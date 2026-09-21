#!/usr/bin/env python3
"""Convert local LeRobot v3.0 to v2.1 without resampling or video re-encoding.

Requires Python >= 3.10, numpy, pyarrow, ffmpeg and ffprobe. Video episodes must
start at a keyframe, with one packet per frame and monotonically increasing PTS.
Unsupported inputs fail before publication instead of silently shifting images.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = CODE_ROOT.parent / "data"
SOURCE = DATA_ROOT / "bund_demo_data_0reset_0827_v002"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
STAT_NAMES = ("min", "max", "mean", "std", "count")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, values):
    Path(path).write_text(
        "".join(
            json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values
        ),
        encoding="utf-8",
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_file(root, name):
    path = (root / name).resolve()
    require(
        root in path.parents and path.is_file(),
        f"Missing file or path outside source: {name}",
    )
    return path


def output_path(folder, template, episode, camera=None):
    return folder / template.format(
        episode_chunk=episode // 1000, episode_index=episode, video_key=camera
    )


def run(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-4000:]}")
    return result.stdout


def probe_video(path):
    data = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,start_time:packet=pts_time,dts_time,flags,data_hash",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    require(len(data.get("streams", [])) == 1, f"Expected one video stream: {path}")
    require(bool(data.get("packets")), f"Empty video: {path}")
    return data


def packet_digest(packets):
    return hashlib.sha256(
        "\n".join(p["data_hash"] for p in packets).encode("ascii")
    ).hexdigest()


def video_slice(probe, start, end, length, fps):
    """Use presentation timestamps, never assume episode-global frame offsets."""
    require(
        math.isfinite(start) and math.isfinite(end) and start >= 0,
        "Invalid video timestamp",
    )
    require(
        abs((end - start) * fps - length) < 1e-4,
        "Video duration and episode length disagree",
    )
    packets = probe["packets"]
    pts = np.asarray([float(p["pts_time"]) for p in packets])
    require(
        np.all(np.diff(pts) > 0),
        "Video PTS must be strictly increasing for lossless splitting",
    )
    begin = int(np.searchsorted(pts, start - 1e-5))
    selected = packets[begin : begin + length]
    require(
        len(selected) == length
        and np.allclose(
            pts[begin : begin + length],
            start + np.arange(length) / fps,
            rtol=0,
            atol=1e-5,
        ),
        "Missing, duplicated or misaligned video frames",
    )
    require(
        "K" in selected[0]["flags"],
        "Episode does not begin on a video keyframe; lossless splitting is unsupported",
    )
    # Reordered packets can need frames outside the requested interval.
    require(
        all(
            abs(float(p.get("dts_time", p["pts_time"])) - float(p["pts_time"])) < 1e-5
            for p in selected
        ),
        "Reordered video packets need a separate decode/re-encode conversion",
    )
    num, den = map(float, probe["streams"][0]["r_frame_rate"].split("/"))
    require(abs(num / den - fps) < 1e-6, "Video FPS differs from dataset FPS")
    return {
        "first_packet": begin,
        "packet_count": length,
        "packet_sha256": packet_digest(selected),
    }


def legacy_hf_metadata(table):
    """datasets 2.x understands Sequence, but not the datasets 4.x List type."""

    def convert(obj):
        if isinstance(obj, dict):
            out = {k: convert(v) for k, v in obj.items()}
            if out.get("_type") == "List":
                out["_type"] = "Sequence"
            return out
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = convert(json.loads(metadata[b"huggingface"]))
        hf.pop("fingerprint", None)
        metadata[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(metadata)


def select_episode(table, episode, fps):
    source_id, n = episode["episode_index"], episode["length"]
    global_indices = np.asarray(table["index"])
    require(
        np.array_equal(
            global_indices, np.arange(global_indices[0], global_indices[0] + len(table))
        ),
        "Source file index is not contiguous",
    )
    offset = episode["dataset_from_index"] - int(global_indices[0])
    require(
        0 <= offset and offset + n <= len(table),
        f"Episode {source_id} spans/misses its data file",
    )
    out = table.slice(offset, n)
    checks = {
        "episode_index": np.full(n, source_id),
        "index": np.arange(episode["dataset_from_index"], episode["dataset_to_index"]),
        "frame_index": np.arange(n),
    }
    for key, expected in checks.items():
        require(
            np.array_equal(np.asarray(out[key]), expected),
            f"Episode {source_id}: invalid {key}",
        )
    require(
        np.allclose(
            np.asarray(out["timestamp"]), np.arange(n) / fps, rtol=0, atol=1e-5
        ),
        f"Episode {source_id}: invalid timestamps",
    )
    require(all(c.null_count == 0 for c in out.columns), "Null source values")
    return out


def reindex_episode(table, output_episode, global_start):
    for key, values in (
        ("episode_index", np.full(len(table), output_episode)),
        ("index", np.arange(global_start, global_start + len(table))),
    ):
        table = table.set_column(
            table.schema.get_field_index(key),
            key,
            pa.array(values, type=table.schema.field(key).type),
        )
    return legacy_hf_metadata(table)


def numeric_stats(table, features):
    stats = {}
    for key, spec in features.items():
        if spec["dtype"] in ("video", "image"):
            continue
        require(key in table.column_names, f"Missing numeric feature: {key}")
        a = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        require(
            list(a.shape[1:]) == spec["shape"] and np.isfinite(a).all(),
            f"Invalid numeric feature {key}",
        )
        stats[key] = {
            name: fn(a, axis=0).tolist()
            for name, fn in [
                ("min", np.min),
                ("max", np.max),
                ("mean", np.mean),
                ("std", np.std),
            ]
        }
        stats[key]["count"] = [len(a)]
    return stats


def visual_stats(episode, key, shape):
    result = {name: episode.get(f"stats/{key}/{name}") for name in STAT_NAMES}
    require(
        all(v is not None for v in result.values()),
        f"Missing per-episode visual statistics: {key}",
    )
    require(
        len(result["count"]) == 1 and result["count"][0] > 0,
        f"Invalid visual statistics count: {key}",
    )
    for name in STAT_NAMES[:-1]:
        value = np.asarray(result[name])
        require(
            value.shape == (shape[-1], 1, 1) and np.isfinite(value).all(),
            f"Invalid visual statistics shape: {key}/{name}",
        )
    require(np.all(np.asarray(result["std"]) >= 0), "Negative standard deviation")
    return result


def aggregate_stats(items):
    result = {}
    for key in items[0]:
        entries = [item[key] for item in items]
        count = np.asarray([v["count"][0] for v in entries], dtype=np.float64)
        means = np.asarray([v["mean"] for v in entries])
        stds = np.asarray([v["std"] for v in entries])
        w = (count / count.sum()).reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (w * means).sum(axis=0)
        variance = (w * (stds**2 + (means - mean) ** 2)).sum(axis=0)
        result[key] = {
            "min": np.min([v["min"] for v in entries], axis=0).tolist(),
            "max": np.max([v["max"] for v in entries], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(count.sum())],
        }
    return result


def read_source(root):
    info_path = local_file(root, "meta/info.json")
    info = json.loads(info_path.read_text())
    require(info["codebase_version"] == "v3.0", "Expected codebase_version v3.0")
    require(math.isfinite(info["fps"]) and info["fps"] > 0, "Invalid FPS")
    tasks_path = local_file(root, "meta/tasks.parquet")
    tasks = pq.ParquetFile(tasks_path).read().to_pylist()
    require(
        all("task_index" in r and "task" in r for r in tasks),
        "Expected task_index/task in tasks.parquet",
    )
    require(
        len({r["task_index"] for r in tasks}) == len(tasks) == info["total_tasks"],
        "Invalid tasks metadata",
    )
    meta_paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    require(bool(meta_paths), "No episode metadata")
    episodes = []
    for path in meta_paths:
        episodes.extend(
            pq.ParquetFile(local_file(root, path.relative_to(root))).read().to_pylist()
        )
    episodes.sort(key=lambda r: r["episode_index"])
    require(
        [r["episode_index"] for r in episodes] == list(range(info["total_episodes"])),
        "Duplicate, missing or non-contiguous episode metadata",
    )
    cursor = 0
    for ep in episodes:
        require(
            ep["length"] > 0
            and ep["dataset_from_index"] == cursor
            and ep["dataset_to_index"] == cursor + ep["length"],
            "Invalid episode global row bounds",
        )
        cursor += ep["length"]
    require(cursor == info["total_frames"], "Source total_frames mismatch")
    return info, tasks, episodes


def make_plan(root, wanted, workers):
    info, tasks, episodes = read_source(root)
    wanted = sorted(set(range(len(episodes))) if wanted is None else set(wanted))
    require(
        bool(wanted) and all(0 <= i < len(episodes) for i in wanted),
        "Unknown/empty episode selection",
    )
    videos = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    require(
        all(
            "/" not in key and "\\" not in key and key not in (".", "..")
            for key in videos
        ),
        "Video feature names must be safe path components",
    )
    source_files = {p.resolve() for p in (root / "meta").rglob("*") if p.is_file()}
    for p in source_files:
        require(root in p.parents, "Metadata resolves outside source")
    records, tables, probes = [], {}, {}
    cursor = 0
    for new_id, old_id in enumerate(wanted):
        ep = episodes[old_id]
        path = local_file(
            root,
            info["data_path"].format(
                chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
            ),
        )
        source_files.add(path)
        record = {
            "source_episode": old_id,
            "output_episode": new_id,
            "length": ep["length"],
            "source_from_index": ep["dataset_from_index"],
            "output_from_index": cursor,
            "data_file": str(path.relative_to(root)),
            "tasks": ep["tasks"],
            "videos": {},
        }
        for key in videos:
            prefix = f"videos/{key}/"
            video = local_file(
                root,
                info["video_path"].format(
                    video_key=key,
                    chunk_index=ep[prefix + "chunk_index"],
                    file_index=ep[prefix + "file_index"],
                ),
            )
            source_files.add(video)
            probes[video] = None
            record["videos"][key] = {
                "source_file": str(video.relative_to(root)),
                "from_timestamp": ep[prefix + "from_timestamp"],
                "to_timestamp": ep[prefix + "to_timestamp"],
            }
        records.append(record)
        cursor += ep["length"]
    # Fingerprint before analysis and compare again before publication.
    hashes = {str(p.relative_to(root)): sha256(p) for p in sorted(source_files)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_video, p): p for p in probes}
        for future in as_completed(futures):
            probes[futures[future]] = future.result()
    schemas = []
    task_names = {r["task_index"]: r["task"] for r in tasks}
    for record in records:
        ep = episodes[record["source_episode"]]
        path = root / record["data_file"]
        if path not in tables:
            tables[path] = pq.ParquetFile(path).read(use_threads=False)
            schemas.append(tables[path].schema.remove_metadata())
        table = select_episode(tables[path], ep, info["fps"])
        task_ids = set(table["task_index"].to_pylist())
        require(
            task_ids <= task_names.keys()
            and {task_names[i] for i in task_ids} == set(ep["tasks"]),
            "Task labels disagree with episode metadata",
        )
        record["stats"] = numeric_stats(
            reindex_episode(
                table, record["output_episode"], record["output_from_index"]
            ),
            info["features"],
        )
        for key, feature in info["features"].items():
            if feature["dtype"] in ("video", "image"):
                record["stats"][key] = visual_stats(ep, key, feature["shape"])
        for key, span in record["videos"].items():
            probe = probes[root / span["source_file"]]
            span.update(
                video_slice(
                    probe,
                    span["from_timestamp"],
                    span["to_timestamp"],
                    ep["length"],
                    info["fps"],
                )
            )
            stream = probe["streams"][0]
            require(
                info["features"][key]["shape"][:2]
                == [stream["height"], stream["width"]],
                f"Video resolution differs from metadata: {key}",
            )
            span["stream"] = stream
    require(all(s == schemas[0] for s in schemas), "Data files have different schemas")
    full = wanted == list(range(len(episodes)))
    if full:
        require(
            sum(len(t) for t in tables.values()) == info["total_frames"],
            "Unreferenced or duplicated data rows",
        )
    return {
        "source_root": str(root),
        "info": info,
        "tasks": tasks,
        "records": records,
        "source_sha256": hashes,
        "all_episodes": full,
        "summary": {
            "episodes": len(records),
            "frames": cursor,
            "videos": len(records) * len(videos),
            "fps": info["fps"],
            "state_shape": info["features"].get("observation.state", {}).get("shape"),
            "action_shape": info["features"].get("action", {}).get("shape"),
            "video_mode": "copy; encoded packets unchanged; timestamps start at zero",
        },
    }


def split_video(source, destination, span, n, fps):
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-ss",
            format(span["from_timestamp"], ".12g"),
            "-i",
            str(source),
            "-t",
            format(n / fps, ".12g"),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    converted = probe_video(destination)
    require(
        len(converted["packets"]) == n,
        f"Wrong output video packet count: {destination}",
    )
    check = video_slice(converted, 0, n / fps, n, fps)
    require(
        check["packet_sha256"] == span["packet_sha256"],
        f"Video/source encoded packets differ: {destination}",
    )
    stream = converted["streams"][0]
    for key in ("codec_name", "width", "height", "pix_fmt"):
        require(stream[key] == span["stream"][key], f"Video stream changed: {key}")
    # Standalone decoding detects missing references despite matching packet bytes.
    progress = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-threads",
            "1",
            "-i",
            str(destination),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ]
    )
    counts = [
        int(line.split("=", 1)[1])
        for line in progress.splitlines()
        if line.startswith("frame=")
    ]
    require(
        bool(counts) and counts[-1] == n,
        f"Wrong decoded video frame count: {destination}",
    )
    return {
        "frames": n,
        "packet_sha256": check["packet_sha256"],
        "sha256": sha256(destination),
        "standalone_decode_verified": True,
        "stream": stream,
    }


def build_data_file(root, stage, records, info):
    table = pq.ParquetFile(root / records[0]["data_file"]).read(use_threads=False)
    results = []
    for r in records:
        ep = {
            "episode_index": r["source_episode"],
            "length": r["length"],
            "dataset_from_index": r["source_from_index"],
            "dataset_to_index": r["source_from_index"] + r["length"],
        }
        original = select_episode(table, ep, info["fps"])
        out = reindex_episode(original, r["output_episode"], r["output_from_index"])
        path = output_path(stage, DATA_PATH, r["output_episode"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, path, compression="snappy")
        loaded = pq.ParquetFile(path).read(use_threads=False)
        require(
            loaded.equals(out, check_metadata=True),
            f"Parquet readback mismatch: {path}",
        )
        for key in original.column_names:
            if key not in ("episode_index", "index"):
                require(
                    loaded[key].equals(original[key]), f"Source column mismatch: {key}"
                )
        results.append(
            (
                r["output_episode"],
                {
                    "source_columns_verified": True,
                    "parquet_readback_verified": True,
                    "sha256": sha256(path),
                },
            )
        )
    return results


def write_metadata(stage, plan, verified):
    records, info = plan["records"], copy.deepcopy(plan["info"])
    meta = stage / "meta"
    meta.mkdir()
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    info.update(
        codebase_version="v2.1",
        total_episodes=len(records),
        total_frames=plan["summary"]["frames"],
        total_videos=plan["summary"]["videos"],
        total_chunks=math.ceil(len(records) / 1000),
        chunks_size=1000,
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
    )
    if not plan["all_episodes"]:
        info["splits"] = {"train": f"0:{len(records)}"}
    for camera, span in records[0]["videos"].items():
        stream = span["stream"]
        info["features"][camera].setdefault("info", {}).update(
            {
                "video.codec": stream["codec_name"],
                "video.height": stream["height"],
                "video.width": stream["width"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.fps": info["fps"],
                "has_audio": False,
            }
        )
    write_json(meta / "info.json", info)
    write_jsonl(meta / "tasks.jsonl", plan["tasks"])
    write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": r["output_episode"],
                "tasks": r["tasks"],
                "length": r["length"],
            }
            for r in records
        ],
    )
    write_jsonl(
        meta / "episodes_stats.jsonl",
        [{"episode_index": r["output_episode"], "stats": r["stats"]} for r in records],
    )
    write_json(meta / "stats.json", aggregate_stats([r["stats"] for r in records]))
    with (stage / "source_frame_map.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_index",
                "source_episode",
                "output_frame",
                "source_frame",
                "output_index",
                "source_index",
            ]
        )
        for r in records:
            writer.writerows(
                (
                    r["output_episode"],
                    r["source_episode"],
                    i,
                    i,
                    r["output_from_index"] + i,
                    r["source_from_index"] + i,
                )
                for i in range(r["length"])
            )
    manifest = {k: v for k, v in plan.items() if k not in ("info", "records", "tasks")}
    manifest.update(
        script_sha256=sha256(__file__),
        strategy="v3_to_v21_lossless_episode_split_v1",
        numeric_statistics="min/max/mean/std/count recomputed from output rows",
        visual_statistics="source per-episode sampled statistics preserved; pooled with source sample counts",
        fields="all retained; only episode_index/index renumbered for a subset; HF List metadata converted to Sequence",
        episodes=[
            {
                **{k: v for k, v in r.items() if k != "stats"},
                "validation": verified[r["output_episode"]],
            }
            for r in records
        ],
    )
    write_json(stage / "conversion.json", manifest)


def validate_dataset(stage, plan):
    info = json.loads((stage / "meta/info.json").read_text())
    episodes = [
        json.loads(s) for s in (stage / "meta/episodes.jsonl").read_text().splitlines()
    ]
    require(len(episodes) == info["total_episodes"], "Output episode count mismatch")
    cursor = 0
    for row in episodes:
        ep, n = row["episode_index"], row["length"]
        table = pq.ParquetFile(output_path(stage, DATA_PATH, ep)).read(
            columns=["episode_index", "index", "frame_index", "timestamp"]
        )
        require(len(table) == n, "Output length mismatch")
        for key, expected in [
            ("episode_index", np.full(n, ep)),
            ("index", np.arange(cursor, cursor + n)),
            ("frame_index", np.arange(n)),
        ]:
            require(
                np.array_equal(np.asarray(table[key]), expected),
                f"Output {key} mismatch",
            )
        require(
            np.allclose(
                np.asarray(table["timestamp"]),
                np.arange(n) / info["fps"],
                rtol=0,
                atol=1e-5,
            ),
            "Output timestamp mismatch",
        )
        cursor += n
    require(
        cursor == info["total_frames"]
        and len(list(stage.glob("videos/*/*/*.mp4"))) == info["total_videos"],
        "Output metadata totals disagree",
    )


def publish(stage, output):
    """Atomic Linux rename with RENAME_NOREPLACE, including the last race window."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def convert(plan, output, workers):
    require(
        not output.exists() and not output.is_symlink(),
        f"Refusing existing output: {output}",
    )
    root = Path(plan["source_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    size = sum((root / p).stat().st_size for p in plan["source_sha256"])
    require(
        shutil.disk_usage(output.parent).free > size * 1.2 + 256 * 1024**2,
        "Insufficient space for staging",
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    print(f"Staging: {stage}", file=sys.stderr, flush=True)
    verified = {r["output_episode"]: {"videos": {}} for r in plan["records"]}
    try:
        groups = {}
        for record in plan["records"]:
            groups.setdefault(record["data_file"], []).append(record)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [
                pool.submit(build_data_file, root, stage, rows, plan["info"])
                for rows in groups.values()
            ]
            for future in as_completed(pending):
                for ep, result in future.result():
                    verified[ep]["data"] = result
        print(
            f"Parquet verified: {len(verified)}/{len(verified)}",
            file=sys.stderr,
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            for r in plan["records"]:
                for key, span in r["videos"].items():
                    dest = output_path(stage, VIDEO_PATH, r["output_episode"], key)
                    future = pool.submit(
                        split_video,
                        root / span["source_file"],
                        dest,
                        span,
                        r["length"],
                        plan["info"]["fps"],
                    )
                    pending[future] = (r["output_episode"], key)
            for done, future in enumerate(as_completed(pending), 1):
                ep, key = pending[future]
                try:
                    verified[ep]["videos"][key] = future.result()
                except BaseException:
                    for job in pending:
                        job.cancel()
                    raise
                if done % 10 == 0 or done == len(pending):
                    print(
                        f"Video copied/decoded/verified: {done}/{len(pending)}",
                        file=sys.stderr,
                        flush=True,
                    )
        write_metadata(stage, plan, verified)
        validate_dataset(stage, plan)
        for name, before in plan["source_sha256"].items():
            require(
                sha256(root / name) == before,
                f"Source changed during conversion: {name}",
            )
        publish(stage, output)
    except BaseException:
        print(
            f"Failed; unpublished staging retained: {stage}",
            file=sys.stderr,
            flush=True,
        )
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument(
        "--output", type=Path, help="Default: sibling <source-name>_v21"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-episodes", action="store_true")
    selection.add_argument(
        "--episode",
        type=int,
        action="append",
        help="Repeat to convert a subset; output episodes start at zero",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate source rows and video packet boundaries; write no files",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    root = args.source_root.resolve()
    output = (
        args.output or args.source_root.with_name(args.source_root.name + "_v21")
    ).absolute()
    resolved = output.resolve()
    if root == resolved or root in resolved.parents or resolved in root.parents:
        parser.error("Output must be outside and independent of source")
    if not args.plan_only and (output.exists() or output.is_symlink()):
        parser.error(f"Refusing existing output: {output}")
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            parser.error(f"{name} is required")
    started = time.monotonic()
    plan = make_plan(root, args.episode, args.workers)
    print(json.dumps(plan["summary"], ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return
    convert(plan, output, args.workers)
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_s": round(time.monotonic() - started, 1),
                **plan["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
