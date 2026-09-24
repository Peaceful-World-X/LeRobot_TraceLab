#!/usr/bin/env python3
"""Plot the duration of every episode in one LeRobot dataset as a PNG.

The input dataset must contain ``meta/info.json`` and episode metadata in
``meta/episodes.jsonl``.  Older exports that only have episode Parquet files
under ``meta/episodes/`` are supported as well.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_DATASET = Path(
    "/16T-2/sudu/code/.data/bund_demo_white_curated_20260917_v21"
)


def read_info(dataset: Path) -> dict:
    path = dataset / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少元数据文件: {path}")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        fps = float(info["fps"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取有效 FPS: {path}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"FPS 必须是正数: {fps}")
    return {"fps": fps, "total_episodes": info.get("total_episodes")}


def _validate_rows(rows: list[dict], source: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for row_number, row in enumerate(rows, 1):
        try:
            episode = int(row["episode_index"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}:{row_number} 的 episode_index/length 无效"
            ) from exc
        if episode < 0 or length <= 0:
            raise ValueError(f"{source}:{row_number} 的 episode 或 length 无效")
        if episode in lengths:
            raise ValueError(f"{source}: episode {episode} 重复")
        lengths[episode] = length
    if not lengths:
        raise ValueError(f"{source} 没有 episode 记录")
    expected = list(range(len(lengths)))
    if sorted(lengths) != expected:
        raise ValueError(
            f"episode_index 必须从 0 连续编号，实际范围为 {min(lengths)}..{max(lengths)}"
        )
    return lengths


def read_lengths(dataset: Path) -> dict[int, int]:
    """Read episode lengths from the canonical JSONL or Parquet metadata."""
    jsonl = dataset / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        rows = []
        for line_number, line in enumerate(
            jsonl.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl}:{line_number} 不是有效 JSON") from exc
        return _validate_rows(rows, jsonl)

    episode_dir = dataset / "meta" / "episodes"
    parquet_paths = sorted(episode_dir.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"缺少 episode 元数据: {jsonl} 或 {episode_dir}/chunk-*/*.parquet"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("读取 Parquet 需要 pyarrow，请安装：pip install pyarrow") from exc
    rows: list[dict] = []
    for path in parquet_paths:
        table = pq.read_table(path, columns=["episode_index", "length"])
        rows.extend(table.to_pylist())
    return _validate_rows(rows, episode_dir)


def tick_step(episode_count: int) -> int:
    """Choose readable episode ticks, preferring multiples of ten."""
    if episode_count <= 20:
        return 1
    # About thirty labels are still searchable on the wide canvas.  This keeps
    # a 194/217-episode dataset at 10-episode intervals; larger datasets scale
    # to 20, 50, ... instead of producing an unreadable axis.
    target_ticks = 30
    rough = max(1, math.ceil(episode_count / target_ticks))
    # Keep labels easy to search while preventing a dense 907-episode axis.
    candidates = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]
    return min((step for step in candidates if step >= rough), default=rough)


def configure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "绘制 PNG 需要 matplotlib。当前解释器为 "
            f"{sys.executable}\n"
            f"请安装：{sys.executable} -m pip install matplotlib\n"
            f"如果这是 uv 环境：uv pip install --python {sys.executable} matplotlib"
        ) from exc

    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        path = Path(font_path)
        if path.is_file():
            from matplotlib import font_manager

            matplotlib.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(path)
            ).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def plot_png(lengths: dict[int, int], fps: float, output: Path, title: str) -> None:
    plt = configure_matplotlib()
    import numpy as np

    episodes = np.asarray(sorted(lengths), dtype=int)
    durations = np.asarray([lengths[int(ep)] / fps for ep in episodes], dtype=float)
    mean = float(np.mean(durations))
    median = float(np.median(durations))
    longest = np.argsort(durations)[-3:][::-1]
    shortest = np.argsort(durations)[:3]

    # A wider canvas keeps all episode bars visible without making labels tiny.
    width = min(42.0, max(14.0, 12.0 + len(episodes) * 0.018))
    fig, axis = plt.subplots(figsize=(width, 8.0), dpi=180)
    axis.bar(episodes, durations, width=0.82, color="#4c89c8", edgecolor="none")
    axis.axhline(mean, color="#d45757", linewidth=1.8, label=f"平均数 {mean:.2f} s")
    axis.axhline(
        median,
        color="#e29b32",
        linewidth=1.8,
        linestyle="--",
        label=f"中位数 {median:.2f} s",
    )

    # Annotate exactly the six extrema, with a small horizontal offset to avoid
    # covering neighboring bars.  Ties are still shown as distinct episodes.
    annotation_specs = [
        (longest, "最长", "#a63d40", "bottom"),
        (shortest, "最短", "#246b45", "top"),
    ]
    for indices, label, color, vertical in annotation_specs:
        for rank, index in enumerate(indices, 1):
            episode = int(episodes[index])
            value = float(durations[index])
            dy = max(float(durations.max()) * 0.035, 0.12)
            if vertical == "bottom":
                xytext = (8 if rank % 2 else -8, 8 + rank * 3)
                va = "bottom"
            else:
                xytext = (8 if rank % 2 else -8, -8 - rank * 3)
                va = "top"
            axis.annotate(
                f"{label} #{rank}\nep {episode}: {value:.2f}s",
                xy=(episode, value),
                xytext=xytext,
                textcoords="offset points",
                ha="left" if xytext[0] > 0 else "right",
                va=va,
                fontsize=8,
                color=color,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8},
            )

    step = tick_step(len(episodes))
    tick_values = np.arange(0, len(episodes), step, dtype=int)
    if tick_values[-1] != len(episodes) - 1:
        tick_values = np.append(tick_values, len(episodes) - 1)
    axis.set_xticks(tick_values)
    axis.set_xticklabels([str(episodes[i]) for i in tick_values])
    axis.set_xlim(-1, len(episodes))
    axis.set_xlabel("episode")
    axis.set_ylabel("时长（秒）")
    axis.set_title(f"{title}：所有 episode 时长（{len(episodes)} 条，{fps:g} Hz）")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG 输出路径，默认写入数据集目录 episode_duration.png",
    )
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset}")
    info = read_info(dataset)
    lengths = read_lengths(dataset)
    output = (args.output or dataset / "episode_duration.png").resolve()
    title = dataset.name
    plot_png(lengths, info["fps"], output, title)
    print(
        f"已生成 PNG：{output}（{len(lengths)} 个 episode，"
        f"平均 {sum(lengths.values()) / info['fps'] / len(lengths):.2f} 秒）"
    )


if __name__ == "__main__":
    main()
