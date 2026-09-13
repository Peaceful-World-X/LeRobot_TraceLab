#!/usr/bin/env python3
"""运行 python ee_video_viewer.py，然后打开 http://localhost:9090。
依赖：pip install fastapi uvicorn numpy pyarrow plotly pyyaml
在网页选择机器人类别、数据集目录与 episode；读取规则见 ee_video_viewer.yaml。
"""

import argparse
import csv
import importlib.util
import json
import re
import secrets
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
import yaml
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
MEDIA = OrderedDict()  # 只允许播放已加载 episode 的视频，最多保留 128 个链接。
MEDIA_LOCK = Lock()  # 同时加载多个页面时保护链接登记。
CONFIG_PATH = Path(__file__).with_suffix(".yaml")  # 默认读取入口旁的配置文件。


class ReaderModel(BaseModel):
    """拒绝拼错的配置键和非有限数，避免悄悄采用错误参数。"""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProfileConfig(ReaderModel):
    """每个类别只声明视频键、位姿键、左右 XYZ 索引和方向。"""
    video_key: str = Field(min_length=1)
    arms_key: str = Field(min_length=1)
    lr_xyz_indices: tuple[int, int, int, int, int, int]
    lr_xyz_direction: tuple[float, float, float, float, float, float]

    @field_validator("lr_xyz_indices")
    @classmethod
    def check_indices(cls, value):
        """左右臂索引必须恰好各有三个非负整数。"""
        if any(v < 0 for v in value):
            raise ValueError("lr_xyz_indices 必须为非负整数")
        return value

    @field_validator("lr_xyz_direction")
    @classmethod
    def check_direction(cls, value):
        """方向数组前后三个值分别对应左右臂 XYZ。"""
        if any(v not in (-1, 1) for v in value):
            raise ValueError("lr_xyz_direction 必须只包含 -1 或 1")
        return value


class ViewerConfig(ReaderModel):
    """保存类别映射；默认类别固定为 h01。"""
    profiles: dict[str, ProfileConfig] = Field(min_length=1)


class ProfileUpdate(ReaderModel):
    """网页自定义类别时允许修改的四个字段。"""
    video_key: str = Field(min_length=1)
    arms_key: str = Field(min_length=1)
    lr_xyz_indices: tuple[int, int, int, int, int, int]
    lr_xyz_direction: tuple[float, float, float, float, float, float]


def read_config() -> ViewerConfig:
    """每次加载重新读取 YAML，语法或字段错误直接返回网页。"""
    try:
        config = ViewerConfig.model_validate(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
        return config
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise HTTPException(422, f"配置文件 {CONFIG_PATH.name} 错误：{exc}") from exc


@app.get("/api/profiles")
def profiles() -> dict:
    """向网页提供类别菜单，类别参数由后端统一解释。"""
    config = read_config()
    default = next(iter(config.profiles))
    return {"default_profile": default, "profiles": [
        # id 和显示文字都直接使用 YAML 键名，保留原始大小写。
        {"id": name, "label": name} for name in config.profiles
    ]}


@app.get("/api/profile-config")
def profile_config(profile: str = Query(..., min_length=1, max_length=128)) -> dict:
    """返回指定类别的四项可编辑配置。"""
    config = read_config()
    if profile not in config.profiles:
        raise HTTPException(404, f"未知类别：{profile}")
    return config.profiles[profile].model_dump()


@app.post("/api/profile-config")
def update_profile_config(profile: str = Query(..., min_length=1, max_length=128), update: ProfileUpdate = Body(...)) -> dict:
    """校验后原子更新 YAML 中指定类别，其他类别保持不变。"""
    config = read_config()
    if profile not in config.profiles and profile != "Custom":
        raise HTTPException(404, f"未知类别：{profile}")
    if any(v < 0 for v in update.lr_xyz_indices):
        raise HTTPException(422, "索引必须为非负整数")
    if any(v not in (-1, 1) for v in update.lr_xyz_direction):
        raise HTTPException(422, "方向系数必须只包含 -1 或 1")
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    raw.setdefault("profiles", {})[profile] = update.model_dump()
    try:
        ViewerConfig.model_validate(raw)
        temp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
        temp.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        temp.replace(CONFIG_PATH)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise HTTPException(422, f"配置保存失败：{exc}") from exc
    return {"profile": profile, "saved": True}


def dataset_file(path: Path, root: Path) -> Path:
    """限制推断出的元数据和视频位于同一数据集，包含符号链接检查。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(403, "文件不在当前数据集目录内")
    if not resolved.is_file():
        raise HTTPException(404, f"找不到 {path.name}")
    return resolved


def source_map(root: Path, episode: int, frames: int, warnings: list) -> list | None:
    """可选来源映射无效时明确提示，不伪造来源帧号。"""
    path = root / "source_frame_map.csv"
    if not path.exists() and not path.is_symlink():
        return None
    path = dataset_file(path, root)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "episode_index" not in (reader.fieldnames or []):
                episodes = root.glob("data/chunk-*/episode_*.parquet")
                next(episodes, None)
                if next(episodes, None) is not None:
                    raise ValueError("多 episode 映射缺少 episode_index")
            result = []
            for row in reader:
                if "episode_index" in row and int(row["episode_index"]) != episode:
                    continue
                frame = int(row["source_frame"])
                if int(row["output_frame"]) != len(result) or frame < 0 or (result and frame <= result[-1]):
                    raise ValueError("来源映射必须连续且递增")
                result.append(frame)
                if len(result) > frames:
                    raise ValueError("映射长度过大")
        if len(result) != frames:
            raise ValueError("映射长度不符")
        return result
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, csv.Error):
        warnings.append("来源映射无效，仅显示当前文件帧号")
        return None


@app.get("/api/episode")
def episode(
    parquet_path: str | None = Query(None, min_length=1, max_length=4096),
    dataset_path: str | None = Query(None, min_length=1, max_length=4096),
    episode_index: int | None = Query(None, ge=0, le=1_000_000),
    profile: str | None = Query(None, min_length=1, max_length=128),
    config_json: str | None = Query(None, max_length=4096),
) -> dict:
    """按选定类别读取位置和视频，统一返回米与米每秒。"""
    config = read_config()
    profile_name = profile or next(iter(config.profiles))
    if config_json:
        try:
            spec = ProfileConfig.model_validate(json.loads(config_json))
            profile_name = "Custom"
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise HTTPException(422, f"自定义配置无效：{exc}") from exc
    elif profile_name not in config.profiles:
        raise HTTPException(422, f"未知类别：{profile_name}")
    else:
        spec = config.profiles[profile_name]
    try:
        # 新接口接收数据集总目录和 episode 序号；保留 parquet_path 兼容旧书签。
        if dataset_path is not None:
            root = Path(dataset_path).expanduser().resolve()
            if episode_index is None:
                raise HTTPException(422, "请选择 episode 序号")
            if not root.is_dir():
                raise HTTPException(404, "数据集目录不存在")
            pattern = f"data/chunk-*/episode_{episode_index:06d}.parquet"
            candidates = sorted(p for p in root.glob(pattern) if p.is_file())
            if len(candidates) != 1:
                raise HTTPException(422 if candidates else 404, f"episode {episode_index} 匹配到 {len(candidates)} 个文件：{pattern}")
            parquet, number = candidates[0], episode_index
        elif parquet_path is not None:
            parquet = Path(parquet_path).expanduser().resolve()
            match = re.fullmatch(r"episode_(\d+)\.parquet", parquet.name)
            if not match or parquet.parent.parent.name != "data" or not re.fullmatch(r"chunk-\d+", parquet.parent.name):
                raise HTTPException(422, "旧 parquet_path 接口需要 data/chunk-NNN/episode_NNNNNN.parquet；其他布局请使用数据集目录")
            root, number = parquet.parent.parent.parent, int(match[1])
        else:
            raise HTTPException(422, "请输入数据集目录")
        # 保留数据集内的逻辑 chunk/stem，再校验符号链接的实际目标。
        video_path = f"videos/{parquet.parent.name}/{spec.video_key}/{parquet.stem}.mp4"
        parquet = dataset_file(parquet, root)
        info = json.loads(dataset_file(root / "meta/info.json", root).read_text(encoding="utf-8"))
        if "fps" not in info:
            raise HTTPException(422, "meta/info.json 缺少频率键：fps")
        fps = float(info["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise HTTPException(422, "FPS 必须是有限正数")
        video = dataset_file(root / video_path, root)
        columns = [spec.arms_key, "timestamp", "frame_index"]
        with pq.ParquetFile(parquet) as reader:
            frames = reader.metadata.num_rows
            if not 1 <= frames <= 100_000:
                raise HTTPException(422, "帧数必须在 1 到 100000 之间")
            missing = set(columns) - set(reader.schema_arrow.names)
            if missing:
                raise HTTPException(422, f"Parquet 缺少配置字段：{', '.join(sorted(missing))}")
            table = reader.read(columns=columns)
        times = np.asarray(table["timestamp"].to_pylist(), dtype=float)
        if times.shape != (frames,) or not np.isfinite(times).all() or not np.allclose(times, np.arange(frames) / fps, rtol=0, atol=1e-5):
            raise HTTPException(422, "timestamp 必须从 0 开始并与 FPS 一致")
        if table["frame_index"].to_pylist() != list(range(frames)):
            raise HTTPException(422, "frame_index 必须从 0 连续编号")
        values = np.asarray(table[spec.arms_key].to_pylist(), dtype=float)
        if values.ndim != 2 or values.shape[1] <= max(spec.lr_xyz_indices):
            raise HTTPException(422, f"{spec.arms_key} 维度不足，无法读取 XYZ 索引 {spec.lr_xyz_indices}")
        xyz = values[:, spec.lr_xyz_indices] * spec.lr_xyz_direction
        left, right = xyz[:, :3], xyz[:, 3:]
        left_speed = np.linalg.norm(np.diff(left, axis=0), axis=1) * fps
        right_speed = np.linalg.norm(np.diff(right, axis=0), axis=1) * fps
        if not all(np.isfinite(value).all() for value in (left, right, left_speed, right_speed)):
            raise HTTPException(422, "位置或速度包含 NaN/Inf")
        warnings = []
        mapping = source_map(root, number, frames, warnings)
        stat = video.stat()
        token = secrets.token_urlsafe(24)
        with MEDIA_LOCK:
            MEDIA[token] = (video, root, (stat.st_ino, stat.st_size, stat.st_mtime_ns))
            while len(MEDIA) > 128:
                MEDIA.popitem(last=False)
        return {
            "profile": profile_name,
            "single_arm": spec.lr_xyz_indices[:3] == spec.lr_xyz_indices[3:] and spec.lr_xyz_direction[:3] == spec.lr_xyz_direction[3:],
            "episode": number, "fps": fps, "frames": frames, "duration": frames / fps,
            "video_url": f"api/video/{token}", "left": left.tolist(), "right": right.tolist(),
            "left_speed": [None, *left_speed.tolist()], "right_speed": [None, *right_speed.tolist()],
            "source_frames": mapping, "warnings": warnings,
        }
    except (OSError, RuntimeError, UnicodeError, ValueError, KeyError, TypeError, pa.ArrowException) as exc:
        raise HTTPException(422, f"无法按类别 {profile_name} 读取数据，请检查配置字段、元数据和位置向量：{exc}") from exc


def byte_range(header: str, size: int) -> tuple[int, int]:
    """解析单个 HTTP Range，包括 bytes=-100 等后缀请求。"""
    invalid = HTTPException(416, "无效的视频字节区间", headers={"Content-Range": f"bytes */{size}"})
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", header.strip()) if len(header) < 200 else None
    if not match or not any(match.groups()) or size <= 0:
        raise invalid
    first, last = match.groups()
    if not first:
        if int(last) <= 0:
            raise invalid
        return max(0, size - int(last)), size - 1
    start, end = int(first), int(last) if last else size - 1
    if start >= size or end < start:
        raise invalid
    return start, min(end, size - 1)


@app.api_route("/api/video/{token}", methods=["GET", "HEAD"])
def video(request: Request, token: str) -> Response:
    """只流式读取已登记的视频；同名文件被重构覆盖后要求重新加载。"""
    with MEDIA_LOCK:
        item = MEDIA.get(token)
    if item is None:
        raise HTTPException(404, "视频链接失效，请重新加载 episode")
    path, root, fingerprint = item
    try:
        path = dataset_file(path, root)
        stat = path.stat()
    except OSError as exc:
        raise HTTPException(404, "视频无法读取") from exc
    if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != fingerprint:
        raise HTTPException(409, "视频已更新，请重新加载 episode")
    size, start, end, status = stat.st_size, 0, stat.st_size - 1, 200
    etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
    headers = {"Accept-Ranges": "bytes", "ETag": etag, "Cache-Control": "private, no-cache"}
    if request.method != "HEAD" and request.headers.get("range") and request.headers.get("if-range", etag) == etag:
        start, end = byte_range(request.headers["range"], size)
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(max(0, end - start + 1))
    if request.method == "HEAD":
        return Response(headers=headers, media_type="video/mp4")

    def chunks():
        """按 1 MiB 读取，避免把视频整体加载进 Python 内存。"""
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(chunks(), status_code=status, headers=headers, media_type="video/mp4")


@app.get("/assets/plotly.min.js")
def plotly_asset() -> FileResponse:
    """使用本地 Plotly 资源，浏览器无需连接 CDN。"""
    spec = importlib.util.find_spec("plotly")
    if not spec or not spec.origin:
        raise HTTPException(503, "请先 pip install plotly")
    return FileResponse(Path(spec.origin).parent / "package_data/plotly.min.js", media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """HTML 与 Python 放在同一目录即可运行。"""
    return HTMLResponse(Path(__file__).with_suffix(".html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


def main() -> None:
    """默认使用同目录 YAML，也可指定独立配置文件。"""
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    CONFIG_PATH = args.config.expanduser().resolve()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
