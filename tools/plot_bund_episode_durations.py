#!/usr/bin/env python3
"""绘制 BUND 数据集每个 episode 重构前后的时长和压缩率。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_SOURCE = Path(
    "/16T-2/sudu/code/.data/bund_demo_data_0reset_0827_v002_v21"
)
DEFAULT_RECONSTRUCTED = Path(
    "/16T-2/sudu/code/.data/bund_demo_data_0reset_0827_v002_v21_reconstructed"
)


def read_info(root: Path) -> dict:
    path = root / "meta/info.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少元数据: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_lengths(root: Path) -> dict[int, int]:
    path = root / "meta/episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 episode 元数据: {path}")
    lengths = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            episode = int(row["episode_index"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number} 的 episode_index/length 无效") from error
        if episode in lengths:
            raise ValueError(f"{path}: episode {episode} 重复")
        if length <= 0:
            raise ValueError(f"{path}: episode {episode} 的 length 必须为正数")
        lengths[episode] = length
    if not lengths:
        raise ValueError(f"{path} 没有 episode 记录")
    return lengths


def load_rows(source: Path, reconstructed: Path) -> tuple[list[dict], float]:
    source_fps = float(read_info(source).get("fps", 10))
    output_fps = float(read_info(reconstructed).get("fps", source_fps))
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"源数据 FPS 无效: {source_fps}")
    if not math.isclose(source_fps, output_fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"源和重构 FPS 不一致: {source_fps} vs {output_fps}")
    before = read_lengths(source)
    after = read_lengths(reconstructed)
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        raise ValueError(f"episode 集合不一致，缺少={missing[:10]}，多出={extra[:10]}")

    rows = []
    for episode in sorted(before):
        source_duration = before[episode] / source_fps
        output_duration = after[episode] / output_fps
        rows.append(
            {
                "episode": episode,
                "source_duration": source_duration,
                "output_duration": output_duration,
                "compression_rate": (1 - output_duration / source_duration) * 100,
            }
        )
    return rows, source_fps


def plot_png(rows: list[dict], fps: float, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "绘制 PNG 需要 matplotlib，请安装：pip install matplotlib"
        ) from error

    # Use an installed CJK font so Chinese axis labels are present in the PNG.
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(font_path).is_file():
            from matplotlib import font_manager

            font_name = font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams["font.family"] = font_name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    episodes = np.asarray([row["episode"] for row in rows])
    source = np.asarray([row["source_duration"] for row in rows])
    rebuilt = np.asarray([row["output_duration"] for row in rows])
    compression = np.asarray([row["compression_rate"] for row in rows])
    width = max(16, min(36, 10 + len(rows) * 0.08))
    fig, axis = plt.subplots(figsize=(width, 8), dpi=180)
    offset = 0.2
    bar_width = 0.38
    axis.bar(
        episodes - offset,
        source,
        width=bar_width,
        color="#b8c0c8",
        label="原始时长",
    )
    axis.bar(
        episodes + offset,
        rebuilt,
        width=bar_width,
        color="#4c89c8",
        label="重构后时长",
    )
    axis.set_xlabel("episode")
    axis.set_ylabel("时长（秒）")
    axis.set_title(f"BUND episode 时长与压缩率（{len(rows)} 条，{fps:g} Hz）")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(episodes.min() - 1, episodes.max() + 1)
    axis.set_xticks(episodes[:: max(1, len(episodes) // 20)])

    rate_axis = axis.twinx()
    rate_axis.plot(
        episodes,
        compression,
        color="#e27d32",
        linewidth=2,
        marker="o",
        markersize=2.5,
        markevery=max(1, len(episodes) // 40),
        label="压缩率",
    )
    rate_axis.set_ylabel("压缩率（%）", color="#c56520")
    rate_axis.tick_params(axis="y", labelcolor="#c56520")
    rate_axis.set_ylim(0, max(100, float(compression.max()) * 1.1))

    handles, labels = axis.get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    axis.legend(handles + rate_handles, labels + rate_labels, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="png", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help="重构前 v2.1 数据集"
    )
    parser.add_argument(
        "--reconstructed", type=Path, default=DEFAULT_RECONSTRUCTED, help="重构后数据集"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RECONSTRUCTED / "episode_duration_comparison.png",
        help="PNG 输出路径",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    reconstructed = args.reconstructed.resolve()
    if not source.is_dir() or not reconstructed.is_dir():
        raise SystemExit(f"数据集目录不存在: source={source}, reconstructed={reconstructed}")
    rows, fps = load_rows(source, reconstructed)
    plot_png(rows, fps, args.output.resolve())
    print(f"已生成 PNG：{args.output.resolve()}（{len(rows)} 个 episode）")


if __name__ == "__main__":
    main()
