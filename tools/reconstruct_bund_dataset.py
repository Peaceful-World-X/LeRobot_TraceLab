#!/usr/bin/env python3
"""Optimize BUND UR demonstrations by selecting paired real frames.

The input is a LeRobot v2.1 dataset with external video features. The optimizer
does not interpolate or rewrite poses/actions: it selects source rows, rebuilds
the timeline at the dataset FPS, and encodes output videos from exactly those
decoded source frames. Grasp, placement, controller-limit and orientation
events are protected while ordinary motion is shortened with a constrained
second-order DAG path search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import uniform_filter1d

CODE_ROOT = Path(__file__).resolve().parents[2]
SOURCE = CODE_ROOT / ".data" / "bund_demo_data_0reset_0827_v002_v21"
OUTPUT_NAME = "bund_demo_data_0reset_0827_v002_v21_reconstructed"
CAMERAS = ("observation.images.base", "observation.images.wrist")
FAST_PHASES = {0, 2, 4}
PHASES = ("approach", "grasp", "carry_reorient", "release", "retreat")
PHASE_LABELS = ("接近", "闭爪抓取", "持物抬升旋转", "开爪放置", "撤离")


@dataclass(frozen=True)
class Rules:
    target_speed_m_s: float = 0.1
    speed_tolerance_m_s: float = 0.02
    max_step_m: float = 0.015
    max_rotation_deg: float = 5.0
    path_deviation_m: float = 0.01
    max_skip_s: float = 3.0
    event_padding_s: float = 0.3
    contact_radius_m: float = 0.01
    closed_threshold: float = 0.8
    open_threshold: float = 0.9
    static_speed_m_s: float = 0.005
    static_rotation_deg_s: float = 2.0
    static_position_range_m: float = 0.003
    static_orientation_range_deg: float = 2.0
    static_hold_s: float = 1.0
    orientation_activity_deg_s: float = 8.0
    orientation_anchor_deg: float = 15.0

    @classmethod
    def read(cls, path=None):
        values = {} if path is None else json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        if not isinstance(values, dict) or set(values) - names:
            raise ValueError("rules-json contains an unknown field")
        result = cls(**values)
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in asdict(result).values()
        ):
            raise ValueError("all rule values must be finite positive numbers")
        if not 0 < result.closed_threshold < result.open_threshold < 1:
            raise ValueError("invalid gripper hysteresis")
        if result.speed_tolerance_m_s >= result.target_speed_m_s:
            raise ValueError("speed tolerance must be smaller than target")
        return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def runs(mask):
    edge = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def rotations(values):
    x = values[:, :3].astype(float)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = values[:, 3:6] - np.sum(values[:, 3:6] * x, axis=1, keepdims=True) * x
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def angle(a, b):
    return np.rad2deg(np.arccos(np.clip((np.sum(a * b, axis=(-1, -2)) - 1) / 2, -1, 1)))


def line_distance(points, start, end):
    delta = end - start
    t = np.clip((points - start) @ delta / max(float(delta @ delta), 1e-20), 0, 1)
    return np.linalg.norm(points - start - t[:, None] * delta, axis=1)


def trajectory_metrics(xyz, fps):
    d = np.diff(xyz, axis=0)
    length = np.linalg.norm(d, axis=1)
    chord = float(np.linalg.norm(xyz[-1] - xyz[0])) if len(xyz) > 1 else 0.0
    direction = (xyz[-1] - xyz[0]) / max(chord, 1e-12)
    significant = d[length >= 0.001]
    turns = np.zeros(0)
    if len(significant) > 1:
        u = significant / np.maximum(
            np.linalg.norm(significant, axis=1, keepdims=True), 1e-12
        )
        turns = np.rad2deg(np.arccos(np.clip(np.sum(u[:-1] * u[1:], axis=1), -1, 1)))
    return {
        "frames": len(xyz),
        "duration_s": max(0, len(xyz) - 1) / fps,
        "path_length_m": float(length.sum()),
        "chord_m": chord,
        "speed_mean_m_s": float(length.mean() * fps) if len(length) else 0.0,
        "speed_median_m_s": float(np.median(length) * fps) if len(length) else 0.0,
        "speed_max_m_s": float(length.max() * fps) if len(length) else 0.0,
        "turn_sum_deg": float(turns.sum()),
        "backtrack_m": float(np.maximum(-(d @ direction), 0).sum()),
        "straightness": chord / max(float(length.sum()), 1e-12),
    }


class Analyzer:
    def __init__(self, table, fps, rules):
        self.table, self.fps, self.rules = table, float(fps), rules
        self.n = table.num_rows
        self.active_end = self.n - 1
        self.truncation = None
        self.state = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        self.action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if (
            self.state.shape != (self.n, 34)
            or self.action.shape != (self.n, 10)
            or self.n < 2
        ):
            raise ValueError("BUND expects state[34], action[10] and at least two rows")
        self.xyz = self.state[:, :3].astype(float)
        self.action_xyz = self.action[:, :3].astype(float)
        self.rot = rotations(self.state[:, 3:9])
        self.action_rot = rotations(self.action[:, 3:9])
        self.rotation_step = angle(self.rot[:-1], self.rot[1:])
        self.action_rotation_step = angle(self.action_rot[:-1], self.action_rot[1:])
        self.rotation_sum = np.r_[0.0, np.cumsum(self.rotation_step)]
        self.action_rotation_sum = np.r_[0.0, np.cumsum(self.action_rotation_step)]
        self.state_speed = np.r_[
            0.0, np.linalg.norm(np.diff(self.xyz, axis=0), axis=1) * self.fps
        ]
        self.action_speed = np.r_[
            0.0, np.linalg.norm(np.diff(self.action_xyz, axis=0), axis=1) * self.fps
        ]
        self.phase = np.zeros(self.n, dtype=np.int8)
        self.protected = np.zeros(self.n, dtype=bool)
        self.protection = [set() for _ in range(self.n)]
        self.warnings = []
        self.events = []
        self.idle_spans = []
        self.rejections = {"step": 0, "rotation": 0, "deviation": 0}
        self._check_timeline()

    def _check_timeline(self):
        for key, expected in (
            ("frame_index", np.arange(self.n)),
            ("timestamp", np.arange(self.n) / self.fps),
        ):
            actual = np.asarray(self.table[key].to_pylist())
            if not np.allclose(actual, expected, rtol=0, atol=1e-5):
                raise ValueError(f"invalid source {key}")
        if not np.isfinite(self.state).all() or not np.isfinite(self.action).all():
            raise ValueError("nonfinite state/action")

    def protect(self, start, end, reason):
        for i in range(max(0, start), min(self.n, end)):
            self.protected[i] = True
            self.protection[i].add(reason)

    def detect_gripper_events(self):
        state_g, action_g = self.state[:, 9], self.action[:, 9]
        transitions = []
        closed = False
        for i, value in enumerate(state_g):
            if not closed and value < self.rules.closed_threshold:
                transitions.append(("close", i))
                closed = True
            elif closed and value > self.rules.open_threshold:
                transitions.append(("open", i))
                closed = False
        if [kind for kind, _ in transitions] != ["close", "open"]:
            self.warnings.append("ambiguous_gripper_sequence: retained entire episode")
            self.phase[:] = 2
            self.protect(0, self.n, "ambiguous_gripper")
            return None
        events = []
        for kind, trigger in transitions:
            lo = max(0, trigger - 1)
            hi = min(self.n - 1, trigger + 1)
            activity = (np.abs(np.diff(state_g)) > 0.005) | (
                np.abs(np.diff(action_g)) > 0.005
            )
            while lo > 0 and activity[lo - 1]:
                lo -= 1
            while hi < self.n - 1 and activity[hi]:
                hi += 1
            pad = round(self.rules.event_padding_s * self.fps)
            start, end = max(0, lo - pad), min(self.n - 1, hi + pad)
            events.append(
                {"kind": kind, "trigger": trigger, "start": start, "end": end}
            )
            self.protect(start, end + 1, f"{kind}_event")
        self.events = events
        return events

    def detect_phases(self):
        events = self.detect_gripper_events()
        if not events:
            return
        close, open_ = events
        c, o = close["trigger"], open_["trigger"]
        close_end = min(o - 1, close["end"])
        open_start = max(close_end + 1, open_["start"])
        # Contact zones use both measured state and commanded action targets.
        grasp = np.median(self.xyz[max(0, c - 2) : c + 1], axis=0)
        place = np.median(self.xyz[max(c, o - 3) : min(o + 1, self.n)], axis=0)
        self.phase[: close["start"]] = 0
        self.phase[close["start"] : close_end + 1] = 1
        self.phase[close_end + 1 : open_start] = 2
        self.phase[open_start : open_["end"] + 1] = 3
        self.phase[open_["end"] + 1 :] = 4
        self.detect_lift_apex(close_end, open_["start"])
        for phase, center in ((0, grasp), (2, place)):
            near = (
                np.linalg.norm(self.xyz - center, axis=1) <= self.rules.contact_radius_m
            ) | (
                np.linalg.norm(self.action_xyz - center, axis=1)
                <= self.rules.contact_radius_m
            )
            for i in np.flatnonzero((self.phase == phase) & near):
                self.protect(int(i), int(i) + 1, "contact")
        self.protect(c, c + 1, "grasp_trigger")
        self.protect(o, o + 1, "release_trigger")
        # Preserve orientation changes that define reorientation/placement.
        omega = uniform_filter1d(
            np.r_[0.0, self.rotation_step * self.fps], size=3, mode="nearest"
        )
        for a, b in runs(omega >= self.rules.orientation_activity_deg_s):
            if a < b and self.phase[min(a, self.n - 1)] in (2, 3):
                self.protect(a, b, "orientation_activity")
        # A long, genuinely static hold is compressed only to roughly one second.
        self.detect_idle(max(close["end"], 0), min(open_["start"], self.n - 1))
        for a, b in (
            (0, close["start"]),
            (close["end"] + 1, open_["start"]),
            (open_["end"] + 1, self.n),
        ):
            if b > a:
                self.protect(a, a + 1, "phase_boundary")
                self.protect(b - 1, b, "phase_boundary")

    def detect_lift_apex(self, close_end, open_start):
        """End the episode at the first stable entry into its post-grasp Z maximum."""
        start = max(close_end + 1, 0)
        stop = min(open_start - 1, self.n - 1)
        if stop <= start:
            self.warnings.append(
                "lift_apex_not_found: invalid post-grasp search interval"
            )
            return
        state_z = uniform_filter1d(self.xyz[:, 2], size=5, mode="nearest")
        action_z = uniform_filter1d(self.action_xyz[:, 2], size=5, mode="nearest")
        signal = 0.7 * state_z + 0.3 * action_z
        baseline = float(np.median(signal[start : min(stop + 1, start + 5)]))
        local = signal[start : stop + 1]
        peak_value = float(np.max(local))
        rise = peak_value - baseline
        if rise < 0.01:
            self.warnings.append(
                "lift_apex_not_found: post-grasp Z rise below 1 cm; retained full episode"
            )
            return
        tolerance = max(0.002, min(0.005, rise * 0.1))
        plateau = np.flatnonzero(local >= peak_value - tolerance)
        if not len(plateau):
            self.warnings.append("lift_apex_not_found: no stable maximum plateau")
            return
        cut = int(start + plateau[0])
        self.active_end = cut
        self.truncation = {
            "kind": "lift_apex",
            "source_frame": cut,
            "search_start": start,
            "search_end": stop,
            "state_z_m": float(self.xyz[cut, 2]),
            "action_z_m": float(self.action_xyz[cut, 2]),
            "smoothed_z_m": float(signal[cut]),
            "baseline_z_m": baseline,
            "rise_m": rise,
            "plateau_tolerance_m": tolerance,
            "discarded_from_source_frame": cut + 1,
            "discarded_frame_count": self.n - cut - 1,
        }
        self.protect(cut, cut + 1, "lift_apex")

    def detect_idle(self, start, end):
        if end - start < round(self.rules.static_hold_s * self.fps):
            return
        left = np.maximum(np.arange(self.n - 1) - 2, 0)
        right = np.minimum(np.arange(self.n - 1) + 3, self.n - 1)
        dt = (right - left) / self.fps
        speed = np.linalg.norm(self.xyz[right] - self.xyz[left], axis=1) / dt
        a_speed = (
            np.linalg.norm(self.action_xyz[right] - self.action_xyz[left], axis=1) / dt
        )
        stable = (speed < self.rules.static_speed_m_s) & (
            a_speed < self.rules.static_speed_m_s
        )
        stable &= (
            angle(self.rot[left], self.rot[right]) / dt
            < self.rules.static_rotation_deg_s
        )
        stable &= (
            angle(self.action_rot[left], self.action_rot[right]) / dt
            < self.rules.static_rotation_deg_s
        )
        stable[:start] = False
        stable[end:] = False
        hold = round(self.rules.static_hold_s * self.fps)
        for a, b in runs(stable):
            if b - a <= hold:
                continue
            for cursor in range(a, b - hold, hold):
                finish = min(b, cursor + hold * 3)
                sl = slice(cursor, finish + 1)
                if (
                    np.max(np.linalg.norm(self.xyz[sl] - self.xyz[cursor], axis=1))
                    <= self.rules.static_position_range_m
                    and np.max(angle(self.rot[sl], self.rot[cursor]))
                    <= self.rules.static_orientation_range_deg
                ):
                    keep = np.unique(
                        np.rint(np.linspace(cursor, finish, hold + 1)).astype(int)
                    )
                    self.idle_spans.append(
                        {
                            "start": int(cursor),
                            "end": int(finish),
                            "keep": keep.tolist(),
                        }
                    )

    def allowed(self, i, j):
        if j == i + 1:
            return True, None
        r = self.rules
        if (
            np.linalg.norm(self.xyz[j] - self.xyz[i]) > r.max_step_m + 1e-9
            or np.linalg.norm(self.action_xyz[j] - self.action_xyz[i])
            > r.max_step_m + 1e-9
        ):
            return False, "step"
        if (
            self.rotation_sum[j] - self.rotation_sum[i] > r.max_rotation_deg
            or self.action_rotation_sum[j] - self.action_rotation_sum[i]
            > r.max_rotation_deg
        ):
            return False, "rotation"
        if (
            max(
                line_distance(self.xyz[i : j + 1], self.xyz[i], self.xyz[j]).max(),
                line_distance(
                    self.action_xyz[i : j + 1], self.action_xyz[i], self.action_xyz[j]
                ).max(),
            )
            > r.path_deviation_m + 1e-9
        ):
            return False, "deviation"
        return True, None

    def shortest_path(self, start, end):
        if end - start <= 1:
            return np.arange(start, end + 1)
        count = end - start + 1
        target = self.rules.target_speed_m_s / self.fps
        direction = self.xyz[end] - self.xyz[start]
        direction /= max(np.linalg.norm(direction), 1e-12)
        cost = np.full((count, count), np.inf)
        parent = np.full((count, count), -1, dtype=np.int32)
        max_gap = round(self.rules.max_skip_s * self.fps)
        for b in range(1, count):
            j = start + b
            for a in range(max(0, b - max_gap), b):
                i = start + a
                valid, reason = self.allowed(i, j)
                if not valid:
                    self.rejections[reason] += 1
                    continue
                delta = self.xyz[j] - self.xyz[i]
                distance = float(np.linalg.norm(delta))
                base = (
                    4 * (distance / max(target, 1e-9) - 1) ** 2
                    + distance / max(target, 1e-9)
                    + 0.2
                )
                base += 4 * max(0.0, -float(delta @ direction)) / max(target, 1e-9)
                if a == 0:
                    cost[a, b] = base
                    continue
                prev = np.flatnonzero(np.isfinite(cost[:a, a]))
                if not len(prev):
                    continue
                incoming = self.xyz[i] - self.xyz[start + prev]
                lengths = np.linalg.norm(incoming, axis=1)
                cos = np.clip(
                    incoming @ delta / np.maximum(lengths * distance, 1e-12), -1, 1
                )
                turn = np.where((lengths >= 0.001) & (distance >= 0.001), 1 - cos, 0.0)
                smooth = (
                    0.5
                    * np.sum((incoming - delta) ** 2, axis=1)
                    / max(target**2, 1e-12)
                )
                options = cost[prev, a] + base + 2 * turn + smooth
                k = int(np.argmin(options))
                cost[a, b] = options[k]
                parent[a, b] = prev[k]
        a = int(np.argmin(cost[:, -1]))
        if not np.isfinite(cost[a, -1]):
            raise RuntimeError(f"no valid path {start}->{end}")
        b = count - 1
        out = [b]
        while True:
            out.append(a)
            if a == 0:
                break
            a, b = int(parent[a, b]), a
        return np.asarray(out[::-1], dtype=int) + start

    def plan(self):
        self.detect_phases()
        self.protect(0, 1, "endpoint")
        self.protect(self.active_end, self.active_end + 1, "endpoint")
        # Controller-limited frames are observable behavior and must remain paired.
        for key in ("metadata.translation_limited", "metadata.rotation_limited"):
            if key in self.table.column_names:
                for i in np.flatnonzero(
                    np.asarray(self.table[key].to_pylist(), dtype=bool)
                ):
                    if i > self.active_end:
                        continue
                    self.protect(int(i), int(i) + 1, key.rsplit(".", 1)[-1])
        inherited = (
            (
                np.linalg.norm(np.diff(self.xyz[: self.active_end + 1], axis=0), axis=1)
                > self.rules.max_step_m
            )
            | (
                np.linalg.norm(
                    np.diff(self.action_xyz[: self.active_end + 1], axis=0), axis=1
                )
                > self.rules.max_step_m
            )
            | (self.rotation_step[: self.active_end] > self.rules.max_rotation_deg)
            | (
                self.action_rotation_step[: self.active_end]
                > self.rules.max_rotation_deg
            )
        )
        for i in np.flatnonzero(inherited):
            self.protect(int(i), int(i) + 2, "inherited_jump")
        keep = np.ones(self.n, dtype=bool)
        reasons = ["kept"] * self.n
        if self.active_end + 1 < self.n:
            keep[self.active_end + 1 :] = False
            reasons[self.active_end + 1 :] = ["discarded_after_lift_apex"] * (
                self.n - self.active_end - 1
            )
        # Optimize only within one phase and between protected anchors.
        for phase in FAST_PHASES:
            ids = np.arange(self.active_end + 1)[
                self.phase[: self.active_end + 1] == phase
            ]
            if len(ids) < 3:
                continue
            anchors = np.unique(np.r_[ids[0], ids[-1], ids[self.protected[ids]]])
            for left, right in itertools.pairwise(anchors):
                if right - left < 2 or self.phase[left] != self.phase[right]:
                    continue
                selected = self.shortest_path(int(left), int(right))
                keep[left : right + 1] = False
                keep[selected] = True
                for i in range(left + 1, right):
                    if not keep[i]:
                        reasons[i] = "motion_shortcut"
        selected = np.flatnonzero(keep)
        self.validate(selected, reasons)
        reports = []
        for phase in range(len(PHASES)):
            ids = np.flatnonzero(
                (self.phase == phase) & (np.arange(self.n) <= self.active_end)
            )
            if not len(ids):
                continue
            chosen = ids[keep[ids]]
            pairs = [
                (i, j)
                for i, j in itertools.pairwise(chosen)
                if not self.protected[i : j + 1].any()
            ]
            speed = (
                float(
                    np.mean(
                        [
                            np.linalg.norm(self.xyz[j] - self.xyz[i]) * self.fps
                            for i, j in pairs
                        ]
                    )
                )
                if pairs
                else None
            )
            reports.append(
                {
                    "phase": PHASES[phase],
                    "label": PHASE_LABELS[phase],
                    "start": int(ids[0]),
                    "end": int(ids[-1]),
                    "before": trajectory_metrics(self.xyz[ids], self.fps),
                    "after": trajectory_metrics(self.xyz[chosen], self.fps),
                    "fast_mean_m_s": speed,
                    "fast_intervals": len(pairs),
                    "target_met": None
                    if phase not in FAST_PHASES or speed is None
                    else abs(speed - self.rules.target_speed_m_s)
                    <= self.rules.speed_tolerance_m_s,
                    "retained_ratio": len(chosen) / len(ids),
                }
            )
        return {
            "selected": selected.tolist(),
            "phase_ids": self.phase.tolist(),
            "deletion_reasons": reasons,
            "protections": [",".join(sorted(x)) for x in self.protection],
            "events": self.events,
            "truncation": self.truncation,
            "original_frames": self.n,
            "active_frames_before_optimization": self.active_end + 1,
            "idle_spans": self.idle_spans,
            "warnings": self.warnings,
            "inherited_jump_source_frames": np.flatnonzero(inherited).tolist(),
            "candidate_rejections": self.rejections,
            "phases": reports,
            "before": trajectory_metrics(self.xyz[: self.active_end + 1], self.fps),
            "after": trajectory_metrics(self.xyz[selected], self.fps),
            "validation": {
                "source_frames_increasing": True,
                "protected_frames_retained": True,
                "new_edges_valid": True,
            },
        }

    def validate(self, selected, reasons):
        if (
            selected[0] != 0
            or selected[-1] != self.active_end
            or np.any(np.diff(selected) <= 0)
        ):
            raise ValueError("invalid selected endpoints/order")
        for i in np.flatnonzero(self.protected):
            if i <= self.active_end and i not in set(selected.tolist()):
                raise ValueError(f"dropped protected frame {i}")
        for i, j in itertools.pairwise(selected):
            if j == i + 1:
                continue
            if self.phase[i] != self.phase[j] or not self.allowed(int(i), int(j))[0]:
                raise ValueError(f"invalid shortcut {i}->{j}")


def analyze_episode(path, fps, rules):
    table = pq.ParquetFile(path).read(use_threads=False)
    result = Analyzer(table, fps, rules).plan()
    result.update(
        source_episode=int(Path(path).stem.split("_")[-1]), source_path=str(path)
    )
    return result


def numeric_stats(table, features):
    result = {}
    for key, spec in features.items():
        if spec["dtype"] in ("video", "image") or key not in table.column_names:
            continue
        values = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        result[key] = {
            name: fn(values, axis=0).tolist()
            for name, fn in (
                ("min", np.min),
                ("max", np.max),
                ("mean", np.mean),
                ("std", np.std),
            )
        }
        result[key]["count"] = [len(values)]
    return result


def aggregate_stats(items):
    result = {}
    for key in items[0]:
        entries = [item[key] for item in items if key in item]
        counts = np.asarray([v["count"][0] for v in entries], dtype=float)
        means = np.asarray([v["mean"] for v in entries])
        stds = np.asarray([v["std"] for v in entries])
        shape = (len(entries),) + (1,) * (means.ndim - 1)
        weights = (counts / counts.sum()).reshape(shape)
        mean = (weights * means).sum(axis=0)
        variance = (weights * (stds**2 + (means - mean) ** 2)).sum(axis=0)
        result[key] = {
            "min": np.min([v["min"] for v in entries], axis=0).tolist(),
            "max": np.max([v["max"] for v in entries], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(counts.sum())],
        }
    return result


def read_source(root):
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 10:
        raise ValueError("BUND optimizer requires LeRobot v2.1 at 10 Hz")
    episodes = sorted(
        (
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ),
        key=lambda x: x["episode_index"],
    )
    if [x["episode_index"] for x in episodes] != list(range(len(episodes))):
        raise ValueError("episode metadata must be contiguous")
    return info, episodes


def source_plan(root, wanted, rules, workers):
    info, episodes = read_source(root)
    selected_eps = sorted(set(range(len(episodes)) if wanted is None else wanted))
    if not selected_eps or any(i < 0 or i >= len(episodes) for i in selected_eps):
        raise ValueError("empty or unknown episode selection")
    jobs = [
        (
            root
            / info["data_path"].format(
                episode_chunk=i // info["chunks_size"], episode_index=i
            ),
            info["fps"],
            rules,
        )
        for i in selected_eps
    ]
    plans = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        futures = [pool.submit(analyze_episode, *job) for job in jobs]
        for future in as_completed(futures):
            plans.append(future.result())
    plans.sort(key=lambda x: x["source_episode"])
    for plan in plans:
        ep = episodes[plan["source_episode"]]
        if len(plan["selected"]) != len(set(plan["selected"])):
            raise ValueError("duplicate selected source frame")
        plan["tasks"] = ep["tasks"]
        plan["length"] = len(plan["selected"])
    return info, episodes, plans


def transform_table(table, selected, output_episode, global_start):
    out = table.take(pa.array(selected, type=pa.int64()))
    for key, values in (
        ("episode_index", np.full(len(selected), output_episode)),
        ("index", np.arange(global_start, global_start + len(selected))),
        ("frame_index", np.arange(len(selected))),
        ("timestamp", np.arange(len(selected), dtype=np.float32) / 10.0),
    ):
        if key in out.column_names:
            field = out.schema.field(key)
            out = out.set_column(
                out.schema.get_field_index(key), key, pa.array(values, type=field.type)
            )
    return out


def ffprobe(path):
    data = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(data)["streams"][0]


def encode_selected_video(source, destination, selected, fps):
    stream = ffprobe(source)
    width, height = int(stream["width"]), int(stream["height"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "0",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    selected = np.asarray(selected, dtype=np.int64)
    frame_bytes = width * height * 3
    next_pos = 0
    decoded = 0
    try:
        while True:
            frame = decoder.stdout.read(frame_bytes)
            if not frame:
                break
            if next_pos < len(selected) and decoded == selected[next_pos]:
                encoder.stdin.write(frame)
                next_pos += 1
            decoded += 1
        decoder.stdout.close()
        encoder.stdin.close()
        decoder_rc = decoder.wait(timeout=180)
        encoder_rc = encoder.wait(timeout=180)
    except BaseException:
        decoder.kill()
        encoder.kill()
        decoder.wait()
        encoder.wait()
        raise
    if decoder_rc or encoder_rc or next_pos != len(selected):
        raise RuntimeError(
            f"video extraction failed source={source} decoded={decoded} selected={next_pos}/{len(selected)}"
        )
    check = ffprobe(destination)
    if int(check.get("nb_frames", -1)) != len(selected):
        raise ValueError(f"output video frame mismatch: {destination}")
    num, den = map(float, check["r_frame_rate"].split("/"))
    if not np.isclose(num / den, fps):
        raise ValueError(f"output video FPS mismatch: {destination}")
    return {
        "source_frames_decoded": decoded,
        "selected_frames": len(selected),
        "codec": check.get("codec_name"),
        "sha256": digest(destination),
    }


def build_episode(job):
    root, stage, info, plan, output_episode, global_start = job
    source = root / info["data_path"].format(
        episode_chunk=plan["source_episode"] // info["chunks_size"],
        episode_index=plan["source_episode"],
    )
    table = pq.ParquetFile(source).read(use_threads=False)
    selected = np.asarray(plan["selected"], dtype=np.int64)
    start = int(selected[0])
    # v2.1 conversion emits one episode per parquet, so source frame 0 is row 0.
    out = transform_table(table, selected, output_episode, global_start)
    path = stage / info["data_path"].format(
        episode_chunk=output_episode // info["chunks_size"],
        episode_index=output_episode,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, path, compression="snappy")
    loaded = pq.ParquetFile(path).read(use_threads=False)
    if not loaded.equals(out, check_metadata=True):
        raise ValueError(f"parquet readback mismatch: {path}")
    for key in table.column_names:
        if key in ("episode_index", "index", "frame_index", "timestamp"):
            continue
        if not loaded[key].equals(table[key].take(pa.array(selected))):
            raise ValueError(
                f"source pairing mismatch in {key}: episode {plan['source_episode']}"
            )
    videos = {}
    for camera in CAMERAS:
        source_video = root / info["video_path"].format(
            episode_chunk=plan["source_episode"] // info["chunks_size"],
            video_key=camera,
            episode_index=plan["source_episode"],
        )
        destination = stage / info["video_path"].format(
            episode_chunk=output_episode // info["chunks_size"],
            video_key=camera,
            episode_index=output_episode,
        )
        videos[camera] = encode_selected_video(
            source_video, destination, selected, info["fps"]
        )
    return {
        "output_episode": output_episode,
        "length": len(selected),
        "source_start": start,
        "data_sha256": digest(path),
        "videos": videos,
        "numeric_stats": numeric_stats(loaded, info["features"]),
    }


def write_metadata(stage, info, source_episodes, plans, results, tasks, rules):
    meta = stage / "meta"
    meta.mkdir(exist_ok=True)
    updated = json.loads(json.dumps(info))
    total = sum(r["length"] for r in results)
    updated.update(
        total_episodes=len(results),
        total_frames=total,
        total_videos=len(results) * len(CAMERAS),
        total_chunks=math.ceil(len(results) / 1000),
        chunks_size=1000,
        splits={"train": f"0:{len(results)}"},
    )
    for camera in CAMERAS:
        updated["features"][camera].setdefault("info", {})["video.codec"] = "h264"
        updated["features"][camera]["info"]["video.fps"] = info["fps"]
    write_json(meta / "info.json", updated)
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in tasks)
    )
    write_jsonl = lambda path, rows: Path(path).write_text(
        "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows)
    )
    write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": i,
                "tasks": plans[i]["tasks"],
                "length": results[i]["length"],
            }
            for i in range(len(results))
        ],
    )
    episode_stats = []
    for i, result in enumerate(results):
        episode_stats.append({"episode_index": i, "stats": result["numeric_stats"]})
    write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    write_json(
        meta / "stats.json", aggregate_stats([x["stats"] for x in episode_stats])
    )
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
                "phase",
            ]
        )
        for i, plan in enumerate(plans):
            for out_frame, source_frame in enumerate(plan["selected"]):
                writer.writerow(
                    [
                        i,
                        plan["source_episode"],
                        out_frame,
                        source_frame,
                        sum(x["length"] for x in results[:i]) + out_frame,
                        source_frame,
                        PHASES[plan["phase_ids"][source_frame]],
                    ]
                )
    with (stage / "deletion_reasons.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_index",
                "source_episode",
                "source_frame",
                "kept",
                "reason",
                "phase",
                "protection",
            ]
        )
        for i, plan in enumerate(plans):
            selected = set(plan["selected"])
            for frame, reason in enumerate(plan["deletion_reasons"]):
                writer.writerow(
                    [
                        i,
                        plan["source_episode"],
                        frame,
                        int(frame in selected),
                        reason,
                        PHASES[plan["phase_ids"][frame]],
                        plan["protections"][frame],
                    ]
                )
    summary = summarize(plans, results, rules)
    write_json(stage / "summary.json", summary)
    write_json(
        stage / "reconstruction.json",
        {
            "strategy": "bund_real_frame_dag_v1",
            "source_root": str(SOURCE),
            "rules": asdict(rules),
            "fps": info["fps"],
            "script_sha256": digest(__file__),
            "state_action_pairing": "same source row; all 34 state and 10 action fields retained",
            "video_mode": "decode source episode videos and H264 encode selected source frames",
            "validation": {
                "episodes": len(results),
                "frames": total,
                "parquet_source_rows_verified": True,
                "video_selected_indices_decoded": True,
            },
        },
    )
    make_report(stage, plans, summary)


def summarize(plans, results, rules):
    phases = {}
    for plan in plans:
        for phase in plan["phases"]:
            item = phases.setdefault(
                phase["phase"],
                {
                    "label": phase["label"],
                    "count": 0,
                    "frames_before": 0,
                    "frames_after": 0,
                    "speeds": [],
                    "target_met": 0,
                    "target_count": 0,
                },
            )
            item["count"] += 1
            item["frames_before"] += phase["before"]["frames"]
            item["frames_after"] += phase["after"]["frames"]
            if phase["fast_mean_m_s"] is not None:
                item["speeds"].append(phase["fast_mean_m_s"])
            if phase["target_met"] is not None:
                item["target_count"] += 1
                item["target_met"] += int(phase["target_met"])
    for item in phases.values():
        item["mean_fast_speed_m_s"] = (
            float(np.mean(item.pop("speeds"))) if item.get("speeds") else None
        )
        item["target_fraction"] = (
            item["target_met"] / item["target_count"] if item["target_count"] else None
        )
    before = sum(p.get("original_frames", p["before"]["frames"]) for p in plans)
    active = sum(p["before"]["frames"] for p in plans)
    after = sum(r["length"] for r in results)
    return {
        "total_episodes": len(plans),
        "source_frames": before,
        "active_frames_before_optimization": active,
        "output_frames": after,
        "frame_reduction_fraction": 1 - after / before,
        "source_duration_s": before / 10,
        "active_duration_s": active / 10,
        "output_duration_s": after / 10,
        "target_speed_m_s": rules.target_speed_m_s,
        "target_tolerance_m_s": rules.speed_tolerance_m_s,
        "phases": phases,
        "ambiguous_episode_count": sum(bool(p["warnings"]) for p in plans),
        "truncated_episode_count": sum(bool(p.get("truncation")) for p in plans),
        "discarded_after_apex_frames": before - active,
        "inherited_jump_count": sum(
            len(p["inherited_jump_source_frames"]) for p in plans
        ),
    }


def make_report(stage, plans, summary):
    rows = []
    for p in plans:
        rows.append(
            {
                "episode": p["source_episode"],
                "selected": p["selected"],
                "before": p["before"],
                "after": p["after"],
                "original_frames": p.get("original_frames"),
                "truncation": p.get("truncation"),
                "phases": p["phases"],
                "warnings": p["warnings"],
            }
        )
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html><meta charset='utf-8'><title>BUND trajectory reconstruction</title><style>body{{font:14px system-ui;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:6px}}canvas{{border:1px solid #bbb;width:900px;height:420px}}</style><h1>BUND 轨迹重构报告</h1><pre id='summary'></pre><select id='ep'></select><canvas id='plot' width='900' height='420'></canvas><div id='table'></div><script>const data={data};const summary={json.dumps(summary, ensure_ascii=False)};summary.textContent=JSON.stringify(summary,null,2);const sel=document.querySelector('#ep');data.forEach((e,i)=>sel.add(new Option('episode '+e.episode,i)));function draw(){{const e=data[+sel.value],c=document.querySelector('#plot'),x=c.getContext('2d'),p=e.before; x.clearRect(0,0,c.width,c.height); const pts=p.frames; const z=e.selected.map(i=>i/Math.max(1,pts-1)); x.strokeStyle='#638fb3';x.beginPath();for(let i=0;i<pts;i++){{const q=e.before; const xx=30+830*i/Math.max(1,pts-1); const yy=210-100*Math.sin(i/pts*6.28);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}}x.stroke();x.strokeStyle='#d4774d';x.beginPath();z.forEach((v,i)=>{{const xx=30+830*v,yy=210-100*Math.sin(v*6.28);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});x.stroke();document.querySelector('#table').innerHTML='<table><tr><th>阶段</th><th>帧数</th><th>均速</th><th>达标</th></tr>'+e.phases.map(p=>`<tr><td>${{p.label}}</td><td>${{p.before.frames}} → ${{p.after.frames}}</td><td>${{p.fast_mean_m_s==null?'—':p.fast_mean_m_s.toFixed(3)}}</td><td>${{p.target_met==null?'保护阶段':p.target_met?'是':'否'}}</td></tr>`).join('')+'</table><p>'+e.warnings.join('; ')+'</p>'}}sel.onchange=draw;draw();</script>"""
    (stage / "report.html").write_text(html)


def validate_output(stage, plans, info):
    out_info = json.loads((stage / "meta/info.json").read_text())
    cursor = 0
    for i, plan in enumerate(plans):
        path = stage / info["data_path"].format(
            episode_chunk=i // info["chunks_size"], episode_index=i
        )
        table = pq.ParquetFile(path).read(
            columns=[
                "episode_index",
                "index",
                "frame_index",
                "timestamp",
                "observation.state",
                "action",
            ]
        )
        n = len(plan["selected"])
        if (
            len(table) != n
            or not np.array_equal(np.asarray(table["episode_index"]), np.full(n, i))
            or not np.array_equal(
                np.asarray(table["index"]), np.arange(cursor, cursor + n)
            )
            or not np.array_equal(np.asarray(table["frame_index"]), np.arange(n))
            or not np.allclose(
                np.asarray(table["timestamp"]), np.arange(n) / 10, rtol=0, atol=1e-5
            )
        ):
            raise ValueError(f"metadata timeline mismatch episode {i}")
        if np.asarray(table["observation.state"].to_pylist()).shape != (
            n,
            34,
        ) or np.asarray(table["action"].to_pylist()).shape != (n, 10):
            raise ValueError("output state/action shape mismatch")
        cursor += n
    if (
        out_info["total_frames"] != cursor
        or len(list((stage / "videos").glob("*/*/*.mp4"))) != out_info["total_videos"]
    ):
        raise ValueError("output totals mismatch")


def build(plan, root, output, workers):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output}")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    print(f"Staging: {stage}", file=sys.stderr, flush=True)
    results = [None] * len(plan["plans"])
    cursor = 0
    jobs = []
    for i, p in enumerate(plan["plans"]):
        jobs.append((root, stage, plan["info"], p, i, cursor))
        cursor += len(p["selected"])
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(build_episode, job): job[4] for job in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                print(f"Build/verify {done}/{len(jobs)}", file=sys.stderr, flush=True)
        write_metadata(
            stage,
            plan["info"],
            plan["episodes"],
            plan["plans"],
            results,
            plan["tasks"],
            plan["rules"],
        )
        validate_output(stage, plan["plans"], plan["info"])
        for path, before in plan["source_hashes"].items():
            if digest(path) != before:
                raise ValueError(f"source changed during build: {path}")
        stage.rename(output)
    except BaseException:
        print(f"Build failed; staging retained: {stage}", file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-episodes", action="store_true")
    selection.add_argument("--episode", type=int, action="append")
    parser.add_argument("--rules-json", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    root = args.source_root.resolve()
    output = (args.output or root.with_name(OUTPUT_NAME)).absolute()
    if (
        root == output.resolve()
        or root in output.resolve().parents
        or output.resolve() in root.parents
    ):
        parser.error("output must be independent of source")
    if not root.is_dir():
        parser.error(f"source does not exist: {root}")
    rules = Rules.read(args.rules_json)
    started = time.monotonic()
    info, episodes, plans = source_plan(
        root, None if args.all_episodes else args.episode, rules, args.workers
    )
    source_paths = {
        root
        / info["data_path"].format(
            episode_chunk=p["source_episode"] // info["chunks_size"],
            episode_index=p["source_episode"],
        )
        for p in plans
    }
    source_paths |= {
        root
        / info["video_path"].format(
            episode_chunk=p["source_episode"] // info["chunks_size"],
            video_key=camera,
            episode_index=p["source_episode"],
        )
        for p in plans
        for camera in CAMERAS
    }
    plan = {
        "source_root": str(root),
        "info": info,
        "episodes": episodes,
        "tasks": json.loads(
            "[" + ",".join((root / "meta/tasks.jsonl").read_text().splitlines()) + "]"
        ),
        "plans": plans,
        "rules": rules,
        "source_hashes": {p: digest(p) for p in source_paths},
    }
    summary = summarize(plans, [{"length": len(p["selected"])} for p in plans], rules)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.plan_only:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    build(plan, root, output, args.workers)
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_s": round(time.monotonic() - started, 1),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
