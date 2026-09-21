#!/usr/bin/env python3
"""同一 H01 洗碗任务的五阶段抽帧重构；只选择真实来源帧，不插值。

数据处理框架（每个 episode 独立并行分析，完成后按源 episode 排序）
===============================================================

  读取 meta/info.json、episodes.jsonl、tasks.jsonl
                         |
                         v
  读取 episode parquet + 原始状态、动作、末端位姿、任务标签
                         |
                         v
  EpisodeAnalyzer：解析字段和左右臂
      |
      +--> 计算末端速度、关节速度、辅助状态变化、四元数角度
      +--> 检测夹爪事件并建立保护帧
      +--> 检测有效起点，删除启动试探和无效前缀
      +--> 检测双臂静止帧、右-左-右-左-右五个阶段
      +--> 标记阶段边界、Z 极值、主要转向和复位方向锚点
                         |
                         v
  plan(include_smooth=False)：原五阶段候选抽帧
      |
      +--> 静止删除、低速谷压缩、冗余回摆压缩
      +--> 阶段二极值保护、阶段四复位密集保帧
      +--> 阶段五单向退离候选和动态约束搜索
      +--> 保留事件、锚点、路径形状和对侧臂状态
                         |
                         v
  [可选] --target-duration -> DurationNormalizer
      |
      +--> 修复旧选帧造成的过大位姿连接
      +--> 二阶路径搜索，向软时长目标靠拢
                         |
                         v
  smooth_plan：平滑点到点后置抽帧
      |
      +--> 只在方向稳定、低曲率、无事件区间尝试抽帧
      +--> 目标速度 0.5 m/s，默认容差 +/-0.15 m/s
      +--> 只从已有选帧中删除，不恢复或重排帧
                         |
                         v
  选帧验证：索引递增、事件完整、路径偏差、位姿/动态/辅助状态约束
                         |
             +-----------+------------+----------------+
             |           |            |                |
             v           v            v                v
       parquet     source_frame_map  vs/episode   选定相机视频
       字段抽取      来源映射 CSV      删除记录       同源帧抽取编码
             |           |            |                |
             +-----------+------------+----------------+
                         |
                         v
  重建 meta、统计、任务/标定引用、PNG 时长图、JSON 报告
                         |
                         v
  临时目录完整验证（来源行、视频帧、哈希、schema）
                         |
                         v
  原子替换 output/<dataset>/；失败则保留上一版

默认处理原数据集 episode 0。--plan-only 只分析，不写数据或编码视频。
依赖：numpy、scipy、pyarrow、plotly，以及 ffmpeg/ffprobe。
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import distance_transform_edt, median_filter, uniform_filter1d
from scipy.signal import find_peaks

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # 启动时固定版本，避免运行中编辑代码污染缓存。
DEFAULT_SOURCE = Path('/shared_disk/users/lv.feng/data/private_data/robot/rect/COL26071359B_rect11/data/chunk-000/episode_000000.parquet')  # 仅为默认输入，算法不依赖路径。
PHASE_ARMS = ('right', 'left', 'right', 'left', 'right')  # 洗碗任务的主臂交替先验。
PHASE_NAMES = ('右臂倒垃圾', '左臂放碗和顶盖', '右臂放碗和顶盖', '左臂放碗和复位', '右臂复位')  # 运动阶段的任务含义。


# 所有规则只含物理阈值、时长和权重，不接受帧号或 episode 特例。
@dataclass(frozen=True)
class Rules:
    activity_speed: float = .025
    static_speed: float = .02
    static_joint_speed: float = .02
    static_aux_speed: float = .02
    chassis_epsilon: float = .001
    detection_smoothing_s: float = .3
    dominance_ratio: float = 3.3
    minimum_activity_s: float = .27
    startup_window_s: float = .27
    startup_persistence_s: float = 1.17
    startup_speed: float = .05
    startup_direction_cos: float = .55
    startup_backtrack_tolerance_m: float = .0001
    gripper_start_rate: float = .015
    gripper_continue_rate: float = .00375
    dense_gap_s: float = 1/6
    min_keep_ratio: float = .7
    min_path_ratio: float = .97
    path_deviation_m: float = .003
    loop_gap_s: float = 4/3
    loop_return_m: float = .06
    loop_min_path_m: float = .025
    loop_path_ratio: float = .65
    departure_backtrack_tolerance_m: float = .001
    departure_gap_s: float = 3.
    departure_dynamic_scales: tuple[float, ...] = (1., 1.25, 1.5, 2., 2.5, 3.)
    max_step_m: float = .12
    max_other_step_m: float = .01
    max_orientation_step_deg: float = 30.
    valley_smoothing_s: float = .1
    valley_distance_s: float = .2
    valley_width_s: float = 2/30
    valley_prominence_ratio: float = .3
    turn_window_s: float = 1/6
    low_turn_deg: float = 30.
    major_turn_deg: float = 60.
    major_turn_excursion_m: float = .015
    dynamics_rtol: float = 1e-5
    dynamics_atol: float = 1e-6
    target_speed_scales: tuple[float, ...] = (.8, 1., 1.2)
    turn_weights: tuple[float, ...] = (.35, 1., 2., 5.)
    edge_penalty: float = .15
    retime_max_gap_s: float = 1/6
    retime_max_step_m: float = .04
    retime_max_joint_step_rad: float = .08
    retime_max_rotation_deg: float = 5.
    retime_event_ramp_s: float = 8/30
    retime_smoothness: float = .2
    retime_min_sampling_ratio: float = .15
    retime_edge_penalty: float = .015
    smooth_speed_target_m_s: float = .5
    smooth_speed_tolerance_m_s: float = .15
    smooth_min_duration_s: float = .3
    smooth_direction_cos: float = .96
    smooth_max_gap_s: float = .5

    # 严格验证共享配置，防止通过配置文件重新引入固定帧策略。
    @classmethod
    def from_json(cls, path=None):
        values = {} if path is None else json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(values, dict) or set(values)-{f.name for f in fields(cls)}:
            raise ValueError('规则只允许已声明的阈值/时长参数，不接受固定帧或未知字段')
        for key in ('target_speed_scales', 'turn_weights', 'departure_dynamic_scales'):
            if key in values:
                if not isinstance(values[key], list) or not 1 <= len(values[key]) <= 32:
                    raise ValueError('候选权重需要包含 1–32 个有限正数')
                values[key] = tuple(values[key])
        rules = cls(**values)
        scalar_values = [item for value in asdict(rules).values() for item in (value if isinstance(value, tuple) else (value,))]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0
               for v in scalar_values):
            raise ValueError('规则参数必须为有限正数')
        if not 0 < rules.min_keep_ratio <= 1 or not 0 < rules.min_path_ratio <= 1 or not 0 < rules.loop_path_ratio < 1:
            raise ValueError('保留比例必须在 (0,1]，回摆路径比例必须在 (0,1)')
        if rules.gripper_continue_rate > rules.gripper_start_rate or rules.low_turn_deg >= rules.major_turn_deg:
            raise ValueError('事件持续阈值或转向阈值顺序不正确')
        if rules.departure_dynamic_scales[0] != 1 or any(b <= a for a, b in zip(rules.departure_dynamic_scales, rules.departure_dynamic_scales[1:])):
            raise ValueError('退离动态候选倍率必须从1开始严格递增')
        if rules.retime_min_sampling_ratio > 1:
            raise ValueError('时长优化最低采样比例不得大于1')
        if not 0 < rules.smooth_direction_cos <= 1:
            raise ValueError('平滑段方向一致性阈值必须在 (0,1]')
        return rules


# 严格递增源帧与分析报告是视频、parquet、元数据重建的唯一输入。
@dataclass
class EpisodePlan:
    selected: np.ndarray
    report: dict
    phase_ids: np.ndarray


# 文件摘要用于来源审计，不用于允许或禁止某一数据集。
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


# 报告拒绝 NaN/Inf，便于浏览器和其他语言直接读取。
def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


# 返回包含起点、不包含终点的连续真值区间。
def runs(mask, minimum=1):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= minimum]


class EpisodeAnalyzer:
    # 按元数据定位特征维度；任何滤波结果都只用于分析。
    def __init__(self, table, info, rules=None):
        self.table, self.info, self.rules = table, info, rules or Rules()
        self.fps, self.n = float(info['fps']), table.num_rows
        if self.n < 1 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('episode 不能为空且 FPS 必须为有限正数')
        self.state = self.vector('observation.state')
        self.action = self.vector('action')
        self.ee = self.vector('observation.state_endpose_quat')
        names = info['features']['observation.state_endpose_quat']['names']
        self.state_names = info['features']['observation.state']['names']
        self.action_names = info['features']['action']['names']
        self.positions, self.quaternions, self.joints = {}, {}, {}
        self.speed, self.filtered_speed = {}, {}
        for arm in ('left', 'right'):
            self.positions[arm] = self.ee[:, [names.index(f'{arm}_eef_pos_{axis}') for axis in 'xyz']]
            quat = self.ee[:, [names.index(f'{arm}_eef_quat_{axis}') for axis in 'xyzw']]
            norm = np.linalg.norm(quat, axis=1)
            if np.any(norm < 1e-6):
                raise ValueError('末端四元数无效')
            self.quaternions[arm] = quat/norm[:, None]
            self.joints[arm] = [i for i, name in enumerate(self.state_names) if name.startswith(arm+'_') and 'gripper' not in name]
            if len(self.joints[arm]) != 7:
                raise ValueError('需要元数据定义的每臂七个关节')
            self.speed[arm] = np.r_[0., np.linalg.norm(np.diff(self.positions[arm], axis=0), axis=1)*self.fps]
            self.filtered_speed[arm] = uniform_filter1d(self.speed[arm], size=self.samples(self.rules.detection_smoothing_s, odd=True), mode='nearest')
        self.gripper_events = self.detect_grippers()
        self.protected = np.zeros(self.n, dtype=bool)
        for event in self.gripper_events:
            self.protected[event['start']:event['end']+1] = True
        tasks = np.asarray(table['task_index'].to_pylist())
        for k in np.flatnonzero(tasks[1:] != tasks[:-1])+1:
            self.protected[k-1:k+1] = True
        self.event_protected = self.protected.copy()
        joint_names = {self.state_names[i] for indices in self.joints.values() for i in indices}
        self.joint_values = np.c_[self.state[:, self.joints['left']+self.joints['right']],
                                  self.action[:, [i for i, name in enumerate(self.action_names) if name in joint_names]]]
        self.joint_arms = np.asarray(['left']*7+['right']*7+
                                    [name.split('_', 1)[0] for name in self.action_names if name in joint_names])
        self.other_values = np.c_[self.state[:, [i for i, name in enumerate(self.state_names) if name not in joint_names]],
                                  self.action[:, [i for i, name in enumerate(self.action_names) if name not in joint_names]]]
        self.static_mask = self.detect_static()
        self.phase_ids = np.full(self.n, -1, dtype=int)
        self.detection = {}

    # 将时间阈值换成当前 FPS 的采样数，奇数窗口避免检测相位偏置。
    def samples(self, seconds, odd=False):
        n = max(1, int(round(seconds*self.fps)))
        return n if not odd or n % 2 else n+1

    # Arrow 列统一读成二维有限数组，兼容 list 和 ndarray 来源。
    def vector(self, name):
        array = np.asarray(self.table[name].to_pylist(), dtype=float)
        if array.ndim != 2 or len(array) != self.n or not np.isfinite(array).all():
            raise ValueError(f'无效数值列：{name}')
        return array

    # 开合事件保留整个连续过程，包括低于启动阈值的收尾。
    def detect_grippers(self):
        events = []
        for arm in ('left', 'right'):
            values = self.state[:, self.state_names.index(arm+'_gripper')]
            rate = abs(np.diff(values))*self.fps
            for a, b in runs(rate > self.rules.gripper_continue_rate):
                if rate[a:b].max() > self.rules.gripper_start_rate:
                    events.append({'arm': arm, 'start': a, 'end': b})
        return sorted(events, key=lambda event: event['start'])

    # 每个静止采样同时检查末端、关节、夹爪、头腰和底盘命令，无最短时长门槛。
    def detect_static(self):
        r = self.rules
        ds = np.r_[np.zeros((1, self.state.shape[1])), abs(np.diff(self.state, axis=0))*self.fps]
        idle = (self.speed['left'] <= r.static_speed) & (self.speed['right'] <= r.static_speed)
        for indices in self.joints.values():
            idle &= np.max(ds[:, indices], axis=1) <= r.static_joint_speed
        aux = [i for i, name in enumerate(self.state_names) if name.startswith(('head_', 'waist_'))]
        if aux:
            idle &= np.max(ds[:, aux], axis=1) <= r.static_aux_speed
        for arm in ('left', 'right'):
            idle &= ds[:, self.state_names.index(arm+'_gripper')] <= r.gripper_continue_rate
        if 'observation.chassis_cmd_vel' in self.table.column_names:
            idle &= np.max(abs(self.vector('observation.chassis_cmd_vel')), axis=1) <= r.chassis_epsilon
        chassis = [i for i, name in enumerate(self.action_names) if name.startswith('cmd_vel_')]
        if chassis:
            idle &= np.max(abs(self.action[:, chassis]), axis=1) <= r.chassis_epsilon
        return idle & ~self.protected

    # 从头识别右臂持续、同向启动，排除左臂活动与双臂短暂试探。
    def find_start(self):
        r = self.rules
        self.startup_note = None
        w, future = self.samples(r.startup_window_s), self.samples(r.startup_persistence_s)
        delta = np.diff(self.positions['right'], axis=0)
        for s in range(max(0, self.n-w-future)):
            right, left = self.speed['right'][s+1:s+w+1], self.speed['left'][s+1:s+w+1]
            if np.mean(right > r.startup_speed) < .75 or np.mean(left <= r.static_speed) < .75:
                continue
            unit = delta[s:s+w]/np.maximum(np.linalg.norm(delta[s:s+w], axis=1, keepdims=True), 1e-12)
            direction = unit.mean(axis=0)
            direction /= max(np.linalg.norm(direction), 1e-12)
            if float(np.mean(unit@direction)) < r.startup_direction_cos:
                continue
            if np.any(delta[s:s+w]@direction < -r.startup_backtrack_tolerance_m):
                continue
            if np.mean(self.speed['right'][s+w+1:s+w+future+1] > r.startup_speed) < .55:
                continue
            first = s+1
            if any(event['end'] < first for event in self.gripper_events):
                self.startup_note = '持续右臂启动前已有完整夹爪事件，保留准备段以避免删除真实抓放'
                return 0
            for event in self.gripper_events:
                if event['start'] < first <= event['end']:
                    first = event['start']
            return first
        return 0

    # 交替主臂块对齐五个有序阶段；真实额外活动归入阶段并标记变体。
    def detect_phases(self):
        first = self.find_start()
        l, r = self.filtered_speed['left'], self.filtered_speed['right']
        labels = np.zeros(self.n, dtype=int)
        labels[(r > self.rules.activity_speed) & (r > self.rules.dominance_ratio*l)] = 1
        labels[(l > self.rules.activity_speed) & (l > self.rules.dominance_ratio*r)] = 2
        blocks = []
        for value, arm in ((1, 'right'), (2, 'left')):
            for a, b in runs(labels == value, self.samples(self.rules.minimum_activity_s)):
                if b > first:
                    blocks.append({'arm': arm, 'start': max(first, a), 'end': b-1})
        blocks.sort(key=lambda block: block['start'])
        merged = []
        for block in blocks:
            if merged and merged[-1]['arm'] == block['arm']:
                merged[-1]['end'] = block['end']
            else:
                merged.append(dict(block))
        sequence = ''.join(block['arm'][0].upper() for block in merged)
        self.detection = {'first_source_frame': first, 'dominant_blocks': merged, 'sequence': sequence,
                          'status': 'standard' if sequence == 'RLRLR' else 'variant',
                          'startup_note': self.startup_note}
        count = len(merged)
        if count < 5 or not merged or merged[0]['arm'] != 'right':
            self.detection['status'] = 'unsupported'
            return []
        cost = np.full((6, count+1), np.inf)
        parent = np.full((6, count+1), -1, dtype=int)
        cost[0, 0] = 0.
        for phase in range(1, 6):
            arm = PHASE_ARMS[phase-1]
            for end in range(phase, count+1):
                for begin in range(phase-1, end):
                    group = merged[begin:end]
                    if not any(block['arm'] == arm for block in group):
                        continue
                    mismatch = sum(block['end']-block['start']+1 for block in group if block['arm'] != arm)
                    value = cost[phase-1, begin]+mismatch
                    if value < cost[phase, end]:
                        cost[phase, end], parent[phase, end] = value, begin
        if not np.isfinite(cost[5, count]):
            self.detection['status'] = 'unsupported'
            return []
        groups, end = [], count
        for phase in range(5, 0, -1):
            begin = int(parent[phase, end]); groups.append(merged[begin:end]); end = begin
        groups.reverse()
        boundaries = [first]+[(a[-1]['end']+b[0]['start']+1)//2 for a, b in zip(groups[:-1], groups[1:])]+[self.n]
        result = []
        for i, (arm, group) in enumerate(zip(PHASE_ARMS, groups)):
            a, b = boundaries[i], boundaries[i+1]-1
            variant = any(block['arm'] != arm for block in group)
            phase = {'id': i+1, 'name': PHASE_NAMES[i], 'arm': arm, 'start': a, 'end': b,
                     'status': 'variant' if variant else 'standard', 'direction_notes': []}
            z = self.positions[arm][a:b+1, 2]
            if i == 1:
                low = int(np.argmin(z))
                if not (0 < low < len(z)-1 and z[0]-z[low] > .01 and z[low:].max()-z[low] > .01):
                    phase['direction_notes'].append('未观察到完整先下后上，保留实际方向')
            if i in (3, 4):
                window = min(len(z)-1, self.samples(.33))
                if z[-1]-z[-1-window] >= 0:
                    phase['direction_notes'].append('末段不是下降，保留实际方向')
            self.phase_ids[a:b+1] = i+1
            result.append(phase)
        return result

    # 速度统计仅对非零速度计算均值/中位数；差分峰值仍包含全部时间步。
    def metrics(self, ids, arm):
        positions = self.positions[arm][np.asarray(ids, dtype=int)]
        velocity = np.diff(positions, axis=0)*self.fps
        speed = np.linalg.norm(velocity, axis=1)
        moving = speed[speed > 0]
        acceleration = np.linalg.norm(np.diff(velocity, axis=0)*self.fps, axis=1)
        jerk = np.linalg.norm(np.diff(velocity, n=2, axis=0)*self.fps**2, axis=1)
        return {'frames': len(positions), 'duration_s': max(0, len(positions)-1)/self.fps,
                'path_length_m': float(speed.sum()/self.fps),
                'speed_mean_m_s': float(moving.mean()) if len(moving) else 0.,
                'speed_median_m_s': float(np.median(moving)) if len(moving) else 0.,
                'speed_cv': float(moving.std()/moving.mean()) if len(moving) else 0.,
                'speed_min_m_s': float(speed.min()) if len(speed) else 0.,
                'speed_max_m_s': float(speed.max(initial=0)), 'zero_speed_steps': int(np.count_nonzero(speed == 0)),
                'acceleration_peak_m_s2': float(acceleration.max(initial=0)),
                'jerk_peak_m_s3': float(jerk.max(initial=0))}

    # 统一低谷统计，同时给出前后运动方向和空间尺度，区分真实转弯与冗余减速。
    def valleys(self, ids, arm):
        ids = np.asarray(ids, dtype=int)
        if len(ids) < 5:
            return []
        positions = self.positions[arm][ids]
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)*self.fps
        moving = speed[speed > 0]
        if not len(moving):
            return []
        smoothed = median_filter(speed, size=self.samples(self.rules.valley_smoothing_s, odd=True), mode='nearest')
        peaks, props = find_peaks(-smoothed, distance=self.samples(self.rules.valley_distance_s),
                                 width=max(1, self.rules.valley_width_s*self.fps),
                                 prominence=float(np.median(moving))*self.rules.valley_prominence_ratio)
        window = self.samples(self.rules.turn_window_s)
        result = []
        for index, k in enumerate(peaks):
            node = int(k)+1
            entering = positions[node]-positions[max(0, node-window)]
            exiting = positions[min(len(ids)-1, node+window)]-positions[node]
            norms = np.linalg.norm(entering)*np.linalg.norm(exiting)
            angle = float(np.degrees(np.arccos(np.clip(entering@exiting/max(norms, 1e-20), -1, 1))))
            extent = float(min(np.linalg.norm(entering), np.linalg.norm(exiting)))
            result.append({'frame': int(ids[node]), 'speed': float(smoothed[k]), 'angle_deg': angle,
                           'excursion_m': extent, 'prominence': float(props['prominences'][index]),
                           'major_turn': bool(angle >= self.rules.major_turn_deg and extent >= self.rules.major_turn_excursion_m)})
        return result

    # 按输出采样周期计算各受控量峰值，包含两臂关节和 action 的同名维度。
    def dynamic_peaks(self, ids):
        ids = np.asarray(ids, dtype=int)
        result = {}
        for arm in ('left', 'right'):
            values = self.positions[arm][ids]
            for order, name in ((1, 'speed'), (2, 'acceleration'), (3, 'jerk')):
                delta = np.diff(values, n=order, axis=0)*self.fps**order
                result[arm+'_'+name] = np.array([np.linalg.norm(delta, axis=1).max(initial=0)])
        for order, name in ((1, 'joint_speed'), (2, 'joint_acceleration'), (3, 'joint_jerk')):
            delta = abs(np.diff(self.joint_values[ids], n=order, axis=0))*self.fps**order
            result[name] = delta.max(axis=0, initial=0)
        result['other_state_action_speed'] = (abs(np.diff(self.other_values[ids], axis=0))*self.fps).max(axis=0, initial=0)
        return result

    # 拼接检查含两侧各三个保留帧，完整覆盖三阶差分。
    def boundary_ids(self, selected, phase):
        first = int(np.searchsorted(selected, phase['start']))
        last = int(np.searchsorted(selected, phase['end'], side='right'))
        return selected[max(0, first-3):min(len(selected), last+3)]

    # 删除边界静止帧后，源对照必须覆盖实际取到的邻接帧，避免比较不同运动范围。
    def source_boundary_ids(self, selected, phase):
        ids = self.boundary_ids(selected, phase)
        start = min(phase['start']-3, int(ids[0])) if len(ids) else phase['start']-3
        end = max(phase['end']+3, int(ids[-1])) if len(ids) else phase['end']+3
        return np.arange(max(0, start), min(self.n, end+1))

    # 与原阶段及同等边界比较，返回实际违反的量而不是隐藏约束失败。
    def violations(self, selected, phase):
        ids = self.boundary_ids(selected, phase)
        baseline = self.source_boundary_ids(selected, phase)
        source, output = self.dynamic_peaks(baseline), self.dynamic_peaks(ids)
        r = self.rules
        return [name for name in source if np.any(output[name] > source[name]*(1+r.dynamics_rtol)+r.dynamics_atol)]

    # 已接受平滑提速后，计划拼接检查仍保留对侧臂和辅助量的原始动态约束。
    def plan_violations(self, selected, phase):
        reference = getattr(self, '_smooth_reference_ids', None)
        baseline = self.dynamic_peaks(self.boundary_ids(reference, phase) if reference is not None
                                      else self.source_boundary_ids(selected, phase))
        output = self.dynamic_peaks(self.boundary_ids(selected, phase))
        failures = [name for name in baseline if np.any(output[name] > baseline[name] *
                    (1+self.rules.dynamics_rtol) + self.rules.dynamics_atol)]
        if not phase.get('smooth_speed_applied'):
            return failures
        arm = phase['arm']
        exempt = {f'{arm}_{name}' for name in ('speed', 'acceleration', 'jerk')}
        for name in ('joint_speed', 'joint_acceleration', 'joint_jerk'):
            if name in failures:
                mask = self.joint_arms != arm
                if np.all(output[name][mask] <= baseline[name][mask] * (1+self.rules.dynamics_rtol) + self.rules.dynamics_atol):
                    exempt.add(name)
        return [name for name in failures if name not in exempt]

    # 第五阶段先退离到下一次抓放位置；用事件分段，避免把后续抬手复位强行投影成同一直线。
    def departure_window(self, phase):
        a, b, arm = phase['start'], phase['end'], phase['arm']
        events = [event for event in self.gripper_events
                  if event['arm'] == arm and a < event['start'] <= b]
        end = min((event['start'] for event in events), default=b)
        origin = self.positions[arm][a]
        vector = self.positions[arm][end]-origin
        distance = float(np.linalg.norm(vector))
        report = {'start': a, 'end': end, 'end_anchor': 'next_gripper_event' if events else 'phase_end',
                  'tolerance_m': self.rules.departure_backtrack_tolerance_m}
        phase['monotonic_departure'] = report
        if distance < self.rules.loop_min_path_m or end-a < 3:
            report.update(status='unsupported', reason='退离净位移不足，不能可靠定义单向方向')
            return None
        direction = vector/distance
        report.update(origin=origin.tolist(), direction=direction.tolist())
        progress = (self.positions[arm][a:end+1]-origin)@direction
        report['before_max_backtrack_m'] = float((np.maximum.accumulate(progress)-progress).max())
        report['status'] = 'pending'
        if phase['status'] != 'standard':
            report.update(status='unsupported', reason='包含额外对侧活动，保留阶段变体')
            return None
        if report['before_max_backtrack_m'] <= self.rules.departure_backtrack_tolerance_m:
            return None
        # 单向退离规则优先于无事件的回摆转向锚点；夹爪及任务事件仍不可删除。
        self.protected[a+1:end] = self.event_protected[a+1:end]
        moving = self.speed[arm][a:end+1]
        moving = moving[moving > 0]
        return {'kind': 'monotonic_departure', 'start': a, 'end': end,
                'origin': origin.tolist(), 'direction': direction.tolist(),
                'target': float(np.median(moving)) if len(moving) else self.rules.activity_speed}

    # 最大累计回撤衡量真正回摆，不能靠增大 FPS 将一次回退拆成许多小步通过验收。
    def departure_backtrack(self, ids, phase):
        rule = phase['monotonic_departure']
        local = ids[(ids >= rule['start']) & (ids <= rule['end'])]
        progress = (self.positions[phase['arm']][local]-rule['origin'])@np.asarray(rule['direction'])
        return float((np.maximum.accumulate(progress)-progress).max(initial=0))

    # 只依据活动与方向识别平滑点到点子段，不依赖已有来源帧列表。
    def smooth_windows(self, phase):
        arm, a, b = phase['arm'], phase['start'], phase['end']
        rows = np.arange(a, b+1)
        if phase['status'] != 'standard':
            return []
        windows = []
        # 对方向稳定的点到点运动建立独立候选，使抽帧后的速度接近统一目标。
        smooth_duration = self.samples(self.rules.smooth_min_duration_s)
        segment_speed = np.linalg.norm(np.diff(self.positions[arm][rows], axis=0), axis=1) * self.fps
        smooth_mask = segment_speed > self.rules.activity_speed
        smooth_mask &= ~self.protected[rows[:-1]] & ~self.protected[rows[1:]]
        max_smooth_samples = max(smooth_duration, self.samples(2.0))
        for run_start, run_stop in runs(smooth_mask, smooth_duration):
            cursor = run_start
            while cursor + smooth_duration <= run_stop:
                best = None
                # 在转向前截断，允许同一阶段包含多个平滑点到点子段。
                for end in range(cursor + smooth_duration, min(run_stop, cursor + max_smooth_samples) + 1):
                    points = self.positions[arm][rows[cursor:end+1]]
                    vector = points[-1] - points[0]
                    path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                    distance = float(np.linalg.norm(vector))
                    if path <= 0 or distance / path < self.rules.smooth_direction_cos:
                        break
                    units = np.diff(points, axis=0)
                    units /= np.maximum(np.linalg.norm(units, axis=1, keepdims=True), 1e-12)
                    direction = vector / max(distance, 1e-12)
                    if np.min(units @ direction) < self.rules.smooth_direction_cos:
                        break
                    best = (end, distance / path)
                if best is None:
                    cursor += 1
                    continue
                end, path_ratio = best
                windows.append({'kind': 'smooth_point_to_point', 'start': int(rows[cursor]),
                                'end': int(rows[end]), 'target': self.rules.smooth_speed_target_m_s,
                                'target_tolerance': self.rules.smooth_speed_tolerance_m_s,
                                'path_ratio': path_ratio})
                cursor = end
        return windows

    # 由低曲率谷、闭合回摆和夹爪后的复位生成局部候选窗口。
    def candidate_windows(self, phase, include_smooth=True):
        arm, a, b = phase['arm'], phase['start'], phase['end']
        rows = np.arange(a, b+1)
        valleys = self.valleys(rows, arm)
        phase['valleys_before'] = valleys
        for valley in valleys:
            if valley['major_turn']:
                self.protected[valley['frame']] = True
        # 阶段二的主要 Z 极值是真实动作锚点；上下模板不能翻转坐标或删掉大运动。
        if phase['id'] == 2:
            z = median_filter(self.positions[arm][rows, 2], size=self.samples(.1, odd=True), mode='nearest')
            for sign in (-1, 1):
                peaks, _ = find_peaks(sign*z, prominence=.01, distance=self.samples(.2))
                self.protected[rows[peaks]] = True
        if phase['status'] != 'standard':
            return []
        windows = []
        if include_smooth:
            windows.extend(self.smooth_windows(phase))
        phase_speed = self.speed[arm][rows]
        moving = phase_speed[phase_speed > 0]
        reference = float(np.median(moving)) if len(moving) else 0.
        smooth = median_filter(self.speed[arm], size=self.samples(.1, odd=True), mode='nearest')
        for valley in valleys:
            if valley['angle_deg'] > self.rules.low_turn_deg or valley['major_turn'] or valley['speed'] >= reference*.7:
                continue
            k = valley['frame']
            left, right = k, k
            threshold = max(reference*.7, valley['speed']*1.25)
            while left > a and smooth[left] < threshold:
                left -= 1
            while right < b and smooth[right] < threshold:
                right += 1
            context = self.samples(2/30)
            left, right = max(a, left-context), min(b, right+context)
            if right-left >= 3:
                flanks = np.r_[self.speed[arm][max(a, left-self.samples(.2)):left+1],
                               self.speed[arm][right:min(b+1, right+self.samples(.2)+1)]]
                flanks = flanks[flanks > 0]
                target = float(np.median(flanks)) if len(flanks) else reference
                windows.append({'kind': 'low_curvature_valley', 'start': left, 'end': right, 'target': max(target, .01)})
        if phase['id'] == 4:
            events = [event for event in self.gripper_events if event['arm'] == arm and a <= event['end'] < b]
            start = max([event['end']+1 for event in events], default=a)
            phase['reset_start'] = start
            windows = [window for window in windows if window['end'] < start]
            if b-start >= 3:
                v = self.speed[arm][start:b+1]; v = v[v > 0]
                windows.append({'kind': 'dense_reset', 'start': start, 'end': b,
                                'target': float(np.median(v)) if len(v) else .01})
        elif phase['id'] in (2, 3, 5):
            positions = self.positions[arm][rows]
            path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
            length = len(rows)
            first = np.arange(length)[:, None]
            second = first+np.arange(self.samples(.27), self.samples(self.rules.loop_gap_s)+1)[None, :]
            valid = second < length
            second = np.minimum(second, length-1)
            distance = np.linalg.norm(positions[second]-positions[first], axis=2)
            arc = path[second]-path[first]
            protection = np.r_[0, np.cumsum(self.protected[rows])]
            valid &= protection[second]-protection[np.minimum(first+1, length)] == 0
            valid &= (arc >= self.rules.loop_min_path_m) & (distance <= self.rules.loop_return_m)
            valid &= distance <= arc*self.rules.loop_path_ratio
            candidates = np.argwhere(valid)
            if len(candidates):
                scores = arc[valid]-distance[valid]
                chosen = []
                for index in np.argsort(scores)[::-1]:
                    x, offset = candidates[index]; y = int(second[x, offset])
                    x, y = int(rows[x]), int(rows[y])
                    if any(not (y < u or x > v) for u, v in chosen):
                        continue
                    chosen.append((x, y))
                    windows.append({'kind': 'return_loop', 'start': x, 'end': y,
                                    'target': max(reference, .01)})
                    if len(chosen) >= 8:
                        break
        if phase['id'] == 5:
            departure = self.departure_window(phase)
            if departure is not None:
                windows = [departure]+[window for window in windows if window['start'] > departure['end']]
                return windows
        return sorted(windows, key=lambda window: (window['kind'] not in ('return_loop', 'smooth_point_to_point'), window['start']))

    # 二阶规划在真实源帧图上选择相邻位移更均匀的序列，事件内部只能逐帧通过。
    def solve_window(self, selected, phase, window, target, weight):
        rows = selected[(selected >= window['start']) & (selected <= window['end'])]
        n = len(rows)
        if n < 4:
            return None
        arm, r = phase['arm'], self.rules
        positions = self.positions[arm][rows]
        disp = positions[None, :, :]-positions[:, None, :]
        speed = np.linalg.norm(disp, axis=2)*self.fps
        i, j = np.indices((n, n))
        departure = window['kind'] == 'monotonic_departure'
        max_gap_seconds = (r.departure_gap_s if departure else r.loop_gap_s if window['kind'] == 'return_loop'
                           else r.smooth_max_gap_s if window['kind'] == 'smooth_point_to_point' else r.dense_gap_s)
        max_gap = self.samples(max_gap_seconds)
        allowed = (j > i) & ((rows[j]-rows[i] <= max_gap) | (j == i+1))
        if departure:
            progress = (positions-window['origin'])@np.asarray(window['direction'])
            allowed &= progress[None, :] >= progress[:, None]-r.departure_backtrack_tolerance_m
        phase_peak = self.metrics(np.arange(phase['start'], phase['end']+1), arm)['speed_max_m_s']
        smooth = window['kind'] == 'smooth_point_to_point'
        inherited = j == i+1
        allowed &= (inherited | (speed <= r.retime_max_step_m*self.fps)) if smooth else speed <= r.max_step_m*self.fps
        smooth_speed_limit = (float(window.get('target', phase_peak)) + r.smooth_speed_tolerance_m_s) if smooth else phase_peak
        allowed &= (inherited | (speed <= smooth_speed_limit)) if smooth else speed <= smooth_speed_limit*(1+r.dynamics_rtol)+r.dynamics_atol
        quat = self.quaternions[arm][rows]
        angles = 2*np.arccos(np.clip(abs(quat@quat.T), 0, 1))
        allowed &= (inherited | (angles <= np.radians(r.retime_max_rotation_deg))) if smooth else angles <= np.radians(r.max_orientation_step_deg)
        other = self.positions['left' if arm == 'right' else 'right'][rows]
        other_steps = np.linalg.norm(other[None, :, :]-other[:, None, :], axis=2)
        allowed &= (inherited | (other_steps <= min(r.max_other_step_m, r.retime_max_step_m))) if smooth else other_steps <= r.max_other_step_m
        if smooth:
            other_quat = self.quaternions['left' if arm == 'right' else 'right'][rows]
            allowed &= inherited | (2*np.arccos(np.clip(abs(other_quat@other_quat.T), 0, 1)) <= np.radians(r.retime_max_rotation_deg))
        reference = getattr(self, '_smooth_reference_ids', None)
        peaks = self.dynamic_peaks(self.boundary_ids(reference, phase) if reference is not None
                                  else self.source_boundary_ids(selected, phase))
        limits = {name: value*(1+r.dynamics_rtol)+r.dynamics_atol for name, value in peaks.items()}
        if smooth:
            # 点到点段的抽帧会提高速度；仅放宽主臂速度包络，几何步长、对侧臂和事件仍受硬约束。
            desired = float(window.get('target', r.smooth_speed_target_m_s))
            limits[arm+'_speed'][0] = max(limits[arm+'_speed'][0], desired * (1 + r.smooth_speed_tolerance_m_s / max(r.smooth_speed_target_m_s, 1e-6)))
            limits[arm+'_acceleration'][0] = max(limits[arm+'_acceleration'][0], desired * self.fps * 2.)
            limits[arm+'_jerk'][0] = max(limits[arm+'_jerk'][0], desired * self.fps**2 * 2.)
            joint_mask = self.joint_arms == arm
            limits['joint_speed'][joint_mask] = np.maximum(limits['joint_speed'][joint_mask], r.retime_max_joint_step_rad * self.fps)
            limits['joint_acceleration'][joint_mask] = np.maximum(limits['joint_acceleration'][joint_mask], r.retime_max_joint_step_rad * self.fps**2)
            limits['joint_jerk'][joint_mask] = np.maximum(limits['joint_jerk'][joint_mask], r.retime_max_joint_step_rad * self.fps**3)
        if departure:
            factor = window.get('dynamic_multiplier', 1.)
            for name in limits:
                if name.startswith('joint_'):
                    limits[name] = limits[name]*np.where(self.joint_arms == arm, factor, 1.)
                elif name.startswith(arm+'_'):
                    limits[name] = limits[name]*factor
        joints = self.joint_values[rows]
        joint_disp = joints[None, :, :]-joints[:, None, :]
        if smooth:
            allowed &= inherited | (np.max(abs(joint_disp), axis=2) <= r.retime_max_joint_step_rad)
        allowed &= np.all(abs(joint_disp)*self.fps <= limits['joint_speed'], axis=2)
        aux = self.other_values[rows]
        allowed &= np.all(abs(aux[None, :, :]-aux[:, None, :])*self.fps <= limits['other_state_action_speed'], axis=2)
        other_disp = other[None, :, :]-other[:, None, :]
        other_arm = 'left' if arm == 'right' else 'right'
        allowed &= np.linalg.norm(other_disp, axis=2)*self.fps <= limits[other_arm+'_speed'][0]
        for k in np.flatnonzero(self.protected[rows]):
            allowed &= ~((i < k) & (j > k))
        if window['kind'] not in ('return_loop', 'monotonic_departure'):
            for x, y in zip(*np.nonzero(allowed & (j > i+1))):
                for checked_arm in ('left', 'right') if smooth else (arm,):
                    path = self.positions[checked_arm]
                    offset = path[rows[x]:rows[y]+1]-path[rows[x]]
                    vector = path[rows[y]]-path[rows[x]]
                    fraction = np.clip(offset@vector/max(float(vector@vector), 1e-20), 0, 1)
                    if np.linalg.norm(offset-fraction[:, None]*vector, axis=1).max() > r.path_deviation_m:
                        allowed[x, y] = False
        target = max(float(target), .001)
        edge_cost = ((speed-target)/target)**2+r.edge_penalty
        if departure:
            # 硬性禁止回退之后优先密集保留正常运动，速度均匀性只作次级代价。
            edge_cost += (j-i-1)*1000.
        dp = np.full((n, n), np.inf)
        previous = np.full((n, n), -1, dtype=int)
        progress_peak = np.full((n, n), -np.inf) if departure else None
        start_index, end_index = np.searchsorted(selected, rows[[0, -1]])
        entering = positions[0]-self.positions[arm][selected[max(0, start_index-1)]]
        exiting = self.positions[arm][selected[min(len(selected)-1, end_index+1)]]-positions[-1]
        prior_id = selected[max(0, start_index-1)]
        joint_entering = joints[0]-self.joint_values[prior_id]
        other_entering = other[0]-self.positions[other_arm][prior_id]
        previous_id = selected[max(0, start_index-2)]
        earlier = self.positions[arm][prior_id]-self.positions[arm][previous_id]
        joint_earlier = self.joint_values[prior_id]-self.joint_values[previous_id]
        other_earlier = self.positions[other_arm][prior_id]-self.positions[other_arm][previous_id]
        for y in np.flatnonzero(allowed[0]):
            if (np.all(abs(joint_disp[0, y]-joint_entering)*self.fps**2 <= limits['joint_acceleration'])
                    and np.linalg.norm(other_disp[0, y]-other_entering)*self.fps**2 <= limits[other_arm+'_acceleration'][0]
                    and np.linalg.norm(disp[0, y]-entering)*self.fps**2 <= limits[arm+'_acceleration'][0]
                    and np.all(abs(joint_disp[0, y]-2*joint_entering+joint_earlier)*self.fps**3 <= limits['joint_jerk'])
                    and np.linalg.norm(disp[0, y]-2*entering+earlier)*self.fps**3 <= limits[arm+'_jerk'][0]
                    and np.linalg.norm(other_disp[0, y]-2*other_entering+other_earlier)*self.fps**3 <= limits[other_arm+'_jerk'][0]):
                dp[0, y] = edge_cost[0, y]+weight*np.sum(((disp[0, y]-entering)*self.fps/target)**2)
                if departure:
                    progress_peak[0, y] = max(progress[0], progress[y])
        for x in range(1, n-1):
            prev = np.flatnonzero(np.isfinite(dp[:, x]))
            if not len(prev):
                continue
            for y in np.flatnonzero(allowed[x]):
                costs = dp[prev, x]+weight*np.sum(((disp[x, y]-disp[prev, x])*self.fps/target)**2, axis=1)
                feasible = np.all(abs(joint_disp[x, y]-joint_disp[prev, x])*self.fps**2 <= limits['joint_acceleration'], axis=1)
                feasible &= np.linalg.norm(disp[x, y]-disp[prev, x], axis=1)*self.fps**2 <= limits[arm+'_acceleration'][0]
                feasible &= np.linalg.norm(other_disp[x, y]-other_disp[prev, x], axis=1)*self.fps**2 <= limits[other_arm+'_acceleration'][0]
                history = previous[prev, x]
                prior_disp = disp[np.maximum(history, 0), prev].copy()
                prior_joints = joint_disp[np.maximum(history, 0), prev].copy()
                prior_other = other_disp[np.maximum(history, 0), prev].copy()
                prior_disp[prev == 0], prior_joints[prev == 0], prior_other[prev == 0] = entering, joint_entering, other_entering
                feasible &= np.all(abs(joint_disp[x, y]-2*joint_disp[prev, x]+prior_joints)*self.fps**3 <= limits['joint_jerk'], axis=1)
                feasible &= np.linalg.norm(disp[x, y]-2*disp[prev, x]+prior_disp, axis=1)*self.fps**3 <= limits[arm+'_jerk'][0]
                feasible &= np.linalg.norm(other_disp[x, y]-2*other_disp[prev, x]+prior_other, axis=1)*self.fps**3 <= limits[other_arm+'_jerk'][0]
                if departure:
                    feasible &= progress[y] >= progress_peak[prev, x]-r.departure_backtrack_tolerance_m
                costs[~feasible] = np.inf
                best = int(np.argmin(costs))
                dp[x, y] = costs[best]+edge_cost[x, y]
                previous[x, y] = prev[best]
                if departure:
                    progress_peak[x, y] = max(progress_peak[prev[best], x], progress[y])
        costs = dp[:, -1]+weight*np.sum(((exiting-disp[:, -1])*self.fps/target)**2, axis=1)
        x, y = int(np.argmin(costs)), n-1
        if not np.isfinite(costs[x]):
            return None
        keep = [y, x]
        while x:
            x, y = int(previous[x, y]), x
            keep.append(x)
        return rows[np.asarray(keep[::-1])]

    # 在共享规则下择优；一次候选必须同时通过路径密度、事件和阶段动态检查。
    def optimize_phase(self, selected, phase, windows):
        phase['attempts'] = []
        for window in windows:
            if window['kind'] == 'monotonic_departure':
                continue
            a, b = window['start'], window['end']
            before_ids = selected[(selected >= a) & (selected <= b)]
            if len(before_ids) < 4:
                continue
            arm = phase['arm']
            before = self.metrics(before_ids, arm)
            choices, rejected = [], {}
            seen = set()
            before_valleys = len(self.valleys(selected[(selected >= phase['start']) & (selected <= phase['end'])], arm))
            for scale in self.rules.target_speed_scales:
                for weight in self.rules.turn_weights:
                    ids = self.solve_window(selected, phase, window, window['target']*scale, weight)
                    if ids is None or len(ids) == len(before_ids) or tuple(ids) in seen:
                        continue
                    seen.add(tuple(ids))
                    after = self.metrics(ids, arm)
                    if (window['kind'] == 'smooth_point_to_point' and
                            abs(after['speed_median_m_s']-window['target']) >=
                            abs(before['speed_median_m_s']-window['target'])-1e-12):
                        continue
                    if (window['kind'] == 'smooth_point_to_point' and
                            abs(after['speed_median_m_s']-window['target']) >
                            self.rules.smooth_speed_tolerance_m_s):
                        continue
                    candidate = np.r_[selected[selected < a], ids, selected[selected > b]]
                    failures = []
                    for stage in self._phases:
                        stage_failures = self.plan_violations(candidate, stage)
                        if window['kind'] == 'smooth_point_to_point' and stage['id'] == phase['id']:
                            # 直线段提速允许主臂动态峰值随目标速度上升；其余阶段、对侧臂和辅助量仍严格检查。
                            exempt = {f'{arm}_{name}' for name in ('speed', 'acceleration', 'jerk')}
                            reference = getattr(self, '_smooth_reference_ids', None)
                            baseline = self.dynamic_peaks(self.boundary_ids(reference, stage) if reference is not None
                                                          else self.source_boundary_ids(candidate, stage))
                            output = self.dynamic_peaks(self.boundary_ids(candidate, stage))
                            for name in ('joint_speed', 'joint_acceleration', 'joint_jerk'):
                                if name in stage_failures:
                                    # 仅当对侧臂关节峰值仍在原包络内时放宽主臂分量。
                                    mask = self.joint_arms != arm
                                    if np.all(output[name][mask] <= baseline[name][mask] * (1+self.rules.dynamics_rtol) + self.rules.dynamics_atol):
                                        exempt.add(name)
                            stage_failures = [reason for reason in stage_failures if reason not in exempt]
                        failures.extend(f"phase_{stage['id']}:{reason}" for reason in stage_failures)
                    if window['kind'] != 'return_loop':
                        if after['path_length_m'] < before['path_length_m']*self.rules.min_path_ratio:
                            failures.append('path_retention')
                        base = np.arange(phase['start'], phase['end']+1)
                        active = base[~self.static_mask[base]]
                        kept_active = candidate[np.isin(candidate, active)]
                        keep_ratio = self.rules.min_keep_ratio
                        if window['kind'] == 'smooth_point_to_point':
                            # 目标速度越高，平滑直线段允许按源速度/目标速度降低采样密度。
                            keep_ratio = min(keep_ratio, max(self.rules.retime_min_sampling_ratio,
                                              before['speed_median_m_s'] / max(window['target'], 1e-6)))
                        if len(kept_active) < np.ceil(len(active)*keep_ratio):
                            failures.append('active_frame_density')
                        if window['kind'] == 'dense_reset' and len(ids) < np.ceil(len(before_ids)*self.rules.min_keep_ratio):
                            failures.append('reset_frame_density')
                    elif after['path_length_m'] > before['path_length_m']*self.rules.loop_path_ratio:
                        failures.append('insufficient_loop_removal')
                    if failures:
                        for reason in failures:
                            rejected[reason] = rejected.get(reason, 0)+1
                        continue
                    local = candidate[(candidate >= phase['start']) & (candidate <= phase['end'])]
                    count = len(self.valleys(local, arm))
                    if window['kind'] == 'smooth_point_to_point':
                        # 平滑直线段优先按非零速度接近目标，其次才比较波动和保留帧数。
                        score = (abs(after['speed_median_m_s'] - window['target']),
                                 after['speed_cv'], -len(ids))
                    else:
                        score = (abs(count-2) if phase['id'] == 1 else 0,
                                 after['path_length_m'] if window['kind'] == 'return_loop' else after['speed_cv'],
                                 -len(ids))
                    if (window['kind'] not in ('return_loop', 'smooth_point_to_point')
                            and after['speed_cv'] >= before['speed_cv'] and count >= before_valleys):
                        continue
                    choices.append((score, candidate, ids, after, scale, weight))
            attempt = {**window, 'candidate_count': len(seen), 'rejections': rejected,
                       'status': 'unchanged under constraints', 'before': before}
            if choices:
                _, selected, ids, after, scale, weight = min(choices, key=lambda item: item[0])
                if window['kind'] == 'smooth_point_to_point':
                    phase['smooth_speed_applied'] = True
                attempt.update(status='compressed', kept_source_frames=ids.tolist(),
                               removed_source_frames=np.setdiff1d(before_ids, ids).tolist(),
                               after=after, target_speed_m_s=window['target']*scale, turn_weight=weight)
            phase['attempts'].append(attempt)
        return selected

    # 用户要求强制单向时只为第五阶段退离连接逐档放宽动态包络，记录原约束实际超限。
    def force_departure(self, selected, phase, window):
        a, b, arm = window['start'], window['end'], phase['arm']
        before_ids = selected[(selected >= a) & (selected <= b)]
        original_peaks = self.dynamic_peaks(self.source_boundary_ids(selected, phase))
        other_arm = 'left' if arm == 'right' else 'right'
        tried, chosen = [], None
        for factor in self.rules.departure_dynamic_scales:
            candidates, seen = [], set()
            for scale in self.rules.target_speed_scales:
                for weight in self.rules.turn_weights:
                    ids = self.solve_window(selected, phase, {**window, 'dynamic_multiplier': factor}, window['target']*scale, weight)
                    if ids is None or tuple(ids) in seen:
                        continue
                    seen.add(tuple(ids))
                    candidate = np.r_[selected[selected < a], ids, selected[selected > b]]
                    if self.departure_backtrack(candidate, phase) > self.rules.departure_backtrack_tolerance_m:
                        continue
                    if any(self.violations(candidate, stage) for stage in self._phases if stage['id'] != phase['id']):
                        continue
                    peaks = self.dynamic_peaks(self.boundary_ids(candidate, phase))
                    limits = {}
                    ratios = []
                    for name, source_peak in original_peaks.items():
                        multipliers = np.full_like(source_peak, factor)
                        if name.startswith(other_arm+'_') or name == 'other_state_action_speed':
                            multipliers[:] = 1.
                        elif name.startswith('joint_'):
                            multipliers[self.joint_arms != arm] = 1.
                        limits[name] = source_peak*multipliers*(1+self.rules.dynamics_rtol)+self.rules.dynamics_atol
                        ratios.append(float(np.max(peaks[name]/np.maximum(source_peak, self.rules.dynamics_atol), initial=0)))
                    if any(np.any(peaks[name] > limits[name]) for name in peaks):
                        continue
                    candidates.append(((max(ratios), -len(ids)), candidate, ids))
            tried.append({'multiplier': factor, 'candidate_count': len(seen), 'feasible_count': len(candidates)})
            if candidates:
                _, selected, ids = min(candidates, key=lambda item: item[0])
                chosen = {'multiplier': factor, 'kept_source_frames': ids.tolist(),
                          'removed_source_frames': np.setdiff1d(before_ids, ids).tolist()}
                break
        phase['monotonic_departure']['search'] = tried
        if chosen is not None:
            phase['monotonic_departure']['selection'] = chosen
            phase['monotonic_departure']['dynamic_exception_applied'] = bool(self.violations(selected, phase))
            output_peaks = self.dynamic_peaks(self.boundary_ids(selected, phase))
            phase['monotonic_departure']['peak_ratios'] = {
                name: float(np.max(value/np.maximum(original_peaks[name], self.rules.dynamics_atol), initial=0))
                for name, value in output_peaks.items()}
            phase['monotonic_departure']['priority'] = '完整事件和对侧臂保护 → 单向退离 → 最小动态增幅'
        return selected

    # 静止段先整段尝试，失败时优先逐帧删除真正重复的状态，而不是恢复整个阶段。
    def remove_static(self, first, phases, static):
        selected = np.arange(first, self.n)
        rejected = []
        motion = self.speed['left']+self.speed['right']
        motion += np.r_[0., np.linalg.norm(np.diff(self.state, axis=0), axis=1)*self.fps]
        for a, b in runs(static & (np.arange(self.n) >= first)):
            candidate = selected[(selected < a) | (selected >= b)]
            failures = sorted({reason for phase in phases for reason in self.violations(candidate, phase)})
            if not failures:
                selected = candidate
                continue
            for frame in sorted(range(a, b), key=lambda frame: motion[frame]):
                candidate = selected[selected != frame]
                if len(candidate) >= 4 and not any(self.violations(candidate, phase) for phase in phases):
                    selected = candidate
            retained = selected[(selected >= a) & (selected < b)]
            if len(retained):
                rejected.append({'source_interval_half_open': [a, b], 'retained_source_frames': retained.tolist(), 'reason': failures})
        return selected, rejected

    # 先做阶段与事件识别，再合并所有规则；参考数据及旧选帧列表不进入此函数。
    def plan(self, include_smooth=True):
        self.protected = self.event_protected.copy()
        self.phase_ids.fill(-1)
        phases = self.detect_phases()
        self._phases = phases
        if not phases:
            ids = np.arange(self.n)
            return EpisodePlan(ids, {'status': 'unoptimized', 'reason': '未可靠识别五阶段，保留原序列',
                                      'source_frames': self.n, 'output_frames': self.n, 'first_source_frame': 0,
                                      'selected_source_frames': ids.tolist(), 'phases': [], 'detection': self.detection,
                                      'gripper_events': self.gripper_events, 'rules': asdict(self.rules)}, self.phase_ids.copy())
        first = phases[0]['start']
        windows = {phase['id']: self.candidate_windows(phase, include_smooth) for phase in phases}
        # 主要转折作为运动锚点；没有事件的重复静止样本直接移除。
        static = self.static_mask & ~self.protected
        selected, rejected_static = self.remove_static(first, phases, static)
        # 静止删除完成后固定每阶段的三阶差分衔接样本；不重新补回静止保护邻帧。
        for phase in phases:
            local = selected[(selected >= phase['start']) & (selected <= phase['end'])]
            self.protected[np.r_[local[:3], local[-3:]]] = True
        base = selected.copy()
        for phase in phases:
            selected = self.optimize_phase(selected, phase, windows[phase['id']])
        # 合并后再检查邻接阶段；局部通过不代表跨阶段拼接也通过。
        for _ in range(len(phases)+1):
            restored = False
            for phase in phases:
                failures = self.plan_violations(selected, phase)
                if failures:
                    restore = self.source_boundary_ids(selected, phase)
                    selected = np.union1d(selected, restore[restore >= first])
                    phase['rollback_reason'] = failures
                    restored = True
            if not restored:
                break
        if any(self.plan_violations(selected, phase) for phase in phases):
            selected = np.arange(first, self.n)
            for phase in phases:
                phase['rollback_reason'] = ['跨阶段约束无法满足，保留起点之后的原序列']
        if any(self.plan_violations(selected, phase) for phase in phases):
            raise ValueError('起点裁剪仍违反拼接约束，需要保留完整原 episode')
        for phase in phases:
            for window in windows[phase['id']]:
                if window['kind'] == 'monotonic_departure':
                    selected = self.force_departure(selected, phase, window)
        if not set(np.flatnonzero(self.protected & (np.arange(self.n) >= first))).issubset(set(selected)):
            raise ValueError('选帧遗漏完整夹爪事件或任务/转折保护点')
        for phase in phases:
            ids = selected[(selected >= phase['start']) & (selected <= phase['end'])]
            raw = np.arange(phase['start'], phase['end']+1)
            phase.update(kept_source_frames=ids.tolist(), before=self.metrics(raw, phase['arm']),
                         after=self.metrics(ids, phase['arm']), valleys_after=self.valleys(ids, phase['arm']),
                         dynamic_violations=self.violations(selected, phase),
                         dynamic_peaks_before={k: v.tolist() for k, v in self.dynamic_peaks(self.source_boundary_ids(selected, phase)).items()},
                         dynamic_peaks_after={k: v.tolist() for k, v in self.dynamic_peaks(self.boundary_ids(selected, phase)).items()})
            departure = phase.get('monotonic_departure')
            if departure and departure['status'] != 'unsupported':
                backtrack = self.departure_backtrack(selected, phase)
                achieved = backtrack <= self.rules.departure_backtrack_tolerance_m
                departure.update(after_max_backtrack_m=backtrack, achieved=achieved,
                                 status='achieved' if achieved else 'infeasible_under_constraints')
                departure['original_dynamic_violations'] = phase['dynamic_violations']
                if not achieved:
                    departure['reason'] = '未找到同时满足单向退离、完整事件和动态峰值的来源帧路径，保留真实动作'
            if phase['id'] == 1:
                valleys = phase['valleys_after']
                phase['two_nonzero_valleys_achieved'] = len(valleys) == 2 and all(valley['speed'] > 0 for valley in valleys)
                if not phase['two_nonzero_valleys_achieved']:
                    phase['target_note'] = '真实动作、完整夹爪过程及动态约束优先，未强制达到两次非零低谷'
        deleted = np.setdiff1d(np.arange(self.n), selected)
        report = {'strategy': 'generic_h01_dishwasher_five_phase', 'status': 'optimized' if len(selected) < self.n else 'unoptimized',
                  'source_frames': self.n, 'output_frames': len(selected), 'removed_frames': len(deleted),
                  'first_source_frame': int(selected[0]), 'fps': self.fps, 'rules': asdict(self.rules),
                  'selected_source_frames': selected.tolist(), 'removed_source_frames': deleted.tolist(),
                  'initial_removed_frames': list(range(first)),
                  'static_intervals_half_open': [list(pair) for pair in runs(static & (np.arange(self.n) >= first))],
                  'static_removal_rejections': rejected_static,
                  'static_removed_frames': np.setdiff1d(np.flatnonzero(static & (np.arange(self.n) >= first)), selected).tolist(),
                  'sampling_removed_frames': np.setdiff1d(base, selected).tolist(),
                  'gripper_events': self.gripper_events, 'detection': self.detection, 'phases': phases,
                  'validation': {'full_gripper_events_preserved': True,
                                 'phase_dynamic_peaks_passed': not any(phase['dynamic_violations'] for phase in phases),
                                 'departure_dynamic_exception_applied': any(phase.get('monotonic_departure', {}).get('dynamic_exception_applied', False) for phase in phases)},
                  'limitations': ['五阶段由主臂交替和任务先验推断，不等同于视觉接触识别。',
                                  '只抽取来源帧，完整夹爪或动态约束可能保留真实减速。',
                                  '二阶代价与历史 jerk 剪枝属于多候选搜索，未找到改进不等于证明全局不可行。']}
        return EpisodePlan(selected, report, self.phase_ids.copy())

    # 在已批准的选帧结果上追加平滑提速，只允许继续删帧，绝不回补或重排现有来源行。
    def smooth_plan(self, plan):
        if len(plan.selected) < 4 or not plan.report.get('phases'):
            return plan
        report = deepcopy(plan.report)
        selected = plan.selected.copy()
        phases = report['phases']
        original_protected = self.protected.copy()
        self._smooth_reference_ids = plan.selected.copy()
        self._phases = phases
        self.protected = self.event_protected.copy()
        for phase in phases:
            local = selected[(selected >= phase['start']) & (selected <= phase['end'])]
            self.protected[np.r_[local[:3], local[-3:]]] = True
            for valley in phase.get('valleys_before', []):
                if valley['major_turn']:
                    self.protected[valley['frame']] = True
        try:
            for phase in phases:
                original_attempts = phase.get('attempts', [])
                windows = self.smooth_windows(phase)
                selected = self.optimize_phase(selected, phase, windows)
                phase['attempts'] = original_attempts + phase['attempts']
            if not np.isin(selected, plan.selected).all():
                raise ValueError('平滑提速必须是已有选帧结果的子序列')
            for phase in phases:
                ids = selected[(selected >= phase['start']) & (selected <= phase['end'])]
                phase.update(kept_source_frames=ids.tolist(), after=self.metrics(ids, phase['arm']),
                             valleys_after=self.valleys(ids, phase['arm']), dynamic_violations=self.violations(selected, phase),
                             dynamic_peaks_after={key: value.tolist() for key, value in self.dynamic_peaks(self.boundary_ids(selected, phase)).items()})
                for attempt in phase.get('attempts', []):
                    if attempt.get('kind') == 'smooth_point_to_point' and attempt.get('status') == 'compressed':
                        local = selected[(selected >= attempt['start']) & (selected <= attempt['end'])]
                        attempt['after'] = self.metrics(local, phase['arm'])
                        attempt['kept_source_frames'] = local.tolist()
                if phase['id'] == 1:
                    phase['two_nonzero_valleys_achieved'] = len(phase['valleys_after']) == 2 and all(v['speed'] > 0 for v in phase['valleys_after'])
            removed = np.setdiff1d(np.arange(self.n), selected)
            additional = np.setdiff1d(plan.selected, selected)
            report.update(selected_source_frames=selected.tolist(), removed_source_frames=removed.tolist(),
                          output_frames=len(selected), removed_frames=len(removed), first_source_frame=int(selected[0]))
            report['sampling_removed_frames'] = np.union1d(report.get('sampling_removed_frames', []), additional).astype(int).tolist()
            report['smooth_speed_retiming'] = {'target_speed_m_s': self.rules.smooth_speed_target_m_s,
                'tolerance_m_s': self.rules.smooth_speed_tolerance_m_s, 'before_frames': len(plan.selected),
                'output_frames': len(selected), 'additional_removed_source_frames': additional.tolist(),
                'source_subset_preserved': True}
            report.setdefault('validation', {})['smooth_pass_is_subset'] = True
            report['validation']['phase_dynamic_peaks_passed'] = not any(phase['dynamic_violations'] for phase in phases)
            return EpisodePlan(selected, report, plan.phase_ids.copy())
        finally:
            self.protected = original_protected
            del self._smooth_reference_ids


# 用同一软时长目标缩短长 episode，几何步长和完整事件优先，不按 episode 编号分支。
class DurationNormalizer:
    # 原始测量是唯一数据源，所有优化变量只控制选取哪些来源帧。
    def __init__(self, analyzer):
        self.a, self.rules = analyzer, analyzer.rules

    # 审计合并后的端点位移、关节差及四元数角距；符号相反的等价四元数不算跳变。
    def unsafe_bridges(self, ids):
        a, r = self.a, self.rules
        start, end = ids[:-1], ids[1:]
        bad = np.max(abs(a.joint_values[end]-a.joint_values[start]), axis=1) > r.retime_max_joint_step_rad
        for arm in ('left', 'right'):
            pos, quat = a.positions[arm], a.quaternions[arm]
            angles = 2*np.arccos(np.clip(abs(np.sum(quat[end]*quat[start], axis=1)), 0, 1))
            bad |= np.linalg.norm(pos[end]-pos[start], axis=1) > r.retime_max_step_m
            bad |= angles > np.radians(r.retime_max_rotation_deg)
        return np.flatnonzero(bad & (end-start > 1))

    # 对完整事件附近渐变采样密度，避免从开合的逐帧保留突然切到大跨度抽帧。
    def weights(self, ids, report, budget):
        a, r = self.a, self.rules
        protected = a.event_protected[ids].copy()
        protected[:3] = protected[-3:] = True
        for phase in report['phases']:
            local = np.flatnonzero((ids >= phase['start']) & (ids <= phase['end']))
            protected[np.r_[local[:3], local[-3:]]] = True
            # 目标时长归一化不得再次跨越已完成的平滑点到点段，避免速度被推高超过目标带宽。
            for attempt in phase.get('attempts', []):
                if attempt.get('kind') == 'smooth_point_to_point' and attempt.get('status') == 'compressed':
                    protected[(ids >= attempt['start']) & (ids <= attempt['end'])] = True
            for valley in phase.get('valleys_before', []):
                if valley['major_turn']:
                    protected[ids == valley['frame']] = True
        ramp = a.samples(r.retime_event_ramp_s)
        distance = np.minimum(distance_transform_edt(~protected), ramp)/ramp
        lower, upper = r.retime_min_sampling_ratio, 1.
        for _ in range(24):
            fraction = (lower+upper)/2
            mass = 1-(1-fraction)*distance
            if mass.sum() > budget:
                upper = fraction
            else:
                lower = fraction
        return protected, 1-(1-lower)*distance

    # 二阶路径代价联合控制目标时长与相邻运动变化，保留必要的正常弯曲路径。
    def solve(self, ids, protected, mass):
        a, r, n = self.a, self.rules, len(ids)
        maximum = a.samples(r.retime_max_gap_s)
        cumulative = np.r_[0., np.cumsum(mass[1:])]
        protection = np.r_[0, np.cumsum(protected)]
        positions = np.c_[a.positions['left'][ids], a.positions['right'][ids]]
        joints = a.joint_values[ids]
        features = np.c_[positions/r.retime_max_step_m, joints[:, :14]/r.retime_max_joint_step_rad]
        delta = np.zeros((n, maximum, features.shape[1]))
        allowed = np.zeros((n, maximum), dtype=bool)
        edge_cost = np.zeros((n, maximum))
        for gap in range(1, min(maximum, n-1)+1):
            end = np.arange(gap, n); start = end-gap
            raw_gap = ids[end]-ids[start]
            valid = (protection[end]-protection[start+1] == 0) & ((raw_gap <= maximum) | (gap == 1))
            if gap > 1:
                valid &= np.max(abs(joints[end]-joints[start]), axis=1) <= r.retime_max_joint_step_rad
                for arm in ('left', 'right'):
                    pos, quat = a.positions[arm], a.quaternions[arm]
                    vector = pos[ids[end]]-pos[ids[start]]
                    valid &= np.linalg.norm(vector, axis=1) <= r.retime_max_step_m
                    angles = 2*np.arccos(np.clip(abs(np.sum(quat[ids[end]]*quat[ids[start]], axis=1)), 0, 1))
                    valid &= angles <= np.radians(r.retime_max_rotation_deg)
                    # 检查全部跳过的原始点，而非仅检查上一轮抽帧后还存在的点。
                    for offset in range(1, maximum):
                        skipped = np.minimum(ids[start]+offset, ids[end])
                        displacement = pos[skipped]-pos[ids[start]]
                        fraction = np.clip(np.sum(displacement*vector, axis=1)/np.maximum(np.sum(vector*vector, axis=1), 1e-20), 0, 1)
                        valid &= (raw_gap <= offset) | (np.linalg.norm(displacement-fraction[:, None]*vector, axis=1) <= r.path_deviation_m)
            allowed[end, gap-1] = valid
            delta[end, gap-1] = features[end]-features[start]
            edge_cost[end, gap-1] = (cumulative[end]-cumulative[start]-1)**2+r.retime_edge_penalty
        cost = np.full((n, maximum), np.inf)
        parent = np.full((n, maximum), -1, dtype=int)
        minimum_count = np.full(n, np.inf); minimum_count[0] = 1
        for end in range(1, n):
            for gap in np.flatnonzero(allowed[end]):
                start = end-gap-1
                minimum_count[end] = min(minimum_count[end], minimum_count[start]+1)
                if start == 0:
                    cost[end, gap] = edge_cost[end, gap]
                    continue
                candidates = cost[start]+r.retime_smoothness*np.sum((delta[end, gap]-delta[start])**2, axis=1)
                previous = int(np.argmin(candidates))
                cost[end, gap] = candidates[previous]+edge_cost[end, gap]
                parent[end, gap] = previous
        end, gap = n-1, int(np.argmin(cost[-1]))
        self.minimum_frames = int(minimum_count[-1]) if np.isfinite(minimum_count[-1]) else n
        if not np.isfinite(cost[end, gap]):
            return ids
        path = [end]
        while end:
            previous = parent[end, gap]
            end -= gap+1
            path.append(end)
            gap = previous
        return ids[path[::-1]]

    # 回补有明显位姿跨度的旧连接，再缩短时长；报告原动态变化，不将加速说成峰值不变。
    def apply(self, plan, target_s):
        a, r = self.a, self.rules
        if not np.isfinite(target_s) or target_s <= 0:
            raise ValueError('目标时长必须为有限正数')
        if len(plan.selected) < 4 or not plan.report.get('phases'):
            return plan
        report = deepcopy(plan.report)
        ids = plan.selected.copy()
        unsafe = self.unsafe_bridges(ids)
        repairs = [[int(ids[index]), int(ids[index+1])] for index in unsafe]
        if repairs:
            ids = np.union1d(ids, np.concatenate([np.arange(start+1, end) for start, end in repairs]))
        budget = min(len(ids), round(target_s*a.fps))
        self.minimum_frames = None
        if len(ids) > budget:
            protected, mass = self.weights(ids, report, budget)
            ids = self.solve(ids, protected, mass)
        if len(self.unsafe_bridges(ids)):
            raise ValueError('时长优化新增了超过几何步长限制的连接')
        required = np.flatnonzero(a.event_protected & (np.arange(a.n) >= ids[0]))
        if not np.isin(required, ids).all():
            raise ValueError('时长优化遗漏夹爪或任务事件')
        for phase in report['phases']:
            local = ids[(ids >= phase['start']) & (ids <= phase['end'])]
            phase.update(kept_source_frames=local.tolist(), after=a.metrics(local, phase['arm']),
                         valleys_after=a.valleys(local, phase['arm']), dynamic_violations=a.violations(ids, phase),
                         dynamic_peaks_before={key: value.tolist() for key, value in a.dynamic_peaks(a.source_boundary_ids(ids, phase)).items()},
                         dynamic_peaks_after={key: value.tolist() for key, value in a.dynamic_peaks(a.boundary_ids(ids, phase)).items()})
            if phase['id'] == 1:
                phase['two_nonzero_valleys_achieved'] = len(phase['valleys_after']) == 2 and all(v['speed'] > 0 for v in phase['valleys_after'])
            departure = phase.get('monotonic_departure')
            if departure and 'direction' in departure:
                backtrack = a.departure_backtrack(ids, phase)
                departure.update(after_max_backtrack_m=backtrack, achieved=backtrack <= r.departure_backtrack_tolerance_m,
                                 status='achieved' if backtrack <= r.departure_backtrack_tolerance_m else 'motion_audit_priority',
                                 original_dynamic_violations=phase['dynamic_violations'])
                if not departure['achieved']:
                    departure['reason'] = '位姿步长和完整运动优先，时长优化会恢复不满足步长约束的强制退离连接'
        removed = np.setdiff1d(np.arange(a.n), ids)
        report.update(selected_source_frames=ids.tolist(), removed_source_frames=removed.tolist(),
                      output_frames=len(ids), removed_frames=len(removed), first_source_frame=int(ids[0]))
        report['static_removed_frames'] = np.setdiff1d(report.get('static_removed_frames', []), ids).astype(int).tolist()
        report['sampling_removed_frames'] = np.setdiff1d(removed, np.r_[report.get('initial_removed_frames', []), report['static_removed_frames']]).astype(int).tolist()
        report['duration_normalization'] = {'target_duration_s': target_s, 'before_frames': len(plan.selected),
            'output_frames': len(ids), 'output_duration_s': len(ids)/a.fps, 'restored_bridges': repairs,
            'candidate_graph_minimum_duration_s': self.minimum_frames/a.fps if self.minimum_frames is not None else None,
            'additional_removed_source_frames': np.setdiff1d(plan.selected, ids).tolist(),
            'restored_source_frames': np.setdiff1d(ids, plan.selected).tolist(), 'geometric_step_checks_passed': True,
            'max_step_m': r.retime_max_step_m, 'max_joint_step_rad': r.retime_max_joint_step_rad,
            'max_rotation_deg': r.retime_max_rotation_deg, 'scope': '合并来源帧的连接；原数据相邻帧自身的高速动作单独审计'}
        report['validation']['phase_dynamic_peaks_passed'] = not any(phase['dynamic_violations'] for phase in report['phases'])
        report['validation']['retiming_geometric_steps_passed'] = True
        return EpisodePlan(ids, report, plan.phase_ids.copy())


# 读取按行 JSON；缺少必需元数据时由调用者终止该数据集。
def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


# 命令失败保留 ffmpeg 原因，禁止带着未验证的视频发布。
def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(' '.join(map(str, args[:4])) + ' failed: ' + result.stderr[-3000:])
    return result.stdout


# 连续源帧合并成 select 区间，减少长序列的命令行长度。
def select_filter(selected):
    groups = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
    return '+'.join(f'between(n\\,{int(group[0])}\\,{int(group[-1])})' for group in groups)


# 按同一源帧映射无损编码，并核验每一帧解码像素及 FPS。
def rewrite_video(source, output, selected, fps):
    source_hash = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    select = select_filter(selected)
    base = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(source)]
    command(base + ['-filter_threads', '1', '-vf', f'select={select},setpts=N/({fps}*TB)',
                    '-r', str(fps), '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
                    '-crf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)])
    original = command(base + ['-filter_threads', '1', '-vf', f'select={select}',
                               '-vsync', '0', '-pix_fmt', 'yuv420p', '-f', 'framemd5', '-'])
    rebuilt = command(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(output),
                       '-pix_fmt', 'yuv420p', '-f', 'framemd5', '-'])
    hashes = lambda text: [line.split(',')[-1].strip() for line in text.splitlines()
                           if line and not line.startswith('#')]
    source_frames, output_frames = hashes(original), hashes(rebuilt)
    if len(output_frames) != len(selected) or source_frames != output_frames:
        raise ValueError(f'视频选帧或解码像素验证失败：{source}')
    probe = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=width,height,nb_frames,avg_frame_rate,duration,codec_name,pix_fmt',
                                '-of', 'json', str(output)]))['streams'][0]
    numerator, denominator = map(float, probe['avg_frame_rate'].split('/'))
    if not np.isclose(numerator / denominator, fps):
        raise ValueError(f'视频 FPS 与元数据不一致：{source}')
    if sha256(source) != source_hash:
        raise ValueError(f'重构期间源视频发生改变：{source}')
    return {'source': str(source), 'source_sha256': source_hash, 'verified_frames': len(output_frames),
            'decoded_pixels_equal': True, 'source_sha256_unchanged': True, 'probe': probe}


# 每个 episode 的分析结果独立保存，不让报告或参考映射反向影响选帧。
@dataclass
class EpisodeRecord:
    source: Path
    table: pa.Table
    source_episode: int
    analyzer: object
    plan: object
    source_hash: str
    cache_hit: bool = False


# 通过相邻临时目录替换派生结果，替换失败时恢复上一版。
def publish(folder, output):
    backup = folder.parent / 'previous-output'
    if output.exists():
        output.rename(backup)
    try:
        folder.rename(output)
    except BaseException:
        if backup.exists():
            backup.rename(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


# 首次正式重建之前备份已有参考数据，后续重跑不覆盖该参考。
def snapshot_reference(output):
    reference = output.parent / '.references' / output.name
    if reference.exists():
        return reference
    if not output.exists():
        return None
    reference.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-', dir=reference.parent) as temporary:
        folder = Path(temporary) / 'reference'
        folder.mkdir()
        files = sorted(set(output.glob('data/chunk-*/episode_*.parquet')) |
                       set(output.glob('*report*.json')) | set(output.glob('meta/**/*')) |
                       ({output / 'source_frame_map.csv'} if (output / 'source_frame_map.csv').is_file() else set()))
        manifest = {}
        for source in files:
            if not source.is_file():
                continue
            relative = source.relative_to(output)
            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest[str(relative)] = sha256(destination)
        write_json(folder / 'reference_manifest.json', {'source_output': str(output), 'files': manifest,
                   'usage': '仅作重构后效果对照，选帧算法不读取这些文件'})
        folder.rename(reference)
    return reference

# 中断时取消尚未开始的视频任务，等待在途任务安全结束后再清理临时输出。
@contextmanager
def video_executor(workers):
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# 缓存键同时包含实现版本和输入摘要；只复用同一代码、数据与规则的结果。
def cache_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


# JSON 校验和用于识别截断或损坏文件；缓存不使用可执行的 pickle 格式。
def read_cache(path, key):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        payload = value['payload']
        if value['key'] == key and value['sha256'] == cache_digest(payload):
            return payload
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


# 临时文件完整关闭后原子替换；缓存失败不影响已经验证的数据输出。
def write_cache(path, key, payload):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.partial-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({'key': key, 'sha256': cache_digest(payload), 'payload': payload},
                      stream, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)
    except OSError as error:
        print(f'缓存写入失败，重构仍继续：{path}: {error}', file=sys.stderr, flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# 只缓存已逐帧像素验证的视频；复用前核验源视频和缓存视频的内容哈希。
def rewrite_video_cached(source, output, selected, fps, cache_dir):
    if cache_dir is None or not source.is_file():
        return rewrite_video(source, output, selected, fps)
    source_hash = sha256(source)
    key = cache_digest({'code': IMPLEMENTATION_SHA256, 'source': str(source),
                        'sha256': source_hash, 'selected': np.asarray(selected).tolist(), 'fps': fps})
    folder = cache_dir / 'video'
    video, manifest = folder / f'{key}.mp4', folder / f'{key}.json'
    cached = read_cache(manifest, key)
    if cached and video.is_file() and sha256(video) == cached.get('video_sha256'):
        report = cached['report']
        if (report.get('decoded_pixels_equal') and report.get('verified_frames') == len(selected)
                and report.get('source_sha256') == source_hash):
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, output)
            if sha256(output) == cached['video_sha256'] and sha256(source) == source_hash:
                return {**report, 'cache_hit': True}
            output.unlink(missing_ok=True)
    report = rewrite_video(source, output, selected, fps)
    if report['source_sha256'] != source_hash:
        raise ValueError(f'视频缓存期间源文件改变：{source}')
    temporary = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.partial-', delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copy2(output, temporary)
        checksum = sha256(temporary)
        temporary.replace(video)
        write_cache(manifest, key, {'video_sha256': checksum, 'report': report})
    except OSError as error:
        print(f'视频缓存写入失败，重构仍继续：{error}', file=sys.stderr, flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {**report, 'cache_hit': False}


# 子进程只接收路径，固定元数据和规则在启动时传入，避免逐任务传输整个构建器。
_ANALYSIS_CONTEXT = None  # 每个分析进程独立持有配置。


# 限制 Arrow 内部线程，避免每个进程再创建几十个计算线程。
def init_analysis_worker(info, rules, target_duration, cache_dir):
    global _ANALYSIS_CONTEXT
    _ANALYSIS_CONTEXT = (info, rules, target_duration, cache_dir)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


# 返回进程号用于并行执行证据，来源行和选帧规则不变。
def run_analysis_worker(source):
    try:
        return analyze_source_episode(source, *_ANALYSIS_CONTEXT), os.getpid()
    except Exception as error:
        error.add_note(f'分析源文件：{source}')
        raise


# 独立分析单条 episode；串行与多进程共用完全相同的算法。
def analyze_source_episode(source, info, rules, target_duration, cache_dir=None):
    source_hash = sha256(source)
    table = pq.read_table(source)
    if table.num_rows < 1:
        raise ValueError(f'空 episode：{source}')
    source_episodes = set(table['episode_index'].to_pylist())
    if len(source_episodes) != 1:
        raise ValueError(f'一个 parquet 只能包含一个 episode：{source}')
    source_episode = int(next(iter(source_episodes)))
    if table['frame_index'].to_pylist() != list(range(table.num_rows)):
        raise ValueError(f'源 frame_index 必须从 0 连续编号：{source}')
    key = cache_digest({'code': IMPLEMENTATION_SHA256, 'source': str(source), 'sha256': source_hash,
                        'info': info, 'rules': asdict(rules), 'target_duration': target_duration})
    cache_path = cache_dir / 'analysis' / f'{key}.json' if cache_dir is not None else None
    cached = read_cache(cache_path, key) if cache_path is not None else None
    analyzer = EpisodeAnalyzer(table, info, rules)
    if cached is not None:
        try:
            selected = np.asarray(cached['selected'], dtype=np.int64)
            phase_ids = np.asarray(cached['phase_ids'], dtype=np.int64)
            report = cached['report']
            if 'rules' in report:
                report['rules'] = asdict(rules)
            if (len(selected) and selected[0] >= 0 and selected[-1] < table.num_rows
                    and np.all(np.diff(selected) > 0) and len(phase_ids) == table.num_rows
                    and report['source_parquet_sha256'] == source_hash
                    and report.get('selected_source_frames', selected.tolist()) == selected.tolist()
                    and report['output_frames'] == len(selected)):
                return EpisodeRecord(source, table, source_episode, analyzer,
                                     EpisodePlan(selected, report, phase_ids), source_hash, cache_hit=True)
        except (KeyError, ValueError, TypeError):
            pass
    try:
        plan = analyzer.plan(include_smooth=False)
    except ValueError as error:
        selected = np.arange(table.num_rows, dtype=np.int64)
        plan = EpisodePlan(selected=selected, phase_ids=np.zeros(table.num_rows, dtype=int),
                           report={'source_frames': table.num_rows, 'output_frames': table.num_rows,
                                   'first_source_frame': 0, 'phases': [], 'status': 'unchanged_fallback',
                                   'reason': str(error)})
    if target_duration is not None:
        plan = DurationNormalizer(analyzer).apply(plan, target_duration)
    plan = analyzer.smooth_plan(plan)
    selected = np.asarray(plan.selected, dtype=np.int64)
    if (not len(selected) or selected[0] < 0 or selected[-1] >= table.num_rows
            or np.any(np.diff(selected) <= 0) or len(plan.phase_ids) != table.num_rows):
        raise ValueError(f'选帧算法返回了无效来源映射：{source}')
    plan.report.update(source=str(source), source_episode=source_episode, source_parquet_sha256=source_hash)
    if sha256(source) != source_hash:
        raise ValueError(f'分析期间源文件改变：{source}')
    if cache_path is not None:
        write_cache(cache_path, key, {'selected': selected.tolist(), 'phase_ids': plan.phase_ids.tolist(),
                                     'report': plan.report})
    return EpisodeRecord(source, table, source_episode, analyzer, plan, source_hash)


# 数据集构建器
class DatasetBuilder:
    # 只接受源数据集外的真实输出目录，禁止覆盖源数据及其祖先。
    def __init__(self, root, output, rules, source_paths=None, episode=0, all_episodes=False, video_keys=None, target_duration=None, workers=16, use_cache=True):
        self.root = Path(root).resolve()
        self.output = Path(output).absolute()
        resolved = self.output.resolve()
        if (self.output.is_symlink() or resolved == self.root or resolved.is_relative_to(self.root)
                or self.root.is_relative_to(resolved)):
            raise ValueError('输出必须是源数据集之外的独立目录，不能通过符号链接覆盖源数据')
        self.cache_dir = self.output.parent / '.reconstruct-cache' / self.root.name if use_cache else None
        self.rules = rules
        self.workers = workers
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('分析和视频并发数必须为正整数')
        self.target_duration = target_duration
        if target_duration is not None and (not np.isfinite(target_duration) or target_duration <= 0):
            raise ValueError('目标时长必须为有限正数')
        self.info = json.loads((self.root / 'meta/info.json').read_text(encoding='utf-8'))
        if self.info.get('codebase_version') != 'v2.1':
            raise ValueError('当前重建器只支持 LeRobot v2.1')
        self.fps = float(self.info['fps'])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('FPS 必须为有限正数')
        self.chunk_size = int(self.info.get('chunks_size', 1000))
        if self.chunk_size < 1:
            raise ValueError('chunks_size 必须为正整数')
        available_videos = sorted(key for key, feature in self.info['features'].items() if feature['dtype'] == 'video')
        self.video_keys = available_videos if video_keys is None else list(dict.fromkeys(video_keys))
        unknown = sorted(set(self.video_keys)-set(available_videos))
        if unknown or (video_keys is not None and not self.video_keys):
            raise ValueError(f'无效视频视角：{unknown}；可选视角：{", ".join(available_videos)}')
        if source_paths is None:
            episodes = json_lines(self.root / 'meta/episodes.jsonl')
            numbers = sorted(int(row['episode_index']) for row in episodes if all_episodes or int(row['episode_index']) == episode)
            if not numbers:
                raise ValueError(f'源数据集没有 episode {episode}')
            source_paths = [self.root / self.info['data_path'].format(episode_chunk=n // self.chunk_size, episode_index=n)
                            for n in numbers]
        self.sources = [Path(path).resolve() for path in source_paths]
        if any(not path.is_relative_to(self.root) for path in self.sources):
            raise ValueError('源 parquet 必须位于指定数据集内')

    # 单进程调试入口与子进程共用分析函数。
    def analyze_episode(self, source):
        return analyze_source_episode(source, self.info, self.rules, self.target_duration, self.cache_dir)

    # 进程隔离绕过 GIL；每十秒提示存活状态，Ctrl+C 会结束所属分析子进程。
    def analyze(self):
        records = []
        count = min(self.workers, len(self.sources))
        self.analysis_worker_pids = set()
        if not count:
            return records
        started = time.monotonic()
        print(f'开始分析 {len(self.sources)} 个 episode，工作进程 {count}', file=sys.stderr, flush=True)
        if count == 1:
            for source in self.sources:
                try:
                    record = self.analyze_episode(source)
                except Exception as error:
                    error.add_note(f'分析源文件：{source}')
                    raise
                records.append(record)
                self.analysis_worker_pids.add(os.getpid())
                print(f'分析 {len(records)}/{len(self.sources)}: episode {record.source_episode}, '
                      f'{record.table.num_rows} -> {len(record.plan.selected)} 帧'
                      f'{"（缓存）" if record.cache_hit else ""}', file=sys.stderr, flush=True)
        else:
            # spawn 前设置环境，保证 NumPy 导入时已限制 BLAS/OpenMP，启动后恢复父进程环境。
            names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
            previous = {name: os.environ.get(name) for name in names}
            previous_pids = {process.pid for process in mp.active_children()}
            try:
                os.environ.update(dict.fromkeys(names, '1'))
                pool = mp.get_context('spawn').Pool(count, initializer=init_analysis_worker,
                                                  initargs=(self.info, self.rules, self.target_duration, self.cache_dir))
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            with pool:
                analysis_processes = [process for process in mp.active_children()
                                      if process.pid not in previous_pids]
                pending = pool.imap_unordered(run_analysis_worker, self.sources, chunksize=1)
                while len(records) < len(self.sources):
                    # Pool 会自动补建崩溃进程，但丢失的任务不会重发；检测后立即失败避免无限等待。
                    dead = [(process.pid, process.exitcode) for process in analysis_processes
                            if process.exitcode is not None]
                    if dead:
                        raise RuntimeError(f'分析子进程异常退出：{dead}；已完成缓存保留，修复后重跑')
                    try:
                        record, pid = pending.next(timeout=10)
                    except mp.TimeoutError:
                        print(f'分析进行中 {len(records)}/{len(self.sources)}，'
                              f'耗时 {time.monotonic()-started:.0f}s；等待正在计算的 episode',
                              file=sys.stderr, flush=True)
                        continue
                    records.append(record)
                    self.analysis_worker_pids.add(pid)
                    print(f'分析 {len(records)}/{len(self.sources)}: episode {record.source_episode}, '
                          f'{record.table.num_rows} -> {len(record.plan.selected)} 帧，'
                          f'耗时 {time.monotonic()-started:.1f}s'
                          f'{"（缓存）" if record.cache_hit else ""}', file=sys.stderr, flush=True)
        records.sort(key=lambda record: record.source_episode)
        if len({record.source_episode for record in records}) != len(records):
            raise ValueError('同一数据集重复提交了 episode')
        return records

    # 保存全部数值特征统计，非数值标签仍逐行保留但不生成无意义数值统计。
    @staticmethod
    def numeric_stats(table):
        stats = {}
        for name in table.column_names:
            try:
                array = np.asarray(table[name].to_pylist(), dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(array).all():
                raise ValueError(f'数值列包含 NaN/Inf：{name}')
            if array.ndim == 1:
                array = array[:, None]
            stats[name] = {key: operation(array, axis=0).tolist()
                           for key, operation in [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
            stats[name]['count'] = [table.num_rows]
        return stats

    # 按重编号后的 episode 更新任务引用、数值统计和相机标定。
    def metadata(self, folder, records, tables):
        meta = folder / 'meta'
        meta.mkdir()
        info = json.loads(json.dumps(self.info))
        info['features'] = {key: feature for key, feature in info['features'].items()
                            if feature['dtype'] != 'video' or key in self.video_keys}
        task_numbers = {int(value) for table in tables for value in table['task_index'].to_pylist()}
        task_rows = [row for row in json_lines(self.root / 'meta/tasks.jsonl') if row['task_index'] in task_numbers]
        if task_numbers != {row['task_index'] for row in task_rows}:
            raise ValueError('缺少 parquet 引用的任务元数据')
        task_names = {row['task_index']: row['task'] for row in task_rows}
        info.update(total_episodes=len(records), total_frames=sum(table.num_rows for table in tables),
                    total_chunks=(len(records) + self.chunk_size - 1) // self.chunk_size,
                    total_videos=len(records) * len(self.video_keys), total_tasks=len(task_rows),
                    chunks_size=self.chunk_size, splits={'train': f'0:{len(records)}'},
                    data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                    video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
        # 特征描述以已验证的输出视频为准，避免继承源数据的旧编码及尺寸信息。
        for key in self.video_keys:
            probes = [record.plan.report['videos'][key]['probe'] for record in records]
            if len({(probe['height'], probe['width']) for probe in probes}) != 1:
                raise ValueError(f'同一相机跨 episode 的视频尺寸不一致：{key}')
            probe = probes[0]
            numerator, denominator = map(float, probe['avg_frame_rate'].split('/'))
            feature = info['features'][key]
            feature.update(shape=[3, probe['height'], probe['width']], names=['channels', 'height', 'width'])
            feature.setdefault('info', {}).update({'video.height': probe['height'], 'video.width': probe['width'],
                'video.codec': probe['codec_name'], 'video.pix_fmt': probe['pix_fmt'],
                'video.fps': numerator / denominator, 'video.channels': 3, 'has_audio': False})
        write_json(meta / 'info.json', info)
        episodes, statistics = [], []
        for number, table in enumerate(tables):
            tasks = [task_names[index] for index in sorted(set(table['task_index'].to_pylist()))]
            episodes.append({'episode_index': number, 'tasks': tasks, 'length': table.num_rows})
            statistics.append({'episode_index': number, 'stats': self.numeric_stats(table)})
        for name, rows in [('episodes', episodes), ('tasks', task_rows), ('episodes_stats', statistics)]:
            (meta / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                                      for row in rows), encoding='utf-8')
        write_json(meta / 'stats.json', self.numeric_stats(pa.concat_tables(tables)))
        calibration = self.root / 'meta/calibration.json'
        if calibration.is_file():
            value = json.loads(calibration.read_text(encoding='utf-8'))
            if 'episodes' in value:
                value['episodes'] = {str(i): value['episodes'][str(record.source_episode)] for i, record in enumerate(records)}
                used = {row['calibration_id'] for row in value['episodes'].values()}
                value['calibrations'] = {key: value['calibrations'][key] for key in used}
            write_json(meta / 'calibration.json', value)
        # 元数据其余静态附件也保留，已重建的索引和统计文件不能被旧版覆盖。
        rebuilt = {path.name for path in meta.iterdir()}
        indexed = {'episodes', 'tasks', 'stats', 'calibration', 'info'}
        for source in (self.root / 'meta').iterdir():
            if source.name in rebuilt or any(source.stem.startswith(prefix) for prefix in indexed):
                continue
            if source.is_dir():
                shutil.copytree(source, meta / source.name)
            else:
                shutil.copy2(source, meta / source.name)

    # 逐 episode 时长对照和柱状图用于验收；最后50条仅作统计分组，不影响选帧。
    def duration_summary(self, folder, records):
        import plotly.graph_objects as go
        original = np.asarray([record.table.num_rows/self.fps for record in records])
        rebuilt = np.asarray([len(record.plan.selected)/self.fps for record in records])
        rows = [{'episode_index': number, 'source_episode': record.source_episode,
                 'source_frames': record.table.num_rows, 'output_frames': len(record.plan.selected),
                 'source_duration_s': float(original[number]), 'output_duration_s': float(rebuilt[number]),
                 'reduction_percent': float((1-rebuilt[number]/original[number])*100)}
                for number, record in enumerate(records)]
        with (folder/'episode_durations.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        groups = {'all': slice(None), 'last50': slice(max(0, len(records)-50), None)}
        if len(records) > 50:
            groups['before_last50'] = slice(None, -50)
        summary = {'target_duration_s': self.target_duration, 'episodes': len(records), 'groups': {}}
        for name, indices in groups.items():
            source, output = original[indices], rebuilt[indices]
            summary['groups'][name] = {'episodes': len(source), 'all_shorter': bool(np.all(output < source)),
                'source_mean_s': float(source.mean()), 'output_mean_s': float(output.mean()),
                'source_median_s': float(np.median(source)), 'output_median_s': float(np.median(output)),
                'source_std_s': float(source.std()), 'output_std_s': float(output.std()),
                'output_min_s': float(output.min()), 'output_max_s': float(output.max()),
                'total_reduction_percent': float((1-output.sum()/source.sum())*100),
                'minimum_reduction_percent': float(np.min(1-output/source)*100)}
        x = np.arange(1, len(records)+1)
        figures = {}
        for label, values, name in [('原始', original, 'duration_original'), ('重构后', rebuilt, 'duration_reconstructed')]:
            fig = go.Figure(go.Bar(x=x, y=values, marker_color='#79a7d8',
                                  customdata=[record.source_episode for record in records],
                                  hovertemplate='源episode %{customdata}<br>%{y:.3f}秒<extra></extra>'))
            fig.add_hline(y=float(values.mean()), line_dash='dash', line_color='#739ece',
                          annotation_text=f'平均 {values.mean():.3f} s', annotation_position='top left')
            fig.add_hline(y=float(np.median(values)), line_dash='dot', line_color='#d49b54',
                          annotation_text=f'中位数 {np.median(values):.3f} s', annotation_position='bottom left')
            fig.update_layout(title=f'{label}每个视频时长（共 {len(records)} 个）',
                              xaxis_title='视频序号（输出episode + 1）', yaxis_title='视频长度（秒）',
                              paper_bgcolor='#fffaf5', plot_bgcolor='#fffaf5', bargap=.15,
                              width=2048, height=760, margin=dict(l=70, r=35, t=65, b=65))
            fig.update_yaxes(range=[0, float(original.max()*1.05)], gridcolor='#eadfd5')
            figures[name] = fig
        comparison = go.Figure()
        comparison.add_bar(x=x, y=original, name='原始', marker_color='#d6d0c8')
        comparison.add_bar(x=x, y=rebuilt, name='重构后', marker_color='#689ed2')
        reduction = (1 - rebuilt / np.maximum(original, 1e-12)) * 100
        comparison.add_trace(go.Scatter(x=x, y=reduction, name='压缩比例', mode='lines+markers',
                                        yaxis='y2', line={'color': '#e08b45', 'width': 2.5},
                                        marker={'color': '#e08b45', 'size': 4},
                                        hovertemplate='源episode %{x}<br>压缩 %{y:.2f}%<extra></extra>'))
        comparison.update_layout(title='所有 episode 时长：原始与重构对照', barmode='overlay',
                                 xaxis_title='视频序号（输出episode + 1）', yaxis_title='视频长度（秒）',
                                 yaxis2={'title': '压缩比例（%）', 'overlaying': 'y', 'side': 'right',
                                         'range': [0, max(100., float(reduction.max()) * 1.1)],
                                         'showgrid': False, 'color': '#b96e2f'},
                                 width=2048, height=760, paper_bgcolor='#fffaf5', plot_bgcolor='#fffaf5')
        figures['duration_comparison'] = comparison
        summary['charts'] = []
        for name, figure in figures.items():
            try:
                figure.write_image(folder/f'{name}.png')
                summary['charts'].append(f'{name}.png')
            except Exception as error:
                summary['png_export_error'] = f'{type(error).__name__}: {error}'
        write_json(folder/'duration_summary.json', summary)
        return summary

    # 对比图使用源帧横轴，保证不同重构长度仍按同一任务进度比较。
    def comparison(self, folder, records, reference):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        comparison_dir = folder / 'vs'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        reference_rows = []
        if reference is not None and (reference / 'source_frame_map.csv').is_file():
            with (reference / 'source_frame_map.csv').open(encoding='utf-8', newline='') as stream:
                reference_rows = list(csv.DictReader(stream))
        reference_reports = sorted(reference.glob('*report*.json')) if reference is not None else []
        reference_source = None
        reference_hashes = {}
        for report_path in reference_reports:
            baseline = json.loads(report_path.read_text(encoding='utf-8'))
            reference_source = baseline.get('source_episode', reference_source)
            for item in baseline.get('episodes', [baseline]):
                if 'source_episode' in item and 'source_parquet_sha256' in item:
                    reference_hashes[int(item['source_episode'])] = item['source_parquet_sha256']
        for output_episode, record in enumerate(records):
            chosen = np.asarray(record.plan.selected, dtype=int)
            baseline_ids = np.asarray([int(row['source_frame']) for row in reference_rows
                                       if int(row.get('source_episode', reference_source if reference_source is not None else 0))
                                       == record.source_episode], dtype=int)
            if len(baseline_ids) and (reference_hashes.get(record.source_episode) != record.source_hash
                                      or baseline_ids.max() >= record.table.num_rows or baseline_ids.min() < 0):
                baseline_ids = np.array([], dtype=int)
            fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'scene'}, {'type': 'xy'}],
                                                       [{'type': 'scene'}, {'type': 'xy'}]],
                                subplot_titles=['左臂轨迹', '左臂速度（源帧对应）', '右臂轨迹', '右臂速度（源帧对应）'])
            comparison = {}
            for row, arm in enumerate(('left', 'right'), 1):
                selections = [(np.arange(record.table.num_rows), '原始', '#999999'),
                              (chosen, '通用规则输出', '#e66b15' if arm == 'left' else '#277be9')]
                if len(baseline_ids):
                    selections.append((baseline_ids, '先前参考', '#9c42c4'))
                for ids, label, color in selections:
                    positions = record.analyzer.positions[arm][ids] * np.array([-1, -1, 1])
                    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) * self.fps
                    fig.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                                              mode='lines+markers', name=f'{arm} {label}', customdata=ids,
                                              marker={'size': 2}, line={'color': color, 'width': 3},
                                              hovertemplate='源帧 %{customdata}<br>(%{x}, %{y}, %{z})<extra>%{fullData.name}</extra>'), row=row, col=1)
                    fig.add_trace(go.Scatter(x=ids[1:], y=speed, name=f'{arm} {label}', line={'color': color},
                                             showlegend=False), row=row, col=2)
                comparison[arm] = {'raw': record.analyzer.metrics(np.arange(record.table.num_rows), arm),
                                   'output': record.analyzer.metrics(chosen, arm)}
                if len(baseline_ids):
                    comparison[arm]['reference'] = record.analyzer.metrics(baseline_ids, arm)
                for phase in record.plan.report.get('phases', []):
                    fig.add_vrect(x0=phase['start'], x1=phase['end'], opacity=.05, line_width=0,
                                  fillcolor='#277be9' if phase['arm'] == 'right' else '#e66b15',
                                  row=row, col=2, exclude_empty_subplots=False)
            fig.update_layout(height=950, title=f'源 episode {record.source_episode} · 通用五阶段重构 · 原始坐标镜像 X/Y 显示',
                              scene={'aspectmode': 'data'}, scene2={'aspectmode': 'data'})
            filename = 'comparison.html' if len(records) == 1 else f'comparison_episode_{output_episode:06d}.html'
            fig.write_html(comparison_dir / filename, include_plotlyjs=True)
            record.plan.report['effect_comparison'] = comparison
            record.plan.report['reference_used_for_selection'] = False

    # 全部 episode、相机和元数据在临时目录验证通过后一起发布。
    def build(self, records):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        reference = snapshot_reference(self.output)
        with tempfile.TemporaryDirectory(prefix='.reconstruct-', dir=self.output.parent) as temporary, \
                video_executor(self.workers) as pool:
            folder = Path(temporary) / 'result'
            folder.mkdir()
            global_index, tables, reports = 0, [], []
            video_jobs = {}
            mapping = folder / 'source_frame_map.csv'
            deletion_dir = folder / 'vs'
            deletion_dir.mkdir()
            with mapping.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['episode_index', 'source_episode', 'output_frame', 'source_frame',
                                 'output_timestamp', 'source_timestamp', 'phase'])
                for number, record in enumerate(records):
                    selected = np.asarray(record.plan.selected, dtype=np.int64)
                    out = record.table.take(pa.array(selected, type=pa.int64()))
                    changed = {'frame_index': np.arange(len(selected)), 'index': np.arange(global_index, global_index + len(selected)),
                               'timestamp': np.arange(len(selected)) / self.fps, 'episode_index': np.full(len(selected), number)}
                    for name, values in changed.items():
                        field = out.schema.field(name)
                        out = out.set_column(out.schema.get_field_index(name), field, pa.array(values, type=field.type))
                    chunk = number // self.chunk_size
                    destination = folder / f'data/chunk-{chunk:03d}/episode_{number:06d}.parquet'
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(out, destination, compression='snappy')
                    loaded = pq.read_table(destination)
                    if not loaded.schema.equals(record.table.schema, check_metadata=True):
                        raise ValueError('输出 parquet schema 改变')
                    for name in loaded.column_names:
                        expected = out[name] if name in changed else record.table[name].take(pa.array(selected))
                        if not loaded[name].equals(expected):
                            raise ValueError(f'来源行验证失败：{record.source} {name}')
                    reports.append(record.plan.report)
                    record.plan.report['output_episode'] = number
                    record.plan.report['videos'] = {}
                    for camera, key in enumerate(self.video_keys, 1):
                        video = self.root / self.info['video_path'].format(episode_chunk=record.source_episode // self.chunk_size,
                                                                          episode_index=record.source_episode, video_key=key)
                        destination = folder / f'videos/chunk-{chunk:03d}' / key / f'episode_{number:06d}.mp4'
                        job = pool.submit(rewrite_video_cached, video, destination, selected, self.fps, self.cache_dir)
                        video_jobs[job] = (record, key)
                    writer.writerows((number, record.source_episode, i, int(source_frame), i / self.fps,
                                      record.table['timestamp'][int(source_frame)].as_py(), int(record.plan.phase_ids[source_frame]))
                                     for i, source_frame in enumerate(selected))
                    # 覆盖所有来源帧，空单元格表示保留；与最终视频和 parquet 共用同一选帧结果。
                    retained = np.zeros(record.table.num_rows, dtype=bool)
                    retained[selected] = True
                    deletion_log = deletion_dir / f'episode_{number:06d}.csv'
                    with deletion_log.open('w', encoding='utf-8-sig', newline='') as deletion_stream:
                        deletion_writer = csv.writer(deletion_stream)
                        deletion_writer.writerow(['原始帧', '是否删除'])
                        deletion_writer.writerows((frame, '' if keep else '1')
                                                  for frame, keep in enumerate(retained))
                    unchanged = sha256(record.source) == record.source_hash
                    if not unchanged:
                        raise ValueError(f'重构期间源 parquet 发生改变：{record.source}')
                    record.plan.report.setdefault('validation', {}).update({'schema_preserved': True, 'all_source_rows_equal': True,
                                                       'indices_timestamps_rebuilt': True, 'video_frames_pixel_equal': True,
                                                       'video_count': len(self.video_keys), 'source_sha256_unchanged': unchanged})
                    tables.append(loaded)
                    global_index += len(selected)
            # 每路视频使用独立来源索引和输出路径；任一失败则等待工作线程结束并清理临时目录。
            for completed, job in enumerate(as_completed(video_jobs), 1):
                record, key = video_jobs[job]
                try:
                    record.plan.report['videos'][key] = job.result()
                except BaseException:
                    for pending in video_jobs:
                        pending.cancel()
                    raise
                hit = record.plan.report['videos'][key].get('cache_hit', False)
                print(f'视频验证 {completed}/{len(video_jobs)}: episode {record.source_episode}, {key}'
                      f'{"（缓存）" if hit else ""}', flush=True)
            for record in records:
                if sha256(record.source) != record.source_hash:
                    raise ValueError(f'视频编码期间源 parquet 发生改变：{record.source}')
            self.metadata(folder, records, tables)
            self.comparison(folder, records, reference)
            duration_summary = self.duration_summary(folder, records) if self.target_duration is not None else None
            report = {'strategy': 'generic_five_phase_v1', 'source_root': str(self.root), 'output': str(self.output),
                      'rules': asdict(self.rules), 'video_keys': self.video_keys,
                      'target_duration_s': self.target_duration, 'duration_summary': duration_summary,
                      'deletion_csv': 'vs/episode_XXXXXX.csv', 'deletion_csv_files': [f'vs/episode_{i:06d}.csv' for i in range(len(records))], 'total_episodes': len(records),
                      'source_frames': sum(record.table.num_rows for record in records), 'output_frames': global_index,
                      'reference': str(reference) if reference else None, 'episodes': reports,
                      'validation': {'all_episodes_verified': True, 'metadata_rebuilt': True}}
            write_json(folder / 'reconstruction_report.json', report)
            publish(folder, self.output)
        return report


# 默认只生成 episode 0；批量分析显式启用，分析命令不会写文件或编码视频。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--source-parquet', type=Path)
    source.add_argument('--source-root', type=Path, action='append', help='数据集目录或包含多个数据集的父目录，可重复指定')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--episode', type=int, default=None)
    selection.add_argument('--all-episodes', action='store_true')
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--output', type=Path)
    destination.add_argument('--output-root', type=Path, default=Path(__file__).parent / 'output')
    parser.add_argument('--rules-json', type=Path)
    parser.add_argument('--target-duration', type=float, metavar='SECONDS',
                        help='在位姿步长和完整事件约束下向共同目标时长优化；不指定时沿用原选帧规则')
    parser.add_argument('--workers', type=int, default=16, help='分析进程数/视频编码并发数，默认16；设为1串行执行')
    parser.add_argument('--video', nargs='+', action='extend', metavar='VIDEO_KEY',
                        help='只生成指定视频视角，支持多个名称或重复 --video；不指定时生成全部视角')
    parser.add_argument('--no-cache', action='store_true', help='禁用分析和已验证视频缓存；默认缓存支持中断后复用')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.source_parquet and (args.all_episodes or args.episode is not None):
        parser.error('--source-parquet 已明确指定 episode，不能再指定 --episode/--all-episodes')
    rules = Rules.from_json(args.rules_json)
    jobs = []
    if args.source_root:
        for candidate in args.source_root:
            if (candidate / 'meta/info.json').is_file():
                jobs.append((candidate, None))
            else:
                datasets = sorted(path for path in candidate.iterdir() if path.is_dir() and (path / 'meta/info.json').is_file())
                if not datasets:
                    parser.error(f'没有找到 LeRobot 数据集：{candidate}')
                jobs.extend((path, None) for path in datasets)
    else:
        parquet = args.source_parquet or DEFAULT_SOURCE
        jobs.append((parquet.parents[2], None if args.all_episodes or args.episode is not None else [parquet]))
    if args.output and len(jobs) != 1:
        parser.error('多个数据集必须使用 --output-root')
    outputs, roots = set(), set()
    results = []
    builders = []
    for root, paths in jobs:
        output = args.output or args.output_root / root.name
        if output.resolve() in outputs or root.resolve() in roots:
            parser.error('重复源数据集或同名输出目录，请分开运行并指定 --output')
        outputs.add(output.resolve())
        roots.add(root.resolve())
        builders.append(DatasetBuilder(root, output, rules, paths, args.episode or 0, args.all_episodes,
                                       video_keys=args.video, target_duration=args.target_duration, workers=args.workers,
                                       use_cache=not (args.no_cache or args.plan_only)))
    for output in outputs:
        if any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
            parser.error('任何输出都不能覆盖本次提交的其他源数据集')
    for number, builder in enumerate(builders, 1):
        print(f'数据集 {number}/{len(builders)} 开始：{builder.root}', file=sys.stderr, flush=True)
        records = builder.analyze()
        if args.plan_only:
            results.append({'source_root': str(builder.root), 'output': str(builder.output), 'rules': asdict(rules),
                            'video_keys': builder.video_keys,
                            'target_duration_s': builder.target_duration,
                            'total_episodes': len(records), 'source_frames': sum(row.table.num_rows for row in records),
                            'output_frames': sum(len(row.plan.selected) for row in records),
                            'episodes': [row.plan.report for row in records]})
        else:
            results.append(builder.build(records))
            print(f'数据集 {number}/{len(builders)} 已验证并发布：{builder.output}', file=sys.stderr, flush=True)
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
