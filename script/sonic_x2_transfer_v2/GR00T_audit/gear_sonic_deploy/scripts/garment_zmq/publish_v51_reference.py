#!/usr/bin/env python3
"""X2 Sonic v5.1 严格参考速度 ZMQ publisher(参考实现)。

规格来源: ``docs/X2_ZMQ_V51_WIRE_SPEC.md``,该文档逐行提取自机器人侧解码器
``src/x2/agi_x2_deploy_onnx_ref/src/zmq_pose_input_source.cpp``。
本文件的目标是:产出的字节能被 ``--strict-reference-velocity`` 模式**接受**,
并显式区分 MuJoCo parity 与实时 HMCP 的速度语义。

速度契约有两个显式模式:

    fixed:     v[t] = (q[t] - q[t-1]) * 50.0
    timestamp: v[t] = (q[t] - q[t-1]) / (timestamp[t] - timestamp[t-1])

  1. ``fixed`` 保留 MuJoCo LiveMotion 的常数 50.0 语义;
  2. ``timestamp`` 用于实时 HMCP,避免低于 50 Hz 时放大物理速度;
  3. 该差分发生在 live-edge clamp **之前**;
  4. clamp 之后 9 个未来 slot 重复最新位姿,**同时重复最新速度,不置零**。

依赖: 仅 numpy(打包/校验)+ pyzmq(仅发送时需要,import 被延迟)。

用法:
    python publish_v51_reference.py --host 0.0.0.0 --port 5556 --rate 50 --duration 10
    python publish_v51_reference.py --source hmcp --host 127.0.0.1 --duration 30
    python publish_v51_reference.py --source jsonl --jsonl full_body_shadow.jsonl
    python publish_v51_reference.py --drop-velocity      # 故意触发 strict 拒帧
    python publish_v51_reference.py --stop-after-s 8     # 故意断流,测 watchdog
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 常量:全部与 C++ 侧一一对应,改这里之前先改 C++
# --------------------------------------------------------------------------

#: policy_parameters.hpp:54
NUM_DOFS = 31
#: policy_parameters.hpp:56 —— 窗口总长 10(slot0 = 当前帧)
NUM_FUTURE_FRAMES = 10
#: zmq_pose_input_source.cpp:244 —— 严格未来 slot 数 = 10 - 1
NUM_FUTURE_SLOTS = NUM_FUTURE_FRAMES - 1
#: policy_parameters.hpp:60 —— 未来 slot 间距;必须落在 C++ 接受的 [0.01, 1.0]
#: 区间内(cpp:328),否则被静默改回 0.1
FUTURE_DT_S = 0.1
#: zmq_packed_message_subscriber.hpp:99 —— header 必须正好这么长
HEADER_SIZE = 1280
#: 速度差分频率。LiveMotion 缓冲区的标称 fps,与真实到帧率解耦(见规格 9.1)
VELOCITY_DIFF_HZ = 50.0
#: 实时 HMCP 时间戳差分的有效区间。区间外重置导数,不制造速度尖峰。
TIMESTAMP_MIN_DT_S = 0.005
TIMESTAMP_RESET_GAP_S = 0.2
VELOCITY_MODES = ("fixed", "timestamp")
POSE_SCALE = 0.7

# HMCP v1:衣服进程发出的 G1 qpos36。格式逐字复用 sonic_support.py。
HMCP_MAGIC = b"HMCP"
HMCP_VERSION = 1
HMCP_HEADER = struct.Struct("<4sBBI8sH")
HMCP_FLAG_HAS_HAND = 0x01
HMCP_FLAG_HAS_BRAINCO = 0x02
HMCP_KNOWN_FLAGS = HMCP_FLAG_HAS_HAND | HMCP_FLAG_HAS_BRAINCO
G1_QPOS_N = 36
HMCP_CAPTURE_SCHEMA = "x2-hmcp-capture-v1"

#: policy_parameters.hpp:63-95 —— 31 维的关节顺序(MJCF 序,唯一正确的顺序)
MUJOCO_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
)

# GMR/HMCP 的 29 关节顺序。腰和腕顺序与 X2 MJCF 不同,所以必须按名字映射。
G1_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# G1 参考基准;肘是 +0.6。X2 训练默认肘是 -0.6,不要把二者混用。
G1_DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)
assert len(G1_JOINT_NAMES) == G1_DEFAULT_ANGLES.shape[0] == 29

#: policy_parameters.hpp:212-244 —— 训练站姿。合成动作以它为中心摆动,
#: 这样即使误接真机也只是围绕训练默认姿态的小幅运动。
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # 左腿
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # 右腿
    0.0, 0.0, 0.0,                             # 腰 yaw/pitch/roll
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,        # 左臂 + 左腕
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,       # 右臂 + 右腕
    0.0, 0.0,                                  # 头 yaw/pitch
], dtype=np.float64)
assert DEFAULT_ANGLES.shape == (NUM_DOFS,)

#: xyzw 顺序的单位四元数(reference_motion.hpp:42 是 scipy 约定)。
#: 注意:wxyz 的单位四元数范数同样是 1,C++ 的范数校验抓不到顺序错误。
IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

#: 允许出现在 wire 上的 dtype 白名单。C++ 用 dtype **字符串**精确比较
#: (cpp:176 要 "f32",cpp:209 要 "i64"),所以不能靠字节宽度蒙对。
_NUMPY_TO_DTYPE_STR = {
    np.dtype(np.float32): "f32",
    np.dtype(np.int64): "i64",
}


@dataclass(frozen=True)
class HmcpPacket:
    sequence: int
    flags: int
    send_timestamp_s: float | None
    qpos36: np.ndarray


def parse_hmcp(raw: bytes) -> HmcpPacket | None:
    """Strictly validate one garment HMCP v1 datagram."""
    if len(raw) < HMCP_HEADER.size:
        return None
    magic, version, flags, sequence, timestamp_raw, count = HMCP_HEADER.unpack_from(raw)
    trailing_floats = (
        (4 if flags & HMCP_FLAG_HAS_HAND else 0)
        + (24 if flags & HMCP_FLAG_HAS_BRAINCO else 0)
    )
    expected_size = HMCP_HEADER.size + 4 * (count + trailing_floats)
    if magic != HMCP_MAGIC or version != HMCP_VERSION:
        return None
    if flags & ~HMCP_KNOWN_FLAGS:
        return None
    if count != G1_QPOS_N or len(raw) != expected_size:
        return None
    qpos = np.frombuffer(raw, dtype="<f4", count=count, offset=HMCP_HEADER.size)
    if qpos.shape != (G1_QPOS_N,) or not np.isfinite(qpos).all():
        return None
    send_timestamp_s = struct.unpack("<d", timestamp_raw)[0]
    if not np.isfinite(send_timestamp_s) or send_timestamp_s <= 0.0:
        send_timestamp_s = None
    return HmcpPacket(
        int(sequence),
        int(flags),
        send_timestamp_s,
        qpos.astype(np.float64, copy=True),
    )


def is_newer_sequence(sequence: int, previous: int | None) -> bool:
    """Return true for a new uint32 sequence, including wraparound."""
    if previous is None:
        return True
    delta = (int(sequence) - int(previous)) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


def hmcp_to_x2_reference(
    qpos36: np.ndarray,
    pose_scale: float = POSE_SCALE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the proven HumanPlus named-joint mapping to one HMCP qpos36.

    Returns ``(joint_pos_mj31, root_quat_xyzw4, root_pos3)``. HMCP stores the
    root quaternion as WXYZ; the ZMQ/C++ contract is XYZW.
    """
    qpos = np.asarray(qpos36, dtype=np.float64).reshape(-1)
    if qpos.shape != (G1_QPOS_N,) or not np.isfinite(qpos).all():
        raise ValueError(f"HMCP qpos must be finite ({G1_QPOS_N},), got {qpos.shape}")
    if not 0.0 <= pose_scale <= 1.0:
        raise ValueError(f"pose_scale must be in [0, 1], got {pose_scale}")

    root_pos = qpos[:3].copy()
    root_quat_wxyz = qpos[3:7].copy()
    quat_norm = float(np.linalg.norm(root_quat_wxyz))
    if quat_norm < 1e-6:
        root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        root_quat_wxyz /= quat_norm
    root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]

    garment_joints = qpos[7:]
    delta_by_name = {
        name: garment_joints[i] - G1_DEFAULT_ANGLES[i]
        for i, name in enumerate(G1_JOINT_NAMES)
    }
    unscaled = np.asarray([
        DEFAULT_ANGLES[i] + delta_by_name.get(name, 0.0)
        for i, name in enumerate(MUJOCO_JOINT_NAMES)
    ], dtype=np.float64)
    joint_pos = DEFAULT_ANGLES + pose_scale * (unscaled - DEFAULT_ANGLES)
    return joint_pos, root_quat_xyzw, root_pos


# --------------------------------------------------------------------------
# 打包:topic 前缀 + 1280 字节 JSON header + 紧密拼接的二进制段
# --------------------------------------------------------------------------


def build_header(fields: Sequence[dict], version: int = 5, count: int = 1) -> bytes:
    """构造正好 HEADER_SIZE 字节的 JSON header。

    * ``separators=(",", ":")`` 去掉空格,给 1280 字节留余量
      (最小字段集实测 475 字节)。参考 zmq_planner_sender.py:24。
    * 必须用 ``\\x00`` 右补齐:C++ 侧用 ``strnlen(data, 1280)`` 取 JSON 长度
      (zmq_packed_message_subscriber.hpp:292),没有 NUL 就会把后面的二进制
      当成 JSON 的一部分。
    * ``endian`` 固定 "le":C++ 的 pose 解码路径无条件按小端读
      (cpp:200 传 needs_swap=false),声明 "be" 不会被拒,只会读出垃圾。
    * ``v`` / ``count`` 被解析但 pose 路径不检查,写 5 / 1 只为日志可读。
    """
    header = {"v": version, "endian": "le", "count": count, "fields": list(fields)}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(raw) > HEADER_SIZE:
        raise ValueError(f"JSON header {len(raw)} 字节 > {HEADER_SIZE},无法表达")
    return raw.ljust(HEADER_SIZE, b"\x00")


def pack_message(
    named_arrays: Sequence[Tuple[str, np.ndarray]],
    topic: str = "pose",
    version: int = 5,
) -> bytes:
    """把 (字段名, ndarray) 序列打包成单个 ZMQ part 的字节串。

    关键约束:
    * header 里 ``fields`` 的顺序**必须**等于二进制段的拼接顺序
      (C++ 用 ComputeByteSize() 累加 offset,subscriber.hpp:334-346)。
    * ``shape`` 必须显式写出,标量写 ``[1]``:shape 为空时 C++ 的
      ComputeByteSize() 返回 0(subscriber.hpp:155),该字段占 0 字节,
      之后所有字段全部错位。
    * 无对齐填充。
    * 用显式白名单(而不是遍历上游 dict)构造字段,是为了杜绝把名为
      ``estop`` 的键顺手带上——C++ 只要看到这个字段名就**无条件永久闭锁
      e-stop,根本不读值**(cpp:265-268)。
    """
    fields: List[dict] = []
    chunks: List[bytes] = []
    for name, arr in named_arrays:
        if name == "estop":
            raise ValueError("拒绝打包 'estop':C++ 只看字段名就会闸停机器人")
        dtype_str = _NUMPY_TO_DTYPE_STR.get(arr.dtype)
        if dtype_str is None:
            raise ValueError(f"字段 '{name}' dtype {arr.dtype} 不在白名单内")
        # C 行主序:未来数组按 [slot, dof] 平铺,C++ 端也是这样连续读的
        # (cpp:300-306 用 future_jpos[0].data() 一次写满 9x31)。
        # 显式转小端:声明了 "le" 就必须真的是小端字节。
        arr = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False)
        fields.append({"name": name, "dtype": dtype_str, "shape": list(arr.shape)})
        chunks.append(arr.tobytes(order="C"))
    return topic.encode("utf-8") + build_header(fields, version=version) + b"".join(chunks)


# --------------------------------------------------------------------------
# 速度契约 + live-edge clamp
# --------------------------------------------------------------------------


class V51ReferenceStream:
    """维护相邻参考位姿,按所选速度语义产出 clamp 后的 v5.1 帧。

    ``fixed`` 逐字保留 MuJoCo LiveMotion 的标称 50 Hz 差分。实时 HMCP
    使用 ``timestamp``:协议头中的发送时间由衣服进程在 HMCP 发布前写入,
    因而不会把 28-35 Hz 的参考误当作 50 Hz。异常时间间隔只重置导数,
    当前位姿仍可继续发送。
    """

    def __init__(
        self,
        diff_hz: float = VELOCITY_DIFF_HZ,
        *,
        velocity_mode: str = "fixed",
        timestamp_min_dt_s: float = TIMESTAMP_MIN_DT_S,
        timestamp_reset_gap_s: float = TIMESTAMP_RESET_GAP_S,
    ) -> None:
        self._diff_hz = float(diff_hz)
        if velocity_mode not in VELOCITY_MODES:
            raise ValueError(f"velocity_mode must be one of {VELOCITY_MODES}, got {velocity_mode!r}")
        if timestamp_min_dt_s <= 0.0:
            raise ValueError("timestamp_min_dt_s must be > 0")
        if timestamp_reset_gap_s <= timestamp_min_dt_s:
            raise ValueError("timestamp_reset_gap_s must be > timestamp_min_dt_s")
        self._velocity_mode = velocity_mode
        self._timestamp_min_dt_s = float(timestamp_min_dt_s)
        self._timestamp_reset_gap_s = float(timestamp_reset_gap_s)
        self._latest_pos: np.ndarray | None = None
        self._latest_vel = np.zeros(NUM_DOFS, dtype=np.float64)
        self._latest_quat = IDENTITY_QUAT_XYZW.copy()
        self._latest_timestamp_s: float | None = None
        self._last_dt_s: float | None = None
        self._derivative_resets = 0
        self._last_reset_reason = "first-frame"

    def push(
        self,
        joint_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        *,
        timestamp_s: float | None = None,
    ) -> None:
        """吃进一个新的参考位姿。**差分在 clamp 之前完成,就在这里。**"""
        pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        quat = np.asarray(root_quat_xyzw, dtype=np.float64).reshape(-1)
        if pos.shape != (NUM_DOFS,):
            raise ValueError(f"joint_pos 必须是 ({NUM_DOFS},),收到 {pos.shape}")
        if quat.shape != (4,):
            raise ValueError(f"root_quat_xyzw 必须是 (4,) 的 xyzw,收到 {quat.shape}")
        if not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise ValueError("参考里有 NaN/Inf;不要发,直接丢这一帧")

        sample_timestamp_s = None
        if timestamp_s is not None and np.isfinite(timestamp_s):
            sample_timestamp_s = float(timestamp_s)

        if self._latest_pos is None:
            # 首帧没有 pos[t-1]。速度填 0,但这一帧照常发:
            # strict 只要求 joint_vel_mj 字段**存在**(cpp:355),不要求非零。
            self._latest_vel = np.zeros(NUM_DOFS, dtype=np.float64)
            self._last_dt_s = None
        elif self._velocity_mode == "fixed":
            self._latest_vel = (pos - self._latest_pos) * self._diff_hz
            self._last_dt_s = 1.0 / self._diff_hz
        else:
            dt_s = None
            if sample_timestamp_s is not None and self._latest_timestamp_s is not None:
                dt_s = sample_timestamp_s - self._latest_timestamp_s
            self._last_dt_s = dt_s
            if dt_s is None:
                self._reset_derivative("missing-timestamp")
            elif dt_s < self._timestamp_min_dt_s:
                reason = "non-monotonic-timestamp" if dt_s <= 0.0 else "timestamp-gap-too-small"
                self._reset_derivative(reason)
            elif dt_s > self._timestamp_reset_gap_s:
                self._reset_derivative("timestamp-gap-too-large")
            else:
                self._latest_vel = (pos - self._latest_pos) / dt_s

        self._latest_pos = pos
        self._latest_timestamp_s = sample_timestamp_s
        # 发送前归一化到 1。C++ 会自己归一化(cpp:59-71),但只接受范数落在
        # [0.5, 2.0] 的输入,发送端归一化能把这条拒帧条件彻底排除。
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError("四元数范数退化,无法归一化")
        self._latest_quat = quat / norm

    def _reset_derivative(self, reason: str) -> None:
        self._latest_vel = np.zeros(NUM_DOFS, dtype=np.float64)
        self._derivative_resets += 1
        self._last_reset_reason = reason

    def velocity_status(self) -> str:
        dt = "n/a" if self._last_dt_s is None else f"{self._last_dt_s * 1000.0:.1f}ms"
        return (
            f"velocity_mode={self._velocity_mode} dt={dt} "
            f"derivative_resets={self._derivative_resets} "
            f"last_reset={self._last_reset_reason}"
        )

    @property
    def ready(self) -> bool:
        return self._latest_pos is not None

    def build_fields(
        self,
        frame_index: int,
        *,
        drop_velocity: bool = False,
    ) -> List[Tuple[str, np.ndarray]]:
        """产出 wire 字段列表(顺序即二进制拼接顺序)。

        live-edge clamp 就在这里:9 个未来 slot **全部重复最新位姿和最新姿态**,
        并且 ``joint_vel_mj_future`` 重复的是**最新的非零速度,不是 0**
        (README_X2_GARMENT_LIVE.md:136-147)。

        由此产生一个反直觉但正确的结果:窗口内位姿斜率为 0,速度却非零。
        不要"修正"它 —— 也正因为如此,``joint_vel_mj_future`` 必须显式发:
        C++ 的回退分支会用 (future_jpos[k]-prev)/future_dt_s 算出**全零**
        (cpp:425-435),那会破坏 parity。

        Args:
            frame_index: 单调计数,仅诊断用(C++ 不校验、不排序)。
            drop_velocity: 故意不发 ``joint_vel_mj``,用来验证 strict 拒帧
                R1 ``strict mode: missing joint_vel_mj field``。
        """
        if self._latest_pos is None:
            raise RuntimeError("还没 push 过任何参考")

        pos = self._latest_pos
        vel = self._latest_vel
        quat = self._latest_quat

        # np.tile 复制 9 份 —— 这就是 live-edge clamp
        jpos_future = np.tile(pos, (NUM_FUTURE_SLOTS, 1))
        jvel_future = np.tile(vel, (NUM_FUTURE_SLOTS, 1))
        quat_future = np.tile(quat, (NUM_FUTURE_SLOTS, 1))

        # 发送前自查有限性。C++ 的 AllFinite 只覆盖 joint_pos_mj / joint_vel_mj
        # (cpp:339, 348),三个 *_future 数组**完全没有有限性校验**(规格 9.3),
        # NaN 会被接受并一路流进 ONNX。这道防线只能由发送端提供。
        for name, arr in (
            ("joint_pos_mj", pos), ("joint_vel_mj", vel), ("root_quat_xyzw", quat),
            ("joint_pos_mj_future", jpos_future),
            ("joint_vel_mj_future", jvel_future),
            ("root_quat_xyzw_future", quat_future),
        ):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} 含 NaN/Inf,拒绝发送")

        # 字段顺序 = 二进制拼接顺序。C++ 按名字匹配,顺序本身不重要,
        # 但 header 与载荷必须同序,所以固定成一个确定的顺序便于比对。
        fields: List[Tuple[str, np.ndarray]] = [
            # strict 必需:唯一的 "body 帧" 触发器(cpp:269-272, 338)
            ("joint_pos_mj", pos.astype(np.float32)),
        ]
        if not drop_velocity:
            # strict 必需:v5.1 的核心字段,C++ 原样转发不重估(cpp:319-324, 373)
            fields.append(("joint_vel_mj", vel.astype(np.float32)))
        fields += [
            # strict 不查它,但缺了 slot0 姿态就是 bootstrap 单位四元数
            # (cpp:118),策略会把身体朝世界 +X 拧。必须发(规格 9.5)。
            ("root_quat_xyzw", quat.astype(np.float32)),
            # strict 必需的三个未来数组,shape 分别 [9,31] / [9,31] / [9,4]
            ("joint_pos_mj_future", jpos_future.astype(np.float32)),
            ("joint_vel_mj_future", jvel_future.astype(np.float32)),
            ("root_quat_xyzw_future", quat_future.astype(np.float32)),
            # 必须是 0.1:它同时决定 Sample(t) 的 slot 索引(cpp:497-503)。
            # 落在 [0.01, 1.0] 之外会被静默改回 0.1(cpp:328)。
            ("future_dt_s", np.array([FUTURE_DT_S], dtype=np.float32)),
            # 可选,纯诊断。dtype 必须是 i64(cpp:209),shape 必须 [1]。
            ("frame_index", np.array([int(frame_index)], dtype=np.int64)),
        ]
        # 刻意不发的字段:
        #   motion_token       —— 被解码缓存但没有任何读取接口,死字段
        #   frame_index_future —— hpp:50 声明了,HandleDecoded 里没有分支
        #   left/right_hand_joints —— 衣服通路不控手;且硬编码必须是 10 个元素
        #   estop              —— 只要字段名出现就无条件永久闭锁 e-stop
        return fields


# --------------------------------------------------------------------------
# 数据源 (a):可复现的合成动作 —— 无衣服时的通路自检
# --------------------------------------------------------------------------


def synthetic_source(
    rate_hz: float,
    amplitude_rad: float = 0.05,
    period_s: float = 10.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """围绕训练站姿的缓慢正弦,完全由帧序号决定(可复现,无随机)。

    只动 4 个上肢关节(双肩 pitch + 双肘),幅值默认 0.05 rad ≈ 2.9°,
    周期 10 s。理由:
      * 腿/腰的参考变化在有承载的支撑测试里风险更高,先不动;
      * 幅值小到即使误接真机也只是围绕默认姿态的微动;
      * 但速度非零,足以让 has_explicit_reference_velocity() 变 true
        并让 tokenizer 真的看到非零参考速度。

    四元数固定单位:根姿态参考是策略的强输入,合成源不该乱给。
    """
    moving = [
        MUJOCO_JOINT_NAMES.index(n) for n in (
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_elbow_joint", "right_elbow_joint",
        )
    ]
    omega = 2.0 * np.pi / float(period_s)
    k = 0
    while True:
        t = k / float(rate_hz)
        pos = DEFAULT_ANGLES.copy()
        s = amplitude_rad * np.sin(omega * t)
        for j in moving:
            pos[j] += s
        yield pos, IDENTITY_QUAT_XYZW.copy()
        k += 1


# --------------------------------------------------------------------------
# 数据源 (b):确定格式的 JSONL 回放
# --------------------------------------------------------------------------


def _wxyz_to_xyzw(values: Sequence[float]) -> np.ndarray:
    quat = np.asarray(values, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"root quaternion must be finite (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        raise ValueError("root quaternion is zero")
    quat /= norm
    return quat[[1, 2, 3, 0]]


def jsonl_source(
    path: str,
    pose_scale: float = POSE_SCALE,
    *,
    include_timestamp: bool = False,
) -> Iterator[
    Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, float | None]
]:
    """Replay either this adapter's HMCP capture or an existing shadow-v1 log.

    Accepted records are deliberately narrow:

    * ``schema=x2-hmcp-capture-v1`` with raw ``qpos36``;
    * ``type=sample`` with ``reference_position`` and the existing shadow
      logger's WXYZ ``reference_root_quaternion``.

    Repeated/out-of-order ``sequence`` or ``garment_sequence`` values are
    skipped. That preserves one LiveMotion advance per new garment frame.
    """
    last_sequence: int | None = None
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("type") == "metadata":
                continue

            if record.get("schema") == HMCP_CAPTURE_SCHEMA:
                sequence = int(record["sequence"])
                if not is_newer_sequence(sequence, last_sequence):
                    continue
                joint_pos, quat_xyzw, _ = hmcp_to_x2_reference(
                    np.asarray(record["qpos36"], dtype=np.float64), pose_scale
                )
                sample_timestamp_s = record.get("velocity_timestamp_s")
                if sample_timestamp_s is None:
                    sample_timestamp_s = record.get("send_timestamp_s")
                if sample_timestamp_s is None and "received_monotonic_ns" in record:
                    sample_timestamp_s = int(record["received_monotonic_ns"]) * 1e-9
            elif record.get("type") == "sample":
                if "reference_position" not in record or "reference_root_quaternion" not in record:
                    raise ValueError(
                        f"{path}:{line_number}: shadow sample lacks reference fields"
                    )
                sequence = int(record.get("garment_sequence", line_number))
                if not is_newer_sequence(sequence, last_sequence):
                    continue
                joint_pos = np.asarray(record["reference_position"], dtype=np.float64).reshape(-1)
                if joint_pos.shape != (NUM_DOFS,) or not np.isfinite(joint_pos).all():
                    raise ValueError(
                        f"{path}:{line_number}: reference_position must be finite ({NUM_DOFS},)"
                    )
                quat_xyzw = _wxyz_to_xyzw(record["reference_root_quaternion"])
                sample_timestamp_s = record.get("timestamp_s")
            else:
                raise ValueError(
                    f"{path}:{line_number}: unsupported JSONL record; expected "
                    f"{HMCP_CAPTURE_SCHEMA!r} or shadow type=sample"
                )

            last_sequence = sequence
            if sample_timestamp_s is not None:
                sample_timestamp_s = float(sample_timestamp_s)
                if not np.isfinite(sample_timestamp_s):
                    sample_timestamp_s = None
            if include_timestamp:
                yield joint_pos, quat_xyzw, sample_timestamp_s
            else:
                yield joint_pos, quat_xyzw


# --------------------------------------------------------------------------
# 数据源 (c):实时 HMCP UDP,每个新 garment sequence 只推进一次
# --------------------------------------------------------------------------


class HmcpUdpSource:
    def __init__(
        self,
        bind_host: str,
        port: int,
        pose_scale: float,
        poll_timeout_s: float,
        capture_jsonl: str | None,
    ) -> None:
        self.pose_scale = float(pose_scale)
        self.received = 0
        self.invalid = 0
        self.duplicate_or_old = 0
        self.accepted = 0
        self.hmcp_send_timestamps = 0
        self.receive_timestamp_fallbacks = 0
        self.last_sequence: int | None = None
        self.last_valid_s = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, int(port)))
        self.sock.settimeout(float(poll_timeout_s))

        self.capture = None
        if capture_jsonl:
            capture_path = Path(capture_jsonl).expanduser()
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            self.capture = capture_path.open("x", encoding="utf-8", buffering=1)
            self.capture.write(json.dumps({
                "type": "metadata",
                "schema": HMCP_CAPTURE_SCHEMA,
                "pose_scale": self.pose_scale,
                "hmcp_bind": bind_host,
                "hmcp_port": int(port),
            }, separators=(",", ":")) + "\n")

        print(
            f"[publish_v51] HMCP listen udp://{bind_host}:{port} "
            f"pose_scale={self.pose_scale:.3f}",
            flush=True,
        )

    def poll(self) -> Tuple[np.ndarray, np.ndarray, float] | None:
        try:
            raw, _ = self.sock.recvfrom(4096)
        except socket.timeout:
            return None
        received_monotonic_ns = time.monotonic_ns()
        self.received += 1
        packet = parse_hmcp(raw)
        if packet is None:
            self.invalid += 1
            return None
        if not is_newer_sequence(packet.sequence, self.last_sequence):
            self.duplicate_or_old += 1
            return None

        self.last_sequence = packet.sequence
        self.accepted += 1
        self.last_valid_s = time.monotonic()
        if packet.send_timestamp_s is None:
            velocity_timestamp_s = received_monotonic_ns * 1e-9
            velocity_timestamp_source = "receive-monotonic"
            self.receive_timestamp_fallbacks += 1
        else:
            velocity_timestamp_s = packet.send_timestamp_s
            velocity_timestamp_source = "hmcp-send"
            self.hmcp_send_timestamps += 1
        if self.capture is not None:
            self.capture.write(json.dumps({
                "schema": HMCP_CAPTURE_SCHEMA,
                "sequence": packet.sequence,
                "flags": packet.flags,
                "send_timestamp_s": packet.send_timestamp_s,
                "received_monotonic_ns": received_monotonic_ns,
                "velocity_timestamp_s": velocity_timestamp_s,
                "velocity_timestamp_source": velocity_timestamp_source,
                "qpos36": packet.qpos36.tolist(),
            }, separators=(",", ":")) + "\n")
        joint_pos, quat_xyzw, _ = hmcp_to_x2_reference(
            packet.qpos36, self.pose_scale
        )
        return joint_pos, quat_xyzw, velocity_timestamp_s

    def status(self) -> str:
        age = float("inf") if self.last_valid_s == 0.0 else time.monotonic() - self.last_valid_s
        return (
            f"rx={self.received} accepted={self.accepted} invalid={self.invalid} "
            f"duplicate_or_old={self.duplicate_or_old} "
            f"timestamp=hmcp:{self.hmcp_send_timestamps}/fallback:{self.receive_timestamp_fallbacks} "
            f"age={age:.3f}s"
        )

    def close(self) -> None:
        self.sock.close()
        if self.capture is not None:
            self.capture.close()


# --------------------------------------------------------------------------
# 发送循环
# --------------------------------------------------------------------------


def run_publisher(args: argparse.Namespace) -> int:
    # pyzmq 只在真的要发时才 import,这样 verify_v51_bytes.py 可以在没装
    # pyzmq 的机器上直接 import 本模块做纯字节校验。
    try:
        import zmq  # noqa: PLC0415
    except ImportError:
        print("需要 pyzmq:pip install pyzmq(纯字节校验请用 verify_v51_bytes.py)",
              file=sys.stderr)
        return 2

    hmcp_source: HmcpUdpSource | None = None
    paced_source: Iterator[
        Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, float | None]
    ] | None = None
    if args.source == "synthetic":
        paced_source = synthetic_source(args.rate, args.amplitude, args.period)
    elif args.source == "jsonl":
        paced_source = jsonl_source(args.jsonl, args.pose_scale, include_timestamp=True)
    else:
        hmcp_source = HmcpUdpSource(
            args.hmcp_bind,
            args.hmcp_port,
            args.pose_scale,
            args.hmcp_poll_timeout,
            args.capture_jsonl,
        )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    endpoint = f"tcp://{args.host}:{args.port}"
    sock.bind(endpoint)
    # slow-joiner:PUB 会丢掉 SUB 完成 SUBSCRIBE 握手之前的消息
    # (x2_zmq_protocol.md:206-218)。必须等 >= 200 ms 再发第一帧。
    time.sleep(0.25)

    stream = V51ReferenceStream(
        diff_hz=args.diff_hz,
        velocity_mode=args.velocity_mode,
        timestamp_min_dt_s=args.velocity_min_dt_s,
        timestamp_reset_gap_s=args.velocity_reset_gap_s,
    )
    period = 1.0 / float(args.rate)
    t0 = time.monotonic()
    sent = 0
    last_status_s = t0
    source_rate = "event-driven" if hmcp_source is not None else f"{args.rate} Hz"
    print(f"[publish_v51] bind {endpoint} topic={args.topic!r} rate={source_rate} "
          f"source={args.source} drop_velocity={args.drop_velocity} "
          f"stop_after_s={args.stop_after_s} velocity_mode={args.velocity_mode}", flush=True)

    try:
        while True:
            now = time.monotonic() - t0
            if args.duration > 0 and now >= args.duration:
                break
            if args.stop_after_s is not None and now >= args.stop_after_s:
                # 故意断流:不发了但进程继续存活,便于观察 C++ 侧
                # --pose-ref-stale-s 0.5 触发 2 秒有界回抓拍静态保持。
                print(f"[publish_v51] {now:.2f}s: 按 --stop-after-s 停止发送,"
                      f"已发 {sent} 帧;进程保持存活,等 watchdog 触发",
                      flush=True)
                while args.duration <= 0 or (time.monotonic() - t0) < args.duration:
                    time.sleep(0.1)
                break

            if hmcp_source is not None:
                sample = hmcp_source.poll()
                if sample is None:
                    absolute_now = time.monotonic()
                    if absolute_now - last_status_s >= 2.0:
                        print(
                            f"[publish_v51] HMCP {hmcp_source.status()} sent={sent} "
                            f"{stream.velocity_status()}",
                            flush=True,
                        )
                        last_status_s = absolute_now
                    continue
                pos, quat, sample_timestamp_s = sample
            else:
                if paced_source is None:
                    raise RuntimeError("paced source was not initialized")
                try:
                    paced_sample = next(paced_source)
                except StopIteration:
                    print(f"[publish_v51] {args.source} source exhausted", flush=True)
                    break
                if len(paced_sample) == 3:
                    pos, quat, sample_timestamp_s = paced_sample
                else:
                    pos, quat = paced_sample
                    sample_timestamp_s = sent / float(args.rate)

            # push 先做差分(clamp 之前),再由 build_fields 做 clamp
            stream.push(pos, quat, timestamp_s=sample_timestamp_s)
            fields = stream.build_fields(sent, drop_velocity=args.drop_velocity)
            sock.send(pack_message(fields, topic=args.topic))
            sent += 1

            if hmcp_source is None:
                # synthetic/JSONL 定速;HMCP 则严格每个新 sequence 立即发一次,
                # 不用 50 Hz 重复旧帧,否则会污染 LiveMotion 速度语义。
                target = t0 + sent * period
                slack = target - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        print("[publish_v51] 收到 Ctrl-C", flush=True)
    finally:
        if hmcp_source is not None:
            hmcp_source.close()
        sock.close(linger=0)
        ctx.term()
    print(f"[publish_v51] 结束,共发 {sent} 帧;{stream.velocity_status()}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="X2 Sonic v5.1 严格参考速度 ZMQ publisher(参考实现)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="PUB bind 地址。机器人上跑衣服通路时用 127.0.0.1")
    p.add_argument("--port", type=int, default=5556,
                   help="端口。garment launcher 里是 5556")
    p.add_argument("--topic", default="pose",
                   help="topic 前缀(裸字节拼在消息最前面,不是 ZMQ 多帧)")
    p.add_argument("--rate", type=float, default=50.0, help="发送频率 Hz")
    p.add_argument("--duration", type=float, default=10.0,
                   help="总时长秒,<=0 表示一直跑")
    p.add_argument("--source", choices=("synthetic", "jsonl", "hmcp"), default="synthetic",
                   help="synthetic=可复现正弦;jsonl=已知格式回放;hmcp=实时衣服 UDP")
    p.add_argument("--jsonl", default=None,
                   help="--source jsonl 的 HMCP capture 或 shadow-v1 文件")
    p.add_argument("--pose-scale", type=float, default=POSE_SCALE,
                   help="围绕 X2 训练默认角缩放 G1 增量;HumanPlus 基线固定 0.7")
    p.add_argument("--hmcp-bind", default="127.0.0.1",
                   help="--source hmcp 的 UDP bind 地址")
    p.add_argument("--hmcp-port", type=int, default=51234,
                   help="--source hmcp 的 UDP 端口")
    p.add_argument("--hmcp-poll-timeout", type=float, default=0.1,
                   help="HMCP socket poll timeout;超时不发送任何替代帧")
    p.add_argument("--capture-jsonl", default=None,
                   help="可选:以 x2-hmcp-capture-v1 保存每个有效新 HMCP 帧;拒绝覆盖")
    p.add_argument("--amplitude", type=float, default=0.05,
                   help="合成正弦幅值,弧度")
    p.add_argument("--period", type=float, default=10.0, help="合成正弦周期,秒")
    p.add_argument("--diff-hz", type=float, default=VELOCITY_DIFF_HZ,
                   help="fixed 模式的速度差分频率;50.0 保持 MuJoCo parity")
    p.add_argument("--velocity-mode", choices=VELOCITY_MODES, default="fixed",
                   help="fixed=MuJoCo 50Hz parity;timestamp=按实时样本时间差分")
    p.add_argument("--velocity-min-dt-s", type=float, default=TIMESTAMP_MIN_DT_S,
                   help="timestamp 模式最小有效间隔;更短则速度清零并重建基线")
    p.add_argument("--velocity-reset-gap-s", type=float, default=TIMESTAMP_RESET_GAP_S,
                   help="timestamp 模式最大有效间隔;更长则速度清零并重建基线")
    p.add_argument("--drop-velocity", action="store_true",
                   help="故意不发 joint_vel_mj,用于验证 strict 拒帧 R1")
    p.add_argument("--stop-after-s", type=float, default=None,
                   help="故意中途停止发送(进程不退出),用于验证断流 watchdog")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.source == "jsonl" and not args.jsonl:
        print("--source jsonl 需要 --jsonl PATH", file=sys.stderr)
        return 2
    if args.rate <= 0:
        print("--rate 必须 > 0", file=sys.stderr)
        return 2
    if not 0.0 <= args.pose_scale <= 1.0:
        print("--pose-scale 必须在 [0, 1]", file=sys.stderr)
        return 2
    if args.hmcp_poll_timeout <= 0:
        print("--hmcp-poll-timeout 必须 > 0", file=sys.stderr)
        return 2
    if args.velocity_min_dt_s <= 0:
        print("--velocity-min-dt-s 必须 > 0", file=sys.stderr)
        return 2
    if args.velocity_reset_gap_s <= args.velocity_min_dt_s:
        print("--velocity-reset-gap-s 必须 > --velocity-min-dt-s", file=sys.stderr)
        return 2
    if args.capture_jsonl and args.source != "hmcp":
        print("--capture-jsonl 只适用于 --source hmcp", file=sys.stderr)
        return 2
    if abs(args.diff_hz - VELOCITY_DIFF_HZ) > 1e-9:
        print(f"[publish_v51] 警告:--diff-hz={args.diff_hz} != {VELOCITY_DIFF_HZ},"
              "速度语义已偏离 MuJoCo LiveMotion parity", file=sys.stderr)
    return run_publisher(args)


if __name__ == "__main__":
    raise SystemExit(main())
