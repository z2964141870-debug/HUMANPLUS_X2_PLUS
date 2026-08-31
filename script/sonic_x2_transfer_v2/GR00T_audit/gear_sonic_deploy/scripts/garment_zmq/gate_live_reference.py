#!/usr/bin/env python3
"""Fail-closed gate between the live garment reference and X2 Sonic.

Wire topology::

    raw timestamp adapter PUB :5555
        -> this process
        -> guarded pose PUB :5556

    X2 deploy x2_debug PUB :5557
        -> this process

The process never talks to ROS, MC, or HAL. It publishes no pose frame until
both streams are fresh and the wearer has remained neutral and still for a
continuous interval. Any source/debug fault after warmup is a terminal
LOCKOUT for this process; restarting the process is the only recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import publish_v51_reference as v51


_DTYPE_INFO = {
    "f32": (np.dtype("<f4"), 4),
    "f64": (np.dtype("<f8"), 8),
    "i32": (np.dtype("<i4"), 4),
    "i64": (np.dtype("<i8"), 8),
    "u8": (np.dtype("u1"), 1),
    "bool": (np.dtype("u1"), 1),
}

_GROUP_INDICES = {
    "leg": np.arange(0, 12),
    "waist": np.arange(12, 15),
    "arm": np.arange(15, 29),
    "head": np.arange(29, 31),
}


class WireError(ValueError):
    """A packed frame violates the proxy's strict wire contract."""


class GateState(str, Enum):
    WAIT_SOURCE = "WAIT_SOURCE"
    WAIT_ROBOT = "WAIT_ROBOT"
    WARMUP = "WARMUP"
    STANDSTILL_READY = "STANDSTILL_READY"
    BLEND = "BLEND"
    LIVE = "LIVE"
    LOCKOUT = "LOCKOUT"


@dataclass(frozen=True)
class GateConfig:
    source_stale_s: float = 0.15
    robot_stale_s: float = 0.10
    still_seconds: float = 2.0
    still_frames: int = 50
    blend_seconds: float = 10.0
    require_powered_debug: bool = True
    stationary_only: bool = True
    root_roll_abs_max_rad: float = math.radians(12.0)
    root_pitch_abs_max_rad: float = math.radians(12.0)
    root_angular_speed_max_rad_s: float = 0.35
    leg_offset_max_rad: float = 0.60
    waist_offset_max_rad: float = 0.20
    arm_offset_max_rad: float = 0.65
    head_offset_max_rad: float = 0.08
    leg_velocity_max_rad_s: float = 0.20
    waist_velocity_max_rad_s: float = 0.20
    arm_velocity_max_rad_s: float = 0.25
    head_velocity_max_rad_s: float = 0.15

    def validate(self) -> None:
        positive = {
            "source_stale_s": self.source_stale_s,
            "robot_stale_s": self.robot_stale_s,
            "still_seconds": self.still_seconds,
            "blend_seconds": self.blend_seconds,
            "root_angular_speed_max_rad_s": self.root_angular_speed_max_rad_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.still_frames < 2:
            raise ValueError("still_frames must be >= 2")


@dataclass(frozen=True)
class PoseFrame:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat_xyzw: np.ndarray
    frame_index: int
    recv_s: float


@dataclass(frozen=True)
class RobotFrame:
    base_quat_wxyz: np.ndarray
    control_tick: int
    ros_timestamp_s: float
    dry_run: bool
    recv_s: float


@dataclass(frozen=True)
class OutputReference:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat_xyzw: np.ndarray
    frame_index: int
    alpha: float


@dataclass(frozen=True)
class SourceGateResult:
    accepted: bool
    reason: str
    root_rpy_rad: tuple[float, float, float]
    root_angular_speed_rad_s: float
    offset_max_by_group: Mapping[str, float]
    velocity_max_by_group: Mapping[str, float]


def _shape_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise WireError("field shape must be a non-empty list")
    shape = tuple(int(item) for item in value)
    if any(item <= 0 for item in shape):
        raise WireError(f"invalid field shape {shape}")
    if math.prod(shape) > 1_000_000:
        raise WireError(f"field shape is unreasonably large: {shape}")
    return shape


def decode_packed(raw: bytes, topic: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Decode one packed-binary frame more strictly than the C++ receiver."""
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        raise WireError(f"topic prefix is not {topic!r}")
    body = raw[len(topic_bytes):]
    if len(body) < v51.HEADER_SIZE:
        raise WireError("short packed header")
    header_block = body[:v51.HEADER_SIZE]
    nul = header_block.find(b"\x00")
    if nul <= 0:
        raise WireError("packed header is not NUL terminated")
    try:
        header = json.loads(header_block[:nul].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireError(f"invalid packed header JSON: {exc}") from exc
    if header.get("endian") != "le":
        raise WireError("only little-endian packed frames are accepted")
    fields = header.get("fields")
    if not isinstance(fields, list):
        raise WireError("packed header has no fields list")

    payload = body[v51.HEADER_SIZE:]
    decoded: dict[str, np.ndarray] = {}
    offset = 0
    for descriptor in fields:
        if not isinstance(descriptor, dict):
            raise WireError("field descriptor is not an object")
        name = descriptor.get("name")
        dtype_name = descriptor.get("dtype")
        if not isinstance(name, str) or not name:
            raise WireError("field has no name")
        if name in decoded:
            raise WireError(f"duplicate field {name!r}")
        if name == "estop":
            raise WireError("estop field is forbidden")
        if dtype_name not in _DTYPE_INFO:
            raise WireError(f"field {name!r} has unsupported dtype {dtype_name!r}")
        shape = _shape_tuple(descriptor.get("shape"))
        dtype, width = _DTYPE_INFO[dtype_name]
        byte_count = math.prod(shape) * width
        if offset + byte_count > len(payload):
            raise WireError(f"field {name!r} extends beyond payload")
        decoded[name] = np.frombuffer(
            payload[offset:offset + byte_count], dtype=dtype
        ).reshape(shape).copy()
        offset += byte_count
    if offset != len(payload):
        raise WireError(f"unexpected trailing payload bytes: {len(payload) - offset}")
    return header, decoded


def _require_field(
    fields: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = fields.get(name)
    if value is None:
        raise WireError(f"missing required field {name!r}")
    if value.dtype != dtype or value.shape != shape:
        raise WireError(
            f"field {name!r} must be {dtype} {shape}, got {value.dtype} {value.shape}"
        )
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise WireError(f"field {name!r} contains NaN/Inf")
    return value


def _normalize_quat(quat: np.ndarray, order: str) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).reshape(4)
    if not np.isfinite(value).all():
        raise WireError(f"{order} quaternion contains NaN/Inf")
    norm = float(np.linalg.norm(value))
    if norm < 0.5 or norm > 2.0:
        raise WireError(f"{order} quaternion norm {norm:.6f} is outside [0.5, 2.0]")
    return value / norm


def decode_pose_frame(raw: bytes, topic: str, recv_s: float) -> PoseFrame:
    header, fields = decode_packed(raw, topic)
    if int(header.get("v", -1)) < 5:
        raise WireError("pose protocol version must be >= 5")
    pos = _require_field(fields, "joint_pos_mj", np.dtype("<f4"), (v51.NUM_DOFS,))
    vel = _require_field(fields, "joint_vel_mj", np.dtype("<f4"), (v51.NUM_DOFS,))
    quat = _require_field(fields, "root_quat_xyzw", np.dtype("<f4"), (4,))
    future_pos = _require_field(
        fields,
        "joint_pos_mj_future",
        np.dtype("<f4"),
        (v51.NUM_FUTURE_SLOTS, v51.NUM_DOFS),
    )
    future_vel = _require_field(
        fields,
        "joint_vel_mj_future",
        np.dtype("<f4"),
        (v51.NUM_FUTURE_SLOTS, v51.NUM_DOFS),
    )
    future_quat = _require_field(
        fields,
        "root_quat_xyzw_future",
        np.dtype("<f4"),
        (v51.NUM_FUTURE_SLOTS, 4),
    )
    future_dt = _require_field(fields, "future_dt_s", np.dtype("<f4"), (1,))
    frame_index = _require_field(fields, "frame_index", np.dtype("<i8"), (1,))

    quat_norm = _normalize_quat(quat, "xyzw")
    if not np.allclose(future_pos, pos[None, :], atol=2e-5, rtol=0.0):
        raise WireError("source future positions are not the proven live-edge clamp")
    if not np.allclose(future_vel, vel[None, :], atol=2e-5, rtol=0.0):
        raise WireError("source future velocities are not the proven live-edge clamp")
    for index, item in enumerate(future_quat):
        normed = _normalize_quat(item, "xyzw")
        if abs(float(np.dot(normed, quat_norm))) < 1.0 - 2e-5:
            raise WireError(f"source future quaternion slot {index} differs from current")
    if abs(float(future_dt[0]) - v51.FUTURE_DT_S) > 1e-6:
        raise WireError(f"future_dt_s must be {v51.FUTURE_DT_S}")
    return PoseFrame(
        joint_pos=pos.astype(np.float64),
        joint_vel=vel.astype(np.float64),
        root_quat_xyzw=quat_norm,
        frame_index=int(frame_index[0]),
        recv_s=float(recv_s),
    )


def decode_robot_frame(raw: bytes, topic: str, recv_s: float) -> RobotFrame:
    header, fields = decode_packed(raw, topic)
    if int(header.get("v", -1)) < 5:
        raise WireError("x2_debug protocol version must be >= 5")
    tick = _require_field(fields, "control_tick", np.dtype("<i8"), (1,))
    ros_timestamp = _require_field(fields, "ros_timestamp", np.dtype("<f8"), (1,))
    quat = _require_field(fields, "base_quat", np.dtype("<f8"), (4,))
    _require_field(fields, "body_q", np.dtype("<f8"), (v51.NUM_DOFS,))
    _require_field(fields, "body_dq", np.dtype("<f8"), (v51.NUM_DOFS,))
    dry_run = _require_field(fields, "dry_run", np.dtype("u1"), (1,))
    if int(dry_run[0]) not in (0, 1):
        raise WireError("x2_debug dry_run must be 0 or 1")
    return RobotFrame(
        base_quat_wxyz=_normalize_quat(quat, "wxyz"),
        control_tick=int(tick[0]),
        ros_timestamp_s=float(ros_timestamp[0]),
        dry_run=bool(dry_run[0]),
        recv_s=float(recv_s),
    )


def yaw_from_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = _normalize_quat(quat, "xyzw")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = _normalize_quat(quat, "wxyz")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_from_xyzw(quat: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = _normalize_quat(quat, "xyzw")
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = _normalize_quat(left, "xyzw")
    rx, ry, rz, rw = _normalize_quat(right, "xyzw")
    result = np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])
    return result / np.linalg.norm(result)


def quat_nlerp_xyzw(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    first = _normalize_quat(start, "xyzw")
    second = _normalize_quat(end, "xyzw")
    if float(np.dot(first, second)) < 0.0:
        second = -second
    result = (1.0 - alpha) * first + alpha * second
    norm = float(np.linalg.norm(result))
    if norm < 1e-9:
        raise WireError("quaternion blend became degenerate")
    return result / norm


def _quat_angle_xyzw(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(
        _normalize_quat(first, "xyzw"), _normalize_quat(second, "xyzw")
    )))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _smoothstep(value: float) -> tuple[float, float]:
    u = max(0.0, min(1.0, value))
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)


def build_output_message(output: OutputReference, topic: str) -> bytes:
    pos = np.asarray(output.joint_pos, dtype=np.float64).reshape(v51.NUM_DOFS)
    vel = np.asarray(output.joint_vel, dtype=np.float64).reshape(v51.NUM_DOFS)
    quat = _normalize_quat(output.root_quat_xyzw, "xyzw")
    for name, value in (("joint_pos", pos), ("joint_vel", vel), ("root_quat", quat)):
        if not np.isfinite(value).all():
            raise WireError(f"output {name} contains NaN/Inf")
    fields = [
        ("joint_pos_mj", pos.astype(np.float32)),
        ("joint_vel_mj", vel.astype(np.float32)),
        ("root_quat_xyzw", quat.astype(np.float32)),
        ("joint_pos_mj_future", np.tile(pos, (v51.NUM_FUTURE_SLOTS, 1)).astype(np.float32)),
        ("joint_vel_mj_future", np.tile(vel, (v51.NUM_FUTURE_SLOTS, 1)).astype(np.float32)),
        ("root_quat_xyzw_future", np.tile(quat, (v51.NUM_FUTURE_SLOTS, 1)).astype(np.float32)),
        ("future_dt_s", np.array([v51.FUTURE_DT_S], dtype=np.float32)),
        ("frame_index", np.array([output.frame_index], dtype=np.int64)),
    ]
    return v51.pack_message(fields, topic=topic, version=5)


class LiveReferenceGate:
    """Pure state machine. Network and files are handled by ``run_proxy``."""

    def __init__(self, config: GateConfig) -> None:
        config.validate()
        self.config = config
        self.state = GateState.WAIT_SOURCE
        self.reason = "waiting for raw live reference"
        self.source: PoseFrame | None = None
        self.robot: RobotFrame | None = None
        self._previous_source: PoseFrame | None = None
        self._both_seen = False
        self._still_since_s: float | None = None
        self._still_frames = 0
        self._standstill_yaw_rad: float | None = None
        self._yaw_delta_rad: float | None = None
        self._blend_start_s: float | None = None
        self._output_index = 0
        self._last_gate: SourceGateResult | None = None
        self._lockout_s: float | None = None

    def _transition(self, state: GateState, reason: str) -> None:
        self.state = state
        self.reason = reason

    def lockout(self, reason: str, now_s: float) -> None:
        if self.state == GateState.LOCKOUT:
            return
        self._lockout_s = float(now_s)
        self._transition(GateState.LOCKOUT, reason)

    def on_robot(self, frame: RobotFrame, now_s: float | None = None) -> None:
        now = frame.recv_s if now_s is None else float(now_s)
        if self.state == GateState.LOCKOUT:
            return
        if self.robot is not None and frame.ros_timestamp_s <= self.robot.ros_timestamp_s:
            self.lockout(
                f"x2_debug ros_timestamp did not increase: {frame.ros_timestamp_s:.9f} <= "
                f"{self.robot.ros_timestamp_s:.9f}",
                now,
            )
            return
        self.robot = frame
        self.tick(now)

    def on_source(self, frame: PoseFrame, now_s: float | None = None) -> OutputReference | None:
        now = frame.recv_s if now_s is None else float(now_s)
        if self.state == GateState.LOCKOUT:
            return None
        if self.source is not None and frame.frame_index <= self.source.frame_index:
            self.lockout(
                f"pose frame_index did not increase: {frame.frame_index} <= "
                f"{self.source.frame_index}",
                now,
            )
            return None
        self._previous_source = self.source
        self.source = frame
        self.tick(now)
        if self.state == GateState.LOCKOUT:
            return None

        self._last_gate = self._evaluate_source(frame)
        if self.state == GateState.WARMUP:
            if self._last_gate.accepted and self._robot_gate_ok():
                if self._still_since_s is None:
                    self._still_since_s = now
                    self._still_frames = 1
                else:
                    self._still_frames += 1
                duration = now - self._still_since_s
                if duration >= self.config.still_seconds and self._still_frames >= self.config.still_frames:
                    assert self.robot is not None
                    self._standstill_yaw_rad = yaw_from_wxyz(self.robot.base_quat_wxyz)
                    self._transition(
                        GateState.STANDSTILL_READY,
                        "continuous neutral stillness passed; publishing anchored StandStill only",
                    )
            else:
                self._still_since_s = None
                self._still_frames = 0
                self.reason = self._last_gate.reason if not self._last_gate.accepted else self._robot_gate_reason()
        elif self.state in (GateState.STANDSTILL_READY, GateState.BLEND, GateState.LIVE):
            if not self._last_gate.accepted:
                self.lockout(f"stationary reference gate opened: {self._last_gate.reason}", now)
                return None
            if not self._robot_gate_ok():
                self.lockout(f"robot debug gate opened: {self._robot_gate_reason()}", now)
                return None

        if self.state in (GateState.STANDSTILL_READY, GateState.BLEND, GateState.LIVE):
            return self.reference_at(now)
        return None

    def tick(self, now_s: float) -> None:
        now = float(now_s)
        if self.state == GateState.LOCKOUT:
            return
        self._update_wait_state(now)
        if not self._both_seen:
            return
        assert self.source is not None and self.robot is not None
        source_age = now - self.source.recv_s
        robot_age = now - self.robot.recv_s
        if source_age > self.config.source_stale_s:
            self.lockout(
                f"raw pose source stale: {source_age:.3f}s > {self.config.source_stale_s:.3f}s",
                now,
            )
        elif robot_age > self.config.robot_stale_s:
            self.lockout(
                f"x2_debug stale: {robot_age:.3f}s > {self.config.robot_stale_s:.3f}s",
                now,
            )

    def arm(self, now_s: float) -> tuple[bool, str]:
        now = float(now_s)
        self.tick(now)
        if self.state != GateState.STANDSTILL_READY:
            return False, f"arm rejected in state {self.state.value}: {self.reason}"
        assert self.source is not None and self.robot is not None
        if self._last_gate is None or not self._last_gate.accepted:
            return False, "arm rejected: source gate is not currently closed"
        robot_yaw = yaw_from_wxyz(self.robot.base_quat_wxyz)
        human_yaw = yaw_from_xyzw(self.source.root_quat_xyzw)
        self._yaw_delta_rad = math.atan2(
            math.sin(robot_yaw - human_yaw), math.cos(robot_yaw - human_yaw)
        )
        self._blend_start_s = now
        self._transition(
            GateState.BLEND,
            f"explicit arm accepted; fixed yaw delta={math.degrees(self._yaw_delta_rad):+.2f}deg",
        )
        return True, self.reason

    def reference_at(self, now_s: float) -> OutputReference:
        if self.state not in (GateState.STANDSTILL_READY, GateState.BLEND, GateState.LIVE):
            raise RuntimeError(f"no output is allowed in state {self.state.value}")
        assert self.source is not None
        assert self._standstill_yaw_rad is not None
        standstill_quat = yaw_quat_xyzw(self._standstill_yaw_rad)
        alpha = 0.0
        alpha_rate = 0.0
        if self.state in (GateState.BLEND, GateState.LIVE):
            assert self._yaw_delta_rad is not None and self._blend_start_s is not None
            elapsed_s = max(0.0, float(now_s) - self._blend_start_s)
            blend_complete = elapsed_s >= self.config.blend_seconds - 1e-12
            u = 1.0 if blend_complete else elapsed_s / self.config.blend_seconds
            alpha, smooth_derivative = _smoothstep(u)
            alpha_rate = smooth_derivative / self.config.blend_seconds
            transformed_quat = quat_mul_xyzw(
                yaw_quat_xyzw(self._yaw_delta_rad), self.source.root_quat_xyzw
            )
            root_quat = quat_nlerp_xyzw(standstill_quat, transformed_quat, alpha)
            joint_delta = self.source.joint_pos - v51.DEFAULT_ANGLES
            joint_pos = v51.DEFAULT_ANGLES + alpha * joint_delta
            joint_vel = alpha * self.source.joint_vel + alpha_rate * joint_delta
            if blend_complete and self.state == GateState.BLEND:
                self._transition(GateState.LIVE, "10-second StandStill-to-live blend completed")
        else:
            root_quat = standstill_quat
            joint_pos = v51.DEFAULT_ANGLES.copy()
            joint_vel = np.zeros(v51.NUM_DOFS, dtype=np.float64)
        output = OutputReference(
            joint_pos=np.asarray(joint_pos, dtype=np.float64),
            joint_vel=np.asarray(joint_vel, dtype=np.float64),
            root_quat_xyzw=np.asarray(root_quat, dtype=np.float64),
            frame_index=self._output_index,
            alpha=float(alpha),
        )
        self._output_index += 1
        return output

    def _update_wait_state(self, now_s: float) -> None:
        if self.state == GateState.LOCKOUT:
            return
        if self.source is None:
            self._transition(GateState.WAIT_SOURCE, "waiting for raw live reference")
            return
        if self.robot is None:
            self._transition(GateState.WAIT_ROBOT, "waiting for x2_debug robot quaternion")
            return
        if not self._both_seen:
            source_age = now_s - self.source.recv_s
            robot_age = now_s - self.robot.recv_s
            if source_age <= self.config.source_stale_s and robot_age <= self.config.robot_stale_s:
                self._both_seen = True
                self._transition(GateState.WARMUP, "both streams seen; waiting for neutral stillness")

    def _robot_gate_ok(self) -> bool:
        if self.robot is None:
            return False
        if self.config.require_powered_debug and self.robot.dry_run:
            return False
        return True

    def _robot_gate_reason(self) -> str:
        if self.robot is None:
            return "x2_debug has not arrived"
        if self.config.require_powered_debug and self.robot.dry_run:
            return "x2_debug reports dry_run=1; powered debug is required"
        return "robot debug gate closed"

    def _evaluate_source(self, frame: PoseFrame) -> SourceGateResult:
        offsets = np.abs(frame.joint_pos - v51.DEFAULT_ANGLES)
        velocities = np.abs(frame.joint_vel)
        offset_limits = {
            "leg": self.config.leg_offset_max_rad,
            "waist": self.config.waist_offset_max_rad,
            "arm": self.config.arm_offset_max_rad,
            "head": self.config.head_offset_max_rad,
        }
        velocity_limits = {
            "leg": self.config.leg_velocity_max_rad_s,
            "waist": self.config.waist_velocity_max_rad_s,
            "arm": self.config.arm_velocity_max_rad_s,
            "head": self.config.head_velocity_max_rad_s,
        }
        offset_max = {
            name: float(np.max(offsets[indices])) for name, indices in _GROUP_INDICES.items()
        }
        velocity_max = {
            name: float(np.max(velocities[indices])) for name, indices in _GROUP_INDICES.items()
        }
        roll, pitch, yaw = rpy_from_xyzw(frame.root_quat_xyzw)
        angular_speed = 0.0
        if self._previous_source is not None:
            dt = frame.recv_s - self._previous_source.recv_s
            if dt <= 0.0:
                angular_speed = float("inf")
            else:
                angular_speed = _quat_angle_xyzw(
                    self._previous_source.root_quat_xyzw, frame.root_quat_xyzw
                ) / dt

        reason = "neutral and still"
        accepted = True
        for name in ("leg", "waist", "arm", "head"):
            if offset_max[name] > offset_limits[name]:
                accepted = False
                reason = (
                    f"{name} offset {offset_max[name]:.3f}rad exceeds "
                    f"{offset_limits[name]:.3f}rad"
                )
                break
        if accepted:
            for name in ("leg", "waist", "arm", "head"):
                if velocity_max[name] > velocity_limits[name]:
                    accepted = False
                    reason = (
                        f"{name} velocity {velocity_max[name]:.3f}rad/s exceeds "
                        f"{velocity_limits[name]:.3f}rad/s"
                    )
                    break
        if accepted and abs(roll) > self.config.root_roll_abs_max_rad:
            accepted = False
            reason = f"root roll {math.degrees(roll):+.2f}deg is outside neutral envelope"
        if accepted and abs(pitch) > self.config.root_pitch_abs_max_rad:
            accepted = False
            reason = f"root pitch {math.degrees(pitch):+.2f}deg is outside neutral envelope"
        if accepted and angular_speed > self.config.root_angular_speed_max_rad_s:
            accepted = False
            reason = (
                f"root angular speed {angular_speed:.3f}rad/s exceeds "
                f"{self.config.root_angular_speed_max_rad_s:.3f}rad/s"
            )
        return SourceGateResult(
            accepted=accepted,
            reason=reason,
            root_rpy_rad=(roll, pitch, yaw),
            root_angular_speed_rad_s=float(angular_speed),
            offset_max_by_group=offset_max,
            velocity_max_by_group=velocity_max,
        )

    def status(self, now_s: float, session_id: str, arm_trigger_file: str) -> dict:
        now = float(now_s)
        source_age = None if self.source is None else max(0.0, now - self.source.recv_s)
        robot_age = None if self.robot is None else max(0.0, now - self.robot.recv_s)
        gate = None
        if self._last_gate is not None:
            gate = {
                **asdict(self._last_gate),
                "root_rpy_deg": [math.degrees(value) for value in self._last_gate.root_rpy_rad],
            }
            gate.pop("root_rpy_rad", None)
        still_duration = 0.0 if self._still_since_s is None else max(0.0, now - self._still_since_s)
        return {
            "schema": "x2-live-reference-gate-status-v1",
            "session_id": session_id,
            "pid": os.getpid(),
            "updated_monotonic_s": now,
            "updated_wall_s": time.time(),
            "state": self.state.value,
            "reason": self.reason,
            "output_active": self.state in (
                GateState.STANDSTILL_READY, GateState.BLEND, GateState.LIVE
            ),
            "stationary_only": self.config.stationary_only,
            "source_age_s": source_age,
            "robot_age_s": robot_age,
            "source_frame_index": None if self.source is None else self.source.frame_index,
            "robot_control_tick": None if self.robot is None else self.robot.control_tick,
            "robot_ros_timestamp_s": None if self.robot is None else self.robot.ros_timestamp_s,
            "debug_dry_run": None if self.robot is None else self.robot.dry_run,
            "still_duration_s": still_duration,
            "still_frames": self._still_frames,
            "blend_alpha": self._current_alpha(now),
            "yaw_delta_deg": None if self._yaw_delta_rad is None else math.degrees(self._yaw_delta_rad),
            "gate": gate,
            "arm_trigger_file": arm_trigger_file,
            "config": asdict(self.config),
        }

    def _current_alpha(self, now_s: float) -> float:
        if self.state == GateState.LIVE:
            return 1.0
        if self.state != GateState.BLEND or self._blend_start_s is None:
            return 0.0
        alpha, _ = _smoothstep((now_s - self._blend_start_s) / self.config.blend_seconds)
        return alpha


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_arm_trigger(path: Path, session_id: str) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid arm trigger: {exc}"
    if payload.get("schema") != "x2-live-reference-arm-v1":
        return False, "invalid arm trigger schema"
    if payload.get("command") != "ARM_LIVE_REFERENCE":
        return False, "invalid arm trigger command"
    if payload.get("session_id") != session_id:
        return False, "arm trigger session_id does not match this proxy"
    return True, "arm trigger accepted"


def run_proxy(args: argparse.Namespace) -> int:
    try:
        import zmq  # noqa: PLC0415
    except ImportError:
        print("gate_live_reference.py requires pyzmq", file=sys.stderr)
        return 2

    config = GateConfig(
        source_stale_s=args.source_stale_s,
        robot_stale_s=args.robot_stale_s,
        still_seconds=args.still_seconds,
        still_frames=args.still_frames,
        blend_seconds=args.blend_seconds,
        require_powered_debug=not args.allow_dry_run_debug,
        stationary_only=True,
    )
    gate = LiveReferenceGate(config)
    session_id = args.session_id or uuid.uuid4().hex
    status_path = Path(args.status_file).expanduser()
    arm_path = (
        Path(args.arm_trigger_file).expanduser()
        if args.arm_trigger_file
        else Path(f"/tmp/x2_live_reference_gate.{os.getpid()}.arm.json")
    )
    if arm_path.exists():
        print(f"[live-gate] refusing stale arm trigger: {arm_path}", file=sys.stderr)
        return 3

    context = zmq.Context()
    source_socket = context.socket(zmq.SUB)
    source_socket.setsockopt(zmq.RCVHWM, 2)
    source_socket.setsockopt(zmq.CONFLATE, 1)
    source_socket.setsockopt(zmq.SUBSCRIBE, args.source_topic.encode("utf-8"))
    source_socket.connect(f"tcp://{args.source_host}:{args.source_port}")

    robot_socket = context.socket(zmq.SUB)
    robot_socket.setsockopt(zmq.RCVHWM, 2)
    robot_socket.setsockopt(zmq.CONFLATE, 1)
    robot_socket.setsockopt(zmq.SUBSCRIBE, args.robot_topic.encode("utf-8"))
    robot_socket.connect(f"tcp://{args.robot_host}:{args.robot_port}")

    output_socket = context.socket(zmq.PUB)
    output_socket.setsockopt(zmq.SNDHWM, 2)
    output_socket.setsockopt(zmq.LINGER, 0)
    output_socket.bind(f"tcp://{args.output_host}:{args.output_port}")

    poller = zmq.Poller()
    poller.register(source_socket, zmq.POLLIN)
    poller.register(robot_socket, zmq.POLLIN)
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    last_state = gate.state
    last_status_s = 0.0
    trigger_consumed = False
    print(
        f"[live-gate] session={session_id} source=tcp://{args.source_host}:{args.source_port} "
        f"robot=tcp://{args.robot_host}:{args.robot_port} "
        f"output=tcp://{args.output_host}:{args.output_port}",
        flush=True,
    )
    print(f"[live-gate] status={status_path} arm_trigger={arm_path}", flush=True)
    print("[live-gate] publishes nothing until STANDSTILL_READY", flush=True)

    try:
        while not stopping:
            events = dict(poller.poll(10))
            now = time.monotonic()
            if robot_socket in events:
                try:
                    raw = robot_socket.recv()
                    gate.on_robot(decode_robot_frame(raw, args.robot_topic, now), now)
                except (WireError, ValueError) as exc:
                    gate.lockout(f"invalid x2_debug frame: {exc}", now)
            if source_socket in events:
                try:
                    raw = source_socket.recv()
                    output = gate.on_source(decode_pose_frame(raw, args.source_topic, now), now)
                    if output is not None and gate.state != GateState.LOCKOUT:
                        output_socket.send(
                            build_output_message(output, args.output_topic),
                            flags=zmq.DONTWAIT,
                        )
                except zmq.Again:
                    pass
                except (WireError, ValueError) as exc:
                    gate.lockout(f"invalid raw pose frame: {exc}", now)
            gate.tick(now)

            if not trigger_consumed and arm_path.exists():
                trigger_consumed = True
                valid, trigger_reason = _read_arm_trigger(arm_path, session_id)
                if not valid:
                    gate.lockout(trigger_reason, now)
                else:
                    accepted, arm_reason = gate.arm(now)
                    print(f"[live-gate] {arm_reason}", flush=True)
                    if not accepted:
                        gate.lockout(arm_reason, now)

            if sys.stdin.isatty():
                readable, _, _ = select.select([sys.stdin], [], [], 0.0)
                if readable:
                    command = sys.stdin.readline().strip()
                    if command == "ARM_LIVE_REFERENCE":
                        accepted, arm_reason = gate.arm(now)
                        print(f"[live-gate] {arm_reason}", flush=True)
                        if not accepted:
                            print("[live-gate] arm did not change state", flush=True)
                    elif command in ("QUIT", "quit", "exit"):
                        stopping = True
                    elif command:
                        print("[live-gate] commands: ARM_LIVE_REFERENCE | QUIT", flush=True)

            if gate.state != last_state:
                print(f"[live-gate] {last_state.value} -> {gate.state.value}: {gate.reason}", flush=True)
                last_state = gate.state
            if now - last_status_s >= 0.04:
                _atomic_write_json(status_path, gate.status(now, session_id, str(arm_path)))
                last_status_s = now
    finally:
        now = time.monotonic()
        if gate.state != GateState.LOCKOUT:
            gate.lockout("proxy process stopping; output ceased", now)
        _atomic_write_json(status_path, gate.status(now, session_id, str(arm_path)))
        source_socket.close(linger=0)
        robot_socket.close(linger=0)
        output_socket.close(linger=0)
        context.term()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed live-reference gate for supported X2 Sonic integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--source-port", type=int, default=5555)
    parser.add_argument("--source-topic", default="pose")
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=5557)
    parser.add_argument("--robot-topic", default="x2_debug")
    parser.add_argument("--output-host", default="127.0.0.1")
    parser.add_argument("--output-port", type=int, default=5556)
    parser.add_argument("--output-topic", default="pose")
    parser.add_argument("--source-stale-s", type=float, default=0.15)
    parser.add_argument("--robot-stale-s", type=float, default=0.10)
    parser.add_argument("--still-seconds", type=float, default=2.0)
    parser.add_argument("--still-frames", type=int, default=50)
    parser.add_argument("--blend-seconds", type=float, default=10.0)
    parser.add_argument(
        "--allow-dry-run-debug",
        action="store_true",
        help="offline tests only; powered candidate leaves this disabled",
    )
    parser.add_argument("--status-file", default="/tmp/x2_live_reference_gate.status.json")
    parser.add_argument("--arm-trigger-file", default=None)
    parser.add_argument("--session-id", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("source_port", "robot_port", "output_port"):
        value = getattr(args, name)
        if not 1 <= value <= 65535:
            print(f"--{name.replace('_', '-')} must be in [1, 65535]", file=sys.stderr)
            return 2
    try:
        return run_proxy(args)
    except (OSError, ValueError) as exc:
        print(f"[live-gate] startup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
