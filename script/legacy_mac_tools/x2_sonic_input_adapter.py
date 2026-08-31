#!/usr/bin/env python3
"""Pure, stateful input adapter for X2-Sonic garment/replay streams.

This module deliberately has no MuJoCo, ONNX, BLE, ROS, or robot dependency.
It maps one incoming human/garment frame to the conservative input domain
identified by the PHUMA closed-loop study and returns telemetry explaining what
was changed.  The same function can be used by an offline replay and by a
future real-time garment process after its I/O boundary has been reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


DEFAULT_ANGLES_MJ = np.asarray(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
     0.0, 0.0], dtype=np.float64)


def _normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("root quaternion is zero or non-finite")
    return q / norm


def _yaw_quat(q: np.ndarray) -> np.ndarray:
    q = _normalize(q)
    yaw = math.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = _normalize(a), _normalize(b)
    dot = float(np.dot(a, b))
    sign = -1.0 if dot < 0.0 else 1.0
    dot = abs(dot)
    if dot > 1.0 - 1e-8:
        return _normalize((1.0 - t) * a + sign * t * b)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    return _normalize(
        math.sin((1.0 - t) * theta) / math.sin(theta) * a
        + sign * math.sin(t * theta) / math.sin(theta) * b
    )


def _tilt_rad(q: np.ndarray) -> float:
    q = _normalize(q)
    # Body down vector expressed in body coordinates; upright is [0,0,-1].
    # Only its z component is needed for the angle to world down.
    gz = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return float(math.acos(max(-1.0, min(1.0, gz))))


def limit_joint_speed(target: np.ndarray, previous: np.ndarray,
                      fps: float, max_speed: float) -> np.ndarray:
    if fps <= 0.0 or max_speed <= 0.0:
        raise ValueError("fps and max_speed must be positive")
    step = float(max_speed) / float(fps)
    return previous + np.clip(target - previous, -step, step)


@dataclass(frozen=True)
class X2SonicAdapterConfig:
    """Conservative input-domain parameters identified offline."""

    root_tilt_scale: float = 0.0
    pose_scale: float = 0.5
    joint_speed_limit_radps: Optional[float] = None

    def validate(self) -> None:
        if not 0.0 <= self.root_tilt_scale <= 1.0:
            raise ValueError("root_tilt_scale must be in [0,1]")
        if not 0.0 < self.pose_scale <= 1.0:
            raise ValueError("pose_scale must be in (0,1]")
        if self.joint_speed_limit_radps is not None and self.joint_speed_limit_radps <= 0.0:
            raise ValueError("joint_speed_limit_radps must be positive")


class X2SonicInputAdapter:
    """Map one frame at a time and expose deterministic safety telemetry."""

    def __init__(self, config: X2SonicAdapterConfig | None = None):
        self.config = config or X2SonicAdapterConfig()
        self.config.validate()
        self._previous_output: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_output = None

    def process(self, joint_pos, root_quat, fps: float) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
        jp = np.asarray(joint_pos, dtype=np.float64)
        rq = _normalize(np.asarray(root_quat, dtype=np.float64))
        if jp.shape != (31,) or not np.all(np.isfinite(jp)):
            raise ValueError(f"joint_pos must be finite shape (31,), got {jp.shape}")
        if rq.shape != (4,) or not np.all(np.isfinite(rq)):
            raise ValueError(f"root_quat must be finite shape (4,), got {rq.shape}")

        yaw = _yaw_quat(rq)
        out_quat = _slerp(yaw, rq, self.config.root_tilt_scale)
        target_jp = DEFAULT_ANGLES_MJ + self.config.pose_scale * (jp - DEFAULT_ANGLES_MJ)
        limited = False
        if self.config.joint_speed_limit_radps is not None and self._previous_output is not None:
            out_jp = limit_joint_speed(target_jp, self._previous_output, fps, self.config.joint_speed_limit_radps)
            limited = bool(np.any(np.abs(out_jp - target_jp) > 1e-12))
        else:
            out_jp = target_jp.copy()
        self._previous_output = out_jp.copy()
        telemetry = {
            "raw_root_tilt_rad": _tilt_rad(rq),
            "adapted_root_tilt_rad": _tilt_rad(out_quat),
            "pose_scale": float(self.config.pose_scale),
            "root_tilt_scale": float(self.config.root_tilt_scale),
            "joint_max_abs_raw_rad": float(np.max(np.abs(jp))),
            "joint_max_abs_adapted_rad": float(np.max(np.abs(out_jp))),
            "joint_rate_limited": limited,
            "finite": bool(np.all(np.isfinite(out_jp)) and np.all(np.isfinite(out_quat))),
        }
        return out_jp, out_quat, telemetry


__all__ = [
    "DEFAULT_ANGLES_MJ",
    "X2SonicAdapterConfig",
    "X2SonicInputAdapter",
    "limit_joint_speed",
]
