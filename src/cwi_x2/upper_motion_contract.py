"""Pure helpers for replaying an X2 upper-body reference as a relative target.

The lower policy owns both legs and the three waist joints.  This contract owns
only the fourteen shoulder/elbow/wrist joints; head joints stay locked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


UPPER_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


@dataclass(frozen=True)
class UpperMotionClip:
    path: Path
    key: str
    fps: float
    joint_names: tuple[str, ...]
    q_rad: np.ndarray

    @property
    def duration_s(self) -> float:
        return (len(self.q_rad) - 1) / self.fps


@dataclass(frozen=True)
class UpperSafetyLimits:
    """Deployable bounds applied after intent decoding and before PD targets."""

    max_excursion_rad: float
    max_velocity_radps: float
    tilt_fallback_rad: float
    height_fallback_m: float

    def validate(self) -> None:
        if self.max_excursion_rad <= 0.0:
            raise ValueError("max upper excursion must be positive")
        if self.max_velocity_radps <= 0.0:
            raise ValueError("max upper velocity must be positive")
        if not 0.0 < self.tilt_fallback_rad <= np.pi:
            raise ValueError("tilt fallback threshold must lie in (0, pi]")


def bounded_upper_target_step(
    desired_q_rad: np.ndarray,
    default_q_rad: np.ndarray,
    previous_q_rad: np.ndarray,
    *,
    dt_s: float,
    limits: UpperSafetyLimits,
    healthy: np.ndarray | bool = True,
) -> np.ndarray:
    """Apply excursion, health-fallback, and slew-rate bounds to one target step."""
    limits.validate()
    if dt_s <= 0.0:
        raise ValueError("upper target dt must be positive")
    desired = np.asarray(desired_q_rad, dtype=np.float32)
    default = np.asarray(default_q_rad, dtype=np.float32)
    previous = np.asarray(previous_q_rad, dtype=np.float32)
    if desired.shape != default.shape or desired.shape != previous.shape:
        raise ValueError(
            f"upper target shape mismatch: {desired.shape}, {default.shape}, {previous.shape}"
        )
    bounded = default + np.clip(
        desired - default,
        -limits.max_excursion_rad,
        limits.max_excursion_rad,
    )
    health = np.asarray(healthy, dtype=bool)
    if health.ndim == bounded.ndim - 1:
        health = np.expand_dims(health, axis=-1)
    bounded = np.where(health, bounded, default)
    max_step = limits.max_velocity_radps * dt_s
    return (
        previous + np.clip(bounded - previous, -max_step, max_step)
    ).astype(np.float32, copy=False)


def wrapped_angle_delta_rad(angle_rad: np.ndarray, reference_rad: np.ndarray) -> np.ndarray:
    """Return the shortest signed angular difference in ``[-pi, pi]``."""
    angle = np.asarray(angle_rad)
    reference = np.asarray(reference_rad)
    delta = angle - reference
    return np.arctan2(np.sin(delta), np.cos(delta))


def load_upper_motion(path: str | Path) -> UpperMotionClip:
    """Load one trusted SONIC motion-cache file and retain the 14 arm joints."""
    resolved = Path(path).expanduser().resolve()
    loaded = joblib.load(resolved)
    if not isinstance(loaded, dict) or len(loaded) != 1:
        raise ValueError(f"expected one motion in {resolved}, got {type(loaded)!r}")
    key, payload = next(iter(loaded.items()))
    names = tuple(payload["joint_names_mujoco"])
    dof = np.asarray(payload["dof"], dtype=np.float32)
    fps = float(payload["fps"])
    if dof.ndim != 2 or dof.shape[1] != len(names):
        raise ValueError(f"invalid dof/name contract: {dof.shape} vs {len(names)}")
    if fps <= 0.0 or len(dof) < 2:
        raise ValueError(f"invalid motion timing: fps={fps}, frames={len(dof)}")
    missing = [name for name in UPPER_JOINT_NAMES if name not in names]
    if missing:
        raise ValueError(f"motion is missing X2 upper joints: {missing}")
    indices = [names.index(name) for name in UPPER_JOINT_NAMES]
    return UpperMotionClip(
        path=resolved,
        key=str(key),
        fps=fps,
        joint_names=UPPER_JOINT_NAMES,
        q_rad=np.ascontiguousarray(dof[:, indices]),
    )


def sample_linear(
    q_rad: np.ndarray,
    fps: float,
    time_s: np.ndarray | float,
    *,
    loop: bool = False,
) -> np.ndarray:
    """Linearly sample a frames×joints trajectory at arbitrary times."""
    q = np.asarray(q_rad)
    times = np.asarray(time_s, dtype=np.float64)
    last = len(q) - 1
    frame = times * fps
    if loop:
        frame = np.mod(frame, last)
    else:
        frame = np.clip(frame, 0.0, float(last))
    lower = np.floor(frame).astype(np.int64)
    upper = np.minimum(lower + 1, last)
    blend = (frame - lower)[..., None]
    return (1.0 - blend) * q[lower] + blend * q[upper]


def relative_upper_target(
    clip: UpperMotionClip,
    default_q_rad: np.ndarray,
    time_s: np.ndarray | float,
    *,
    start_s: float,
    scale: float,
    time_scale: float = 1.0,
    lower_limits_rad: np.ndarray | None = None,
    upper_limits_rad: np.ndarray | None = None,
    loop: bool = False,
) -> np.ndarray:
    """Return default + scaled(reference(t)-reference(start)), optionally clipped."""
    if scale < 0.0:
        raise ValueError("upper motion scale must be non-negative")
    if time_scale <= 0.0:
        raise ValueError("upper motion time scale must be positive")
    baseline = sample_linear(clip.q_rad, clip.fps, start_s, loop=loop)
    sample = sample_linear(
        clip.q_rad,
        clip.fps,
        np.asarray(time_s) * time_scale + start_s,
        loop=loop,
    )
    target = np.asarray(default_q_rad) + scale * (sample - baseline)
    if lower_limits_rad is not None or upper_limits_rad is not None:
        if lower_limits_rad is None or upper_limits_rad is None:
            raise ValueError("both lower and upper joint limits are required")
        target = np.clip(target, lower_limits_rad, upper_limits_rad)
    return target.astype(np.float32, copy=False)
