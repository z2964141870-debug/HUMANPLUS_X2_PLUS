#!/usr/bin/env python3
"""Pure, offline-testable safety boundary for the X2 policy controller.

This module never opens a socket and never publishes a command.  It validates
one real-state snapshot and converts a raw 31-joint policy target into a
bounded, rate-limited target.  Callers must stop sending when ``accepted`` is
false; the class deliberately has no fallback that invents robot state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from x2_state_feedback import FeedbackSnapshot, N_QPOS, N_QVEL, REQUIRED_FLAGS


N_JOINTS = 31
# Must match eval_official_sonic_x2.DEFAULT_ANGLES_MJ.  Kept here so this
# safety boundary can be unit-tested without MuJoCo/ONNX Runtime installed.
POLICY_DEFAULT_ANGLES = np.asarray([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
    0.0, 0.0,
], dtype=np.float64)


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    qi = np.asarray([w, -x, -y, -z])
    vx, vy, vz = np.asarray(v, dtype=np.float64)
    _, ix, iy, iz = qi
    tx = 2.0 * (iy * vz - iz * vy)
    ty = 2.0 * (iz * vx - ix * vz)
    tz = 2.0 * (ix * vy - iy * vx)
    return np.asarray([
        vx + qi[0] * tx + (iy * tz - iz * ty),
        vy + qi[0] * ty + (iz * tx - ix * tz),
        vz + qi[0] * tz + (ix * ty - iy * tx),
    ])


@dataclass(frozen=True)
class SafetyConfig:
    feedback_timeout_s: float = 0.15
    max_joint_speed_rad_s: float = 15.0
    max_tilt_rad: float = 0.85
    max_policy_offset_rad: float = 0.40
    max_tracking_error_rad: float = 0.55
    max_target_rate_rad_s: float = 1.50
    control_hz: float = 50.0


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    reason: str
    targets: Optional[np.ndarray] = None
    detail: str = ""


class ClosedLoopSafetyFilter:
    """Fail-closed state checks plus deterministic target filtering."""

    def __init__(self, config: SafetyConfig = SafetyConfig()):
        self.config = config
        self._previous_target: Optional[np.ndarray] = None
        self._target_center = POLICY_DEFAULT_ANGLES.copy()

    def reset(self, measured_joints: Optional[np.ndarray] = None,
              nominal_joints: Optional[np.ndarray] = None) -> None:
        self._previous_target = None
        self._target_center = POLICY_DEFAULT_ANGLES.copy()
        if nominal_joints is not None:
            center = np.asarray(nominal_joints, dtype=np.float64).reshape(-1)
            if center.shape != (N_JOINTS,) or not np.all(np.isfinite(center)):
                raise ValueError("nominal_joints must be finite shape (31,)")
            self._target_center = center.copy()
        if measured_joints is not None:
            q = np.asarray(measured_joints, dtype=np.float64).reshape(-1)
            if q.shape == (N_JOINTS,) and np.all(np.isfinite(q)):
                self._previous_target = q.copy()

    def validate_feedback(self, state: Optional[FeedbackSnapshot]) -> Optional[str]:
        cfg = self.config
        if state is None:
            return "feedback_missing_or_stale"
        if state.age_s > cfg.feedback_timeout_s or state.source_age_s > cfg.feedback_timeout_s:
            return "feedback_stale"
        if (state.flags & REQUIRED_FLAGS) != REQUIRED_FLAGS:
            return "feedback_validity_flags_missing"
        qpos = np.asarray(state.qpos, dtype=np.float64)
        qvel = np.asarray(state.qvel, dtype=np.float64)
        if qpos.shape != (N_QPOS,) or qvel.shape != (N_QVEL,):
            return "feedback_shape_invalid"
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            return "feedback_nonfinite"
        quat_norm = float(np.linalg.norm(qpos[3:7]))
        if not 0.95 <= quat_norm <= 1.05:
            return "feedback_quaternion_invalid"
        if float(np.max(np.abs(qvel[6:37]))) > cfg.max_joint_speed_rad_s:
            return "joint_speed_limit"
        gravity_body = _quat_rotate_inv(qpos[3:7], [0.0, 0.0, -1.0])
        tilt = float(np.arccos(np.clip(-gravity_body[2], -1.0, 1.0)))
        if tilt > cfg.max_tilt_rad:
            return "base_tilt_limit"
        return None

    def filter(self, raw_targets: np.ndarray,
               state: Optional[FeedbackSnapshot], *,
               enforce_tracking: bool = True) -> SafetyDecision:
        reason = self.validate_feedback(state)
        if reason is not None:
            self._previous_target = None
            return SafetyDecision(False, reason)

        return self._filter_targets(
            raw_targets, state.qpos[7:38],
            enforce_tracking=enforce_tracking,
        )

    def filter_offline(self, raw_targets: np.ndarray,
                       synthetic_joints: np.ndarray) -> SafetyDecision:
        """Exercise target filtering without feedback; callers must be dry-run."""
        return self._filter_targets(
            raw_targets, synthetic_joints, enforce_tracking=False,
        )

    def _filter_targets(self, raw_targets: np.ndarray,
                        measured_joints: np.ndarray, *,
                        enforce_tracking: bool) -> SafetyDecision:
        targets = np.asarray(raw_targets, dtype=np.float64).reshape(-1)
        if targets.shape != (N_JOINTS,) or not np.all(np.isfinite(targets)):
            self._previous_target = None
            return SafetyDecision(False, "policy_target_invalid")

        measured = np.asarray(measured_joints, dtype=np.float64).reshape(-1)
        if measured.shape != (N_JOINTS,) or not np.all(np.isfinite(measured)):
            self._previous_target = None
            return SafetyDecision(False, "measured_joints_invalid")
        cfg = self.config
        # Normal operation is centered on Sonic's trained default.  Suspended
        # operation explicitly replaces this center with the measured
        # load-bearing handoff pose; otherwise enabling Sonic would pull the
        # robot toward the unsupported ground-stand posture.
        envelope_lo = self._target_center - cfg.max_policy_offset_rad
        envelope_hi = self._target_center + cfg.max_policy_offset_rad
        bounded = np.clip(targets, envelope_lo, envelope_hi)
        if self._previous_target is None:
            self._previous_target = measured.copy()
        elif enforce_tracking:
            # This robot's two head joints are intentionally not commanded
            # because HAL reports head error 1026.  They must not trip the
            # tracking gate for the 29 body joints that we actually own.
            tracking_error = np.abs(self._previous_target[:29] - measured[:29])
            worst = int(np.argmax(tracking_error))
            worst_error = float(tracking_error[worst])
            if worst_error > cfg.max_tracking_error_rad:
                self._previous_target = None
                return SafetyDecision(
                    False, "tracking_error_limit", detail=(
                        f"joint_index={worst} error={worst_error:.5f}rad"
                    ),
                )
        max_step = cfg.max_target_rate_rad_s / cfg.control_hz
        command = self._previous_target + np.clip(
            bounded - self._previous_target, -max_step, max_step,
        )
        self._previous_target = command.copy()
        return SafetyDecision(True, "ok", command)
