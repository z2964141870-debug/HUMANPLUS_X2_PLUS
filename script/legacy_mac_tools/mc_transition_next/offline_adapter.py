#!/usr/bin/env python3
"""Offline integration of the handoff machine, PD contract, and supervisor.

The module is deliberately side-effect free: no ROS, HAL, sockets, subprocess,
or robot SDK calls.  It turns already-observed evidence into an atomic 31-joint
PD frame suitable for replay and design verification only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .state_machine import Command, Directive, Machine, Observation, TargetMode


JOINT_COUNT = 31


def _finite_vector(values: Any, name: str, *, nonnegative: bool = False) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (JOINT_COUNT,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite shape ({JOINT_COUNT},)")
    if nonnegative and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return np.frombuffer(vector.tobytes(), dtype=np.float64)


@dataclass(frozen=True)
class PdFrame:
    target: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    contract_fingerprint: str

    @classmethod
    def build(
        cls,
        *,
        target: Any,
        velocity: Any,
        effort: Any,
        stiffness: Any,
        damping: Any,
        contract_fingerprint: str,
    ) -> "PdFrame":
        fingerprint = str(contract_fingerprint)
        if not fingerprint:
            raise ValueError("contract_fingerprint is required")
        return cls(
            target=_finite_vector(target, "target"),
            velocity=_finite_vector(velocity, "velocity"),
            effort=_finite_vector(effort, "effort"),
            stiffness=_finite_vector(stiffness, "stiffness", nonnegative=True),
            damping=_finite_vector(damping, "damping", nonnegative=True),
            contract_fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class AdapterInput:
    observation: Observation
    supervisor_snapshot: Any
    official_frame: PdFrame
    measured_position: np.ndarray
    measured_velocity: np.ndarray
    sonic_action: np.ndarray


@dataclass(frozen=True)
class AdapterResult:
    directive: Directive
    frame: PdFrame | None
    supervisor_decision: Any
    blocked_reason: str | None


class OfflineTransitionAdapter:
    """Compose safety evidence without creating a live control surface."""

    def __init__(
        self,
        *,
        machine: Machine,
        contract: Any,
        supervisor: Any,
        consumption_monitor: Any,
        max_target_rate_rad_s: float = 0.20,
        max_target_offset_rad: float = 0.03,
    ) -> None:
        if not math.isfinite(max_target_rate_rad_s) or max_target_rate_rad_s <= 0.0:
            raise ValueError("max_target_rate_rad_s must be positive and finite")
        if not math.isfinite(max_target_offset_rad) or max_target_offset_rad <= 0.0:
            raise ValueError("max_target_offset_rad must be positive and finite")
        self.machine = machine
        self.contract = contract
        self.supervisor = supervisor
        self.consumption_monitor = consumption_monitor
        self.max_target_rate_rad_s = float(max_target_rate_rad_s)
        self.max_target_offset_rad = float(max_target_offset_rad)
        self._captured_official: PdFrame | None = None
        self._candidate_anchor: np.ndarray | None = None
        self._last_accepted: PdFrame | None = None
        self._last_frame_time_s: float | None = None

    def _candidate_frame(self, target: Any) -> PdFrame:
        return PdFrame.build(
            target=target,
            velocity=np.zeros(JOINT_COUNT),
            effort=np.zeros(JOINT_COUNT),
            stiffness=self.contract.stiffness,
            damping=self.contract.damping,
            contract_fingerprint=self.contract.fingerprint,
        )

    def _blend_to_candidate(
        self,
        blend: float,
        measured_position: np.ndarray,
        measured_velocity: np.ndarray,
    ) -> PdFrame:
        if self._captured_official is None:
            raise RuntimeError("official PD frame was not captured before overlap")
        alpha = min(1.0, max(0.0, float(blend)))
        official = self._captured_official
        q = _finite_vector(measured_position, "measured_position")
        dq = _finite_vector(measured_velocity, "measured_velocity")
        stiffness = official.stiffness + alpha * (
            self.contract.stiffness - official.stiffness
        )
        damping = official.damping + alpha * (
            self.contract.damping - official.damping
        )
        if np.any(stiffness <= 0.0):
            raise ValueError("blended stiffness must remain strictly positive")
        velocity = (1.0 - alpha) * official.velocity
        effort = (1.0 - alpha) * official.effort
        official_torque = (
            official.effort
            + official.stiffness * (official.target - q)
            + official.damping * (official.velocity - dq)
        )
        target = q + (
            official_torque - effort - damping * (velocity - dq)
        ) / stiffness
        frame = PdFrame.build(
            target=target,
            velocity=velocity,
            effort=effort,
            stiffness=stiffness,
            damping=damping,
            contract_fingerprint=self.contract.fingerprint,
        )
        if alpha >= 1.0:
            self._candidate_anchor = frame.target
        return frame

    def _anchored_sonic_target(self, action: np.ndarray) -> np.ndarray:
        if self._captured_official is None:
            raise RuntimeError("official PD frame was not captured before Sonic")
        if self._candidate_anchor is None:
            raise RuntimeError("torque-equivalent candidate anchor was not established")
        anchor = self._candidate_anchor
        model_target = self.contract.targets_from_action(action)
        desired = anchor + (model_target - self.contract.default_position)
        lower = np.maximum(
            self.contract.lower_limit,
            anchor - self.max_target_offset_rad,
        )
        upper = np.minimum(
            self.contract.upper_limit,
            anchor + self.max_target_offset_rad,
        )
        # A calibrated official target may sit slightly outside generic model
        # limits. Preserve that exact target, but never command farther out.
        lower = np.minimum(lower, anchor)
        upper = np.maximum(upper, anchor)
        return np.clip(desired, lower, upper)

    def _rate_limited_sonic(self, now_s: float, action: np.ndarray) -> PdFrame:
        desired = self._anchored_sonic_target(action)
        if self._last_accepted is None or self._last_frame_time_s is None:
            if self._candidate_anchor is None:
                raise RuntimeError("candidate anchor was not established before Sonic")
            previous = self._candidate_anchor
            elapsed = 0.0
        else:
            previous = self._last_accepted.target
            elapsed = max(0.0, now_s - self._last_frame_time_s)
        delta = self.max_target_rate_rad_s * elapsed
        target = np.clip(desired, previous - delta, previous + delta)
        return self._candidate_frame(target)

    def _frame_for(
        self, directive: Directive, now_s: float, data: AdapterInput
    ) -> PdFrame | None:
        mode = directive.target_mode
        if not directive.publish or mode == TargetMode.NONE:
            return None
        if mode == TargetMode.MIRROR_CAPTURED_OFFICIAL:
            if self._captured_official is None:
                raise RuntimeError("official PD frame was not captured")
            return self._captured_official
        if mode == TargetMode.CANDIDATE_ANCHOR:
            return self._blend_to_candidate(
                directive.contract_blend,
                data.measured_position,
                data.measured_velocity,
            )
        if mode == TargetMode.SONIC_RATE_LIMITED:
            return self._rate_limited_sonic(now_s, data.sonic_action)
        if mode == TargetMode.FREEZE_LAST_ACCEPTED:
            if self._last_accepted is None:
                raise RuntimeError("cannot freeze before an atomic frame was accepted")
            return self._last_accepted
        if mode == TargetMode.MIRROR_LIVE_OFFICIAL:
            return data.official_frame
        raise AssertionError(f"unhandled target mode {mode}")

    def step(
        self,
        now_s: float,
        data: AdapterInput,
        command: Command = Command.NONE,
    ) -> AdapterResult:
        self.contract.validate_runtime_fingerprint(
            data.supervisor_snapshot.contract_fingerprint
        )
        consumers_proven = self.consumption_monitor.all_groups_proven(now_s)
        snapshot = replace(
            data.supervisor_snapshot,
            now_s=now_s,
            consumers_proven=consumers_proven,
            ownership_active=self.machine.ownership_uncertain,
        )
        decision = self.supervisor.evaluate(snapshot)
        action = getattr(decision.action, "value", str(decision.action))
        blocked_reason: str | None = None

        if action == "KEEP_MIRROR_REQUEST_OFFICIAL_RECOVERY":
            directive = self.machine.latch_external_fault(now_s, decision.reason)
        elif action == "BLOCK":
            blocked_reason = decision.reason
            directive = self.machine.latch_external_fault(now_s, decision.reason)
        else:
            directive = self.machine.step(now_s, data.observation, command)

        if (
            directive.target_mode == TargetMode.MIRROR_CAPTURED_OFFICIAL
            and self._captured_official is None
        ):
            self._captured_official = data.official_frame

        frame = self._frame_for(directive, now_s, data)
        if frame is not None:
            self._last_accepted = frame
            self._last_frame_time_s = now_s
        return AdapterResult(directive, frame, decision, blocked_reason)
