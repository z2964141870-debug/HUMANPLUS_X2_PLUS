#!/usr/bin/env python3
"""Pure, offline state machine for a future Sonic -> Develop_MC handoff.

This module deliberately has no ROS, HAL, socket, subprocess, or robot SDK
dependency.  It converts observations and operator commands into directives;
an integration adapter would remain responsible for all side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


UINT32_HALF = 1 << 31
UINT32_MASK = (1 << 32) - 1


def sequence_is_newer(sequence: int, previous: int) -> bool:
    delta = (int(sequence) - int(previous)) & UINT32_MASK
    return 0 < delta < UINT32_HALF


@dataclass(frozen=True)
class HealthLimits:
    max_local_age_s: float
    max_source_age_s: float
    max_group_skew_s: float
    max_inter_sample_gap_s: float


@dataclass(frozen=True)
class StreamHealth:
    complete: bool
    sequence_ok: bool
    local_age_s: float
    source_age_s: float
    group_skew_s: float
    max_gap_s: float
    stable_for_s: float
    reason: str = ""

    @classmethod
    def missing(cls, reason: str = "missing") -> "StreamHealth":
        return cls(False, False, math.inf, math.inf, math.inf, math.inf, 0.0, reason)

    def problem(self, limits: HealthLimits, stable_s: float = 0.0) -> str | None:
        if not self.complete:
            return self.reason or "groups_incomplete"
        if not self.sequence_ok:
            return self.reason or "sequence_discontinuity"
        if not all(
            math.isfinite(value)
            for value in (
                self.local_age_s,
                self.source_age_s,
                self.group_skew_s,
                self.max_gap_s,
                self.stable_for_s,
            )
        ):
            return "nonfinite_stream_health"
        if self.local_age_s > limits.max_local_age_s:
            return "local_receive_stale"
        if self.source_age_s > limits.max_source_age_s:
            return "source_sample_stale"
        if self.group_skew_s > limits.max_group_skew_s:
            return "group_cohort_skew"
        if self.max_gap_s > limits.max_inter_sample_gap_s:
            return "sample_gap"
        if self.stable_for_s < stable_s:
            return "stream_not_stable"
        return None


@dataclass
class _GroupSample:
    sequence: int
    received_at_s: float
    source_age_at_receive_s: float
    max_gap_s: float = 0.0


class GroupedFreshness:
    """Tracks freshness from both source age and local receive time.

    Sequence resets are never inferred.  An adapter may create a new tracker
    only before candidate ownership, after validating a new stream session.
    """

    def __init__(self, groups: Iterable[str]) -> None:
        expected = tuple(groups)
        if not expected or len(expected) != len(set(expected)):
            raise ValueError("groups must be non-empty and unique")
        self.groups = expected
        self._samples: dict[str, _GroupSample] = {}
        self._stable_since_s: float | None = None
        self._sequence_ok = True
        self._reason = ""

    def observe(
        self,
        group: str,
        sequence: int,
        received_at_s: float,
        source_age_s: float,
    ) -> bool:
        if group not in self.groups:
            raise KeyError(group)
        values = (received_at_s, source_age_s)
        if not all(math.isfinite(value) for value in values) or source_age_s < 0.0:
            self._sequence_ok = False
            self._reason = "invalid_timestamp"
            self._stable_since_s = None
            return False
        previous = self._samples.get(group)
        if previous is not None:
            if received_at_s < previous.received_at_s:
                self._sequence_ok = False
                self._reason = "receive_clock_regressed"
                self._stable_since_s = None
                return False
            if not sequence_is_newer(sequence, previous.sequence):
                self._sequence_ok = False
                self._reason = "replayed_or_out_of_order_sequence"
                self._stable_since_s = None
                return False
            gap = received_at_s - previous.received_at_s
            max_gap = max(previous.max_gap_s, gap)
        else:
            max_gap = 0.0
        self._samples[group] = _GroupSample(
            int(sequence) & UINT32_MASK,
            float(received_at_s),
            float(source_age_s),
            max_gap,
        )
        if len(self._samples) == len(self.groups) and self._sequence_ok:
            if self._stable_since_s is None:
                self._stable_since_s = min(
                    sample.received_at_s for sample in self._samples.values()
                )
        return True

    def health(self, now_s: float) -> StreamHealth:
        if not math.isfinite(now_s):
            return StreamHealth.missing("invalid_now")
        if len(self._samples) != len(self.groups):
            return StreamHealth.missing("groups_incomplete")
        received = [self._samples[group].received_at_s for group in self.groups]
        if now_s < max(received):
            return StreamHealth.missing("receive_clock_regressed")
        local_ages = [now_s - value for value in received]
        source_ages = [
            self._samples[group].source_age_at_receive_s
            + now_s
            - self._samples[group].received_at_s
            for group in self.groups
        ]
        stable_since = self._stable_since_s
        return StreamHealth(
            complete=True,
            sequence_ok=self._sequence_ok,
            local_age_s=max(local_ages),
            source_age_s=max(source_ages),
            group_skew_s=max(received) - min(received),
            max_gap_s=max(sample.max_gap_s for sample in self._samples.values()),
            stable_for_s=0.0 if stable_since is None else max(0.0, now_s - stable_since),
            reason=self._reason,
        )


class State(str, Enum):
    OBSERVE_OFFICIAL = "OBSERVE_OFFICIAL"
    OVERLAP_OFFICIAL = "OVERLAP_OFFICIAL"
    WAIT_DEVELOP = "WAIT_DEVELOP"
    RAMP_CANDIDATE = "RAMP_CANDIDATE"
    CANDIDATE_HOLD = "CANDIDATE_HOLD"
    ACTIVE = "ACTIVE"
    RETURN_RAMP = "RETURN_RAMP"
    WAIT_READY = "WAIT_READY"
    VERIFY_OFFICIAL = "VERIFY_OFFICIAL"
    LOCKED_RECOVERY = "LOCKED_RECOVERY"
    ABORTED = "ABORTED"
    COMPLETE = "COMPLETE"
    ESTOP = "ESTOP"


class Command(str, Enum):
    NONE = "NONE"
    START = "START"
    ACTIVATE = "ACTIVATE"
    HANDBACK = "HANDBACK"
    MIGRATION_REJECTED = "MIGRATION_REJECTED"
    EXIT = "EXIT"
    ESTOP = "ESTOP"


class TargetMode(str, Enum):
    NONE = "NONE"
    MIRROR_CAPTURED_OFFICIAL = "MIRROR_CAPTURED_OFFICIAL"
    CANDIDATE_ANCHOR = "CANDIDATE_ANCHOR"
    SONIC_RATE_LIMITED = "SONIC_RATE_LIMITED"
    FREEZE_LAST_ACCEPTED = "FREEZE_LAST_ACCEPTED"
    MIRROR_LIVE_OFFICIAL = "MIRROR_LIVE_OFFICIAL"


@dataclass(frozen=True)
class OutputContinuity:
    complete: bool
    sequence_ok: bool
    last_publish_age_s: float
    max_publish_gap_s: float
    all_groups_same_cycle: bool

    def problem(self, max_age_s: float, max_gap_s: float) -> str | None:
        if not self.complete or not self.all_groups_same_cycle:
            return "output_groups_incomplete"
        if not self.sequence_ok:
            return "output_sequence_discontinuity"
        if not math.isfinite(self.last_publish_age_s) or self.last_publish_age_s > max_age_s:
            return "output_publish_stale"
        if not math.isfinite(self.max_publish_gap_s) or self.max_publish_gap_s > max_gap_s:
            return "output_publish_gap"
        return None


@dataclass(frozen=True)
class Observation:
    system_state: str
    system_status_ready: bool
    feedback: StreamHealth
    official: StreamHealth
    sonic: StreamHealth
    output: OutputContinuity | None
    official_powered: bool
    official_stationary: bool
    official_contract_complete: bool
    official_sample_after_handback: bool
    target_error_max_rad: float
    measured_speed_max_rad_s: float
    tracking_error_max_rad: float
    vibration_velocity_rms_rad_s: float
    vibration_accel_rms_rad_s2: float
    vibration_reversal_hz: float


@dataclass(frozen=True)
class Config:
    feedback_limits: HealthLimits = HealthLimits(0.08, 0.08, 0.012, 0.025)
    official_limits: HealthLimits = HealthLimits(0.10, 0.10, 0.020, 0.040)
    sonic_limits: HealthLimits = HealthLimits(0.12, 0.12, 0.020, 0.050)
    preflight_stable_s: float = 1.0
    overlap_s: float = 0.75
    output_grace_s: float = 0.05
    output_max_age_s: float = 0.008
    output_max_gap_s: float = 0.010
    migration_timeout_s: float = 15.0
    contract_ramp_s: float = 3.0
    candidate_hold_s: float = 1.0
    return_ramp_s: float = 3.0
    return_settle_s: float = 0.75
    official_verify_s: float = 1.0
    max_preflight_target_error_rad: float = 0.05
    max_preflight_speed_rad_s: float = 0.08
    max_tracking_error_rad: float = 0.08
    max_vibration_velocity_rms_rad_s: float = 0.12
    max_vibration_accel_rms_rad_s2: float = 8.0
    max_vibration_reversal_hz: float = 5.0
    vibration_dwell_s: float = 0.08


@dataclass(frozen=True)
class Directive:
    state: State
    publish: bool
    target_mode: TargetMode
    contract_blend: float
    transition_request: str | None
    can_exit: bool
    fault_latched: bool
    reason: str


def smoothstep5(progress: float) -> float:
    x = min(1.0, max(0.0, float(progress)))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


class Machine:
    """Deterministic authority and recovery state machine.

    Once a Develop_MC request is emitted, failures latch candidate ownership.
    The machine never interprets fresh data as permission to resume and never
    permits process exit until verified official overlap or physical E-stop.
    """

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.state = State.OBSERVE_OFFICIAL
        self.reason = "awaiting_preflight"
        self._last_now_s: float | None = None
        self._preflight_since_s: float | None = None
        self._state_since_s = 0.0
        self._healthy_since_s: float | None = None
        self._hazard_since_s: float | None = None
        self._migration_intent_emitted = False
        self._ready_intent_emitted = False
        self._live_official_verified = False
        self._blend = 0.0
        self._return_start_blend = 0.0

    @property
    def ownership_uncertain(self) -> bool:
        return self._migration_intent_emitted and self.state not in (
            State.COMPLETE,
            State.ESTOP,
        )

    def _enter(self, state: State, now_s: float, reason: str) -> None:
        self.state = state
        self._state_since_s = now_s
        self._healthy_since_s = None
        self.reason = reason
        if state == State.VERIFY_OFFICIAL:
            self._live_official_verified = False

    def _preflight_problem(self, obs: Observation) -> str | None:
        if obs.system_state.lower() != "business" or not obs.system_status_ready:
            return "official_business_not_ready"
        problem = obs.feedback.problem(
            self.config.feedback_limits, self.config.preflight_stable_s
        )
        if problem:
            return f"feedback_{problem}"
        problem = obs.official.problem(
            self.config.official_limits, self.config.preflight_stable_s
        )
        if problem:
            return f"official_{problem}"
        if not obs.official_contract_complete:
            return "official_contract_incomplete"
        if not obs.official_powered:
            return "official_contract_unpowered"
        if not obs.official_stationary:
            return "official_contract_moving"
        if not math.isfinite(obs.target_error_max_rad):
            return "target_error_nonfinite"
        if obs.target_error_max_rad > self.config.max_preflight_target_error_rad:
            return "official_target_mismatch"
        if not math.isfinite(obs.measured_speed_max_rad_s):
            return "measured_speed_nonfinite"
        if obs.measured_speed_max_rad_s > self.config.max_preflight_speed_rad_s:
            return "robot_not_stationary"
        return None

    def _output_problem(self, obs: Observation, now_s: float) -> str | None:
        if now_s - self._state_since_s <= self.config.output_grace_s:
            return None
        if obs.output is None:
            return "output_unobserved"
        return obs.output.problem(
            self.config.output_max_age_s, self.config.output_max_gap_s
        )

    def _dynamic_problem(self, obs: Observation, needs_sonic: bool) -> str | None:
        problem = obs.feedback.problem(self.config.feedback_limits)
        if problem:
            return f"feedback_{problem}"
        if needs_sonic:
            problem = obs.sonic.problem(self.config.sonic_limits)
            if problem:
                return f"sonic_{problem}"
        if not math.isfinite(obs.tracking_error_max_rad):
            return "tracking_error_nonfinite"
        if obs.tracking_error_max_rad > self.config.max_tracking_error_rad:
            return "tracking_error"
        return None

    def _vibration_problem(self, obs: Observation, now_s: float) -> str | None:
        metrics = (
            obs.vibration_velocity_rms_rad_s,
            obs.vibration_accel_rms_rad_s2,
            obs.vibration_reversal_hz,
        )
        if not all(math.isfinite(value) for value in metrics):
            return "vibration_metric_nonfinite"
        violation = (
            metrics[0] > self.config.max_vibration_velocity_rms_rad_s
            or metrics[1] > self.config.max_vibration_accel_rms_rad_s2
            or metrics[2] > self.config.max_vibration_reversal_hz
        )
        if not violation:
            self._hazard_since_s = None
            return None
        if self._hazard_since_s is None:
            self._hazard_since_s = now_s
            return None
        if now_s - self._hazard_since_s >= self.config.vibration_dwell_s:
            return "vibration_detected"
        return None

    def _lock(self, now_s: float, reason: str) -> None:
        self._enter(State.LOCKED_RECOVERY, now_s, reason)

    def _request_return(self, now_s: float, reason: str) -> None:
        self._return_start_blend = self._blend
        self._enter(State.RETURN_RAMP, now_s, reason)

    def _directive(self, transition_request: str | None = None) -> Directive:
        state = self.state
        if state in (State.OBSERVE_OFFICIAL, State.ABORTED, State.COMPLETE, State.ESTOP):
            mode = TargetMode.NONE
            publish = False
        elif state in (State.OVERLAP_OFFICIAL, State.WAIT_DEVELOP):
            mode = TargetMode.MIRROR_CAPTURED_OFFICIAL
            publish = True
        elif state in (State.RAMP_CANDIDATE, State.CANDIDATE_HOLD, State.RETURN_RAMP):
            mode = TargetMode.CANDIDATE_ANCHOR
            publish = True
        elif state == State.ACTIVE:
            mode = TargetMode.SONIC_RATE_LIMITED
            publish = True
        elif state == State.VERIFY_OFFICIAL and self._live_official_verified:
            mode = TargetMode.MIRROR_LIVE_OFFICIAL
            publish = True
        else:
            mode = TargetMode.FREEZE_LAST_ACCEPTED
            publish = True
        return Directive(
            state=state,
            publish=publish,
            target_mode=mode,
            contract_blend=min(1.0, max(0.0, self._blend)),
            transition_request=transition_request,
            can_exit=state in (State.OBSERVE_OFFICIAL, State.ABORTED, State.COMPLETE, State.ESTOP),
            fault_latched=state == State.LOCKED_RECOVERY,
            reason=self.reason,
        )

    def latch_external_fault(self, now_s: float, reason: str) -> Directive:
        """Latch a fault found by an independent supervisor.

        This is a pure state transition.  It exists so an adapter does not
        have to disguise supervisor failures as missing feedback or migration
        rejection.  Before migration intent, the same fault only blocks start;
        after intent, candidate ownership is retained for explicit recovery.
        """
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._last_now_s is not None and now_s < self._last_now_s:
            raise ValueError("monotonic time regressed")
        self._last_now_s = now_s
        normalized = str(reason).strip() or "unspecified_external_fault"
        if self.ownership_uncertain:
            self._lock(now_s, f"supervisor:{normalized}")
        else:
            self.reason = f"supervisor_blocked:{normalized}"
        return self._directive()

    def step(
        self,
        now_s: float,
        obs: Observation,
        command: Command = Command.NONE,
    ) -> Directive:
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._last_now_s is not None and now_s < self._last_now_s:
            raise ValueError("monotonic time regressed")
        self._last_now_s = now_s

        if command == Command.ESTOP:
            self._enter(State.ESTOP, now_s, "physical_estop_confirmed")
            return self._directive()
        if self.state in (State.COMPLETE, State.ESTOP, State.ABORTED):
            return self._directive()

        if self.state == State.OBSERVE_OFFICIAL:
            if obs.system_state.lower() == "develop_mc":
                self._lock(now_s, "unexpected_develop_without_candidate_output")
                return self._directive()
            problem = self._preflight_problem(obs)
            if problem:
                self._preflight_since_s = None
                self.reason = problem
            else:
                if self._preflight_since_s is None:
                    self._preflight_since_s = now_s
                self.reason = "preflight_stabilizing"
            if command == Command.START:
                stable = (
                    problem is None
                    and self._preflight_since_s is not None
                    and now_s - self._preflight_since_s >= self.config.preflight_stable_s
                )
                if stable:
                    self._blend = 0.0
                    self._enter(State.OVERLAP_OFFICIAL, now_s, "official_overlap")
                else:
                    self.reason = f"start_rejected:{problem or 'preflight_dwell'}"
            return self._directive()

        if command == Command.EXIT:
            if self.state == State.OVERLAP_OFFICIAL and not self._migration_intent_emitted:
                self._enter(State.ABORTED, now_s, "exit_before_migration")
            else:
                self._lock(now_s, "exit_requested_while_ownership_uncertain")
            return self._directive()

        if self.state == State.OVERLAP_OFFICIAL:
            problem = self._preflight_problem(obs) or self._output_problem(obs, now_s)
            if problem:
                self._enter(State.ABORTED, now_s, f"overlap_failed:{problem}")
                return self._directive()
            if now_s - self._state_since_s >= self.config.overlap_s:
                self._migration_intent_emitted = True
                self._enter(State.WAIT_DEVELOP, now_s, "develop_request_emitted")
                return self._directive("Develop_MC")
            return self._directive()

        if command == Command.MIGRATION_REJECTED:
            self._lock(now_s, "migration_rejected_or_outcome_uncertain")
            return self._directive()

        if self.state == State.WAIT_DEVELOP:
            problem = self._output_problem(obs, now_s)
            if problem:
                self._lock(now_s, problem)
            elif obs.system_state.lower() == "develop_mc" and obs.system_status_ready:
                self._blend = 0.0
                self._enter(State.RAMP_CANDIDATE, now_s, "candidate_contract_ramp")
            elif now_s - self._state_since_s > self.config.migration_timeout_s:
                self._lock(now_s, "develop_confirmation_timeout")
            return self._directive()

        if self.state == State.LOCKED_RECOVERY:
            if command == Command.HANDBACK:
                self._request_return(now_s, "operator_recovery_handback")
            return self._directive()

        if self.state in (
            State.RAMP_CANDIDATE,
            State.CANDIDATE_HOLD,
            State.ACTIVE,
            State.RETURN_RAMP,
        ):
            if obs.system_state.lower() != "develop_mc" or not obs.system_status_ready:
                self._enter(State.VERIFY_OFFICIAL, now_s, "external_handback_detected")
                return self._directive()
            needs_sonic = self.state == State.ACTIVE
            problem = (
                self._output_problem(obs, now_s)
                or self._dynamic_problem(obs, needs_sonic)
                or self._vibration_problem(obs, now_s)
            )
            if problem:
                self._lock(now_s, problem)
                return self._directive()

        if self.state == State.RAMP_CANDIDATE:
            elapsed = now_s - self._state_since_s
            self._blend = smoothstep5(elapsed / self.config.contract_ramp_s)
            if elapsed >= self.config.contract_ramp_s:
                self._blend = 1.0
                self._enter(State.CANDIDATE_HOLD, now_s, "candidate_hold_validation")
            return self._directive()

        if self.state == State.CANDIDATE_HOLD:
            if command == Command.HANDBACK:
                self._request_return(now_s, "operator_handback")
            elif command == Command.ACTIVATE:
                sonic_problem = obs.sonic.problem(
                    self.config.sonic_limits, self.config.candidate_hold_s
                )
                held_long_enough = (
                    now_s - self._state_since_s >= self.config.candidate_hold_s
                )
                if sonic_problem is None and held_long_enough:
                    self._enter(State.ACTIVE, now_s, "sonic_active")
                else:
                    self.reason = f"activate_rejected:{sonic_problem or 'hold_dwell'}"
            return self._directive()

        if self.state == State.ACTIVE:
            if command == Command.HANDBACK:
                self._request_return(now_s, "operator_handback")
            return self._directive()

        if self.state == State.RETURN_RAMP:
            elapsed = now_s - self._state_since_s
            remaining = 1.0 - smoothstep5(elapsed / self.config.return_ramp_s)
            self._blend = self._return_start_blend * remaining
            if elapsed >= self.config.return_ramp_s:
                self._blend = 0.0
                if obs.tracking_error_max_rad <= self.config.max_tracking_error_rad:
                    if self._healthy_since_s is None:
                        self._healthy_since_s = now_s
                    if now_s - self._healthy_since_s >= self.config.return_settle_s:
                        self._ready_intent_emitted = True
                        self._enter(State.WAIT_READY, now_s, "ready_request_emitted")
                        return self._directive("Ready")
                else:
                    self._healthy_since_s = None
            return self._directive()

        if self.state == State.WAIT_READY:
            problem = self._output_problem(obs, now_s)
            if problem:
                self._lock(now_s, problem)
            elif (
                obs.system_state.lower() in ("ready", "business")
                and obs.system_status_ready
            ):
                self._enter(State.VERIFY_OFFICIAL, now_s, "verify_official_overlap")
            return self._directive()

        if self.state == State.VERIFY_OFFICIAL:
            problem = self._output_problem(obs, now_s)
            if problem:
                self._lock(now_s, problem)
                return self._directive()
            official_ok = (
                obs.system_state.lower() == "business"
                and obs.system_status_ready
                and obs.official.problem(self.config.official_limits) is None
                and obs.feedback.problem(self.config.feedback_limits) is None
                and obs.official_contract_complete
                and obs.official_powered
                and obs.official_stationary
                and obs.official_sample_after_handback
            )
            if official_ok:
                self._live_official_verified = True
                if self._healthy_since_s is None:
                    self._healthy_since_s = now_s
                if now_s - self._healthy_since_s >= self.config.official_verify_s:
                    self._enter(State.COMPLETE, now_s, "official_ownership_verified")
            else:
                self._live_official_verified = False
                self._healthy_since_s = None
                self.reason = "official_overlap_not_verified"
            return self._directive()

        raise AssertionError(f"unhandled state {self.state}")
