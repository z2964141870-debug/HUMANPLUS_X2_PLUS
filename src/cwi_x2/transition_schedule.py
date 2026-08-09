"""Pure command schedule used by the X2 start/cruise/stop curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransitionEventAudit:
    """Static proof that one rollout contains an aligned stop event."""

    event_end_s: float
    required_rollout_s: float
    rollout_s: float
    terminal_gait_phase: float
    terminal_in_double_support: bool
    random_episode_phase: bool
    maximum_phase_offset_s: float


def stratified_phase_offsets(
    num_envs: int,
    maximum_offset_s: float,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Deterministically cover transition phases across parallel envs."""
    if num_envs <= 0:
        raise ValueError("number of environments must be positive")
    if maximum_offset_s < 0.0:
        raise ValueError("maximum phase offset must be non-negative")
    if num_envs == 1:
        return torch.zeros(1, device=device)
    return torch.linspace(0.0, maximum_offset_s, num_envs, device=device)


def smooth_transition_speed(
    elapsed_s: torch.Tensor,
    cruise_speed: torch.Tensor,
    *,
    stand_s: float,
    accelerate_s: float,
    cruise_s: float,
    decelerate_s: float,
) -> torch.Tensor:
    """Return a C1 stand -> cruise -> stop velocity schedule."""
    durations = (stand_s, accelerate_s, cruise_s, decelerate_s)
    if any(value < 0.0 for value in durations):
        raise ValueError("transition durations must be non-negative")
    if accelerate_s <= 0.0 or decelerate_s <= 0.0:
        raise ValueError("acceleration and deceleration durations must be positive")
    if elapsed_s.shape != cruise_speed.shape:
        raise ValueError("elapsed time and cruise speed must have identical shapes")

    acceleration_phase = torch.clamp(
        (elapsed_s - stand_s) / accelerate_s,
        min=0.0,
        max=1.0,
    )
    acceleration_blend = acceleration_phase.square() * (
        3.0 - 2.0 * acceleration_phase
    )
    deceleration_start = stand_s + accelerate_s + cruise_s
    deceleration_phase = torch.clamp(
        (elapsed_s - deceleration_start) / decelerate_s,
        min=0.0,
        max=1.0,
    )
    deceleration_blend = 1.0 - deceleration_phase.square() * (
        3.0 - 2.0 * deceleration_phase
    )
    return cruise_speed * acceleration_blend * deceleration_blend


def audit_phase_consistent_event(
    *,
    stand_s: float,
    accelerate_s: float,
    cruise_s: float,
    decelerate_s: float,
    terminal_hold_s: float,
    gait_cycle_s: float,
    double_support_fraction: float,
    rollout_s: float,
    random_episode_phase: bool,
    maximum_phase_offset_s: float,
) -> TransitionEventAudit:
    """Reject a reset/command clock mismatch before IsaacLab is launched.

    A full event rollout must start from the physical reset at event time zero,
    contain the complete stand/accelerate/cruise/decelerate sequence, and keep
    a terminal hold window.  The final moving instant must land inside a
    scheduled double-support interval so contact and velocity objectives do
    not prescribe contradictory terminal states.
    """
    durations = (stand_s, accelerate_s, cruise_s, decelerate_s, terminal_hold_s)
    if any(value < 0.0 for value in durations):
        raise ValueError("transition event durations must be non-negative")
    if accelerate_s <= 0.0 or decelerate_s <= 0.0:
        raise ValueError("transition acceleration and deceleration must be positive")
    if gait_cycle_s <= 0.0:
        raise ValueError("gait cycle must be positive")
    if not 0.0 < double_support_fraction < 1.0:
        raise ValueError("double-support fraction must lie in (0, 1)")
    if rollout_s <= 0.0:
        raise ValueError("rollout duration must be positive")
    if maximum_phase_offset_s < 0.0:
        raise ValueError("maximum phase offset must be non-negative")
    if random_episode_phase:
        raise ValueError("phase-consistent event forbids random episode phase")
    if maximum_phase_offset_s != 0.0:
        raise ValueError("phase-consistent event requires zero command phase offset")

    event_end_s = stand_s + accelerate_s + cruise_s + decelerate_s
    required_rollout_s = event_end_s + terminal_hold_s
    if rollout_s + 1.0e-9 < required_rollout_s:
        raise ValueError(
            f"rollout {rollout_s:.3f}s does not contain required "
            f"{required_rollout_s:.3f}s event"
        )

    terminal_phase = (event_end_s / gait_cycle_s) % 1.0
    half_ds_width = double_support_fraction / 4.0
    distance_to_zero = min(terminal_phase, 1.0 - terminal_phase)
    distance_to_half = abs(terminal_phase - 0.5)
    in_double_support = min(distance_to_zero, distance_to_half) <= half_ds_width + 1.0e-9
    if not in_double_support:
        raise ValueError(
            f"terminal gait phase {terminal_phase:.6f} is outside double support"
        )

    return TransitionEventAudit(
        event_end_s=event_end_s,
        required_rollout_s=required_rollout_s,
        rollout_s=rollout_s,
        terminal_gait_phase=terminal_phase,
        terminal_in_double_support=True,
        random_episode_phase=False,
        maximum_phase_offset_s=0.0,
    )


__all__ = [
    "TransitionEventAudit",
    "audit_phase_consistent_event",
    "smooth_transition_speed",
    "stratified_phase_offsets",
]
