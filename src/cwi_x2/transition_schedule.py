"""Pure command schedule used by the X2 start/cruise/stop curriculum."""

from __future__ import annotations

import torch


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


__all__ = ["smooth_transition_speed", "stratified_phase_offsets"]
