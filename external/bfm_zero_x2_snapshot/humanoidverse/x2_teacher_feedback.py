"""Pure state-feedback command utilities for the X2 scratch teacher screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class FeedbackCommandSpec:
    """One preregistered command-feedback candidate."""

    name: str
    role: str
    ramp_s: float
    kp_vx: float
    kp_yaw: float
    max_vx_mps: float = 0.65
    max_abs_yaw_radps: float = 0.80

    def __post_init__(self) -> None:
        if not self.name or not self.role:
            raise ValueError("feedback candidate name and role are required")
        if self.ramp_s <= 0.0:
            raise ValueError("ramp_s must be positive")
        if min(self.kp_vx, self.kp_yaw) < 0.0:
            raise ValueError("feedback gains must be non-negative")
        if self.max_vx_mps <= 0.0 or self.max_abs_yaw_radps <= 0.0:
            raise ValueError("command bounds must be positive")

    def record(self) -> dict[str, str | float]:
        return asdict(self)


def low_speed_candidates() -> tuple[FeedbackCommandSpec, ...]:
    """Frozen 4x4 ramp/forward-feedback grid for the 0.20 m/s role."""

    return tuple(
        FeedbackCommandSpec(
            name=f"low_ramp{ramp:g}_kp{gain:g}",
            role="vx_0p20",
            ramp_s=ramp,
            kp_vx=gain,
            kp_yaw=0.0,
            max_vx_mps=0.45,
        )
        for ramp in (0.5, 1.0, 1.5, 2.0)
        for gain in (0.0, 0.5, 1.0, 1.5)
    )


def right_turn_candidates() -> tuple[FeedbackCommandSpec, ...]:
    """Frozen 4x4 ramp/yaw-feedback grid for the -0.30 rad/s role."""

    return tuple(
        FeedbackCommandSpec(
            name=f"right_ramp{ramp:g}_kp{gain:g}",
            role="turn_right",
            ramp_s=ramp,
            kp_vx=0.5,
            kp_yaw=gain,
        )
        for ramp in (0.5, 1.0, 1.5, 2.0)
        for gain in (0.5, 1.0, 2.0, 3.0)
    )


def state_feedback_command(
    desired: torch.Tensor,
    measured: torch.Tensor,
    elapsed_s: torch.Tensor,
    ramp_s: torch.Tensor,
    kp_vx: torch.Tensor,
    kp_yaw: torch.Tensor,
    max_vx_mps: torch.Tensor,
    max_abs_yaw_radps: torch.Tensor,
) -> torch.Tensor:
    """Return bounded ramped P-feedback commands for batched X2 lanes.

    ``desired`` and ``measured`` are ``[N, 3]`` body-frame ``vx, vy, wz``.
    All parameter tensors are ``[N]``.  The lateral command is deliberately
    open-loop zero in this first feasibility screen.
    """

    if desired.ndim != 2 or desired.shape[-1] != 3 or measured.shape != desired.shape:
        raise ValueError("desired and measured commands must have shape [N, 3]")
    lane_count = desired.shape[0]
    parameters = (elapsed_s, ramp_s, kp_vx, kp_yaw, max_vx_mps, max_abs_yaw_radps)
    if any(value.shape != (lane_count,) for value in parameters):
        raise ValueError("feedback parameter tensors must have shape [N]")
    tensors = (desired, measured, *parameters)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("state-feedback command input is non-finite")
    if torch.any(ramp_s <= 0.0) or torch.any(max_vx_mps <= 0.0) or torch.any(
        max_abs_yaw_radps <= 0.0
    ):
        raise ValueError("ramp and bounds must be positive")
    if torch.any(kp_vx < 0.0) or torch.any(kp_yaw < 0.0):
        raise ValueError("feedback gains must be non-negative")

    ramp = torch.clamp(elapsed_s / ramp_s, 0.0, 1.0)
    target = desired * ramp[:, None]
    command = torch.zeros_like(desired)
    command[:, 0] = target[:, 0] + kp_vx * (target[:, 0] - measured[:, 0])
    command[:, 2] = target[:, 2] + kp_yaw * (target[:, 2] - measured[:, 2])
    command[:, 0] = torch.clamp(command[:, 0], min=0.0)
    command[:, 0] = torch.minimum(command[:, 0], max_vx_mps)
    command[:, 2] = torch.maximum(
        torch.minimum(command[:, 2], max_abs_yaw_radps), -max_abs_yaw_radps
    )
    return command


__all__ = [
    "FeedbackCommandSpec",
    "low_speed_candidates",
    "right_turn_candidates",
    "state_feedback_command",
]
