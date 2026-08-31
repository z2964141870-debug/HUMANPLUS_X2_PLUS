"""Compact direct-action X2 teacher used by the feasibility screen.

The controller emits the exact normalized 15-D lower-body targets consumed by
the scratch X2 plant.  It contains no learned-policy weights.  A zero-mean gait
cycle is only a periodic prior; all command/posture correction is state
feedback in the direct action space.
"""

from __future__ import annotations

import torch


PARAMETER_NAMES = (
    "template_scale",
    "period_s",
    "hip_pitch_bias",
    "knee_bias",
    "ankle_pitch_bias",
    "hip_roll_antisym_bias",
    "ankle_roll_antisym_bias",
    "hip_yaw_antisym_bias",
    "waist_yaw_bias",
    "waist_pitch_bias",
    "vx_error_to_hip_pitch",
    "vx_error_to_ankle_pitch",
    "gravity_x_to_hip_pitch",
    "pitch_rate_to_ankle_pitch",
    "yaw_error_to_hip_yaw_antisym",
    "yaw_error_to_waist_yaw",
    "gravity_y_to_hip_roll_antisym",
    "roll_rate_to_ankle_roll_antisym",
    "yaw_error_to_hip_roll_antisym",
    "yaw_rate_to_hip_roll_antisym",
    "action_smoothing_alpha",
    "phase_offset",
)

LOWER_BOUNDS = torch.tensor(
    (
        0.00, 0.55, -0.35, -0.35, -0.35, -0.30, -0.30, -0.40,
        -0.40, -0.30, -1.50, -1.50, -1.50, -1.00, -2.00, -2.00,
        -1.50, -1.00, -1.00, -1.00, 0.15, 0.00,
    ),
    dtype=torch.float32,
)
UPPER_BOUNDS = torch.tensor(
    (
        0.35, 1.20, 0.35, 0.35, 0.35, 0.30, 0.30, 0.40,
        0.40, 0.30, 1.50, 1.50, 1.50, 1.00, 2.00, 2.00,
        1.50, 1.00, 1.00, 1.00, 1.00, 1.00,
    ),
    dtype=torch.float32,
)

INITIAL_MEAN = torch.tensor(
    (
        0.15, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.45, 0.0,
    ),
    dtype=torch.float32,
)
INITIAL_STD = 0.28 * (UPPER_BOUNDS - LOWER_BOUNDS)


def interpolate_cycle(cycle: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate a periodic ``[bins, 15]`` cycle per environment."""

    if cycle.ndim != 2 or cycle.shape[1] != 15:
        raise ValueError(f"cycle must be [bins, 15], got {tuple(cycle.shape)}")
    if phase.ndim != 1:
        raise ValueError(f"phase must be [envs], got {tuple(phase.shape)}")
    position = torch.remainder(phase, 1.0) * cycle.shape[0] - 0.5
    lower_unwrapped = torch.floor(position)
    blend = position - lower_unwrapped
    lower = lower_unwrapped.to(torch.long) % cycle.shape[0]
    upper = (lower + 1) % cycle.shape[0]
    return (1.0 - blend[:, None]) * cycle[lower] + blend[:, None] * cycle[upper]


def clamp_parameters(parameters: torch.Tensor) -> torch.Tensor:
    lower = LOWER_BOUNDS.to(parameters)
    upper = UPPER_BOUNDS.to(parameters)
    return torch.maximum(torch.minimum(parameters, upper), lower)


def direct_teacher_action(
    parameters: torch.Tensor,
    cycle_position_rad: torch.Tensor,
    action_scale_rad: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    command: torch.Tensor,
    previous_action: torch.Tensor,
    ramp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clipped and pre-clipped direct normalized lower-body actions."""

    envs = parameters.shape[0]
    expected = {
        "parameters": (envs, len(PARAMETER_NAMES)),
        "cycle_position_rad": (envs, 15),
        "root_lin_vel_b": (envs, 3),
        "root_ang_vel_b": (envs, 3),
        "projected_gravity_b": (envs, 3),
        "command": (envs, 3),
        "previous_action": (envs, 15),
        "ramp": (envs,),
    }
    actual = {
        "parameters": tuple(parameters.shape),
        "cycle_position_rad": tuple(cycle_position_rad.shape),
        "root_lin_vel_b": tuple(root_lin_vel_b.shape),
        "root_ang_vel_b": tuple(root_ang_vel_b.shape),
        "projected_gravity_b": tuple(projected_gravity_b.shape),
        "command": tuple(command.shape),
        "previous_action": tuple(previous_action.shape),
        "ramp": tuple(ramp.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must be {shape}, got {actual[name]}")
    if action_scale_rad.shape != (15,):
        raise ValueError("action_scale_rad must be [15]")

    p = clamp_parameters(parameters)
    target = p[:, 0:1] * cycle_position_rad / action_scale_rad[None]

    def symmetric(left: int, right: int, value: torch.Tensor) -> None:
        target[:, left] += value
        target[:, right] += value

    def antisymmetric(left: int, right: int, value: torch.Tensor) -> None:
        target[:, left] += value
        target[:, right] -= value

    symmetric(0, 6, p[:, 2])
    symmetric(3, 9, p[:, 3])
    symmetric(4, 10, p[:, 4])
    antisymmetric(1, 7, p[:, 5])
    antisymmetric(5, 11, p[:, 6])
    antisymmetric(2, 8, p[:, 7])
    target[:, 12] += p[:, 8]
    target[:, 13] += p[:, 9]

    vx_error = command[:, 0] - root_lin_vel_b[:, 0]
    yaw_error = command[:, 2] - root_ang_vel_b[:, 2]
    symmetric(0, 6, p[:, 10] * vx_error + p[:, 12] * projected_gravity_b[:, 0])
    symmetric(4, 10, p[:, 11] * vx_error + p[:, 13] * root_ang_vel_b[:, 1])
    antisymmetric(2, 8, p[:, 14] * yaw_error)
    target[:, 12] += p[:, 15] * yaw_error
    antisymmetric(1, 7, p[:, 16] * projected_gravity_b[:, 1] + p[:, 18] * yaw_error)
    antisymmetric(5, 11, p[:, 17] * root_ang_vel_b[:, 0] + p[:, 19] * root_ang_vel_b[:, 2])

    target *= ramp[:, None]
    alpha = p[:, 20:21]
    preclip = previous_action + alpha * (target - previous_action)
    return preclip.clamp(-1.0, 1.0), preclip


def update_cem(
    mean: torch.Tensor,
    std: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    *,
    elite_count: int,
    momentum: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded diagonal CEM update; returns mean, std, elite indices."""

    if not 1 <= elite_count <= candidates.shape[0]:
        raise ValueError("invalid elite_count")
    elite_ids = torch.topk(scores, k=elite_count, largest=True).indices
    elite = candidates[elite_ids]
    elite_mean = elite.mean(0)
    elite_std = elite.std(0, unbiased=False)
    new_mean = momentum * mean + (1.0 - momentum) * elite_mean
    floor = 0.03 * (UPPER_BOUNDS.to(std) - LOWER_BOUNDS.to(std))
    new_std = torch.maximum(momentum * std + (1.0 - momentum) * elite_std, floor)
    return clamp_parameters(new_mean), new_std, elite_ids
