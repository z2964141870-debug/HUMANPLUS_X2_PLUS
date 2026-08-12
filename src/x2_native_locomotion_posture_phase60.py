"""Directional posture and actual-support rewards for X2 native locomotion.

The posture term is deliberately one-sided: it penalizes excessive backward
root pitch while leaving modest forward pitch unpenalized.  The support term
uses live PhysX body masses and ground-filtered foot contacts; it does not use
reference contact labels or constrain the waist joint directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving_mask(
    env: "ManagerBasedRLEnv", command_name: str, command_threshold_mps: float
) -> torch.Tensor:
    if not math.isfinite(command_threshold_mps) or command_threshold_mps < 0.0:
        raise ValueError("command threshold must be finite and nonnegative")
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :2], dim=-1) > command_threshold_mps


def signed_root_pitch_rad(robot) -> torch.Tensor:
    """Return X2 signed root pitch from body-frame projected gravity.

    X2 uses +X forward.  Under the frozen observation convention, negative
    pitch is backward lean and positive pitch is forward lean.
    """

    gravity = robot.data.projected_gravity_b
    if gravity.ndim != 2 or gravity.shape[-1] != 3:
        raise RuntimeError("projected gravity must have shape (envs, 3)")
    return torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))


def signed_backward_pitch_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    tolerance_rad: float,
    normalization_rad: float,
    command_threshold_mps: float = 0.10,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Penalize only backward pitch beyond ``-tolerance_rad`` while moving."""

    if not math.isfinite(tolerance_rad) or tolerance_rad < 0.0:
        raise ValueError("pitch tolerance must be finite and nonnegative")
    if not math.isfinite(normalization_rad) or normalization_rad <= 0.0:
        raise ValueError("pitch normalization must be finite and positive")
    pitch = signed_root_pitch_rad(env.scene[asset_name])
    backward_excess = torch.clamp(-pitch - tolerance_rad, min=0.0)
    penalty = torch.square(backward_excess / normalization_rad)
    return penalty * _moving_mask(env, command_name, command_threshold_mps).to(penalty.dtype)


def _ground_vertical_force(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    forces_w = env.scene[sensor_name].data.force_matrix_w
    if forces_w is None:
        raise RuntimeError(f"ground contact sensor {sensor_name!r} has no force matrix")
    return forces_w[..., 2].abs().reshape(forces_w.shape[0], -1).amax(dim=-1)


def _yaw_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise RuntimeError("quaternion must have shape (envs, 4) in wxyz order")
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _world_xy_to_heading(delta_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * delta_xy[:, 0] + sine * delta_xy[:, 1],
            -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1],
        ),
        dim=-1,
    )


def _whole_body_com_w(robot) -> torch.Tensor:
    positions = robot.data.body_com_pos_w
    masses = robot.root_physx_view.get_masses()
    if not torch.is_tensor(masses):
        masses = torch.as_tensor(masses, device=positions.device, dtype=positions.dtype)
    else:
        masses = masses.to(device=positions.device, dtype=positions.dtype)
    if tuple(masses.shape) != tuple(positions.shape[:2]):
        raise RuntimeError(
            f"body mass/COM shapes differ: masses={tuple(masses.shape)}, "
            f"positions={tuple(positions.shape)}"
        )
    if not torch.isfinite(masses).all() or torch.any(masses <= 0.0):
        raise RuntimeError("live body masses must be finite and positive")
    return (positions * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(dim=1, keepdim=True)


def actual_support_com_outside_distance(
    env: "ManagerBasedRLEnv",
    force_threshold_n: float,
    left_sensor_name: str = "left_foot_ground_contact",
    right_sensor_name: str = "right_foot_ground_contact",
    asset_name: str = "robot",
    left_foot_body_name: str = "left_ankle_roll_link",
    right_foot_body_name: str = "right_ankle_roll_link",
    support_x_min_m: float = -0.070,
    support_x_max_m: float = 0.144,
    support_y_min_m: float = -0.065,
    support_y_max_m: float = 0.065,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return COM distance outside the actual contact support and contact count.

    Single support uses the stance-foot heading and its rectangular sole.  In
    double support, both sole rectangles are enclosed in the root-heading
    frame.  This is a conservative horizontal support envelope, not a claim
    about COP or full dynamic feasibility.  Flight frames return zero distance
    and are handled by the existing contact/air-time rewards.
    """

    if not math.isfinite(force_threshold_n) or force_threshold_n <= 0.0:
        raise ValueError("contact force threshold must be finite and positive")
    if support_x_max_m <= support_x_min_m or support_y_max_m <= support_y_min_m:
        raise ValueError("support bounds must have min < max")

    robot = env.scene[asset_name]
    com = _whole_body_com_w(robot)
    body_ids = robot.find_bodies(
        [left_foot_body_name, right_foot_body_name], preserve_order=True
    )[0]
    if len(body_ids) != 2:
        raise RuntimeError("actual support reward requires exactly two foot bodies")
    foot_pos = robot.data.body_pos_w[:, body_ids]
    foot_quat = robot.data.body_quat_w[:, body_ids]
    force = torch.stack(
        (
            _ground_vertical_force(env, left_sensor_name),
            _ground_vertical_force(env, right_sensor_name),
        ),
        dim=-1,
    )
    contact = force > force_threshold_n
    contact_count = contact.to(torch.int64).sum(dim=-1)
    outside = torch.zeros(com.shape[0], device=com.device, dtype=com.dtype)

    single = contact_count == 1
    if torch.any(single):
        side = torch.argmax(contact.to(torch.int64), dim=-1)
        env_ids = torch.arange(com.shape[0], device=com.device)
        stance_pos = foot_pos[env_ids, side]
        stance_yaw = _yaw_wxyz(foot_quat[env_ids, side])
        local = _world_xy_to_heading(com[:, :2] - stance_pos[:, :2], stance_yaw)
        dx = torch.maximum(
            (support_x_min_m - local[:, 0]).clamp_min(0.0),
            (local[:, 0] - support_x_max_m).clamp_min(0.0),
        )
        dy = torch.maximum(
            (support_y_min_m - local[:, 1]).clamp_min(0.0),
            (local[:, 1] - support_y_max_m).clamp_min(0.0),
        )
        outside = torch.where(single, torch.hypot(dx, dy), outside)

    double = contact_count == 2
    if torch.any(double):
        root_yaw = _yaw_wxyz(robot.data.root_quat_w)
        left = _world_xy_to_heading(foot_pos[:, 0, :2] - com[:, :2], root_yaw)
        right = _world_xy_to_heading(foot_pos[:, 1, :2] - com[:, :2], root_yaw)
        centers = torch.stack((left, right), dim=1)
        # COM is the origin in this frame.  The distance below is therefore
        # its overshoot beyond the envelope of the two contact soles.
        lower_x = centers[:, :, 0].amin(dim=1) + support_x_min_m
        upper_x = centers[:, :, 0].amax(dim=1) + support_x_max_m
        lower_y = centers[:, :, 1].amin(dim=1) + support_y_min_m
        upper_y = centers[:, :, 1].amax(dim=1) + support_y_max_m
        dx = torch.maximum(lower_x.clamp_min(0.0), (-upper_x).clamp_min(0.0))
        dy = torch.maximum(lower_y.clamp_min(0.0), (-upper_y).clamp_min(0.0))
        outside = torch.where(double, torch.hypot(dx, dy), outside)

    return outside, contact_count


def actual_support_com_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    normalization_m: float,
    force_threshold_n: float = 10.0,
    command_threshold_mps: float = 0.10,
    left_sensor_name: str = "left_foot_ground_contact",
    right_sensor_name: str = "right_foot_ground_contact",
    asset_name: str = "robot",
    left_foot_body_name: str = "left_ankle_roll_link",
    right_foot_body_name: str = "right_ankle_roll_link",
    support_x_min_m: float = -0.070,
    support_x_max_m: float = 0.144,
    support_y_min_m: float = -0.065,
    support_y_max_m: float = 0.065,
) -> torch.Tensor:
    """Squared normalized COM-support overshoot during actual ground support."""

    if not math.isfinite(normalization_m) or normalization_m <= 0.0:
        raise ValueError("COM support normalization must be finite and positive")
    outside, contact_count = actual_support_com_outside_distance(
        env,
        force_threshold_n=force_threshold_n,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        asset_name=asset_name,
        left_foot_body_name=left_foot_body_name,
        right_foot_body_name=right_foot_body_name,
        support_x_min_m=support_x_min_m,
        support_x_max_m=support_x_max_m,
        support_y_min_m=support_y_min_m,
        support_y_max_m=support_y_max_m,
    )
    active = (contact_count > 0) & _moving_mask(env, command_name, command_threshold_mps)
    return torch.square(outside / normalization_m) * active.to(outside.dtype)
