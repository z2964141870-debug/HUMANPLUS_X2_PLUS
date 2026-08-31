"""Deployable six-link goal contract for X2 garment teleoperation."""

from __future__ import annotations

from collections.abc import Sequence

import torch


X2_EXTREMITY_LINK_NAMES = (
    "base_link",
    "torso_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

X2_LOWER_POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)

X2_EXTERNAL_ARM_JOINT_NAMES = (
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

X2_LOCKED_HEAD_JOINT_NAMES = ("head_yaw_joint", "head_pitch_joint")


def resolve_named_indices(available: Sequence[str], required: Sequence[str]) -> tuple[int, ...]:
    """Resolve an ordered, unique name contract and fail on ambiguity."""

    available_names = tuple(str(name) for name in available)
    required_names = tuple(str(name) for name in required)
    if len(set(available_names)) != len(available_names):
        raise ValueError("available names must be unique")
    if len(set(required_names)) != len(required_names):
        raise ValueError("required names must be unique")
    missing = tuple(name for name in required_names if name not in available_names)
    if missing:
        raise ValueError(f"required names are unavailable: {missing}")
    return tuple(available_names.index(name) for name in required_names)


def _validate_pose_tensor(value: torch.Tensor, final_dim: int, name: str) -> None:
    if value.ndim < 3 or value.shape[-2] != len(X2_EXTREMITY_LINK_NAMES):
        raise ValueError(
            f"{name} must end in ({len(X2_EXTREMITY_LINK_NAMES)}, {final_dim}), "
            f"got {tuple(value.shape)}"
        )
    if value.shape[-1] != final_dim:
        raise ValueError(f"{name} final dimension must be {final_dim}")


def quat_conjugate(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    output = quaternion_wxyz.clone()
    output[..., 1:] = -output[..., 1:]
    return output


def quat_multiply(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    """Hamilton product for scalar-first unit quaternions."""

    lw, lx, ly, lz = left_wxyz.unbind(dim=-1)
    rw, rx, ry, rz = right_wxyz.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_rotate_inverse(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a scalar-first unit quaternion."""

    vector_quaternion = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return quat_multiply(
        quat_multiply(quat_conjugate(quaternion_wxyz), vector_quaternion),
        quaternion_wxyz,
    )[..., 1:]


def quat_to_rotation_vector(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Shortest rotation vector, invariant to quaternion sign."""

    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True).clamp_min(1.0e-12)
    quaternion = quaternion_wxyz / norm
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[..., :1].clamp_min(1.0e-12))
    scale = torch.where(vector_norm > 1.0e-7, angle / vector_norm, torch.full_like(vector_norm, 2.0))
    return vector * scale


def extremity_future_pose_error(
    current_position_w: torch.Tensor,
    current_quaternion_wxyz: torch.Tensor,
    target_position_w: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
    *,
    anchor_index: int = 0,
) -> torch.Tensor:
    """Encode future six-link SE(3) errors in the current pelvis frame.

    Current poses use shape ``(batch, 6, 3|4)``. Targets may either use the
    same shape or add a future-horizon dimension, ``(batch, horizon, 6, 3|4)``.
    The result always has shape ``(batch, horizon, 6, 6)`` and stores position
    error followed by shortest rotation-vector error. The target pelvis is not
    removed, so commanded pelvis translation, height, and orientation remain
    observable to the policy.
    """

    _validate_pose_tensor(current_position_w, 3, "current_position_w")
    _validate_pose_tensor(current_quaternion_wxyz, 4, "current_quaternion_wxyz")
    if current_position_w.ndim != 3 or current_quaternion_wxyz.ndim != 3:
        raise ValueError("current poses must have shape (batch, 6, 3|4)")
    if target_position_w.ndim == 3:
        target_position_w = target_position_w.unsqueeze(1)
    if target_quaternion_wxyz.ndim == 3:
        target_quaternion_wxyz = target_quaternion_wxyz.unsqueeze(1)
    _validate_pose_tensor(target_position_w, 3, "target_position_w")
    _validate_pose_tensor(target_quaternion_wxyz, 4, "target_quaternion_wxyz")
    if target_position_w.ndim != 4 or target_quaternion_wxyz.ndim != 4:
        raise ValueError("target poses must have shape (batch, horizon, 6, 3|4)")
    if current_position_w.shape[0] != target_position_w.shape[0]:
        raise ValueError("current and target batch sizes differ")
    if current_quaternion_wxyz.shape[:2] != current_position_w.shape[:2]:
        raise ValueError("current position and quaternion shapes differ")
    if target_quaternion_wxyz.shape[:3] != target_position_w.shape[:3]:
        raise ValueError("target position and quaternion shapes differ")
    if not 0 <= anchor_index < len(X2_EXTREMITY_LINK_NAMES):
        raise IndexError("anchor_index is outside the six-link contract")

    anchor_position = current_position_w[:, anchor_index : anchor_index + 1]
    anchor_quaternion = current_quaternion_wxyz[:, anchor_index : anchor_index + 1]
    current_local_position = quat_rotate_inverse(
        anchor_quaternion,
        current_position_w - anchor_position,
    )
    target_local_position = quat_rotate_inverse(
        anchor_quaternion.unsqueeze(1),
        target_position_w - anchor_position.unsqueeze(1),
    )
    position_error = target_local_position - current_local_position.unsqueeze(1)

    current_inverse = quat_conjugate(current_quaternion_wxyz).unsqueeze(1)
    rotation_error = quat_to_rotation_vector(
        quat_multiply(current_inverse, target_quaternion_wxyz)
    )
    return torch.cat((position_error, rotation_error), dim=-1)

