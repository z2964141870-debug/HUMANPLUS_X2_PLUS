"""Pure X2 data contracts for a from-scratch BFM-Zero training run.

This module deliberately has no Isaac Sim dependency.  It is shared by the
offline expert-data converter and the future live X2 environment wrapper so
that observation and action semantics can be tested before launching Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


X2_JOINTS_31 = (
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
    "head_yaw_joint",
    "head_pitch_joint",
)

X2_LOWER_JOINTS_15 = X2_JOINTS_31[:15]

# Isaac/URDF articulation order used by the frozen Stage219 critic rollout.
# It is interleaved by kinematic tree and differs from the semantic BFM order.
X2_ISAAC_JOINTS_31 = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint",
)

# The immutable X2 articulation default pose used by the existing native
# environment.  The order is X2_JOINTS_31.
X2_DEFAULT_JOINT_POS_31 = np.asarray(
    (
        -0.248,
        0.0,
        0.0,
        0.5303,
        -0.2823,
        0.0,
        -0.248,
        0.0,
        0.0,
        0.5303,
        -0.2823,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
        0.0,
        -1.2,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
        0.0,
        -1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float32,
)

# BFM-Zero's actor is intrinsically bounded to [-1, 1].  These scales map the
# already-qualified four-clip X2 gait panel into that range without relying on
# the Stage219 gait template or any pretrained policy.
X2_SCRATCH_ACTION_SCALE_15 = np.asarray(
    (
        0.50,
        0.40,
        0.40,
        0.90,
        0.85,
        0.15,
        0.50,
        0.40,
        0.40,
        0.90,
        0.85,
        0.15,
        0.40,
        0.22,
        0.20,
    ),
    dtype=np.float32,
)

# The base linear velocity is deliberately part of the deployable state.  A
# periodic joint trajectory without it aliases forward locomotion, backward
# locomotion, and in-place slipping, which made the first scratch contract
# incapable of identifying the direction represented by an expert gait.
STATE_DIM = 71  # 31 q + 31 dq + 3 gravity + 3 base angular + 3 base linear velocity
ACTION_DIM = 15
COMMAND_DIM = 3
GAIT_PHASE_DIM = 4

# Raw IsaacLab costs used only by BFM's auxiliary critic.  The adapter removes
# the live reward weights before replay storage, so these scales are explicit
# and independent of the environment's locomotion reward sum.
X2_AUX_REWARD_SCALES = {
    "track_lin_vel_xy_exp": 1.0,
    # Match the live reward manager's integrated -200 * 0.02 terminal cost.
    "termination": -4.0,
    # Keep the direct scratch actor away from persistent [-1, 1] saturation.
    "action_magnitude_l2": -0.02,
    "dof_torques_l2": -1.0e-6,
    "action_rate_l2": -0.1,
    "flat_orientation_l2": -0.4,
    "dof_pos_limits": -10.0,
    "feet_slide": -2.0,
}

# A short safety-finetune can instead expose the environment's complete,
# already-integrated locomotion reward as one auxiliary target.  This keeps
# every posture/contact/velocity term in the original task contract together
# rather than trying another hand-picked subset of penalties.
X2_TASK_AUX_REWARD_SCALES = {
    "locomotion_total_reward": 1.0,
    "action_magnitude_l2": -0.005,
}

# The live adapter emits a stable superset so replay schemas stay compatible
# when a scratch checkpoint switches auxiliary objectives during a finetune.
X2_ALL_AUX_REWARD_NAMES = tuple(
    dict.fromkeys((*X2_AUX_REWARD_SCALES, *X2_TASK_AUX_REWARD_SCALES))
)


@dataclass(frozen=True)
class X2ScratchDataConfig:
    history_length: int = 4
    control_dt_s: float = 0.02
    max_action_abs: float = 1.0
    max_action_clip_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        if self.control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive")
        if not 0 <= self.max_action_clip_fraction <= 1:
            raise ValueError("max_action_clip_fraction must be in [0, 1]")

    @property
    def history_actor_dim(self) -> int:
        return self.history_length * (STATE_DIM + ACTION_DIM)


def _as_float32(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _quat_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    output = quaternion.copy()
    output[..., :3] *= -1.0
    return output


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("root_rot contains a zero quaternion")
    return quaternion / norm


def projected_gravity_from_xyzw(root_rot_xyzw: np.ndarray) -> np.ndarray:
    """Rotate world gravity (0, 0, -1) into the root frame."""

    quaternion = _normalize_quaternion_xyzw(_as_float32(root_rot_xyzw, name="root_rot", ndim=2))
    gravity_quat = np.zeros_like(quaternion)
    gravity_quat[..., 2] = -1.0
    rotated = _quat_multiply_xyzw(
        _quat_multiply_xyzw(_quat_conjugate_xyzw(quaternion), gravity_quat),
        quaternion,
    )
    return rotated[..., :3].astype(np.float32)


def angular_velocity_from_xyzw(root_rot_xyzw: np.ndarray, dt_s: float) -> np.ndarray:
    """Finite-difference root angular velocity, expressed in the root frame."""

    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    quaternion = _normalize_quaternion_xyzw(_as_float32(root_rot_xyzw, name="root_rot", ndim=2))
    relative = _quat_multiply_xyzw(_quat_conjugate_xyzw(quaternion[:-1]), quaternion[1:])
    relative = np.where(relative[:, 3:4] < 0.0, -relative, relative)
    vector = relative[:, :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 3:4], -1.0, 1.0))
    axis = np.divide(vector, vector_norm, out=np.zeros_like(vector), where=vector_norm > 1e-8)
    velocity = axis * (angle / dt_s)
    return np.concatenate((velocity, velocity[-1:]), axis=0).astype(np.float32)


def finite_difference(values: np.ndarray, dt_s: float) -> np.ndarray:
    values = _as_float32(values, name="values", ndim=2)
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    velocity = np.diff(values, axis=0) / dt_s
    return np.concatenate((velocity, velocity[-1:]), axis=0).astype(np.float32)


def normalized_lower_action(
    target_joint_pos_31: np.ndarray,
    joint_names: Sequence[str],
) -> np.ndarray:
    target = _as_float32(target_joint_pos_31, name="action", ndim=2)
    if tuple(joint_names) != X2_JOINTS_31:
        raise ValueError("X2 motion joint order mismatch")
    return (
        (target[:, :ACTION_DIM] - X2_DEFAULT_JOINT_POS_31[:ACTION_DIM])
        / X2_SCRATCH_ACTION_SCALE_15
    ).astype(np.float32)


def normalized_lower_physical_targets(target_joint_pos_15: np.ndarray) -> np.ndarray:
    """Map physical lower-body targets into the direct scratch action contract.

    Unlike :func:`normalized_lower_action`, this entry point accepts an already
    ordered 15-DOF target.  It is used for importing controller targets that
    were recorded after an external policy/template/action contract had been
    resolved.  No source policy action or gait-template residual is reused.
    """

    target = _as_float32(target_joint_pos_15, name="physical_lower_target", ndim=2)
    if target.shape[1] != ACTION_DIM:
        raise ValueError(
            f"physical lower target must have {ACTION_DIM} columns, got {target.shape}"
        )
    return (
        (target - X2_DEFAULT_JOINT_POS_31[:ACTION_DIM])
        / X2_SCRATCH_ACTION_SCALE_15
    ).astype(np.float32)


def root_quaternion_wxyz_from_projected_gravity_yaw(
    projected_gravity: np.ndarray,
    yaw_rad: np.ndarray,
) -> np.ndarray:
    """Reconstruct root orientation from deploy-observable gravity and yaw."""

    gravity = _as_float32(projected_gravity, name="projected_gravity", ndim=2)
    yaw = _as_float32(yaw_rad, name="yaw_rad", ndim=1)
    if gravity.shape[1] != 3 or gravity.shape[0] != yaw.shape[0]:
        raise ValueError("projected gravity/yaw batch shapes differ")
    norm = np.linalg.norm(gravity, axis=1)
    if not np.allclose(norm, 1.0, rtol=0.0, atol=2.0e-4):
        raise ValueError("projected gravity must be unit length")
    pitch = np.arcsin(np.clip(gravity[:, 0], -1.0, 1.0))
    roll = np.arctan2(-gravity[:, 1], -gravity[:, 2])
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quaternion = np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion.astype(np.float32)


def rotate_body_to_world_wxyz(vector_body: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate batched body-frame vectors into world coordinates."""

    vector = _as_float32(vector_body, name="vector_body", ndim=2)
    quaternion = _as_float32(quaternion_wxyz, name="quaternion_wxyz", ndim=2)
    if vector.shape[1] != 3 or quaternion.shape != (vector.shape[0], 4):
        raise ValueError("body vector/quaternion batch shapes differ")
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w = quaternion[:, 0:1]
    xyz = quaternion[:, 1:4]
    # Quaternion-vector rotation without constructing a dense matrix.
    twice_cross = 2.0 * np.cross(xyz, vector)
    return (vector + w * twice_cross + np.cross(xyz, twice_cross)).astype(np.float32)


def rotate_world_to_body_xyzw(
    vector_world: np.ndarray, quaternion_xyzw: np.ndarray
) -> np.ndarray:
    """Rotate batched world-frame vectors into the root/body frame."""

    vector = _as_float32(vector_world, name="vector_world", ndim=2)
    quaternion = _normalize_quaternion_xyzw(
        _as_float32(quaternion_xyzw, name="quaternion_xyzw", ndim=2)
    )
    if vector.shape[1] != 3 or quaternion.shape != (vector.shape[0], 4):
        raise ValueError("world vector/quaternion batch shapes differ")
    inverse_wxyz = np.concatenate(
        (quaternion[:, 3:4], -quaternion[:, :3]), axis=-1
    )
    return rotate_body_to_world_wxyz(vector, inverse_wxyz)


def stack_causal_history(state: np.ndarray, last_action: np.ndarray, length: int) -> np.ndarray:
    """Return current-and-past [state,last_action] frames with zero left padding."""

    state = _as_float32(state, name="state", ndim=2)
    last_action = _as_float32(last_action, name="last_action", ndim=2)
    if state.shape[0] != last_action.shape[0]:
        raise ValueError("state and last_action lengths differ")
    if length < 1:
        raise ValueError("history length must be positive")
    frame = np.concatenate((state, last_action), axis=-1)
    padded = np.pad(frame, ((length - 1, 0), (0, 0)))
    return np.stack([padded[index : index + len(frame)] for index in range(length)], axis=1).reshape(len(frame), -1)


def deployable_gait_phase_features(
    elapsed_steps: np.ndarray,
    command: np.ndarray,
    *,
    control_dt_s: float = 0.02,
    cycle_time_s: float = 0.8,
    double_support_fraction: float = 0.30,
) -> np.ndarray:
    """Reproduce Stage219's four deployable gait-clock features.

    This pure NumPy implementation intentionally uses only controller time and
    commanded planar velocity.  It is shared by offline dataset conversion and
    unit tests; the live adapter applies the identical equations on tensors.
    """

    steps = np.asarray(elapsed_steps)
    command = _as_float32(command, name="command")
    if steps.ndim not in (1, 2):
        raise ValueError("elapsed_steps must be rank 1 or 2")
    if command.shape != (*steps.shape, COMMAND_DIM):
        raise ValueError("elapsed_steps and command shapes differ")
    if control_dt_s <= 0.0 or cycle_time_s <= 0.0:
        raise ValueError("gait timing must be positive")
    if not 0.0 < double_support_fraction < 1.0:
        raise ValueError("double support fraction must lie in (0, 1)")

    moving = np.linalg.norm(command[..., :2], axis=-1) > 0.1
    phase = np.remainder(
        steps.astype(np.float32) * control_dt_s / cycle_time_s, 1.0
    )
    half_ds_width = double_support_fraction / 4.0
    right_swing = (phase >= half_ds_width) & (phase < 0.5 - half_ds_width)
    left_swing = (phase >= 0.5 + half_ds_width) & (phase < 1.0 - half_ds_width)
    desired_contact = np.stack((~left_swing, ~right_swing), axis=-1)
    desired_contact = np.where(
        moving[..., None], desired_contact, np.ones_like(desired_contact)
    )
    angle = 2.0 * np.pi * phase
    clock = np.stack((np.sin(angle), np.cos(angle)), axis=-1)
    clock *= moving[..., None]
    return np.concatenate(
        (clock, desired_contact.astype(np.float32)), axis=-1
    ).astype(np.float32)


def convert_x2_motion_entry(
    entry: Mapping[str, Any],
    *,
    motion_id: int,
    config: X2ScratchDataConfig = X2ScratchDataConfig(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one native X2 motion-cache entry into a BFM expert episode."""

    required = {
        "dof",
        "action",
        "root_rot",
        "root_trans_offset",
        "fps",
        "joint_names_mujoco",
    }
    missing = sorted(required.difference(entry))
    if missing:
        raise ValueError(f"X2 motion entry misses fields: {missing}")
    joint_names = tuple(str(name) for name in entry["joint_names_mujoco"])
    if joint_names != X2_JOINTS_31:
        raise ValueError("X2 motion joint order mismatch")
    dof = _as_float32(entry["dof"], name="dof", ndim=2)
    action_target = _as_float32(entry["action"], name="action", ndim=2)
    root_rot = _as_float32(entry["root_rot"], name="root_rot", ndim=2)
    root_trans = _as_float32(
        entry["root_trans_offset"], name="root_trans_offset", ndim=2
    )
    if dof.shape[1] != len(X2_JOINTS_31) or action_target.shape != dof.shape:
        raise ValueError("X2 dof/action shape mismatch")
    if root_rot.shape != (len(dof), 4):
        raise ValueError("X2 root_rot shape mismatch")
    if root_trans.shape != (len(dof), 3):
        raise ValueError("X2 root_trans_offset shape mismatch")
    fps = float(entry["fps"])
    if not np.isclose(fps * config.control_dt_s, 1.0, atol=1e-6):
        raise ValueError(f"motion fps {fps} does not match control_dt_s {config.control_dt_s}")

    dof_rel = dof - X2_DEFAULT_JOINT_POS_31
    dof_vel = finite_difference(dof, config.control_dt_s)
    projected_gravity = projected_gravity_from_xyzw(root_rot)
    base_ang_vel = angular_velocity_from_xyzw(root_rot, config.control_dt_s)
    root_lin_vel_world = finite_difference(root_trans, config.control_dt_s)
    base_lin_vel = rotate_world_to_body_xyzw(root_lin_vel_world, root_rot)
    state = np.concatenate(
        (dof_rel, dof_vel, projected_gravity, base_ang_vel, base_lin_vel), axis=-1
    ).astype(np.float32)
    if state.shape != (len(dof), STATE_DIM):
        raise AssertionError(f"unexpected X2 state shape: {state.shape}")

    action_unclipped = normalized_lower_action(action_target, joint_names)
    clip_mask = np.abs(action_unclipped) > config.max_action_abs
    clip_fraction = float(clip_mask.mean())
    if clip_fraction > config.max_action_clip_fraction:
        raise ValueError(
            f"expert action clip fraction {clip_fraction:.6f} exceeds "
            f"{config.max_action_clip_fraction:.6f}"
        )
    action = np.clip(action_unclipped, -config.max_action_abs, config.max_action_abs).astype(np.float32)
    last_action = np.concatenate((np.zeros_like(action[:1]), action[:-1]), axis=0)
    history_actor = stack_causal_history(state, last_action, config.history_length).astype(np.float32)
    truncated = np.zeros((len(dof), 1), dtype=np.bool_)
    truncated[-1] = True
    terminated = np.zeros_like(truncated)

    episode = {
        "observation": {
            "state": state,
            # The first X2 scratch run intentionally has no asymmetric critic
            # state.  Keeping this key equal to state preserves the BFM API
            # without leaking unavailable simulation-only features.
            "privileged_state": state.copy(),
            "last_action": last_action,
            "history_actor": history_actor,
        },
        "action": action,
        "terminated": terminated,
        "truncated": truncated,
        "motion_id": np.full((len(dof), 1), motion_id, dtype=np.int64),
    }
    audit = {
        "frames": len(dof),
        "fps": fps,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "history_actor_dim": config.history_actor_dim,
        "base_linear_velocity_mean": base_lin_vel.mean(axis=0).tolist(),
        "action_abs_max_before_clip": float(np.abs(action_unclipped).max()),
        "action_clip_fraction": clip_fraction,
        "action_clip_frame_fraction": float(np.any(clip_mask, axis=1).mean()),
        "finite": all(
            np.isfinite(value).all()
            for value in (state, action, last_action, history_actor, base_lin_vel)
        ),
    }
    return episode, audit


def convert_stage219_critic_rollout(
    critic_observation: np.ndarray,
    *,
    direct_action_labels: np.ndarray | None = None,
    include_command_phase: bool = False,
    motion_id_offset: int = 0,
    config: X2ScratchDataConfig = X2ScratchDataConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert a closed-loop Stage219 93-D rollout into BFM expert episodes.

    Stage219's uncorrupted critic observation is ordered as base linear
    velocity (3), base angular velocity (3), gravity (3), command (3), joint
    position/velocity (31+31), previous action (15), and gait phase (4).
    The legacy contract uses only physical proprioception and previous action.
    The optional phase-conditioned contract also exposes the same deployable
    command and gait-clock suffix already consumed by Stage219.
    """

    rollout = _as_float32(
        critic_observation, name="critic_observation", ndim=3
    )
    if rollout.shape[2] != 93 or rollout.shape[0] < 9 or rollout.shape[1] < 1:
        raise ValueError("Stage219 critic rollout must be [T,N,93]")
    base_lin_vel = rollout[..., 0:3]
    base_ang_vel = rollout[..., 3:6]
    projected_gravity = rollout[..., 6:9]
    command = rollout[..., 9:12]
    gait_phase = rollout[..., 89:93]
    semantic_ids = np.asarray(
        [X2_ISAAC_JOINTS_31.index(name) for name in X2_JOINTS_31],
        dtype=np.int64,
    )
    joint_pos_rel = rollout[..., 12:43][..., semantic_ids]
    joint_vel = rollout[..., 43:74][..., semantic_ids]
    stage219_previous_action = rollout[..., 74:89]
    if direct_action_labels is None:
        previous_action = stage219_previous_action
        executed_action = np.concatenate(
            (previous_action[1:], previous_action[-1:]), axis=0
        ).astype(np.float32)
    else:
        labels = _as_float32(
            direct_action_labels, name="direct_action_labels", ndim=3
        )
        expected_shape = (rollout.shape[0] - 1, rollout.shape[1], ACTION_DIM)
        if labels.shape != expected_shape:
            raise ValueError(
                f"direct action labels must be {expected_shape}, got {labels.shape}"
            )
        previous_action = np.zeros(
            (rollout.shape[0], rollout.shape[1], ACTION_DIM), dtype=np.float32
        )
        previous_action[1:] = labels
        executed_action = np.concatenate((labels, labels[-1:]), axis=0)
    state = np.concatenate(
        (
            joint_pos_rel,
            joint_vel,
            projected_gravity,
            base_ang_vel,
            base_lin_vel,
        ),
        axis=-1,
    ).astype(np.float32)
    if state.shape[2] != STATE_DIM:
        raise AssertionError(f"unexpected closed-loop state shape: {state.shape}")
    if np.abs(previous_action).max() > config.max_action_abs + 1e-6:
        raise ValueError("Stage219 previous action exceeds the BFM action bounds")

    # Without explicit direct labels, the action that produced observation
    # t+1 is Stage219's raw residual stored at t+1.  With labels, both the
    # observation history and action field use the actual template-free BFM
    # direct-action contract.
    episodes: list[dict[str, Any]] = []
    for env_id in range(rollout.shape[1]):
        lane_state = state[:, env_id]
        lane_last_action = previous_action[:, env_id].astype(np.float32)
        history_actor = stack_causal_history(
            lane_state, lane_last_action, config.history_length
        ).astype(np.float32)
        truncated = np.zeros((rollout.shape[0], 1), dtype=np.bool_)
        truncated[-1] = True
        observation = {
            "state": lane_state.copy(),
            "privileged_state": lane_state.copy(),
            "last_action": lane_last_action.copy(),
            "history_actor": history_actor,
        }
        if include_command_phase:
            observation["command"] = command[:, env_id].copy()
            observation["gait_phase"] = gait_phase[:, env_id].copy()
        episodes.append(
            {
                "observation": observation,
                "action": executed_action[:, env_id].copy(),
                "terminated": np.zeros_like(truncated),
                "truncated": truncated,
                "motion_id": np.full(
                    (rollout.shape[0], 1),
                    motion_id_offset + env_id,
                    dtype=np.int64,
                ),
            }
        )
    return episodes, {
        "schema": "bfm_zero_x2_stage219_closed_loop_dataset_audit_v1",
        "episode_count": len(episodes),
        "frames_per_episode": int(rollout.shape[0]),
        "total_frames": int(rollout.shape[0] * rollout.shape[1]),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "all_finite": bool(
            np.isfinite(state).all()
            and np.isfinite(previous_action).all()
            and np.isfinite(executed_action).all()
        ),
        "forward_velocity_mean_mps": float(base_lin_vel[..., 0].mean()),
        "forward_velocity_std_mps": float(base_lin_vel[..., 0].std()),
        "previous_action_abs_max": float(np.abs(previous_action).max()),
        "direct_action_observation_contract": direct_action_labels is not None,
        "command_phase_observation_contract": include_command_phase,
        "command_dim": COMMAND_DIM if include_command_phase else 0,
        "gait_phase_dim": GAIT_PHASE_DIM if include_command_phase else 0,
        "source_joint_order": list(X2_ISAAC_JOINTS_31),
        "target_joint_order": list(X2_JOINTS_31),
        "joint_order_reindexed": X2_ISAAC_JOINTS_31 != X2_JOINTS_31,
    }


def reconstruct_stage219_direct_action_labels(
    critic_observation: np.ndarray,
    q_cycle_zero_mean_rad: np.ndarray,
    action_scale_rad: np.ndarray,
    *,
    template_scale: float = 0.15,
    period_s: float = 0.8,
    control_dt_s: float = 0.02,
) -> np.ndarray:
    """Reconstruct the normalized direct target executed by Stage219.

    The rollout observation at t+1 stores the residual action issued at t.
    Stage219 added its deterministic gait template in normalized coordinates
    before clipping.  Its normalized coordinates are *not* the BFM scratch
    plant's normalized coordinates: the two action terms use different
    per-joint physical scales.  Reconstruct the Stage219 combined command in
    radians first, then normalize it by the scratch plant scale.
    """

    rollout = _as_float32(
        critic_observation, name="critic_observation", ndim=3
    )
    cycle = _as_float32(
        q_cycle_zero_mean_rad, name="q_cycle_zero_mean_rad", ndim=2
    )
    scale = _as_float32(action_scale_rad, name="action_scale_rad", ndim=1)
    if rollout.shape[2] != 93 or cycle.shape[1] != ACTION_DIM:
        raise ValueError("invalid Stage219 rollout/template dimensions")
    if scale.shape != (ACTION_DIM,) or np.any(scale <= 0):
        raise ValueError("invalid Stage219 action scale")
    if template_scale < 0 or period_s <= 0 or control_dt_s <= 0:
        raise ValueError("invalid Stage219 template timing/scaling")

    step = np.arange(rollout.shape[0] - 1, dtype=np.float32)
    phase = np.remainder(step * control_dt_s / period_s, 1.0)
    phase_position = phase * cycle.shape[0] - 0.5
    lower_unwrapped = np.floor(phase_position)
    blend = phase_position - lower_unwrapped
    lower = lower_unwrapped.astype(np.int64) % cycle.shape[0]
    upper = (lower + 1) % cycle.shape[0]
    q_bias = (
        (1.0 - blend[:, None]) * cycle[lower]
        + blend[:, None] * cycle[upper]
    )
    normalized_bias = template_scale * q_bias / scale[None]
    residual_action = rollout[1:, :, 74:89]
    stage219_combined = np.clip(
        residual_action + normalized_bias[:, None, :], -1.0, 1.0
    )
    labels = np.clip(
        stage219_combined * scale[None, None, :]
        / X2_SCRATCH_ACTION_SCALE_15[None, None, :],
        -1.0,
        1.0,
    ).astype(np.float32)
    if not np.isfinite(labels).all():
        raise ValueError("Stage219 reconstructed labels are non-finite")
    return labels


def load_x2_motion_files(
    paths: Iterable[str | Path],
    *,
    config: X2ScratchDataConfig = X2ScratchDataConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and convert one-entry native X2 joblib motion files."""

    import joblib

    episodes: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for motion_id, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser().resolve()
        payload = joblib.load(path)
        if not isinstance(payload, Mapping) or len(payload) != 1:
            raise ValueError(f"{path} must contain exactly one motion entry")
        name, entry = next(iter(payload.items()))
        episode, audit = convert_x2_motion_entry(entry, motion_id=motion_id, config=config)
        episodes.append(episode)
        files.append({"path": str(path), "motion": str(name), **audit})
    if not episodes:
        raise ValueError("at least one X2 motion file is required")
    return episodes, {
        "schema": "bfm_zero_x2_scratch_dataset_audit_v1",
        "config": {
            "history_length": config.history_length,
            "control_dt_s": config.control_dt_s,
            "max_action_abs": config.max_action_abs,
            "max_action_clip_fraction": config.max_action_clip_fraction,
        },
        "motion_count": len(episodes),
        "total_frames": sum(item["frames"] for item in files),
        "files": files,
        "all_finite": all(item["finite"] for item in files),
    }


def build_x2_scratch_replay_buffers(
    episodes: Sequence[dict[str, Any]],
    *,
    seq_length: int = 8,
    z_dim: int = 64,
    seed: int = 770002,
    device: str = "cpu",
):
    """Build the official expert slicer plus a minimal scratch seed replay.

    The seed replay is made only from expert transitions and random normalized
    latent goals.  It exists to validate the complete BFM update path before
    an online X2 collection loop is allowed to run; it is not sufficient data
    for a locomotion training campaign.
    """

    if not episodes:
        raise ValueError("at least one X2 expert episode is required")
    if seq_length < 2 or z_dim < 1:
        raise ValueError("invalid sequence or latent dimension")

    from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
    from humanoidverse.agents.buffers.transition import DictBuffer

    expert = TrajectoryDictBuffer(
        episodes=list(episodes),
        device=device,
        seq_length=seq_length,
        output_key_t=["observation"],
        output_key_tp1=["observation"],
        end_key="truncated",
    )

    rng = np.random.default_rng(seed)
    transitions: dict[str, Any] = {
        "observation": {},
        "action": [],
        "z": [],
        "next": {"observation": {}, "terminated": []},
    }
    observation_keys = tuple(episodes[0]["observation"])
    for key in observation_keys:
        transitions["observation"][key] = []
        transitions["next"]["observation"][key] = []

    for episode in episodes:
        if tuple(episode["observation"]) != observation_keys:
            raise ValueError("expert episode observation keys differ")
        length = int(episode["action"].shape[0])
        if length <= seq_length:
            raise ValueError("expert episode is too short for the BFM sequence")
        for key in observation_keys:
            transitions["observation"][key].append(episode["observation"][key][:-1])
            transitions["next"]["observation"][key].append(episode["observation"][key][1:])
        transitions["action"].append(episode["action"][:-1])
        transitions["next"]["terminated"].append(episode["terminated"][1:])
        raw_z = rng.standard_normal((length - 1, z_dim), dtype=np.float32)
        norm = np.linalg.norm(raw_z, axis=-1, keepdims=True)
        transitions["z"].append(
            (np.sqrt(np.float32(z_dim)) * raw_z / np.maximum(norm, 1.0e-8)).astype(np.float32)
        )

    for key in observation_keys:
        transitions["observation"][key] = np.concatenate(
            transitions["observation"][key], axis=0
        )
        transitions["next"]["observation"][key] = np.concatenate(
            transitions["next"]["observation"][key], axis=0
        )
    transitions["action"] = np.concatenate(transitions["action"], axis=0)
    transitions["z"] = np.concatenate(transitions["z"], axis=0)
    transitions["next"]["terminated"] = np.concatenate(
        transitions["next"]["terminated"], axis=0
    )

    transition_count = int(transitions["action"].shape[0])
    train = DictBuffer(capacity=transition_count, device=device)
    train.extend(transitions)
    return {
        "expert_slicer": expert,
        "train": train,
        "audit": {
            "expert_frames": len(expert),
            "seed_transitions": len(train),
            "motion_count": len(episodes),
            "seq_length": seq_length,
            "z_dim": z_dim,
            "device": device,
        },
    }
