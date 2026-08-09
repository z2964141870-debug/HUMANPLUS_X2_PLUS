"""Pure sagittal-reflection helpers for the X2 locomotion actor.

This module deliberately has no ROS or ONNX dependencies.  It defines the
same 93-D observation reflection used by the IsaacLab training contract and a
bounded equivariance projection that can be tested before official MuJoCo is
started.
"""

from __future__ import annotations

import numpy as np


def mirror_joint_sign(name: str) -> float:
    return -1.0 if any(axis in name for axis in ("_roll_", "_yaw_")) else 1.0


def mirror_named_vector(
    values: np.ndarray,
    names: tuple[str, ...],
) -> np.ndarray:
    """Reflect a named joint vector across the sagittal plane."""
    values = np.asarray(values)
    if values.shape != (len(names),):
        raise ValueError(f"expected {(len(names),)}, got {values.shape}")
    index = {name: i for i, name in enumerate(names)}
    mirrored = np.empty_like(values)
    for output_index, name in enumerate(names):
        if name.startswith("left_"):
            source_name = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            source_name = "left_" + name[len("right_") :]
        else:
            source_name = name
        mirrored[output_index] = mirror_joint_sign(name) * values[index[source_name]]
    return mirrored


def mirror_stage208_observation(
    observation: np.ndarray,
    *,
    isaac_joint_names: tuple[str, ...],
    action_joint_names: tuple[str, ...],
) -> np.ndarray:
    """Mirror the exact Stage208 93-D observation contract.

    Layout: body linear velocity (3), angular velocity (3), gravity (3),
    command (3), 31 q, 31 dq, previous 15-D action, and 4 phase/contact
    features.  Both sin/cos gait-clock terms change sign and the two contact
    indicators swap, matching the authoritative training transform.
    """
    observation = np.asarray(observation)
    if observation.shape != (93,):
        raise ValueError(f"expected (93,), got {observation.shape}")
    mirrored = observation.copy()
    mirrored[0:3] *= np.asarray([1.0, -1.0, 1.0], dtype=observation.dtype)
    mirrored[3:6] *= np.asarray([-1.0, 1.0, -1.0], dtype=observation.dtype)
    mirrored[6:9] *= np.asarray([1.0, -1.0, 1.0], dtype=observation.dtype)
    mirrored[9:12] *= np.asarray([1.0, -1.0, -1.0], dtype=observation.dtype)
    mirrored[12:43] = mirror_named_vector(observation[12:43], isaac_joint_names)
    mirrored[43:74] = mirror_named_vector(observation[43:74], isaac_joint_names)
    mirrored[74:89] = mirror_named_vector(observation[74:89], action_joint_names)
    mirrored[89:91] = -observation[89:91]
    mirrored[91:93] = observation[[92, 91]]
    return mirrored


def action_projection_mask(
    names: tuple[str, ...],
    mode: str,
) -> np.ndarray:
    if mode == "all":
        return np.ones(len(names), dtype=bool)
    if mode == "roll_yaw":
        return np.asarray(
            ["_roll_" in name or "_yaw_" in name for name in names],
            dtype=bool,
        )
    raise ValueError(f"unsupported actor symmetry mask: {mode}")


def project_actor_action(
    raw_action: np.ndarray,
    mirrored_observation_action: np.ndarray,
    *,
    action_joint_names: tuple[str, ...],
    alpha: float,
    mask_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project selected action components toward sagittal equivariance.

    ``mirrored_observation_action`` is the actor output for ``M(obs)``.  It is
    reflected back before averaging.  Alpha zero is an exact no-op; alpha one
    is the group average on the selected coordinates.
    """
    raw_action = np.asarray(raw_action)
    mirrored_observation_action = np.asarray(mirrored_observation_action)
    expected = (len(action_joint_names),)
    if raw_action.shape != expected or mirrored_observation_action.shape != expected:
        raise ValueError(
            f"expected two {expected} actions, got {raw_action.shape} and "
            f"{mirrored_observation_action.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    mirrored_back = mirror_named_vector(
        mirrored_observation_action,
        action_joint_names,
    )
    equivariance_error = raw_action - mirrored_back
    group_average = 0.5 * (raw_action + mirrored_back)
    mask = action_projection_mask(action_joint_names, mask_mode)
    projected = raw_action.copy()
    projected[mask] = (
        raw_action[mask] + alpha * (group_average[mask] - raw_action[mask])
    )
    applied_delta = projected - raw_action
    metrics = {
        "equivariance_rmse": float(np.sqrt(np.mean(equivariance_error**2))),
        "equivariance_abs_max": float(np.max(np.abs(equivariance_error))),
        "applied_delta_rmse": float(np.sqrt(np.mean(applied_delta**2))),
        "applied_delta_abs_max": float(np.max(np.abs(applied_delta))),
    }
    return projected.astype(raw_action.dtype, copy=False), metrics
