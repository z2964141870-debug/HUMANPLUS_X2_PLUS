"""Controlled lower/upper reference composition for conditional capability tests."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np


ARM_TOKENS = ("shoulder_", "elbow", "wrist_")


def arm_indices(joint_names: list[str]) -> list[int]:
    return [
        index
        for index, name in enumerate(joint_names)
        if any(token in name for token in ARM_TOKENS)
    ]


def select_dynamic_window(
    dof: np.ndarray,
    indexes: list[int],
    *,
    source_fps: float,
    duration_s: float,
) -> tuple[int, int, float]:
    """Select the fixed-duration arm window with the largest mean variance."""
    window = min(len(dof), max(2, int(math.ceil(duration_s * source_fps)) + 1))
    best_start = 0
    best_score = -math.inf
    for start in range(0, len(dof) - window + 1):
        segment = dof[start : start + window, indexes]
        score = float(np.mean(np.var(segment, axis=0)))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, window, best_score


def interpolate_window(
    values: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
    target_length: int,
    source_start: int,
) -> np.ndarray:
    source_t = np.arange(len(values), dtype=np.float64) / float(source_fps)
    target_t = source_start / float(source_fps) + np.arange(target_length) / float(target_fps)
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    flat = values.reshape(len(values), -1)
    output = np.empty((target_length, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target_t, source_t, flat[:, column])
    return output.reshape((target_length,) + values.shape[1:]).astype(values.dtype)


def raised_cosine_envelope(length: int) -> np.ndarray:
    if length < 2:
        return np.zeros(length, dtype=np.float32)
    phase = np.linspace(0.0, math.pi, length, dtype=np.float64)
    envelope = np.square(np.sin(phase)).astype(np.float32)
    envelope[0] = 0.0
    envelope[-1] = 0.0
    return envelope


def _joint_axes(indexes: list[int], *motions: Mapping[str, Any]) -> np.ndarray:
    names = list(motions[0]["joint_names_mujoco"])
    axes = np.zeros((len(names), 3), dtype=np.float32)
    for joint_index in indexes:
        for motion in motions:
            q = np.asarray(motion["dof"])[:, joint_index]
            pose = np.asarray(motion["pose_aa"])[:, joint_index + 1]
            nonzero = np.flatnonzero(np.abs(q) > 1.0e-5)
            if len(nonzero):
                sample = int(nonzero[0])
                axis = pose[sample] / q[sample]
                norm = float(np.linalg.norm(axis))
                if norm > 0.9:
                    axes[joint_index] = axis / norm
                    break
        if not np.any(axes[joint_index]):
            # A permanently-zero arm DOF has no observable axis in the cache,
            # but its composed value is also zero and its original pose is safe.
            if any(np.any(np.asarray(motion["dof"])[:, joint_index]) for motion in motions):
                raise ValueError(f"cannot infer axis for joint {names[joint_index]}")
    return axes


def compose_gait_with_upper(
    base: Mapping[str, Any],
    upper: Mapping[str, Any],
    *,
    alpha: float = 0.60,
    joint_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blend only arm DOFs while preserving gait root, legs, waist, and contacts."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    names = list(base["joint_names_mujoco"])
    if names != list(upper["joint_names_mujoco"]):
        raise ValueError("base and upper joint orders differ")
    base_q = np.asarray(base["dof"], dtype=np.float32)
    upper_q = np.asarray(upper["dof"], dtype=np.float32)
    indexes = arm_indices(names)
    duration_s = (len(base_q) - 1) / float(base["fps"])
    start, source_window, score = select_dynamic_window(
        upper_q,
        indexes,
        source_fps=float(upper["fps"]),
        duration_s=duration_s,
    )
    sampled_upper = interpolate_window(
        upper_q,
        source_fps=float(upper["fps"]),
        target_fps=float(base["fps"]),
        target_length=len(base_q),
        source_start=start,
    )
    envelope = raised_cosine_envelope(len(base_q))[:, None]
    blend = alpha * envelope
    composed_q = base_q.copy()
    composed_q[:, indexes] = (
        (1.0 - blend) * base_q[:, indexes] + blend * sampled_upper[:, indexes]
    )

    clipped = 0
    if joint_ranges:
        for index in indexes:
            limits = joint_ranges.get(names[index])
            if limits is None:
                continue
            before = composed_q[:, index].copy()
            composed_q[:, index] = np.clip(composed_q[:, index], limits[0], limits[1])
            clipped += int(np.count_nonzero(before != composed_q[:, index]))

    result = deepcopy(dict(base))
    result["dof"] = composed_q.astype(np.float32)
    axes = _joint_axes(indexes, base, upper)
    pose = np.asarray(base["pose_aa"], dtype=np.float32).copy()
    pose[:, np.asarray(indexes) + 1, :] = (
        composed_q[:, indexes, None] * axes[None, indexes, :]
    )
    result["pose_aa"] = pose
    # The official gait action is a measured command. It is no longer truthful
    # after synthetic arm composition, so remove it rather than silently lying.
    result.pop("action", None)
    result.pop("action_semantics", None)
    result["source_type"] = "synthetic_gait_upper_composition_probe"

    step = np.abs(np.diff(composed_q, axis=0)) if len(composed_q) > 1 else np.zeros_like(composed_q)
    report = {
        "alpha": alpha,
        "upper_source_start_frame": start,
        "upper_source_window_frames": source_window,
        "upper_source_window_score": score,
        "arm_joint_count": len(indexes),
        "clipped_arm_values": clipped,
        "max_dof_step_rad": float(step.max(initial=0.0)),
        "max_arm_step_rad": float(step[:, indexes].max(initial=0.0)),
        "start_arm_delta_rad": float(np.max(np.abs(composed_q[0, indexes] - base_q[0, indexes]))),
        "end_arm_delta_rad": float(np.max(np.abs(composed_q[-1, indexes] - base_q[-1, indexes]))),
        "lower_waist_exact": bool(
            np.array_equal(
                composed_q[:, [i for i in range(len(names)) if i not in indexes]],
                base_q[:, [i for i in range(len(names)) if i not in indexes]],
            )
        ),
        "pose_dof_norm_max_error": float(
            np.max(
                np.abs(
                    np.linalg.norm(pose[:, np.asarray(indexes) + 1, :], axis=-1)
                    - np.abs(composed_q[:, indexes])
                )
            )
        ),
    }
    return result, report
