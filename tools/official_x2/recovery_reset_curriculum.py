#!/usr/bin/env python3
"""Validated Stage335 reset-state curriculum for the X2 stand backend.

This module deliberately changes only the *initial physical state*.  It does
not treat official MuJoCo rollouts as reference motion and it does not restore
the actor's previous-action/gait-clock buffers.  Isaac Lab resets those manager
buffers after reset events, so pretending to restore the full 93-D observation
would create a false contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


REQUIRED_ARRAY_SHAPES = {
    "source_index": (None,),
    "trace_index": (None,),
    "elapsed_s": (None,),
    "eventual_pass": (None,),
    "root_z_m": (None,),
    "root_yaw_rad": (None,),
    "root_roll_rad": (None,),
    "root_pitch_rad": (None,),
    "base_lin_vel_body_mps": (None, 3),
    "base_ang_vel_body_radps": (None, 3),
    "projected_gravity": (None, 3),
    "joint_pos_rel_rad": (None, 31),
    "joint_vel_radps": (None, 31),
    "previous_action": (None, 15),
    "gait_phase": (None, 4),
    "joint_names": (31,),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecoveryDatasetAudit:
    path: str
    sha256: str
    state_count: int
    eventual_pass_count: int
    eventual_fail_count: int
    source_counts: dict[int, int]
    root_z_range_m: tuple[float, float]
    max_abs_tilt_rad: float
    max_body_speed_mps: float
    gravity_norm_max_error: float


def _shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...], n: int) -> bool:
    resolved = tuple(n if value is None else value for value in expected)
    return actual == resolved


@lru_cache(maxsize=8)
def load_recovery_dataset(path_string: str, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    """Load and strictly validate one Stage335-compatible NPZ file."""
    path = Path(path_string).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"recovery dataset SHA-256 mismatch: {digest} != {expected_sha256}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    missing = sorted(set(REQUIRED_ARRAY_SHAPES) - set(arrays))
    extra = sorted(set(arrays) - set(REQUIRED_ARRAY_SHAPES))
    if missing or extra:
        raise ValueError(f"recovery dataset keys differ: missing={missing} extra={extra}")
    n = int(arrays["root_z_m"].shape[0])
    if n <= 0:
        raise ValueError("recovery dataset is empty")
    for name, expected in REQUIRED_ARRAY_SHAPES.items():
        if not _shape_matches(arrays[name].shape, expected, n):
            raise ValueError(f"invalid {name} shape: {arrays[name].shape}, expected {expected}")
        if arrays[name].dtype.kind in "fiu" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite values in {name}")
    joint_names = arrays["joint_names"].tolist()
    if len(set(joint_names)) != 31:
        raise ValueError("joint_names must contain 31 unique names")
    if arrays["eventual_pass"].dtype.kind != "b":
        raise ValueError("eventual_pass must be boolean")
    gravity_error = np.abs(np.linalg.norm(arrays["projected_gravity"], axis=1) - 1.0)
    if float(gravity_error.max()) > 1.0e-4:
        raise ValueError("projected gravity is not normalized")
    if np.any(arrays["root_z_m"] <= 0.0):
        raise ValueError("root_z_m must be positive")
    if np.any(np.abs(arrays["previous_action"]) > 1.00001):
        raise ValueError("previous_action violates the clipped Stage208 contract")
    return arrays


def audit_recovery_dataset(path: str | Path, expected_sha256: str | None = None) -> RecoveryDatasetAudit:
    arrays = load_recovery_dataset(str(Path(path).expanduser().resolve()), expected_sha256)
    labels = arrays["eventual_pass"].astype(bool)
    sources, counts = np.unique(arrays["source_index"], return_counts=True)
    speed = np.linalg.norm(arrays["base_lin_vel_body_mps"][:, :2], axis=1)
    tilt = np.hypot(arrays["root_roll_rad"], arrays["root_pitch_rad"])
    gravity_error = np.abs(np.linalg.norm(arrays["projected_gravity"], axis=1) - 1.0)
    resolved = Path(path).expanduser().resolve()
    return RecoveryDatasetAudit(
        path=str(resolved),
        sha256=sha256_file(resolved),
        state_count=int(labels.size),
        eventual_pass_count=int(labels.sum()),
        eventual_fail_count=int((~labels).sum()),
        source_counts={int(source): int(count) for source, count in zip(sources, counts)},
        root_z_range_m=(float(arrays["root_z_m"].min()), float(arrays["root_z_m"].max())),
        max_abs_tilt_rad=float(tilt.max()),
        max_body_speed_mps=float(speed.max()),
        gravity_norm_max_error=float(gravity_error.max()),
    )


def joint_reorder_indices(dataset_joint_names: Sequence[str], asset_joint_names: Sequence[str]) -> np.ndarray:
    """Return indices that reorder dataset vectors into articulation order."""
    dataset_names = list(dataset_joint_names)
    asset_names = list(asset_joint_names)
    if len(dataset_names) != len(asset_names) or set(dataset_names) != set(asset_names):
        missing = sorted(set(asset_names) - set(dataset_names))
        extra = sorted(set(dataset_names) - set(asset_names))
        raise ValueError(f"joint-name contract mismatch: missing={missing} extra={extra}")
    return np.asarray([dataset_names.index(name) for name in asset_names], dtype=np.int64)


def balanced_sample_indices(labels: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample approximately 50/50 eventual-pass/fail states with replacement."""
    labels = np.asarray(labels, dtype=bool)
    pass_ids = np.flatnonzero(labels)
    fail_ids = np.flatnonzero(~labels)
    if pass_ids.size == 0 or fail_ids.size == 0:
        raise ValueError("balanced sampling requires both eventual-pass and eventual-fail states")
    requested_labels = np.arange(count) % 2 == 0
    rng.shuffle(requested_labels)
    result = np.empty(count, dtype=np.int64)
    result[requested_labels] = rng.choice(pass_ids, int(requested_labels.sum()), replace=True)
    result[~requested_labels] = rng.choice(fail_ids, int((~requested_labels).sum()), replace=True)
    return result


def reset_from_recovery_dataset(
    env,
    env_ids,
    dataset_path: str,
    recovery_fraction: float,
    sampling_mode: str = "balanced",
    expected_sha256: str | None = None,
    asset_name: str = "robot",
):
    """Final reset event that overwrites a subset with Stage335 physical states.

    This term must be appended *after* the inherited ``reset_base`` and
    ``reset_robot_joints`` terms.  Non-selected environments are untouched,
    making ``recovery_fraction=0`` an exact control.
    """
    import torch

    if not 0.0 <= recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must be in [0, 1]")
    if sampling_mode not in {"all", "balanced", "eventual_pass", "eventual_fail"}:
        raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
    if recovery_fraction == 0.0 or len(env_ids) == 0:
        env._x2_recovery_reset_last = {
            "selected_env_ids": torch.empty(0, dtype=torch.long, device=env.device),
            "sample_indices": torch.empty(0, dtype=torch.long, device=env.device),
        }
        return

    arrays = load_recovery_dataset(str(Path(dataset_path).expanduser().resolve()), expected_sha256)
    asset = env.scene[asset_name]
    device = asset.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if recovery_fraction == 1.0:
        selected = env_ids
    else:
        selected = env_ids[torch.rand(len(env_ids), device=device) < recovery_fraction]
    if len(selected) == 0:
        env._x2_recovery_reset_last = {
            "selected_env_ids": selected,
            "sample_indices": torch.empty(0, dtype=torch.long, device=device),
        }
        return

    labels = torch.as_tensor(arrays["eventual_pass"], device=device, dtype=torch.bool)
    if sampling_mode == "all":
        candidate_ids = torch.arange(len(labels), device=device)
        sample_ids = candidate_ids[torch.randint(len(candidate_ids), (len(selected),), device=device)]
    elif sampling_mode in {"eventual_pass", "eventual_fail"}:
        target = sampling_mode == "eventual_pass"
        candidate_ids = torch.nonzero(labels == target, as_tuple=False).flatten()
        if len(candidate_ids) == 0:
            raise ValueError(f"dataset has no {sampling_mode} states")
        sample_ids = candidate_ids[torch.randint(len(candidate_ids), (len(selected),), device=device)]
    else:
        pass_ids = torch.nonzero(labels, as_tuple=False).flatten()
        fail_ids = torch.nonzero(~labels, as_tuple=False).flatten()
        if len(pass_ids) == 0 or len(fail_ids) == 0:
            raise ValueError("balanced sampling requires both label classes")
        choose_pass = torch.arange(len(selected), device=device) % 2 == 0
        choose_pass = choose_pass[torch.randperm(len(selected), device=device)]
        sample_ids = torch.empty(len(selected), dtype=torch.long, device=device)
        sample_ids[choose_pass] = pass_ids[
            torch.randint(len(pass_ids), (int(choose_pass.sum().item()),), device=device)
        ]
        sample_ids[~choose_pass] = fail_ids[
            torch.randint(len(fail_ids), (int((~choose_pass).sum().item()),), device=device)
        ]

    sample_cpu = sample_ids.detach().cpu().numpy()
    reorder = joint_reorder_indices(arrays["joint_names"].tolist(), asset.joint_names)
    reorder_t = torch.as_tensor(reorder, device=device, dtype=torch.long)
    state_dtype = asset.data.default_joint_pos.dtype

    joint_rel = torch.as_tensor(
        arrays["joint_pos_rel_rad"][sample_cpu], device=device, dtype=state_dtype
    )
    joint_vel = torch.as_tensor(
        arrays["joint_vel_radps"][sample_cpu], device=device, dtype=state_dtype
    )
    joint_pos = asset.data.default_joint_pos[selected] + joint_rel[:, reorder_t]
    joint_vel = joint_vel[:, reorder_t]
    limits = asset.data.soft_joint_pos_limits[selected]
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits[selected]
    joint_vel = torch.clamp(joint_vel, -vel_limits, vel_limits)

    roll = torch.as_tensor(arrays["root_roll_rad"][sample_cpu], device=device, dtype=state_dtype)
    pitch = torch.as_tensor(arrays["root_pitch_rad"][sample_cpu], device=device, dtype=state_dtype)
    yaw = torch.as_tensor(arrays["root_yaw_rad"][sample_cpu], device=device, dtype=state_dtype)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    # ZYX Euler to wxyz quaternion.
    quat = torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )
    positions = env.scene.env_origins[selected].clone()
    positions[:, 2] += torch.as_tensor(
        arrays["root_z_m"][sample_cpu], device=device, dtype=state_dtype
    )

    def rotate_body_to_world(v_body):
        q_xyz = quat[:, 1:4]
        t = 2.0 * torch.linalg.cross(q_xyz, v_body, dim=-1)
        return v_body + quat[:, 0:1] * t + torch.linalg.cross(q_xyz, t, dim=-1)

    lin_body = torch.as_tensor(
        arrays["base_lin_vel_body_mps"][sample_cpu], device=device, dtype=state_dtype
    )
    ang_body = torch.as_tensor(
        arrays["base_ang_vel_body_radps"][sample_cpu], device=device, dtype=state_dtype
    )
    root_velocity = torch.cat((rotate_body_to_world(lin_body), rotate_body_to_world(ang_body)), dim=-1)

    asset.write_root_pose_to_sim(torch.cat((positions, quat), dim=-1), env_ids=selected)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=selected)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=selected)
    env._x2_recovery_reset_last = {
        "selected_env_ids": selected.clone(),
        "sample_indices": sample_ids.clone(),
    }
