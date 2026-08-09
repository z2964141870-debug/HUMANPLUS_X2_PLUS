#!/usr/bin/env python3
"""Validated Stage335 reset-state curriculum for the X2 stand backend.

The phase-1 reset term changes only physical state.  The optional phase-4
contract adds a post-manager-reset finalizer: Isaac Lab runs reset events
*before* it clears action/command buffers and episode time, so logical state
cannot be restored honestly from an event callback alone.  A dedicated env
mixin calls :func:`finalize_stateful_recovery` after ``super()._reset_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
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


@dataclass(frozen=True)
class StatefulRecoveryAudit:
    source_report_path: str
    source_report_sha256: str
    source_count: int
    state_count: int
    moving_phase_count: int
    stationary_phase_count: int
    low_command_forced_moving_count: int
    max_observation_crosscheck_error: float
    previous_issued_action_missing_count: int


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


def gait_phase_from_steps(
    steps: np.ndarray,
    moving: np.ndarray,
    *,
    step_dt: float = 0.02,
    cycle_time_s: float = 0.8,
    double_support_fraction: float = 0.30,
) -> np.ndarray:
    """Pure NumPy mirror of the X2 four-value gait observation."""
    steps = np.asarray(steps, dtype=np.int64)
    moving = np.asarray(moving, dtype=bool)
    phase = np.remainder(steps.astype(np.float64) * step_dt / cycle_time_s, 1.0)
    angle = 2.0 * math.pi * phase
    clock = np.stack((np.sin(angle), np.cos(angle)), axis=-1)
    clock *= moving[:, None]
    half_ds = double_support_fraction / 4.0
    right_swing = (phase >= half_ds) & (phase < 0.5 - half_ds)
    left_swing = (phase >= 0.5 + half_ds) & (phase < 1.0 - half_ds)
    contacts = np.stack((~left_swing, ~right_swing), axis=-1)
    contacts = np.where(moving[:, None], contacts, np.ones_like(contacts))
    return np.concatenate((clock, contacts.astype(np.float64)), axis=-1)


@lru_cache(maxsize=4)
def load_stateful_recovery_sidecar(
    dataset_path_string: str,
    source_report_path_string: str,
    expected_dataset_sha256: str | None = None,
    expected_source_report_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Rejoin Stage335 rows with their immutable official trace snapshots.

    The original NPZ intentionally contains the 93-D observation slices but
    omitted the command and the issued (post-template) action.  ``source_index``
    and ``trace_index`` form a lossless foreign key into the five hashed source
    traces.  This function validates that join before returning any logical
    reset state; it never modifies the phase-1 NPZ.
    """
    dataset_path = Path(dataset_path_string).expanduser().resolve()
    source_report_path = Path(source_report_path_string).expanduser().resolve()
    arrays = load_recovery_dataset(str(dataset_path), expected_dataset_sha256)
    report_digest = sha256_file(source_report_path)
    if expected_source_report_sha256 is not None and report_digest != expected_source_report_sha256:
        raise ValueError(
            f"source report SHA-256 mismatch: {report_digest} != {expected_source_report_sha256}"
        )
    report = json.loads(source_report_path.read_text(encoding="utf-8"))
    sources = report.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source report has no source_files provenance")

    payloads: list[dict] = []
    for source in sources:
        path = Path(source["path"]).expanduser().resolve()
        digest = sha256_file(path)
        if digest != source["sha256"]:
            raise ValueError(f"source trace SHA-256 mismatch: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))

    commands = []
    clock_steps = []
    event_remaining = []
    force_moving = []
    force_moving_remaining = []
    previous_issued = []
    current_issued = []
    max_crosscheck_error = 0.0
    missing_previous = 0
    control_dt = 0.02
    obs_slices = (
        (slice(0, 3), "base_lin_vel_body_mps"),
        (slice(3, 6), "base_ang_vel_body_radps"),
        (slice(6, 9), "projected_gravity"),
        (slice(12, 43), "joint_pos_rel_rad"),
        (slice(43, 74), "joint_vel_radps"),
        (slice(74, 89), "previous_action"),
        (slice(89, 93), "gait_phase"),
    )
    for row, (source_index, trace_index) in enumerate(
        zip(arrays["source_index"], arrays["trace_index"])
    ):
        source_index = int(source_index)
        trace_index = int(trace_index)
        if not 0 <= source_index < len(payloads):
            raise ValueError(f"invalid source_index at row {row}: {source_index}")
        payload = payloads[source_index]
        trace = payload.get("trace", [])
        if not 0 <= trace_index < len(trace):
            raise ValueError(f"invalid trace_index at row {row}: {trace_index}")
        item = trace[trace_index]
        if item.get("stage") != "stop":
            raise ValueError(f"stateful row {row} does not join to a stop event")
        elapsed = float(arrays["elapsed_s"][row])
        if abs(float(item["elapsed_s"]) - elapsed) > 1.0e-8:
            raise ValueError(f"elapsed-time join mismatch at row {row}")
        obs = np.asarray(item.get("obs", []), dtype=np.float64)
        if obs.shape != (93,) or not np.isfinite(obs).all():
            raise ValueError(f"source observation is not finite 93-D at row {row}")
        for source_slice, dataset_key in obs_slices:
            error = float(np.max(np.abs(obs[source_slice] - arrays[dataset_key][row])))
            max_crosscheck_error = max(max_crosscheck_error, error)
            if error > 1.0e-6:
                raise ValueError(
                    f"source/NPZ observation mismatch at row {row} key={dataset_key}: {error}"
                )

        command = obs[9:12]
        moving = bool(np.linalg.norm(obs[89:91]) > 0.5)
        steps = int(round(elapsed / control_dt))
        reconstructed = gait_phase_from_steps(
            np.asarray([steps]), np.asarray([moving]), step_dt=control_dt
        )[0]
        if float(np.max(np.abs(reconstructed - obs[89:93]))) > 1.0e-5:
            raise ValueError(f"episode clock cannot reconstruct gait phase at row {row}")

        prior_action = None
        if trace_index > 0:
            candidate = np.asarray(trace[trace_index - 1].get("action", []), dtype=np.float64)
            if candidate.shape == (15,) and np.isfinite(candidate).all():
                prior_action = candidate
        if prior_action is None:
            missing_previous += 1
            prior_action = np.full(15, np.nan, dtype=np.float64)
        issued = np.asarray(item.get("action", []), dtype=np.float64)
        if issued.shape != (15,) or not np.isfinite(issued).all():
            raise ValueError(f"current issued action is missing at row {row}")

        summary = payload["summary"]
        stop_seconds = float(summary["stop_seconds"])
        latch = summary.get("stop_hold_latch_s")
        latch = stop_seconds if latch is None else float(latch)
        commands.append(command)
        clock_steps.append(steps)
        event_remaining.append(max(control_dt, stop_seconds - elapsed))
        force_moving.append(moving)
        force_moving_remaining.append(max(0.0, latch - elapsed) if moving else 0.0)
        previous_issued.append(prior_action)
        current_issued.append(issued)

    if missing_previous:
        raise ValueError(f"{missing_previous} stateful rows lack a previous issued action")
    result = {
        "command_velocity_mps_radps": np.asarray(commands, dtype=np.float32),
        "episode_clock_steps": np.asarray(clock_steps, dtype=np.int64),
        "event_remaining_s": np.asarray(event_remaining, dtype=np.float32),
        "force_moving": np.asarray(force_moving, dtype=bool),
        "force_moving_remaining_s": np.asarray(force_moving_remaining, dtype=np.float32),
        "previous_issued_action": np.asarray(previous_issued, dtype=np.float32),
        "current_issued_action": np.asarray(current_issued, dtype=np.float32),
        "source_control_dt_s": np.asarray(control_dt, dtype=np.float64),
        "source_report_sha256": np.asarray(report_digest),
        "source_count": np.asarray(len(sources), dtype=np.int64),
        "max_observation_crosscheck_error": np.asarray(max_crosscheck_error, dtype=np.float64),
    }
    return result


def audit_stateful_recovery_sidecar(
    dataset_path: str | Path,
    source_report_path: str | Path,
    expected_dataset_sha256: str | None = None,
    expected_source_report_sha256: str | None = None,
) -> StatefulRecoveryAudit:
    dataset = load_recovery_dataset(
        str(Path(dataset_path).expanduser().resolve()), expected_dataset_sha256
    )
    sidecar = load_stateful_recovery_sidecar(
        str(Path(dataset_path).expanduser().resolve()),
        str(Path(source_report_path).expanduser().resolve()),
        expected_dataset_sha256,
        expected_source_report_sha256,
    )
    moving = sidecar["force_moving"]
    commands = sidecar["command_velocity_mps_radps"]
    low_command_forced = moving & (np.linalg.norm(commands[:, :2], axis=1) <= 0.1)
    return StatefulRecoveryAudit(
        source_report_path=str(Path(source_report_path).expanduser().resolve()),
        source_report_sha256=str(sidecar["source_report_sha256"].item()),
        source_count=int(sidecar["source_count"].item()),
        state_count=int(len(dataset["root_z_m"])),
        moving_phase_count=int(moving.sum()),
        stationary_phase_count=int((~moving).sum()),
        low_command_forced_moving_count=int(low_command_forced.sum()),
        max_observation_crosscheck_error=float(sidecar["max_observation_crosscheck_error"].item()),
        previous_issued_action_missing_count=0,
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
    stateful_source_report: str | None = None,
    stateful_source_report_sha256: str | None = None,
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
    vel_limits = asset.data.soft_joint_vel_limits[selected].clone()
    # IsaacLab initializes soft_joint_vel_limits to zeros and fills them only
    # in Articulation._apply_actuator_model(), which runs after the first reset
    # events.  Clamping against that transient zero tensor silently erased all
    # Stage335 dq on the first reset.  The PhysX/URDF hard limits are already
    # initialized at this point and are the fail-closed fallback only for
    # non-positive soft entries; later resets keep the actuator soft limits.
    uninitialized_vel_limits = vel_limits <= 0.0
    velocity_limit_fallback_count = int(uninitialized_vel_limits.sum().item())
    if velocity_limit_fallback_count:
        hard_vel_limits = asset.data.joint_vel_limits[selected]
        if bool(torch.any(hard_vel_limits <= 0.0)):
            raise RuntimeError("recovery reset encountered non-positive hard joint velocity limits")
        vel_limits = torch.where(uninitialized_vel_limits, hard_vel_limits, vel_limits)
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
        # Scalar observability only: this distinguishes source-side clipping
        # from an Isaac/PhysX write failure without retaining reset tensors.
        "requested_joint_velocity_abs_max_radps": float(torch.abs(joint_vel).max().item()),
        "velocity_limit_fallback_count": velocity_limit_fallback_count,
    }
    if stateful_source_report is not None:
        # This pending payload is intentionally finalized only after Isaac
        # Lab's action/command managers and episode clock have been reset.
        # Merely setting it in this event is not a stateful reset contract.
        env._x2_recovery_stateful_pending = {
            "selected_env_ids": selected.clone(),
            "sample_indices": sample_ids.clone(),
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "expected_dataset_sha256": expected_sha256,
            "source_report_path": str(Path(stateful_source_report).expanduser().resolve()),
            "expected_source_report_sha256": stateful_source_report_sha256,
        }


def finalize_stateful_recovery(
    env,
    *,
    action_name: str = "joint_pos",
    command_name: str = "base_velocity",
) -> dict[str, object]:
    """Restore logical buffers after Isaac Lab's normal manager reset.

    This must run *after* ``ManagerBasedRLEnv._reset_idx``.  It fails closed
    when the expected manager internals are unavailable; silently restoring a
    subset of the 93-D contract would make the curriculum invalid.
    """
    import torch

    pending = getattr(env, "_x2_recovery_stateful_pending", None)
    if pending is None:
        return {"selected_env_count": 0, "no_op": True}
    delattr(env, "_x2_recovery_stateful_pending")
    selected = pending["selected_env_ids"]
    sample_ids = pending["sample_indices"]
    if len(selected) == 0:
        return {"selected_env_count": 0, "no_op": True}
    sidecar = load_stateful_recovery_sidecar(
        pending["dataset_path"],
        pending["source_report_path"],
        pending["expected_dataset_sha256"],
        pending["expected_source_report_sha256"],
    )
    dataset = load_recovery_dataset(
        pending["dataset_path"], pending["expected_dataset_sha256"]
    )
    sample_cpu = sample_ids.detach().cpu().numpy()
    step_dt = float(env.step_dt)
    source_dt = float(sidecar["source_control_dt_s"].item())
    if abs(step_dt - source_dt) > 1.0e-9:
        raise RuntimeError(
            f"stateful recovery requires matched 50 Hz control: env={step_dt} source={source_dt}"
        )

    if not hasattr(env, "episode_length_buf"):
        raise RuntimeError("environment does not expose episode_length_buf")
    action_manager = env.action_manager
    for name in ("_action", "_prev_action"):
        if not hasattr(action_manager, name):
            raise RuntimeError(f"action manager lacks required buffer {name}")
    action_term = action_manager.get_term(action_name)
    required_action_buffers = (
        "_raw_actions",
        "_processed_actions",
        "_combined_normalized_actions",
        "_preclip_combined_actions",
        "_normalized_template_bias",
        "_scale",
        "_offset",
    )
    missing = [name for name in required_action_buffers if not hasattr(action_term, name)]
    if missing:
        raise RuntimeError(f"stateful gait action lacks required buffers: {missing}")

    command_term = env.command_manager.get_term(command_name)
    required_command_buffers = (
        "vel_command_b",
        "is_standing_env",
        "is_heading_env",
        "heading_target",
        "time_left",
        "command_counter",
    )
    missing = [name for name in required_command_buffers if not hasattr(command_term, name)]
    if missing:
        raise RuntimeError(f"velocity command term lacks required buffers: {missing}")

    device = action_manager._action.device
    dtype = action_manager._action.dtype
    previous_action = torch.as_tensor(
        dataset["previous_action"][sample_cpu], device=device, dtype=dtype
    )
    issued_action = torch.as_tensor(
        sidecar["previous_issued_action"][sample_cpu], device=device, dtype=dtype
    )
    clock_steps = torch.as_tensor(
        sidecar["episode_clock_steps"][sample_cpu],
        device=env.episode_length_buf.device,
        dtype=env.episode_length_buf.dtype,
    )
    command = torch.as_tensor(
        sidecar["command_velocity_mps_radps"][sample_cpu],
        device=command_term.vel_command_b.device,
        dtype=command_term.vel_command_b.dtype,
    )
    moving = torch.as_tensor(
        sidecar["force_moving"][sample_cpu], device=device, dtype=torch.bool
    )
    event_remaining = torch.as_tensor(
        sidecar["event_remaining_s"][sample_cpu],
        device=command_term.time_left.device,
        dtype=command_term.time_left.dtype,
    )
    moving_remaining = torch.as_tensor(
        sidecar["force_moving_remaining_s"][sample_cpu],
        device=env.episode_length_buf.device,
        dtype=torch.float32,
    )

    env.episode_length_buf[selected] = clock_steps
    action_manager._action[selected] = previous_action
    action_manager._prev_action[selected] = previous_action
    action_term._raw_actions[selected] = previous_action
    action_term._combined_normalized_actions[selected] = issued_action
    action_term._preclip_combined_actions[selected] = issued_action
    action_term._normalized_template_bias[selected] = issued_action - previous_action
    scale = torch.as_tensor(action_term._scale, device=device, dtype=dtype)
    offset = torch.as_tensor(action_term._offset, device=device, dtype=dtype)
    processed = issued_action * scale[selected] + offset[selected]
    clip = getattr(action_term, "_clip", None)
    if clip is not None:
        processed = torch.clamp(processed, min=clip[selected, :, 0], max=clip[selected, :, 1])
    action_term._processed_actions[selected] = processed

    command_term.vel_command_b[selected] = command
    command_term.is_standing_env[selected] = ~moving.to(command_term.is_standing_env.device)
    command_term.is_heading_env[selected] = False
    # Direct-yaw Stage326 commands do not carry a separate heading target.
    command_term.heading_target[selected] = 0.0
    command_term.time_left[selected] = event_remaining
    command_term.command_counter[selected] = 1
    for optional in (
        "heading_error_integral",
        "heading_integral_contribution",
        "heading_error",
    ):
        if hasattr(command_term, optional):
            getattr(command_term, optional)[selected] = 0.0

    if not hasattr(env, "_x2_recovery_force_moving"):
        env._x2_recovery_force_moving = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        env._x2_recovery_force_moving_until_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    extra_steps = torch.ceil(moving_remaining / step_dt).to(dtype=torch.long)
    env._x2_recovery_force_moving[selected] = moving
    env._x2_recovery_force_moving_until_step[selected] = torch.where(
        moving,
        clock_steps.to(env.device) + torch.clamp(extra_steps.to(env.device), min=1),
        torch.full_like(clock_steps.to(env.device), -1),
    )

    result = {
        "selected_env_count": int(len(selected)),
        "no_op": False,
        "episode_clock_steps": clock_steps.clone(),
        "previous_action": previous_action.clone(),
        "previous_issued_action": issued_action.clone(),
        "command": command.clone(),
        "force_moving": moving.clone(),
    }
    env._x2_recovery_reset_last["stateful_finalized"] = True
    env._x2_recovery_reset_last["stateful_result"] = result
    return result
