#!/usr/bin/env python3
"""Contracts for on-policy closed-loop recovery suffix aggregation.

The official Stage208 adapter may optionally emit one immutable sidecar after
an episode.  This module validates that sidecar and defines a virtual union
with the frozen Stage335 reset set; it deliberately does not train, reset a
simulator, or rewrite the Stage335 NPZ.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from official_x2.controller_snapshot_contract import validate_controller_state


SNAPSHOT_SCHEMA = "aimdk_x2_recovery_suffix_snapshot_v1"
SIDECAR_SCHEMA = "aimdk_x2_recovery_suffix_sidecar_v1"
AGGREGATION_SCHEMA = "x2_stage335_recovery_suffix_virtual_union_v1"
CAPTURE_BOUNDARY = "post_inference_pre_physics"
CONTROL_DT_S = 0.02
OBS_DIM = 93
ACTION_DIM = 15
JOINT_DIM = 31


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape ({size},), got {array.shape}")
    return array


def recovery_capture_eligible(
    *,
    output_enabled: bool,
    authority_slot: str,
    time_after_handoff_s: float,
    horizon_s: float = 1.5,
) -> bool:
    """Pure default-off and authority/horizon gate for suffix recording."""
    if horizon_s < 0.0:
        raise ValueError("capture horizon must be non-negative")
    if not output_enabled:
        return False
    if authority_slot != "recovery":
        return False
    return -1.0e-9 <= float(time_after_handoff_s) <= horizon_s + 1.0e-9


def physical_state_payload(
    *,
    joint_names: Sequence[str],
    joint_position_rad: Sequence[float],
    joint_velocity_radps: Sequence[float],
    root_position_m: Sequence[float],
    root_quaternion_xyzw: Sequence[float],
    root_linear_velocity_world_mps: Sequence[float],
    root_angular_velocity_world_radps: Sequence[float],
) -> dict[str, Any]:
    names = tuple(str(name) for name in joint_names)
    if len(names) != JOINT_DIM or len(set(names)) != JOINT_DIM:
        raise ValueError("joint_names must contain 31 unique names")
    position = _finite_vector("joint_position_rad", joint_position_rad, JOINT_DIM)
    velocity = _finite_vector("joint_velocity_radps", joint_velocity_radps, JOINT_DIM)
    root_position = _finite_vector("root_position_m", root_position_m, 3)
    quaternion = _finite_vector("root_quaternion_xyzw", root_quaternion_xyzw, 4)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-4:
        raise ValueError(f"root quaternion is not normalized: {norm}")
    root_linear = _finite_vector(
        "root_linear_velocity_world_mps", root_linear_velocity_world_mps, 3
    )
    root_angular = _finite_vector(
        "root_angular_velocity_world_radps", root_angular_velocity_world_radps, 3
    )
    return {
        "joint_names": list(names),
        "joint_position_rad": position.tolist(),
        "joint_velocity_radps": velocity.tolist(),
        "root_position_m": root_position.tolist(),
        "root_quaternion_xyzw": quaternion.tolist(),
        "root_linear_velocity_world_mps": root_linear.tolist(),
        "root_angular_velocity_world_radps": root_angular.tolist(),
    }


def build_recovery_suffix_snapshot(
    *,
    episode_id: str,
    source_tick: int,
    time_after_handoff_s: float,
    physical_state: Mapping[str, Any],
    observation: Sequence[float],
    previous_action_input: Sequence[float],
    issued_action: Sequence[float],
    command: Sequence[float],
    gait_phase: Sequence[float],
    controller_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one hash-bound post-inference/pre-physics recovery row."""
    if not episode_id:
        raise ValueError("episode_id is required")
    if not isinstance(source_tick, int) or source_tick < 0:
        raise ValueError("source_tick must be a non-negative integer")
    if time_after_handoff_s < -1.0e-9:
        raise ValueError("time_after_handoff_s must be non-negative")
    physical = physical_state_payload(**dict(physical_state))
    physical_sha = canonical_sha256(physical)
    obs = _finite_vector("observation", observation, OBS_DIM)
    previous = _finite_vector("previous_action_input", previous_action_input, ACTION_DIM)
    issued = _finite_vector("issued_action", issued_action, ACTION_DIM)
    command_array = _finite_vector("command", command, 3)
    gait = _finite_vector("gait_phase", gait_phase, 4)
    if np.max(np.abs(obs[9:12] - command_array)) > 1.0e-6:
        raise ValueError("command does not match observation[9:12]")
    if np.max(np.abs(obs[74:89] - previous)) > 1.0e-6:
        raise ValueError("previous action does not match observation[74:89]")
    if np.max(np.abs(obs[89:93] - gait)) > 1.0e-6:
        raise ValueError("gait phase does not match observation[89:93]")
    validate_controller_state(controller_state)
    if controller_state["physical_state_sha256"] != physical_sha:
        raise ValueError("controller snapshot is not bound to the physical state")
    controller_issued = _finite_vector(
        "controller issued recovery action",
        controller_state["issued_actions"]["recovery"],
        ACTION_DIM,
    )
    if np.max(np.abs(controller_issued - issued)) > 1.0e-6:
        raise ValueError("issued action differs from controller recovery slot")
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "capture_boundary": CAPTURE_BOUNDARY,
        "episode_id": str(episode_id),
        "authority_slot": "recovery",
        "source_tick": source_tick,
        "time_after_handoff_s": float(time_after_handoff_s),
        "control_dt_s": CONTROL_DT_S,
        "physical_state": physical,
        "physical_state_sha256": physical_sha,
        "observation_93d": obs.tolist(),
        "previous_action_input": previous.tolist(),
        "actual_issued_action": issued.tolist(),
        "command_velocity_mps_radps": command_array.tolist(),
        "gait_phase": gait.tolist(),
        "episode_clock": {
            "controller_sequence_step": source_tick,
            "time_after_handoff_s": float(time_after_handoff_s),
            "control_dt_s": CONTROL_DT_S,
        },
        "controller_state": _jsonable(controller_state),
        "controller_state_sha256": canonical_sha256(controller_state),
        "truth_boundary": (
            "post-inference controller state plus the same pre-physics observation state; "
            "replay must apply actual_issued_action for one physics interval before the next actor tick"
        ),
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    validate_recovery_suffix_snapshot(payload)
    return payload


def validate_recovery_suffix_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported recovery suffix snapshot schema")
    if snapshot.get("capture_boundary") != CAPTURE_BOUNDARY:
        raise ValueError("unsupported recovery suffix capture boundary")
    if snapshot.get("authority_slot") != "recovery":
        raise ValueError("snapshot was not recorded under recovery authority")
    if abs(float(snapshot.get("control_dt_s", -1.0)) - CONTROL_DT_S) > 1.0e-12:
        raise ValueError("snapshot is not on the 50 Hz control contract")
    physical = physical_state_payload(**dict(snapshot["physical_state"]))
    if canonical_sha256(physical) != snapshot.get("physical_state_sha256"):
        raise ValueError("physical state hash mismatch")
    obs = _finite_vector("observation_93d", snapshot["observation_93d"], OBS_DIM)
    previous = _finite_vector(
        "previous_action_input", snapshot["previous_action_input"], ACTION_DIM
    )
    issued = _finite_vector(
        "actual_issued_action", snapshot["actual_issued_action"], ACTION_DIM
    )
    command = _finite_vector(
        "command_velocity_mps_radps", snapshot["command_velocity_mps_radps"], 3
    )
    gait = _finite_vector("gait_phase", snapshot["gait_phase"], 4)
    if np.max(np.abs(obs[9:12] - command)) > 1.0e-6:
        raise ValueError("snapshot command/observation mismatch")
    if np.max(np.abs(obs[74:89] - previous)) > 1.0e-6:
        raise ValueError("snapshot previous-action/observation mismatch")
    if np.max(np.abs(obs[89:93] - gait)) > 1.0e-6:
        raise ValueError("snapshot gait/observation mismatch")
    controller = snapshot["controller_state"]
    validate_controller_state(controller)
    if controller["physical_state_sha256"] != snapshot["physical_state_sha256"]:
        raise ValueError("controller/physical hash mismatch")
    if canonical_sha256(controller) != snapshot.get("controller_state_sha256"):
        raise ValueError("controller state hash mismatch")
    controller_issued = _finite_vector(
        "controller issued recovery action",
        controller["issued_actions"]["recovery"],
        ACTION_DIM,
    )
    if np.max(np.abs(controller_issued - issued)) > 1.0e-6:
        raise ValueError("controller/issued-action mismatch")
    unsigned = dict(snapshot)
    digest = unsigned.pop("snapshot_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("snapshot content hash mismatch")


def build_suffix_sidecar(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    source_trace_path: str | Path,
    source_trace_sha256: str,
    adapter_path: str | Path,
    adapter_sha256: str,
    recovery_model_path: str | Path,
    recovery_model_sha256: str,
    episode_outcome: Mapping[str, Any],
    horizon_s: float,
) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("enabled suffix capture produced no snapshots")
    if horizon_s < 0.0:
        raise ValueError("horizon_s must be non-negative")
    rows = [_jsonable(snapshot) for snapshot in snapshots]
    for row in rows:
        validate_recovery_suffix_snapshot(row)
        if float(row["time_after_handoff_s"]) > horizon_s + 1.0e-9:
            raise ValueError("snapshot exceeds declared capture horizon")
    ticks = [int(row["source_tick"]) for row in rows]
    if ticks != sorted(ticks) or len(ticks) != len(set(ticks)):
        raise ValueError("suffix snapshot ticks must be strictly increasing")
    times = [float(row["time_after_handoff_s"]) for row in rows]
    if times != sorted(times):
        raise ValueError("suffix snapshot times must be monotonic")
    for name, digest in (
        ("source_trace_sha256", source_trace_sha256),
        ("adapter_sha256", adapter_sha256),
        ("recovery_model_sha256", recovery_model_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid {name}")
    outcome = {
        "full_gate_pass": bool(episode_outcome.get("full_gate_pass")),
        "stand_gate_pass": bool(episode_outcome.get("stand_gate_pass")),
        "startup_gate_pass": bool(episode_outcome.get("startup_gate_pass")),
        "move_gate_pass": bool(episode_outcome.get("move_gate_pass")),
        "stop_gate_pass": bool(episode_outcome.get("stop_gate_pass")),
    }
    payload = {
        "schema": SIDECAR_SCHEMA,
        "purpose": "on-policy closed-loop recovery reset curriculum; not a motion reference",
        "capture_boundary": CAPTURE_BOUNDARY,
        "capture_horizon_s": float(horizon_s),
        "source_trace": {
            "path": str(source_trace_path),
            "sha256": source_trace_sha256,
        },
        "adapter": {"path": str(adapter_path), "sha256": adapter_sha256},
        "recovery_model": {
            "path": str(recovery_model_path),
            "sha256": recovery_model_sha256,
        },
        "episode_outcome": outcome,
        "row_count": len(rows),
        "row_snapshot_sha256": [row["snapshot_sha256"] for row in rows],
        "rows": rows,
        "truth_boundary": (
            "rows preserve a closed-loop post-inference/pre-physics sequence; "
            "they are curriculum evidence, not proof of deterministic ROS replay"
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    validate_suffix_sidecar(payload)
    return payload


def validate_suffix_sidecar(sidecar: Mapping[str, Any]) -> None:
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise ValueError("unsupported suffix sidecar schema")
    rows = sidecar.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("suffix sidecar has no rows")
    if int(sidecar.get("row_count", -1)) != len(rows):
        raise ValueError("suffix sidecar row count mismatch")
    horizon = float(sidecar.get("capture_horizon_s", -1.0))
    if not 0.0 <= horizon <= 1.5:
        raise ValueError("suffix sidecar capture horizon is outside [0, 1.5]")
    for field in ("source_trace", "adapter", "recovery_model"):
        record = sidecar.get(field)
        if not isinstance(record, Mapping) or not record.get("path"):
            raise ValueError(f"suffix sidecar lacks {field} provenance")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"suffix sidecar has invalid {field} SHA-256")
    outcome = sidecar.get("episode_outcome")
    outcome_fields = {
        "full_gate_pass", "stand_gate_pass", "startup_gate_pass",
        "move_gate_pass", "stop_gate_pass",
    }
    if not isinstance(outcome, Mapping) or set(outcome) != outcome_fields:
        raise ValueError("suffix sidecar episode outcome contract differs")
    if any(not isinstance(outcome[name], bool) for name in outcome_fields):
        raise ValueError("suffix sidecar episode outcomes must be boolean")
    for row in rows:
        validate_recovery_suffix_snapshot(row)
        if float(row["time_after_handoff_s"]) > horizon + 1.0e-9:
            raise ValueError("suffix sidecar row exceeds its capture horizon")
    ticks = [int(row["source_tick"]) for row in rows]
    times = [float(row["time_after_handoff_s"]) for row in rows]
    if ticks != sorted(ticks) or len(ticks) != len(set(ticks)):
        raise ValueError("suffix sidecar ticks are not strictly increasing")
    if times != sorted(times):
        raise ValueError("suffix sidecar times are not monotonic")
    if len({row["episode_id"] for row in rows}) != 1:
        raise ValueError("one suffix sidecar must contain exactly one episode")
    if sidecar.get("row_snapshot_sha256") != [row["snapshot_sha256"] for row in rows]:
        raise ValueError("suffix sidecar row hash list mismatch")
    unsigned = dict(sidecar)
    digest = unsigned.pop("content_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("suffix sidecar content hash mismatch")


def write_suffix_sidecar(path: str | Path, sidecar: Mapping[str, Any]) -> None:
    validate_suffix_sidecar(sidecar)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(sidecar), indent=2) + "\n", encoding="utf-8")


def load_suffix_sidecar(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise ValueError("suffix sidecar file SHA-256 mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_suffix_sidecar(payload)
    return payload


# Conservative, fixed bins for duplicate suppression.  The signature is root
# XY/yaw invariant but keeps root height, actor input, issued action, command,
# gait and event time.  Full rows remain in the immutable sidecar; only the
# representative index used for reset sampling is deduplicated.
_DEDUP_STEPS = np.concatenate(
    (
        np.full(3, 0.01),   # body linear velocity
        np.full(3, 0.02),   # body angular velocity
        np.full(3, 0.002),  # projected gravity
        np.full(3, 0.01),   # command
        np.full(31, 0.005), # q relative
        np.full(31, 0.02),  # dq
        np.full(15, 0.01),  # previous action input
        np.full(4, 0.002),  # gait phase/contact
        np.full(15, 0.01),  # actual issued action
        np.asarray([0.002, CONTROL_DT_S]),  # root z, time after handoff
    )
)


def recovery_dedup_key(snapshot: Mapping[str, Any]) -> str:
    validate_recovery_suffix_snapshot(snapshot)
    obs = np.asarray(snapshot["observation_93d"], dtype=np.float64)
    issued = np.asarray(snapshot["actual_issued_action"], dtype=np.float64)
    root_z = float(snapshot["physical_state"]["root_position_m"][2])
    event_time = float(snapshot["time_after_handoff_s"])
    vector = np.concatenate((obs, issued, [root_z, event_time]))
    if vector.shape != _DEDUP_STEPS.shape:
        raise RuntimeError("internal suffix dedup contract shape mismatch")
    quantized = np.rint(vector / _DEDUP_STEPS).astype("<i8", copy=False)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def deduplicate_suffix_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    representatives: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}
    for source_index, raw in enumerate(snapshots):
        snapshot = _jsonable(raw)
        key = recovery_dedup_key(snapshot)
        if key not in key_to_index:
            key_to_index[key] = len(representatives)
            representatives.append(snapshot)
        assignments.append(
            {
                "source_index": source_index,
                "representative_index": key_to_index[key],
                "dedup_key_sha256": key,
                "snapshot_sha256": snapshot["snapshot_sha256"],
            }
        )
    return representatives, assignments


def suffix_source_mask(
    count: int, suffix_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Choose virtual-union source without consuming RNG when fraction is 0."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if not 0.0 <= suffix_fraction <= 1.0:
        raise ValueError("suffix_fraction must be in [0, 1]")
    if suffix_fraction == 0.0:
        return np.zeros(count, dtype=bool)
    if suffix_fraction == 1.0:
        return np.ones(count, dtype=bool)
    return rng.random(count) < suffix_fraction


def build_aggregation_manifest(
    *,
    base_dataset_path: str | Path,
    base_dataset_sha256: str,
    base_state_count: int,
    suffix_sidecar_paths: Sequence[str | Path],
    suffix_fraction: float,
) -> dict[str, Any]:
    """Build a hash-bound virtual union while keeping Stage335 immutable.

    ``suffix_fraction=0`` returns before opening any sidecar.  A caller using
    that value therefore retains the exact Stage335 data and RNG path.
    """
    if not 0.0 <= suffix_fraction <= 1.0:
        raise ValueError("suffix_fraction must be in [0, 1]")
    base = Path(base_dataset_path).expanduser().resolve()
    if sha256_file(base) != base_dataset_sha256:
        raise ValueError("base Stage335 SHA-256 mismatch")
    if base_state_count <= 0:
        raise ValueError("base_state_count must be positive")
    payload: dict[str, Any] = {
        "schema": AGGREGATION_SCHEMA,
        "base_dataset": {
            "path": str(base),
            "sha256": base_dataset_sha256,
            "state_count": int(base_state_count),
            "immutable": True,
        },
        "suffix_fraction": float(suffix_fraction),
        "sampler_contract": "per-reset source mixture; fraction=0 consumes no suffix RNG",
        "materialization": "virtual union only; Stage335 NPZ is never rewritten",
    }
    if suffix_fraction == 0.0:
        payload.update(
            {
                "strict_no_op": True,
                "suffix_sidecars": [],
                "suffix_raw_state_count": 0,
                "suffix_representative_state_count": 0,
                "dedup_assignments": [],
            }
        )
        payload["content_sha256"] = canonical_sha256(payload)
        return payload

    sidecar_records = []
    snapshots = []
    for path in suffix_sidecar_paths:
        resolved = Path(path).expanduser().resolve()
        file_sha = sha256_file(resolved)
        sidecar = load_suffix_sidecar(resolved, file_sha)
        if bool(sidecar["episode_outcome"]["full_gate_pass"]):
            raise ValueError("Phase15 aggregation accepts failed on-policy suffixes only")
        sidecar_records.append(
            {
                "path": str(resolved),
                "file_sha256": file_sha,
                "content_sha256": sidecar["content_sha256"],
                "row_count": sidecar["row_count"],
            }
        )
        snapshots.extend(sidecar["rows"])
    if not snapshots:
        raise ValueError("non-zero suffix fraction requires at least one failed suffix row")
    representatives, assignments = deduplicate_suffix_snapshots(snapshots)
    payload.update(
        {
            "strict_no_op": False,
            "suffix_sidecars": sidecar_records,
            "suffix_raw_state_count": len(snapshots),
            "suffix_representative_state_count": len(representatives),
            "representative_snapshot_sha256": [
                row["snapshot_sha256"] for row in representatives
            ],
            "dedup_assignments": assignments,
            "dedup_contract": {
                "root_xy_yaw_invariant": True,
                "event_time_retained": True,
                "quantization": {
                    "body_linear_velocity_mps": 0.01,
                    "body_angular_velocity_radps": 0.02,
                    "projected_gravity": 0.002,
                    "command": 0.01,
                    "joint_position_rad": 0.005,
                    "joint_velocity_radps": 0.02,
                    "previous_and_issued_action": 0.01,
                    "gait_phase": 0.002,
                    "root_z_m": 0.002,
                    "event_time_s": CONTROL_DT_S,
                },
            },
        }
    )
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "AGGREGATION_SCHEMA",
    "CAPTURE_BOUNDARY",
    "SIDECAR_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "build_aggregation_manifest",
    "build_recovery_suffix_snapshot",
    "build_suffix_sidecar",
    "canonical_sha256",
    "deduplicate_suffix_snapshots",
    "load_suffix_sidecar",
    "physical_state_payload",
    "recovery_capture_eligible",
    "recovery_dedup_key",
    "sha256_file",
    "suffix_source_mask",
    "validate_recovery_suffix_snapshot",
    "validate_suffix_sidecar",
    "write_suffix_sidecar",
]
