#!/usr/bin/env python3
"""Hash-bound, stop-event-aligned stateful evidence for BASE Phase17."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from official_x2.controller_snapshot_contract import validate_controller_state
from official_x2.recovery_suffix_aggregation import (
    ACTION_DIM,
    CONTROL_DT_S,
    OBS_DIM,
    _jsonable,
    canonical_sha256,
    physical_state_payload,
)


SNAPSHOT_SCHEMA = "aimdk_x2_stop_event_suffix_snapshot_v1"
SIDECAR_SCHEMA = "aimdk_x2_stop_event_suffix_sidecar_v1"
CAPTURE_BOUNDARY = "post_inference_pre_physics"
MAX_HORIZON_S = 4.0
AUTHORITY_SLOTS = {"brake_main", "brake_stationary_blend", "stationary", "recovery"}


def _vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape ({size},)")
    return array


def stop_event_capture_eligible(
    *, output_enabled: bool, stage: str, stop_elapsed_s: float, horizon_s: float
) -> bool:
    if not 0.0 <= horizon_s <= MAX_HORIZON_S:
        raise ValueError("stop-event capture horizon must be in [0, 4.0]")
    return bool(
        output_enabled
        and stage == "stop"
        and -1.0e-9 <= float(stop_elapsed_s) <= horizon_s + 1.0e-9
    )


def build_stop_event_snapshot(
    *,
    episode_id: str,
    source_tick: int,
    stop_elapsed_s: float,
    authority_slot: str,
    observation_policy_slot: str,
    physical_state: Mapping[str, Any],
    observation: Sequence[float],
    actor_proposal_action: Sequence[float],
    actual_issued_action: Sequence[float],
    physical_lower_target_rad: Sequence[float],
    default_lower_target_rad: Sequence[float],
    lower_action_scale_rad: Sequence[float],
    command: Sequence[float],
    gait_phase: Sequence[float],
    controller_state: Mapping[str, Any],
) -> dict[str, Any]:
    if not episode_id or source_tick < 0 or stop_elapsed_s < -1.0e-9:
        raise ValueError("invalid stop-event identity/time")
    if authority_slot not in AUTHORITY_SLOTS:
        raise ValueError(f"unsupported stop-event authority: {authority_slot}")
    if observation_policy_slot not in {"main", "stationary", "recovery"}:
        raise ValueError("invalid observation policy slot")
    physical = physical_state_payload(**dict(physical_state))
    physical_sha = canonical_sha256(physical)
    obs = _vector("observation", observation, OBS_DIM)
    proposal = _vector("actor_proposal_action", actor_proposal_action, ACTION_DIM)
    actual = _vector("actual_issued_action", actual_issued_action, ACTION_DIM)
    target = _vector("physical_lower_target_rad", physical_lower_target_rad, ACTION_DIM)
    default_target = _vector("default_lower_target_rad", default_lower_target_rad, ACTION_DIM)
    action_scale = _vector("lower_action_scale_rad", lower_action_scale_rad, ACTION_DIM)
    if np.any(action_scale <= 0.0):
        raise ValueError("lower action scales must be positive")
    if np.max(np.abs(default_target + actual * action_scale - target)) > 1.0e-6:
        raise ValueError("actual issued action does not reconstruct physical lower target")
    command_array = _vector("command", command, 3)
    gait = _vector("gait_phase", gait_phase, 4)
    if np.max(np.abs(obs[9:12] - command_array)) > 1.0e-6:
        raise ValueError("command/observation mismatch")
    if np.max(np.abs(obs[89:93] - gait)) > 1.0e-6:
        raise ValueError("gait/observation mismatch")
    validate_controller_state(controller_state)
    if controller_state["physical_state_sha256"] != physical_sha:
        raise ValueError("controller snapshot is not bound to physical state")
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "capture_boundary": CAPTURE_BOUNDARY,
        "episode_id": episode_id,
        "source_tick": int(source_tick),
        "stop_elapsed_s": float(stop_elapsed_s),
        "control_dt_s": CONTROL_DT_S,
        "authority_slot": authority_slot,
        "observation_policy_slot": observation_policy_slot,
        "physical_state": physical,
        "physical_state_sha256": physical_sha,
        "observation_93d": obs.tolist(),
        "actual_previous_action_input": obs[74:89].tolist(),
        "actor_proposal_action": proposal.tolist(),
        "actual_issued_action": actual.tolist(),
        "physical_lower_target_rad": target.tolist(),
        "default_lower_target_rad": default_target.tolist(),
        "lower_action_scale_rad": action_scale.tolist(),
        "command_velocity_mps_radps": command_array.tolist(),
        "gait_phase": gait.tolist(),
        "controller_state": _jsonable(controller_state),
        "controller_state_sha256": canonical_sha256(controller_state),
        "truth_boundary": (
            "post-inference/pre-physics: actual_issued_action is reverse-mapped from the final "
            "physical lower target; actor_proposal_action may differ during brake-policy blending"
        ),
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    validate_stop_event_snapshot(payload)
    return payload


def validate_stop_event_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema") != SNAPSHOT_SCHEMA or snapshot.get("capture_boundary") != CAPTURE_BOUNDARY:
        raise ValueError("unsupported stop-event snapshot schema/boundary")
    if snapshot.get("authority_slot") not in AUTHORITY_SLOTS:
        raise ValueError("invalid stop-event authority")
    if abs(float(snapshot.get("control_dt_s", -1.0)) - CONTROL_DT_S) > 1.0e-12:
        raise ValueError("stop-event snapshot is not 50 Hz")
    physical = physical_state_payload(**dict(snapshot["physical_state"]))
    if canonical_sha256(physical) != snapshot.get("physical_state_sha256"):
        raise ValueError("physical hash mismatch")
    obs = _vector("observation_93d", snapshot["observation_93d"], OBS_DIM)
    previous = _vector("actual_previous_action_input", snapshot["actual_previous_action_input"], ACTION_DIM)
    _vector("actor_proposal_action", snapshot["actor_proposal_action"], ACTION_DIM)
    actual = _vector("actual_issued_action", snapshot["actual_issued_action"], ACTION_DIM)
    target = _vector("physical_lower_target_rad", snapshot["physical_lower_target_rad"], ACTION_DIM)
    default_target = _vector("default_lower_target_rad", snapshot["default_lower_target_rad"], ACTION_DIM)
    action_scale = _vector("lower_action_scale_rad", snapshot["lower_action_scale_rad"], ACTION_DIM)
    if np.any(action_scale <= 0.0) or np.max(np.abs(default_target + actual * action_scale - target)) > 1.0e-6:
        raise ValueError("actual-issued/physical-target reconstruction mismatch")
    command = _vector("command", snapshot["command_velocity_mps_radps"], 3)
    gait = _vector("gait", snapshot["gait_phase"], 4)
    if np.max(np.abs(obs[74:89] - previous)) > 1.0e-6:
        raise ValueError("previous-action/observation mismatch")
    if np.max(np.abs(obs[9:12] - command)) > 1.0e-6 or np.max(np.abs(obs[89:93] - gait)) > 1.0e-6:
        raise ValueError("command/gait observation mismatch")
    controller = snapshot["controller_state"]
    validate_controller_state(controller)
    if controller["physical_state_sha256"] != snapshot["physical_state_sha256"]:
        raise ValueError("controller/physical hash mismatch")
    if canonical_sha256(controller) != snapshot.get("controller_state_sha256"):
        raise ValueError("controller state hash mismatch")
    unsigned = dict(snapshot)
    digest = unsigned.pop("snapshot_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("stop-event snapshot content hash mismatch")


def build_stop_event_sidecar(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    source_trace_path: str | Path,
    source_trace_sha256: str,
    adapter_path: str | Path,
    adapter_sha256: str,
    model_assets: Mapping[str, Mapping[str, str]],
    episode_outcome: Mapping[str, Any],
    horizon_s: float,
    frozen_control_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not snapshots or not 0.0 <= horizon_s <= MAX_HORIZON_S:
        raise ValueError("invalid stop-event sidecar rows/horizon")
    rows = [_jsonable(row) for row in snapshots]
    for row in rows:
        validate_stop_event_snapshot(row)
        if float(row["stop_elapsed_s"]) > horizon_s + 1.0e-9:
            raise ValueError("row exceeds stop-event horizon")
    times = [float(row["stop_elapsed_s"]) for row in rows]
    ticks = [int(row["source_tick"]) for row in rows]
    if times != sorted(times) or ticks != sorted(ticks) or len(set(ticks)) != len(ticks):
        raise ValueError("stop-event rows are not strictly ordered")
    payload = {
        "schema": SIDECAR_SCHEMA,
        "purpose": "outcome-stratified closed-loop stop/recovery curriculum evidence; not a motion reference",
        "capture_boundary": CAPTURE_BOUNDARY,
        "capture_anchor": "stop_command_start",
        "capture_horizon_s": float(horizon_s),
        "source_trace": {"path": str(source_trace_path), "sha256": source_trace_sha256},
        "adapter": {"path": str(adapter_path), "sha256": adapter_sha256},
        "model_assets": _jsonable(model_assets),
        "frozen_control_contract": _jsonable(frozen_control_contract),
        "episode_outcome": {
            key: bool(episode_outcome.get(key))
            for key in (
                "full_gate_pass", "stand_gate_pass", "startup_gate_pass",
                "move_gate_pass", "stop_gate_pass", "survived_stop_height_gate",
            )
        },
        "row_count": len(rows),
        "row_snapshot_sha256": [row["snapshot_sha256"] for row in rows],
        "rows": rows,
        "truth_boundary": "stateful official-MuJoCo evidence; not X2 real-foot-force truth or ROS mid-event restore proof",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    validate_stop_event_sidecar(payload)
    return payload


def validate_stop_event_sidecar(sidecar: Mapping[str, Any]) -> None:
    if sidecar.get("schema") != SIDECAR_SCHEMA or sidecar.get("capture_anchor") != "stop_command_start":
        raise ValueError("unsupported stop-event sidecar")
    rows = sidecar.get("rows")
    if not isinstance(rows, list) or not rows or int(sidecar.get("row_count", -1)) != len(rows):
        raise ValueError("invalid stop-event sidecar row count")
    for row in rows:
        validate_stop_event_snapshot(row)
    if sidecar.get("row_snapshot_sha256") != [row["snapshot_sha256"] for row in rows]:
        raise ValueError("stop-event sidecar row hash list mismatch")
    unsigned = dict(sidecar)
    digest = unsigned.pop("content_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("stop-event sidecar content hash mismatch")


def write_stop_event_sidecar(path: str | Path, sidecar: Mapping[str, Any]) -> None:
    import json

    validate_stop_event_sidecar(sidecar)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(sidecar), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "AUTHORITY_SLOTS", "CAPTURE_BOUNDARY", "MAX_HORIZON_S", "SIDECAR_SCHEMA",
    "SNAPSHOT_SCHEMA", "build_stop_event_sidecar", "build_stop_event_snapshot",
    "stop_event_capture_eligible", "validate_stop_event_sidecar",
    "validate_stop_event_snapshot", "write_stop_event_sidecar",
]
