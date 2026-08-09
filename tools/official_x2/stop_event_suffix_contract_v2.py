#!/usr/bin/env python3
"""Lossless actor-input evidence for BASE Phase19.

V1 stop-event rows captured the final 93-D vector, but not the sensor and
clock state which produced it.  V2 deliberately keeps both: the exact vector
fed to the actor and the independent sources required to reconstruct it.
This is an evidence/reset contract, never an imitation-label contract.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from official_x2.recovery_suffix_aggregation import _jsonable, canonical_sha256
from official_x2.stop_event_suffix_contract import (
    CAPTURE_BOUNDARY,
    MAX_HORIZON_S,
    build_stop_event_snapshot,
    validate_stop_event_snapshot,
)


SNAPSHOT_SCHEMA_V2 = "aimdk_x2_stop_event_suffix_snapshot_v2"
SIDECAR_SCHEMA_V2 = "aimdk_x2_stop_event_suffix_sidecar_v2"
SELF_CONSISTENCY_ATOL = 1.0e-6
OUTCOME_KEYS = (
    "full_gate_pass", "stand_gate_pass", "startup_gate_pass",
    "move_gate_pass", "stop_gate_pass", "survived_stop_height_gate",
)

ISAAC_JOINTS = (
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
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)
LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)


def _finite_vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape ({size},)")
    return array


def _optional_vector(name: str, value: Any, size: int) -> np.ndarray | None:
    if value is None:
        return None
    return _finite_vector(name, value, size)


def _world_to_body(quaternion_xyzw: np.ndarray, vector_w: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    rotation_body_to_world = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation_body_to_world.T @ vector_w


def _gravity_from_imu(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    return np.asarray(
        [2.0 * (-z * x + w * y), -2.0 * (z * y + w * x), 1.0 - 2.0 * (w * w + z * z)],
        dtype=np.float32,
    )


def _mirror_named(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    result = np.empty_like(values)
    for index, name in enumerate(names):
        other = name
        if name.startswith("left_"):
            other = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            other = "left_" + name[len("right_"):]
        sign = -1.0 if any(axis in name for axis in ("_roll_", "_yaw_")) else 1.0
        result[index] = sign * values[lookup[other]]
    return result


def _predict_physical(source: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    odom = source["odom_source"]
    imu = source["torso_imu_source"]
    joints = source["joint_source"]
    predictor = source["predictor_state"]
    base_lin = _world_to_body(
        _finite_vector("odom quaternion", odom["root_quaternion_xyzw"], 4),
        _finite_vector("odom linear velocity", odom["root_linear_velocity_world_mps"], 3),
    )
    base_ang = _finite_vector("imu angular velocity", imu["angular_velocity_radps"], 3)
    gravity = _gravity_from_imu(
        _finite_vector("imu orientation", imu["orientation_xyzw"], 4)
    )
    q = _finite_vector("joint position", joints["position_rad"], 31)
    default = _finite_vector("default position", joints["default_position_rad"], 31)
    dq = _finite_vector("joint velocity", joints["velocity_radps"], 31)
    physical = np.concatenate((base_lin, base_ang, gravity, q - default, dq)).astype(np.float32)
    saved_raw = _finite_vector(
        "raw physical observation", predictor["raw_physical_observation_71d"], 71
    )
    if np.max(np.abs(physical - saved_raw)) > SELF_CONSISTENCY_ATOL:
        raise ValueError("raw sensor/joint sources do not reconstruct raw physical observation")

    tau = float(predictor["prediction_seconds"])
    sequence_step = int(predictor["sequence_step"])
    predicted_step_before = int(predictor["predicted_step_before"])
    previous_before = _optional_vector(
        "previous physical before", predictor.get("previous_physical_before"), 71
    )
    predicted_before = _optional_vector(
        "predicted physical before", predictor.get("predicted_physical_before"), 71
    )
    if tau <= 0.0:
        predicted = physical.copy()
        expected_after = {
            "previous_physical_after": previous_before,
            "predicted_physical_after": predicted_before,
            "predicted_step_after": predicted_step_before,
        }
    elif predicted_step_before == sequence_step:
        if predicted_before is None:
            raise ValueError("cached predictor step lacks predicted physical state")
        predicted = predicted_before.copy()
        expected_after = {
            "previous_physical_after": previous_before,
            "predicted_physical_after": predicted_before,
            "predicted_step_after": predicted_step_before,
        }
    else:
        acceleration = np.zeros_like(physical)
        if previous_before is not None:
            acceleration = (physical - previous_before) / np.float32(0.02)
        predicted = physical.copy()
        delayed_joint_velocity = physical[40:71].copy()
        predicted[0:6] += np.float32(tau) * acceleration[0:6]
        predicted[40:71] += np.float32(tau) * acceleration[40:71]
        omega = predicted[3:6].copy()
        predicted_gravity = physical[6:9] - np.float32(tau) * np.cross(omega, physical[6:9])
        norm = float(np.linalg.norm(predicted_gravity))
        if norm > 1.0e-8:
            predicted[6:9] = predicted_gravity / np.float32(norm)
        predicted[9:40] += (
            np.float32(tau) * delayed_joint_velocity
            + np.float32(0.5 * tau * tau) * acceleration[40:71]
        )
        expected_after = {
            "previous_physical_after": physical,
            "predicted_physical_after": predicted,
            "predicted_step_after": sequence_step,
        }
    return predicted, expected_after


def _phase_features(source: Mapping[str, Any]) -> np.ndarray:
    input_elapsed = float(source["phase_elapsed_input_s"])
    actor_elapsed = float(source["actor_phase_elapsed_s"])
    offset = float(source["phase_offset_s"])
    should_offset = bool(source["force_moving"]) or abs(float(source["command_vx_before_mirror"])) > 0.1
    expected_elapsed = input_elapsed + (offset if should_offset else 0.0)
    if abs(actor_elapsed - expected_elapsed) > 1.0e-12:
        raise ValueError("gait actor clock does not match exact input clock/offset")
    if not bool(source["moving"]):
        result = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        saved = _finite_vector("saved gait output", source["gait_output"], 4)
        if np.max(np.abs(result - saved)) > SELF_CONSISTENCY_ATOL:
            raise ValueError("saved stationary gait output mismatch")
        return result
    elapsed = actor_elapsed
    period = float(source["period_s"])
    ds_fraction = float(source["double_support_fraction"])
    phase = (elapsed / period) % 1.0
    half_ds_width = ds_fraction / 4.0
    right_swing = half_ds_width <= phase < 0.5 - half_ds_width
    left_swing = 0.5 + half_ds_width <= phase < 1.0 - half_ds_width
    result = np.asarray(
        [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase), not left_swing, not right_swing],
        dtype=np.float32,
    )
    saved = _finite_vector("saved gait output", source["gait_output"], 4)
    if np.max(np.abs(result - saved)) > SELF_CONSISTENCY_ATOL:
        raise ValueError("saved gait output mismatch")
    return result


def reconstruct_actor_observation(
    actor_state: Mapping[str, Any], *, validate_predictor_after: bool = True
) -> np.ndarray:
    """Reconstruct the actor's 93-D input from independently saved sources."""
    physical, expected_after = _predict_physical(actor_state)
    predictor = actor_state["predictor_state"]
    if validate_predictor_after:
        for key in ("previous_physical_after", "predicted_physical_after"):
            actual = _optional_vector(key, predictor.get(key), 71)
            expected = expected_after[key]
            if (actual is None) != (expected is None):
                raise ValueError(f"predictor {key} nullability mismatch")
            if actual is not None and np.max(np.abs(actual - expected)) > SELF_CONSISTENCY_ATOL:
                raise ValueError(f"predictor {key} transition mismatch")
        if int(predictor["predicted_step_after"]) != int(expected_after["predicted_step_after"]):
            raise ValueError("predictor step transition mismatch")

    command = _finite_vector("command", actor_state["command_velocity_mps_radps"], 3)
    previous = _finite_vector("previous action", actor_state["previous_action_pre_inference"], 15)
    gait = _phase_features(actor_state["gait_generator_state"])
    if bool(actor_state["mirror_policy"]):
        physical[0:3] *= np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        physical[3:6] *= np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
        physical[6:9] *= np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        command *= np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
        physical[9:40] = _mirror_named(physical[9:40], ISAAC_JOINTS)
        physical[40:71] = _mirror_named(physical[40:71], ISAAC_JOINTS)
        previous = _mirror_named(previous, LOWER_JOINTS)
    return np.concatenate((physical[0:9], command, physical[9:71], previous, gait)).astype(np.float32)


def validate_actor_observation_state(actor_state: Mapping[str, Any]) -> dict[str, float]:
    if actor_state.get("schema") != "aimdk_x2_actor_observation_source_v2":
        raise ValueError("unsupported actor observation source schema")
    saved = _finite_vector("actor_observation_93d", actor_state["actor_observation_93d"], 93)
    if tuple(actor_state["joint_source"].get("joint_names", ())) != ISAAC_JOINTS:
        raise ValueError("actor joint source order is not the Stage208 ISAAC_JOINTS contract")
    model_input = np.asarray(actor_state["model_input"], dtype=np.float32)
    if model_input.ndim != 1 or model_input.size not in (93, 121, 123) or not np.isfinite(model_input).all():
        raise ValueError("model_input must be finite actor width 93/121/123")
    reconstructed = reconstruct_actor_observation(actor_state)
    error = np.abs(reconstructed - saved)
    model_error = np.abs(model_input[:93] - saved)
    max_error = float(np.max(error))
    model_max_error = float(np.max(model_error))
    if max_error > SELF_CONSISTENCY_ATOL:
        raise ValueError(f"actor 93D source reconstruction mismatch: {max_error:.9g}")
    if model_max_error > SELF_CONSISTENCY_ATOL:
        raise ValueError(f"model input/base observation mismatch: {model_max_error:.9g}")
    return {"reconstruction_abs_max": max_error, "model_prefix_abs_max": model_max_error}


def build_stop_event_snapshot_v2(*, actor_observation_state: Mapping[str, Any], **v1_kwargs: Any) -> dict[str, Any]:
    state = _jsonable(actor_observation_state)
    metrics = validate_actor_observation_state(state)
    v1 = build_stop_event_snapshot(**v1_kwargs)
    if np.max(
        np.abs(
            np.asarray(v1["observation_93d"], dtype=np.float32)
            - np.asarray(state["actor_observation_93d"], dtype=np.float32)
        )
    ) > SELF_CONSISTENCY_ATOL:
        raise ValueError("v1 evidence observation differs from captured actor input")
    payload = copy.deepcopy(v1)
    payload["schema"] = SNAPSHOT_SCHEMA_V2
    payload["actor_observation_state"] = state
    payload["actor_observation_state_sha256"] = canonical_sha256(state)
    payload["self_consistency"] = metrics
    payload["truth_boundary"] = (
        "post-inference/pre-physics physical/controller evidence plus the exact pre-inference "
        "actor input sources; actual actions are history/state only, never expert labels"
    )
    payload.pop("snapshot_sha256", None)
    payload["snapshot_sha256"] = canonical_sha256(payload)
    validate_stop_event_snapshot_v2(payload)
    return payload


def validate_stop_event_snapshot_v2(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema") != SNAPSHOT_SCHEMA_V2 or snapshot.get("capture_boundary") != CAPTURE_BOUNDARY:
        raise ValueError("unsupported v2 stop-event snapshot")
    v1 = copy.deepcopy(dict(snapshot))
    for key in ("actor_observation_state", "actor_observation_state_sha256", "self_consistency"):
        v1.pop(key, None)
    v1["schema"] = "aimdk_x2_stop_event_suffix_snapshot_v1"
    v1.pop("snapshot_sha256", None)
    # V2 truth text differs and participates in the V1 content hash; regenerate
    # a temporary valid V1 hash solely to reuse all structural validators.
    v1["truth_boundary"] = (
        "post-inference/pre-physics: actual_issued_action is reverse-mapped from the final "
        "physical lower target; actor_proposal_action may differ during brake-policy blending"
    )
    v1["snapshot_sha256"] = canonical_sha256(v1)
    validate_stop_event_snapshot(v1)
    state = snapshot["actor_observation_state"]
    if canonical_sha256(state) != snapshot.get("actor_observation_state_sha256"):
        raise ValueError("actor observation source hash mismatch")
    metrics = validate_actor_observation_state(state)
    if any(abs(float(snapshot["self_consistency"][key]) - value) > 1.0e-12 for key, value in metrics.items()):
        raise ValueError("self-consistency metrics mismatch")
    unsigned = dict(snapshot)
    digest = unsigned.pop("snapshot_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("v2 snapshot content hash mismatch")


def finite_gate_outcome(episode_outcome: Mapping[str, Any]) -> dict[str, bool]:
    """Project a possibly non-finite deploy summary onto finite gate labels."""
    return {key: bool(episode_outcome.get(key)) for key in OUTCOME_KEYS}


def build_stop_event_sidecar_v2(
    snapshots: Sequence[Mapping[str, Any]],
    *, source_trace_path: str | Path,
    source_trace_sha256: str,
    adapter_path: str | Path,
    adapter_sha256: str,
    model_assets: Mapping[str, Mapping[str, str]],
    episode_outcome: Mapping[str, Any],
    horizon_s: float,
    frozen_control_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not snapshots or not 0.0 <= horizon_s <= MAX_HORIZON_S:
        raise ValueError("invalid v2 stop-event sidecar rows/horizon")
    rows = [_jsonable(row) for row in snapshots]
    for row in rows:
        validate_stop_event_snapshot_v2(row)
        if float(row["stop_elapsed_s"]) > horizon_s + 1.0e-9:
            raise ValueError("row exceeds stop-event horizon")
    ticks = [int(row["source_tick"]) for row in rows]
    times = [float(row["stop_elapsed_s"]) for row in rows]
    if ticks != sorted(ticks) or len(set(ticks)) != len(ticks) or times != sorted(times):
        raise ValueError("v2 rows are not strictly ordered")
    payload = {
        "schema": SIDECAR_SCHEMA_V2,
        "purpose": "lossless actor-input stop/recovery reset evidence; not imitation labels",
        "capture_boundary": CAPTURE_BOUNDARY,
        "capture_anchor": "stop_command_start",
        "capture_horizon_s": float(horizon_s),
        "source_trace": {"path": str(source_trace_path), "sha256": source_trace_sha256},
        "adapter": {"path": str(adapter_path), "sha256": adapter_sha256},
        "model_assets": _jsonable(model_assets),
        "frozen_control_contract": _jsonable(frozen_control_contract),
        # The source trace retains the full summary.  The hash-bound sidecar
        # needs only finite gate labels; deploy summaries may legitimately
        # contain Infinity for a disabled threshold.
        "episode_outcome": finite_gate_outcome(episode_outcome),
        "row_count": len(rows),
        "row_snapshot_sha256": [row["snapshot_sha256"] for row in rows],
        "rows": rows,
        "truth_boundary": "official MuJoCo/ROS evidence only; no real foot-force truth and no expert action labels",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    validate_stop_event_sidecar_v2(payload)
    return payload


def validate_stop_event_sidecar_v2(sidecar: Mapping[str, Any]) -> None:
    if sidecar.get("schema") != SIDECAR_SCHEMA_V2 or sidecar.get("capture_anchor") != "stop_command_start":
        raise ValueError("unsupported v2 sidecar")
    rows = sidecar.get("rows")
    if not isinstance(rows, list) or not rows or int(sidecar.get("row_count", -1)) != len(rows):
        raise ValueError("invalid v2 sidecar row count")
    for row in rows:
        validate_stop_event_snapshot_v2(row)
    if sidecar.get("row_snapshot_sha256") != [row["snapshot_sha256"] for row in rows]:
        raise ValueError("v2 row hash list mismatch")
    unsigned = dict(sidecar)
    digest = unsigned.pop("content_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("v2 sidecar content hash mismatch")


def write_stop_event_sidecar_v2(path: str | Path, sidecar: Mapping[str, Any]) -> None:
    import json

    validate_stop_event_sidecar_v2(sidecar)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(sidecar), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "SELF_CONSISTENCY_ATOL", "SIDECAR_SCHEMA_V2", "SNAPSHOT_SCHEMA_V2",
    "build_stop_event_sidecar_v2", "build_stop_event_snapshot_v2",
    "finite_gate_outcome",
    "reconstruct_actor_observation", "validate_actor_observation_state",
    "validate_stop_event_sidecar_v2", "validate_stop_event_snapshot_v2",
    "write_stop_event_sidecar_v2",
]
