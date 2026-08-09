#!/usr/bin/env python3
"""Hash-bound Phase17 role-aware reset curriculum and Isaac injection glue.

The manifest contains source references only.  Actions are restored exclusively
as actor/controller history; they are never exposed as imitation labels.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from official_x2.recovery_suffix_aggregation import canonical_sha256, sha256_file
from official_x2.stop_event_suffix_contract import validate_stop_event_sidecar


MANIFEST_SCHEMA = "x2_phase17_role_aware_reset_curriculum_v1"
OUTCOME_ROLES = ("success", "critical", "height_fail")
CONTROL_DT_S = 0.02


def outcome_role(outcome: Mapping[str, Any]) -> str:
    if bool(outcome.get("full_gate_pass")):
        return "success"
    if bool(outcome.get("survived_stop_height_gate")):
        return "critical"
    return "height_fail"


def root_tilt_rad(row: Mapping[str, Any]) -> float:
    x, y, z, w = (float(v) for v in row["physical_state"]["root_quaternion_xyzw"])
    gravity_z = 1.0 - 2.0 * (w * w + z * z)
    gravity_x = 2.0 * (-z * x + w * y)
    gravity_y = -2.0 * (z * y + w * x)
    norm = math.sqrt(gravity_x * gravity_x + gravity_y * gravity_y + gravity_z * gravity_z)
    return math.acos(float(np.clip(-gravity_z / norm, -1.0, 1.0)))


def load_urdf_limits(path: str | Path) -> dict[str, dict[str, float]]:
    resolved = Path(path).expanduser().resolve()
    result: dict[str, dict[str, float]] = {}
    for joint in ET.parse(resolved).getroot().findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint lacks limit: {joint.attrib.get('name')}")
        result[joint.attrib["name"]] = {
            key: float(limit.attrib[key]) for key in ("lower", "upper", "velocity")
        }
    return result


def eligibility_reasons(
    row: Mapping[str, Any], limits: Mapping[str, Mapping[str, float]], *,
    min_root_z_m: float = 0.55, max_root_tilt_rad: float = 0.30,
    min_remaining_s: float = 0.50, capture_horizon_s: float = 4.0,
) -> list[str]:
    physical = row["physical_state"]
    reasons = []
    if float(physical["root_position_m"][2]) < min_root_z_m:
        reasons.append("root_z_below_safe_reset_floor")
    if root_tilt_rad(row) > max_root_tilt_rad:
        reasons.append("root_tilt_above_safe_reset_limit")
    if capture_horizon_s - float(row["stop_elapsed_s"]) < min_remaining_s - 1.0e-9:
        reasons.append("insufficient_recorded_continuation")
    for name, q in zip(physical["joint_names"], physical["joint_position_rad"]):
        if name not in limits:
            reasons.append("joint_missing_from_urdf")
            break
        if not limits[name]["lower"] - 1.0e-6 <= float(q) <= limits[name]["upper"] + 1.0e-6:
            reasons.append("joint_position_outside_urdf_limit")
            break
    for name, dq in zip(physical["joint_names"], physical["joint_velocity_radps"]):
        if name not in limits or abs(float(dq)) > limits[name]["velocity"] + 1.0e-6:
            reasons.append("joint_velocity_outside_urdf_limit")
            break
    return reasons


def build_manifest(
    sidecar_paths: Sequence[str | Path], urdf_path: str | Path, *,
    min_root_z_m: float = 0.55, max_root_tilt_rad: float = 0.30,
    min_remaining_s: float = 0.50,
) -> dict[str, Any]:
    urdf = Path(urdf_path).expanduser().resolve()
    limits = load_urdf_limits(urdf)
    sources, rows = [], []
    for source_index, path in enumerate(sidecar_paths):
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        validate_stop_event_sidecar(payload)
        role = outcome_role(payload["episode_outcome"])
        sources.append({
            "source_index": source_index,
            "path": str(resolved),
            "file_sha256": sha256_file(resolved),
            "content_sha256": payload["content_sha256"],
            "row_count": len(payload["rows"]),
            "outcome_role": role,
            "immutable": True,
        })
        for row_index, row in enumerate(payload["rows"]):
            reasons = eligibility_reasons(
                row, limits, min_root_z_m=min_root_z_m,
                max_root_tilt_rad=max_root_tilt_rad,
                min_remaining_s=min_remaining_s,
                capture_horizon_s=float(payload["capture_horizon_s"]),
            )
            rows.append({
                "manifest_row_index": len(rows),
                "source_index": source_index,
                "source_row_index": row_index,
                "source_snapshot_sha256": row["snapshot_sha256"],
                "source_controller_state_sha256": row["controller_state_sha256"],
                "outcome_role": role,
                "controller_role": row["authority_slot"],
                "observation_policy_slot": row["observation_policy_slot"],
                "stop_elapsed_s": float(row["stop_elapsed_s"]),
                "root_z_m": float(row["physical_state"]["root_position_m"][2]),
                "root_tilt_rad": root_tilt_rad(row),
                "eligible": not reasons,
                "ineligible_reasons": reasons,
            })
    counts = {}
    for role in OUTCOME_ROLES:
        subset = [row for row in rows if row["outcome_role"] == role]
        counts[role] = {
            "raw": len(subset),
            "eligible": sum(row["eligible"] for row in subset),
            "ineligible": sum(not row["eligible"] for row in subset),
            "eligible_controller_roles": dict(Counter(
                row["controller_role"] for row in subset if row["eligible"]
            )),
            "ineligible_reason_counts": dict(Counter(
                reason for row in subset for reason in row["ineligible_reasons"]
            )),
        }
        if counts[role]["eligible"] == 0:
            raise ValueError(f"no eligible Phase17 states for outcome role {role}")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "role-aware stateful reset curriculum; actions are history/state only, never expert imitation labels",
        "materialization": "source references only; Phase17 sidecars are never copied or rewritten",
        "control_dt_s": CONTROL_DT_S,
        "urdf": {"path": str(urdf), "sha256": sha256_file(urdf)},
        "eligibility_contract": {
            "min_root_z_m": min_root_z_m,
            "max_root_tilt_rad": max_root_tilt_rad,
            "min_recorded_continuation_s": min_remaining_s,
            "joint_position_and_velocity": "must lie inside sole12 URDF hard limits; no runtime projection",
            "post_collapse_normal_reset": False,
        },
        "action_semantics": {
            "actual_previous_action_input": "restored actor history only",
            "actual_issued_action": "restored executed/controller history only",
            "imitation_target": False,
        },
        "sources": sources,
        "rows": rows,
        "counts": counts,
        "training_fraction": None,
        "training_unlocked": False,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    validate_manifest(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any], *, validate_sources: bool = True) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported role-aware manifest schema")
    unsigned = dict(manifest)
    digest = unsigned.pop("content_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("role-aware manifest content hash mismatch")
    if manifest["action_semantics"].get("imitation_target") is not False:
        raise ValueError("Phase17 actions must not be imitation labels")
    if set(manifest["counts"]) != set(OUTCOME_ROLES):
        raise ValueError("role-aware manifest does not have all outcome roles")
    if any(int(manifest["counts"][role]["eligible"]) <= 0 for role in OUTCOME_ROLES):
        raise ValueError("role-aware manifest has an outcome role without safe states")
    if not validate_sources:
        return
    sources = manifest["sources"]
    payloads = []
    for index, source in enumerate(sources):
        if int(source["source_index"]) != index:
            raise ValueError("non-contiguous source index")
        path = Path(source["path"])
        if sha256_file(path) != source["file_sha256"]:
            raise ValueError("Phase17 source sidecar file hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_stop_event_sidecar(payload)
        if payload["content_sha256"] != source["content_sha256"]:
            raise ValueError("Phase17 source sidecar content hash mismatch")
        payloads.append(payload)
    for index, ref in enumerate(manifest["rows"]):
        if int(ref["manifest_row_index"]) != index:
            raise ValueError("non-contiguous manifest row index")
        row = payloads[int(ref["source_index"])]["rows"][int(ref["source_row_index"])]
        if row["snapshot_sha256"] != ref["source_snapshot_sha256"]:
            raise ValueError("Phase17 source row hash mismatch")
        if row["controller_state_sha256"] != ref["source_controller_state_sha256"]:
            raise ValueError("Phase17 controller hash mismatch")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise ValueError("role-aware manifest file hash mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def resolve_rows(manifest: Mapping[str, Any], indices: Sequence[int]) -> list[dict[str, Any]]:
    source_cache: dict[int, dict[str, Any]] = {}
    result = []
    for raw_index in indices:
        ref = manifest["rows"][int(raw_index)]
        if not ref["eligible"]:
            raise ValueError(f"attempted to inject ineligible manifest row {raw_index}")
        source_index = int(ref["source_index"])
        if source_index not in source_cache:
            source_cache[source_index] = json.loads(
                Path(manifest["sources"][source_index]["path"]).read_text(encoding="utf-8")
            )
        row = source_cache[source_index]["rows"][int(ref["source_row_index"])]
        if row["snapshot_sha256"] != ref["source_snapshot_sha256"]:
            raise ValueError("resolved Phase17 source row hash mismatch")
        result.append(row)
    return result


def fixed_batch_indices(manifest: Mapping[str, Any], role: str, count: int = 16) -> list[int]:
    if role not in OUTCOME_ROLES:
        raise ValueError(f"unsupported outcome role: {role}")
    eligible = [
        int(row["manifest_row_index"]) for row in manifest["rows"]
        if row["outcome_role"] == role and row["eligible"]
    ]
    if not eligible:
        raise ValueError(f"no eligible states for role {role}")
    positions = np.linspace(0, len(eligible) - 1, count).round().astype(np.int64)
    return [eligible[int(position)] for position in positions]


def reset_from_role_aware_suffix(
    env, env_ids, *, manifest_path: str, reset_fraction: float,
    outcome_role_name: str, expected_manifest_sha256: str | None = None,
    fixed_manifest_indices: Sequence[int] | None = None, asset_name: str = "robot",
) -> None:
    """Final physical reset event. Fraction zero returns before opening manifest."""
    import torch

    if not 0.0 <= reset_fraction <= 1.0:
        raise ValueError("reset_fraction must be in [0, 1]")
    if reset_fraction == 0.0 or len(env_ids) == 0:
        env._x2_role_reset_last = {
            "selected_env_ids": torch.empty(0, dtype=torch.long, device=env.device),
            "manifest_indices": torch.empty(0, dtype=torch.long, device=env.device),
            "strict_no_op": True,
        }
        return
    manifest = load_manifest(manifest_path, expected_manifest_sha256)
    if outcome_role_name not in OUTCOME_ROLES:
        raise ValueError(f"unsupported outcome role: {outcome_role_name}")
    asset = env.scene[asset_name]
    device = asset.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if reset_fraction == 1.0:
        selected = env_ids
    else:
        selected = env_ids[torch.rand(len(env_ids), device=device) < reset_fraction]
    if len(selected) == 0:
        env._x2_role_reset_last = {
            "selected_env_ids": selected,
            "manifest_indices": torch.empty(0, dtype=torch.long, device=device),
            "strict_no_op": False,
        }
        return
    if fixed_manifest_indices is not None:
        indices = list(map(int, fixed_manifest_indices))
        if len(indices) != len(selected):
            raise ValueError("fixed manifest index count differs from selected env count")
    else:
        candidates = [
            int(row["manifest_row_index"]) for row in manifest["rows"]
            if row["outcome_role"] == outcome_role_name and row["eligible"]
        ]
        candidate_tensor = torch.as_tensor(candidates, device=device, dtype=torch.long)
        indices = candidate_tensor[
            torch.randint(len(candidate_tensor), (len(selected),), device=device)
        ].cpu().tolist()
    refs = [manifest["rows"][index] for index in indices]
    if any(ref["outcome_role"] != outcome_role_name or not ref["eligible"] for ref in refs):
        raise ValueError("fixed batch violates role/eligibility contract")
    rows = resolve_rows(manifest, indices)
    names = rows[0]["physical_state"]["joint_names"]
    if any(row["physical_state"]["joint_names"] != names for row in rows):
        raise ValueError("Phase17 rows do not share a joint order")
    from official_x2.recovery_reset_curriculum import joint_reorder_indices
    reorder = joint_reorder_indices(names, asset.joint_names)
    reorder_t = torch.as_tensor(reorder, device=device, dtype=torch.long)
    dtype = asset.data.default_joint_pos.dtype
    q = torch.as_tensor(
        [row["physical_state"]["joint_position_rad"] for row in rows], device=device, dtype=dtype
    )[:, reorder_t]
    dq = torch.as_tensor(
        [row["physical_state"]["joint_velocity_radps"] for row in rows], device=device, dtype=dtype
    )[:, reorder_t]
    limits = asset.data.soft_joint_pos_limits[selected]
    if bool(torch.any(q < limits[..., 0] - 1.0e-6) or torch.any(q > limits[..., 1] + 1.0e-6)):
        raise RuntimeError("eligible Phase17 q violates target Isaac limits; projection forbidden")
    vel_limits = asset.data.soft_joint_vel_limits[selected].clone()
    fallback = vel_limits <= 0.0
    if bool(torch.any(fallback)):
        hard = asset.data.joint_vel_limits[selected]
        if bool(torch.any(hard <= 0.0)):
            raise RuntimeError("target Isaac velocity limits are unavailable")
        vel_limits = torch.where(fallback, hard, vel_limits)
    if bool(torch.any(torch.abs(dq) > vel_limits + 1.0e-6)):
        raise RuntimeError("eligible Phase17 dq violates target Isaac limits; projection forbidden")
    relative_position = torch.as_tensor(
        [row["physical_state"]["root_position_m"] for row in rows], device=device, dtype=dtype
    )
    root_position = env.scene.env_origins[selected] + relative_position
    xyzw = torch.as_tensor(
        [row["physical_state"]["root_quaternion_xyzw"] for row in rows], device=device, dtype=dtype
    )
    wxyz = xyzw[:, [3, 0, 1, 2]]
    root_velocity = torch.as_tensor(
        [
            row["physical_state"]["root_linear_velocity_world_mps"]
            + row["physical_state"]["root_angular_velocity_world_radps"]
            for row in rows
        ], device=device, dtype=dtype,
    )
    asset.write_root_pose_to_sim(torch.cat((root_position, wxyz), dim=-1), env_ids=selected)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=selected)
    asset.write_joint_state_to_sim(q, dq, env_ids=selected)
    manifest_indices_t = torch.as_tensor(indices, device=device, dtype=torch.long)
    env._x2_role_reset_last = {
        "selected_env_ids": selected.clone(),
        "manifest_indices": manifest_indices_t.clone(),
        "strict_no_op": False,
    }
    env._x2_role_aware_pending = {
        "selected_env_ids": selected.clone(),
        "manifest_indices": manifest_indices_t.clone(),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "expected_manifest_sha256": expected_manifest_sha256,
    }


def finalize_role_aware_suffix(env, *, action_name: str = "joint_pos", command_name: str = "base_velocity") -> dict[str, Any]:
    """Restore actor-visible buffers and hash-bound controller snapshot registry."""
    import torch

    pending = getattr(env, "_x2_role_aware_pending", None)
    if pending is None:
        return {"selected_env_count": 0, "no_op": True}
    delattr(env, "_x2_role_aware_pending")
    selected = pending["selected_env_ids"]
    indices = pending["manifest_indices"]
    manifest = load_manifest(pending["manifest_path"], pending["expected_manifest_sha256"])
    rows = resolve_rows(manifest, indices.detach().cpu().tolist())
    if abs(float(env.step_dt) - CONTROL_DT_S) > 1.0e-9:
        raise RuntimeError("role-aware reset requires matched 50 Hz control")
    action_manager = env.action_manager
    action_term = action_manager.get_term(action_name)
    command_term = env.command_manager.get_term(command_name)
    previous = torch.as_tensor(
        [row["actual_previous_action_input"] for row in rows],
        device=action_manager._action.device, dtype=action_manager._action.dtype,
    )
    issued = torch.as_tensor(
        [row["actual_issued_action"] for row in rows],
        device=action_manager._action.device, dtype=action_manager._action.dtype,
    )
    clock = torch.as_tensor(
        [round(float(row["stop_elapsed_s"]) / CONTROL_DT_S) for row in rows],
        device=env.episode_length_buf.device, dtype=env.episode_length_buf.dtype,
    )
    command = torch.as_tensor(
        [row["command_velocity_mps_radps"] for row in rows],
        device=command_term.vel_command_b.device, dtype=command_term.vel_command_b.dtype,
    )
    gait = torch.as_tensor(
        [row["gait_phase"] for row in rows], device=env.device, dtype=torch.float32
    )
    moving = torch.linalg.vector_norm(gait[:, :2], dim=-1) > 0.5
    env.episode_length_buf[selected] = clock
    action_manager._action[selected] = previous
    action_manager._prev_action[selected] = previous
    action_term._raw_actions[selected] = previous
    action_term._combined_normalized_actions[selected] = issued
    action_term._preclip_combined_actions[selected] = issued
    action_term._normalized_template_bias[selected] = issued - previous
    scale = torch.as_tensor(action_term._scale, device=env.device, dtype=previous.dtype)
    offset = torch.as_tensor(action_term._offset, device=env.device, dtype=previous.dtype)
    processed = issued * scale[selected] + offset[selected]
    clip = getattr(action_term, "_clip", None)
    if clip is not None:
        processed = torch.clamp(processed, min=clip[selected, :, 0], max=clip[selected, :, 1])
    action_term._processed_actions[selected] = processed
    command_term.vel_command_b[selected] = command
    command_term.is_standing_env[selected] = ~moving.to(command_term.is_standing_env.device)
    command_term.is_heading_env[selected] = False
    command_term.heading_target[selected] = 0.0
    command_term.time_left[selected] = torch.as_tensor(
        [max(CONTROL_DT_S, 4.0 - float(row["stop_elapsed_s"])) for row in rows],
        device=command_term.time_left.device, dtype=command_term.time_left.dtype,
    )
    command_term.command_counter[selected] = 1
    if not hasattr(env, "_x2_recovery_force_moving"):
        env._x2_recovery_force_moving = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._x2_recovery_force_moving_until_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    env._x2_recovery_force_moving[selected] = moving
    env._x2_recovery_force_moving_until_step[selected] = torch.where(
        moving, clock.to(env.device) + 1, torch.full_like(clock.to(env.device), -1)
    )
    if not hasattr(env, "_x2_role_controller_state_by_env"):
        env._x2_role_controller_state_by_env = {}
        env._x2_role_source_ref_by_env = {}
    for env_id, manifest_index, row in zip(
        selected.detach().cpu().tolist(), indices.detach().cpu().tolist(), rows
    ):
        env._x2_role_controller_state_by_env[int(env_id)] = copy.deepcopy(row["controller_state"])
        env._x2_role_source_ref_by_env[int(env_id)] = {
            "manifest_index": int(manifest_index),
            "snapshot_sha256": row["snapshot_sha256"],
            "controller_state_sha256": row["controller_state_sha256"],
        }
    result = {
        "selected_env_count": int(len(selected)), "no_op": False,
        "episode_clock_steps": clock.clone(), "previous_action": previous.clone(),
        "actual_issued_action": issued.clone(), "command": command.clone(),
        "gait_phase": gait.clone(), "controller_hashes": [row["controller_state_sha256"] for row in rows],
    }
    env._x2_role_reset_last["stateful_finalized"] = True
    env._x2_role_reset_last["stateful_result"] = result
    return result


__all__ = [
    "MANIFEST_SCHEMA", "OUTCOME_ROLES", "build_manifest", "eligibility_reasons",
    "finalize_role_aware_suffix", "fixed_batch_indices", "load_manifest",
    "outcome_role", "reset_from_role_aware_suffix", "resolve_rows", "root_tilt_rad",
    "validate_manifest", "write_manifest",
]
