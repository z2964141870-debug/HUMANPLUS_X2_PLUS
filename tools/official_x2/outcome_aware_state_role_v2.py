#!/usr/bin/env python3
"""Outcome-aware state roles from immutable Phase19 recorder-v2 sidecars.

Episode outcome and reset-state role are intentionally different concepts:
an episode may end in height collapse while an earlier, still-safe state is a
useful critical recovery reset.  Failure actions remain history/provenance and
are never behavior-cloning labels.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from official_x2.recovery_suffix_aggregation import canonical_sha256, sha256_file
from official_x2.stop_event_suffix_contract_v2 import validate_stop_event_sidecar_v2


MANIFEST_SCHEMA = "x2_phase19_outcome_aware_state_role_manifest_v1"
STATE_ROLES = ("success_safe", "critical_from_failure")
CONTROL_DT_S = 0.02


def root_tilt_rad(row: Mapping[str, Any]) -> float:
    x, y, z, w = (float(v) for v in row["physical_state"]["root_quaternion_xyzw"])
    gravity_z = 1.0 - 2.0 * (w * w + z * z)
    gravity_x = 2.0 * (-z * x + w * y)
    gravity_y = -2.0 * (z * y + w * x)
    norm = math.sqrt(gravity_x * gravity_x + gravity_y * gravity_y + gravity_z * gravity_z)
    return math.acos(float(np.clip(-gravity_z / norm, -1.0, 1.0)))


def load_urdf_limits(path: str | Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for joint in ET.parse(Path(path).expanduser().resolve()).getroot().findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint lacks limit: {joint.attrib.get('name')}")
        result[joint.attrib["name"]] = {
            key: float(limit.attrib[key]) for key in ("lower", "upper", "velocity")
        }
    return result


def safe_state_reasons(
    row: Mapping[str, Any], limits: Mapping[str, Mapping[str, float]], *,
    min_root_z_m: float = 0.55, max_root_tilt_rad: float = 0.30,
) -> list[str]:
    physical = row["physical_state"]
    reasons: list[str] = []
    if float(physical["root_position_m"][2]) < min_root_z_m:
        reasons.append("root_z_below_safe_reset_floor")
    if root_tilt_rad(row) > max_root_tilt_rad:
        reasons.append("root_tilt_above_safe_reset_limit")
    for name, q in zip(physical["joint_names"], physical["joint_position_rad"]):
        limit = limits.get(name)
        if limit is None:
            reasons.append("joint_missing_from_urdf")
            break
        if not limit["lower"] - 1.0e-6 <= float(q) <= limit["upper"] + 1.0e-6:
            reasons.append("joint_position_outside_urdf_limit")
            break
    for name, dq in zip(physical["joint_names"], physical["joint_velocity_radps"]):
        limit = limits.get(name)
        if limit is None or abs(float(dq)) > limit["velocity"] + 1.0e-6:
            reasons.append("joint_velocity_outside_urdf_limit")
            break
    return reasons


def _episode_outcome(sidecar: Mapping[str, Any]) -> str:
    outcome = sidecar["episode_outcome"]
    if bool(outcome.get("full_gate_pass")):
        return "full_success"
    if not bool(outcome.get("survived_stop_height_gate")):
        return "height_failure"
    return "critical_episode"


def _first_height_collapse_time(sidecar: Mapping[str, Any], threshold_m: float) -> float | None:
    for row in sidecar["rows"]:
        if float(row["physical_state"]["root_position_m"][2]) < threshold_m:
            return float(row["stop_elapsed_s"])
    return None


def build_manifest(
    sidecar_paths: Sequence[str | Path], urdf_path: str | Path, *,
    min_root_z_m: float = 0.55, max_root_tilt_rad: float = 0.30,
    height_collapse_threshold_m: float = 0.45,
    critical_future_min_s: float = 0.50,
    critical_future_max_s: float = 1.50,
) -> dict[str, Any]:
    urdf = Path(urdf_path).expanduser().resolve()
    limits = load_urdf_limits(urdf)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for source_index, raw_path in enumerate(sidecar_paths):
        path = Path(raw_path).expanduser().resolve()
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        validate_stop_event_sidecar_v2(sidecar)
        outcome = _episode_outcome(sidecar)
        collapse_time = _first_height_collapse_time(sidecar, height_collapse_threshold_m)
        sources.append(
            {
                "source_index": source_index,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "content_sha256": sidecar["content_sha256"],
                "row_count": int(sidecar["row_count"]),
                "episode_outcome": outcome,
                "first_height_collapse_s": collapse_time,
                "immutable": True,
            }
        )
        for source_row_index, row in enumerate(sidecar["rows"]):
            safety_reasons = safe_state_reasons(
                row, limits, min_root_z_m=min_root_z_m,
                max_root_tilt_rad=max_root_tilt_rad,
            )
            elapsed = float(row["stop_elapsed_s"])
            time_to_collapse = None if collapse_time is None else collapse_time - elapsed
            state_role: str | None = None
            role_reasons = list(safety_reasons)
            if outcome == "full_success":
                state_role = "success_safe"
            elif outcome == "height_failure":
                if collapse_time is None:
                    role_reasons.append("height_failure_without_observed_height_crossing")
                elif not critical_future_min_s - 1.0e-9 <= float(time_to_collapse) <= critical_future_max_s + 1.0e-9:
                    role_reasons.append("height_collapse_not_in_future_0p5_to_1p5s")
                else:
                    state_role = "critical_from_failure"
            else:
                role_reasons.append("unsupported_episode_outcome_for_phase20")
            eligible = state_role is not None and not role_reasons
            rows.append(
                {
                    "manifest_row_index": len(rows),
                    "source_index": source_index,
                    "source_row_index": source_row_index,
                    "source_snapshot_sha256": row["snapshot_sha256"],
                    "source_controller_state_sha256": row["controller_state_sha256"],
                    "episode_outcome": outcome,
                    "state_role": state_role,
                    "stop_elapsed_s": elapsed,
                    "time_to_height_collapse_s": time_to_collapse,
                    "root_z_m": float(row["physical_state"]["root_position_m"][2]),
                    "root_tilt_rad": root_tilt_rad(row),
                    "authority_slot": row["authority_slot"],
                    "observation_policy_slot": row["observation_policy_slot"],
                    "eligible": bool(eligible),
                    "ineligible_reasons": role_reasons,
                }
            )
    counts: dict[str, Any] = {}
    for role in STATE_ROLES:
        subset = [row for row in rows if row["state_role"] == role]
        counts[role] = {
            "candidate": len(subset),
            "eligible": sum(bool(row["eligible"]) for row in subset),
            "source_episode_count": len(
                {int(row["source_index"]) for row in subset if row["eligible"]}
            ),
            "eligible_authority_slots": dict(
                Counter(row["authority_slot"] for row in subset if row["eligible"])
            ),
        }
        if counts[role]["eligible"] <= 0:
            raise ValueError(f"Phase20 state role has no eligible rows: {role}")
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "outcome-aware state-role reset curriculum; episode outcome is not state role",
        "materialization": "immutable source references only; Phase19 v2 sidecars are never copied or rewritten",
        "control_dt_s": CONTROL_DT_S,
        "urdf": {"path": str(urdf), "sha256": sha256_file(urdf)},
        "eligibility_contract": {
            "min_root_z_m": min_root_z_m,
            "max_root_tilt_rad": max_root_tilt_rad,
            "height_collapse_threshold_m": height_collapse_threshold_m,
            "critical_future_window_s": [critical_future_min_s, critical_future_max_s],
            "joint_position_and_velocity": "inside sole12 URDF hard limits; no projection",
            "terminal_or_post_collapse_reset": False,
        },
        "role_semantics": {
            "success_safe": "safe state from a full-success episode",
            "critical_from_failure": "currently safe state from a height-failure episode whose first height collapse is 0.5-1.5s ahead",
            "critical_episode_fabricated": False,
        },
        "action_semantics": {
            "actual_previous_action_input": "history/state only",
            "actual_issued_action": "history/state only",
            "failure_action_is_expert_label": False,
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
        raise ValueError("unsupported Phase20 manifest schema")
    unsigned = dict(manifest)
    digest = unsigned.pop("content_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("Phase20 manifest content hash mismatch")
    if set(manifest.get("counts", {})) != set(STATE_ROLES):
        raise ValueError("Phase20 manifest state-role set mismatch")
    if any(int(manifest["counts"][role]["eligible"]) <= 0 for role in STATE_ROLES):
        raise ValueError("Phase20 manifest has an empty state role")
    if manifest["action_semantics"].get("imitation_target") is not False:
        raise ValueError("Phase20 actions must never be imitation targets")
    if manifest["role_semantics"].get("critical_episode_fabricated") is not False:
        raise ValueError("Phase20 must not fabricate a critical episode")
    if not validate_sources:
        return
    sidecars = []
    for index, source in enumerate(manifest["sources"]):
        if int(source["source_index"]) != index:
            raise ValueError("non-contiguous Phase20 source index")
        path = Path(source["path"])
        if sha256_file(path) != source["file_sha256"]:
            raise ValueError("Phase20 source sidecar file hash mismatch")
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        validate_stop_event_sidecar_v2(sidecar)
        if sidecar["content_sha256"] != source["content_sha256"]:
            raise ValueError("Phase20 source content hash mismatch")
        sidecars.append(sidecar)
    for index, ref in enumerate(manifest["rows"]):
        if int(ref["manifest_row_index"]) != index:
            raise ValueError("non-contiguous Phase20 manifest row index")
        row = sidecars[int(ref["source_index"])]["rows"][int(ref["source_row_index"])]
        if row["snapshot_sha256"] != ref["source_snapshot_sha256"]:
            raise ValueError("Phase20 source snapshot hash mismatch")
        if row["controller_state_sha256"] != ref["source_controller_state_sha256"]:
            raise ValueError("Phase20 controller snapshot hash mismatch")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


@lru_cache(maxsize=4)
def _load_manifest_cached(path_string: str, expected_sha256: str | None) -> dict[str, Any]:
    resolved = Path(path_string)
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise ValueError("Phase20 manifest file hash mismatch")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def load_manifest(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    return _load_manifest_cached(resolved, expected_sha256)


@lru_cache(maxsize=8)
def _load_source_cached(path_string: str, expected_file_sha256: str) -> dict[str, Any]:
    path = Path(path_string)
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("Phase20 cached source file hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_stop_event_sidecar_v2(payload)
    return payload


def resolve_rows(manifest: Mapping[str, Any], indices: Sequence[int]) -> list[dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    result = []
    for raw_index in indices:
        ref = manifest["rows"][int(raw_index)]
        if not ref["eligible"]:
            raise ValueError(f"attempted to inject ineligible Phase20 row {raw_index}")
        source_index = int(ref["source_index"])
        if source_index not in cache:
            source = manifest["sources"][source_index]
            cache[source_index] = _load_source_cached(
                source["path"], source["file_sha256"]
            )
        row = cache[source_index]["rows"][int(ref["source_row_index"])]
        if row["snapshot_sha256"] != ref["source_snapshot_sha256"]:
            raise ValueError("resolved Phase20 row hash mismatch")
        result.append(row)
    return result


def fixed_batch_indices(manifest: Mapping[str, Any], role: str, count: int = 16) -> list[int]:
    if role not in STATE_ROLES:
        raise ValueError(f"unsupported Phase20 state role: {role}")
    eligible = [
        int(row["manifest_row_index"]) for row in manifest["rows"]
        if row["state_role"] == role and row["eligible"]
    ]
    if not eligible:
        raise ValueError(f"no eligible Phase20 states for role {role}")
    positions = np.linspace(0, len(eligible) - 1, count).round().astype(np.int64)
    return [eligible[int(position)] for position in positions]


def reset_from_outcome_aware_v2(
    env, env_ids, *, manifest_path: str, reset_fraction: float,
    state_role_name: str, expected_manifest_sha256: str | None = None,
    fixed_manifest_indices: Sequence[int] | None = None, asset_name: str = "robot",
) -> None:
    """Inject physical state; fraction zero returns before I/O or RNG."""
    import torch

    if not 0.0 <= reset_fraction <= 1.0:
        raise ValueError("reset_fraction must be in [0, 1]")
    if reset_fraction == 0.0 or len(env_ids) == 0:
        env._x2_phase20_reset_last = {
            "selected_env_ids": torch.empty(0, dtype=torch.long, device=env.device),
            "manifest_indices": torch.empty(0, dtype=torch.long, device=env.device),
            "strict_no_op": True,
        }
        return
    manifest = load_manifest(manifest_path, expected_manifest_sha256)
    if state_role_name not in (*STATE_ROLES, "balanced"):
        raise ValueError(f"unsupported Phase20 state role: {state_role_name}")
    asset = env.scene[asset_name]
    device = asset.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    selected = env_ids if reset_fraction == 1.0 else env_ids[
        torch.rand(len(env_ids), device=device) < reset_fraction
    ]
    if len(selected) == 0:
        env._x2_phase20_reset_last = {
            "selected_env_ids": selected,
            "manifest_indices": torch.empty(0, dtype=torch.long, device=device),
            "strict_no_op": False,
        }
        return
    intended_roles: list[str]
    if fixed_manifest_indices is None:
        if state_role_name == "balanced":
            role_ids = torch.randint(len(STATE_ROLES), (len(selected),), device=device).cpu().tolist()
            intended_roles = [STATE_ROLES[int(role_id)] for role_id in role_ids]
            by_role = {
                role: [
                    int(row["manifest_row_index"]) for row in manifest["rows"]
                    if row["state_role"] == role and row["eligible"]
                ]
                for role in STATE_ROLES
            }
            indices = [
                int(by_role[role][torch.randint(len(by_role[role]), (1,), device=device).item()])
                for role in intended_roles
            ]
        else:
            intended_roles = [state_role_name] * len(selected)
            candidates = [
                int(row["manifest_row_index"]) for row in manifest["rows"]
                if row["state_role"] == state_role_name and row["eligible"]
            ]
            choices = torch.as_tensor(candidates, device=device, dtype=torch.long)
            indices = choices[torch.randint(len(choices), (len(selected),), device=device)].cpu().tolist()
    else:
        indices = list(map(int, fixed_manifest_indices))
        if len(indices) != len(selected):
            raise ValueError("Phase20 fixed index count differs from selected env count")
        intended_roles = [state_role_name] * len(selected)
    refs = [manifest["rows"][index] for index in indices]
    if any(
        ref["state_role"] != intended_role or not ref["eligible"]
        for ref, intended_role in zip(refs, intended_roles)
    ):
        raise ValueError("Phase20 fixed batch violates role/eligibility contract")
    rows = resolve_rows(manifest, indices)
    from official_x2.recovery_reset_curriculum import joint_reorder_indices
    names = rows[0]["physical_state"]["joint_names"]
    reorder = torch.as_tensor(
        joint_reorder_indices(names, asset.joint_names), device=device, dtype=torch.long
    )
    dtype = asset.data.default_joint_pos.dtype
    q = torch.as_tensor(
        [row["physical_state"]["joint_position_rad"] for row in rows], device=device, dtype=dtype
    )[:, reorder]
    dq = torch.as_tensor(
        [row["physical_state"]["joint_velocity_radps"] for row in rows], device=device, dtype=dtype
    )[:, reorder]
    limits = asset.data.soft_joint_pos_limits[selected]
    if bool(torch.any(q < limits[..., 0] - 1.0e-6) or torch.any(q > limits[..., 1] + 1.0e-6)):
        raise RuntimeError("eligible Phase20 q violates Isaac limits; projection forbidden")
    vel_limits = asset.data.soft_joint_vel_limits[selected].clone()
    fallback = vel_limits <= 0.0
    if bool(torch.any(fallback)):
        vel_limits = torch.where(fallback, asset.data.joint_vel_limits[selected], vel_limits)
    if bool(torch.any(vel_limits <= 0.0)) or bool(torch.any(torch.abs(dq) > vel_limits + 1.0e-6)):
        raise RuntimeError("eligible Phase20 dq violates Isaac limits; projection forbidden")
    relative_position = torch.as_tensor(
        [row["physical_state"]["root_position_m"] for row in rows], device=device, dtype=dtype
    )
    root_position = env.scene.env_origins[selected] + relative_position
    xyzw = torch.as_tensor(
        [row["physical_state"]["root_quaternion_xyzw"] for row in rows], device=device, dtype=dtype
    )
    root_velocity = torch.as_tensor(
        [
            row["physical_state"]["root_linear_velocity_world_mps"]
            + row["physical_state"]["root_angular_velocity_world_radps"]
            for row in rows
        ], device=device, dtype=dtype,
    )
    asset.write_root_pose_to_sim(
        torch.cat((root_position, xyzw[:, [3, 0, 1, 2]]), dim=-1), env_ids=selected
    )
    asset.write_root_velocity_to_sim(root_velocity, env_ids=selected)
    asset.write_joint_state_to_sim(q, dq, env_ids=selected)
    indices_t = torch.as_tensor(indices, device=device, dtype=torch.long)
    env._x2_phase20_reset_last = {
        "selected_env_ids": selected.clone(),
        "manifest_indices": indices_t.clone(),
        "strict_no_op": False,
    }
    env._x2_phase20_pending = {
        "selected_env_ids": selected.clone(),
        "manifest_indices": indices_t.clone(),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "expected_manifest_sha256": expected_manifest_sha256,
    }


def finalize_outcome_aware_v2(
    env, *, action_name: str = "joint_pos", command_name: str = "base_velocity"
) -> dict[str, Any]:
    """Restore exact first-observation sources and controller/history state."""
    import torch

    pending = getattr(env, "_x2_phase20_pending", None)
    if pending is None:
        return {"selected_env_count": 0, "no_op": True}
    delattr(env, "_x2_phase20_pending")
    selected = pending["selected_env_ids"]
    indices = pending["manifest_indices"]
    manifest = load_manifest(pending["manifest_path"], pending["expected_manifest_sha256"])
    rows = resolve_rows(manifest, indices.detach().cpu().tolist())
    if abs(float(env.step_dt) - CONTROL_DT_S) > 1.0e-9:
        raise RuntimeError("Phase20 reset requires matched 50 Hz control")
    action_manager = env.action_manager
    action_term = action_manager.get_term(action_name)
    command_term = env.command_manager.get_term(command_name)
    previous = torch.as_tensor(
        [row["actual_previous_action_input"] for row in rows],
        device=env.device, dtype=action_manager._action.dtype,
    )
    issued = torch.as_tensor(
        [row["actual_issued_action"] for row in rows],
        device=env.device, dtype=action_manager._action.dtype,
    )
    command = torch.as_tensor(
        [row["command_velocity_mps_radps"] for row in rows],
        device=env.device, dtype=command_term.vel_command_b.dtype,
    )
    clock = torch.as_tensor(
        [round(float(row["stop_elapsed_s"]) / CONTROL_DT_S) for row in rows],
        device=env.episode_length_buf.device, dtype=env.episode_length_buf.dtype,
    )
    expected_obs = torch.as_tensor(
        [row["observation_93d"] for row in rows], device=env.device, dtype=torch.float32
    )
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
    command_term.is_standing_env[selected] = torch.linalg.vector_norm(command[:, :2], dim=-1) <= 0.1
    command_term.is_heading_env[selected] = False
    command_term.heading_target[selected] = 0.0
    command_term.time_left[selected] = torch.as_tensor(
        [max(CONTROL_DT_S, 4.0 - float(row["stop_elapsed_s"])) for row in rows],
        device=command_term.time_left.device, dtype=command_term.time_left.dtype,
    )
    command_term.command_counter[selected] = 1

    # Exact source values are active only for reset-return.  After the first
    # simulator step the actor sees the live Isaac state again.
    if not hasattr(env, "_x2_phase20_obs_override"):
        env._x2_phase20_obs_override = {
            key: torch.zeros((env.num_envs, size), device=env.device, dtype=torch.float32)
            for key, size in (("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("gait_phase", 4))
        }
        env._x2_phase20_obs_override_until_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    for key, section in (
        ("base_lin_vel", slice(0, 3)),
        ("base_ang_vel", slice(3, 6)),
        ("projected_gravity", slice(6, 9)),
        ("gait_phase", slice(89, 93)),
    ):
        env._x2_phase20_obs_override[key][selected] = expected_obs[:, section]
    env._x2_phase20_obs_override_until_step[selected] = clock.to(env.device)

    if not hasattr(env, "_x2_phase20_controller_state_by_env"):
        env._x2_phase20_controller_state_by_env = {}
        env._x2_phase20_actor_source_by_env = {}
        env._x2_phase20_source_ref_by_env = {}
    for env_id, manifest_index, row in zip(
        selected.detach().cpu().tolist(), indices.detach().cpu().tolist(), rows
    ):
        env._x2_phase20_controller_state_by_env[int(env_id)] = copy.deepcopy(row["controller_state"])
        env._x2_phase20_actor_source_by_env[int(env_id)] = copy.deepcopy(row["actor_observation_state"])
        env._x2_phase20_source_ref_by_env[int(env_id)] = {
            "manifest_index": int(manifest_index),
            "snapshot_sha256": row["snapshot_sha256"],
            "controller_state_sha256": row["controller_state_sha256"],
            "actor_observation_state_sha256": row["actor_observation_state_sha256"],
        }
    result = {
        "selected_env_count": int(len(selected)), "no_op": False,
        "episode_clock_steps": clock.clone(), "previous_action": previous.clone(),
        "actual_issued_action": issued.clone(), "command": command.clone(),
        "expected_observation_93d": expected_obs.clone(),
    }
    env._x2_phase20_reset_last["stateful_finalized"] = True
    env._x2_phase20_reset_last["stateful_result"] = result
    return result


__all__ = [
    "CONTROL_DT_S", "MANIFEST_SCHEMA", "STATE_ROLES", "build_manifest",
    "finalize_outcome_aware_v2", "fixed_batch_indices", "load_manifest",
    "reset_from_outcome_aware_v2", "resolve_rows", "root_tilt_rad",
    "safe_state_reasons", "validate_manifest", "write_manifest",
]
