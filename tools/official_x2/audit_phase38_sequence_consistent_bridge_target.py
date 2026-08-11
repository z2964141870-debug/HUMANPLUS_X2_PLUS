#!/usr/bin/env python3
"""Select a real, sequence-consistent Phase19 success suffix for a BASE bridge.

This is deliberately read-only.  It does not integrate physics, change a
controller, or turn Phase19 controller history into an imitation label.  It
answers the narrower question left by Phase26: which *recorded* success-safe
state is closest to the bridge endpoint while also having one full second of
recorded success-safe continuation?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from official_x2.audit_phase25_role_divergence import calibrate, pair_distance
from official_x2.outcome_aware_state_role_v2 import load_manifest, resolve_rows
from official_x2.replay_official_trace_direct_mujoco import LOWER_JOINTS


CONTROL_DT_S = 0.02
SUFFIX_TICKS = 51  # current tick plus the next 50 ticks = 1.00 s


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_distance(left: np.ndarray, right: np.ndarray, calibration: dict[str, Any]) -> float:
    distance = float(pair_distance(left[None], right[None], calibration["scale"])[0, 0])
    return distance / float(calibration["success_threshold"])


def consecutive_candidates(
    manifest: dict[str, Any], resolved_by_manifest_index: dict[int, dict[str, Any]]
) -> list[tuple[int, list[int]]]:
    eligible = {
        int(ref["manifest_row_index"]): ref
        for ref in manifest["rows"]
        if ref.get("eligible") and ref.get("state_role") == "success_safe"
    }
    by_source_row = {
        (int(ref["source_index"]), int(ref["source_row_index"])): index
        for index, ref in eligible.items()
    }
    result: list[tuple[int, list[int]]] = []
    for index, ref in eligible.items():
        source = int(ref["source_index"])
        source_row = int(ref["source_row_index"])
        indices = [by_source_row.get((source, source_row + offset)) for offset in range(SUFFIX_TICKS)]
        if any(value is None for value in indices):
            continue
        typed = [int(value) for value in indices]
        ticks = [int(resolved_by_manifest_index[value]["source_tick"]) for value in typed]
        if ticks != list(range(ticks[0], ticks[0] + SUFFIX_TICKS)):
            continue
        result.append((index, typed))
    return result


def top_deltas(names: list[str], delta: np.ndarray, count: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(delta))[::-1][:count]
    return [
        {"name": names[int(index)], "delta_target_minus_endpoint": float(delta[int(index)])}
        for index in order
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    eligible_indices = [
        int(ref["manifest_row_index"])
        for ref in manifest["rows"]
        if ref.get("eligible")
    ]
    resolved = resolve_rows(manifest, eligible_indices)
    row_by_index = dict(zip(eligible_indices, resolved))
    success = [row_by_index[int(ref["manifest_row_index"])] for ref in manifest["rows"] if ref.get("eligible") and ref.get("state_role") == "success_safe"]
    critical = [row_by_index[int(ref["manifest_row_index"])] for ref in manifest["rows"] if ref.get("eligible") and ref.get("state_role") == "critical_from_failure"]

    q_extract: Callable[[dict[str, Any]], np.ndarray] = lambda row: np.asarray(row["observation_93d"][12:43], dtype=np.float64)
    action_extract: Callable[[dict[str, Any]], np.ndarray] = lambda row: np.asarray(row["actual_previous_action_input"], dtype=np.float64)
    q_cal = calibrate(np.asarray([q_extract(row) for row in success]), np.asarray([q_extract(row) for row in critical]))
    action_cal = calibrate(np.asarray([action_extract(row) for row in success]), np.asarray([action_extract(row) for row in critical]))

    physical_names = list(success[0]["physical_state"]["joint_names"])
    if len(physical_names) != 31 or any(list(row["physical_state"]["joint_names"]) != physical_names for row in resolved):
        raise ValueError("Phase19 physical joint-name contract is not a stable 31-DoF order")
    defaults = np.asarray([
        np.asarray(row["physical_state"]["joint_position_rad"], dtype=np.float64) - q_extract(row)
        for row in resolved
    ])
    default = np.median(defaults, axis=0)
    default_error = float(np.max(np.abs(defaults - default)))
    if default_error > 1e-6:
        raise ValueError(f"cannot recover one exact default pose: max error {default_error}")

    phase26 = json.loads(args.phase26.read_text(encoding="utf-8"))
    endpoint = phase26["best"]["rollout"]["bridge_rows"][-1]
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    qpos = np.asarray(endpoint["qpos"], dtype=np.float64)
    endpoint_absolute_q = np.asarray([
        qpos[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])]
        for name in physical_names
    ])
    endpoint_q = endpoint_absolute_q - default
    endpoint_action = np.asarray(endpoint["action"], dtype=np.float64)
    if endpoint_action.shape != (15,):
        raise ValueError("Phase26 endpoint action is not the expected lower12+waist3 15-D contract")

    candidates = consecutive_candidates(manifest, row_by_index)
    scored = []
    for manifest_index, suffix_indices in candidates:
        row = row_by_index[manifest_index]
        q_distance = calibration_distance(endpoint_q, q_extract(row), q_cal)
        action_distance = calibration_distance(endpoint_action, action_extract(row), action_cal)
        composite = float(np.sqrt(0.5 * (q_distance**2 + action_distance**2)))
        scored.append((composite, q_distance, action_distance, manifest_index, suffix_indices))
    if not scored:
        raise ValueError("no success-safe row has a one-second consecutive success-safe suffix")
    scored.sort(key=lambda value: (value[0], value[3]))
    composite, q_distance, action_distance, manifest_index, suffix_indices = scored[0]
    target = row_by_index[manifest_index]
    target_q = q_extract(target)
    target_action = action_extract(target)
    q_delta = target_q - endpoint_q
    action_delta = target_action - endpoint_action
    suffix_rows = [row_by_index[index] for index in suffix_indices]

    return {
        "stage": "BASE Phase38 sequence-consistent bridge target audit",
        "execution": {"physics_steps": 0, "optimizer_steps": 0, "training": False, "read_only": True},
        "assets": {
            "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
            "phase26": str(args.phase26), "phase26_sha256": sha256(args.phase26),
            "scene": str(args.scene), "scene_sha256": sha256(args.scene),
        },
        "contracts": {
            "control_dt_s": CONTROL_DT_S,
            "suffix_ticks_including_anchor": SUFFIX_TICKS,
            "suffix_duration_s": (SUFFIX_TICKS - 1) * CONTROL_DT_S,
            "eligible_success_safe_rows": len(success),
            "eligible_critical_rows": len(critical),
            "sequence_consistent_candidate_count": len(candidates),
            "default_pose_reconstruction_max_abs_rad": default_error,
            "distance": "Phase25 robust per-feature scale and success LOO-p95; q/action groups combined with equal RMS weight",
            "action_semantics": "controller previous-action/history state only; not an expert imitation target",
        },
        "selected_target": {
            "manifest_row_index": manifest_index,
            "episode_id": target["episode_id"],
            "source_tick": int(target["source_tick"]),
            "stop_elapsed_s": float(target["stop_elapsed_s"]),
            "snapshot_sha256": target["snapshot_sha256"],
            "controller_state_sha256": target["controller_state_sha256"],
            "composite_normalized_distance": composite,
            "joint_position_normalized_distance": q_distance,
            "previous_action_normalized_distance": action_distance,
            "joint_position_rmse_rad": float(np.sqrt(np.mean(q_delta**2))),
            "previous_action_rmse": float(np.sqrt(np.mean(action_delta**2))),
            "joint_position_top_deltas": top_deltas(physical_names, q_delta),
            "previous_action_top_deltas": top_deltas(list(LOWER_JOINTS), action_delta),
            "target_joint_position_relative_default_rad": target_q.tolist(),
            "target_previous_action": target_action.tolist(),
            "target_actual_issued_action": list(map(float, target["actual_issued_action"])),
        },
        "candidate_frontier": [
            {
                "manifest_row_index": int(item[3]),
                "episode_id": row_by_index[int(item[3])]["episode_id"],
                "source_tick": int(row_by_index[int(item[3])]["source_tick"]),
                "stop_elapsed_s": float(row_by_index[int(item[3])]["stop_elapsed_s"]),
                "composite_normalized_distance": float(item[0]),
                "joint_position_normalized_distance": float(item[1]),
                "previous_action_normalized_distance": float(item[2]),
            }
            for item in scored[:10]
        ],
        "single_group_nearest": {
            "joint_position": {
                "manifest_row_index": int(min(scored, key=lambda item: item[1])[3]),
                "normalized_distance": float(min(scored, key=lambda item: item[1])[1]),
            },
            "previous_action": {
                "manifest_row_index": int(min(scored, key=lambda item: item[2])[3]),
                "normalized_distance": float(min(scored, key=lambda item: item[2])[2]),
            },
        },
        "suffix": {
            "manifest_row_indices": suffix_indices,
            "source_ticks": [int(row["source_tick"]) for row in suffix_rows],
            "snapshot_sha256": [row["snapshot_sha256"] for row in suffix_rows],
            "controller_state_sha256": [row["controller_state_sha256"] for row in suffix_rows],
            "all_success_safe_and_eligible": True,
        },
        "decision": {
            "sequence_consistent_target_available": True,
            "endpoint_already_inside_one_jointly_sequence_consistent_target": bool(composite <= 1.0),
            "direct_bridge_training_or_physics_unlocked": False,
            "next_representation": "track this frozen 51-tick physical/controller suffix with explicit q and previous/issued-action history synchronization",
            "reject": "more population on the old five hand-built q-mode bridge without changing its target representation",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/x2_phase19_outcome_aware_state_role.json"))
    parser.add_argument("--phase26", type=Path, default=Path("reports/baseline/x2_recovery_phase26_bridge_cem.json"))
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/official_x2/phase38_sequence_consistent_bridge_target.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_target": result["selected_target"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
