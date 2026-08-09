#!/usr/bin/env python3
"""Read-only common-instability attribution for BASE Phase21 official traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median, pvariance

import numpy as np

from official_x2.audit_phase14_recovery_ood import (
    ALL_GROUPS,
    FEATURE_NAMES,
    ISAAC_JOINTS,
    LOWER_JOINTS,
    OBS_GROUPS,
    group_pair_distance,
    load_stage335,
    robust_center_scale,
    root_state,
)
from official_x2.outcome_aware_state_role_v2 import load_manifest, resolve_rows


EXPECTED_SLOT_COUNTS = {"main": 360, "stationary": 100, "recovery": 300}
HANDOFF_S = 2.0
DT_S = 0.02
LOWER_DEFAULT = dict(zip(LOWER_JOINTS, (
    -0.248, 0.0, 0.0, 0.5303, -0.2823, 0.0,
    -0.248, 0.0, 0.0, 0.5303, -0.2823, 0.0,
    0.0, 0.0, 0.0,
)))
JOINT_GROUPS = {
    "left_leg": tuple(name for name in LOWER_JOINTS if name.startswith("left_")),
    "right_leg": tuple(name for name in LOWER_JOINTS if name.startswith("right_")),
    "waist": tuple(name for name in LOWER_JOINTS if name.startswith("waist_")),
    "upper_head": tuple(name for name in ISAAC_JOINTS if name not in LOWER_JOINTS),
}
PHASE21_LABELS = ("source", "f000", "f005")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float | None]) -> dict:
    present = [float(value) for value in values if value is not None]
    return {
        "count": len(present),
        "median": median(present) if present else None,
        "population_variance": pvariance(present) if len(present) >= 2 else 0.0 if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }


def _phase(row: dict) -> str:
    left, right = float(row["obs"][91]), float(row["obs"][92])
    if left > 0.0 and right > 0.0:
        return "generator_double_support"
    if left > 0.0:
        return "generator_left_support"
    if right > 0.0:
        return "generator_right_support"
    return "generator_flight_or_transition"


def _first_time(rows: list[dict], predicate) -> float | None:
    for row in rows:
        if predicate(row):
            return float(row["elapsed_s"]) - HANDOFF_S
    return None


def _nearest_phase(rows: list[dict], time_s: float | None) -> str | None:
    if time_s is None:
        return None
    row = min(
        rows,
        key=lambda item: abs(
            float(item.get("_phase22_relative", float(item["elapsed_s"]) - HANDOFF_S))
            - time_s
        ),
    )
    return _phase(row)


def _actual_lower_q(row: dict) -> np.ndarray:
    rel = np.asarray(row["obs"][12:43], dtype=np.float64)
    values = []
    for name in LOWER_JOINTS:
        values.append(rel[ISAAC_JOINTS.index(name)] + LOWER_DEFAULT[name])
    return np.asarray(values)


def _groups_from_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    obs = np.asarray([row["obs"] for row in rows], dtype=np.float64)
    result = {name: obs[:, source_slice] for name, source_slice in OBS_GROUPS.items()}
    result["current_action"] = np.asarray([row["action"] for row in rows], dtype=np.float64)
    result["root_state"] = np.asarray([root_state(row) for row in rows], dtype=np.float64)
    return result


def load_v2_reference(manifest_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest = load_manifest(manifest_path)
    indices = [int(row["manifest_row_index"]) for row in manifest["rows"] if row["eligible"]]
    rows = resolve_rows(manifest, indices)
    obs = np.asarray([row["observation_93d"] for row in rows], dtype=np.float64)
    groups = {name: obs[:, source_slice] for name, source_slice in OBS_GROUPS.items()}
    groups["current_action"] = np.asarray(
        [row["actual_issued_action"] for row in rows], dtype=np.float64
    )
    groups["root_state"] = np.asarray(
        [
            [
                row["physical_state"]["root_position_m"][2],
                # Recorder-v2 already stores the same signed pitch and gravity
                # sources used by actor reconstruction.  Tilt is recovered
                # from projected gravity without inventing a root quaternion.
                math.acos(float(np.clip(-row["observation_93d"][8], -1.0, 1.0))),
                math.asin(float(np.clip(row["observation_93d"][6], -1.0, 1.0))),
                math.hypot(*row["physical_state"]["root_linear_velocity_world_mps"][:2]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return groups, {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "rows": len(rows),
        "actor_input": "recorder-v2 exact observation_93d consumed by actor",
        "contact_boundary": "gait_phase/contact generator state, not physical foot force or hardware contact",
    }


def reference_contract(reference: dict[str, np.ndarray]) -> dict[str, dict]:
    result = {}
    for name in ALL_GROUPS:
        center, scale = robust_center_scale(reference[name])
        pair = group_pair_distance(reference[name], reference[name], scale)
        np.fill_diagonal(pair, np.inf)
        result[name] = {
            "center": center,
            "scale": scale,
            "loo_p95": float(np.percentile(pair.min(axis=1), 95.0)),
            "p01": np.percentile(reference[name], 1.0, axis=0),
            "p99": np.percentile(reference[name], 99.0, axis=0),
        }
    return result


def group_first_ood(rows: list[dict], reference: dict[str, np.ndarray], contract: dict[str, dict]) -> dict[str, float | None]:
    query = _groups_from_rows(rows)
    result = {}
    for name in ALL_GROUPS:
        spec = contract[name]
        distance = group_pair_distance(query[name], reference[name], spec["scale"]).min(axis=1)
        index = next((i for i, value in enumerate(distance) if value > spec["loo_p95"]), None)
        result[name] = None if index is None else float(rows[index]["elapsed_s"]) - HANDOFF_S
    return result


def joint_group_first_exit(rows: list[dict], contract: dict[str, dict], field: str) -> dict[str, float | None]:
    query = _groups_from_rows(rows)[field]
    low, high = contract[field]["p01"], contract[field]["p99"]
    names = ISAAC_JOINTS
    result = {}
    for group, group_names in JOINT_GROUPS.items():
        indices = [names.index(name) for name in group_names]
        first = None
        for row_index, values in enumerate(query[:, indices]):
            if np.any(values < low[indices]) or np.any(values > high[indices]):
                first = float(rows[row_index]["elapsed_s"]) - HANDOFF_S
                break
        result[group] = first
    return result


def first_step_exceeding_pre_max(stop: list[dict], field: str, transform=lambda value: np.asarray(value, dtype=np.float64)) -> tuple[float | None, float, float]:
    pre = [row for row in stop if HANDOFF_S - 1.0 - 1e-9 <= float(row["elapsed_s"]) < HANDOFF_S]
    post = [row for row in stop if float(row["elapsed_s"]) >= HANDOFF_S - DT_S - 1e-9]
    pre_steps = [float(np.linalg.norm(transform(right[field]) - transform(left[field]))) for left, right in zip(pre, pre[1:])]
    threshold = max(pre_steps) if pre_steps else 0.0
    first = None
    peak = 0.0
    for left, right in zip(post, post[1:]):
        value = float(np.linalg.norm(transform(right[field]) - transform(left[field])))
        peak = max(peak, value)
        if first is None and value > threshold + 1.0e-12:
            first = float(right["elapsed_s"]) - HANDOFF_S
    return first, threshold, peak


def audit_episode(path: Path, label: str, v2_ref: dict, v2_contract: dict, stage335_ref: dict, stage335_contract: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if summary.get("policy_slot_inference_counts") != EXPECTED_SLOT_COUNTS:
        raise ValueError(f"slot mismatch: {path}")
    trace = payload["trace"]
    stop = sorted((row for row in trace if row["stage"] == "stop"), key=lambda row: float(row["elapsed_s"]))
    recovery = [row for row in stop if float(row["elapsed_s"]) >= HANDOFF_S - 1.0e-9]
    if not recovery or abs(float(recovery[0]["elapsed_s"]) - HANDOFF_S) > 1.0e-8:
        raise ValueError(f"handoff not bracketed: {path}")

    action_first, action_pre_max, action_peak = first_step_exceeding_pre_max(stop, "action")
    target_first, target_pre_max, target_peak = first_step_exceeding_pre_max(stop, "physical_lower_target_rad")
    post_target_pairs = [
        (left, right)
        for left, right in zip(stop, stop[1:])
        if float(right["elapsed_s"]) >= HANDOFF_S - 1.0e-9
    ]
    peak_target_delta = max(
        (
            np.asarray(right["physical_lower_target_rad"], dtype=np.float64)
            - np.asarray(left["physical_lower_target_rad"], dtype=np.float64)
            for left, right in post_target_pairs
        ),
        key=lambda value: float(np.linalg.norm(value)),
    )
    peak_target_by_group = {
        "left_leg": float(np.linalg.norm(peak_target_delta[:6])),
        "right_leg": float(np.linalg.norm(peak_target_delta[6:12])),
        "waist": float(np.linalg.norm(peak_target_delta[12:])),
    }
    errors = [np.asarray(row["physical_lower_target_rad"]) - _actual_lower_q(row) for row in stop]
    pre_error = [float(np.linalg.norm(error)) for row, error in zip(stop, errors) if HANDOFF_S - 1.0 <= float(row["elapsed_s"]) < HANDOFF_S]
    tracking_limit = max(pre_error)
    tracking_first = next((float(row["elapsed_s"]) - HANDOFF_S for row, error in zip(stop, errors) if float(row["elapsed_s"]) >= HANDOFF_S and np.linalg.norm(error) > tracking_limit + 1e-12), None)

    move = [row for row in trace if row["stage"] == "move"]
    origin = move[0]
    yaw0 = float(origin["root_yaw_rad"])
    x0, y0 = float(origin["root_x_m"]), float(origin["root_y_m"])
    timeline = move + stop
    for row in move:
        row["_phase22_relative"] = float(row["elapsed_s"]) - float(move[-1]["elapsed_s"]) - HANDOFF_S
    for row in stop:
        row["_phase22_relative"] = float(row["elapsed_s"]) - HANDOFF_S
    heading_first = next((float(row["_phase22_relative"]) for row in timeline if abs(math.atan2(math.sin(float(row["root_yaw_rad"])-yaw0), math.cos(float(row["root_yaw_rad"])-yaw0))) > 0.30), None)
    lateral_first = next((float(row["_phase22_relative"]) for row in timeline if abs(-(float(row["root_x_m"])-x0)*math.sin(yaw0)+(float(row["root_y_m"])-y0)*math.cos(yaw0)) > 0.30), None)

    physical = {
        "root_tilt_gt_0p30": _first_time(recovery, lambda row: float(row["root_tilt_rad"]) > 0.30),
        "root_z_lt_0p45": _first_time(recovery, lambda row: float(row["root_z_m"]) < 0.45),
        "heading_gt_0p30_from_move_start": heading_first,
        "lateral_gt_0p30_from_move_start": lateral_first,
    }
    v2_ood = group_first_ood(recovery, v2_ref, v2_contract)
    old_ood = group_first_ood(recovery, stage335_ref, stage335_contract)
    q_exit = joint_group_first_exit(recovery, v2_contract, "joint_position")
    dq_exit = joint_group_first_exit(recovery, v2_contract, "joint_velocity")
    events = {
        "heading_official_threshold": heading_first,
        "lateral_official_threshold": lateral_first,
        "action_step_above_pre_handoff_max": action_first,
        "target_step_above_pre_handoff_max": target_first,
        "pd_tracking_error_above_pre_handoff_max": tracking_first,
        "v2_previous_action_ood": v2_ood["previous_action"],
        "v2_current_action_ood": v2_ood["current_action"],
        "v2_gait_contact_generator_ood": v2_ood["gait_phase"],
        "v2_projected_gravity_imu_ood": v2_ood["projected_gravity"],
        "v2_base_angular_velocity_imu_ood": v2_ood["base_angular_velocity"],
        "root_tilt_gt_0p30": physical["root_tilt_gt_0p30"],
        "root_z_lt_0p45": physical["root_z_lt_0p45"],
    }
    sequence = [
        {"event": name, "time_after_handoff_s": value, "support_phase": _nearest_phase(timeline, value)}
        for name, value in sorted(events.items(), key=lambda item: float("inf") if item[1] is None else item[1])
        if value is not None
    ]
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "valid": True,
        "subgates": {key: bool(summary.get(f"{key}_gate_pass")) for key in ("stand", "startup", "move", "stop")},
        "handoff_state": {"root_z_m": recovery[0]["root_z_m"], "root_tilt_rad": recovery[0]["root_tilt_rad"], "support_phase": _phase(recovery[0])},
        "step_contract": {
            "action_first_above_pre_max_s": action_first, "action_pre_max_l2": action_pre_max, "action_post_peak_l2": action_peak,
            "target_first_above_pre_max_s": target_first, "target_pre_max_l2_rad": target_pre_max, "target_post_peak_l2_rad": target_peak,
            "pd_tracking_first_above_pre_max_s": tracking_first, "pd_tracking_pre_max_l2_rad": tracking_limit,
            "post_handoff_peak_target_step_by_joint_group_l2_rad": peak_target_by_group,
        },
        "physical_first": physical,
        "v2_actor_source_first_ood": v2_ood,
        "stage335_first_ood": old_ood,
        "v2_joint_group_p01_p99_first_exit": {"q": q_exit, "dq": dq_exit},
        "event_sequence": sequence,
        "upper_body": {"fixed_target_excursion_abs_max_rad": summary.get("upper_target_excursion_abs_max_rad"), "tracking_rmse_rad": summary.get("upper_tracking_rmse_rad")},
    }


def build_report(result_root: Path, manifest: Path, dataset: Path, source_report: Path) -> dict:
    v2_ref, v2_provenance = load_v2_reference(manifest)
    stage335_ref, _labels, stage335_provenance = load_stage335(dataset, source_report)
    v2_contract, old_contract = reference_contract(v2_ref), reference_contract(stage335_ref)
    episodes = []
    for label in PHASE21_LABELS:
        paths = sorted(result_root.glob(f"phase21_outcome_aware_{label}_matched_blend0p5_stiff1p2_fixed_r*.json"))
        if len(paths) != 5:
            raise ValueError(f"expected 5 Phase21 {label} traces, got {len(paths)}")
        episodes.extend(audit_episode(path, label, v2_ref, v2_contract, stage335_ref, old_contract) for path in paths)
    event_names = tuple(episodes[0]["event_sequence"][i]["event"] for i in range(min(3, len(episodes[0]["event_sequence"]))))
    groups = {}
    all_event_names = sorted({item["event"] for episode in episodes for item in episode["event_sequence"]})
    for label in PHASE21_LABELS:
        subset = [episode for episode in episodes if episode["label"] == label]
        event_times = {
            name: _summary([
                next((item["time_after_handoff_s"] for item in episode["event_sequence"] if item["event"] == name), None)
                for episode in subset
            ])
            for name in all_event_names
        }
        groups[label] = {
            "episodes": 5,
            "event_times_after_handoff_s": event_times,
            "first_three_sequences": [[item["event"] for item in episode["event_sequence"][:3]] for episode in subset],
            "support_phase_at_tilt": [next((item["support_phase"] for item in episode["event_sequence"] if item["event"] == "root_tilt_gt_0p30"), None) for episode in subset],
            "v2_joint_group_p01_p99_first_exit_s": {
                field: {
                    joint_group: _summary([
                        episode["v2_joint_group_p01_p99_first_exit"][field][joint_group]
                        for episode in subset
                    ])
                    for joint_group in JOINT_GROUPS
                }
                for field in ("q", "dq")
            },
            "post_handoff_peak_target_step_by_joint_group_l2_rad": {
                joint_group: _summary([
                    episode["step_contract"]["post_handoff_peak_target_step_by_joint_group_l2_rad"][joint_group]
                    for episode in subset
                ])
                for joint_group in ("left_leg", "right_leg", "waist")
            },
        }
    return {
        "stage": "BASE Phase22 common-instability attribution",
        "analysis_only": True,
        "causal_claim": False,
        "trace_count": len(episodes),
        "references": {"recorder_v2": v2_provenance, "stage335": stage335_provenance},
        "threshold_contract": {
            "root_and_world": "existing official gates only: z 0.45m, tilt/heading/lateral 0.30",
            "actor_groups": "reference LOO nearest-neighbor p95, robust median/IQR scaling, same as Phase14",
            "joint_fields": "recorder-v2 reference p01-p99, same descriptive coverage convention as Phase14",
            "step": "first post-handoff one-tick L2 exceeding same-episode pre-handoff 1s maximum; no tuned multiplier",
        },
        "groups": groups,
        "episodes": episodes,
        "interpretation": {
            "candidate_changes_binary_outcome_or_event_order": False,
            "post_fall_smaller_drift_is_recovery": False,
            "physical_contact_observed": False,
            "contact_field_is_generator_state_only": True,
            "upper_body_is_common_cause_supported": False,
            "pd_is_causal_supported": False,
            "correlation_not_causation": True,
        },
        "next_single_falsifiable_intervention": {
            "name": "support_and_actor-history_gated_recovery_authority",
            "control": "existing fixed 2.0s recovery authority handoff",
            "candidate": "after 2.0s, retain existing brake until generator double-support AND previous-action is within frozen recorder-v2 LOO-p95 support; then apply the same 0.5s blend and same recovery actor",
            "unchanged": "models, PD, command, stop trajectory, reward, training, thresholds",
            "falsification": "if tilt/root-z sequence and collapse remain unchanged despite delayed in-support handoff, previous-action/support timing is not the dominant cause",
            "training_required": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/x2_phase19_outcome_aware_state_role.json"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"))
    parser.add_argument("--source-report", type=Path, default=Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.result_root, args.manifest, args.dataset, args.source_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trace_count": report["trace_count"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
