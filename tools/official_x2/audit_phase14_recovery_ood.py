#!/usr/bin/env python3
"""Pure-offline OOD audit for BASE Phase13 recovery handoff traces.

Distances are computed per semantic group after robust feature scaling and are
normalized by group width.  The composite distance gives every semantic group
equal weight, preventing 31 joint coordinates from silently dominating three
base-velocity coordinates.  Eventual-pass/fail labels describe the immutable
source episode continuation, not a causal state-success oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np


OBS_GROUPS = {
    "base_linear_velocity": slice(0, 3),
    "base_angular_velocity": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "command": slice(9, 12),
    "joint_position": slice(12, 43),
    "joint_velocity": slice(43, 74),
    "previous_action": slice(74, 89),
    "gait_phase": slice(89, 93),
}
EXTRA_GROUPS = ("current_action", "root_state")
ALL_GROUPS = tuple(OBS_GROUPS) + EXTRA_GROUPS
EXPECTED_SLOT_COUNTS = {"main": 360, "stationary": 100, "recovery": 300}
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
FEATURE_NAMES = {
    "base_linear_velocity": ("body_vx", "body_vy", "body_vz"),
    "base_angular_velocity": ("body_wx", "body_wy", "body_wz"),
    "projected_gravity": ("gravity_x", "gravity_y", "gravity_z"),
    "command": ("command_vx", "command_vy", "command_wz"),
    "joint_position": tuple(f"{name}:position_rel" for name in ISAAC_JOINTS),
    "joint_velocity": tuple(f"{name}:velocity" for name in ISAAC_JOINTS),
    "previous_action": tuple(f"{name}:previous_action" for name in LOWER_JOINTS),
    "gait_phase": ("phase_sin", "phase_cos", "left_contact", "right_contact"),
    "current_action": tuple(f"{name}:current_action" for name in LOWER_JOINTS),
    "root_state": ("root_z", "root_tilt", "root_pitch", "root_horizontal_speed"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_state(row: dict) -> np.ndarray:
    speed = math.hypot(float(row["root_vx_w_mps"]), float(row["root_vy_w_mps"]))
    if "root_pitch_rad" in row:
        pitch = float(row["root_pitch_rad"])
    else:
        # Older immutable Stage326 traces predate the explicit signed-pitch
        # field. Recover the same ZYX pitch used by Stage335 extraction from
        # projected gravity; do not invent a quaternion.
        gravity = np.asarray(row["obs"][6:9], dtype=np.float64)
        pitch = math.asin(float(np.clip(gravity[0] / np.linalg.norm(gravity), -1.0, 1.0)))
    return np.asarray(
        [row["root_z_m"], row["root_tilt_rad"], pitch, speed],
        dtype=np.float64,
    )


def robust_center_scale(reference: np.ndarray, *, scale_floor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    center = np.median(reference, axis=0)
    q25, q75 = np.percentile(reference, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    std = np.std(reference, axis=0)
    scale = np.where(scale > 1.0e-8, scale, std)
    if scale_floor is not None:
        scale = np.maximum(scale, np.asarray(scale_floor, dtype=np.float64))
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return center, scale


def group_pair_distance(query: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> np.ndarray:
    delta = (query[:, None, :] - reference[None, :, :]) / scale[None, None, :]
    return np.sqrt(np.mean(delta * delta, axis=2))


def nearest_and_loo(query: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_distance = group_pair_distance(query, reference, scale).min(axis=1)
    pair = group_pair_distance(reference, reference, scale)
    np.fill_diagonal(pair, np.inf)
    return query_distance, pair.min(axis=1)


def composite_pair_distance(
    query_groups: dict[str, np.ndarray],
    reference_groups: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    names: Iterable[str],
) -> np.ndarray:
    per_group = [
        group_pair_distance(query_groups[name], reference_groups[name], scales[name])
        for name in names
    ]
    return np.sqrt(np.mean(np.stack(per_group, axis=0) ** 2, axis=0))


def composite_nearest_and_loo(
    query_groups: dict[str, np.ndarray],
    reference_groups: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    query_pair = composite_pair_distance(query_groups, reference_groups, scales, names)
    ref_pair = composite_pair_distance(reference_groups, reference_groups, scales, names)
    np.fill_diagonal(ref_pair, np.inf)
    return query_pair.min(axis=1), ref_pair.min(axis=1)


def load_stage335(dataset: Path, report_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    archive = np.load(dataset, allow_pickle=False)
    if tuple(archive["joint_names"].tolist()) != ISAAC_JOINTS:
        raise ValueError("Stage335 joint order differs from the 93D feature contract")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256(dataset) != report["dataset_sha256"]:
        raise ValueError("Stage335 dataset SHA mismatch")
    sources = report["source_files"]
    payloads = []
    for source in sources:
        path = Path(source["path"])
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Stage335 source SHA mismatch: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    obs_rows, action_rows, root_rows = [], [], []
    for row, (source_index, trace_index) in enumerate(
        zip(archive["source_index"], archive["trace_index"])
    ):
        item = payloads[int(source_index)]["trace"][int(trace_index)]
        obs = np.asarray(item["obs"], dtype=np.float64)
        action = np.asarray(item["action"], dtype=np.float64)
        if obs.shape != (93,) or action.shape != (15,):
            raise ValueError(f"invalid Stage335 source row {row}")
        # Re-validate every immutable slice stored in the NPZ.
        checks = (
            (slice(0, 3), "base_lin_vel_body_mps"),
            (slice(3, 6), "base_ang_vel_body_radps"),
            (slice(6, 9), "projected_gravity"),
            (slice(12, 43), "joint_pos_rel_rad"),
            (slice(43, 74), "joint_vel_radps"),
            (slice(74, 89), "previous_action"),
            (slice(89, 93), "gait_phase"),
        )
        for source_slice, key in checks:
            if np.max(np.abs(obs[source_slice] - archive[key][row])) > 1.0e-6:
                raise ValueError(f"Stage335 observation rejoin mismatch row={row} key={key}")
        obs_rows.append(obs)
        action_rows.append(action)
        root_rows.append(root_state(item))
    obs = np.stack(obs_rows)
    groups = {name: obs[:, source_slice] for name, source_slice in OBS_GROUPS.items()}
    groups["current_action"] = np.stack(action_rows)
    groups["root_state"] = np.stack(root_rows)
    labels = archive["eventual_pass"].astype(bool)
    provenance = {
        "dataset": str(dataset),
        "dataset_sha256": report["dataset_sha256"],
        "source_report": str(report_path),
        "state_count": int(len(labels)),
        "eventual_pass": int(labels.sum()),
        "eventual_fail": int((~labels).sum()),
    }
    return groups, labels, provenance


def load_phase13(
    result_root: Path,
) -> tuple[dict[str, np.ndarray], list[dict], dict[str, np.ndarray], dict]:
    query_obs, query_action, query_root, meta = [], [], [], []
    stand_obs, stand_action, stand_root = [], [], []
    query_paths, excluded = [], []
    for repeat in range(1, 6):
        path = result_root / f"phase13_source_recovery_blend0p5_stiff1p2_fixed_r{repeat}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        if summary.get("policy_slot_inference_counts") != EXPECTED_SLOT_COUNTS:
            raise ValueError(f"Phase13 role-count mismatch: {path}")
        healthy = bool(summary.get("stand_gate_pass") and summary.get("startup_gate_pass"))
        if not healthy:
            excluded.append({"path": str(path), "reason": "pre-handoff stand/start failure"})
            continue
        query_paths.append({"path": str(path), "sha256": sha256(path)})
        for row in payload["trace"]:
            if row["stage"] == "stop" and 2.0 - 1e-9 <= float(row["elapsed_s"]) <= 3.5 + 1e-9:
                obs = np.asarray(row["obs"], dtype=np.float64)
                action = np.asarray(row["action"], dtype=np.float64)
                if obs.shape != (93,) or action.shape != (15,):
                    raise ValueError(f"invalid Phase13 query row: {path}")
                query_obs.append(obs)
                query_action.append(action)
                query_root.append(root_state(row))
                meta.append({
                    "episode": repeat,
                    "time_after_handoff_s": max(0.0, float(row["elapsed_s"]) - 2.0),
                    "root_z_m": float(row["root_z_m"]),
                    "root_tilt_rad": float(row["root_tilt_rad"]),
                })
    # Source stand stable support: final 1.0 s of every healthy blend=0 control stand.
    stand_paths = []
    for repeat in range(1, 6):
        path = result_root / f"phase13_source_recovery_blend0p0_stiff1p2_fixed_r{repeat}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload["summary"].get("stand_gate_pass"):
            continue
        stand_paths.append({"path": str(path), "sha256": sha256(path)})
        for row in payload["trace"]:
            if row["stage"] == "stand" and float(row["elapsed_s"]) >= 1.0 - 1e-9:
                stand_obs.append(np.asarray(row["obs"], dtype=np.float64))
                stand_action.append(np.asarray(row["action"], dtype=np.float64))
                stand_root.append(root_state(row))
    query_obs_array = np.stack(query_obs)
    stand_obs_array = np.stack(stand_obs)
    query_groups = {name: query_obs_array[:, source_slice] for name, source_slice in OBS_GROUPS.items()}
    query_groups["current_action"] = np.stack(query_action)
    query_groups["root_state"] = np.stack(query_root)
    stand_groups = {name: stand_obs_array[:, source_slice] for name, source_slice in OBS_GROUPS.items()}
    stand_groups["current_action"] = np.stack(stand_action)
    stand_groups["root_state"] = np.stack(stand_root)
    provenance = {
        "healthy_query_episodes": len(query_paths),
        "query_rows": len(meta),
        "query_paths": query_paths,
        "excluded_query_episodes": excluded,
        "stand_rows": len(stand_obs),
        "stand_paths": stand_paths,
    }
    return query_groups, meta, stand_groups, provenance


def summarize_distance(distance: np.ndarray, threshold: float, meta: list[dict]) -> dict:
    ood = distance > threshold
    first_by_episode = {}
    for index, item in enumerate(meta):
        episode = str(item["episode"])
        if episode not in first_by_episode and ood[index]:
            first_by_episode[episode] = float(item["time_after_handoff_s"])
    return {
        "reference_loo_p95_threshold": float(threshold),
        "query_distance_median": float(np.median(distance)),
        "query_distance_p95": float(np.percentile(distance, 95.0)),
        "query_distance_max": float(distance.max()),
        "query_ood_fraction": float(ood.mean()),
        "first_ood_s_by_episode": first_by_episode,
        "first_ood_s_median": median(first_by_episode.values()) if first_by_episode else None,
    }


def analyze_reference(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    meta: list[dict],
    *,
    global_floors: dict[str, np.ndarray] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    scales = {}
    group_results = {}
    for name in ALL_GROUPS:
        floor = None if global_floors is None else global_floors[name]
        _, scale = robust_center_scale(reference[name], scale_floor=floor)
        scales[name] = scale
        query_distance, loo = nearest_and_loo(query[name], reference[name], scale)
        threshold = float(np.percentile(loo, 95.0))
        result = summarize_distance(query_distance, threshold, meta)
        low, high = np.percentile(reference[name], [1.0, 99.0], axis=0)
        outside = (query[name] < low[None, :]) | (query[name] > high[None, :])
        result["robust_p01_p99_outside_feature_fraction"] = float(outside.mean())
        result["dimensions"] = int(reference[name].shape[1])
        feature_rows = []
        for dimension, feature_name in enumerate(FEATURE_NAMES[name]):
            first_by_episode = {}
            for index, item in enumerate(meta):
                episode = str(item["episode"])
                if episode not in first_by_episode and outside[index, dimension]:
                    first_by_episode[episode] = float(item["time_after_handoff_s"])
            feature_rows.append({
                "feature": feature_name,
                "outside_fraction": float(outside[:, dimension].mean()),
                "first_exit_s_by_episode": first_by_episode,
                "first_exit_s": min(first_by_episode.values()) if first_by_episode else None,
            })
        feature_rows.sort(key=lambda row: row["outside_fraction"], reverse=True)
        result["top_outside_features"] = feature_rows[:5]
        exits = [row for row in feature_rows if row["first_exit_s"] is not None]
        result["earliest_percentile_exit"] = (
            None
            if not exits
            else {
                "time_s": min(row["first_exit_s"] for row in exits),
                "features": [
                    row["feature"]
                    for row in exits
                    if abs(row["first_exit_s"] - min(item["first_exit_s"] for item in exits)) < 1e-9
                ],
            }
        )
        group_results[name] = result
    composite, composite_loo = composite_nearest_and_loo(query, reference, scales, ALL_GROUPS)
    composite_result = summarize_distance(
        composite, float(np.percentile(composite_loo, 95.0)), meta
    )
    return {"groups": group_results, "equal_group_composite": composite_result}, scales


def analyze_ablations(
    query: dict[str, np.ndarray], reference: dict[str, np.ndarray], scales: dict[str, np.ndarray], meta: list[dict]
) -> dict:
    result = {}
    for omitted in ALL_GROUPS:
        names = tuple(name for name in ALL_GROUPS if name != omitted)
        distance, loo = composite_nearest_and_loo(query, reference, scales, names)
        result[f"without_{omitted}"] = summarize_distance(
            distance, float(np.percentile(loo, 95.0)), meta
        )
    return result


def first_physical_violation(meta: list[dict]) -> dict:
    result = {}
    for episode in sorted({item["episode"] for item in meta}):
        rows = [item for item in meta if item["episode"] == episode]
        tilt = next((row["time_after_handoff_s"] for row in rows if row["root_tilt_rad"] > 0.30), None)
        height = next((row["time_after_handoff_s"] for row in rows if row["root_z_m"] < 0.45), None)
        result[str(episode)] = {"tilt_gt_0p30_s": tilt, "root_z_lt_0p45_s": height}
    return result


def summarize_query_window(query: dict[str, np.ndarray], meta: list[dict]) -> dict:
    """Report physical query ranges without converting them into a new gate."""
    result = {}
    for episode in sorted({item["episode"] for item in meta}):
        indices = [index for index, item in enumerate(meta) if item["episode"] == episode]
        root = query["root_state"][indices]
        action = query["current_action"][indices]
        result[str(episode)] = {
            "rows": len(indices),
            "root_z_start_m": float(root[0, 0]),
            "root_z_end_m": float(root[-1, 0]),
            "root_z_min_m": float(root[:, 0].min()),
            "root_tilt_max_rad": float(root[:, 1].max()),
            "signed_root_pitch_min_rad": float(root[:, 2].min()),
            "signed_root_pitch_max_rad": float(root[:, 2].max()),
            "root_horizontal_speed_max_mps": float(root[:, 3].max()),
            "current_action_abs_max": float(np.abs(action).max()),
        }
    return result


def build_report(dataset: Path, source_report: Path, result_root: Path) -> dict:
    training, labels, training_provenance = load_stage335(dataset, source_report)
    query, meta, stand, phase13_provenance = load_phase13(result_root)
    training_all, training_scales = analyze_reference(query, training, meta)
    pass_reference = {name: values[labels] for name, values in training.items()}
    fail_reference = {name: values[~labels] for name, values in training.items()}
    training_floor = {name: training_scales[name] * 0.10 for name in ALL_GROUPS}
    training_pass, _ = analyze_reference(query, pass_reference, meta, global_floors=training_floor)
    training_fail, _ = analyze_reference(query, fail_reference, meta, global_floors=training_floor)
    stand_reference, _ = analyze_reference(query, stand, meta, global_floors=training_floor)
    return {
        "stage": "BASE Phase14 recovery OOD audit",
        "analysis_only": True,
        "causal_claim": False,
        "contracts": {
            "query": "Phase13 blend0.5 healthy episodes, recovery input rows +0.0..+1.5s",
            "training": "Stage335 immutable 90-state deterministic 93D rejoin",
            "stand": "Phase13 blend0 final 1.0s of source stand actor",
            "scaling": "per-feature median/IQR scale with fallback; per-group RMS width normalization",
            "composite": "equal weight over ten semantic groups",
            "ood_threshold": "reference leave-one-out nearest-neighbor distance p95",
            "eventual_label_warning": "pass/fail labels belong to source episode continuation, not a causal state oracle",
        },
        "provenance": {"training": training_provenance, "phase13": phase13_provenance},
        "query_window_summary": summarize_query_window(query, meta),
        "physical_first_violation": first_physical_violation(meta),
        "references": {
            "training_all": training_all,
            "training_eventual_pass": training_pass,
            "training_eventual_fail": training_fail,
            "source_stand_stable": stand_reference,
        },
        "training_all_equal_group_ablations": analyze_ablations(
            query, training, training_scales, meta
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"),
    )
    parser.add_argument(
        "--source-report", type=Path,
        default=Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"),
    )
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.dataset, args.source_report, args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "query_rows": report["provenance"]["phase13"]["query_rows"]}, indent=2))


if __name__ == "__main__":
    main()
