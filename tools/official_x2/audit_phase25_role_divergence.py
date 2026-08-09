#!/usr/bin/env python3
"""Offline role-aware divergence audit for BASE Phase25."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from official_x2.outcome_aware_state_role_v2 import load_manifest, resolve_rows


STABLE_TICKS = 5
DISCRIMINATIVE_BALANCED_ACCURACY = 0.65
ROLES = ("success_safe", "critical_from_failure")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robust_scale(values: np.ndarray) -> np.ndarray:
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    std = np.std(values, axis=0)
    return np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))


def pair_distance(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(((left[:, None, :] - right[None, :, :]) / scale) ** 2, axis=2))


def calibrate(success: np.ndarray, critical: np.ndarray) -> dict[str, Any]:
    scale = robust_scale(np.concatenate((success, critical), axis=0))
    within_s = pair_distance(success, success, scale)
    within_c = pair_distance(critical, critical, scale)
    np.fill_diagonal(within_s, np.inf)
    np.fill_diagonal(within_c, np.inf)
    loo_s = within_s.min(axis=1)
    loo_c = within_c.min(axis=1)
    threshold_s = max(float(np.percentile(loo_s, 95)), 1e-8)
    threshold_c = max(float(np.percentile(loo_c, 95)), 1e-8)
    cross_sc = pair_distance(success, critical, scale)
    pred_s = loo_s / threshold_s <= cross_sc.min(axis=1) / threshold_c
    pred_c = within_c.min(axis=1) / threshold_c < cross_sc.min(axis=0) / threshold_s
    success_recall = float(np.mean(pred_s))
    critical_recall = float(np.mean(pred_c))
    union = np.concatenate((success, critical), axis=0)
    union_pair = pair_distance(union, union, scale)
    np.fill_diagonal(union_pair, np.inf)
    union_threshold = max(float(np.percentile(union_pair.min(axis=1), 95)), 1e-8)
    return {
        "scale": scale,
        "success": success,
        "critical": critical,
        "success_threshold": threshold_s,
        "critical_threshold": threshold_c,
        "union": union,
        "union_threshold": union_threshold,
        "success_recall": success_recall,
        "critical_recall": critical_recall,
        "balanced_accuracy": 0.5 * (success_recall + critical_recall),
    }


def classify(value: np.ndarray, calibration: dict[str, Any]) -> dict[str, Any]:
    query = np.asarray(value, dtype=np.float64)[None]
    scale = calibration["scale"]
    ds = float(pair_distance(query, calibration["success"], scale).min())
    dc = float(pair_distance(query, calibration["critical"], scale).min())
    du = float(pair_distance(query, calibration["union"], scale).min())
    score_s = ds / calibration["success_threshold"]
    score_c = dc / calibration["critical_threshold"]
    return {
        "success_distance": ds, "critical_distance": dc,
        "success_normalized": score_s, "critical_normalized": score_c,
        "assigned_role": "success_safe" if score_s <= score_c else "critical_from_failure",
        "union_distance": du,
        "union_threshold": calibration["union_threshold"],
        "union_ood": bool(du > calibration["union_threshold"]),
    }


def first_stable(series: list[dict], predicate) -> float | None:
    for start in range(0, len(series) - STABLE_TICKS + 1):
        window = series[start : start + STABLE_TICKS]
        if all(predicate(row) for row in window):
            return float(window[0]["offset_s"])
    return None


def nearest_at(series: list[dict], offset: float) -> dict:
    return min(series, key=lambda row: abs(float(row["offset_s"]) - offset))


def median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else statistics.median(clean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/x2_phase19_outcome_aware_state_role.json"))
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    role_indices = {
        role: [int(row["manifest_row_index"]) for row in manifest["rows"] if row.get("eligible") and row.get("state_role") == role]
        for role in ROLES
    }
    resolved = {role: resolve_rows(manifest, indices) for role, indices in role_indices.items()}

    def matrix(role: str, fn) -> np.ndarray:
        return np.asarray([fn(row) for row in resolved[role]], dtype=np.float64)

    groups = {
        "base_lin_vel": lambda row: row["observation_93d"][0:3],
        "base_ang_vel": lambda row: row["observation_93d"][3:6],
        "projected_gravity": lambda row: row["observation_93d"][6:9],
        "joint_position": lambda row: row["observation_93d"][12:43],
        "joint_velocity": lambda row: row["observation_93d"][43:74],
        "previous_action": lambda row: row["observation_93d"][74:89],
        "gait_phase": lambda row: row["observation_93d"][89:93],
        "issued_action": lambda row: row["actual_issued_action"],
        "actor_proposal": lambda row: row["actor_proposal_action"],
        "root_posture": lambda row: [row["physical_state"]["root_position_m"][2], _root_tilt(row["physical_state"]["root_quaternion_xyzw"])],
    }
    calibrations = {
        name: calibrate(matrix("success_safe", fn), matrix("critical_from_failure", fn))
        for name, fn in groups.items()
    }

    episodes = []
    for repeat in range(1, 6):
        path = args.result_root / f"phase24_physical_gated_handoff_f005_stiff1p2_fixed_r{repeat}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        handoff = float(payload["summary"]["curriculum_recovery_handoff_gate_handoff_s"])
        source_rows = [
            row for row in payload["trace"]
            if row.get("stage") == "stop" and handoff - 0.5 - 1e-9 <= float(row["elapsed_s"]) <= handoff + 1.0 + 1e-9 and len(row.get("obs", [])) == 93
        ]
        series_by_group: dict[str, list[dict]] = {name: [] for name in groups}
        for row in source_rows:
            obs = np.asarray(row["obs"], dtype=np.float64)
            values = {
                "base_lin_vel": obs[0:3], "base_ang_vel": obs[3:6], "projected_gravity": obs[6:9],
                "joint_position": obs[12:43], "joint_velocity": obs[43:74], "previous_action": obs[74:89],
                "gait_phase": obs[89:93], "issued_action": np.asarray(row["action"], dtype=np.float64),
                "actor_proposal": np.asarray(row.get("unblended_policy_action") or row["action"], dtype=np.float64),
                "root_posture": np.asarray([row["root_z_m"], row["root_tilt_rad"]], dtype=np.float64),
            }
            for name, value in values.items():
                entry = classify(value, calibrations[name])
                entry["offset_s"] = float(row["elapsed_s"]) - handoff
                series_by_group[name].append(entry)
        group_results = {}
        for name, series in series_by_group.items():
            post = [row for row in series if row["offset_s"] >= -1e-9]
            group_results[name] = {
                "at_handoff": nearest_at(series, 0.0),
                "at_0p5s": nearest_at(series, 0.5),
                "at_1p0s": nearest_at(series, 1.0),
                "first_stable_critical_offset_s": first_stable(post, lambda row: row["assigned_role"] == "critical_from_failure"),
                "first_stable_union_ood_offset_s": first_stable(post, lambda row: row["union_ood"]),
            }
        episodes.append({
            "repeat": repeat, "path": str(path), "file_sha256": sha256(path), "handoff_stop_elapsed_s": handoff,
            "group_results": group_results,
        })

    group_summary = {}
    for name, calibration in calibrations.items():
        critical_onsets = [episode["group_results"][name]["first_stable_critical_offset_s"] for episode in episodes]
        ood_onsets = [episode["group_results"][name]["first_stable_union_ood_offset_s"] for episode in episodes]
        group_summary[name] = {
            "dimension": int(calibration["success"].shape[1]),
            "reference_counts": {"success_safe": len(resolved["success_safe"]), "critical_from_failure": len(resolved["critical_from_failure"])},
            "loo_success_recall": calibration["success_recall"],
            "loo_critical_recall": calibration["critical_recall"],
            "loo_balanced_accuracy": calibration["balanced_accuracy"],
            "role_discriminative": calibration["balanced_accuracy"] >= DISCRIMINATIVE_BALANCED_ACCURACY,
            "stable_critical_episode_count": sum(value is not None for value in critical_onsets),
            "stable_critical_median_offset_s": median(critical_onsets),
            "stable_union_ood_episode_count": sum(value is not None for value in ood_onsets),
            "stable_union_ood_median_offset_s": median(ood_onsets),
            "assigned_critical_counts": {
                label: sum(
                    episode["group_results"][name][key]["assigned_role"] == "critical_from_failure"
                    for episode in episodes
                )
                for label, key in (("handoff", "at_handoff"), ("0p5s", "at_0p5s"), ("1p0s", "at_1p0s"))
            },
            "union_ood_counts": {
                label: sum(episode["group_results"][name][key]["union_ood"] for episode in episodes)
                for label, key in (("handoff", "at_handoff"), ("0p5s", "at_0p5s"), ("1p0s", "at_1p0s"))
            },
        }

    # Only the action actually issued to physics participates in precedence.
    # Phase19 actor proposals came from a different controller and are not
    # labels for f005, so proposal distance remains diagnostic-only.
    action_names = ("issued_action",)
    physical_names = ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_position", "joint_velocity", "root_posture")
    action_onsets = [group_summary[name]["stable_union_ood_median_offset_s"] for name in action_names if group_summary[name]["role_discriminative"]]
    physical_onsets = [group_summary[name]["stable_union_ood_median_offset_s"] for name in physical_names if group_summary[name]["role_discriminative"]]
    action_first = min((x for x in action_onsets if x is not None), default=None)
    physical_first = min((x for x in physical_onsets if x is not None), default=None)
    if action_first is None and physical_first is not None:
        ordering = "physical_ood_present_without_stable_executed_action_ood"
    elif action_first is None or physical_first is None:
        ordering = "indeterminate_missing_discriminative_stable_ood"
    elif action_first + 0.04 < physical_first:
        ordering = "actor_action_ood_precedes_physical_ood"
    elif physical_first + 0.04 < action_first:
        ordering = "physical_ood_precedes_actor_action_ood"
    else:
        ordering = "simultaneous_within_0p04s"

    serializable_calibration = {
        name: {
            key: value for key, value in calibration.items()
            if key not in ("scale", "success", "critical", "union")
        }
        for name, calibration in calibrations.items()
    }
    result = {
        "stage": "BASE Phase25 offline role-aware divergence",
        "manifest": str(args.manifest), "manifest_file_sha256": sha256(args.manifest),
        "role_counts": {role: len(rows) for role, rows in resolved.items()},
        "pre_registered_rules": {
            "stable_run_ticks": STABLE_TICKS, "stable_run_seconds": STABLE_TICKS * 0.02,
            "role_discriminative_balanced_accuracy_min": DISCRIMINATIVE_BALANCED_ACCURACY,
            "role_assignment": "smaller nearest-neighbor distance normalized by class-specific within-role LOO-p95",
            "union_ood": "nearest union distance > union LOO-p95",
            "groups_equal_weight_boundary": "no raw concatenated full-93D distance; groups reported separately to prevent dimensional dominance",
        },
        "calibration": serializable_calibration,
        "group_summary": group_summary,
        "episodes": episodes,
        "ordering": {
            "action_first_stable_ood_median_s": action_first,
            "physical_first_stable_ood_median_s": physical_first,
            "classification": ordering,
        },
        "causal_boundary": {
            "actions_are_not_expert_labels": True,
            "reference_actions_semantics": "Phase19 controller history/state only, never imitation targets",
            "actor_proposal_comparability": "diagnostic only: Phase19 proposals are from another controller, excluded from precedence裁决",
            "phase24_trace_lacks_full_controller_snapshot": True,
            "controller_state_comparison": "unavailable beyond gate/latch/slot timing; no values fabricated",
            "interpretation": "temporal ordering is descriptive and can reject simple precedence hypotheses; it cannot identify an unobserved contact/PD cause",
        },
        "training_unlocked": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role_counts": result["role_counts"], "ordering": result["ordering"], "groups": group_summary}, indent=2))


def _root_tilt(quaternion_xyzw: list[float]) -> float:
    x, y, z, w = map(float, quaternion_xyzw)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(max(-1.0, min(1.0, up_z))))


if __name__ == "__main__":
    main()
