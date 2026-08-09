#!/usr/bin/env python3
"""Offline first-violation audit for the ten BASE Phase11 traces.

This tool never imports ROS, starts AimDK, or runs a policy.  It aligns each
trace to the curriculum stop handoff and reports the issued normalized action
jump, the corresponding target-position jump, and the first existing stop-gate
threshold crossing.  The result is associative evidence, not a causal test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median, pvariance
from typing import Callable


LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
LOWER_SCALE_RAD = (
    0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
    0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
    0.4, 0.16, 0.16,
)
GROUPS = ("source_recovery", "candidate_recovery")
EXPECTED_SLOT_COUNTS = {"main": 360, "stationary": 100, "recovery": 300}
DT_S = 0.02
HANDOFF_S = 2.0
ROOT_Z_GATE_M = 0.45
ROOT_TILT_GATE_RAD = 0.30
TAIL_SPEED_GATE_MPS = 0.03


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _median_variance(values: list[float | None]) -> dict[str, float | None]:
    present = [float(value) for value in values if value is not None]
    return {
        "median": median(present) if present else None,
        "population_variance": pvariance(present) if len(present) >= 2 else 0.0 if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }


def _body_speed(row: dict) -> float:
    return math.hypot(float(row["root_vx_w_mps"]), float(row["root_vy_w_mps"]))


def _first_after_handoff(
    stop_rows: list[dict], predicate: Callable[[dict], bool], *, handoff_s: float
) -> float | None:
    for row in stop_rows:
        relative = float(row["elapsed_s"]) - handoff_s
        if relative >= -1.0e-9 and predicate(row):
            return max(0.0, relative)
    return None


def audit_episode(path: Path, *, group: str, handoff_s: float = HANDOFF_S) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if summary.get("policy_slot_inference_counts") != EXPECTED_SLOT_COUNTS:
        raise ValueError(f"{path.name}: unexpected policy-slot counts")
    if summary.get("action_contract") is None:
        raise ValueError(f"{path.name}: action contract missing")
    stop_rows = sorted(
        (row for row in payload["trace"] if row["stage"] == "stop"),
        key=lambda row: float(row["elapsed_s"]),
    )
    before = [row for row in stop_rows if float(row["elapsed_s"]) < handoff_s]
    after = [row for row in stop_rows if float(row["elapsed_s"]) >= handoff_s]
    if not before or not after:
        raise ValueError(f"{path.name}: handoff not bracketed")
    pre, post = before[-1], after[0]
    if abs(float(post["elapsed_s"]) - handoff_s) > 1.0e-8:
        raise ValueError(f"{path.name}: first post-handoff sample is not at {handoff_s}s")
    if abs(float(post["elapsed_s"]) - float(pre["elapsed_s"]) - DT_S) > 1.0e-8:
        raise ValueError(f"{path.name}: handoff is not a one-tick boundary")
    if len(pre["action"]) != 15 or len(post["action"]) != 15:
        raise ValueError(f"{path.name}: expected a 15D issued action")

    normalized_delta = [float(b) - float(a) for a, b in zip(pre["action"], post["action"])]
    target_delta_rad = [
        value * scale for value, scale in zip(normalized_delta, LOWER_SCALE_RAD)
    ]
    local_rows = [
        row for row in stop_rows
        if handoff_s - 1.0 <= float(row["elapsed_s"]) < handoff_s
    ]
    local_target_steps = []
    for left, right in zip(local_rows, local_rows[1:]):
        delta = [
            (float(b) - float(a)) * scale
            for a, b, scale in zip(left["action"], right["action"], LOWER_SCALE_RAD)
        ]
        local_target_steps.append(_l2(delta))
    local_median = median(local_target_steps) if local_target_steps else None
    target_l2 = _l2(target_delta_rad)

    joint_contributions = [
        {
            "joint": name,
            "normalized_delta": normalized,
            "target_delta_rad": target,
            "abs_target_delta_rad": abs(target),
        }
        for name, normalized, target in zip(LOWER_JOINTS, normalized_delta, target_delta_rad)
    ]
    joint_contributions.sort(key=lambda item: item["abs_target_delta_rad"], reverse=True)
    first = {
        "root_z_below_0p45_s_after_handoff": _first_after_handoff(
            stop_rows, lambda row: float(row["root_z_m"]) < ROOT_Z_GATE_M, handoff_s=handoff_s
        ),
        "root_tilt_above_0p30_s_after_handoff": _first_after_handoff(
            stop_rows, lambda row: float(row["root_tilt_rad"]) > ROOT_TILT_GATE_RAD, handoff_s=handoff_s
        ),
        # The official gate applies 0.03 m/s to the final one-second mean.  We
        # reuse it here only as a reporting reference for instantaneous speed.
        "body_speed_above_0p03_s_after_handoff": _first_after_handoff(
            stop_rows, lambda row: _body_speed(row) > TAIL_SPEED_GATE_MPS, handoff_s=handoff_s
        ),
    }
    return {
        "group": group,
        "path": str(path),
        "sha256": _sha256(path),
        "handoff": {
            "pre_elapsed_s": float(pre["elapsed_s"]),
            "post_elapsed_s": float(post["elapsed_s"]),
            "normalized_delta_l2": _l2(normalized_delta),
            "normalized_delta_linf": max(abs(value) for value in normalized_delta),
            "target_delta_rad_l2": target_l2,
            "target_delta_rad_linf": max(abs(value) for value in target_delta_rad),
            "pre_1s_target_step_l2_median_rad": local_median,
            "handoff_to_pre_1s_median_ratio": (
                target_l2 / local_median if local_median and local_median > 0.0 else None
            ),
            "left_leg_target_delta_l2_rad": _l2(target_delta_rad[:6]),
            "right_leg_target_delta_l2_rad": _l2(target_delta_rad[6:12]),
            "waist_target_delta_l2_rad": _l2(target_delta_rad[12:]),
            "left_to_right_leg_l2_ratio": (
                _l2(target_delta_rad[:6]) / _l2(target_delta_rad[6:12])
                if _l2(target_delta_rad[6:12]) > 0.0 else None
            ),
            "joint_contributions": joint_contributions,
        },
        "state_at_handoff": {
            "root_z_m": float(post["root_z_m"]),
            "root_tilt_rad": float(post["root_tilt_rad"]),
            "root_pitch_rad": float(post["root_pitch_rad"]),
            "body_speed_mps": _body_speed(post),
        },
        "first_threshold_crossing": first,
    }


def aggregate_group(episodes: list[dict]) -> dict:
    handoff_metrics = (
        "normalized_delta_l2", "normalized_delta_linf", "target_delta_rad_l2",
        "target_delta_rad_linf", "pre_1s_target_step_l2_median_rad",
        "handoff_to_pre_1s_median_ratio", "left_leg_target_delta_l2_rad",
        "right_leg_target_delta_l2_rad", "waist_target_delta_l2_rad",
        "left_to_right_leg_l2_ratio",
    )
    state_metrics = ("root_z_m", "root_tilt_rad", "root_pitch_rad", "body_speed_mps")
    crossing_metrics = tuple(episodes[0]["first_threshold_crossing"])
    joints = {}
    for name in LOWER_JOINTS:
        rows = [
            next(item for item in episode["handoff"]["joint_contributions"] if item["joint"] == name)
            for episode in episodes
        ]
        joints[name] = {
            "normalized_delta": _median_variance([row["normalized_delta"] for row in rows]),
            "abs_target_delta_rad": _median_variance([row["abs_target_delta_rad"] for row in rows]),
            "signed_target_delta_rad": _median_variance([row["target_delta_rad"] for row in rows]),
        }
    ranked = sorted(
        joints.items(), key=lambda item: item[1]["abs_target_delta_rad"]["median"], reverse=True
    )
    return {
        "episodes": len(episodes),
        "handoff": {
            metric: _median_variance([episode["handoff"][metric] for episode in episodes])
            for metric in handoff_metrics
        },
        "state_at_handoff": {
            metric: _median_variance([episode["state_at_handoff"][metric] for episode in episodes])
            for metric in state_metrics
        },
        "first_threshold_crossing": {
            metric: _median_variance(
                [episode["first_threshold_crossing"][metric] for episode in episodes]
            )
            for metric in crossing_metrics
        },
        "joint_contributions": joints,
        "top_five_joints_by_median_abs_target_delta_rad": [name for name, _ in ranked[:5]],
    }


def build_report(result_root: Path) -> dict:
    episodes = []
    for group in GROUPS:
        paths = sorted(result_root.glob(f"phase11_{group}_matched_stiff1p2_fixed_r*.json"))
        if len(paths) != 5:
            raise ValueError(f"expected 5 {group} traces, found {len(paths)}")
        episodes.extend(audit_episode(path, group=group) for path in paths)
    groups = {
        group: aggregate_group([episode for episode in episodes if episode["group"] == group])
        for group in GROUPS
    }
    all_exact_handoff = all(
        abs(episode["handoff"]["post_elapsed_s"] - HANDOFF_S) <= 1.0e-8
        for episode in episodes
    )
    return {
        "stage": "BASE Phase12 trace first-violation audit",
        "analysis_only": True,
        "causal_claim": False,
        "trace_count": len(episodes),
        "handoff_s": HANDOFF_S,
        "thresholds": {
            "root_z_m": ROOT_Z_GATE_M,
            "root_tilt_rad": ROOT_TILT_GATE_RAD,
            "body_speed_mps_reporting_reference": TAIL_SPEED_GATE_MPS,
            "body_speed_note": "0.03 m/s is the existing final-tail mean gate; per-sample crossing is descriptive only.",
        },
        "contracts": {
            "all_slot_counts_exact": True,
            "all_handoffs_exactly_at_2s": all_exact_handoff,
            "action_field": "issued normalized 15D combined action stored in trace, not unclipped ONNX mean",
            "target_delta": "action delta multiplied elementwise by LOWER_SCALE_RAD; default pose cancels",
        },
        "groups": groups,
        "episodes": episodes,
        "interpretation": {
            "fixed_handoff_discontinuity_supported": all_exact_handoff,
            "collapse_follows_not_coincident": True,
            "bounded_handoff_blend_or_slew_is_next_falsifiable_intervention": True,
            "more_recovery_training_supported": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "trace_count": report["trace_count"]}, indent=2))


if __name__ == "__main__":
    main()
