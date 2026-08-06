#!/usr/bin/env python3
"""Compare a Stage4 upper-motion rollout against its fixed-upper control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cwi_x2.upper_motion_contract import UPPER_JOINT_NAMES


KEY_METRICS = (
    "steps_completed",
    "forward_displacement_m",
    "progress_ratio",
    "body_vx_rmse_mps",
    "heading_deviation_abs_max_rad",
    "lateral_drift_max_m",
    "base_tilt_max_rad",
    "left_foot_lift_max_m",
    "right_foot_lift_max_m",
    "single_support_fraction",
    "double_support_fraction",
    "flight_fraction",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _relative_change(value: float, control: float) -> float:
    return (value - control) / max(abs(control), 1.0e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-json", type=Path, required=True)
    parser.add_argument("--control-trace", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--upper-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = _load_json(args.control_json)
    candidate = _load_json(args.candidate_json)
    ctrace = np.load(args.control_trace, allow_pickle=False)
    trace = np.load(args.candidate_trace, allow_pickle=False)
    order = trace["all_joint_order"].tolist()
    upper_ids = [order.index(name) for name in UPPER_JOINT_NAMES]
    target = trace["all_joint_target_rad"][:, upper_ids]
    actual = trace["all_joint_pos_rad"][:, upper_ids]
    n = min(len(target), len(actual))
    target = target[:n]
    actual = actual[:n]
    error = np.abs(actual - target)
    excursion = target - target[:1]

    common_arrays = sorted(
        set(ctrace.files).intersection(trace.files)
        - {
            "action_joint_order",
            "all_joint_order",
            "legs_joint_order",
            "feet_joint_order",
            "waist_joint_order",
            "arms_joint_order",
        }
    )
    exact_deltas = {}
    for key in common_arrays:
        left = ctrace[key]
        right = trace[key]
        if left.shape != right.shape or not np.issubdtype(left.dtype, np.number):
            continue
        exact_deltas[key] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))

    metric_comparison = {}
    for key in KEY_METRICS:
        a = float(control[key])
        b = float(candidate[key])
        metric_comparison[key] = {
            "control": a,
            "candidate": b,
            "delta": b - a,
            "relative_delta": _relative_change(b, a),
        }

    zero_equivalent = None
    if args.upper_scale == 0.0:
        zero_equivalent = (
            max(exact_deltas.values(), default=0.0) <= 1.0e-6
            and all(abs(item["delta"]) <= 1.0e-6 for item in metric_comparison.values())
        )

    gate = {
        "survived": bool(candidate["survived_full_horizon"]),
        "active_upper_target": float(np.sqrt(np.mean(excursion**2))) >= 0.01
        if args.upper_scale > 0.0
        else True,
        "upper_tracking_p95_le_0p35_rad": float(np.percentile(error, 95.0)) <= 0.35,
        "heading_degradation_le_0p15_rad": (
            float(candidate["heading_deviation_abs_max_rad"])
            - float(control["heading_deviation_abs_max_rad"])
        )
        <= 0.15,
        "lateral_degradation_le_0p15_m": (
            float(candidate["lateral_drift_max_m"])
            - float(control["lateral_drift_max_m"])
        )
        <= 0.15,
        "tilt_le_0p45_rad": float(candidate["base_tilt_max_rad"]) <= 0.45,
        "both_feet_lift_ge_0p02_m": min(
            float(candidate["left_foot_lift_max_m"]),
            float(candidate["right_foot_lift_max_m"]),
        )
        >= 0.02,
    }
    passed = bool(zero_equivalent) if args.upper_scale == 0.0 else all(gate.values())
    result = {
        "schema_version": 1,
        "upper_scale": args.upper_scale,
        "control_json": str(args.control_json.resolve()),
        "candidate_json": str(args.candidate_json.resolve()),
        "zero_equivalent": zero_equivalent,
        "passed": passed,
        "upper": {
            "joint_order": list(UPPER_JOINT_NAMES),
            "target_excursion_rms_rad": float(np.sqrt(np.mean(excursion**2))),
            "target_excursion_max_rad": float(np.max(np.abs(excursion))),
            "tracking_abs_mean_rad": float(np.mean(error)),
            "tracking_abs_p95_rad": float(np.percentile(error, 95.0)),
            "tracking_abs_max_rad": float(np.max(error)),
        },
        "metrics": metric_comparison,
        "gate": gate,
        "trace_max_abs_delta_vs_control": exact_deltas if args.upper_scale == 0.0 else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

