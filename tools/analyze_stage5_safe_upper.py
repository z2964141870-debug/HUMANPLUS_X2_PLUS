#!/usr/bin/env python3
"""Add bounded-adapter checks to one Stage4-compatible comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cwi_x2.upper_motion_contract import UPPER_JOINT_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--max-excursion-rad", type=float, required=True)
    parser.add_argument("--max-velocity-radps", type=float, required=True)
    parser.add_argument("--tilt-fallback-rad", type=float, required=True)
    parser.add_argument("--height-fallback-m", type=float, required=True)
    parser.add_argument("--heading-fallback-rad", type=float, default=float("inf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.comparison.read_text())
    trace = np.load(args.candidate_trace, allow_pickle=False)
    order = trace["all_joint_order"].tolist()
    ids = [order.index(name) for name in UPPER_JOINT_NAMES]
    target = trace["all_joint_target_rad"][:, ids].astype(np.float64)
    time_s = trace["time_s"].astype(np.float64)
    excursion = target - target[:1]
    dt = np.diff(time_s)
    velocity = np.diff(target, axis=0) / np.maximum(dt[:, None], 1.0e-9)
    tilt = trace["base_tilt_rad"].astype(np.float64)
    root_height = trace["root_pos_w_m"][:, 2].astype(np.float64)
    hazard = (tilt > args.tilt_fallback_rad) | (root_height < args.height_fallback_m)
    heading = np.abs(trace["heading_delta_rad"].astype(np.float64))
    heading_risk = heading > args.heading_fallback_rad

    safety_metrics = {
        "target_excursion_max_rad": float(np.max(np.abs(excursion))),
        "target_velocity_abs_p95_radps": float(np.percentile(np.abs(velocity), 95.0)),
        "target_velocity_abs_max_radps": float(np.max(np.abs(velocity))),
        "hazard_fraction": float(np.mean(hazard)),
        "hazard_first_time_s": (
            None if not np.any(hazard) else float(time_s[np.flatnonzero(hazard)[0]])
        ),
        "heading_risk_fraction": float(np.mean(heading_risk)),
        "heading_risk_first_time_s": (
            None
            if not np.any(heading_risk)
            else float(time_s[np.flatnonzero(heading_risk)[0]])
        ),
    }
    safety_gate = {
        "excursion_within_bound": (
            safety_metrics["target_excursion_max_rad"] <= args.max_excursion_rad + 5.0e-4
        ),
        "velocity_within_bound": (
            safety_metrics["target_velocity_abs_max_radps"]
            <= args.max_velocity_radps + 5.0e-3
        ),
        "hazard_fraction_le_0p02": safety_metrics["hazard_fraction"] <= 0.02,
    }
    result["stage5_safety"] = {
        "limits": {
            "max_excursion_rad": args.max_excursion_rad,
            "max_velocity_radps": args.max_velocity_radps,
            "tilt_fallback_rad": args.tilt_fallback_rad,
            "height_fallback_m": args.height_fallback_m,
            "heading_fallback_rad": args.heading_fallback_rad,
        },
        "metrics": safety_metrics,
        "gate": safety_gate,
    }
    result["stage4_comparison_passed"] = bool(result["passed"])
    result["passed"] = bool(result["passed"] and all(safety_gate.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
