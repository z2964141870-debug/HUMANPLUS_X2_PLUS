#!/usr/bin/env python3
"""Rank Phase62 sagittal action directions by closed-loop central difference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    resource = json.loads(args.resource.read_text(encoding="utf-8"))
    legacy_eval_tag = (
        report.get("mode") == "eval"
        and report.get("posture_variant") == "action_sensitivity"
        and isinstance(report.get("action_sensitivity"), dict)
    )
    if report.get("phase") != 62 or (report.get("mode") != "screen" and not legacy_eval_tag):
        raise RuntimeError("input is not a Phase62 screen")
    epsilon = report["action_sensitivity"]["epsilon_normalized_action"]
    groups = report["groups"]
    base = groups["base"]
    rows = []
    for joint in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
        negative = groups[f"{joint}_neg"]
        positive = groups[f"{joint}_pos"]

        def derivative(path: tuple[str, ...]) -> float:
            def value(row: dict) -> float:
                selected = row
                for name in path:
                    selected = selected[name]
                return float(selected)
            return (value(positive) - value(negative)) / (2.0 * epsilon)

        pitch_derivative = derivative(("signed_pitch_rad", "mean"))
        support_derivative = derivative(("com_support_outside_m", "mean"))
        velocity_derivative = derivative(("velocity_tracking_rmse",))
        rows.append(
            {
                "joint_coordinate": joint,
                "pitch_mean_derivative_rad_per_action": pitch_derivative,
                "support_mean_derivative_m_per_action": support_derivative,
                "velocity_rmse_derivative_mps_per_action": velocity_derivative,
                "recommended_sign": 1 if pitch_derivative > 0.0 else -1,
                "absolute_pitch_sensitivity": abs(pitch_derivative),
                "negative_group": negative,
                "positive_group": positive,
            }
        )
    ranked = sorted(rows, key=lambda row: row["absolute_pitch_sensitivity"], reverse=True)
    valid_groups = all(
        row["survival_s_mean"] >= 4.0 - 1.0e-6 and row["termination_rate"] == 0.0
        for row in groups.values()
    )
    result = {
        "schema": "x2_action_sensitivity_phase62_result_v1",
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "source_report_mode": report.get("mode"),
        "legacy_eval_mode_tag_accepted": legacy_eval_tag,
        "resource": {"path": str(args.resource), "sha256": sha256(args.resource), "report": resource},
        "checkpoint_sha256": report["checkpoint_sha256"],
        "epsilon_normalized_action": epsilon,
        "base": base,
        "ranked_sagittal_coordinates": ranked,
        "all_groups_finite": report.get("finite") is True,
        "all_groups_survived": valid_groups,
        "decision": "PASS_DIAGNOSTIC_DIRECTION_ONLY" if report.get("finite") and valid_groups else "FAIL_DIAGNOSTIC_STOP",
        "training_unlocked": False,
        "deployment_unlocked": False,
        "boundary": "Finite differences identify local closed-loop directions only; they are not a checkpoint, reward gate, or deployment intervention.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "ranked": ranked}, indent=2))


if __name__ == "__main__":
    main()
