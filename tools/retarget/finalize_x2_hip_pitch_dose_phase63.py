#!/usr/bin/env python3
"""Apply the preregistered Phase63 small-dose gates without changing weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested(row: dict, *names: str) -> float:
    value = row
    for name in names:
        value = value[name]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    resource = json.loads(args.resource.read_text(encoding="utf-8"))
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if report.get("phase") != 63 or report.get("mode") != "screen":
        raise RuntimeError("input is not a Phase63 screen")
    if report.get("posture_variant") != "hip_pitch_dose":
        raise RuntimeError("Phase63 posture variant changed")
    observed_doses = report["hip_pitch_dose_screen"]["normalized_action_doses"]
    expected_doses = prereg["screen"]["normalized_action_doses"]
    if observed_doses != expected_doses:
        raise RuntimeError(f"Phase63 dose contract changed: {observed_doses}")
    if report.get("checkpoint_sha256") != prereg["source"]["checkpoint_sha256"]:
        raise RuntimeError("Phase63 source checkpoint changed")

    groups = report["groups"]
    base = groups["base"]
    gates = prereg["local_gates"]
    names = [
        "hip_pitch_m004", "hip_pitch_m006", "hip_pitch_m008",
        "hip_pitch_m010", "hip_pitch_m012", "hip_pitch_m016", "hip_pitch_m020",
    ]
    rows = []
    for name, dose in zip(names, expected_doses[1:], strict=True):
        candidate = groups[name]
        delta = {
            "signed_pitch_mean_rad": nested(candidate, "signed_pitch_rad", "mean") - nested(base, "signed_pitch_rad", "mean"),
            "signed_pitch_p05_rad": nested(candidate, "signed_pitch_rad", "p05") - nested(base, "signed_pitch_rad", "p05"),
            "lateral_abs_m": nested(candidate, "lateral_abs") - nested(base, "lateral_abs"),
            "yaw_abs_rad": nested(candidate, "yaw_abs") - nested(base, "yaw_abs"),
            "velocity_tracking_rmse_mps": nested(candidate, "velocity_tracking_rmse") - nested(base, "velocity_tracking_rmse"),
            "com_support_outside_mean_m": nested(candidate, "com_support_outside_m", "mean") - nested(base, "com_support_outside_m", "mean"),
            "stance_slip_p95_mps": nested(candidate, "stance_slip_mps", "p95") - nested(base, "stance_slip_mps", "p95"),
            "root_height_min_mean_m": nested(candidate, "root_height_min_mean") - nested(base, "root_height_min_mean"),
            "root_tilt_max_mean_rad": nested(candidate, "root_tilt_max_mean") - nested(base, "root_tilt_max_mean"),
            "action_delta_abs": nested(candidate, "action_delta_abs") - nested(base, "action_delta_abs"),
        }
        checks = {
            "finite": report.get("finite") is True,
            "survival": nested(candidate, "survival_s_mean") >= gates["survival_s_min"] - 1.0e-6,
            "termination": nested(candidate, "termination_rate") <= gates["termination_rate_max"],
            "pitch_mean": delta["signed_pitch_mean_rad"] >= gates["signed_pitch_mean_improvement_min_rad"],
            "pitch_p05": delta["signed_pitch_p05_rad"] >= gates["signed_pitch_p05_improvement_min_rad"],
            "lateral": delta["lateral_abs_m"] <= gates["lateral_abs_regression_max_m"],
            "yaw": delta["yaw_abs_rad"] <= gates["yaw_abs_regression_max_rad"],
            "velocity": delta["velocity_tracking_rmse_mps"] <= gates["velocity_rmse_regression_max_mps"],
            "support": delta["com_support_outside_mean_m"] <= gates["com_support_mean_regression_max_m"],
            "slip": delta["stance_slip_p95_mps"] <= gates["stance_slip_p95_regression_max_mps"],
            "root_height": delta["root_height_min_mean_m"] >= -gates["root_height_min_regression_max_m"],
            "root_tilt": delta["root_tilt_max_mean_rad"] <= gates["root_tilt_max_regression_max_rad"],
            "action_delta": delta["action_delta_abs"] <= gates["action_delta_abs_regression_max"],
        }
        rows.append({
            "group": name,
            "normalized_action_dose": dose,
            "delta_vs_base": delta,
            "checks": checks,
            "passed": all(checks.values()),
            "metrics": candidate,
        })

    passed = [row for row in rows if row["passed"]]
    selected = passed[0] if passed else None
    decision = "PASS_LOCAL_DOSE_WINDOW" if selected else "FAIL_LOCAL_DOSE_WINDOW_STOP"
    result = {
        "schema": "x2_hip_pitch_dose_phase63_result_v1",
        "decision": decision,
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "resource": {"path": str(args.resource), "sha256": sha256(args.resource), "report": resource},
        "preregistration": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
        "checkpoint_sha256": report["checkpoint_sha256"],
        "base": base,
        "dose_rows": rows,
        "selected_group": selected["group"] if selected else None,
        "selected_normalized_action_dose": selected["normalized_action_dose"] if selected else None,
        "selection_rule": prereg["selection_rule"],
        "targeted_residual_experiment_unlocked": bool(selected),
        "training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
        "boundary": "A passing constant-dose screen is local causal evidence only; it does not alter the checkpoint or prove official MuJoCo gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# X2 native posture Phase63 — bilateral hip-pitch dose screen",
        "",
        f"Decision: **{decision}**.",
        "",
        "| group | dose | pitch mean Δ | pitch p05 Δ | lateral Δ | yaw Δ | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        delta = row["delta_vs_base"]
        lines.append(
            f"| `{row['group']}` | {row['normalized_action_dose']:.3f} | "
            f"{delta['signed_pitch_mean_rad']:+.6f} | {delta['signed_pitch_p05_rad']:+.6f} | "
            f"{delta['lateral_abs_m']:+.6f} | {delta['yaw_abs_rad']:+.6f} | {row['passed']} |"
        )
    lines.extend([
        "",
        f"Selected group: `{result['selected_group']}`.",
        "",
        "No checkpoint, ONNX, optimizer update, or deployment backend was produced by this screen.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "selected": result["selected_group"]}, indent=2))


if __name__ == "__main__":
    main()
