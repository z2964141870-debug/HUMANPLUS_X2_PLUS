#!/usr/bin/env python3
"""Apply the preregistered Phase64 mirrored knee-dose gates fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOSE_SPECS = (
    ("m008", -0.008),
    ("m010", -0.010),
    ("m012", -0.012),
)
PHASE_REGIONS = (
    "double_support_zero",
    "right_swing_left_support",
    "double_support_half",
    "left_swing_right_support",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested(row: dict, *names: str) -> float:
    value = row
    for name in names:
        value = value[name]
    if value is None:
        raise RuntimeError(f"missing finite metric at {'.'.join(names)}")
    return float(value)


def metric_delta(candidate: dict, base: dict) -> dict[str, float]:
    return {
        "signed_pitch_mean_rad": nested(candidate, "signed_pitch_rad", "mean")
        - nested(base, "signed_pitch_rad", "mean"),
        "signed_pitch_p05_rad": nested(candidate, "signed_pitch_rad", "p05")
        - nested(base, "signed_pitch_rad", "p05"),
        "lateral_abs_m": nested(candidate, "lateral_abs") - nested(base, "lateral_abs"),
        "yaw_abs_rad": nested(candidate, "yaw_abs") - nested(base, "yaw_abs"),
        "velocity_tracking_rmse_mps": nested(candidate, "velocity_tracking_rmse")
        - nested(base, "velocity_tracking_rmse"),
        "com_support_outside_mean_m": nested(candidate, "com_support_outside_m", "mean")
        - nested(base, "com_support_outside_m", "mean"),
        "stance_slip_p95_mps": nested(candidate, "stance_slip_mps", "p95")
        - nested(base, "stance_slip_mps", "p95"),
        "root_height_min_mean_m": nested(candidate, "root_height_min_mean")
        - nested(base, "root_height_min_mean"),
        "root_tilt_max_mean_rad": nested(candidate, "root_tilt_max_mean")
        - nested(base, "root_tilt_max_mean"),
        "action_delta_abs": nested(candidate, "action_delta_abs")
        - nested(base, "action_delta_abs"),
        "swing_clearance_p05_m": nested(candidate, "swing_sole_clearance_m", "p05")
        - nested(base, "swing_sole_clearance_m", "p05"),
        "knee_excursion_mean_rad": nested(candidate, "knee_excursion_rad_mean")
        - nested(base, "knee_excursion_rad_mean"),
        "knee_left_excursion_mean_rad": nested(
            candidate, "knee_joint_range", "left", "excursion_rad_mean"
        )
        - nested(base, "knee_joint_range", "left", "excursion_rad_mean"),
        "knee_right_excursion_mean_rad": nested(
            candidate, "knee_joint_range", "right", "excursion_rad_mean"
        )
        - nested(base, "knee_joint_range", "right", "excursion_rad_mean"),
        "knee_left_right_asymmetry_rad": nested(
            candidate, "knee_excursion_left_right_abs_diff_rad_mean"
        )
        - nested(base, "knee_excursion_left_right_abs_diff_rad_mean"),
        "flight_fraction": nested(candidate, "flight_fraction") - nested(base, "flight_fraction"),
        "intervention_clip_fraction": nested(candidate, "knee_intervention_clip_fraction")
        - nested(base, "knee_intervention_clip_fraction"),
    }


def global_checks(
    report_finite: bool,
    candidate: dict,
    delta: dict[str, float],
    gates: dict,
) -> dict[str, bool]:
    requested = nested(candidate, "knee_requested_bias_mean")
    effective = nested(candidate, "knee_effective_bias_mean")
    ratio = abs(effective) / max(abs(requested), 1.0e-12)
    return {
        "finite": report_finite,
        "survival": nested(candidate, "survival_s_mean") >= gates["survival_s_min"] - 1.0e-6,
        "termination": nested(candidate, "termination_rate") <= gates["termination_rate_max"],
        "pitch_mean": delta["signed_pitch_mean_rad"]
        >= gates["signed_pitch_mean_improvement_min_rad"],
        "pitch_p05": delta["signed_pitch_p05_rad"]
        >= gates["signed_pitch_p05_improvement_min_rad"],
        "lateral": delta["lateral_abs_m"] <= gates["lateral_abs_regression_max_m"],
        "yaw": delta["yaw_abs_rad"] <= gates["yaw_abs_regression_max_rad"],
        "velocity": delta["velocity_tracking_rmse_mps"]
        <= gates["velocity_rmse_regression_max_mps"],
        "support": delta["com_support_outside_mean_m"]
        <= gates["com_support_mean_regression_max_m"],
        "slip": delta["stance_slip_p95_mps"] <= gates["stance_slip_p95_regression_max_mps"],
        "root_height": delta["root_height_min_mean_m"]
        >= -gates["root_height_min_regression_max_m"],
        "root_tilt": delta["root_tilt_max_mean_rad"]
        <= gates["root_tilt_max_regression_max_rad"],
        "action_delta": delta["action_delta_abs"] <= gates["action_delta_abs_regression_max"],
        "swing_clearance": delta["swing_clearance_p05_m"]
        >= -gates["swing_clearance_p05_regression_max_m"],
        "knee_excursion_mean": abs(delta["knee_excursion_mean_rad"])
        <= gates["knee_excursion_abs_change_max_rad"],
        "knee_excursion_left": abs(delta["knee_left_excursion_mean_rad"])
        <= gates["knee_excursion_abs_change_max_rad"],
        "knee_excursion_right": abs(delta["knee_right_excursion_mean_rad"])
        <= gates["knee_excursion_abs_change_max_rad"],
        "knee_asymmetry": delta["knee_left_right_asymmetry_rad"]
        <= gates["knee_left_right_asymmetry_regression_max_rad"],
        "flight": delta["flight_fraction"] <= gates["flight_fraction_regression_max"],
        "effective_bias": requested < 0.0
        and effective < 0.0
        and ratio >= gates["effective_requested_bias_ratio_min"],
        "intervention_clip": delta["intervention_clip_fraction"]
        <= gates["intervention_clip_fraction_regression_max"],
    }


def phase_comparison_checks(
    candidate: dict,
    base: dict,
    gates: dict,
    count_minimum: int,
) -> dict:
    result = {}
    for region in PHASE_REGIONS:
        candidate_region = candidate[region]
        base_region = base[region]
        delta = {
            "signed_pitch_mean_rad": nested(candidate_region, "signed_pitch_rad", "mean")
            - nested(base_region, "signed_pitch_rad", "mean"),
            "signed_pitch_p05_rad": nested(candidate_region, "signed_pitch_rad", "p05")
            - nested(base_region, "signed_pitch_rad", "p05"),
            "velocity_tracking_rmse_mps": nested(candidate_region, "velocity_tracking_rmse_mps")
            - nested(base_region, "velocity_tracking_rmse_mps"),
            "com_support_outside_mean_m": nested(
                candidate_region, "com_support_outside_m", "mean"
            )
            - nested(base_region, "com_support_outside_m", "mean"),
            "stance_slip_p95_mps": nested(candidate_region, "stance_slip_mps", "p95")
            - nested(base_region, "stance_slip_mps", "p95"),
            "flight_fraction": nested(candidate_region, "flight_fraction", "mean")
            - nested(base_region, "flight_fraction", "mean"),
        }
        count_checks = {}
        for side_name, phase_row in (("candidate", candidate_region), ("base", base_region)):
            count_checks[f"{side_name}_velocity_count"] = (
                int(phase_row["sample_count"]) >= count_minimum
            )
            for metric in (
                "signed_pitch_rad",
                "com_support_outside_m",
                "stance_slip_mps",
                "flight_fraction",
            ):
                count_checks[f"{side_name}_{metric}_count"] = (
                    int(phase_row[metric]["finite_count"]) >= count_minimum
                )
        checks = {
            **count_checks,
            "pitch_mean": delta["signed_pitch_mean_rad"]
            >= -gates["signed_pitch_mean_regression_max_rad"],
            "pitch_p05": delta["signed_pitch_p05_rad"]
            >= -gates["signed_pitch_p05_regression_max_rad"],
            "velocity": delta["velocity_tracking_rmse_mps"]
            <= gates["velocity_rmse_regression_max_mps"],
            "support": delta["com_support_outside_mean_m"]
            <= gates["com_support_mean_regression_max_m"],
            "slip": delta["stance_slip_p95_mps"]
            <= gates["stance_slip_p95_regression_max_mps"],
            "flight": delta["flight_fraction"] <= gates["flight_fraction_regression_max"],
        }
        result[region] = {"delta_vs_base": delta, "checks": checks, "passed": all(checks.values())}
    return result


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
    if report.get("phase") != 64 or report.get("mode") != "screen":
        raise RuntimeError("input is not a Phase64 screen")
    if report.get("posture_variant") != "knee_pitch_mirrored_dose":
        raise RuntimeError("Phase64 posture variant changed")
    screen = report.get("knee_pitch_dose_screen", {})
    if screen.get("normalized_action_doses") != prereg["screen"]["normalized_action_doses"]:
        raise RuntimeError("Phase64 dose contract changed")
    if screen.get("grouping") != "8x8_latin_square_slot_equals_row_plus_column_modulo_8":
        raise RuntimeError("Phase64 Latin-square allocation changed")
    if report.get("checkpoint_sha256") != prereg["source"]["checkpoint_sha256"]:
        raise RuntimeError("Phase64 source checkpoint changed")
    if resource.get("exit_code") != 0:
        raise RuntimeError("Phase64 resource ledger reports a failed screen")
    if nested(resource, "gpu", "memory_used_peak_mib") > prereg["resource_limits"]["gpu_peak_memory_mib_max"]:
        raise RuntimeError("Phase64 exceeded preregistered GPU memory limit")

    groups = report["groups"]
    phase_groups = report["gait_phase_semantics"]["metrics_by_group"]
    required_groups = {
        "A_base", "B_base", "base",
        *(f"{lane}_knee_{suffix}" for lane in ("A", "B") for suffix, _ in DOSE_SPECS),
        *(f"knee_{suffix}" for suffix, _ in DOSE_SPECS),
    }
    if not required_groups.issubset(groups) or not required_groups.issubset(phase_groups):
        raise RuntimeError("Phase64 report lacks a preregistered lane or pooled group")

    base_gate = prereg["base_balance_gates"]
    base_delta = metric_delta(groups["A_base"], groups["B_base"])
    base_checks = {
        "signed_pitch_mean": abs(base_delta["signed_pitch_mean_rad"])
        <= base_gate["signed_pitch_mean_abs_diff_max_rad"],
        "signed_pitch_p05": abs(base_delta["signed_pitch_p05_rad"])
        <= base_gate["signed_pitch_p05_abs_diff_max_rad"],
        "lateral": abs(base_delta["lateral_abs_m"]) <= base_gate["lateral_abs_diff_max_m"],
        "yaw": abs(base_delta["yaw_abs_rad"]) <= base_gate["yaw_abs_diff_max_rad"],
        "velocity": abs(base_delta["velocity_tracking_rmse_mps"])
        <= base_gate["velocity_rmse_abs_diff_max_mps"],
        "support": abs(base_delta["com_support_outside_mean_m"])
        <= base_gate["com_support_mean_abs_diff_max_m"],
    }
    base_balance_passed = all(base_checks.values())

    global_gate = prereg["global_gates"]
    phase_gate = prereg["phase_local_gates"]
    interaction_gate = prereg["lane_interaction_gates"]
    dose_rows = []
    passing = []
    for suffix, dose in DOSE_SPECS:
        comparisons = {}
        for comparison_name in ("A", "B", "pooled"):
            candidate_name = (
                f"{comparison_name}_knee_{suffix}"
                if comparison_name != "pooled"
                else f"knee_{suffix}"
            )
            base_name = f"{comparison_name}_base" if comparison_name != "pooled" else "base"
            delta = metric_delta(groups[candidate_name], groups[base_name])
            checks = global_checks(bool(report.get("finite")), groups[candidate_name], delta, global_gate)
            count_minimum = (
                phase_gate["lane_finite_count_min_per_region"]
                if comparison_name != "pooled"
                else phase_gate["pooled_finite_count_min_per_region"]
            )
            phase_checks = phase_comparison_checks(
                phase_groups[candidate_name], phase_groups[base_name], phase_gate, count_minimum
            )
            comparisons[comparison_name] = {
                "candidate_group": candidate_name,
                "base_group": base_name,
                "delta_vs_base": delta,
                "checks": checks,
                "phase_regions": phase_checks,
                "passed": all(checks.values()) and all(row["passed"] for row in phase_checks.values()),
            }

        delta_a = comparisons["A"]["delta_vs_base"]
        delta_b = comparisons["B"]["delta_vs_base"]
        interaction = {
            "signed_pitch_mean": abs(delta_a["signed_pitch_mean_rad"] - delta_b["signed_pitch_mean_rad"])
            <= interaction_gate["signed_pitch_mean_delta_abs_diff_max_rad"],
            "signed_pitch_p05": abs(delta_a["signed_pitch_p05_rad"] - delta_b["signed_pitch_p05_rad"])
            <= interaction_gate["signed_pitch_p05_delta_abs_diff_max_rad"],
            "lateral": abs(delta_a["lateral_abs_m"] - delta_b["lateral_abs_m"])
            <= interaction_gate["lateral_delta_abs_diff_max_m"],
            "yaw": abs(delta_a["yaw_abs_rad"] - delta_b["yaw_abs_rad"])
            <= interaction_gate["yaw_delta_abs_diff_max_rad"],
            "velocity": abs(
                delta_a["velocity_tracking_rmse_mps"] - delta_b["velocity_tracking_rmse_mps"]
            )
            <= interaction_gate["velocity_rmse_delta_abs_diff_max_mps"],
            "support": abs(
                delta_a["com_support_outside_mean_m"] - delta_b["com_support_outside_mean_m"]
            )
            <= interaction_gate["com_support_delta_abs_diff_max_m"],
            "slip": abs(delta_a["stance_slip_p95_mps"] - delta_b["stance_slip_p95_mps"])
            <= interaction_gate["stance_slip_p95_delta_abs_diff_max_mps"],
        }
        passed = (
            base_balance_passed
            and all(row["passed"] for row in comparisons.values())
            and all(interaction.values())
        )
        row = {
            "dose_suffix": suffix,
            "normalized_action_dose": dose,
            "comparisons": comparisons,
            "lane_interaction_checks": interaction,
            "passed": passed,
        }
        dose_rows.append(row)
        if passed:
            passing.append(row)

    selected = passing[0] if passing else None
    if not base_balance_passed:
        decision = "FAIL_INVALID_BASE_BALANCE_STOP"
    elif selected:
        decision = "PASS_LOCAL_KNEE_DOSE_WINDOW"
    else:
        decision = "FAIL_LOCAL_KNEE_DOSE_WINDOW_STOP"
    result = {
        "schema": "x2_knee_pitch_dose_phase64_result_v1",
        "decision": decision,
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "resource": {"path": str(args.resource), "sha256": sha256(args.resource), "report": resource},
        "preregistration": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
        "checkpoint_sha256": report["checkpoint_sha256"],
        "base_balance": {
            "A_group": "A_base",
            "B_group": "B_base",
            "delta_A_minus_B": base_delta,
            "checks": base_checks,
            "passed": base_balance_passed,
        },
        "dose_rows": dose_rows,
        "selected_group": f"knee_{selected['dose_suffix']}" if selected else None,
        "selected_normalized_action_dose": selected["normalized_action_dose"] if selected else None,
        "selection_rule": prereg["selection_rule"],
        "coordination_or_phase_diagnostic_unlocked": bool(selected),
        "phase_conditioned_stance_diagnostic_required": base_balance_passed and not bool(selected),
        "training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
        "boundary": "This zero-optimizer screen can select one diagnostic dose only; it cannot promote a checkpoint or satisfy official MuJoCo gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# X2 native posture Phase64 — mirrored knee-pitch dose screen",
        "",
        f"Decision: **{decision}**.",
        "",
        f"Base A/B balance: **{base_balance_passed}**.",
        "",
        "| dose | A pitch Δ | A velocity Δ | B pitch Δ | B velocity Δ | pooled pitch Δ | pooled velocity Δ | pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in dose_rows:
        a = row["comparisons"]["A"]["delta_vs_base"]
        b = row["comparisons"]["B"]["delta_vs_base"]
        pooled = row["comparisons"]["pooled"]["delta_vs_base"]
        lines.append(
            f"| {row['normalized_action_dose']:.3f} | {a['signed_pitch_mean_rad']:+.6f} | "
            f"{a['velocity_tracking_rmse_mps']:+.6f} | {b['signed_pitch_mean_rad']:+.6f} | "
            f"{b['velocity_tracking_rmse_mps']:+.6f} | {pooled['signed_pitch_mean_rad']:+.6f} | "
            f"{pooled['velocity_tracking_rmse_mps']:+.6f} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Selected group: `{result['selected_group']}`.",
            "",
            "Every lane, direct pooled, interaction, effective-dose, and four-region phase gate was applied fail-closed.",
            "No checkpoint, optimizer update, ONNX, or deployment backend was produced.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "selected": result["selected_group"]}, indent=2))


if __name__ == "__main__":
    main()
