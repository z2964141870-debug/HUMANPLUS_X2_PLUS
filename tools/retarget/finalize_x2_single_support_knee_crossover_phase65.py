#!/usr/bin/env python3
"""Pair the two Phase65 crossover passes and apply all gates fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REGIONS = (
    "double_support_zero",
    "right_swing_left_support",
    "double_support_half",
    "left_swing_right_support",
)
SINGLE_SUPPORT = {"right_swing_left_support", "left_swing_right_support"}
EXPECTED_COUNTS = (30, 70, 30, 70, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        raise RuntimeError("a required Phase65 sample distribution is empty")
    position = probability * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    blend = position - lower
    return finite[lower] * (1.0 - blend) + finite[upper] * blend


def sample_summary(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise RuntimeError("a required Phase65 sample distribution is empty")
    return {
        "finite_count": len(finite),
        "mean": sum(finite) / len(finite),
        "p05": quantile(finite, 0.05),
        "p50": quantile(finite, 0.50),
        "p95": quantile(finite, 0.95),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def raw(report: dict, name: str) -> list[list[float]]:
    return report["single_support_knee_crossover"]["raw_samples"][name]


def records(report: dict) -> dict[int, dict]:
    return {
        int(row["env_id"]): row
        for row in report["single_support_knee_crossover"]["per_env_records"]
    }


def select_raw(
    report: dict,
    metric: str,
    env_ids: list[int],
    *,
    region: int | None = None,
) -> list[float]:
    values = raw(report, metric)
    region_rows = report["single_support_knee_crossover"]["pre_step_region_index"]
    selected = []
    for step, row in enumerate(values):
        for env_id in env_ids:
            if region is None or int(region_rows[step][env_id]) == region:
                selected.append(float(row[env_id]))
    return selected


def aggregate(report: dict, env_ids: list[int], *, region: int | None = None) -> dict:
    row_by_env = records(report)
    result = {
        "signed_pitch_rad": sample_summary(select_raw(report, "signed_pitch_rad", env_ids, region=region)),
        "com_support_outside_m": sample_summary(
            select_raw(report, "com_support_outside_m", env_ids, region=region)
        ),
        "stance_slip_mps": sample_summary(
            select_raw(report, "stance_slip_mps", env_ids, region=region)
        ),
        "swing_sole_clearance_m": sample_summary(
            select_raw(report, "swing_sole_clearance_m", env_ids, region=region)
        ),
        "flight_fraction": sample_summary(
            select_raw(report, "flight_fraction", env_ids, region=region)
        ),
    }
    velocity_sq = select_raw(report, "velocity_tracking_sq", env_ids, region=region)
    finite_velocity = [value for value in velocity_sq if math.isfinite(value)]
    result["velocity_tracking_rmse_mps"] = math.sqrt(
        sum(finite_velocity) / len(finite_velocity)
    )
    result["velocity_sample_count"] = len(finite_velocity)
    if region is None:
        result.update(
            lateral_abs_m=sum(row_by_env[i]["lateral_abs_mean_m"] for i in env_ids) / len(env_ids),
            yaw_abs_rad=sum(row_by_env[i]["yaw_abs_mean_rad"] for i in env_ids) / len(env_ids),
            action_delta_abs=sum(row_by_env[i]["action_delta_abs_mean"] for i in env_ids) / len(env_ids),
            root_height_min_mean_m=sum(row_by_env[i]["root_height_min_m"] for i in env_ids) / len(env_ids),
            root_tilt_max_mean_rad=sum(row_by_env[i]["root_tilt_max_rad"] for i in env_ids) / len(env_ids),
            knee_left_excursion_mean_rad=sum(
                row_by_env[i]["knee_left_excursion_rad"] for i in env_ids
            ) / len(env_ids),
            knee_right_excursion_mean_rad=sum(
                row_by_env[i]["knee_right_excursion_rad"] for i in env_ids
            ) / len(env_ids),
            knee_left_right_asymmetry_mean_rad=sum(
                row_by_env[i]["knee_left_right_excursion_abs_diff_rad"] for i in env_ids
            ) / len(env_ids),
            survival_s_mean=sum(row_by_env[i]["survival_s"] for i in env_ids) / len(env_ids),
            termination_rate=sum(bool(row_by_env[i]["terminated"]) for i in env_ids) / len(env_ids),
        )
    return result


def delta(candidate: dict, control: dict, *, region: bool = False) -> dict[str, float]:
    result = {
        "signed_pitch_mean_rad": candidate["signed_pitch_rad"]["mean"]
        - control["signed_pitch_rad"]["mean"],
        "signed_pitch_p05_rad": candidate["signed_pitch_rad"]["p05"]
        - control["signed_pitch_rad"]["p05"],
        "velocity_tracking_rmse_mps": candidate["velocity_tracking_rmse_mps"]
        - control["velocity_tracking_rmse_mps"],
        "com_support_outside_mean_m": candidate["com_support_outside_m"]["mean"]
        - control["com_support_outside_m"]["mean"],
        "stance_slip_p95_mps": candidate["stance_slip_mps"]["p95"]
        - control["stance_slip_mps"]["p95"],
        "swing_clearance_p05_m": candidate["swing_sole_clearance_m"]["p05"]
        - control["swing_sole_clearance_m"]["p05"],
        "flight_fraction": candidate["flight_fraction"]["mean"]
        - control["flight_fraction"]["mean"],
    }
    if not region:
        result.update(
            lateral_abs_m=candidate["lateral_abs_m"] - control["lateral_abs_m"],
            yaw_abs_rad=candidate["yaw_abs_rad"] - control["yaw_abs_rad"],
            action_delta_abs=candidate["action_delta_abs"] - control["action_delta_abs"],
            root_height_min_mean_m=(
                candidate["root_height_min_mean_m"] - control["root_height_min_mean_m"]
            ),
            root_tilt_max_mean_rad=(
                candidate["root_tilt_max_mean_rad"] - control["root_tilt_max_mean_rad"]
            ),
            knee_left_excursion_mean_rad=(
                candidate["knee_left_excursion_mean_rad"]
                - control["knee_left_excursion_mean_rad"]
            ),
            knee_right_excursion_mean_rad=(
                candidate["knee_right_excursion_mean_rad"]
                - control["knee_right_excursion_mean_rad"]
            ),
            knee_left_right_asymmetry_mean_rad=(
                candidate["knee_left_right_asymmetry_mean_rad"]
                - control["knee_left_right_asymmetry_mean_rad"]
            ),
        )
    return result


def global_checks(
    comparison_name: str, candidate: dict, comparison_delta: dict, gates: dict
) -> dict[str, bool]:
    pitch_gate = (
        gates["pooled_signed_pitch_mean_improvement_min_rad"]
        if comparison_name == "pooled"
        else gates["sequence_signed_pitch_mean_improvement_min_rad"]
    )
    return {
        "survival": candidate["survival_s_mean"] >= gates["survival_s_min"] - 1.0e-6,
        "termination": candidate["termination_rate"] <= gates["termination_rate_max"],
        "pitch_mean": comparison_delta["signed_pitch_mean_rad"] >= pitch_gate,
        "pitch_p05": comparison_delta["signed_pitch_p05_rad"]
        >= gates["signed_pitch_p05_improvement_min_rad"],
        "lateral": comparison_delta["lateral_abs_m"] <= gates["lateral_abs_regression_max_m"],
        "yaw": comparison_delta["yaw_abs_rad"] <= gates["yaw_abs_regression_max_rad"],
        "velocity": comparison_delta["velocity_tracking_rmse_mps"]
        <= gates["velocity_rmse_regression_max_mps"],
        "support": comparison_delta["com_support_outside_mean_m"]
        <= gates["com_support_mean_regression_max_m"],
        "slip": comparison_delta["stance_slip_p95_mps"]
        <= gates["stance_slip_p95_regression_max_mps"],
        "root_height": comparison_delta["root_height_min_mean_m"]
        >= -gates["root_height_min_regression_max_m"],
        "root_tilt": comparison_delta["root_tilt_max_mean_rad"]
        <= gates["root_tilt_max_regression_max_rad"],
        "action_delta": comparison_delta["action_delta_abs"]
        <= gates["action_delta_abs_regression_max"],
        "clearance": comparison_delta["swing_clearance_p05_m"]
        >= -gates["swing_clearance_p05_regression_max_m"],
        "knee_left_excursion": abs(comparison_delta["knee_left_excursion_mean_rad"])
        <= gates["knee_excursion_abs_change_max_rad"],
        "knee_right_excursion": abs(comparison_delta["knee_right_excursion_mean_rad"])
        <= gates["knee_excursion_abs_change_max_rad"],
        "knee_asymmetry": comparison_delta["knee_left_right_asymmetry_mean_rad"]
        <= gates["knee_left_right_asymmetry_regression_max_rad"],
        "flight": comparison_delta["flight_fraction"] <= gates["flight_fraction_regression_max"],
    }


def phase_checks(
    comparison_name: str,
    region_name: str,
    candidate: dict,
    control: dict,
    comparison_delta: dict,
    gates: dict,
) -> dict[str, bool]:
    expected_per_env = EXPECTED_COUNTS[REGIONS.index(region_name)]
    env_count = 64 if comparison_name == "pooled" else 32
    expected = expected_per_env * env_count
    slip_expected = max(1, int(math.floor(expected * gates["slip_finite_fraction_min"])))
    non_slip_expected = int(expected * gates["non_slip_finite_fraction_min"])
    checks = {
        "candidate_pitch_coverage": candidate["signed_pitch_rad"]["finite_count"] >= non_slip_expected,
        "control_pitch_coverage": control["signed_pitch_rad"]["finite_count"] >= non_slip_expected,
        "candidate_support_coverage": candidate["com_support_outside_m"]["finite_count"] >= non_slip_expected,
        "control_support_coverage": control["com_support_outside_m"]["finite_count"] >= non_slip_expected,
        "candidate_velocity_coverage": candidate["velocity_sample_count"] >= non_slip_expected,
        "control_velocity_coverage": control["velocity_sample_count"] >= non_slip_expected,
        "candidate_slip_coverage": candidate["stance_slip_mps"]["finite_count"] >= slip_expected,
        "control_slip_coverage": control["stance_slip_mps"]["finite_count"] >= slip_expected,
        "velocity": comparison_delta["velocity_tracking_rmse_mps"]
        <= gates["velocity_rmse_regression_max_mps"],
        "support": comparison_delta["com_support_outside_mean_m"]
        <= gates["com_support_mean_regression_max_m"],
        "slip": comparison_delta["stance_slip_p95_mps"]
        <= gates["stance_slip_p95_regression_max_mps"],
        "flight": comparison_delta["flight_fraction"] <= gates["flight_fraction_regression_max"],
    }
    if region_name in SINGLE_SUPPORT:
        pitch_gate = (
            gates["single_support_pooled_signed_pitch_mean_improvement_min_rad"]
            if comparison_name == "pooled"
            else gates["single_support_sequence_signed_pitch_mean_improvement_min_rad"]
        )
        checks.update(
            pitch_mean=comparison_delta["signed_pitch_mean_rad"] >= pitch_gate,
            pitch_p05=comparison_delta["signed_pitch_p05_rad"]
            >= gates["single_support_signed_pitch_p05_improvement_min_rad"],
        )
    else:
        checks.update(
            pitch_mean=comparison_delta["signed_pitch_mean_rad"]
            >= -gates["double_support_signed_pitch_mean_regression_max_rad"],
            pitch_p05=comparison_delta["signed_pitch_p05_rad"]
            >= -gates["double_support_signed_pitch_p05_regression_max_rad"],
        )
    return checks


def realized_bias(report: dict, env_ids: list[int]) -> dict:
    region_rows = report["single_support_knee_crossover"]["pre_step_region_index"]
    active_regions = {1, 3}
    result = {}
    for side in ("left", "right"):
        requested_rows = raw(report, f"knee_requested_bias_{side}")
        effective_rows = raw(report, f"knee_effective_bias_{side}")
        target_rows = raw(report, f"knee_effective_target_offset_{side}_rad")
        requested = []
        effective = []
        targets = []
        for step in range(len(region_rows)):
            for env_id in env_ids:
                if int(region_rows[step][env_id]) in active_regions:
                    requested.append(float(requested_rows[step][env_id]))
                    effective.append(float(effective_rows[step][env_id]))
                    targets.append(float(target_rows[step][env_id]))
        requested_summary = sample_summary(requested)
        effective_summary = sample_summary(effective)
        target_summary = sample_summary(targets)
        result[side] = {
            "requested": requested_summary,
            "effective": effective_summary,
            "target_offset_rad": target_summary,
            "effective_to_requested_mean_ratio": abs(effective_summary["mean"])
            / max(abs(requested_summary["mean"]), 1.0e-12),
            "wrong_sign_fraction": sum(value >= 0.0 for value in effective) / len(effective),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass0", type=Path, required=True)
    parser.add_argument("--pass1", type=Path, required=True)
    parser.add_argument("--resource0", type=Path, required=True)
    parser.add_argument("--resource1", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    pass0 = json.loads(args.pass0.read_text(encoding="utf-8"))
    pass1 = json.loads(args.pass1.read_text(encoding="utf-8"))
    resource0 = json.loads(args.resource0.read_text(encoding="utf-8"))
    resource1 = json.loads(args.resource1.read_text(encoding="utf-8"))
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    reports = (pass0, pass1)
    resources = (resource0, resource1)
    for index, (report, resource) in enumerate(zip(reports, resources, strict=True)):
        screen = report.get("single_support_knee_crossover", {})
        if report.get("phase") != 65 or report.get("mode") != "screen":
            raise RuntimeError(f"pass{index} is not a Phase65 screen")
        if report.get("posture_variant") != "single_support_knee_crossover":
            raise RuntimeError(f"pass{index} posture variant changed")
        if screen.get("pass_index") != index:
            raise RuntimeError(f"pass{index} index contract changed")
        if screen.get("normalized_action_dose") != prereg["screen"]["nominal_normalized_action_dose"]:
            raise RuntimeError(f"pass{index} dose contract changed")
        if report.get("checkpoint_sha256") != prereg["source"]["checkpoint_sha256"]:
            raise RuntimeError(f"pass{index} checkpoint changed")
        if resource.get("exit_code") != 0:
            raise RuntimeError(f"pass{index} resource ledger reports failure")
        if resource["gpu"]["memory_used_peak_mib"] > prereg["resource_limits"]["gpu_peak_memory_mib_max_each_run"]:
            raise RuntimeError(f"pass{index} exceeded GPU memory limit")

    screen0 = pass0["single_support_knee_crossover"]
    screen1 = pass1["single_support_knee_crossover"]
    pairing_checks = {
        "checkpoint": pass0["checkpoint_sha256"] == pass1["checkpoint_sha256"],
        "runner": screen0["runner_sha256"] == screen1["runner_sha256"],
        "domain_upper": pass0["domain_upper_counts"] == pass1["domain_upper_counts"],
        "initial_fingerprints": screen0["initial_fingerprints"] == screen1["initial_fingerprints"],
        "first_three_ds0_replay": (
            screen0["first_three_ds0_replay_hashes"]
            == screen1["first_three_ds0_replay_hashes"]
        ),
        "pre_step_regions": screen0["pre_step_region_index"] == screen1["pre_step_region_index"],
        "region_counts": all(
            counts == [expected] * 64
            for screen in (screen0, screen1)
            for counts, expected in zip(
                screen["pre_step_region_counts_per_env"].values(), EXPECTED_COUNTS, strict=True
            )
        ),
        "invalid_contact_suffix": all(
            screen["bias_contract"]["invalid_contact_suffix_count"]
            <= prereg["pairing_gates"]["invalid_contact_suffix_count_max"]
            for screen in (screen0, screen1)
        ),
    }
    pairing_passed = all(pairing_checks.values())

    sequence_ids = {
        "TC": [env_id for env_id in range(64) if (env_id // 8 + env_id % 8) % 2 == 0],
        "CT": [env_id for env_id in range(64) if (env_id // 8 + env_id % 8) % 2 == 1],
        "pooled": list(range(64)),
    }
    comparisons = {}
    realized = {}
    global_gate = prereg["global_gates"]
    phase_gate = prereg["phase_local_gates"]
    for name, env_ids in sequence_ids.items():
        if name == "TC":
            treatment_report, control_report = pass0, pass1
        elif name == "CT":
            treatment_report, control_report = pass1, pass0
        else:
            treatment_report = control_report = None
        if name == "pooled":
            treatment_global_parts = [
                (pass0, sequence_ids["TC"]),
                (pass1, sequence_ids["CT"]),
            ]
            control_global_parts = [
                (pass1, sequence_ids["TC"]),
                (pass0, sequence_ids["CT"]),
            ]

            def combined_aggregate(parts, region=None):
                # Rebuild one synthetic report by directly concatenating the
                # two original distributions; never average quantiles/RMSE.
                metrics = {}
                raw_names = pass0["single_support_knee_crossover"]["raw_samples"]
                for metric in raw_names:
                    values = []
                    for report_part, ids_part in parts:
                        values.extend(select_raw(report_part, metric, ids_part, region=region))
                    metrics[metric] = values
                summary = {
                    "signed_pitch_rad": sample_summary(metrics["signed_pitch_rad"]),
                    "com_support_outside_m": sample_summary(metrics["com_support_outside_m"]),
                    "stance_slip_mps": sample_summary(metrics["stance_slip_mps"]),
                    "swing_sole_clearance_m": sample_summary(metrics["swing_sole_clearance_m"]),
                    "flight_fraction": sample_summary(metrics["flight_fraction"]),
                    "velocity_tracking_rmse_mps": math.sqrt(
                        sum(metrics["velocity_tracking_sq"]) / len(metrics["velocity_tracking_sq"])
                    ),
                    "velocity_sample_count": len(metrics["velocity_tracking_sq"]),
                }
                if region is None:
                    record_rows = [
                        records(report_part)[env_id]
                        for report_part, ids_part in parts
                        for env_id in ids_part
                    ]
                    summary.update(
                        lateral_abs_m=sum(r["lateral_abs_mean_m"] for r in record_rows) / 64,
                        yaw_abs_rad=sum(r["yaw_abs_mean_rad"] for r in record_rows) / 64,
                        action_delta_abs=sum(r["action_delta_abs_mean"] for r in record_rows) / 64,
                        root_height_min_mean_m=sum(r["root_height_min_m"] for r in record_rows) / 64,
                        root_tilt_max_mean_rad=sum(r["root_tilt_max_rad"] for r in record_rows) / 64,
                        knee_left_excursion_mean_rad=sum(r["knee_left_excursion_rad"] for r in record_rows) / 64,
                        knee_right_excursion_mean_rad=sum(r["knee_right_excursion_rad"] for r in record_rows) / 64,
                        knee_left_right_asymmetry_mean_rad=sum(
                            r["knee_left_right_excursion_abs_diff_rad"] for r in record_rows
                        ) / 64,
                        survival_s_mean=sum(r["survival_s"] for r in record_rows) / 64,
                        termination_rate=sum(bool(r["terminated"]) for r in record_rows) / 64,
                    )
                return summary

            candidate = combined_aggregate(treatment_global_parts)
            control = combined_aggregate(control_global_parts)
        else:
            candidate = aggregate(treatment_report, env_ids)
            control = aggregate(control_report, env_ids)
        comparison_delta = delta(candidate, control)
        checks = global_checks(name, candidate, comparison_delta, global_gate)
        phase_rows = {}
        for region_index, region_name in enumerate(REGIONS):
            if name == "pooled":
                phase_candidate = combined_aggregate(treatment_global_parts, region_index)
                phase_control = combined_aggregate(control_global_parts, region_index)
            else:
                phase_candidate = aggregate(treatment_report, env_ids, region=region_index)
                phase_control = aggregate(control_report, env_ids, region=region_index)
            phase_delta = delta(phase_candidate, phase_control, region=True)
            checks_region = phase_checks(
                name, region_name, phase_candidate, phase_control, phase_delta, phase_gate
            )
            phase_rows[region_name] = {
                "candidate": phase_candidate,
                "control": phase_control,
                "delta": phase_delta,
                "checks": checks_region,
                "passed": all(checks_region.values()),
            }
        comparisons[name] = {
            "candidate": candidate,
            "control": control,
            "delta": comparison_delta,
            "checks": checks,
            "phase_regions": phase_rows,
            "passed": all(checks.values()) and all(row["passed"] for row in phase_rows.values()),
        }
        if name != "pooled":
            realized[name] = realized_bias(treatment_report, env_ids)

    # Pooled realized distributions are direct concatenations of the two
    # sequence-specific active treatment distributions.
    realized["pooled"] = {
        side: {
            key: (
                sample_summary(
                    [
                        value
                        for sequence in ("TC", "CT")
                        for value in (
                            [realized[sequence][side][key]["mean"]]
                            if key in {"requested", "effective", "target_offset_rad"}
                            else []
                        )
                    ]
                )
                if key in {"requested", "effective", "target_offset_rad"}
                else None
            )
            for key in ("requested", "effective", "target_offset_rad")
        }
        for side in ("left", "right")
    }
    # Replace the compact pooled approximation above with exact count-weighted
    # summaries for the mean-based realized-dose gates.
    for side in ("left", "right"):
        for key in ("requested", "effective", "target_offset_rad"):
            total = sum(realized[s][side][key]["finite_count"] for s in ("TC", "CT"))
            mean = sum(
                realized[s][side][key]["mean"] * realized[s][side][key]["finite_count"]
                for s in ("TC", "CT")
            ) / total
            realized["pooled"][side][key] = {"finite_count": total, "mean": mean}
        realized["pooled"][side]["effective_to_requested_mean_ratio"] = abs(
            realized["pooled"][side]["effective"]["mean"]
        ) / abs(realized["pooled"][side]["requested"]["mean"])
        realized["pooled"][side]["wrong_sign_fraction"] = sum(
            realized[s][side]["wrong_sign_fraction"] for s in ("TC", "CT")
        ) / 2.0

    bias_gate = prereg["realized_bias_gates"]
    realized_checks = {}
    for name in ("TC", "CT", "pooled"):
        row = realized[name]
        checks = {
            f"{side}_ratio": bias_gate["effective_to_requested_mean_ratio_min"]
            <= row[side]["effective_to_requested_mean_ratio"]
            <= bias_gate["effective_to_requested_mean_ratio_max"]
            for side in ("left", "right")
        }
        for side in ("left", "right"):
            checks[f"{side}_range"] = (
                bias_gate["effective_normalized_mean_min"]
                <= row[side]["effective"]["mean"]
                <= bias_gate["effective_normalized_mean_max"]
            )
            checks[f"{side}_sign"] = (
                row[side]["wrong_sign_fraction"] <= bias_gate["wrong_sign_fraction_max"]
            )
        checks["left_right"] = abs(
            row["left"]["effective"]["mean"] - row["right"]["effective"]["mean"]
        ) <= bias_gate["left_right_effective_mean_abs_diff_max"]
        realized_checks[name] = checks
    sequence_effective_diff = max(
        abs(realized["TC"][side]["effective"]["mean"] - realized["CT"][side]["effective"]["mean"])
        for side in ("left", "right")
    )
    inactive_checks = {
        f"pass{index}_inactive_requested": screen["bias_contract"]["inactive_requested_max_abs"]
        <= bias_gate["inactive_requested_max_abs"]
        for index, screen in enumerate((screen0, screen1))
    }
    inactive_checks.update(
        {
            f"pass{index}_inactive_effective": screen["bias_contract"]["inactive_effective_max_abs"]
            <= bias_gate["inactive_effective_max_abs"]
            for index, screen in enumerate((screen0, screen1))
        }
    )
    inactive_checks.update(
        {
            f"pass{index}_shadow_standing": screen["bias_contract"]["shadow_standing_requested_max_abs"]
            <= bias_gate["shadow_standing_requested_max_abs"]
            for index, screen in enumerate((screen0, screen1))
        }
    )
    realized_passed = (
        all(all(row.values()) for row in realized_checks.values())
        and all(inactive_checks.values())
        and sequence_effective_diff <= bias_gate["sequence_effective_mean_abs_diff_max"]
    )

    tc_delta = comparisons["TC"]["delta"]
    ct_delta = comparisons["CT"]["delta"]
    interaction_gate = prereg["sequence_interaction_gates"]
    interaction_checks = {
        "signed_pitch_mean": abs(tc_delta["signed_pitch_mean_rad"] - ct_delta["signed_pitch_mean_rad"])
        <= interaction_gate["signed_pitch_mean_delta_abs_diff_max_rad"],
        "signed_pitch_p05": abs(tc_delta["signed_pitch_p05_rad"] - ct_delta["signed_pitch_p05_rad"])
        <= interaction_gate["signed_pitch_p05_delta_abs_diff_max_rad"],
        "lateral": abs(tc_delta["lateral_abs_m"] - ct_delta["lateral_abs_m"])
        <= interaction_gate["lateral_delta_abs_diff_max_m"],
        "yaw": abs(tc_delta["yaw_abs_rad"] - ct_delta["yaw_abs_rad"])
        <= interaction_gate["yaw_delta_abs_diff_max_rad"],
        "velocity": abs(
            tc_delta["velocity_tracking_rmse_mps"] - ct_delta["velocity_tracking_rmse_mps"]
        )
        <= interaction_gate["velocity_rmse_delta_abs_diff_max_mps"],
        "support": abs(
            tc_delta["com_support_outside_mean_m"] - ct_delta["com_support_outside_mean_m"]
        )
        <= interaction_gate["com_support_delta_abs_diff_max_m"],
        "slip": abs(tc_delta["stance_slip_p95_mps"] - ct_delta["stance_slip_p95_mps"])
        <= interaction_gate["stance_slip_p95_delta_abs_diff_max_mps"],
    }

    if not pairing_passed:
        decision = "FAIL_CROSSOVER_REPLAY_INVALID_STOP"
    elif (
        all(report.get("finite") is True for report in reports)
        and all(row["passed"] for row in comparisons.values())
        and all(interaction_checks.values())
        and realized_passed
    ):
        decision = "PASS_LOCAL_SINGLE_SUPPORT_KNEE_CROSSOVER"
    else:
        decision = "FAIL_LOCAL_SINGLE_SUPPORT_KNEE_CROSSOVER_STOP"
    result = {
        "schema": "x2_single_support_knee_crossover_phase65_result_v1",
        "decision": decision,
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in (args.pass0, args.pass1)
        ],
        "resources": [
            {"path": str(path), "sha256": sha256(path), "report": report}
            for path, report in zip((args.resource0, args.resource1), resources, strict=True)
        ],
        "preregistration": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
        "pairing": {"checks": pairing_checks, "passed": pairing_passed},
        "comparisons": comparisons,
        "sequence_interaction_checks": interaction_checks,
        "realized_bias": realized,
        "realized_bias_checks": realized_checks,
        "inactive_bias_checks": inactive_checks,
        "sequence_effective_mean_max_abs_diff": sequence_effective_diff,
        "realized_bias_passed": realized_passed,
        "selection_rule": prereg["selection_rule"],
        "stance_swing_role_diagnostic_unlocked": decision == "PASS_LOCAL_SINGLE_SUPPORT_KNEE_CROSSOVER",
        "training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
        "boundary": "This paired zero-optimizer screen tests one deployable timing rule only; it cannot promote the frozen checkpoint or satisfy official MuJoCo gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# X2 native posture Phase65 — single-support knee crossover",
        "",
        f"Decision: **{decision}**.",
        "",
        f"Exact crossover pairing: **{pairing_passed}**. Realized-dose contract: **{realized_passed}**.",
        "",
        "| comparison | pitch mean Δ | pitch p05 Δ | support Δ | velocity Δ | pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("TC", "CT", "pooled"):
        row = comparisons[name]
        value = row["delta"]
        lines.append(
            f"| `{name}` | {value['signed_pitch_mean_rad']:+.6f} | "
            f"{value['signed_pitch_p05_rad']:+.6f} | "
            f"{value['com_support_outside_mean_m']:+.6f} | "
            f"{value['velocity_tracking_rmse_mps']:+.6f} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            "Nonlinear p05/p95 and RMSE values were recomputed from the original matched samples; per-env quantiles were not averaged.",
            "Coverage counts prove complete phase coverage only and are not interpreted as independent samples.",
            "No checkpoint, optimizer update, ONNX, or deployment backend was produced.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "pairing": pairing_passed}, indent=2))


if __name__ == "__main__":
    main()
