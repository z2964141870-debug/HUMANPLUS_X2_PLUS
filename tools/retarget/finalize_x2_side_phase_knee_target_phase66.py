#!/usr/bin/env python3
"""Pair Phase66 side-by-phase cells with same-index controls fail-closed."""

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
EXPECTED_COUNTS = (29, 70, 30, 71, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float | int]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        raise RuntimeError("Phase66 required distribution is empty")

    def q(probability: float) -> float:
        position = probability * (len(finite) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return finite[lo] * (hi - position) + finite[hi] * (position - lo) if hi != lo else finite[lo]

    return {
        "finite_count": len(finite),
        "mean": sum(finite) / len(finite),
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "minimum": finite[0],
        "maximum": finite[-1],
    }


def screen(report: dict) -> dict:
    return report["side_phase_physical_knee_target"]


def select_raw(
    report: dict, metric: str, env_ids: list[int], region_index: int | None = None
) -> list[float]:
    rows = screen(report)["raw_samples"][metric]
    regions = screen(report)["pre_step_region_index"]
    return [
        float(row[env_id])
        for step, row in enumerate(rows)
        for env_id in env_ids
        if region_index is None or int(regions[step][env_id]) == region_index
    ]


def record_map(report: dict) -> dict[int, dict]:
    return {int(row["env_id"]): row for row in screen(report)["per_env_records"]}


def aggregate(report: dict, env_ids: list[int], region_index: int | None = None) -> dict:
    pitch = summary(select_raw(report, "signed_pitch_rad", env_ids, region_index))
    support = summary(select_raw(report, "com_support_outside_m", env_ids, region_index))
    slip = summary(select_raw(report, "stance_slip_mps", env_ids, region_index))
    clearance = summary(select_raw(report, "swing_sole_clearance_m", env_ids, region_index))
    flight = summary(select_raw(report, "flight_fraction", env_ids, region_index))
    velocity = [
        value
        for value in select_raw(report, "velocity_tracking_sq", env_ids, region_index)
        if math.isfinite(value)
    ]
    result = {
        "signed_pitch_rad": pitch,
        "com_support_outside_m": support,
        "stance_slip_mps": slip,
        "swing_sole_clearance_m": clearance,
        "flight_fraction": flight,
        "velocity_tracking_rmse_mps": math.sqrt(sum(velocity) / len(velocity)),
        "velocity_sample_count": len(velocity),
    }
    if region_index is None:
        records = record_map(report)
        result.update(
            lateral_abs_m=sum(records[i]["lateral_abs_mean_m"] for i in env_ids) / len(env_ids),
            yaw_abs_rad=sum(records[i]["yaw_abs_mean_rad"] for i in env_ids) / len(env_ids),
            action_delta_abs=sum(records[i]["action_delta_abs_mean"] for i in env_ids) / len(env_ids),
            root_height_min_mean_m=sum(records[i]["root_height_min_m"] for i in env_ids) / len(env_ids),
            root_tilt_max_mean_rad=sum(records[i]["root_tilt_max_rad"] for i in env_ids) / len(env_ids),
            knee_left_excursion_mean_rad=sum(
                records[i]["knee_left_excursion_rad"] for i in env_ids
            ) / len(env_ids),
            knee_right_excursion_mean_rad=sum(
                records[i]["knee_right_excursion_rad"] for i in env_ids
            ) / len(env_ids),
            knee_left_right_asymmetry_mean_rad=sum(
                records[i]["knee_left_right_excursion_abs_diff_rad"] for i in env_ids
            ) / len(env_ids),
            survival_s_mean=sum(records[i]["survival_s"] for i in env_ids) / len(env_ids),
            termination_rate=sum(bool(records[i]["terminated"]) for i in env_ids) / len(env_ids),
        )
    return result


def difference(candidate: dict, control: dict, *, phase: bool = False) -> dict[str, float]:
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
    if not phase:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate-resource", type=Path, required=True)
    parser.add_argument("--control-resource", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8"))
    candidate_resource = json.loads(args.candidate_resource.read_text(encoding="utf-8"))
    control_resource = json.loads(args.control_resource.read_text(encoding="utf-8"))
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    reports = (candidate, control)
    resources = (candidate_resource, control_resource)
    for index, (report, resource) in enumerate(zip(reports, resources, strict=True)):
        if report.get("phase") != 66 or report.get("mode") != "screen":
            raise RuntimeError(f"pass{index} is not Phase66 screen output")
        if report.get("posture_variant") != "side_phase_physical_knee_target":
            raise RuntimeError(f"pass{index} posture variant changed")
        if screen(report)["pass_index"] != index:
            raise RuntimeError(f"pass{index} contract changed")
        if report["checkpoint_sha256"] != prereg["source"]["checkpoint_sha256"]:
            raise RuntimeError(f"pass{index} checkpoint changed")
        if resource.get("exit_code") != 0:
            raise RuntimeError(f"pass{index} resource ledger reports failure")
        if resource["gpu"]["memory_used_peak_mib"] > prereg["resource_limits"]["gpu_peak_memory_mib_max_each_run"]:
            raise RuntimeError(f"pass{index} exceeded GPU memory limit")

    candidate_screen, control_screen = screen(candidate), screen(control)
    non_ds0_envs = [env_id for env_id, slot in enumerate(candidate_screen["condition_slot_by_env"]) if slot >= 2]
    replay_exact = True
    for candidate_step, control_step in zip(
        candidate_screen["first_three_ds0_per_env_replay_hashes"],
        control_screen["first_three_ds0_per_env_replay_hashes"],
        strict=True,
    ):
        for name in candidate_step["per_env"]:
            replay_exact &= all(
                candidate_step["per_env"][name][env_id]
                == control_step["per_env"][name][env_id]
                for env_id in non_ds0_envs
            )
    counts_exact = all(
        counts == [expected] * 64
        for observed in (candidate_screen, control_screen)
        for counts, expected in zip(
            observed["pre_step_region_counts_per_env"].values(), EXPECTED_COUNTS, strict=True
        )
    )
    pairing_checks = {
        "checkpoint": candidate["checkpoint_sha256"] == control["checkpoint_sha256"],
        "runner": candidate_screen["runner_sha256"] == control_screen["runner_sha256"],
        "domain_upper": candidate["domain_upper_counts"] == control["domain_upper_counts"],
        "initial_fingerprints": (
            candidate_screen["initial_fingerprints"] == control_screen["initial_fingerprints"]
        ),
        "non_DS0_first_three_replay": replay_exact,
        "pre_step_regions": (
            candidate_screen["pre_step_region_index"] == control_screen["pre_step_region_index"]
        ),
        "region_counts": counts_exact,
        "invalid_contact_suffix": all(
            observed["contract"]["invalid_contact_suffix_count"]
            <= prereg["pairing_gates"]["invalid_contact_suffix_count_max"]
            for observed in (candidate_screen, control_screen)
        ),
    }
    pairing_passed = all(pairing_checks.values())

    names = candidate_screen["condition_names"]
    gates = prereg["condition_gates"]
    target_gates = prereg["target_contract_gates"]
    rows = []
    for slot, name in enumerate(names):
        env_ids = [env_id for env_id, value in enumerate(candidate_screen["condition_slot_by_env"]) if value == slot]
        candidate_global = aggregate(candidate, env_ids)
        control_global = aggregate(control, env_ids)
        global_delta = difference(candidate_global, control_global)
        active_region_index = slot // 2
        next_region_index = (active_region_index + 1) % 4
        candidate_active = aggregate(candidate, env_ids, active_region_index)
        control_active = aggregate(control, env_ids, active_region_index)
        active_delta = difference(candidate_active, control_active, phase=True)
        candidate_next = aggregate(candidate, env_ids, next_region_index)
        control_next = aggregate(control, env_ids, next_region_index)
        next_delta = difference(candidate_next, control_next, phase=True)

        side = "left" if slot % 2 == 0 else "right"
        other = "right" if side == "left" else "left"
        requested = select_raw(candidate, f"requested_target_{side}_rad", env_ids, active_region_index)
        effective = select_raw(candidate, f"effective_target_{side}_rad", env_ids, active_region_index)
        inactive_requested = [
            value
            for region in range(4)
            if region != active_region_index
            for value in select_raw(candidate, f"requested_target_{side}_rad", env_ids, region)
        ] + select_raw(candidate, f"requested_target_{other}_rad", env_ids)
        inactive_effective = [
            value
            for region in range(4)
            if region != active_region_index
            for value in select_raw(candidate, f"effective_target_{side}_rad", env_ids, region)
        ] + select_raw(candidate, f"effective_target_{other}_rad", env_ids)
        requested_summary = summary(requested)
        effective_summary = summary(effective)
        target_checks = {
            "requested": abs(
                requested_summary["mean"] - target_gates["active_requested_target_mean_rad"]
            ) <= 1.0e-8,
            "effective_ratio": abs(effective_summary["mean"])
            / max(abs(requested_summary["mean"]), 1.0e-12)
            >= target_gates["active_effective_to_requested_mean_ratio_min"],
            "effective_sign": sum(value >= 0.0 for value in effective) / len(effective)
            <= target_gates["active_wrong_sign_fraction_max"],
            "inactive_requested": max(abs(value) for value in inactive_requested)
            <= target_gates["inactive_requested_max_abs_rad"],
            "inactive_effective": max(abs(value) for value in inactive_effective)
            <= target_gates["inactive_effective_max_abs_rad"],
            "source_limit": max(
                abs(value)
                for axis in ("left", "right")
                for value in select_raw(candidate, f"source_target_{axis}_rad", env_ids)
            ) <= target_gates["source_target_limit_abs_max_rad"],
            "intervention_limit": max(
                abs(value)
                for axis in ("left", "right")
                for value in select_raw(candidate, f"intervention_target_{axis}_rad", env_ids)
            ) <= target_gates["intervention_target_limit_abs_max_rad"],
        }
        checks = {
            "finite": candidate.get("finite") is True and control.get("finite") is True,
            "survival": candidate_global["survival_s_mean"] >= gates["survival_s_min"] - 1.0e-6,
            "termination": candidate_global["termination_rate"] <= gates["termination_rate_max"],
            "global_pitch_mean": global_delta["signed_pitch_mean_rad"]
            >= gates["global_signed_pitch_mean_improvement_min_rad"],
            "global_pitch_p05": global_delta["signed_pitch_p05_rad"]
            >= gates["global_signed_pitch_p05_improvement_min_rad"],
            "active_pitch_mean": active_delta["signed_pitch_mean_rad"]
            >= gates["active_phase_signed_pitch_mean_improvement_min_rad"],
            "active_pitch_p05": active_delta["signed_pitch_p05_rad"]
            >= gates["active_phase_signed_pitch_p05_improvement_min_rad"],
            "lateral": global_delta["lateral_abs_m"] <= gates["global_lateral_abs_regression_max_m"],
            "yaw": global_delta["yaw_abs_rad"] <= gates["global_yaw_abs_regression_max_rad"],
            "global_velocity": global_delta["velocity_tracking_rmse_mps"]
            <= gates["global_velocity_rmse_regression_max_mps"],
            "global_support": global_delta["com_support_outside_mean_m"]
            <= gates["global_com_support_mean_regression_max_m"],
            "global_slip": global_delta["stance_slip_p95_mps"]
            <= gates["global_stance_slip_p95_regression_max_mps"],
            "global_flight": global_delta["flight_fraction"]
            <= gates["global_flight_fraction_regression_max"],
            "active_velocity": active_delta["velocity_tracking_rmse_mps"]
            <= gates["active_phase_velocity_rmse_regression_max_mps"],
            "active_support": active_delta["com_support_outside_mean_m"]
            <= gates["active_phase_com_support_mean_regression_max_m"],
            "active_slip": active_delta["stance_slip_p95_mps"]
            <= gates["active_phase_stance_slip_p95_regression_max_mps"],
            "active_flight": active_delta["flight_fraction"]
            <= gates["active_phase_flight_fraction_regression_max"],
            "next_support": next_delta["com_support_outside_mean_m"]
            <= gates["next_phase_com_support_mean_regression_max_m"],
            "next_slip": next_delta["stance_slip_p95_mps"]
            <= gates["next_phase_stance_slip_p95_regression_max_mps"],
            "root_height": global_delta["root_height_min_mean_m"]
            >= -gates["root_height_min_regression_max_m"],
            "root_tilt": global_delta["root_tilt_max_mean_rad"]
            <= gates["root_tilt_max_regression_max_rad"],
            "action_delta": global_delta["action_delta_abs"]
            <= gates["action_delta_abs_regression_max"],
            "clearance": global_delta["swing_clearance_p05_m"]
            >= -gates["swing_clearance_p05_regression_max_m"],
            "knee_left_excursion": abs(global_delta["knee_left_excursion_mean_rad"])
            <= gates["knee_excursion_abs_change_max_rad"],
            "knee_right_excursion": abs(global_delta["knee_right_excursion_mean_rad"])
            <= gates["knee_excursion_abs_change_max_rad"],
            "knee_asymmetry": global_delta["knee_left_right_asymmetry_mean_rad"]
            <= gates["knee_left_right_asymmetry_regression_max_rad"],
        }
        passed = all(checks.values()) and all(target_checks.values())
        rows.append(
            {
                "condition": name,
                "env_ids": env_ids,
                "active_region": REGIONS[active_region_index],
                "next_region": REGIONS[next_region_index],
                "global_delta": global_delta,
                "active_phase_delta": active_delta,
                "next_phase_delta": next_delta,
                "requested_target": requested_summary,
                "effective_target": effective_summary,
                "target_checks": target_checks,
                "checks": checks,
                "passed": passed,
            }
        )

    row_by_name = {row["condition"]: row for row in rows}
    role_rule = prereg["role_rule"]
    role_passes = {
        "stance": all(row_by_name[name]["passed"] for name in role_rule["stance_pair"]),
        "swing": all(row_by_name[name]["passed"] for name in role_rule["swing_pair"]),
        "double_support_zero": all(
            row_by_name[name]["passed"] for name in role_rule["double_support_pairs"][0]
        ),
        "double_support_half": all(
            row_by_name[name]["passed"] for name in role_rule["double_support_pairs"][1]
        ),
    }
    identified_roles = [name for name, passed in role_passes.items() if passed]
    if not pairing_passed:
        decision = "FAIL_SIDE_PHASE_REPLAY_INVALID_STOP"
    elif identified_roles:
        decision = "PASS_LOCAL_MIRRORED_PHASE_ROLE_IDENTIFIED"
    else:
        decision = "FAIL_LOCAL_SIDE_PHASE_DECOMPOSITION_STOP"
    result = {
        "schema": "x2_side_phase_knee_target_phase66_result_v1",
        "decision": decision,
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in (args.candidate, args.control)
        ],
        "resources": [
            {"path": str(path), "sha256": sha256(path), "report": report}
            for path, report in zip(
                (args.candidate_resource, args.control_resource), resources, strict=True
            )
        ],
        "preregistration": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
        "pairing": {"checks": pairing_checks, "passed": pairing_passed},
        "condition_rows": rows,
        "role_passes": role_passes,
        "identified_mirrored_roles": identified_roles,
        "composition_diagnostic_unlocked": pairing_passed and bool(identified_roles),
        "training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
        "boundary": "A cell or mirrored role is diagnostic causal evidence only; no target shim, checkpoint, or deployment artifact is promoted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# X2 native posture Phase66 — side-by-phase physical knee target",
        "",
        f"Decision: **{decision}**. Pairing exact: **{pairing_passed}**.",
        "",
        "| condition | global pitch Δ | active pitch Δ | global support Δ | next support Δ | pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['global_delta']['signed_pitch_mean_rad']:+.6f} | "
            f"{row['active_phase_delta']['signed_pitch_mean_rad']:+.6f} | "
            f"{row['global_delta']['com_support_outside_mean_m']:+.6f} | "
            f"{row['next_phase_delta']['com_support_outside_mean_m']:+.6f} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Mirrored roles identified: `{identified_roles}`.",
            "",
            "All nonlinear quantiles were recomputed from original same-index samples. No optimizer, checkpoint, ONNX, or deployment backend was produced.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "roles": identified_roles}, indent=2))


if __name__ == "__main__":
    main()
