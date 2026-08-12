#!/usr/bin/env python3
"""Apply the preregistered Phase68 technical, event-safety, and efficacy gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-resource", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-resource", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    train = json.loads(args.train.read_text())
    source_report = json.loads(args.source.read_text())
    candidate_report = json.loads(args.candidate.read_text())
    resources = {
        "train": json.loads(args.train_resource.read_text()),
        "source": json.loads(args.source_resource.read_text()),
        "candidate": json.loads(args.candidate_resource.read_text()),
    }
    source = source_report["groups"]["all"]
    candidate = candidate_report["groups"]["all"]
    source_contract = source_report["phase68_residual_event"]["contract"]
    candidate_contract = candidate_report["phase68_residual_event"]["contract"]
    expected = prereg["source"]
    technical_checks = {
        "prereg_sidecar": sha256(args.prereg) == "fedc6bc83a270444c0c2004313a2bae0457266e8fff281d4adfa42543b453e12",
        "base_checkpoint": train["base_checkpoint_sha256"] == expected["base_checkpoint_sha256"],
        "runner_hash": all(
            report["phase68_residual_event"]["runner_sha256"] == expected["runner_sha256"]
            for report in (source_report, candidate_report)
        ),
        "module_hash": all(
            report["phase68_residual_event"]["module_sha256"] == expected["residual_module_sha256"]
            for report in (source_report, candidate_report)
        ),
        "interface_hash": all(
            report["phase68_residual_event"]["interface_sha256"] == expected["ppo_interface_sha256"]
            for report in (source_report, candidate_report)
        ),
        "train_decision": train["decision"] == "UPDATE_TECHNICAL_PASS_PENDING_EVENT_EVAL",
        "all_train_gates": all(train["technical_checks"].values()),
        "optimizer_step_exact_one": train["optimizer_steps"] == 1,
        "checkpoint_count_exact_two": bool(
            train["source_residual_checkpoint_sha256"]
            and train["candidate_residual_checkpoint_sha256"]
        ),
        "resource_exit_codes": all(item["exit_code"] == 0 for item in resources.values()),
        "resource_launch_count": len(resources) == prereg["resource_limits"]["maximum_isaac_launches"],
        "resource_disk_delta": sum(item["disk_used_delta_bytes"] for item in resources.values())
        <= prereg["resource_limits"]["maximum_total_disk_delta_bytes"],
        "resource_gpu_peak": max(item["gpu"]["memory_used_peak_mib"] for item in resources.values())
        <= prereg["resource_limits"]["gpu_peak_memory_mib_max"],
        "all_finite": finite_tree(train) and finite_tree(source_report) and finite_tree(candidate_report),
    }
    source_gate = prereg["complete_event_evaluation"]["source_validity"]
    source_checks = {
        "finite": source_report["finite"] is True and source_contract["finite"] is True,
        "role": source_contract["role"] == "source",
        "survival": source["survival_s_mean"] >= source_gate["survival_s_mean_min"] - 1.0e-6,
        "termination": source["termination_rate"] <= source_gate["termination_rate_max"],
        "zero_residual": source_contract["residual_output_max_abs_rad"] == 0.0,
        "zero_processed_delta": source_contract["processed_target_delta_max_abs_rad"] == 0.0,
    }
    absolute = prereg["complete_event_evaluation"]["candidate_absolute"]
    candidate_absolute_checks = {
        "finite": candidate_report["finite"] is True and candidate_contract["finite"] is True,
        "role": candidate_contract["role"] == "candidate",
        "survival": candidate["survival_s_mean"] >= absolute["survival_s_mean_min"] - 1.0e-6,
        "termination": candidate["termination_rate"] <= absolute["termination_rate_max"],
        "terminal_speed_mean": candidate["terminal_base_speed_mps"]["mean"] <= absolute["terminal_base_speed_mean_mps_max"],
        "terminal_speed_p95": candidate["terminal_base_speed_mps"]["p95"] <= absolute["terminal_base_speed_p95_mps_max"],
        "terminal_double_support": candidate["terminal_double_support"]["mean"] >= absolute["terminal_double_support_mean_min"],
        "standing_zero": candidate_contract["standing_shadow_output_max_abs_rad"] <= absolute["standing_shadow_output_max_abs_rad"],
        "invalid_zero": candidate_contract["invalid_contact_shadow_output_max_abs_rad"] <= absolute["invalid_contact_shadow_output_max_abs_rad"],
        "non_knee_zero": candidate_contract["non_knee_output_max_abs_rad"] <= absolute["non_knee_output_max_abs_rad"],
        "residual_bound": candidate_contract["residual_output_max_abs_rad"] <= absolute["residual_output_max_abs_rad"] + 1.0e-8,
        "residual_nonempty": candidate_contract["residual_output_rms_rad"] >= absolute["residual_output_rms_rad_min"],
        "effective_ratio": candidate_contract["effective_requested_abs_ratio"] >= absolute["effective_requested_abs_ratio_min"],
    }
    maximum = prereg["complete_event_evaluation"]["candidate_relative_safety_max_regression"]
    delta = {
        "velocity_tracking_rmse_mps": candidate["velocity_tracking_rmse"] - source["velocity_tracking_rmse"],
        "lateral_abs_mean_m": candidate["lateral_abs"] - source["lateral_abs"],
        "yaw_abs_mean_rad": candidate["yaw_abs"] - source["yaw_abs"],
        "com_support_outside_mean_m": candidate["com_support_outside_m"]["mean"] - source["com_support_outside_m"]["mean"],
        "stance_slip_p95_mps": candidate["stance_slip_mps"]["p95"] - source["stance_slip_mps"]["p95"],
        "flight_fraction": candidate["flight_fraction"] - source["flight_fraction"],
        "root_height_min_mean_m": candidate["root_height_min_mean"] - source["root_height_min_mean"],
        "root_tilt_max_mean_rad": candidate["root_tilt_max_mean"] - source["root_tilt_max_mean"],
        "action_delta_abs_mean": candidate["action_delta_abs"] - source["action_delta_abs"],
        "swing_clearance_p05_m": candidate["swing_sole_clearance_m"]["p05"] - source["swing_sole_clearance_m"]["p05"],
        "knee_left_excursion_rad": candidate["knee_joint_range"]["left"]["excursion_rad_mean"] - source["knee_joint_range"]["left"]["excursion_rad_mean"],
        "knee_right_excursion_rad": candidate["knee_joint_range"]["right"]["excursion_rad_mean"] - source["knee_joint_range"]["right"]["excursion_rad_mean"],
        "knee_excursion_asymmetry_rad": candidate["knee_excursion_left_right_abs_diff_rad_mean"] - source["knee_excursion_left_right_abs_diff_rad_mean"],
    }
    safety_checks = {
        "velocity": delta["velocity_tracking_rmse_mps"] <= maximum["velocity_tracking_rmse_mps"],
        "lateral": delta["lateral_abs_mean_m"] <= maximum["lateral_abs_mean_m"],
        "yaw": delta["yaw_abs_mean_rad"] <= maximum["yaw_abs_mean_rad"],
        "support": delta["com_support_outside_mean_m"] <= maximum["com_support_outside_mean_m"],
        "slip": delta["stance_slip_p95_mps"] <= maximum["stance_slip_p95_mps"],
        "flight": delta["flight_fraction"] <= maximum["flight_fraction"],
        "root_height": delta["root_height_min_mean_m"] >= -maximum["root_height_min_mean_m"],
        "root_tilt": delta["root_tilt_max_mean_rad"] <= maximum["root_tilt_max_mean_rad"],
        "action_delta": delta["action_delta_abs_mean"] <= maximum["action_delta_abs_mean"],
        "clearance": delta["swing_clearance_p05_m"] >= -maximum["swing_clearance_p05_m"],
        "knee_left": abs(delta["knee_left_excursion_rad"]) <= maximum["knee_excursion_each_rad"],
        "knee_right": abs(delta["knee_right_excursion_rad"]) <= maximum["knee_excursion_each_rad"],
        "knee_asymmetry": delta["knee_excursion_asymmetry_rad"] <= maximum["knee_excursion_asymmetry_rad"],
    }
    signal_gate = prereg["complete_event_evaluation"]["scientific_signal"]
    pitch_delta = {
        key: candidate["signed_pitch_rad"][key] - source["signed_pitch_rad"][key]
        for key in ("mean", "p05", "p50", "p95")
    }
    terminal_delta = {
        "base_speed_mean_mps": candidate["terminal_base_speed_mps"]["mean"] - source["terminal_base_speed_mps"]["mean"],
        "double_support_mean": candidate["terminal_double_support"]["mean"] - source["terminal_double_support"]["mean"],
        "root_height_mean_m": candidate["terminal_root_height_m"]["mean"] - source["terminal_root_height_m"]["mean"],
        "root_tilt_mean_rad": candidate["terminal_root_tilt_rad"]["mean"] - source["terminal_root_tilt_rad"]["mean"],
    }
    scientific_checks = {
        "pitch_mean": pitch_delta["mean"] >= signal_gate["moving_signed_pitch_mean_delta_rad_min"],
        "pitch_p05": pitch_delta["p05"] >= signal_gate["moving_signed_pitch_p05_delta_rad_min"],
        "terminal_speed": terminal_delta["base_speed_mean_mps"] <= 0.005,
        "terminal_double_support": terminal_delta["double_support_mean"] >= 0.0,
        "terminal_root_height": terminal_delta["root_height_mean_m"] >= -0.005,
        "terminal_root_tilt": terminal_delta["root_tilt_mean_rad"] <= 0.01,
    }
    if not all(technical_checks.values()):
        decision = "FAIL_INVALID_UPDATE_STOP"
    elif not all(source_checks.values()):
        decision = "FAIL_SOURCE_EVENT_INVALID_STOP"
    elif not all(candidate_absolute_checks.values()) or not all(safety_checks.values()):
        decision = "FAIL_CANDIDATE_SAFETY_STOP"
    elif not all(scientific_checks.values()):
        decision = "FAIL_NO_LOCAL_SIGNAL_STOP"
    else:
        decision = "PASS_ONE_UPDATE_SANITY_STOP"
    result = {
        "schema": "x2_phase_conditioned_residual_phase68_result_v1",
        "decision": decision,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "prereg": args.prereg,
                "train": args.train,
                "train_resource": args.train_resource,
                "source": args.source,
                "source_resource": args.source_resource,
                "candidate": args.candidate,
                "candidate_resource": args.candidate_resource,
            }.items()
        },
        "technical_checks": technical_checks,
        "source_checks": source_checks,
        "candidate_absolute_checks": candidate_absolute_checks,
        "candidate_relative_safety_checks": safety_checks,
        "scientific_checks": scientific_checks,
        "candidate_minus_source": {"safety": delta, "pitch": pitch_delta, "terminal": terminal_delta},
        "resources": {
            "launch_count": len(resources),
            "total_disk_delta_bytes": sum(item["disk_used_delta_bytes"] for item in resources.values()),
            "peak_gpu_memory_mib": max(item["gpu"]["memory_used_peak_mib"] for item in resources.values()),
            "elapsed_s": {name: item["elapsed_s"] for name, item in resources.items()},
        },
        "diagnostic_note": "The final_target_clip_sample_count field is a subtraction-tolerance diagnostic, not a clipping gate; the preregistered effective/requested absolute ratio is authoritative.",
        "five_update_unlocked": decision == "PASS_ONE_UPDATE_SANITY_STOP",
        "long_training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
        "boundary": "Even a pass would only authorize a separate bounded pilot preregistration. This result does not authorize long training, export, deployment, or Task2 completion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(
        "\n".join(
            [
                "# X2 native posture Phase68 — one-step physical residual PPO",
                "",
                f"Decision: **{decision}**.",
                "",
                f"Technical checks: `{technical_checks}`.",
                "",
                f"Source checks: `{source_checks}`.",
                "",
                f"Candidate safety checks: `{candidate_absolute_checks | safety_checks}`.",
                "",
                f"Signed-pitch candidate minus source: `{pitch_delta}`.",
                "",
                "The update was technically valid and safety-bounded, but it is rejected unless every preregistered efficacy check passes. No long training, export, or deployment was unlocked.",
            ]
        ) + "\n"
    )
    print(json.dumps({"decision": decision, "pitch_delta": pitch_delta}, indent=2))


if __name__ == "__main__":
    main()
