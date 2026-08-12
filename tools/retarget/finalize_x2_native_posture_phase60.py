#!/usr/bin/env python3
"""Fail-closed multi-seed selection for the Phase60 A/B/C posture pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate(paths: list[Path]) -> dict:
    reports = [load(path) for path in paths]
    if [report["seed"] for report in reports] != [40, 41, 42]:
        raise RuntimeError("Phase60 evaluation seeds must be exactly 40, 41, 42")
    if not all(report.get("finite") and report.get("phase") == 60 for report in reports):
        raise RuntimeError("Phase60 evaluation report is non-finite or wrong phase")
    rows = [report["groups"]["all"] for report in reports]
    aggregate_result = {
        "seeds": [40, 41, 42],
        "reports": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        ],
        "survival_s_min": min(float(row["survival_s_mean"]) for row in rows),
        "termination_rate_max": max(float(row["termination_rate"]) for row in rows),
    }
    for key in (
        "velocity_tracking_rmse", "yaw_tracking_rmse", "lateral_abs", "yaw_abs",
        "action_abs", "action_delta_abs", "flight_fraction", "single_support_fraction",
        "double_support_fraction", "root_height_min_mean", "root_tilt_max_mean",
        "knee_excursion_rad_mean",
    ):
        aggregate_result[key] = mean(rows, key)
    for key in (
        "signed_pitch_rad", "com_support_outside_m", "stance_slip_mps",
        "swing_sole_clearance_m",
    ):
        aggregate_result[key] = {
            statistic: mean([row[key] for row in rows], statistic)
            for statistic in ("mean", "p05", "p50", "p95", "max")
        }
    return aggregate_result


def candidate_checks(prereg: dict, baseline: dict, candidate: dict, training: dict, group: str) -> dict:
    gate = prereg["local_trend_gates"]
    retention = training["fixed_source_retention"]
    checks = {
        "train_finite": training.get("finite") is True,
        "optimizer_steps_exact": training.get("optimizer_steps") == gate["optimizer_steps_exact"],
        "checkpoint_count_exact_two": training.get("checkpoint_count") == 2,
        "dense_unchanged": training.get("dense_hash_before") == training.get("dense_hash_after"),
        "std_unchanged": training.get("std_hash_before") == training.get("std_hash_after"),
        "lora_only": bool(training.get("trainable_names"))
        and all("lora_" in name for name in training["trainable_names"]),
        "kl_mean": retention["kl_mean"] <= gate["fixed_batch_kl_mean_max"],
        "kl_peak": retention["kl_max"] <= gate["fixed_batch_kl_peak_max"],
        "action_drift": retention["action_max_abs"] <= gate["fixed_batch_action_drift_max"],
        "survival": candidate["survival_s_min"] >= gate["survival_s_min"],
        "termination": candidate["termination_rate_max"] <= gate["termination_rate_max"],
        "velocity": candidate["velocity_tracking_rmse"]
        <= baseline["velocity_tracking_rmse"] + gate["velocity_rmse_regression_max"],
        "signed_pitch_mean": candidate["signed_pitch_rad"]["mean"]
        >= baseline["signed_pitch_rad"]["mean"] + gate["signed_pitch_mean_improvement_min_rad"],
        "signed_pitch_p05": candidate["signed_pitch_rad"]["p05"]
        >= baseline["signed_pitch_rad"]["p05"],
        "flight": candidate["flight_fraction"]
        <= baseline["flight_fraction"] + gate["flight_fraction_regression_max"],
        "stance_slip_p95": candidate["stance_slip_mps"]["p95"]
        <= baseline["stance_slip_mps"]["p95"] + gate["stance_slip_p95_regression_max_mps"],
        "action_delta": candidate["action_delta_abs"]
        <= baseline["action_delta_abs"] + gate["action_delta_abs_regression_max"],
        "root_height": candidate["root_height_min_mean"]
        >= baseline["root_height_min_mean"] - gate["root_height_regression_max_m"],
        "root_tilt": candidate["root_tilt_max_mean"]
        <= baseline["root_tilt_max_mean"] + gate["root_tilt_regression_max_rad"],
        "knee_excursion": abs(candidate["knee_excursion_rad_mean"] - baseline["knee_excursion_rad_mean"])
        <= gate["knee_excursion_abs_change_max_rad"],
        "swing_clearance_p05": candidate["swing_sole_clearance_m"]["p05"]
        >= baseline["swing_sole_clearance_m"]["p05"] - gate["swing_clearance_p05_regression_max_m"],
    }
    checks["support"] = (
        candidate["com_support_outside_m"]["mean"]
        <= baseline["com_support_outside_m"]["mean"]
        if group == "C" else True
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports/retarget"
    prereg_path = report_dir / "x2_native_posture_phase60_prereg.json"
    prereg = load(prereg_path)
    eval_paths = {
        "A": [
            report_dir / f"x2_native_posture_phase60_A_seed{seed}_{'smoke' if seed == 42 else 'eval'}.json"
            for seed in (40, 41, 42)
        ],
        "B": [report_dir / f"x2_native_posture_phase60_B_seed{seed}_eval.json" for seed in (40, 41, 42)],
        "C": [report_dir / f"x2_native_posture_phase60_C_seed{seed}_eval.json" for seed in (40, 41, 42)],
    }
    evaluations = {group: aggregate(paths) for group, paths in eval_paths.items()}
    training_paths = {
        group: report_dir / f"x2_native_posture_phase60_{group}_train.json"
        for group in ("B", "C")
    }
    training = {group: load(path) for group, path in training_paths.items()}
    checks = {
        group: candidate_checks(prereg, evaluations["A"], evaluations[group], training[group], group)
        for group in ("B", "C")
    }
    passed = {group: all(group_checks.values()) for group, group_checks in checks.items()}
    scores = {
        group: (
            evaluations[group]["signed_pitch_rad"]["mean"]
            - evaluations["A"]["signed_pitch_rad"]["mean"]
            + 0.5 * (
                evaluations["A"]["com_support_outside_m"]["mean"]
                - evaluations[group]["com_support_outside_m"]["mean"]
            )
        )
        for group in ("B", "C")
    }
    eligible = [group for group in ("B", "C") if passed[group]]
    selected = max(eligible, key=lambda group: (scores[group], group == "B")) if eligible else None
    result = {
        "schema": "x2_native_posture_phase60_result_v1",
        "prereg": {"path": str(prereg_path), "sha256": sha256(prereg_path)},
        "evaluations": evaluations,
        "training": {
            group: {
                "path": str(training_paths[group]),
                "sha256": sha256(training_paths[group]),
                "report": training[group],
            }
            for group in ("B", "C")
        },
        "checks": checks,
        "passed": passed,
        "selection_scores": scores,
        "selected_local_candidate": selected,
        "evaluation_randomness": {
            "observation_corruption": False,
            "reset_pose": "fixed",
            "command": "fixed [0.35, 0, 0]",
            "seed_outputs_numerically_identical": all(
                all(
                    report["groups"]["all"] == load(eval_paths[group][0])["groups"]["all"]
                    for report in [load(path) for path in eval_paths[group][1:]]
                )
                for group in ("A", "B", "C")
            ),
            "interpretation": "Seeds are reproducibility checks, not independent stochastic trials.",
        },
        "decision": (
            f"PASS_LOCAL_TREND_{selected}_STOP" if selected else "FAIL_BOTH_LOCAL_TREND_STOP"
        ),
        "training_unlocked": False,
        "boundary": "A/B/C one-update local trend only; no long training, official MuJoCo promotion, or Task2 completion claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
