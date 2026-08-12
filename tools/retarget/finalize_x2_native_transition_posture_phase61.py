#!/usr/bin/env python3
"""Fail-closed paired local gate for the Phase61 complete-event update."""

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
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / "reports/retarget"
    paths = {
        "prereg": report_dir / "x2_native_transition_posture_phase61_prereg.json",
        "source_eval": report_dir / "x2_native_transition_posture_phase61_source_eval.json",
        "train": report_dir / "x2_native_transition_posture_phase61_train.json",
        "candidate_eval": report_dir / "x2_native_transition_posture_phase61_candidate_eval.json",
        "source_resource": report_dir / "x2_native_transition_posture_phase61_source_resource.json",
        "train_resource": report_dir / "x2_native_transition_posture_phase61_train_resource.json",
        "candidate_resource": report_dir / "x2_native_transition_posture_phase61_candidate_resource.json",
    }
    data = {name: load(path) for name, path in paths.items()}
    prereg = data["prereg"]
    source_report = data["source_eval"]
    candidate_report = data["candidate_eval"]
    train = data["train"]
    source = source_report["groups"]["all"]
    candidate = candidate_report["groups"]["all"]
    gate = prereg["local_gates"]
    retention = train["fixed_source_retention"]
    train_gpu = data["train_resource"].get("gpu")
    tolerance = 1.0e-6

    for label, report in (("source", source_report), ("candidate", candidate_report)):
        if report.get("phase") != 61 or report.get("eval_steps") != 512:
            raise RuntimeError(f"{label} evaluation violates the Phase61 contract")
        if report.get("transition_event", {}).get("event_end_s") != 7.2:
            raise RuntimeError(f"{label} evaluation lacks the frozen transition event")

    checks = {
        "source_finite": source_report.get("finite") is True,
        "candidate_finite": candidate_report.get("finite") is True,
        "train_finite": train.get("finite") is True,
        "optimizer_steps_exact": train.get("optimizer_steps") == gate["optimizer_steps_exact"],
        "checkpoint_count_exact_two": train.get("checkpoint_count") == 2,
        "dense_unchanged": train.get("dense_hash_before") == train.get("dense_hash_after"),
        "std_unchanged": train.get("std_hash_before") == train.get("std_hash_after"),
        "lora_only": bool(train.get("trainable_names"))
        and all("lora_" in name for name in train["trainable_names"]),
        "kl_mean": retention["kl_mean"] <= gate["fixed_batch_kl_mean_max"],
        "kl_peak": retention["kl_max"] <= gate["fixed_batch_kl_peak_max"],
        "action_drift": retention["action_max_abs"] <= gate["fixed_batch_action_drift_max"],
        "source_survival": source["survival_s_mean"] + tolerance
        >= gate["source_and_candidate_survival_s_min"],
        "candidate_survival": candidate["survival_s_mean"] + tolerance
        >= gate["source_and_candidate_survival_s_min"],
        "source_termination": source["termination_rate"]
        <= gate["source_and_candidate_termination_rate_max"],
        "candidate_termination": candidate["termination_rate"]
        <= gate["source_and_candidate_termination_rate_max"],
        "moving_pitch_mean": candidate["signed_pitch_rad"]["mean"]
        >= source["signed_pitch_rad"]["mean"]
        + gate["moving_signed_pitch_mean_improvement_min_rad"],
        "moving_pitch_p05": candidate["signed_pitch_rad"]["p05"]
        >= source["signed_pitch_rad"]["p05"],
        "moving_support": candidate["com_support_outside_m"]["mean"]
        <= source["com_support_outside_m"]["mean"],
        "velocity": candidate["velocity_tracking_rmse"]
        <= source["velocity_tracking_rmse"] + gate["velocity_rmse_regression_max_mps"],
        "terminal_speed_mean": candidate["terminal_base_speed_mps"]["mean"]
        <= source["terminal_base_speed_mps"]["mean"]
        + gate["terminal_speed_mean_regression_max_mps"],
        "terminal_speed_p95": candidate["terminal_base_speed_mps"]["p95"]
        <= source["terminal_base_speed_mps"]["p95"]
        + gate["terminal_speed_p95_regression_max_mps"],
        "terminal_double_support": candidate["terminal_double_support"]["mean"]
        >= source["terminal_double_support"]["mean"]
        - gate["terminal_double_support_mean_regression_max"],
        "terminal_root_height": candidate["terminal_root_height_m"]["p05"]
        >= source["terminal_root_height_m"]["p05"]
        - gate["terminal_root_height_p05_regression_max_m"],
        "terminal_root_tilt": candidate["terminal_root_tilt_rad"]["p95"]
        <= source["terminal_root_tilt_rad"]["p95"]
        + gate["terminal_root_tilt_p95_regression_max_rad"],
        "stance_slip": candidate["stance_slip_mps"]["p95"]
        <= source["stance_slip_mps"]["p95"]
        + gate["stance_slip_p95_regression_max_mps"],
        "action_delta": candidate["action_delta_abs"]
        <= source["action_delta_abs"] + gate["action_delta_abs_regression_max"],
        "knee_excursion": abs(
            candidate["knee_excursion_rad_mean"] - source["knee_excursion_rad_mean"]
        ) <= gate["knee_excursion_abs_change_max_rad"],
        "swing_clearance": candidate["swing_sole_clearance_m"]["p05"]
        >= source["swing_sole_clearance_m"]["p05"]
        - gate["swing_clearance_p05_regression_max_m"],
        "event_static_audit": train["transition_event_coverage"]["static_audit_pass"] is True,
        "event_terminal_coverage": train["transition_event_coverage"]["terminal_zero_steps"]
        >= gate["train_terminal_zero_steps_min"],
        "gpu_sampled": train_gpu is not None
        and data["train_resource"]["gpu_sample_count"] > 0,
        "gpu_peak": train_gpu is not None
        and train_gpu["memory_used_peak_mib"] <= gate["gpu_peak_memory_mib_max"],
        "run_exit_codes": all(
            data[name].get("exit_code") == 0
            for name in ("source_resource", "train_resource", "candidate_resource")
        ),
    }
    passed = all(checks.values())
    result = {
        "schema": "x2_native_transition_posture_phase61_result_v1",
        "prereg": {"path": str(paths["prereg"]), "sha256": sha256(paths["prereg"])},
        "inputs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
            if name != "prereg"
        },
        "checks": checks,
        "metrics": {
            "source": source,
            "candidate": candidate,
            "deltas": {
                "moving_pitch_mean_rad": candidate["signed_pitch_rad"]["mean"]
                - source["signed_pitch_rad"]["mean"],
                "moving_pitch_p05_rad": candidate["signed_pitch_rad"]["p05"]
                - source["signed_pitch_rad"]["p05"],
                "support_mean_m": candidate["com_support_outside_m"]["mean"]
                - source["com_support_outside_m"]["mean"],
                "terminal_speed_mean_mps": candidate["terminal_base_speed_mps"]["mean"]
                - source["terminal_base_speed_mps"]["mean"],
            },
        },
        "training": train,
        "resources": {
            name.removesuffix("_resource"): data[name]
            for name in ("source_resource", "train_resource", "candidate_resource")
        },
        "local_pass": passed,
        "decision": "PASS_LOCAL_JOINT_TRANSITION_STOP" if passed else "FAIL_LOCAL_JOINT_TRANSITION_STOP",
        "export_unlocked": passed,
        "official_gate_unlocked": passed,
        "long_training_unlocked": False,
        "task2_complete": False,
        "boundary": "A local pass only unlocks exact LoRA merge and official physical evaluation with the fixed stationary backend.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    failed = [name for name, value in checks.items() if not value]
    lines = [
        "# X2 native posture Phase61",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        f"- Moving signed pitch mean delta: `{result['metrics']['deltas']['moving_pitch_mean_rad']:+.6f} rad`.",
        f"- Moving signed pitch p05 delta: `{result['metrics']['deltas']['moving_pitch_p05_rad']:+.6f} rad`.",
        f"- Actual support outside-mean delta: `{result['metrics']['deltas']['support_mean_m']:+.6f} m`.",
        f"- Terminal speed mean delta: `{result['metrics']['deltas']['terminal_speed_mean_mps']:+.6f} m/s`.",
        f"- Training peak GPU memory: `{train_gpu['memory_used_peak_mib'] if train_gpu else None} MiB`.",
        f"- Failed local gates: `{failed}`.",
        "",
        "This single complete-event update is not long-training evidence. The default official stop phase remains owned by the frozen stationary actor; the candidate can only alter locomotion and the state delivered at handoff.",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
