#!/usr/bin/env python3
"""Apply the repaired Phase67b suffix and live-zero gates fail-closed."""

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
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    resource = json.loads(args.resource.read_text(encoding="utf-8"))
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if report.get("phase") != 67 or report.get("mode") != "screen":
        raise RuntimeError("input is not the Phase67-family live-zero screen")
    if report.get("posture_variant") != "phase_conditioned_residual_live_zero":
        raise RuntimeError("Phase67b posture variant changed")
    if report.get("checkpoint_sha256") != prereg["source"]["checkpoint_sha256"]:
        raise RuntimeError("Phase67b checkpoint changed")
    if resource.get("exit_code") != 0:
        raise RuntimeError("Phase67b resource ledger reports failure")
    if resource["gpu"]["memory_used_peak_mib"] > prereg["resource_limits"]["gpu_peak_memory_mib_max"]:
        raise RuntimeError("Phase67b exceeded GPU memory limit")
    live = report["phase_conditioned_residual_live_zero"]
    contract = live["contract"]
    manifest = live["trainable_manifest"]
    hashes_equal = all(
        row["source_processed_target"] == row["intervention_processed_target"]
        for row in live["sampled_step_hashes"]
    )
    checks = {
        "report_finite": report.get("finite") is True,
        "contract_finite": contract["finite"] is True,
        "zero_residual": contract["residual_output_max_abs_rad"] <= prereg["live_screen"]["residual_output_max_abs_rad"],
        "zero_standing": contract["standing_shadow_output_max_abs_rad"] <= prereg["live_screen"]["standing_shadow_output_max_abs_rad"],
        "zero_invalid_contact": contract["invalid_contact_shadow_output_max_abs_rad"] <= prereg["live_screen"]["invalid_contact_shadow_output_max_abs_rad"],
        "zero_processed_delta": contract["processed_target_delta_max_abs_rad"] <= prereg["live_screen"]["processed_target_delta_max_abs_rad"],
        "zero_non_knee": contract["non_knee_output_max_abs_rad"] <= prereg["live_screen"]["non_knee_output_max_abs_rad"],
        "sampled_target_hashes": hashes_equal,
        "trainable_parameter_count": manifest["trainable_parameters"] == prereg["repaired_contract"]["expected_trainable_parameters"],
        "optimizer_steps": live["optimizer_steps"] == 0,
        "checkpoint_count": live["checkpoint_count"] == 0,
        "survival": report["groups"]["all"]["survival_s_mean"] >= prereg["live_screen"]["survival_s_min"] - 1.0e-6,
        "termination": report["groups"]["all"]["termination_rate"] <= prereg["live_screen"]["termination_rate_max"],
    }
    passed = all(checks.values())
    decision = "PASS_SUFFIX_REPAIR_LIVE_ZERO_ONLY" if passed else "FAIL_SUFFIX_REPAIR_BLOCK_RESIDUAL"
    result = {
        "schema": "x2_phase_conditioned_residual_phase67b_result_v1",
        "decision": decision,
        "supersedes_contract_result_sha256": prereg["source"]["phase67_result_sha256"],
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "resource": {"path": str(args.resource), "sha256": sha256(args.resource), "report": resource},
        "preregistration": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
        "module_sha256": live["module_sha256"],
        "trainable_manifest": manifest,
        "checks": checks,
        "phase68_preregistration_unlocked": passed,
        "optimizer_update_unlocked": False,
        "long_training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
        "boundary": "Pass repairs only the gait-suffix safety contract and preserves zero behavior; no optimization or checkpoint was produced.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(
        "\n".join(
            [
                "# X2 native posture Phase67b — fail-closed suffix repair",
                "",
                f"Decision: **{decision}**.",
                "",
                f"Checks: `{checks}`.",
                "",
                "The contact-suffix safety contract was repaired and live-zero was repeated. No optimizer step, checkpoint, ONNX, or deployment backend was produced.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
