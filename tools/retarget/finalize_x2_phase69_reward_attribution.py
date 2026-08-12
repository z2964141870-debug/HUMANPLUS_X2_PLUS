#!/usr/bin/env python3
"""Finalize the zero-optimizer Phase69 reward-gradient attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_sign(summary: dict[str, object]) -> int:
    """Return +1/-1 only when the environment bootstrap is sign-stable."""

    loo = float(summary["leave_one_env_out_same_sign_fraction"])
    same = float(summary["bootstrap_same_sign_fraction"])
    if (
        float(summary["bootstrap_p025"]) > 0.0
        and float(summary["point"]) > 0.0
        and loo >= 0.90
        and same >= 0.95
    ):
        return 1
    if (
        float(summary["bootstrap_p975"]) < 0.0
        and float(summary["point"]) < 0.0
        and loo >= 0.90
        and same >= 0.95
    ):
        return -1
    return 0


def classify_alignment(
    pitch_reward_alignment: dict[str, object],
    total_alignment: dict[str, object],
) -> str:
    """Apply the preregistered diagnostic-only attribution decision tree."""

    pitch_sign = stable_sign(pitch_reward_alignment)
    total_sign = stable_sign(total_alignment)
    if pitch_sign < 0:
        return "FAIL_REWARD_CREDIT_DIRECTION_STOP"
    if pitch_sign > 0 and total_sign < 0:
        return "PASS_REWARD_CONFLICT_DIAG_ONLY"
    if total_sign > 0:
        return "FAIL_STOCHASTIC_TO_MEAN_OR_ADAM_MISMATCH_STOP"
    return "INCONCLUSIVE_VARIANCE_STOP"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.markdown.exists():
        raise RuntimeError("refusing to overwrite immutable Phase69 result")
    prereg = json.loads(args.prereg.read_text())
    screen = json.loads(args.screen.read_text())
    resource = json.loads(args.resource.read_text())
    prereg_sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    expected_sidecar = f"{sha256(args.prereg)}  {args.prereg.name}\n"
    technical = screen.get("technical_checks", {})
    resource_checks = {
        "exit_code": resource.get("exit_code") == 0,
        "disk_delta": int(resource.get("disk_used_delta_bytes", 2**63))
        <= int(prereg["resource_limits"]["disk_delta_bytes_max"]),
        "gpu_peak": int(resource.get("gpu", {}).get("memory_used_peak_mib", 2**31))
        <= int(prereg["resource_limits"]["gpu_peak_memory_mib_max"]),
    }
    validity_checks = {
        "prereg_sidecar": prereg_sidecar.is_file()
        and prereg_sidecar.read_text() == expected_sidecar,
        "screen_schema": screen.get("schema")
        == "x2_phase69_reward_attribution_screen_v1",
        "screen_pending_decision": screen.get("decision")
        == "ATTRIBUTION_VALID_PENDING_FINALIZATION",
        "all_technical_checks": bool(technical) and all(technical.values()),
        "optimizer_steps_zero": screen.get("optimizer_steps") == 0,
        "checkpoint_count_zero": screen.get("checkpoint_count") == 0,
        "resource_checks": all(resource_checks.values()),
    }
    total_alignment = screen.get("alignment", {}).get(
        "total_vs_signed_pitch_metric", {}
    )
    terminal_alignment = screen.get("alignment", {}).get(
        "terminal_bootstrap_total_vs_signed_pitch_metric", {}
    )
    cutoff = screen.get("cutoff_diagnostic", {}).get(
        "zero_vs_terminal_bootstrap_direction", {}
    )
    influence = screen.get("environment_influence", {})
    attribution_quality_checks = {
        "zero_terminal_direction_cosine": float(cutoff.get("cosine", -1.0))
        >= 0.90,
        "zero_terminal_pitch_projection_sign_consistent": (
            stable_sign(total_alignment) != 0
            and stable_sign(total_alignment) == stable_sign(terminal_alignment)
        ),
        "kish_effective_sample_size": float(
            influence.get("kish_effective_sample_size", 0.0)
        )
        >= 16.0,
        "maximum_env_absolute_contribution": float(
            influence.get("maximum_absolute_env_contribution_fraction", 1.0)
        )
        <= 0.20,
    }
    if not all(validity_checks.values()):
        decision = "FAIL_ATTRIBUTION_INVALID_STOP"
    elif not all(attribution_quality_checks.values()):
        decision = "OBJECTIVE_ESTIMATOR_OR_CUTOFF_MISALIGNMENT_STOP"
    else:
        decision = classify_alignment(
            screen["alignment"]["pitch_reward_vs_signed_pitch_metric"],
            screen["alignment"]["total_vs_signed_pitch_metric"],
        )
    report = {
        "schema": "x2_phase69_reward_attribution_result_v1",
        "decision": decision,
        "inputs": {
            "prereg": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
            "screen": {"path": str(args.screen), "sha256": sha256(args.screen)},
            "resource": {"path": str(args.resource), "sha256": sha256(args.resource)},
            "rollout_bundle": {
                "path": screen.get("rollout_bundle"),
                "sha256": screen.get("rollout_bundle_sha256"),
                "artifact_role": "raw rollout evidence; not a model checkpoint",
            },
        },
        "validity_checks": validity_checks,
        "resource_checks": resource_checks,
        "attribution_quality_checks": attribution_quality_checks,
        "alignment": screen.get("alignment"),
        "additive_components": screen.get("additive_components"),
        "phase_components": screen.get("phase_components"),
        "gradient_replay": screen.get("gradient_replay"),
        "cutoff_diagnostic": screen.get("cutoff_diagnostic"),
        "environment_influence": screen.get("environment_influence"),
        "interpretation_boundary": (
            "This is an environment-level finite-sample policy-gradient diagnostic, "
            "not independent statistical significance and not a training promotion."
        ),
        "phase70_preregistration_only_unlocked": decision
        in {
            "PASS_REWARD_CONFLICT_DIAG_ONLY",
            "FAIL_STOCHASTIC_TO_MEAN_OR_ADAM_MISMATCH_STOP",
        },
        "optimizer_unlocked": False,
        "five_update_unlocked": False,
        "long_training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    result_sha = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{result_sha}  {args.output.name}\n"
    )
    pitch = report["alignment"]["pitch_reward_vs_signed_pitch_metric"]
    total = report["alignment"]["total_vs_signed_pitch_metric"]
    args.markdown.write_text(
        "# X2 Phase69 reward-gradient attribution\n\n"
        f"- Decision: `{decision}`\n"
        "- Optimizer steps/checkpoints: `0 / 0`\n"
        f"- Pitch-reward → signed-pitch projection: `{pitch['point']:.6f}` "
        f"(95% env-bootstrap `{pitch['bootstrap_p025']:.6f}` to "
        f"`{pitch['bootstrap_p975']:.6f}`)\n"
        f"- Total-gradient → signed-pitch projection: `{total['point']:.6f}` "
        f"(95% env-bootstrap `{total['bootstrap_p025']:.6f}` to "
        f"`{total['bootstrap_p975']:.6f}`)\n"
        "- This phase does not unlock optimization, long training, export, or deployment.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
