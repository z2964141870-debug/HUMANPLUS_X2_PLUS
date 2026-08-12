#!/usr/bin/env python3
"""Finalize the zero-optimizer Phase70 long-lookahead qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_sign(summary: dict[str, object]) -> int:
    point = float(summary.get("point", 0.0))
    lower = float(summary.get("bootstrap_p025", 0.0))
    upper = float(summary.get("bootstrap_p975", 0.0))
    bootstrap = float(summary.get("bootstrap_same_sign_fraction", 0.0))
    loo = float(summary.get("leave_one_env_out_same_sign_fraction", 0.0))
    if point > 0.0 and lower > 0.0 and bootstrap >= 0.95 and loo >= 0.90:
        return 1
    if point < 0.0 and upper < 0.0 and bootstrap >= 0.95 and loo >= 0.90:
        return -1
    return 0


def classify_scientific_outcome(
    pitch_total: dict[str, object], support_total: dict[str, object]
) -> str:
    pitch_sign = stable_sign(pitch_total)
    support_sign = stable_sign(support_total)
    if pitch_sign < 0 and support_sign < 0:
        return "PASS_REWARD_CONFLICT_DIAG_ONLY"
    if pitch_sign > 0 and support_sign > 0:
        return "PASS_LONG_LOOKAHEAD_POSITIVE_DIAG_ONLY"
    return "INCONCLUSIVE_TOTAL_DIRECTION_STOP"


def reward_conflict_confirmed(alignment: dict[str, object]) -> bool:
    """Require reward-only and locomotion gradients to oppose both metrics."""

    required_negative = (
        "primary_total_vs_positive_pitch",
        "primary_total_vs_lower_support_outside",
        "reward_only_total_vs_positive_pitch",
        "reward_only_total_vs_lower_support_outside",
        "locomotion_reward_vs_positive_pitch",
        "locomotion_reward_vs_lower_support_outside",
    )
    return all(stable_sign(alignment.get(name, {})) < 0 for name in required_negative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result_sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or result_sidecar.exists() or args.markdown.exists():
        raise RuntimeError("refusing to overwrite immutable Phase70 result")
    prereg = json.loads(args.prereg.read_text())
    screen = json.loads(args.screen.read_text())
    resource = json.loads(args.resource.read_text())
    sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    expected_sidecar = f"{sha256(args.prereg)}  {args.prereg.name}\n"
    root = Path(__file__).resolve().parents[2]
    code_paths = {
        "runner_sha256": root / "scripts/run_x2_upper_robust_one_update_phase56.py",
        "run_script_sha256": root / "scripts/run_x2_phase70_long_lookahead.sh",
        "lookahead_module_sha256": root / "src/cwi_x2/phase70_long_lookahead.py",
        "finalizer_sha256": Path(__file__).resolve(),
        "lookahead_test_sha256": root / "tests/test_phase70_long_lookahead.py",
        "finalizer_test_sha256": root / "tests/test_phase70_long_lookahead_finalizer.py",
        "phase69_module_sha256": root / "src/cwi_x2/phase69_reward_attribution.py",
        "phase68_interface_sha256": root / "src/cwi_x2/phase68_residual_ppo.py",
        "residual_module_sha256": root / "src/cwi_x2/phase_conditioned_knee_residual.py",
    }
    code_hash_checks = {
        name: path.is_file()
        and sha256(path) == prereg.get("immutable_code", {}).get(name)
        for name, path in code_paths.items()
    }
    prior = prereg.get("prior_attempt", {})

    def resolve(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else root / path

    prior_paths = {
        "failure": resolve(prior.get("failure", {}).get("path", "missing")),
        "resource": resolve(prior.get("resource", {}).get("path", "missing")),
        "log": resolve(prior.get("log", {}).get("path", "missing")),
        "frozen_runner_snapshot": resolve(
            prior.get("frozen_runner_snapshot", {}).get("path", "missing")
        ),
        "frozen_run_script_snapshot": resolve(
            prior.get("frozen_run_script_snapshot", {}).get("path", "missing")
        ),
        "repair_patch": resolve(
            prior.get("repair_patch", {}).get("path", "missing")
        ),
    }
    prior_hash_checks = {
        name: path.is_file()
        and sha256(path) == prior.get(name, {}).get("sha256")
        for name, path in prior_paths.items()
    }
    prior_resource = (
        json.loads(prior_paths["resource"].read_text())
        if prior_paths["resource"].is_file()
        else {}
    )
    bundle = Path(screen.get("rollout_bundle", ""))
    technical = screen.get("technical_checks", {})
    resource_checks = {
        "exit_code": resource.get("exit_code") == 0,
        "disk_delta": int(resource.get("disk_used_delta_bytes", 2**63))
        <= int(prereg["resource_limits"]["disk_delta_bytes_max"]),
        "gpu_peak": int(resource.get("gpu", {}).get("memory_used_peak_mib", 2**31))
        <= int(prereg["resource_limits"]["gpu_peak_memory_mib_max"]),
        "bundle_size": bundle.is_file()
        and bundle.stat().st_size
        <= int(prereg["resource_limits"]["raw_evidence_bundle_bytes_max"]),
        "cumulative_disk_delta": int(
            prior_resource.get("disk_used_delta_bytes", 2**62)
        )
        + int(resource.get("disk_used_delta_bytes", 2**62))
        <= int(prereg["resource_limits"]["cumulative_disk_delta_bytes_max"]),
    }
    validity_checks = {
        "prereg_sidecar": sidecar.is_file() and sidecar.read_text() == expected_sidecar,
        "immutable_code_hashes": bool(code_hash_checks)
        and all(code_hash_checks.values()),
        "prior_attempt_evidence_hashes": bool(prior_hash_checks)
        and all(prior_hash_checks.values()),
        "screen_schema": screen.get("schema") == "x2_phase70_long_lookahead_screen_v1",
        "screen_pending": screen.get("decision") == "LOOKAHEAD_VALID_PENDING_FINALIZATION",
        "all_technical": bool(technical) and all(technical.values()),
        "optimizer_steps_zero": screen.get("optimizer_steps") == 0,
        "checkpoint_count_zero": screen.get("checkpoint_count") == 0,
        "bundle_hash": bundle.is_file()
        and sha256(bundle) == screen.get("rollout_bundle_sha256"),
        "resource_checks": all(resource_checks.values()),
    }
    stability = screen.get("stability", {})
    cohort = stability.get("cohort_0_31_vs_32_63", {})
    centered = stability.get("observed400_vs_centered_locomotion_tail", {})
    reference = stability.get("observed400_vs_phase69_reference", {})
    reward_only = stability.get("primary_with_baseline_vs_reward_only", {})
    influence = stability.get("environment_influence", {})
    alignment = screen.get("alignment", {})
    reward_alignment_checks = {
        "pitch_reward_matches_pitch_metric": stable_sign(
            alignment.get("pitch_reward_vs_positive_pitch", {})
        )
        > 0,
        "support_reward_matches_support_metric": stable_sign(
            alignment.get("support_reward_vs_lower_support_outside", {})
        )
        > 0,
    }
    internal_quality_checks = {
        "cohort_direction_cosine": float(cohort.get("cosine", -1.0)) >= 0.90,
        "centered_tail_cosine": float(centered.get("cosine", -1.0)) >= 0.99,
        "centered_tail_relative_l2": float(centered.get("relative_l2", 2.0)) <= 0.10,
        "kish_effective_sample_size": float(
            influence.get("kish_effective_sample_size", 0.0)
        )
        >= 16.0,
        "maximum_env_contribution": float(
            influence.get("maximum_absolute_env_contribution_fraction", 1.0)
        )
        <= 0.20,
    }
    phase69_reference_consistent = float(reference.get("cosine", -1.0)) >= 0.90
    baseline_quality_checks = {
        "primary_vs_reward_only_cosine": float(reward_only.get("cosine", -1.0))
        >= 0.90,
        "primary_vs_reward_only_pitch_sign": stable_sign(
            alignment.get("primary_total_vs_positive_pitch", {})
        )
        != 0
        and stable_sign(alignment.get("primary_total_vs_positive_pitch", {}))
        == stable_sign(alignment.get("reward_only_total_vs_positive_pitch", {})),
        "primary_vs_reward_only_support_sign": stable_sign(
            alignment.get("primary_total_vs_lower_support_outside", {})
        )
        != 0
        and stable_sign(alignment.get("primary_total_vs_lower_support_outside", {}))
        == stable_sign(
            alignment.get("reward_only_total_vs_lower_support_outside", {})
        ),
    }
    if not all(validity_checks.values()):
        decision = "FAIL_INVALID_STOP"
    elif not all(reward_alignment_checks.values()):
        decision = "FAIL_REWARD_METRIC_ALIGNMENT_STOP"
    elif not all(internal_quality_checks.values()):
        decision = "INCONCLUSIVE_VARIANCE_OR_CUTOFF_STOP"
    elif not phase69_reference_consistent:
        decision = "HORIZON_SENSITIVE_STOP"
    elif not all(baseline_quality_checks.values()):
        decision = "BASELINE_SENSITIVE_OR_INCONCLUSIVE_STOP"
    elif reward_conflict_confirmed(alignment):
        decision = "PASS_REWARD_CONFLICT_DIAG_ONLY"
    else:
        decision = classify_scientific_outcome(
            alignment["primary_total_vs_positive_pitch"],
            alignment["primary_total_vs_lower_support_outside"],
        )
    report = {
        "schema": "x2_phase70_long_lookahead_result_v1",
        "decision": decision,
        "inputs": {
            "prereg": {"path": str(args.prereg), "sha256": sha256(args.prereg)},
            "screen": {"path": str(args.screen), "sha256": sha256(args.screen)},
            "resource": {"path": str(args.resource), "sha256": sha256(args.resource)},
            "rollout_bundle": {
                "path": str(bundle),
                "sha256": screen.get("rollout_bundle_sha256"),
                "artifact_role": "raw rollout evidence; not a checkpoint",
            },
            "prior_attempt": prior,
        },
        "validity_checks": validity_checks,
        "resource_checks": resource_checks,
        "code_hash_checks": code_hash_checks,
        "prior_hash_checks": prior_hash_checks,
        "reward_alignment_checks": reward_alignment_checks,
        "internal_quality_checks": internal_quality_checks,
        "baseline_quality_checks": baseline_quality_checks,
        "phase69_reference_consistent": phase69_reference_consistent,
        "estimand": screen.get("estimand"),
        "stability": stability,
        "alignment": alignment,
        "interpretation_boundary": (
            "This estimates the first-200 action-score gradient of a 400-step "
            "observed-reward lookahead. It is neither a full 400-step PPO gradient "
            "nor an unbiased infinite-horizon total-reward gradient."
        ),
        "optimizer_unlocked": False,
        "five_update_unlocked": False,
        "long_training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    result_sidecar.write_text(
        f"{sha256(args.output)}  {args.output.name}\n"
    )
    pitch = alignment.get("primary_total_vs_positive_pitch", {})
    support = alignment.get("primary_total_vs_lower_support_outside", {})
    args.markdown.write_text(
        "# X2 Phase70 long-lookahead estimator qualification\n\n"
        f"- Decision: `{decision}`\n"
        "- Physics/optimizer/checkpoints: `64 env × 400 / 0 / 0`\n"
        f"- Primary total → positive pitch projection: `{float(pitch.get('point', 0.0)):.6f}`\n"
        f"- Primary total → lower support-outside projection: `{float(support.get('point', 0.0)):.6f}`\n"
        f"- 32/32 cohort cosine: `{float(cohort.get('cosine', 0.0)):.6f}`\n"
        f"- Primary vs centered locomotion-tail cosine: `{float(centered.get('cosine', 0.0)):.6f}`\n"
        "- This phase cannot unlock an optimizer, long training, export, or deployment.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
