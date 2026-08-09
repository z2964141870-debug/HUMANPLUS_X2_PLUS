#!/usr/bin/env python3
"""Finalize the preregistered Phase56 source-vs-final A/B gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_score(group: dict) -> float:
    return (
        group["survival_s_mean"]
        - 2.0 * group["termination_rate"]
        - group["velocity_tracking_rmse"]
        - group["yaw_tracking_rmse"]
        - group["lateral_abs"]
        - group["yaw_abs"]
        - 0.1 * group["action_delta_abs"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(args.pre.read_text())
    train = json.loads(args.train.read_text())
    post = json.loads(args.post.read_text())
    prereg = json.loads(args.prereg.read_text())
    phase = int(train["phase"])
    if pre["phase"] != phase or post["phase"] != phase:
        raise RuntimeError("pre/train/post phase mismatch")
    A0, A1 = pre["groups"]["A_none"], post["groups"]["A_none"]
    B0, B1 = pre["groups"]["B_bounded"], post["groups"]["B_bounded"]
    checks = {
        "pre_finite": pre["finite"],
        "train_finite": train["finite"],
        "post_finite": post["finite"],
        "optimizer_steps_exact_20": train["optimizer_steps"] == 20,
        "checkpoint_count_exact_2": train["checkpoint_count"] == 2,
        "std_hash_unchanged": train["std_hash_before"] == train["std_hash_after"],
        "fixed_kl_mean_le_0p02": train["fixed_source_retention"]["kl_mean"] <= 0.02,
        "fixed_kl_max_le_0p20": train["fixed_source_retention"]["kl_max"] <= 0.20,
        "A_survival_not_regressed": A1["survival_s_mean"] >= A0["survival_s_mean"] - 0.10,
        "A_termination_not_regressed": A1["termination_rate"] <= A0["termination_rate"] + 0.05,
        "A_velocity_tracking_not_regressed": A1["velocity_tracking_rmse"] <= A0["velocity_tracking_rmse"] + max(0.02, 0.10 * A0["velocity_tracking_rmse"]),
        "A_lateral_not_regressed": A1["lateral_abs"] <= A0["lateral_abs"] + 0.02,
        "A_yaw_not_regressed": A1["yaw_abs"] <= A0["yaw_abs"] + 0.03,
        "B_survival_not_regressed": B1["survival_s_mean"] >= B0["survival_s_mean"] - 0.10,
        "B_termination_not_regressed": B1["termination_rate"] <= B0["termination_rate"] + 0.05,
        "B_robust_score_improved": robust_score(B1) > robust_score(B0),
        "A_root_height_not_regressed": A1["root_height_min_mean"] >= A0["root_height_min_mean"] - 0.03,
        "A_root_tilt_not_regressed": A1["root_tilt_max_mean"] <= A0["root_tilt_max_mean"] + 0.05,
        "B_root_height_not_regressed": B1["root_height_min_mean"] >= B0["root_height_min_mean"] - 0.03,
        "B_root_tilt_not_regressed": B1["root_tilt_max_mean"] <= B0["root_tilt_max_mean"] + 0.05,
        "A_action_abs_not_regressed": A1["action_abs"] <= A0["action_abs"] + 0.05,
        "A_action_delta_not_regressed": A1["action_delta_abs"] <= A0["action_delta_abs"] + 0.03,
        "B_action_abs_not_regressed": B1["action_abs"] <= B0["action_abs"] + 0.05,
        "B_action_delta_not_regressed": B1["action_delta_abs"] <= B0["action_delta_abs"] + 0.03,
        "fixed_action_drift_le_0p05": train["fixed_source_retention"]["action_max_abs"] <= 0.05,
        "fixed_value_drift_le_0p25": train["fixed_source_retention"]["value_max_abs"] <= 0.25,
    }
    if phase in (58, 59):
        checks.update({
            "learning_rate_exact_5e_5": train["learning_rate"] == 5.0e-5,
            "dense_hash_unchanged": train["dense_hash_before"] == train["dense_hash_after"],
            "trainable_scope_lora_only": bool(train["trainable_names"]) and all(
                "lora_" in name for name in train["trainable_names"]
            ),
        })
    gate_pass = all(checks.values())
    result = {
        "phase": phase,
        "decision": (
            ("PASS_LORA_ONE_UPDATE_TREND_GATE_STOP" if gate_pass else "FAIL_LORA_ONE_UPDATE_TREND_GATE_STOP")
            if phase in (58, 59) else
            ("PASS_ONE_UPDATE_TREND_GATE_STOP" if gate_pass else "FAIL_ONE_UPDATE_TREND_GATE_STOP")
        ),
        "prereg_sha256": sha256(args.prereg),
        "inputs": {"pre": str(args.pre), "train": str(args.train), "post": str(args.post)},
        "checks": checks,
        "scores": {
            "A_pre": robust_score(A0), "A_post": robust_score(A1),
            "B_pre": robust_score(B0), "B_post": robust_score(B1),
            "A_delta": robust_score(A1) - robust_score(A0),
            "B_delta": robust_score(B1) - robust_score(B0),
        },
        "metrics": {"A_pre": A0, "A_post": A1, "B_pre": B0, "B_post": B1},
        "training": train,
        "boundary": "One update only. A pass is a local trend gate, not robustness or deployment proof; never auto-unlocks five updates.",
    }
    if not all(math.isfinite(float(value)) for value in result["scores"].values()):
        raise RuntimeError("Phase56 score is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
