#!/usr/bin/env python3
"""Apply the frozen Phase48 physical/numerical gates to one Phase49 update."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", type=int)
    args = parser.parse_args()
    segment = args.segment
    tag = f"{segment:02d}"
    runtime_path = REPO / f"logs/x2_faithful_exact_s7_phase49_update{tag}/phase49_segment_runtime.json"
    train_path = REPO / f"logs/x2_faithful_phase49_regression_phase49_u{tag}_train/phase47_regression_phase49_u{tag}_train.json"
    gold_path = REPO / f"logs/x2_faithful_phase49_regression_phase49_u{tag}_held_out/phase47_regression_phase49_u{tag}_held_out.json"
    baseline_train = load(REPO / "logs/x2_faithful_phase47_regression_pre_train/phase47_regression_pre_train.json")
    baseline_gold = load(REPO / "logs/x2_faithful_phase47_regression_pre_held_out/phase47_regression_pre_held_out.json")
    runtime, train, gold = load(runtime_path), load(train_path), load(gold_path)
    lunge = "PHUMA-LUNGE-R-001"
    a, b = baseline_train["per_motion"][lunge], train["per_motion"][lunge]
    tracking_relative = {
        name: b["tracking"][name] / value - 1.0
        for name, value in a["tracking"].items()
    }
    lunge_checks = {
        "finite": all(math.isfinite(value) for value in b["tracking"].values()),
        "survival_drop_le_0p1s": a["survival_s_mean"] - b["survival_s_mean"] <= 0.1 + 1e-12,
        "termination_increase_le_0p05": b["termination_rate"] - a["termination_rate"] <= 0.05 + 1e-12,
        "at_least_one_tracking_error_decreases": any(value < 0 for value in tracking_relative.values()),
        "no_tracking_error_increases_over_10pct": max(tracking_relative.values()) <= 0.1 + 1e-12,
    }
    gold_rows = {}
    for key, source in baseline_gold["per_motion"].items():
        current = gold["per_motion"][key]
        survival_fraction = (
            (source["survival_s_mean"] - current["survival_s_mean"]) / source["survival_s_mean"]
            if source["survival_s_mean"] else 0.0
        )
        tracking = {
            name: current["tracking"][name] / value - 1.0
            for name, value in source["tracking"].items()
        }
        gold_rows[key] = {
            "survival_drop_fraction": survival_fraction,
            "termination_rate_increase": current["termination_rate"] - source["termination_rate"],
            "tracking_relative_change": tracking,
            "pass": survival_fraction <= 0.05 + 1e-12
            and current["termination_rate"] - source["termination_rate"] <= 0.05 + 1e-12
            and max(tracking.values()) <= 0.1 + 1e-12,
        }
    numerical_checks = {
        "runtime_pass": runtime["decision"] == "PASS_NUMERICAL_SEGMENT_PENDING_PHYSICAL_GATE",
        "segment_exact": runtime["segment"] == segment,
        "finite_kl": math.isfinite(float(runtime["training_diagnostics"]["policy/approxkl_avg"])),
        "kl_below_0p02": float(runtime["training_diagnostics"]["policy/approxkl_avg"]) < 0.02,
        "optimizer_steps_20": runtime["truth_boundary"]["optimizer_steps_this_process"] == 20,
        "frozen_hash_unchanged": runtime["frozen_parameter_hash"]["unchanged"] is True,
        "held_optimizer_samples_zero": runtime["truth_boundary"]["held_optimizer_samples"] == 0,
    }
    passed = all(numerical_checks.values()) and all(lunge_checks.values()) and all(
        row["pass"] for row in gold_rows.values()
    )
    report = {
        "schema_version": "x2_faithful_phase49_update_gate_v1",
        "segment": segment,
        "decision": "PASS_ALLOW_NEXT_SEGMENT" if passed and segment < 5 else (
            "PASS_FIVE_UPDATE_COMPLETE" if passed else "HARD_STOP_REJECT"
        ),
        "numerical_checks": numerical_checks,
        "lunge": {
            "baseline": a,
            "current": b,
            "tracking_relative_change": tracking_relative,
            "checks": lunge_checks,
        },
        "native_gold": gold_rows,
        "truth_boundary": {
            "comparison_baseline": "fresh original SONIC source deterministic regression",
            "next_segment_authorized": passed and segment < 5,
            "long_training_authorized": False,
        },
    }
    output = REPO / f"reports/retarget/x2_faithful_phase49_update{tag}_gate.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"decision": report["decision"], "output": str(output)}))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
