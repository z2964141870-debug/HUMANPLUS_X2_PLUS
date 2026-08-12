#!/usr/bin/env python3
"""Fail-closed two-seed, lane-swapped gate for the Phase77p pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


SOURCE_SHA256 = "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb"
TRAIN_SEEDS = (770101, 770102)
EVAL_SEEDS = (771001, 771002, 771003)
LANES = ("A", "B")
UPDATE_GATE_KEYS = {
    "finite",
    "post_minibatch_source_kl_mean",
    "post_minibatch_source_kl_max",
    "post_minibatch_incremental_kl_mean",
    "post_minibatch_incremental_kl_max",
    "rollout_source_kl_mean",
    "rollout_source_kl_max",
    "rollout_action_drift",
    "fixed_action_drift",
    "termination",
    "no_time_outs",
    "root_height",
    "root_tilt",
    "source_immutable",
    "candidate_std_immutable",
}
EXPECTED_CONFIG = {
    "num_envs": 256,
    "steps_per_env": 48,
    "updates_total": 10,
    "ppo_epochs": 1,
    "ppo_minibatches": 4,
    "ppo_clip": 0.10,
    "gamma": 0.99,
    "lambda": 0.95,
    "entropy_coefficient": 0.0,
    "actor_lr": 1.0e-5,
    "critic_lr": 1.0e-4,
    "source_kl_coefficient": 0.10,
    "source_kl_mean_max": 0.005,
    "source_kl_max": 0.02,
    "incremental_kl_mean_max": 0.0025,
    "incremental_kl_max": 0.01,
    "action_drift_max": 0.05,
    "posture_reward_weight": -0.5,
    "source_std_tensor_hash": "2d49dce85d503d6ec9cc69351cb1285f54e600bbdaed1afacf6d69d69f711c51",
    "source_std_min": 0.08823904395103455,
    "source_std_max": 0.7206376791000366,
    "command_allocation": {
        "vx_0p20": 64,
        "vx_0p35": 64,
        "vx_0p50": 64,
        "turn_vx_0p35_yaw_pm_0p15": 32,
        "transition_vx_0p35": 32,
    },
    "template_scale": 0.15,
    "fixed_upper": True,
    "ideal_actuator": True,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != (
        f"{sha256(path)}  {path.name}\n"
    ):
        raise RuntimeError(f"missing or invalid report sidecar: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite result")
        return value
    raise TypeError(type(value).__name__)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric(rows: list[dict], treatment: str, name: str) -> list[float]:
    values = [
        row[name]
        for row in rows
        if row["treatment"] == treatment and row.get(name) is not None
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"missing/non-finite {treatment} metric {name}")
    return values


def recompute_deltas(rows: list[dict]) -> dict[str, float]:
    source = [row for row in rows if row["treatment"] == "source"]
    candidate = [row for row in rows if row["treatment"] == "candidate"]

    def values(group: list[dict], name: str) -> list[float]:
        result = [row[name] for row in group if row.get(name) is not None]
        if not result or not all(math.isfinite(value) for value in result):
            raise RuntimeError(f"cannot recompute eval delta {name}")
        return result

    def mean(group: list[dict], name: str) -> float:
        result = values(group, name)
        return sum(result) / len(result)

    source_terminal = [row for row in source if int(row["role_id"]) == 7]
    candidate_terminal = [row for row in candidate if int(row["role_id"]) == 7]
    return {
        "signed_pitch_mean_rad": mean(candidate, "signed_pitch_mean_rad")
        - mean(source, "signed_pitch_mean_rad"),
        "signed_pitch_p05_rad": percentile(
            values(candidate, "signed_pitch_mean_rad"), 0.05
        )
        - percentile(values(source, "signed_pitch_mean_rad"), 0.05),
        "support_outside_mean_m": mean(candidate, "support_mean_m")
        - mean(source, "support_mean_m"),
        "velocity_rmse_mps": math.sqrt(mean(candidate, "velocity_mse"))
        - math.sqrt(mean(source, "velocity_mse")),
        "yaw_rmse_radps": math.sqrt(mean(candidate, "yaw_mse"))
        - math.sqrt(mean(source, "yaw_mse")),
        "termination_rate": sum(bool(row["terminated"]) for row in candidate)
        / len(candidate)
        - sum(bool(row["terminated"]) for row in source) / len(source),
        "stance_slip_p95_mps": percentile(
            values(candidate, "stance_slip_p95_mps"), 0.95
        )
        - percentile(values(source, "stance_slip_p95_mps"), 0.95),
        "terminal_speed_mean_mps": mean(
            candidate_terminal, "terminal_speed_mean_mps"
        )
        - mean(source_terminal, "terminal_speed_mean_mps"),
        "terminal_double_support_mean": mean(
            candidate_terminal, "terminal_double_support_mean"
        )
        - mean(source_terminal, "terminal_double_support_mean"),
    }


def two_level_pitch_bootstrap(rows: list[dict], seed: int) -> dict:
    by_eval = {
        eval_seed: [row for row in rows if int(row["eval_seed"]) == eval_seed]
        for eval_seed in EVAL_SEEDS
    }
    if any(not group for group in by_eval.values()):
        raise RuntimeError("missing eval seed for bootstrap")
    rng = random.Random(seed)
    estimates = []
    for _ in range(4096):
        selected_eval = [rng.choice(EVAL_SEEDS) for _ in EVAL_SEEDS]
        source_values: list[float] = []
        candidate_values: list[float] = []
        for eval_seed in selected_eval:
            group = by_eval[eval_seed]
            source = metric(group, "source", "signed_pitch_mean_rad")
            candidate = metric(group, "candidate", "signed_pitch_mean_rad")
            source_values.extend(rng.choices(source, k=len(source)))
            candidate_values.extend(rng.choices(candidate, k=len(candidate)))
        estimates.append(
            sum(candidate_values) / len(candidate_values)
            - sum(source_values) / len(source_values)
        )
    return {
        "draws": len(estimates),
        "cluster_order": "eval_seed_then_environment",
        "ci95_lower": percentile(estimates, 0.025),
        "ci95_upper": percentile(estimates, 0.975),
    }


def validate_train(row: dict, stage: int) -> bool:
    expected_updates = list(range(1, 6)) if stage == 5 else list(range(6, 11))
    updates = row.get("updates", [])
    return bool(
        row.get("schema") == "x2_phase77p_performance_train_v1"
        and row.get("decision") == "SEGMENT_VALID"
        and row.get("source_checkpoint_sha256") == SOURCE_SHA256
        and int(row.get("num_envs", -1)) == 256
        and int(row.get("steps_per_env", -1)) == 48
        and int(row.get("start_update", -1)) == (0 if stage == 5 else 5)
        and int(row.get("end_update", -1)) == stage
        and int(row.get("optimizer_steps", -1)) == 20
        and row.get("campaign_config") == EXPECTED_CONFIG
        and [int(item.get("update_index", -1)) for item in updates] == expected_updates
        and all(float(item.get("posture_reward_weight", 0.0)) == -0.5 for item in updates)
        and all(
            set(item.get("gates", {})) == UPDATE_GATE_KEYS
            and all(item["gates"].values())
            and int(item.get("losses", {}).get("optimizer_steps", -1)) == 4
            and float(item["losses"]["source_kl"]) <= 0.005
            and float(item["losses"]["source_kl_max"]) <= 0.02
            and float(item["losses"]["incremental_kl"]) <= 0.0025
            and float(item["losses"]["incremental_kl_max"]) <= 0.01
            and int(item["rollout_metrics"]["time_outs_count"]) == 0
            for item in updates
        )
        and row.get("terminal_bootstrap")
        == "installed RSL-RL policy.evaluate(s_T)"
        and int(row.get("checkpoint_count", -1)) == 1
        and sha256(Path(row["checkpoint"])) == row.get("checkpoint_sha256")
    )


def validate_eval(row: dict, stage: int) -> bool:
    per_env = row.get("per_env", [])
    expected_treatment = lambda env_id: (
        "source"
        if ((env_id % 2 == 0) == (row.get("lane") == "A"))
        else "candidate"
    )
    reported_deltas = row.get("candidate_minus_source", {})
    recalculated = recompute_deltas(per_env) if per_env else {}
    deltas_match = set(reported_deltas) == set(recalculated) and all(
        abs(float(reported_deltas[name]) - recalculated[name]) <= 1.0e-8
        for name in recalculated
    )
    return bool(
        row.get("schema") == "x2_phase77p_performance_eval_v1"
        and row.get("decision") == "EVAL_FINITE"
        and row.get("finite") is True
        and row.get("per_env_complete") is True
        and row.get("source_checkpoint_sha256") == SOURCE_SHA256
        and int(row.get("candidate_update_index", -1)) == stage
        and int(row.get("num_envs", -1)) == 128
        and int(row.get("eval_steps", -1)) == 512
        and row.get("lane") in LANES
        and int(row.get("optimizer_steps", -1)) == 0
        and int(row.get("checkpoint_writes", -1)) == 0
        and len(per_env) == 128
        and {int(item["env_id"]) for item in per_env} == set(range(128))
        and all(item["lane"] == row["lane"] for item in per_env)
        and deltas_match
        and all(
            int(item["pair_id"]) == int(item["env_id"]) // 2
            and int(item["role_id"]) == (int(item["env_id"]) // 2) % 8
            and item["treatment"] == expected_treatment(int(item["env_id"]))
            for item in per_env
        )
        and all(
            int(item.get("moving_sample_count", 1)) > 0
            and (
                int(item["role_id"]) != 7
                or int(item.get("terminal_sample_count", 1)) > 0
            )
            for item in per_env
        )
        and all(
            row["groups"][treatment]["role_counts"]
            == {
                "vx_0p20_a": 8,
                "vx_0p20_b": 8,
                "vx_0p35_a": 8,
                "vx_0p35_b": 8,
                "vx_0p50_a": 8,
                "vx_0p50_b": 8,
                "turn_vx_0p35_yaw_pm_0p15": 8,
                "transition_vx_0p35": 8,
            }
            for treatment in ("source", "candidate")
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(5, 10), required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--stage5-train", type=Path, action="append", default=[])
    parser.add_argument("--stage5-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(args.output)
    if len(args.train) != 2 or len(args.eval) != 12:
        raise ValueError("Phase77p requires 2 train and 12 lane-swapped eval reports")
    if args.stage == 10 and (len(args.stage5_train) != 2 or args.stage5_gate is None):
        raise ValueError("stage10 requires the two stage5 reports and stage5 gate")
    if args.stage == 5 and (args.stage5_train or args.stage5_gate is not None):
        raise ValueError("stage5 forbids continuation evidence")

    train_rows = [load(path) for path in args.train]
    eval_rows = [load(path) for path in args.eval]
    train_by_seed = {int(row["seed"]): row for row in train_rows}
    technical = {
        "train_seed_set": set(train_by_seed) == set(TRAIN_SEEDS),
        "train_reports": all(validate_train(row, args.stage) for row in train_rows),
        "eval_reports": all(validate_eval(row, args.stage) for row in eval_rows),
    }

    eval_by_train_seed: dict[int, list[dict]] = {seed: [] for seed in TRAIN_SEEDS}
    for row in eval_rows:
        train_seed = int(row["candidate_train_seed"])
        if train_seed not in eval_by_train_seed:
            raise RuntimeError(f"eval references unknown train seed {train_seed}")
        if row["candidate_checkpoint_sha256"] != train_by_seed[train_seed]["checkpoint_sha256"]:
            raise RuntimeError("eval candidate does not match train checkpoint")
        eval_by_train_seed[train_seed].append(row)
    technical["eval_factorial"] = all(
        {(int(row["eval_seed"]), row["lane"]) for row in rows}
        == {(seed, lane) for seed in EVAL_SEEDS for lane in LANES}
        for rows in eval_by_train_seed.values()
    )

    if args.stage == 10:
        stage5_rows = [load(path) for path in args.stage5_train]
        stage5_by_seed = {int(row["seed"]): row for row in stage5_rows}
        stage5_path_by_seed = {
            int(row["seed"]): path
            for row, path in zip(stage5_rows, args.stage5_train, strict=True)
        }
        stage5_gate = load(args.stage5_gate)
        technical["stage5_seed_set"] = set(stage5_by_seed) == set(TRAIN_SEEDS)
        technical["stage5_gate"] = bool(
            stage5_gate.get("decision") == "PASS_UPDATE5_CONTINUE_TO_UPDATE10"
            and stage5_gate.get("update10_training_unlocked") is True
            and all(validate_train(row, 5) for row in stage5_rows)
        )
        technical["resume_chain"] = all(
            train_by_seed[seed].get("resume_checkpoint_sha256")
            == stage5_by_seed[seed].get("checkpoint_sha256")
            and train_by_seed[seed].get("resume_train_report_sha256")
            == sha256(stage5_path_by_seed[seed])
            and train_by_seed[seed].get("resume_gate_sha256") == sha256(args.stage5_gate)
            for seed in TRAIN_SEEDS
        )

    efficacy_mean_min = 0.005 if args.stage == 5 else 0.015
    efficacy_p05_min = 0.0 if args.stage == 5 else 0.010
    seed_summaries = {}
    for seed, reports in sorted(eval_by_train_seed.items()):
        per_env = [
            {**item, "eval_seed": int(report["eval_seed"])}
            for report in reports
            for item in report["per_env"]
        ]
        source_pitch = metric(per_env, "source", "signed_pitch_mean_rad")
        candidate_pitch = metric(per_env, "candidate", "signed_pitch_mean_rad")
        pitch_mean_delta = sum(candidate_pitch) / len(candidate_pitch) - sum(source_pitch) / len(source_pitch)
        pitch_p05_delta = percentile(candidate_pitch, 0.05) - percentile(source_pitch, 0.05)
        bootstrap = two_level_pitch_bootstrap(per_env, 777000 + args.stage * 10 + seed)
        deltas = [row["candidate_minus_source"] for row in reports]
        candidates = [row["groups"]["candidate"] for row in reports]
        sources = [row["groups"]["source"] for row in reports]
        all_groups = candidates + sources
        gates = {
            "pitch_mean": pitch_mean_delta >= efficacy_mean_min,
            "pitch_mean_ci": bootstrap["ci95_lower"] > 0.0,
            "pitch_p05": pitch_p05_delta >= efficacy_p05_min,
            "support": max(row["support_outside_mean_m"] for row in deltas) <= 0.001,
            "velocity": max(row["velocity_rmse_mps"] for row in deltas) <= 0.010,
            "yaw": max(row["yaw_rmse_radps"] for row in deltas) <= 0.015,
            "slip": max(row["stance_slip_p95_mps"] for row in deltas) <= 0.030,
            "termination": max(row["termination_rate"] for row in all_groups) == 0.0,
            "survival": min(row["survival_s_min"] for row in all_groups) >= 10.24 - 1.0e-6,
            "no_time_outs": sum(row["time_outs_count"] for row in all_groups) == 0,
            "root_height_absolute": min(row["root_height_min_global"] for row in all_groups) >= 0.60,
            "root_height_relative": min(
                candidate["root_height_min_mean"] - source["root_height_min_mean"]
                for candidate, source in zip(candidates, sources, strict=True)
            ) >= -0.005,
            "tilt_absolute": max(row["root_tilt_max_global"] for row in all_groups) <= 0.35,
            "tilt_relative": max(
                candidate["root_tilt_max_mean"] - source["root_tilt_max_mean"]
                for candidate, source in zip(candidates, sources, strict=True)
            ) <= 0.010,
            "terminal_speed_absolute": max(
                row["terminal_base_speed_mps"]["mean"] for row in candidates
            ) <= 0.12,
            "terminal_speed_relative": max(
                row["terminal_speed_mean_mps"] for row in deltas
            ) <= 0.020,
            "terminal_double_support_absolute": min(
                row["terminal_double_support"]["mean"] for row in candidates
            ) >= 0.95,
            "terminal_double_support_relative": min(
                row["terminal_double_support_mean"] for row in deltas
            ) >= -0.020,
            "action_drift": max(
                row["candidate_source_action_drift_max"] for row in reports
            ) <= 0.05,
        }
        seed_summaries[str(seed)] = {
            "eval_seeds": list(EVAL_SEEDS),
            "lanes": list(LANES),
            "environment_summaries_per_treatment": len(source_pitch),
            "pitch_mean_delta_rad": pitch_mean_delta,
            "pitch_p05_delta_rad": pitch_p05_delta,
            "pitch_mean_two_level_bootstrap": bootstrap,
            "gates": gates,
            "passed": all(gates.values()),
        }

    valid = all(technical.values())
    passed = valid and all(row["passed"] for row in seed_summaries.values())
    if not valid:
        decision = "FAIL_INVALID_STOP"
    elif passed and args.stage == 5:
        decision = "PASS_UPDATE5_CONTINUE_TO_UPDATE10"
    elif passed:
        decision = "PASS_UPDATE10_LOCAL_PENDING_OFFICIAL_PANEL"
    else:
        decision = f"FAIL_UPDATE{args.stage}_NO_SAFE_SIGNAL_STOP"
    selected_seed = None
    if passed and args.stage == 10:
        selected_seed = max(
            TRAIN_SEEDS,
            key=lambda seed: seed_summaries[str(seed)]["pitch_mean_delta_rad"],
        )

    result = native(
        {
            "schema": "x2_phase77p_performance_local_gate_v2",
            "stage": args.stage,
            "decision": decision,
            "technical_checks": technical,
            "thresholds": {
                "pitch_mean_delta_rad_min": efficacy_mean_min,
                "pitch_mean_ci95_lower_min_exclusive": 0.0,
                "pitch_p05_delta_rad_min": efficacy_p05_min,
                "support_regression_m_max": 0.001,
                "velocity_rmse_regression_mps_max": 0.010,
                "yaw_rmse_regression_radps_max": 0.015,
                "action_drift_max": 0.05,
            },
            "train_reports": [
                {"path": str(path), "sha256": sha256(path)} for path in args.train
            ],
            "eval_reports": [
                {"path": str(path), "sha256": sha256(path)} for path in args.eval
            ],
            "seed_summaries": seed_summaries,
            "selected_train_seed": selected_seed,
            "update10_training_unlocked": decision == "PASS_UPDATE5_CONTINUE_TO_UPDATE10",
            "official_panel_unlocked": decision == "PASS_UPDATE10_LOCAL_PENDING_OFFICIAL_PANEL",
            "long_training_unlocked": False,
            "deployment_unlocked": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sidecar.write_text(f"{sha256(args.output)}  {args.output.name}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
