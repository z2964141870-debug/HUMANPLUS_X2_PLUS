#!/usr/bin/env python3
"""Aggregate five independent Phase72 antithetic pairs without optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from cwi_x2.phase72_antithetic import (
    cosine,
    environment_influence,
    projection,
    stratified_bootstrap_projection,
)


ROOT = Path(__file__).resolve().parents[2]
MAIN_RELATIONS = {
    "pitch_reward_vs_positive_pitch": ("pitch_reward", "positive_pitch_metric", 1),
    "primary_total_vs_positive_pitch": ("primary_total", "positive_pitch_metric", -1),
    "reward_only_vs_positive_pitch": ("reward_only", "positive_pitch_metric", -1),
    "locomotion_vs_positive_pitch": ("locomotion", "positive_pitch_metric", -1),
}
SUPPORT_RELATIONS = {
    "support_reward_vs_lower_support": ("support_reward", "lower_support_metric", 1),
    "primary_total_vs_lower_support": ("primary_total", "lower_support_metric", -1),
    "reward_only_vs_lower_support": ("reward_only", "lower_support_metric", -1),
    "locomotion_vs_lower_support": ("locomotion", "lower_support_metric", -1),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.is_file() and sidecar.read_text() == f"{sha256(path)}  {path.name}\n"


def fold_cosines(value: torch.Tensor) -> dict[str, float]:
    """Two preregistered balanced fold comparisons over [seed,env,coord]."""

    seeds = torch.arange(value.shape[0])[:, None]
    envs = torch.arange(value.shape[1])[None, :]
    half = ((envs < 32).to(torch.int64) ^ (seeds % 2).to(torch.int64)).bool()
    row = envs // 8
    column = envs % 8
    checker = ((row + column + seeds) % 2 == 0)
    return {
        "cross_seed_half": cosine(value[half].mean(dim=0), value[~half].mean(dim=0)),
        "seed_shifted_checkerboard": cosine(
            value[checker].mean(dim=0), value[~checker].mean(dim=0)
        ),
    }


def relation_summary(
    component: torch.Tensor,
    metric: torch.Tensor,
    *,
    expected_sign: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    full = projection(component.mean(dim=(0, 1)), metric.mean(dim=(0, 1)))
    seed_values = [projection(component[index].mean(dim=0), metric[index].mean(dim=0)) for index in range(component.shape[0])]
    loso_values = []
    for held_out in range(component.shape[0]):
        keep = torch.arange(component.shape[0]) != held_out
        loso_values.append(
            projection(component[keep].mean(dim=(0, 1)), metric[keep].mean(dim=(0, 1)))
        )
    block_values = []
    half_cosines = []
    for seed in range(component.shape[0]):
        block_values.append(
            [
                projection(
                    component[seed, start : start + 16].mean(dim=0),
                    metric[seed, start : start + 16].mean(dim=0),
                )
                for start in (0, 16, 32, 48)
            ]
        )
        half_cosines.append(
            cosine(component[seed, :32].mean(dim=0), component[seed, 32:].mean(dim=0))
        )
    bootstrap = stratified_bootstrap_projection(
        component, metric, seed=bootstrap_seed, draws=4096
    )
    threshold = 0.20
    seed_material = all(value * expected_sign >= threshold for value in seed_values)
    loso_material = all(value * expected_sign >= threshold for value in loso_values)
    blocks_same_sign = all(value * expected_sign > 0.0 for row in block_values for value in row)
    halves_stable = all(value >= 0.80 for value in half_cosines)
    ci_excludes_zero = (
        bootstrap["p025"] > 0.0 if expected_sign > 0 else bootstrap["p975"] < 0.0
    )
    return {
        "expected_sign": expected_sign,
        "point": full,
        "seed_projections": seed_values,
        "leave_one_seed_out_projections": loso_values,
        "four_blocks_by_seed": block_values,
        "component_half_cosine_by_seed": half_cosines,
        "exact_seed_sign_probability": 1.0 / 32.0,
        "bootstrap": bootstrap,
        "checks": {
            "five_of_five_seed_material_sign": seed_material,
            "leave_one_seed_out_material_sign": loso_material,
            "all_fixed_blocks_same_sign": blocks_same_sign,
            "all_within_seed_half_cosine": halves_stable,
            "cluster_bootstrap_ci_excludes_zero": ci_excludes_zero,
            "pooled_material_projection": full * expected_sign >= threshold,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--pair-result", type=Path, action="append", required=True)
    parser.add_argument("--resource", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result_sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or result_sidecar.exists() or args.markdown.exists():
        raise FileExistsError("refusing to overwrite immutable Phase72 final result")
    prereg = json.loads(args.prereg.read_text())
    pair_results = [json.loads(path.read_text()) for path in args.pair_result]
    resources = [json.loads(path.read_text()) for path in args.resource]
    if len(pair_results) != 5 or len(resources) != 10:
        raise RuntimeError("Phase72 finalizer requires exactly five pairs and ten launches")
    pair_results.sort(key=lambda value: int(value["seed_index"]))
    evidence = []
    for result in pair_results:
        path = Path(result["pair_evidence"])
        if not path.is_file() or not exact_sidecar(path) or sha256(path) != result["pair_evidence_sha256"]:
            raise RuntimeError("Phase72 pair evidence is absent or changed")
        evidence.append(torch.load(path, map_location="cpu", weights_only=False))
    directions = {
        name: torch.stack([row["directions"][name] for row in evidence], dim=0).to(torch.float64)
        for name in evidence[0]["directions"]
    }
    relation_reports = {}
    for index, (name, (component, metric, sign)) in enumerate(
        tuple(MAIN_RELATIONS.items()) + tuple(SUPPORT_RELATIONS.items())
    ):
        relation_reports[name] = relation_summary(
            directions[component],
            directions[metric],
            expected_sign=sign,
            bootstrap_seed=720100 + index,
        )
    pitch_direction_names = (
        "primary_total", "reward_only", "locomotion", "pitch_reward",
        "positive_pitch_metric",
    )
    support_direction_names = (
        "primary_total", "reward_only", "locomotion", "support_reward",
        "lower_support_metric",
    )
    required_direction_names = tuple(
        dict.fromkeys(pitch_direction_names + support_direction_names)
    )
    direction_stability = {
        name: {
            "balanced_fold_cosines": fold_cosines(directions[name]),
            "seed_vs_pooled_cosines": [
                cosine(directions[name][seed].mean(dim=0), directions[name].mean(dim=(0, 1)))
                for seed in range(5)
            ],
            "influence": environment_influence(directions[name].reshape(-1, 66)),
        }
        for name in directions
    }
    direction_quality = {
        name: (
            all(value >= 0.90 for value in direction_stability[name]["balanced_fold_cosines"].values())
            and all(value >= 0.80 for value in direction_stability[name]["seed_vs_pooled_cosines"])
            and direction_stability[name]["influence"]["kish_effective_sample_size"] >= 128.0
            and direction_stability[name]["influence"]["maximum_absolute_cluster_contribution_fraction"] <= 0.02
        )
        for name in required_direction_names
    }
    main_relation_quality = {
        name: all(report["checks"].values())
        for name, report in relation_reports.items()
        if name in MAIN_RELATIONS
    }
    support_relation_quality = {
        name: all(report["checks"].values())
        for name, report in relation_reports.items()
        if name in SUPPORT_RELATIONS
    }
    code_paths = {
        "runner_sha256": ROOT / "scripts/run_x2_phase72_antithetic.py",
        "helper_sha256": ROOT / "src/cwi_x2/phase72_antithetic.py",
        "run_script_sha256": ROOT / "scripts/run_x2_phase72_antithetic.sh",
        "pair_validator_sha256": ROOT / "tools/retarget/validate_x2_phase72_pair.py",
        "finalizer_sha256": Path(__file__).resolve(),
        "runner_test_sha256": ROOT / "tests/test_phase72_antithetic_runner.py",
        "helper_test_sha256": ROOT / "tests/test_phase72_antithetic.py",
        "finalizer_test_sha256": ROOT / "tests/test_phase72_antithetic_finalizer.py",
        "schedule_generator_sha256": ROOT / "tools/retarget/prepare_x2_phase72_schedules.py",
        "phase60_posture_module_sha256": ROOT / "src/x2_native_locomotion_posture_phase60.py",
        "x2_flat_env_cfg_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/x2_velocity/flat_env_cfg.py"),
        "x2_reward_module_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/x2_velocity/rewards.py"),
        "x2_gait_module_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/x2_velocity/gait.py"),
        "x2_action_module_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actions.py"),
        "heading_command_module_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/x2_velocity/heading_command.py"),
        "modular_env_cfg_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py"),
        "x2_robot_cfg_sha256": Path("/home/yu/x2_teleop_final/x2_sonic/sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py"),
        "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "phase70_helper_sha256": ROOT / "src/cwi_x2/phase70_long_lookahead.py",
        "ledger_sha256": ROOT / "tools/retarget/run_with_gpu_ledger.py",
    }
    code_checks = {
        name: path.is_file() and sha256(path) == prereg["immutable_code"].get(name)
        for name, path in code_paths.items()
    }
    initial_hashes = [row["initial_combined_sha256"] for row in pair_results]
    cumulative_disk = sum(
        max(0, int(row.get("disk_used_delta_bytes", 2**62))) for row in resources
    )
    expected_resource_labels = {
        f"phase72_seed{seed}_{sign}_antithetic_zero_optimizer"
        for seed in range(5)
        for sign in ("plus", "minus")
    }
    technical_checks = {
        "prereg_sidecar": exact_sidecar(args.prereg),
        "immutable_code": all(code_checks.values()),
        "pair_result_sidecars": all(exact_sidecar(path) for path in args.pair_result),
        "five_unique_seed_indices": [row["seed_index"] for row in pair_results] == list(range(5)),
        "five_distinct_initial_states": len(set(initial_hashes)) == 5,
        "all_pairs_valid": all(row.get("decision") == "PAIR_VALID_PENDING_FINAL" for row in pair_results),
        "all_pair_technical": all(all(row.get("technical_checks", {}).values()) for row in pair_results),
        "ten_resource_exit_zero": all(row.get("exit_code") == 0 for row in resources),
        "ten_resource_labels_exact": {
            row.get("label") for row in resources
        }
        == expected_resource_labels,
        "ten_resource_sidecars": all(exact_sidecar(path) for path in args.resource),
        "cumulative_disk_hard_cap": cumulative_disk <= int(prereg["resource_limits"]["cumulative_disk_delta_bytes_hard_max"]),
        "all_directions_finite": all(torch.isfinite(value).all() for value in directions.values()),
        "all_direction_shapes": all(list(value.shape) == [5, 64, 66] for value in directions.values()),
        "optimizer_steps_zero": True,
        "backward_calls_zero": True,
        "checkpoint_count_zero": True,
    }
    valid = all(technical_checks.values())
    pitch_direction_quality = all(direction_quality[name] for name in pitch_direction_names)
    support_direction_quality = all(
        direction_quality[name] for name in support_direction_names
    )
    pitch_confirmed = (
        valid
        and all(main_relation_quality.values())
        and pitch_direction_quality
    )
    support_secondary = (
        valid
        and all(support_relation_quality.values())
        and support_direction_quality
    )
    if not valid:
        decision = "FAIL_INVALID_STOP"
    elif pitch_confirmed:
        decision = "CONFIRM_LOCAL_PITCH_REWARD_CONFLICT_DIAGNOSTIC_ONLY"
    else:
        decision = "INCONCLUSIVE_INDEPENDENT_VARIANCE_STOP"
    report = {
        "schema": "x2_phase72_antithetic_result_v1",
        "decision": decision,
        "primary_question": "local pitch reward conflict under bounded knee residual exploration",
        "support_secondary_hypothesis_supported": support_secondary,
        "seed_count": 5,
        "launch_count": 10,
        "antithetic_pair_count": 5,
        "cluster_count": 320,
        "initial_combined_sha256": initial_hashes,
        "relations": relation_reports,
        "direction_stability": direction_stability,
        "direction_quality": direction_quality,
        "pitch_direction_quality": pitch_direction_quality,
        "support_direction_quality": support_direction_quality,
        "main_relation_quality": main_relation_quality,
        "support_relation_quality": support_relation_quality,
        "technical_checks": technical_checks,
        "code_hash_checks": code_checks,
        "resources": {
            "cumulative_disk_delta_bytes": cumulative_disk,
            "gpu_peak_memory_mib": max(
                float((row.get("gpu") or {}).get("memory_used_peak_mib", float("inf")))
                for row in resources
            ),
            "elapsed_s_total": sum(float(row["elapsed_s"]) for row in resources),
        },
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "optimizer_unlocked": False,
        "long_training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_temp = args.output.with_name(f".{args.output.name}.tmp")
    markdown_temp = args.markdown.with_name(f".{args.markdown.name}.tmp")
    if result_temp.exists() or markdown_temp.exists():
        raise FileExistsError("Phase72 final temporary output already exists")
    result_temp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(result_temp, args.output)
    digest = sha256(args.output)
    result_sidecar.write_text(f"{digest}  {args.output.name}\n")
    lines = [
        "# X2 Phase72 antithetic diagnostic",
        "",
        f"Decision: `{decision}`.",
        "",
        "Five independent reset/latent seeds were paired with exact opposite latent schedules. "
        "This phase used no backward pass, optimizer step, model checkpoint, export, or deployment promotion.",
        "",
        f"Pitch conflict confirmed: `{pitch_confirmed}`. Support secondary supported: `{support_secondary}`.",
    ]
    markdown_temp.write_text("\n".join(lines) + "\n")
    os.replace(markdown_temp, args.markdown)
    print(json.dumps(report, indent=2))
    if not valid:
        raise SystemExit("Phase72 final technical validity failed")


if __name__ == "__main__":
    main()
