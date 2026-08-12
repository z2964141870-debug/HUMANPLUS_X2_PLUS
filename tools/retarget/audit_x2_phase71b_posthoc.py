#!/usr/bin/env python3
"""Repair-only CPU posthoc audit after the frozen invalid Phase71 v1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from cwi_x2.phase69_reward_attribution import (
    PHASE_NAMES,
    analytical_head_ascent_per_env,
    bootstrap_projection,
)
from cwi_x2.phase70_long_lookahead import (
    anchor_returns,
    standalone_credit as anchor_standalone_credit,
    vector_comparison,
)
from cwi_x2.phase_conditioned_knee_residual import PhaseConditionedKneeTargetResidual
from tools.retarget.audit_x2_phase71_posthoc import (
    cosine,
    fixed_groups,
    prefixed_tensor_hash,
    projection,
    raw_tensor_hash,
    relative_l2,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "phase71_v1_registration": "9febbaeb944498bbf25a172693b1a61dc6d89d851945e01bb14ec352d527f34d",
    "phase71_v1_result": "6dd70a608ee5199fbed4f5175745f4be4689897a046c90af99b3700464403c5f",
    "phase70_screen": "92533777eb3fba781043995b18cae03e6eaea7440cdc2ad21cfdd52505bb24f2",
    "phase70_bundle": "0b7f5f4026eb62ea474a2204b11b67e7c500024e1f3b814add0dffc482a07d81",
    "phase70_failure": "f084540b7ee37bfb648a60c5d76be0354318b8399217266ee169c725cd745fbb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_residual() -> PhaseConditionedKneeTargetResidual:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(680042)
        return PhaseConditionedKneeTargetResidual().eval()


def common_scale_components(
    total_reward: torch.Tensor,
    values: torch.Tensor,
    pitch_reward: torch.Tensor,
    support_reward: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    total = total_reward.to(dtype)
    value = values.to(dtype)
    pitch = pitch_reward.to(dtype)
    support = support_reward.to(dtype)
    # Recompute locomotion after dtype promotion so float64 tests arithmetic,
    # not float32 rounding already baked into a saved intermediate.
    locomotion = total - pitch - support
    raw = anchor_returns(total, anchor_steps=200) - value[:200]
    denominator = raw.std() + torch.tensor(1.0e-8, dtype=dtype)
    total_credit = (raw - raw.mean()) / denominator
    component_credit = {}
    for name, component in {
        "locomotion": locomotion,
        "pitch": pitch,
        "support": support,
    }.items():
        returns = anchor_returns(component, anchor_steps=200)
        component_credit[name] = (returns - returns.mean()) / denominator
    baseline = -value[:200]
    component_credit["baseline"] = (baseline - baseline.mean()) / denominator
    closure = sum(component_credit.values()) - total_credit
    return total_credit, component_credit, {
        "credit_relative_l2": relative_l2(sum(component_credit.values()), total_credit),
        "credit_max_abs": float(closure.abs().max()),
    }


def paired_stability(
    direction: torch.Tensor,
    metric: torch.Tensor,
    *,
    seed: int,
) -> dict[str, object]:
    if direction.shape != metric.shape or direction.shape[0] != 64:
        raise ValueError("paired stability requires [64, coordinate] vectors")
    groups = fixed_groups()
    group_projection = {}
    for name, indices in groups.items():
        index = torch.tensor(indices)
        group_projection[name] = projection(
            direction[index].mean(0), metric[index].mean(0)
        )
    full_projection = projection(direction.mean(0), metric.mean(0))
    sign = 1.0 if full_projection >= 0.0 else -1.0
    block_names = [
        "contiguous_00_15",
        "contiguous_16_31",
        "contiguous_32_47",
        "contiguous_48_63",
    ]
    lobo = {}
    for block, start in enumerate(range(0, 64, 16)):
        keep = torch.tensor([index for index in range(64) if not start <= index < start + 16])
        lobo[f"holdout_block_{block}"] = projection(
            direction[keep].mean(0), metric[keep].mean(0)
        )
    loeo = []
    ids = torch.arange(64)
    for held_out in range(64):
        keep = ids != held_out
        loeo.append(projection(direction[keep].mean(0), metric[keep].mean(0)))
    bootstrap = bootstrap_projection(direction, metric, seed=seed, draws=2048)
    direction_half_cosine = cosine(direction[:32].mean(0), direction[32:].mean(0))
    metric_half_cosine = cosine(metric[:32].mean(0), metric[32:].mean(0))
    excludes_zero = (
        bootstrap["bootstrap_p025"] > 0.0
        if sign > 0.0
        else bootstrap["bootstrap_p975"] < 0.0
    )
    same = lambda values: all(value * sign > 0.0 for value in values)
    stable = (
        same(group_projection[name] for name in block_names)
        and same(
            [group_projection["even_env_id"], group_projection["odd_env_id"]]
        )
        and same(lobo.values())
        and sum(value * sign > 0.0 for value in loeo) / 64.0 >= 0.90
        and direction_half_cosine >= 0.90
        and metric_half_cosine >= 0.90
        and excludes_zero
    )
    return {
        "full_projection": full_projection,
        "bootstrap": bootstrap,
        "fixed_group_projection": group_projection,
        "direction_first_second_half_cosine": direction_half_cosine,
        "metric_first_second_half_cosine": metric_half_cosine,
        "leave_one_block_out_projection": lobo,
        "leave_one_env_out_same_sign_fraction": sum(
            value * sign > 0.0 for value in loeo
        )
        / 64.0,
        "strict_stable": stable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--phase71-v1-registration", type=Path, required=True)
    parser.add_argument("--phase71-v1-result", type=Path, required=True)
    parser.add_argument("--phase70-screen", type=Path, required=True)
    parser.add_argument("--phase70-bundle", type=Path, required=True)
    parser.add_argument("--phase70-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    output_sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or output_sidecar.exists() or args.markdown.exists():
        raise RuntimeError("refusing to overwrite immutable Phase71b output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("Phase71b forbids CUDA visibility or initialization")
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    free_before = shutil.disk_usage(ROOT).free
    paths = {
        "phase71_v1_registration": args.phase71_v1_registration,
        "phase71_v1_result": args.phase71_v1_result,
        "phase70_screen": args.phase70_screen,
        "phase70_bundle": args.phase70_bundle,
        "phase70_failure": args.phase70_failure,
    }
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    registration = json.loads(args.registration.read_text())
    registration_sidecar = args.registration.with_suffix(args.registration.suffix + ".sha256")
    if registration.get("schema") != "x2_phase71b_posthoc_prereg_v1":
        raise RuntimeError("Phase71b registration schema changed")
    if input_hashes != INPUTS or input_hashes != registration["immutable_inputs"]:
        raise RuntimeError("Phase71b immutable input mismatch")
    if not registration_sidecar.is_file() or registration_sidecar.read_text() != (
        f"{sha256(args.registration)}  {args.registration.name}\n"
    ):
        raise RuntimeError("Phase71b registration sidecar mismatch")
    code_paths = {
        "audit_script_sha256": Path(__file__).resolve(),
        "audit_test_sha256": ROOT / "tests/test_phase71b_posthoc_audit.py",
        "phase71_v1_audit_sha256": ROOT / "tools/retarget/audit_x2_phase71_posthoc.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "phase70_helper_sha256": ROOT / "src/cwi_x2/phase70_long_lookahead.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
    }
    code_hashes = {name: sha256(path) for name, path in code_paths.items()}
    if code_hashes != registration["immutable_code"]:
        raise RuntimeError("Phase71b immutable code mismatch")
    screen = json.loads(args.phase70_screen.read_text())
    phase71_v1 = json.loads(args.phase71_v1_result.read_text())
    if screen.get("decision") != "FAIL_INVALID_STOP" or phase71_v1.get("decision") != "POSTHOC_RECONSTRUCTION_INVALID_STOP":
        raise RuntimeError("source invalid decisions changed")

    with torch.inference_mode():
        bundle = torch.load(args.phase70_bundle, map_location="cpu", weights_only=False)
        observation = bundle["policy_observation"]
        latent = bundle["latent_action"][:200]
        total_reward = bundle["total_reward"]
        values = bundle["value"]
        term_reward = bundle["reward_by_term"]
        names = tuple(bundle["reward_term_names"])
        pitch_reward = term_reward[..., names.index("signed_backward_pitch")]
        support_reward = term_reward[..., names.index("actual_support_com")]
        total_credit32, component_credit32, float32_closure = common_scale_components(
            total_reward, values, pitch_reward, support_reward, dtype=torch.float32
        )
        _, _, float64_closure = common_scale_components(
            total_reward, values, pitch_reward, support_reward, dtype=torch.float64
        )
        stored_credit = bundle["anchor_primary_normalized_advantage"]
        residual = build_residual()
        encoded = residual.encoder(observation[:200].flatten(0, 1)).reshape(200, 64, -1)
        primary = analytical_head_ascent_per_env(encoded, latent, stored_credit)
        stored_primary = bundle["anchor_head_ascent_per_env"]
        component = {
            name: analytical_head_ascent_per_env(encoded, latent, credit)
            for name, credit in component_credit32.items()
        }
        residual_credit = stored_credit - (
            component_credit32["locomotion"]
            + component_credit32["pitch"]
            + component_credit32["support"]
        )
        residual_direction = analytical_head_ascent_per_env(
            encoded, latent, residual_credit
        )
        residual_reconstructed = (
            component["locomotion"]
            + component["pitch"]
            + component["support"]
            + residual_direction
        )
        residual_closure = {
            "relative_l2": relative_l2(residual_reconstructed, primary),
            "max_abs": float((residual_reconstructed - primary).abs().max()),
        }
        reward_only = component["locomotion"] + component["pitch"] + component["support"]
        metric = {
            "positive_pitch": analytical_head_ascent_per_env(
                encoded,
                latent,
                anchor_standalone_credit(
                    bundle["signed_pitch_rad"] * 0.02, anchor_steps=200
                ),
            ),
            "lower_support_outside": analytical_head_ascent_per_env(
                encoded,
                latent,
                anchor_standalone_credit(
                    -bundle["support_outside_m"] * 0.02, anchor_steps=200
                ),
            ),
        }
        independent_reward = {
            "pitch_reward": analytical_head_ascent_per_env(
                encoded,
                latent,
                anchor_standalone_credit(pitch_reward, anchor_steps=200),
            ),
            "support_reward": analytical_head_ascent_per_env(
                encoded,
                latent,
                anchor_standalone_credit(support_reward, anchor_steps=200),
            ),
        }
        directions = {
            "primary": primary,
            "reward_only": reward_only,
            "locomotion": component["locomotion"],
            **independent_reward,
        }
        stability = {}
        seed = 713000
        for direction_name, direction in directions.items():
            for metric_name, metric_value in metric.items():
                seed += 1
                stability[f"{direction_name}_vs_{metric_name}"] = paired_stability(
                    direction, metric_value, seed=seed
                )
        phase_id = bundle["phase_id"][:200]
        phase = {
            name: analytical_head_ascent_per_env(
                encoded,
                latent,
                stored_credit,
                sample_mask=phase_id == index,
            )
            for index, name in enumerate(PHASE_NAMES)
        }
        phase_summary = {
            name: {
                "vs_primary": vector_comparison(value.mean(0), primary.mean(0)),
                "block_projection_to_positive_pitch": {
                    block: projection(
                        value[torch.tensor(indices)].mean(0),
                        metric["positive_pitch"][torch.tensor(indices)].mean(0),
                    )
                    for block, indices in fixed_groups().items()
                    if block.startswith("contiguous_")
                },
            }
            for name, value in phase.items()
        }
        term_directions = {}
        primary_raw = anchor_returns(total_reward, anchor_steps=200) - values[:200]
        denominator = primary_raw.std() + 1.0e-8
        for index, name in enumerate(names[:21]):
            reward = term_reward[..., index]
            if float(reward.abs().max()) == 0.0:
                term_directions[name] = {"zero_reward_term": True}
                continue
            independent = analytical_head_ascent_per_env(
                encoded,
                latent,
                anchor_standalone_credit(reward, anchor_steps=200),
            )
            returns = anchor_returns(reward, anchor_steps=200)
            additive = analytical_head_ascent_per_env(
                encoded, latent, (returns - returns.mean()) / denominator
            )
            term_directions[name] = {
                "zero_reward_term": False,
                "independent_vs_positive_pitch": paired_stability(
                    independent, metric["positive_pitch"], seed=714000 + index * 2
                ),
                "independent_vs_lower_support_outside": paired_stability(
                    independent, metric["lower_support_outside"], seed=714001 + index * 2
                ),
                "common_denominator_additive_vs_primary": vector_comparison(
                    additive.mean(0), primary.mean(0)
                ),
            }

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(700042)
    cpu_state = cpu_generator.get_state()
    cuda_state = torch.tensor(
        list((700042).to_bytes(8, "little") + (0).to_bytes(8, "little")),
        dtype=torch.uint8,
    )
    reconstruction = {
        "input_hashes": input_hashes == INPUTS,
        "code_hashes": code_hashes == registration["immutable_code"],
        "cpu_rng_raw_and_prefixed": (
            raw_tensor_hash(cpu_state) == "0721c6550d700ede33ea154260c940ab9d4be05b1d54bf192df7d51a89c668a8"
            and prefixed_tensor_hash(cpu_state) == screen["rng"]["cpu_state_sha256"]
        ),
        "cuda_seed_offset_convention_consistent_without_cuda": (
            raw_tensor_hash(cuda_state) == "2b6852ce608b9c5209ace6e7e6ca7006f4c8de288cbc5cc843a2a0bc3002e9a2"
            and prefixed_tensor_hash(cuda_state) == screen["rng"]["cuda_state_sha256"]
        ),
        "stored_credit_cpu_sensitivity_max_abs": float(
            (total_credit32 - stored_credit).abs().max()
        ) <= 1.0e-6,
        "stored_ascent_reconstructed": (
            cosine(primary.mean(0), stored_primary.mean(0)) >= 0.999999
            and relative_l2(primary, stored_primary) <= 1.0e-6
            and float((primary - stored_primary).abs().max()) <= 2.0e-7
        ),
        "float64_common_scale_closure": (
            float64_closure["credit_relative_l2"] <= 1.0e-12
            and float64_closure["credit_max_abs"] <= 1.0e-12
        ),
        "float32_residual_credit_closure": (
            residual_closure["relative_l2"] <= 1.0e-6
            and residual_closure["max_abs"] <= 1.0e-7
        ),
        "phase_closure": (
            relative_l2(sum(phase.values()), primary) <= 1.0e-6
        ),
        "all_finite": all(
            torch.isfinite(value).all()
            for value in (
                observation,
                latent,
                total_reward,
                term_reward,
                primary,
                *component.values(),
                *metric.values(),
                *independent_reward.values(),
                *phase.values(),
            )
        ),
        "cpu_only": not torch.cuda.is_initialized(),
        "grad_disabled": not torch.is_grad_enabled(),
    }
    pitch_required = {
        "pitch_reward_positive": stability["pitch_reward_vs_positive_pitch"],
        "primary_negative": stability["primary_vs_positive_pitch"],
        "reward_only_negative": stability["reward_only_vs_positive_pitch"],
        "locomotion_negative": stability["locomotion_vs_positive_pitch"],
    }
    pitch_conflict_stable = (
        all(value["strict_stable"] for value in pitch_required.values())
        and pitch_required["pitch_reward_positive"]["full_projection"] > 0.0
        and all(
            pitch_required[name]["full_projection"] < 0.0
            for name in ("primary_negative", "reward_only_negative", "locomotion_negative")
        )
    )
    support_required_names = (
        "support_reward_vs_lower_support_outside",
        "primary_vs_lower_support_outside",
        "reward_only_vs_lower_support_outside",
        "locomotion_vs_lower_support_outside",
    )
    support_all_stable = all(stability[name]["strict_stable"] for name in support_required_names)
    any_required_unstable = not (
        all(value["strict_stable"] for value in pitch_required.values())
        and support_all_stable
    )
    if not all(reconstruction.values()):
        decision = "POSTHOC_REPAIR_RECONSTRUCTION_INVALID_STOP"
    elif any_required_unstable:
        decision = "POSTHOC_VARIANCE_INSTABILITY_CONFIRMED_NO_PROMOTION"
    elif pitch_conflict_stable and not support_all_stable:
        decision = "POSTHOC_PITCH_CONFLICT_HYPOTHESIS_ONLY"
    else:
        decision = "POSTHOC_NUMERICAL_AUDIT_NO_PROMOTION"
    report = {
        "schema": "x2_phase71b_posthoc_audit_v1",
        "decision": decision,
        "analysis_mode": "repair_only_registered_posthoc_on_previously_inspected_data",
        "phase70_decision_unchanged": "FAIL_INVALID_STOP",
        "phase71_v1_decision_unchanged": "POSTHOC_RECONSTRUCTION_INVALID_STOP",
        "repair_provenance": {
            "review_was_requested_before_phase71_v1_execution": True,
            "review_identified_code_issues_without_using_phase71_v1_metrics": True,
            "fixed_metric_semantics": "anchor-normalize first-200 scores against full-400 returns",
            "fixed_stability_aggregation": "all preregistered primary/reward-only/locomotion/pitch/support directions",
        },
        "immutable_inputs": input_hashes,
        "immutable_code": code_hashes,
        "reconstruction_checks": reconstruction,
        "float32_common_scale_closure": float32_closure,
        "float64_common_scale_closure": float64_closure,
        "float32_residual_credit_closure": residual_closure,
        "phase70_historical_gpu_float32_closure": screen["gradient_validation"][
            "additive_closure"
        ],
        "stored_credit_cpu_sensitivity": {
            "max_abs": float((total_credit32 - stored_credit).abs().max()),
            "role": "CPU/GPU reduction-order sensitivity; stored bundle credit remains authoritative",
        },
        "stored_ascent_reconstruction": {
            "cosine": cosine(primary.mean(0), stored_primary.mean(0)),
            "relative_l2": relative_l2(primary, stored_primary),
            "max_abs": float((primary - stored_primary).abs().max()),
        },
        "fixed_group_stability": stability,
        "pitch_conflict_strict": pitch_conflict_stable,
        "support_required_all_stable": support_all_stable,
        "phase_local_associations": phase_summary,
        "locomotion_term_exploration": term_directions,
        "isaac_launches": 0,
        "physics_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint_count": 0,
        "resource": {
            "elapsed_seconds": time.perf_counter() - started,
            "disk_free_before_bytes": free_before,
            "disk_free_after_bytes_before_outputs": shutil.disk_usage(ROOT).free,
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "boundaries": {
            "confirmatory_claim": False,
            "promotion": "none",
            "optimizer_unlocked": False,
            "long_training_unlocked": False,
            "deployment_unlocked": False,
            "task2_complete": False,
            "future_confirmation_requires_new_independent_prereg_and_data": True,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    output_sidecar.write_text(f"{sha256(args.output)}  {args.output.name}\n")
    args.markdown.write_text(
        "# Phase71b repair-only registered posthoc audit\n\n"
        f"Decision: `{decision}`. Phase70 and Phase71 v1 decisions are unchanged.\n\n"
        f"- Stored ascent reconstruction relative-L2: `{report['stored_ascent_reconstruction']['relative_l2']:.9g}`.\n"
        f"- Float64 common-scale credit closure relative-L2: `{float64_closure['credit_relative_l2']:.9g}`.\n"
        f"- Strict pitch-conflict hypothesis: `{pitch_conflict_stable}`.\n"
        f"- All required support directions stable: `{support_all_stable}`.\n\n"
        "This registered posthoc repair cannot promote, train, export, or deploy a policy.\n"
    )
    print(json.dumps({"decision": decision, "reconstruction_checks": reconstruction}))
    if not all(reconstruction.values()):
        raise RuntimeError("Phase71b repair reconstruction gates failed")


if __name__ == "__main__":
    main()
