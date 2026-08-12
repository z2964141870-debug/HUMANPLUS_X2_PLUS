#!/usr/bin/env python3
"""CPU-only registered posthoc audit of the immutable Phase70 invalid rollout."""

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
    standalone_credit,
    vector_summary,
)
from cwi_x2.phase_conditioned_knee_residual import PhaseConditionedKneeTargetResidual
from cwi_x2.phase70_long_lookahead import anchor_returns, normalized_advantage, vector_comparison


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "phase70_prereg": "430104815519552dc8ab4ce569f56ac6c76a4f2bd8f83ba0a6d3d2c108a266dc",
    "screen": "92533777eb3fba781043995b18cae03e6eaea7440cdc2ad21cfdd52505bb24f2",
    "resource": "e8972f21fc129b2aa475f565a169783c9937dd02dd4bf7042612457dacc844f5",
    "failure": "f084540b7ee37bfb648a60c5d76be0354318b8399217266ee169c725cd745fbb",
    "bundle": "0b7f5f4026eb62ea474a2204b11b67e7c500024e1f3b814add0dffc482a07d81",
    "log": "7006499189a6266cee5836cf08818dd18b046857cd841f00972f8c8f65e52b3e",
}
RAW_CPU_RNG_SHA = "0721c6550d700ede33ea154260c940ab9d4be05b1d54bf192df7d51a89c668a8"
PREFIXED_CPU_RNG_SHA = "408382543c7b8165648ae6e434929c29dc7f97601d49a0135c1a9ec1a1d0130f"
RAW_CUDA_RNG_SHA = "2b6852ce608b9c5209ace6e7e6ca7006f4c8de288cbc5cc843a2a0bc3002e9a2"
PREFIXED_CUDA_RNG_SHA = "27b790394093efd7178dd2fc462278b20f3c5a49c8ec46bbe56ebae37a0c0d4f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def prefixed_tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(f"{value.dtype}:{value.shape}".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_seeded_residual() -> PhaseConditionedKneeTargetResidual:
    """Rebuild the frozen Phase68/70 encoder without importing RSL-RL."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(680042)
        return PhaseConditionedKneeTargetResidual()


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    value = value.flatten().to(torch.float64)
    reference = reference.flatten().to(torch.float64)
    return float(
        torch.linalg.vector_norm(value - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1.0e-30)
    )


def cosine(value: torch.Tensor, reference: torch.Tensor) -> float:
    value = value.flatten().to(torch.float64)
    reference = reference.flatten().to(torch.float64)
    return float(
        torch.dot(value, reference)
        / (
            torch.linalg.vector_norm(value)
            * torch.linalg.vector_norm(reference)
        ).clamp_min(1.0e-30)
    )


def projection(value: torch.Tensor, reference: torch.Tensor) -> float:
    value = value.flatten().to(torch.float64)
    reference = reference.flatten().to(torch.float64)
    denominator = torch.dot(reference, reference)
    if denominator <= 1.0e-30:
        raise ValueError("projection reference is zero")
    return float(torch.dot(value, reference) / denominator)


def component_audit(
    total_reward: torch.Tensor,
    values: torch.Tensor,
    locomotion_reward: torch.Tensor,
    pitch_reward: torch.Tensor,
    support_reward: torch.Tensor,
    encoded: torch.Tensor,
    latent: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> dict[str, object]:
    total_reward = total_reward.to(dtype)
    values = values.to(dtype)
    rewards = {
        "locomotion": locomotion_reward.to(dtype),
        "pitch": pitch_reward.to(dtype),
        "support": support_reward.to(dtype),
    }
    encoded = encoded.to(dtype)
    latent = latent.to(dtype)
    raw, primary_credit = normalized_advantage(
        total_reward, values, anchor_steps=200, gamma=0.99
    )
    denominator = raw.std() + torch.tensor(1.0e-8, dtype=dtype)
    credits = {
        name: (value - value.mean()) / denominator
        for name, reward in rewards.items()
        for value in (anchor_returns(reward, anchor_steps=200, gamma=0.99),)
    }
    baseline = -values[:200]
    credits["baseline"] = (baseline - baseline.mean()) / denominator
    per_env = {
        name: analytical_head_ascent_per_env(encoded, latent, credit)
        for name, credit in credits.items()
    }
    primary_per_env = analytical_head_ascent_per_env(
        encoded, latent, primary_credit
    )
    additive = sum(per_env.values())
    additive_mean = additive.mean(dim=0)
    primary_mean = primary_per_env.mean(dim=0)
    credit_sum = sum(credits.values())
    return {
        "primary_credit": primary_credit,
        "primary_per_env": primary_per_env,
        "component_per_env": per_env,
        "credit_closure_max_abs": float((credit_sum - primary_credit).abs().max()),
        "credit_closure_relative_l2": relative_l2(credit_sum, primary_credit),
        "gradient_closure_cosine": cosine(additive_mean, primary_mean),
        "gradient_closure_relative_l2": relative_l2(additive_mean, primary_mean),
        "gradient_closure_max_abs": float((additive_mean - primary_mean).abs().max()),
    }


def residual_credit_audit(
    primary_credit: torch.Tensor,
    component_per_env: dict[str, torch.Tensor],
    component_credits: dict[str, torch.Tensor],
    encoded: torch.Tensor,
    latent: torch.Tensor,
) -> dict[str, float]:
    non_residual = (
        component_credits["locomotion"]
        + component_credits["pitch"]
        + component_credits["support"]
    )
    residual = primary_credit - non_residual
    residual_per_env = analytical_head_ascent_per_env(encoded, latent, residual)
    reconstructed = (
        component_per_env["locomotion"]
        + component_per_env["pitch"]
        + component_per_env["support"]
        + residual_per_env
    )
    primary_per_env = analytical_head_ascent_per_env(encoded, latent, primary_credit)
    return {
        "credit_max_abs": float((non_residual + residual - primary_credit).abs().max()),
        "gradient_relative_l2": relative_l2(
            reconstructed.mean(dim=0), primary_per_env.mean(dim=0)
        ),
        "gradient_max_abs": float(
            (reconstructed.mean(dim=0) - primary_per_env.mean(dim=0)).abs().max()
        ),
    }


def fixed_groups() -> dict[str, list[int]]:
    return {
        "contiguous_00_15": list(range(0, 16)),
        "contiguous_16_31": list(range(16, 32)),
        "contiguous_32_47": list(range(32, 48)),
        "contiguous_48_63": list(range(48, 64)),
        "even_env_id": list(range(0, 64, 2)),
        "odd_env_id": list(range(1, 64, 2)),
        "first_half": list(range(0, 32)),
        "second_half": list(range(32, 64)),
    }


def group_summary(per_env: torch.Tensor) -> dict[str, object]:
    full = per_env.mean(dim=0)
    groups = fixed_groups()
    vectors = {
        name: per_env[torch.tensor(indices)].mean(dim=0)
        for name, indices in groups.items()
    }
    summaries = {
        name: vector_comparison(vector, full) for name, vector in vectors.items()
    }
    pairwise = {
        "first_vs_second_half": vector_comparison(
            vectors["first_half"], vectors["second_half"]
        ),
        "even_vs_odd": vector_comparison(vectors["even_env_id"], vectors["odd_env_id"]),
    }
    blocks = [list(range(start, start + 16)) for start in range(0, 64, 16)]
    leave_one_block_out = {}
    for block_id, held_out in enumerate(blocks):
        keep = [index for index in range(64) if index not in held_out]
        leave_one_block_out[f"holdout_block_{block_id}"] = vector_comparison(
            per_env[torch.tensor(keep)].mean(dim=0), full
        )
    return {
        "groups_vs_full": summaries,
        "pairwise": pairwise,
        "leave_one_16_env_block_out": leave_one_block_out,
    }


def direction_stability(
    direction_per_env: torch.Tensor,
    metric_per_env: torch.Tensor,
    *,
    seed: int,
) -> dict[str, object]:
    """Evaluate one posthoc direction against one metric on fixed env groups."""

    if direction_per_env.shape != metric_per_env.shape or direction_per_env.shape[0] != 64:
        raise ValueError("direction and metric must share [64, coordinate]")
    groups = fixed_groups()
    group_projection = {}
    for name, indices in groups.items():
        index = torch.tensor(indices)
        group_projection[name] = projection(
            direction_per_env[index].mean(dim=0),
            metric_per_env[index].mean(dim=0),
        )
    full = projection(direction_per_env.mean(dim=0), metric_per_env.mean(dim=0))
    sign = 1.0 if full >= 0.0 else -1.0
    blocks = [list(range(start, start + 16)) for start in range(0, 64, 16)]
    lobo = {}
    for block_id, held_out in enumerate(blocks):
        keep = torch.tensor([index for index in range(64) if index not in held_out])
        lobo[f"holdout_block_{block_id}"] = projection(
            direction_per_env[keep].mean(dim=0), metric_per_env[keep].mean(dim=0)
        )
    loeo = []
    all_indices = torch.arange(64)
    for held_out in range(64):
        keep = all_indices != held_out
        loeo.append(
            projection(
                direction_per_env[keep].mean(dim=0),
                metric_per_env[keep].mean(dim=0),
            )
        )
    bootstrap = bootstrap_projection(
        direction_per_env, metric_per_env, seed=seed, draws=2048
    )
    block_values = [group_projection[f"contiguous_{start:02d}_{start + 15:02d}"] for start in range(0, 64, 16)]
    odd_even = [group_projection["even_env_id"], group_projection["odd_env_id"]]
    lobo_values = list(lobo.values())
    same = lambda values: all(value * sign > 0.0 for value in values)
    bootstrap_excludes_zero = (
        float(bootstrap["bootstrap_p025"]) > 0.0
        if sign > 0.0
        else float(bootstrap["bootstrap_p975"]) < 0.0
    )
    half_cosine = cosine(
        direction_per_env[:32].mean(dim=0),
        direction_per_env[32:].mean(dim=0),
    )
    stable = (
        same(block_values)
        and same(odd_even)
        and same(lobo_values)
        and sum(value * sign > 0.0 for value in loeo) / 64.0 >= 0.90
        and half_cosine >= 0.90
        and bootstrap_excludes_zero
    )
    return {
        "full_projection": full,
        "full_bootstrap": bootstrap,
        "fixed_group_projection": group_projection,
        "first_half_vs_second_half_direction_cosine": half_cosine,
        "leave_one_16_env_block_out_projection": lobo,
        "leave_one_env_out_same_sign_fraction": sum(
            value * sign > 0.0 for value in loeo
        )
        / 64.0,
        "strict_direction_stable": stable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--phase70-prereg", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists() or args.markdown.exists():
        raise RuntimeError("refusing to overwrite immutable Phase71 output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Phase71 requires CUDA_VISIBLE_DEVICES='' for CPU-only audit")
    if torch.cuda.is_initialized():
        raise RuntimeError("Phase71 forbids CUDA initialization")
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    disk_before = shutil.disk_usage(ROOT).free
    paths = {
        "phase70_prereg": args.phase70_prereg,
        "screen": args.screen,
        "resource": args.resource,
        "failure": args.failure,
        "bundle": args.bundle,
        "log": args.log,
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    prereg = json.loads(args.registration.read_text())
    registration_sidecar = args.registration.with_suffix(
        args.registration.suffix + ".sha256"
    )
    expected_registration_sidecar = (
        f"{sha256(args.registration)}  {args.registration.name}\n"
    )
    screen = json.loads(args.screen.read_text())
    failure = json.loads(args.failure.read_text())
    if prereg.get("schema") != "x2_phase71_posthoc_prereg_v1":
        raise RuntimeError("Phase71 prereg schema changed")
    if (
        not registration_sidecar.is_file()
        or registration_sidecar.read_text() != expected_registration_sidecar
    ):
        raise RuntimeError("Phase71 registration sidecar mismatch")
    code_paths = {
        "audit_script_sha256": Path(__file__).resolve(),
        "audit_test_sha256": ROOT / "tests/test_phase71_posthoc_audit.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "phase70_helper_sha256": ROOT / "src/cwi_x2/phase70_long_lookahead.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
    }
    code_hashes = {name: sha256(path) for name, path in code_paths.items()}
    if code_hashes != prereg.get("immutable_code"):
        raise RuntimeError("Phase71 immutable code hash mismatch")
    if screen.get("decision") != "FAIL_INVALID_STOP":
        raise RuntimeError("Phase70 screen is no longer invalid")
    if failure.get("decision") != "FAIL_INVALID_STOP":
        raise RuntimeError("Phase70 terminal failure record changed")
    if hashes != EXPECTED or hashes != prereg.get("immutable_inputs"):
        raise RuntimeError("Phase71 immutable input hash mismatch")

    with torch.inference_mode():
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
        if bundle.get("schema") != "x2_phase70_long_lookahead_evidence_v1":
            raise RuntimeError("Phase70 evidence schema changed")
        observation = bundle["policy_observation"]
        latent = bundle["latent_action"][:200]
        total_reward = bundle["total_reward"]
        values = bundle["value"]
        term_reward = bundle["reward_by_term"]
        names = tuple(bundle["reward_term_names"])
        weights = tuple(float(value) for value in bundle["reward_term_weights"])
        pitch_reward = term_reward[..., names.index("signed_backward_pitch")]
        support_reward = term_reward[..., names.index("actual_support_com")]
        locomotion_reward = total_reward - pitch_reward - support_reward
        residual = build_seeded_residual().eval()
        encoded = residual.encoder(observation[:200].flatten(0, 1)).reshape(200, 64, -1)

        float32 = component_audit(
            total_reward,
            values,
            locomotion_reward,
            pitch_reward,
            support_reward,
            encoded,
            latent,
            dtype=torch.float32,
        )
        float64 = component_audit(
            total_reward,
            values,
            locomotion_reward,
            pitch_reward,
            support_reward,
            encoded,
            latent,
            dtype=torch.float64,
        )
        primary_credit32 = float32["primary_credit"]
        denominator32 = (
            anchor_returns(total_reward, anchor_steps=200) - values[:200]
        ).std() + 1.0e-8
        component_credits32 = {
            name: (anchor_returns(reward, anchor_steps=200) - anchor_returns(reward, anchor_steps=200).mean()) / denominator32
            for name, reward in {
                "locomotion": locomotion_reward,
                "pitch": pitch_reward,
                "support": support_reward,
            }.items()
        }
        residual_closure = residual_credit_audit(
            primary_credit32,
            float32["component_per_env"],
            component_credits32,
            encoded,
            latent,
        )

        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(700042)
        cpu_state = cpu_generator.get_state()
        cuda_state_bytes = (700042).to_bytes(8, "little") + (0).to_bytes(8, "little")
        cuda_state = torch.tensor(list(cuda_state_bytes), dtype=torch.uint8)
        rng = {
            "cpu_raw_sha256": raw_tensor_hash(cpu_state),
            "cpu_prefixed_sha256": prefixed_tensor_hash(cpu_state),
            "cuda_state_reconstructed_without_cuda": True,
            "cuda_state_semantics": "little-endian uint64 seed followed by uint64 offset zero",
            "cuda_raw_sha256": raw_tensor_hash(cuda_state),
            "cuda_prefixed_sha256": prefixed_tensor_hash(cuda_state),
        }

        primary_per_env = float32["primary_per_env"]
        signed_pitch = bundle["signed_pitch_rad"]
        support_outside = bundle["support_outside_m"]
        metric_per_env = {
            "positive_pitch": analytical_head_ascent_per_env(
                encoded,
                latent,
                standalone_credit(signed_pitch * 0.02)[:200],
            ),
            "lower_support_outside": analytical_head_ascent_per_env(
                encoded,
                latent,
                standalone_credit(-support_outside * 0.02)[:200],
            ),
        }
        reward_per_env = {
            "pitch": analytical_head_ascent_per_env(
                encoded, latent, standalone_credit(pitch_reward)[:200]
            ),
            "support": analytical_head_ascent_per_env(
                encoded, latent, standalone_credit(support_reward)[:200]
            ),
        }
        reward_only_per_env = (
            float32["component_per_env"]["locomotion"]
            + float32["component_per_env"]["pitch"]
            + float32["component_per_env"]["support"]
        )
        direction_vectors = {
            "primary": primary_per_env,
            "reward_only": reward_only_per_env,
            "locomotion": float32["component_per_env"]["locomotion"],
            "pitch_reward": reward_per_env["pitch"],
            "support_reward": reward_per_env["support"],
        }
        alignment = {}
        alignment_seed = 710100
        for direction_name, direction in direction_vectors.items():
            for metric_name, metric in metric_per_env.items():
                alignment_seed += 1
                alignment[f"{direction_name}_vs_{metric_name}"] = direction_stability(
                    direction, metric, seed=alignment_seed
                )
        phase_id = bundle["phase_id"][:200]
        phase_per_env = {
            name: analytical_head_ascent_per_env(
                encoded,
                latent,
                primary_credit32,
                sample_mask=phase_id == index,
            )
            for index, name in enumerate(PHASE_NAMES)
        }
        phase_summary = {}
        phase_seed = 711000
        for phase_name, direction in phase_per_env.items():
            phase_summary[phase_name] = {
                "vs_primary": vector_summary(
                    direction.mean(dim=0), primary_per_env.mean(dim=0)
                ),
                "metric_associations": {},
            }
            for metric_name, metric in metric_per_env.items():
                phase_seed += 1
                phase_summary[phase_name]["metric_associations"][metric_name] = direction_stability(
                    direction, metric, seed=phase_seed
                )
        term_direction = {}
        primary_raw32 = anchor_returns(total_reward, anchor_steps=200) - values[:200]
        primary_denominator32 = primary_raw32.std() + 1.0e-8
        term_seed = 712000
        for index, name in enumerate(names[:21]):
            reward = term_reward[..., index]
            is_zero = float(reward.abs().max()) == 0.0
            if is_zero:
                term_direction[name] = {
                    "zero_reward_term": True,
                    "independent_direction": "undefined",
                    "common_denominator_additive": "zero",
                }
                continue
            independent_direction = analytical_head_ascent_per_env(
                encoded, latent, standalone_credit(reward)[:200]
            )
            term_return = anchor_returns(reward, anchor_steps=200)
            term_credit = (term_return - term_return.mean()) / primary_denominator32
            additive_direction = analytical_head_ascent_per_env(
                encoded, latent, term_credit
            )
            term_direction[name] = {
                "zero_reward_term": False,
                "common_denominator_additive_vs_primary": vector_summary(
                    additive_direction.mean(dim=0), primary_per_env.mean(dim=0)
                ),
                "independent_metric_direction": {},
            }
            for metric_name, metric in metric_per_env.items():
                term_seed += 1
                term_direction[name]["independent_metric_direction"][metric_name] = direction_stability(
                    independent_direction, metric, seed=term_seed
                )

    expected_shapes = {
        "policy_observation": (400, 64, 93),
        "critic_observation": (400, 64, 93),
        "latent_action": (400, 64, 2),
        "total_reward": (400, 64),
        "reward_by_term": (400, 64, 23),
        "value": (400, 64),
        "phase_id": (400, 64),
        "anchor_primary_normalized_advantage": (200, 64),
        "anchor_head_ascent_per_env": (64, 66),
    }
    shape_checks = {
        key: tuple(bundle[key].shape) == expected
        for key, expected in expected_shapes.items()
    }
    reward_closure_max = float((total_reward - term_reward.sum(dim=-1)).abs().max())
    phase_full = bundle["phase_id"]
    phase_valid = bool(torch.all((phase_full >= 0) & (phase_full < len(PHASE_NAMES))))
    anchor_phase_complete = all(
        bool(torch.all((phase_full[:200] == index).sum(dim=0) > 0))
        for index in range(len(PHASE_NAMES))
    )
    ascent_reconstruction = {
        "cosine": cosine(
            primary_per_env.mean(dim=0),
            bundle["anchor_head_ascent_per_env"].mean(dim=0),
        ),
        "relative_l2": relative_l2(
            primary_per_env, bundle["anchor_head_ascent_per_env"]
        ),
        "max_abs": float(
            (primary_per_env - bundle["anchor_head_ascent_per_env"]).abs().max()
        ),
    }
    phase_closure = {
        "relative_l2": relative_l2(sum(phase_per_env.values()), primary_per_env),
        "max_abs": float((sum(phase_per_env.values()) - primary_per_env).abs().max()),
    }
    full_latent_sha = prefixed_tensor_hash(bundle["latent_action"])
    anchor_latent_sha = prefixed_tensor_hash(bundle["latent_action"][:200])
    reconstruction_checks = {
        "immutable_input_hashes": hashes == EXPECTED == prereg["immutable_inputs"],
        "immutable_code_hashes": code_hashes == prereg["immutable_code"],
        "registration_sidecar": registration_sidecar.read_text()
        == expected_registration_sidecar,
        "bundle_shapes": all(shape_checks.values()),
        "reward_terms_frozen": (
            list(names) == prereg["reward_terms"]["names"]
            and list(weights) == prereg["reward_terms"]["weights"]
        ),
        "reward_closure": reward_closure_max <= 1.0e-7,
        "phase_ids_valid": phase_valid,
        "anchor_phase_complete_per_env": anchor_phase_complete,
        "latent_hashes_match_screen": (
            full_latent_sha == screen["rng"]["full_latent_sha256"]
            and anchor_latent_sha == screen["rng"]["anchor_latent_sha256"]
        ),
        "rng_cpu_raw_reproduced": rng["cpu_raw_sha256"] == RAW_CPU_RNG_SHA,
        "rng_cpu_prefixed_reproduced": rng["cpu_prefixed_sha256"] == PREFIXED_CPU_RNG_SHA,
        "rng_cuda_seed_offset_structure_matches_frozen": (
            rng["cuda_raw_sha256"] == RAW_CUDA_RNG_SHA
            and rng["cuda_prefixed_sha256"] == PREFIXED_CUDA_RNG_SHA
            and screen["rng"]["cuda_state_sha256"] == PREFIXED_CUDA_RNG_SHA
        ),
        "stored_primary_credit_reproduced": float(
            (primary_credit32 - bundle["anchor_primary_normalized_advantage"]).abs().max()
        ) <= 1.0e-7,
        "stored_ascent_reproduced": (
            ascent_reconstruction["cosine"] >= 0.999999
            and ascent_reconstruction["relative_l2"] <= 1.0e-5
            and ascent_reconstruction["max_abs"] <= 1.0e-7
        ),
        "phase_gradient_closure": (
            phase_closure["relative_l2"] <= 1.0e-6
            and phase_closure["max_abs"] <= 1.0e-7
        ),
        "all_finite": all(
            torch.isfinite(value).all()
            for value in (
                observation,
                latent,
                total_reward,
                values,
                term_reward,
                primary_per_env,
                *metric_per_env.values(),
                *reward_per_env.values(),
                *phase_per_env.values(),
            )
        ),
        "cpu_only": not torch.cuda.is_initialized(),
        "grad_disabled": not torch.is_grad_enabled(),
    }
    numerical_checks = {
        "historical_float32_failure_reproduced": (
            abs(float32["gradient_closure_relative_l2"] - 1.3954803919087401e-05)
            <= 1.0e-9
        ),
        "float64_gradient_closure": (
            float64["gradient_closure_relative_l2"] <= 1.0e-12
            and float64["gradient_closure_max_abs"] <= 1.0e-12
        ),
        "residual_credit_float32_closure": (
            residual_closure["gradient_relative_l2"] <= 1.0e-6
            and residual_closure["gradient_max_abs"] <= 1.0e-7
        ),
    }
    technical_checks = {**reconstruction_checks, **numerical_checks}
    reconstruction_valid = all(reconstruction_checks.values())
    numerical_valid = all(numerical_checks.values())
    pitch_reward_stable = alignment["pitch_reward_vs_positive_pitch"][
        "strict_direction_stable"
    ]
    primary_pitch_stable = alignment["primary_vs_positive_pitch"][
        "strict_direction_stable"
    ]
    pitch_conflict_hypothesis = (
        pitch_reward_stable
        and primary_pitch_stable
        and alignment["pitch_reward_vs_positive_pitch"]["full_projection"] > 0.0
        and alignment["primary_vs_positive_pitch"]["full_projection"] < 0.0
    )
    primary_support_bootstrap = alignment["primary_vs_lower_support_outside"][
        "full_bootstrap"
    ]
    support_inconclusive = (
        primary_support_bootstrap["bootstrap_p025"] <= 0.0
        <= primary_support_bootstrap["bootstrap_p975"]
    )
    groups = group_summary(primary_per_env)
    high_variance = groups["pairwise"]["first_vs_second_half"]["cosine"] < 0.90
    if not reconstruction_valid:
        decision = "POSTHOC_RECONSTRUCTION_INVALID_STOP"
    elif not numerical_valid:
        decision = "NUMERICAL_DECOMPOSITION_UNRESOLVED_STOP"
    elif high_variance or not primary_pitch_stable:
        decision = "POSTHOC_VARIANCE_INSTABILITY_CONFIRMED_NO_PROMOTION"
    elif pitch_conflict_hypothesis and support_inconclusive:
        decision = "POSTHOC_PITCH_CONFLICT_HYPOTHESIS_ONLY"
    else:
        decision = "POSTHOC_NUMERICAL_AUDIT_NO_PROMOTION"
    valid = reconstruction_valid and numerical_valid
    report = {
        "schema": "x2_phase71_posthoc_audit_v1",
        "decision": decision,
        "phase70_decision_unchanged": "FAIL_INVALID_STOP",
        "promotion": "none",
        "analysis_mode": "registered_posthoc_cpu_only_on_previously_inspected_data",
        "isaac_launches": 0,
        "physics_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "immutable_inputs": hashes,
        "technical_checks": technical_checks,
        "shape_checks": shape_checks,
        "reward_closure_max_abs": reward_closure_max,
        "stored_ascent_reconstruction": ascent_reconstruction,
        "phase_gradient_closure": phase_closure,
        "rng_hash_convention": rng,
        "float32_historical_path": {
            key: value for key, value in float32.items() if not isinstance(value, (torch.Tensor, dict))
        },
        "float64_reference": {
            key: value for key, value in float64.items() if not isinstance(value, (torch.Tensor, dict))
        },
        "residual_credit_sensitivity": residual_closure,
        "variance": groups,
        "phase_components_vs_primary": phase_summary,
        "reward_metric_alignment": alignment,
        "locomotion_term_directions": term_direction,
        "descriptive_hypotheses": {
            "full_sample_pitch_conflict_pattern": (
                alignment["pitch_reward_vs_positive_pitch"]["full_projection"] > 0.0
                and alignment["primary_vs_positive_pitch"]["full_projection"] < 0.0
            ),
            "pitch_reward_conflict": pitch_conflict_hypothesis,
            "support_total_direction_inconclusive": support_inconclusive,
            "environment_group_direction_high_variance": high_variance,
            "confirmatory_claim": False,
        },
        "resource": {
            "elapsed_seconds": time.perf_counter() - started,
            "disk_free_before_bytes": disk_before,
            "disk_free_after_bytes_before_outputs": shutil.disk_usage(ROOT).free,
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "boundaries": {
            "phase70_rerun_allowed": False,
            "optimizer_unlocked": False,
            "long_training_unlocked": False,
            "deployment_unlocked": False,
            "task2_complete": False,
            "future_confirmation_requires_new_independent_prereg_and_data": True,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    sidecar.write_text(f"{sha256(args.output)}  {args.output.name}\n")
    args.markdown.write_text(
        "# Phase71 CPU-only posthoc audit\n\n"
        f"Decision: `{decision}`. Phase70 remains `FAIL_INVALID_STOP`.\n\n"
        f"- CPU RNG raw/prefixed convention reproduced: {reconstruction_checks['rng_cpu_raw_reproduced'] and reconstruction_checks['rng_cpu_prefixed_reproduced']}.\n"
        f"- Float32 additive gradient relative-L2: `{float32['gradient_closure_relative_l2']:.9g}`.\n"
        f"- Float64 additive gradient relative-L2: `{float64['gradient_closure_relative_l2']:.9g}`.\n"
        f"- First-half/second-half cosine: `{groups['pairwise']['first_vs_second_half']['cosine']:.6f}`.\n"
        f"- Strict pitch conflict hypothesis: `{pitch_conflict_hypothesis}`; support direction inconclusive: `{support_inconclusive}`.\n\n"
        "This is registered posthoc analysis of already inspected data. It cannot promote, train, export, or deploy a policy.\n"
    )
    print(json.dumps({"decision": decision, "technical_checks": technical_checks}))
    if not valid:
        raise RuntimeError("Phase71 posthoc technical checks failed")


if __name__ == "__main__":
    main()
