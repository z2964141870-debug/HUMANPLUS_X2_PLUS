#!/usr/bin/env python3
"""Posthoc-only numerical calibration of the immutable Phase69 rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cwi_x2.phase68_residual_ppo import build_seeded_residual
from cwi_x2.phase69_reward_attribution import (
    analytical_head_ascent_per_env,
    bootstrap_projection,
    standalone_credit,
    vector_summary,
)


EXPECTED_BUNDLE_SHA = "db249ea619739c3128e2311039c65d26f19c61e7aa0f0451e9c8fc580c2c2f18"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rsl_lambda_one_advantage(
    reward: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float = 0.99,
    last_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match RSL RolloutStorage float32 recursion for lambda exactly one."""

    if reward.shape != values.shape or reward.ndim != 2:
        raise ValueError("reward and values must share [time, env]")
    next_value = (
        torch.zeros_like(values[0])
        if last_value is None
        else last_value.to(device=values.device, dtype=values.dtype)
    )
    advantage = torch.zeros_like(values[0])
    output = torch.zeros_like(values)
    for step in range(reward.shape[0] - 1, -1, -1):
        if step != reward.shape[0] - 1:
            next_value = values[step + 1]
        delta = reward[step] + gamma * next_value - values[step]
        advantage = delta + gamma * advantage
        output[step] = advantage
    return output


def normalize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean()) / (value.std() + 1.0e-8)


def metric_summary(vector: torch.Tensor, metric: torch.Tensor) -> dict[str, float]:
    """Name vector-summary fields for a metric-gradient reference."""

    raw = vector_summary(vector, metric)
    return {
        "norm": raw["norm"],
        "dot_with_metric_gradient": raw["dot_with_total"],
        "projection_on_metric_gradient": raw["projection_on_total"],
        "cosine_with_metric_gradient": raw["cosine_with_total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.markdown.exists():
        raise RuntimeError("refusing to overwrite immutable posthoc output")
    if sha256(args.bundle) != EXPECTED_BUNDLE_SHA:
        raise RuntimeError("Phase69 immutable bundle hash changed")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "x2_phase69_reward_attribution_evidence_v1":
        raise RuntimeError("Phase69 bundle schema changed")

    observation = bundle["policy_observation"]
    latent = bundle["latent_action"]
    total_reward = bundle["total_reward"]
    values = bundle["value"]
    term_reward = bundle["reward_by_term"]
    names = tuple(bundle["reward_term_names"])
    signed_pitch = bundle["signed_pitch_rad"]
    support_outside = bundle["support_outside_m"]
    terminal_value = bundle["terminal_bootstrap_value"]
    residual = build_seeded_residual().eval()
    with torch.no_grad():
        encoded = residual.encoder(observation.flatten(0, 1)).reshape(200, 64, -1)

    rsl_order_advantage = rsl_lambda_one_advantage(total_reward, values)
    rsl_order_normalized = normalize(rsl_order_advantage)
    bundle_normalized = bundle["normalized_total_advantage"]
    reward_closure = total_reward - term_reward.sum(dim=-1)

    component_reward = {
        "locomotion": term_reward[..., :21].sum(dim=-1),
        "pitch": term_reward[..., names.index("signed_backward_pitch")],
        "support": term_reward[..., names.index("actual_support_com")],
        "reward_closure_residual": reward_closure,
    }
    component_advantage = {
        name: rsl_lambda_one_advantage(
            reward,
            torch.zeros_like(values),
        )
        for name, reward in component_reward.items()
    }
    baseline_advantage = rsl_lambda_one_advantage(
        torch.zeros_like(total_reward), values
    )
    component_advantage["baseline"] = baseline_advantage
    denominator = rsl_order_advantage.std() + 1.0e-8
    component_credit = {
        name: (value - value.mean()) / denominator
        for name, value in component_advantage.items()
    }
    component_credit_sum = sum(component_credit.values())
    component_per_env = {
        name: analytical_head_ascent_per_env(encoded, latent, credit)
        for name, credit in component_credit.items()
    }
    total_per_env = analytical_head_ascent_per_env(encoded, latent, rsl_order_normalized)
    total = total_per_env.mean(dim=0)

    metric_per_env = {
        "signed_pitch": analytical_head_ascent_per_env(
            encoded, latent, standalone_credit(signed_pitch * 0.02)
        ),
        "negative_support_outside": analytical_head_ascent_per_env(
            encoded, latent, standalone_credit(-support_outside * 0.02)
        ),
    }
    standalone_reward_per_env = {
        "pitch": analytical_head_ascent_per_env(
            encoded, latent, standalone_credit(component_reward["pitch"])
        ),
        "support": analytical_head_ascent_per_env(
            encoded, latent, standalone_credit(component_reward["support"])
        ),
    }
    individual_term_per_env = {}
    for index, name in enumerate(names[:21]):
        individual_term_per_env[name] = analytical_head_ascent_per_env(
            encoded,
            latent,
            standalone_credit(term_reward[..., index]),
        )

    pitch_additive_projection = bootstrap_projection(
        component_per_env["pitch"], metric_per_env["signed_pitch"], seed=699001
    )
    support_additive_projection = bootstrap_projection(
        component_per_env["support"],
        metric_per_env["negative_support_outside"],
        seed=699002,
    )
    pitch_standalone_projection = bootstrap_projection(
        standalone_reward_per_env["pitch"],
        metric_per_env["signed_pitch"],
        seed=699003,
    )
    support_standalone_projection = bootstrap_projection(
        standalone_reward_per_env["support"],
        metric_per_env["negative_support_outside"],
        seed=699004,
    )
    term_pitch_alignment = {
        name: bootstrap_projection(value, metric_per_env["signed_pitch"], seed=699100 + index)
        for index, (name, value) in enumerate(individual_term_per_env.items())
    }

    terminal_mean = terminal_value.mean()
    centered_terminal_value = terminal_value - terminal_mean
    full_terminal_advantage = rsl_lambda_one_advantage(
        total_reward, values, last_value=terminal_value
    )
    centered_terminal_advantage = rsl_lambda_one_advantage(
        total_reward, values, last_value=centered_terminal_value
    )
    full_terminal_per_env = analytical_head_ascent_per_env(
        encoded, latent, normalize(full_terminal_advantage)
    )
    centered_terminal_per_env = analytical_head_ascent_per_env(
        encoded, latent, normalize(centered_terminal_advantage)
    )
    zero_full = vector_summary(full_terminal_per_env.mean(dim=0), total)
    zero_centered = vector_summary(centered_terminal_per_env.mean(dim=0), total)

    # Keep the Phase68 denominator fixed here so the terminal common-mode and
    # state-varying effects form an actually additive sensitivity decomposition.
    terminal_exponent = torch.arange(
        total_reward.shape[0], 0, -1, dtype=total_reward.dtype
    ).unsqueeze(1)
    terminal_discount = torch.pow(
        torch.tensor(0.99, dtype=total_reward.dtype), terminal_exponent
    )
    common_terminal_trace = (terminal_discount * terminal_mean).expand_as(total_reward)
    centered_terminal_trace = terminal_discount * centered_terminal_value.unsqueeze(0)
    common_terminal_credit = (
        common_terminal_trace - common_terminal_trace.mean()
    ) / denominator
    centered_terminal_credit = (
        centered_terminal_trace - centered_terminal_trace.mean()
    ) / denominator
    common_terminal_per_env = analytical_head_ascent_per_env(
        encoded, latent, common_terminal_credit
    )
    centered_terminal_delta_per_env = analytical_head_ascent_per_env(
        encoded, latent, centered_terminal_credit
    )
    fixed_scale_full_per_env = (
        total_per_env + common_terminal_per_env + centered_terminal_delta_per_env
    )
    fixed_scale_direct_per_env = analytical_head_ascent_per_env(
        encoded,
        latent,
        rsl_order_normalized + common_terminal_credit + centered_terminal_credit,
    )
    fixed_scale_error = fixed_scale_full_per_env - fixed_scale_direct_per_env

    result = {
        "schema": "x2_phase69_posthoc_calibration_v2",
        "decision": "POSTHOC_CALIBRATION_NO_PROMOTION",
        "bundle": {"path": str(args.bundle), "sha256": EXPECTED_BUNDLE_SHA},
        "numerical_reconstruction": {
            "rsl_order_cpu_vs_bundle_offline_normalized_advantage_max_abs": float(
                (rsl_order_normalized - bundle_normalized).abs().max()
            ),
            "bundle_does_not_contain_rsl_storage_advantage": True,
            "phase69_original_storage_validity_error_from_screen": 5.304813385009766e-06,
            "weighted_terms_vs_authoritative_reward_max_abs": float(
                reward_closure.abs().max()
            ),
            "additive_credit_vs_rsl_normalized_max_abs": float(
                (component_credit_sum - rsl_order_normalized).abs().max()
            ),
            "additive_credit_vs_rsl_normalized_relative_l2": float(
                torch.linalg.vector_norm(component_credit_sum - rsl_order_normalized)
                / torch.linalg.vector_norm(rsl_order_normalized)
            ),
        },
        "component_vs_phase68_total": {
            name: {
                **vector_summary(value.mean(dim=0), total),
                "bootstrap": bootstrap_projection(
                    value, total_per_env, seed=699010 + index
                ),
            }
            for index, (name, value) in enumerate(component_per_env.items())
        },
        "reward_metric_alignment": {
            "interpretation": (
                "Direction cosine is scale-free. Common-denominator projection "
                "retains the Phase68 advantage scale and measures a reward-gradient "
                "projection onto a metric gradient; "
                "standalone projection independently normalizes each reward and "
                "measures directional agreement only."
            ),
            "pitch_reward_vs_positive_signed_pitch": {
                "additive_common_scale_projection": pitch_additive_projection,
                "additive_common_scale_direction": metric_summary(
                    component_per_env["pitch"].mean(dim=0),
                    metric_per_env["signed_pitch"].mean(dim=0),
                ),
                "standalone_projection": pitch_standalone_projection,
                "standalone_direction": metric_summary(
                    standalone_reward_per_env["pitch"].mean(dim=0),
                    metric_per_env["signed_pitch"].mean(dim=0),
                ),
            },
            "support_reward_vs_negative_support_outside": {
                "additive_common_scale_projection": support_additive_projection,
                "additive_common_scale_direction": metric_summary(
                    component_per_env["support"].mean(dim=0),
                    metric_per_env["negative_support_outside"].mean(dim=0),
                ),
                "standalone_projection": support_standalone_projection,
                "standalone_direction": metric_summary(
                    standalone_reward_per_env["support"].mean(dim=0),
                    metric_per_env["negative_support_outside"].mean(dim=0),
                ),
            },
        },
        "locomotion_term_vs_positive_signed_pitch": term_pitch_alignment,
        "terminal_value_sensitivity": {
            "terminal_value_mean": float(terminal_mean),
            "terminal_value_std": float(terminal_value.std()),
            "full_terminal_direction_vs_zero": zero_full,
            "centered_terminal_direction_vs_zero": zero_centered,
            "fixed_phase68_denominator_additive_decomposition": {
                "zero": vector_summary(total, total),
                "empirical_common_mode_terminal_offset_delta": vector_summary(
                    common_terminal_per_env.mean(dim=0), total
                ),
                "state_varying_centered_terminal_delta": vector_summary(
                    centered_terminal_delta_per_env.mean(dim=0), total
                ),
                "full_sum": vector_summary(
                    fixed_scale_full_per_env.mean(dim=0), total
                ),
                "direct_full": vector_summary(
                    fixed_scale_direct_per_env.mean(dim=0), total
                ),
                "additive_closure_max_abs": float(fixed_scale_error.abs().max()),
                "additive_closure_relative_l2": float(
                    torch.linalg.vector_norm(fixed_scale_error)
                    / torch.linalg.vector_norm(fixed_scale_direct_per_env)
                ),
            },
            "interpretation": (
                "The full old-critic terminal value is not a matched total-reward "
                "bootstrap. Its empirical common-mode offset would have zero score "
                "gradient in expectation, but finite-sample leakage dominates this "
                "batch. The centered result only isolates the state-varying old-critic "
                "sensitivity; it is not a corrected total-reward estimator."
            ),
        },
        "boundary": {
            "phase69_decision_changed": False,
            "optimizer_unlocked": False,
            "phase70_execution_unlocked": False,
            "long_training_unlocked": False,
            "deployment_unlocked": False,
            "task2_complete": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n"
    )
    worst_terms = sorted(
        term_pitch_alignment.items(), key=lambda row: float(row[1]["point"])
    )[:5]
    args.markdown.write_text(
        "# X2 Phase69 posthoc calibration\n\n"
        "- Decision: `POSTHOC_CALIBRATION_NO_PROMOTION`\n"
        "- Phase69 remains `FAIL_ATTRIBUTION_INVALID_STOP`; the bundle does not contain the actual RSL storage advantage vector.\n"
        f"- RSL-order CPU recurrence vs saved offline advantage max error: `{result['numerical_reconstruction']['rsl_order_cpu_vs_bundle_offline_normalized_advantage_max_abs']:.3e}`\n"
        f"- Pitch reward → positive signed pitch direction cosine: `{result['reward_metric_alignment']['pitch_reward_vs_positive_signed_pitch']['standalone_direction']['cosine_with_metric_gradient']:.6f}`; standalone projection `{pitch_standalone_projection['point']:.6f}`; common-denominator reward-gradient projection `{pitch_additive_projection['point']:.6f}`\n"
        f"- Support reward → lower support outside direction cosine: `{result['reward_metric_alignment']['support_reward_vs_negative_support_outside']['standalone_direction']['cosine_with_metric_gradient']:.6f}`; standalone projection `{support_standalone_projection['point']:.6f}`; common-denominator reward-gradient projection `{support_additive_projection['point']:.6f}`\n"
        f"- Components projected on the Phase68 total: locomotion `{result['component_vs_phase68_total']['locomotion']['projection_on_total']:.4f}`, pitch `{result['component_vs_phase68_total']['pitch']['projection_on_total']:.4f}`, support `{result['component_vs_phase68_total']['support']['projection_on_total']:.4f}`; the formal Phase69 total → positive pitch projection remained negative.\n"
        f"- Full old-critic terminal vs zero direction cosine: `{zero_full['cosine_with_total']:.6f}`\n"
        f"- Mean-centered old-critic terminal vs zero direction cosine: `{zero_centered['cosine_with_total']:.6f}`\n"
        "- Most opposing independently normalized locomotion-term directions: "
        + ", ".join(f"`{name}` ({value['point']:.4f})" for name, value in worst_terms if value["point"] < 0.0)
        + "\n- Full old-critic sensitivity is retained for provenance; the centered sensitivity only isolates its state-varying part and is not a corrected estimator.\n"
        "- This posthoc analysis does not change Phase69 or unlock Phase70 execution or any optimizer.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
