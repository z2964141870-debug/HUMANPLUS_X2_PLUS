#!/usr/bin/env python3
"""Audit a trained Stage6 checkpoint against its frozen Stage208 source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cwi_x2.future_intent_actor_critic import (
    BASE_ACTOR_OBS_DIM,
    NUM_LOWER_ACTIONS,
    UPPER_INTENT_DIM,
    FutureIntentActorCritic,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-mode", type=str, required=True)
    parser.add_argument("--coordination-blend", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    return parser.parse_args()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def main() -> None:
    args = _parse_args()
    if args.probe_batch_size <= 0:
        raise ValueError("--probe-batch-size must be positive")
    source_path = args.source_checkpoint.expanduser().resolve()
    trained_path = args.trained_checkpoint.expanduser().resolve()
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    trained_payload = torch.load(
        trained_path,
        map_location="cpu",
        weights_only=False,
    )
    source_state = source_payload["model_state_dict"]
    trained_state = trained_payload["model_state_dict"]

    observations = {
        "policy": torch.zeros(1, BASE_ACTOR_OBS_DIM + UPPER_INTENT_DIM),
        "critic": torch.zeros(1, BASE_ACTOR_OBS_DIM),
    }
    model = FutureIntentActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_LOWER_ACTIONS,
        adapter_mode=args.adapter_mode,
        coordination_blend=args.coordination_blend,
    )
    model.load_state_dict(trained_state, strict=True)
    model.eval()

    actor_differences = {
        f"actor.{name}": _max_abs(
            tensor,
            source_state[f"actor.{name}"],
        )
        for name, tensor in model.actor.state_dict().items()
    }
    critic_differences = {
        f"critic.{name}": _max_abs(
            tensor,
            source_state[f"critic.{name}"],
        )
        for name, tensor in model.critic.state_dict().items()
    }
    finite_tensors = {
        name: bool(torch.isfinite(tensor).all())
        for name, tensor in trained_state.items()
    }

    torch.manual_seed(args.seed)
    base_observation = torch.randn(
        args.probe_batch_size,
        BASE_ACTOR_OBS_DIM,
    )
    intent = 0.12 * torch.rand(
        args.probe_batch_size,
        UPPER_INTENT_DIM,
    ) - 0.06
    with torch.inference_mode():
        model._mean_from_actor_observation(
            torch.cat((base_observation, intent), dim=-1)
        )
    residual = model._last_coordination_residual
    if residual is None:
        raise RuntimeError("Stage6 model did not expose its coordination residual")
    masked_indices = [3, 4, 5, 9, 10, 11, 13]
    adapter_state = {
        name: tensor
        for name, tensor in trained_state.items()
        if name.startswith("coordination_adapter.")
    }
    adapter_parameter_l2 = float(
        torch.sqrt(
            sum(tensor.square().sum() for tensor in adapter_state.values())
        ).item()
    )
    adapter_parameter_abs_max = max(
        float(tensor.abs().max().item()) for tensor in adapter_state.values()
    )
    max_actor_difference = max(actor_differences.values())
    max_critic_difference = max(critic_differences.values())
    residual_abs_max = float(residual.abs().max().item())
    masked_residual_abs_max = float(
        residual[:, masked_indices].abs().max().item()
    )
    all_finite = all(finite_tensors.values())
    actor_frozen = all(
        not parameter.requires_grad for parameter in model.actor.parameters()
    )
    passed = (
        max_actor_difference == 0.0
        and all_finite
        and actor_frozen
        and residual_abs_max
        <= model.coordination_output_scale * model.coordination_blend + 1.0e-7
        and masked_residual_abs_max == 0.0
        and bool(adapter_state)
    )
    report = {
        "schema_version": 1,
        "source_checkpoint": str(source_path),
        "trained_checkpoint": str(trained_path),
        "source_iteration": int(source_payload.get("iter", -1)),
        "trained_iteration": int(trained_payload.get("iter", -1)),
        "adapter_mode": args.adapter_mode,
        "coordination_blend": model.coordination_blend,
        "seed": args.seed,
        "probe_batch_size": args.probe_batch_size,
        "max_frozen_actor_tensor_difference": max_actor_difference,
        "max_critic_tensor_difference": max_critic_difference,
        "all_checkpoint_tensors_finite": all_finite,
        "nonfinite_tensor_names": [
            name for name, finite in finite_tensors.items() if not finite
        ],
        "actor_frozen_after_reload": actor_frozen,
        "adapter_tensor_count": len(adapter_state),
        "adapter_parameter_l2": adapter_parameter_l2,
        "adapter_parameter_abs_max": adapter_parameter_abs_max,
        "probe_residual_abs_max": residual_abs_max,
        "residual_budget": (
            model.coordination_output_scale * model.coordination_blend
        ),
        "masked_residual_abs_max": masked_residual_abs_max,
        "actor_tensor_differences": actor_differences,
        "critic_tensor_differences": critic_differences,
        "passed": passed,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("Stage6 checkpoint audit failed")


if __name__ == "__main__":
    main()
