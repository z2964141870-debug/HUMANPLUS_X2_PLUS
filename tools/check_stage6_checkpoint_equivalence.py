#!/usr/bin/env python3
"""Verify that Stage6 is an exact zero-residual extension of Stage208.

This check intentionally avoids Isaac Lab.  It validates the checkpoint and
policy contracts before an expensive simulator rollout is allowed to start.
"""

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source_state = payload["model_state_dict"]
    torch.manual_seed(args.seed)
    observations = {
        "policy": torch.zeros(
            args.batch_size,
            BASE_ACTOR_OBS_DIM + UPPER_INTENT_DIM,
        ),
        "critic": torch.zeros(args.batch_size, BASE_ACTOR_OBS_DIM),
    }
    model = FutureIntentActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_LOWER_ACTIONS,
        adapter_mode="future",
    )
    model.load_state_dict(source_state, strict=True)
    model.eval()

    actor_tensor_differences: dict[str, float] = {}
    for name, tensor in model.actor.state_dict().items():
        source_name = f"actor.{name}"
        actor_tensor_differences[source_name] = _max_abs_difference(
            tensor,
            source_state[source_name],
        )
    critic_tensor_differences: dict[str, float] = {}
    for name, tensor in model.critic.state_dict().items():
        source_name = f"critic.{name}"
        critic_tensor_differences[source_name] = _max_abs_difference(
            tensor,
            source_state[source_name],
        )

    base_observation = torch.randn(args.batch_size, BASE_ACTOR_OBS_DIM)
    arbitrary_intent = torch.randn(args.batch_size, UPPER_INTENT_DIM)
    zero_intent = torch.zeros_like(arbitrary_intent)
    with torch.inference_mode():
        source_action = model.actor(base_observation)
        zero_initialized_action = model._mean_from_actor_observation(
            torch.cat((base_observation, arbitrary_intent), dim=-1)
        )
        zero_intent_action = model._mean_from_actor_observation(
            torch.cat((base_observation, zero_intent), dim=-1)
        )

    adapter_parameters = list(model.coordination_adapter.parameters())
    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_names = [name for name, _ in trainable_parameters]
    allowed_trainable_prefixes = (
        "coordination_adapter.",
        "critic.",
        "std",
        "log_std",
    )
    unexpected_trainable_names = [
        name
        for name in trainable_names
        if not name.startswith(allowed_trainable_prefixes)
    ]
    actor_frozen = all(
        not parameter.requires_grad for parameter in model.actor.parameters()
    )
    adapter_last_layer_zero = bool(
        torch.count_nonzero(model.coordination_adapter[-1].weight) == 0
        and torch.count_nonzero(model.coordination_adapter[-1].bias) == 0
    )

    max_actor_tensor_difference = max(actor_tensor_differences.values())
    max_critic_tensor_difference = max(critic_tensor_differences.values())
    max_arbitrary_intent_action_difference = _max_abs_difference(
        source_action,
        zero_initialized_action,
    )
    max_zero_intent_action_difference = _max_abs_difference(
        source_action,
        zero_intent_action,
    )
    passed = (
        max_actor_tensor_difference == 0.0
        and max_critic_tensor_difference == 0.0
        and max_arbitrary_intent_action_difference == 0.0
        and max_zero_intent_action_difference == 0.0
        and actor_frozen
        and adapter_last_layer_zero
        and not unexpected_trainable_names
    )

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(payload.get("iter", -1)),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "contracts": {
            "base_actor_observation_dim": BASE_ACTOR_OBS_DIM,
            "upper_intent_dim": UPPER_INTENT_DIM,
            "actor_observation_dim": BASE_ACTOR_OBS_DIM + UPPER_INTENT_DIM,
            "num_lower_actions": NUM_LOWER_ACTIONS,
        },
        "parameter_counts": {
            "adapter_total": sum(
                parameter.numel() for parameter in adapter_parameters
            ),
            "model_trainable": sum(
                parameter.numel() for _, parameter in trainable_parameters
            ),
        },
        "trainable_parameter_names": trainable_names,
        "unexpected_trainable_parameter_names": unexpected_trainable_names,
        "actor_frozen": actor_frozen,
        "adapter_last_layer_zero": adapter_last_layer_zero,
        "max_actor_tensor_difference": max_actor_tensor_difference,
        "max_critic_tensor_difference": max_critic_tensor_difference,
        "max_arbitrary_intent_action_difference": (
            max_arbitrary_intent_action_difference
        ),
        "max_zero_intent_action_difference": max_zero_intent_action_difference,
        "actor_tensor_differences": actor_tensor_differences,
        "critic_tensor_differences": critic_tensor_differences,
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
        raise SystemExit("Stage6 checkpoint equivalence gate failed")


if __name__ == "__main__":
    main()
