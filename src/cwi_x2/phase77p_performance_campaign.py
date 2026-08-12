"""Core policy and PPO helpers for the outcome-first X2 posture campaign.

This module deliberately contains no IsaacLab imports.  It defines the
parameter-efficient actor, the fully-trainable critic contract, and the
source-policy KL anchor used by the Phase77p performance pilot.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from x2_upper_robust_lora_phase57 import ZeroOutputLoRALinear


ACTOR_LINEAR_INDICES = (0, 2, 4, 6)
CRITIC_LINEAR_INDICES = (0, 2, 4, 6)


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash a tensor with dtype and shape provenance."""

    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(f"{value.dtype}:{value.shape}".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_hash(module: nn.Module) -> str:
    """Hash a module state deterministically."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in eval mode and make all of its parameters immutable."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def inject_actor_lora_train_critic(
    policy: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 4.0,
) -> dict[str, object]:
    """Freeze the dense actor, add zero-output LoRA, and unfreeze the critic.

    The deployed action remains the original 15-dimensional actor output.  No
    residual action MDP is introduced.  Critic parameters are deliberately
    trained densely because the posture/support reward changes its target.
    """

    if rank <= 0 or alpha <= 0.0:
        raise ValueError("rank and alpha must be positive")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    actor_scopes: list[str] = []
    for index in ACTOR_LINEAR_INDICES:
        layer = policy.actor[index]
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"actor.{index} is not a Linear layer")
        policy.actor[index] = ZeroOutputLoRALinear(
            layer,
            rank=rank,
            alpha=alpha,
        )
        actor_scopes.append(f"actor.{index}")
    for index in CRITIC_LINEAR_INDICES:
        layer = policy.critic[index]
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"critic.{index} is not a Linear layer")
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    policy.std.requires_grad_(False)

    actor_names = sorted(
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and name.startswith("actor.")
    )
    critic_names = sorted(
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and name.startswith("critic.")
    )
    expected_actor = sorted(
        f"actor.{index}.lora_{suffix}"
        for index in ACTOR_LINEAR_INDICES
        for suffix in ("A", "B")
    )
    expected_critic = sorted(
        f"critic.{index}.{suffix}"
        for index in CRITIC_LINEAR_INDICES
        for suffix in ("weight", "bias")
    )
    if actor_names != expected_actor:
        raise RuntimeError(f"actor trainable scope changed: {actor_names}")
    if critic_names != expected_critic:
        raise RuntimeError(f"critic trainable scope changed: {critic_names}")
    trainable = actor_names + critic_names
    frozen = sorted(
        name for name, parameter in policy.named_parameters() if not parameter.requires_grad
    )
    return {
        "rank": rank,
        "alpha": alpha,
        "actor_scopes": actor_scopes,
        "actor_trainable_names": actor_names,
        "critic_trainable_names": critic_names,
        "trainable_names": sorted(trainable),
        "frozen_names": frozen,
        "actor_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if name in actor_names
        ),
        "critic_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if name in critic_names
        ),
    }


def trainable_parameter_groups(
    policy: nn.Module,
    *,
    actor_lr: float,
    critic_lr: float,
) -> list[dict[str, object]]:
    """Return disjoint Adam groups for actor LoRA and dense critic tensors."""

    if not (math.isfinite(actor_lr) and actor_lr > 0.0):
        raise ValueError("actor_lr must be finite and positive")
    if not (math.isfinite(critic_lr) and critic_lr > 0.0):
        raise ValueError("critic_lr must be finite and positive")
    actor = [
        parameter
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and name.startswith("actor.")
    ]
    critic = [
        parameter
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and name.startswith("critic.")
    ]
    if not actor or not critic:
        raise RuntimeError("actor and critic trainable groups must both be non-empty")
    if len({id(parameter) for parameter in actor + critic}) != len(actor) + len(critic):
        raise RuntimeError("optimizer parameter groups overlap")
    return [
        {"params": actor, "lr": float(actor_lr), "name": "actor_lora"},
        {"params": critic, "lr": float(critic_lr), "name": "critic_dense"},
    ]


def diagonal_fixed_std_kl(
    mean: torch.Tensor,
    source_mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """KL(current || source) for equal diagonal Gaussian standard deviations."""

    if mean.shape != source_mean.shape:
        raise ValueError("mean tensors must have identical shapes")
    if std.ndim == 1:
        std = std.reshape(*([1] * (mean.ndim - 1)), -1)
    if std.shape[-1] != mean.shape[-1] or torch.any(std <= 0):
        raise ValueError("std must be positive and match the action width")
    return 0.5 * torch.square((mean - source_mean) / std).sum(dim=-1)


def posture_reward_weight(update_index: int) -> float:
    """Return the frozen pilot posture/support weight for updates 1..10.

    The performance pilot deliberately does not ramp this coefficient.  A
    reward change at the segment-5 restart would otherwise be inseparable from
    a process restart and command-cycle restart.
    """

    if update_index < 1 or update_index > 10:
        raise ValueError("update_index must lie in [1, 10]")
    return -0.5


@dataclass(frozen=True)
class AnchoredUpdateLimits:
    source_kl_mean_max: float = 0.005
    source_kl_max: float = 0.02
    incremental_kl_mean_max: float = 0.0025
    incremental_kl_max: float = 0.01
    action_drift_max: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.source_kl_mean_max,
            self.source_kl_max,
            self.incremental_kl_mean_max,
            self.incremental_kl_max,
            self.action_drift_max,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("anchored update limits must be finite and positive")


def _all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def anchored_ppo_update(
    algorithm,
    source_policy: nn.Module,
    *,
    source_kl_coefficient: float,
    limits: AnchoredUpdateLimits = AnchoredUpdateLimits(),
) -> dict[str, float | int | bool]:
    """Run one PPO update with an explicit immutable source-actor KL anchor.

    This follows the installed RSL-RL PPO numerical path for the non-recurrent,
    non-RND, non-symmetry case.  The source actor is evaluated deterministically
    on every minibatch.  Its parameters are never part of the optimizer.
    """

    if not math.isfinite(source_kl_coefficient) or source_kl_coefficient <= 0.0:
        raise ValueError("source_kl_coefficient must be finite and positive")
    if algorithm.policy.is_recurrent:
        raise RuntimeError("Phase77p supports only a feed-forward policy")
    if getattr(algorithm, "rnd", None) is not None:
        raise RuntimeError("Phase77p does not support RND")
    if getattr(algorithm, "symmetry", None) is not None:
        raise RuntimeError("Phase77p does not support RSL symmetry augmentation")

    totals = {
        "value_function": 0.0,
        "surrogate": 0.0,
        "entropy": 0.0,
        "source_kl": 0.0,
        "incremental_kl": 0.0,
    }
    source_kl_peak = 0.0
    incremental_kl_peak = 0.0
    action_drift_peak = 0.0
    optimizer_steps = 0
    generator = algorithm.storage.mini_batch_generator(
        algorithm.num_mini_batches,
        algorithm.num_learning_epochs,
    )
    for (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
    ) in generator:
        if algorithm.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages_batch = (
                    advantages_batch - advantages_batch.mean()
                ) / (advantages_batch.std() + 1.0e-8)

        algorithm.policy.act(
            obs_batch,
            masks=masks_batch,
            hidden_states=hid_states_batch[0],
        )
        actions_log_prob_batch = algorithm.policy.get_actions_log_prob(actions_batch)
        value_batch = algorithm.policy.evaluate(
            obs_batch,
            masks=masks_batch,
            hidden_states=hid_states_batch[1],
        )
        mu_batch = algorithm.policy.action_mean
        sigma_batch = algorithm.policy.action_std
        entropy_batch = algorithm.policy.entropy
        with torch.no_grad():
            source_mean = source_policy.act_inference(obs_batch)

        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
        surrogate = -advantages_batch.squeeze(-1) * ratio
        surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
            ratio,
            1.0 - algorithm.clip_param,
            1.0 + algorithm.clip_param,
        )
        surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()

        if algorithm.use_clipped_value_loss:
            value_clipped = target_values_batch + (
                value_batch - target_values_batch
            ).clamp(-algorithm.clip_param, algorithm.clip_param)
            value_losses = torch.square(value_batch - returns_batch)
            value_losses_clipped = torch.square(value_clipped - returns_batch)
            value_loss = torch.maximum(value_losses, value_losses_clipped).mean()
        else:
            value_loss = torch.square(returns_batch - value_batch).mean()

        source_kl_pre = diagonal_fixed_std_kl(
            mu_batch,
            source_mean,
            source_policy.std.detach(),
        )
        loss = (
            surrogate_loss
            + algorithm.value_loss_coef * value_loss
            - algorithm.entropy_coef * entropy_batch.mean()
            + source_kl_coefficient * source_kl_pre.mean()
        )
        if not _all_finite(
            (loss, value_loss, surrogate_loss, source_kl_pre)
        ):
            raise FloatingPointError("non-finite Phase77p PPO loss or KL")

        trainable = [
            parameter
            for parameter in algorithm.policy.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise FloatingPointError("Phase77p has no trainable parameters")
        parameter_snapshot = [parameter.detach().clone() for parameter in trainable]
        optimizer_snapshot = copy.deepcopy(algorithm.optimizer.state_dict())
        algorithm.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in trainable]
        if any(gradient is None for gradient in gradients) or not _all_finite(
            gradient for gradient in gradients if gradient is not None
        ):
            raise FloatingPointError("missing or non-finite Phase77p gradient")
        nn.utils.clip_grad_norm_(trainable, algorithm.max_grad_norm)
        algorithm.optimizer.step()

        # The trust region is a post-step acceptance rule.  A pre-step KL only
        # regularizes the loss; it cannot prevent the Adam step from leaving
        # the admissible region.
        with torch.no_grad():
            if hasattr(algorithm.policy, "get_actor_obs"):
                actor_obs = algorithm.policy.get_actor_obs(obs_batch)
                actor_obs = algorithm.policy.actor_obs_normalizer(actor_obs)
            else:
                actor_obs = obs_batch
            algorithm.policy.update_distribution(actor_obs)
            post_mu = algorithm.policy.action_mean
            post_sigma = algorithm.policy.action_std
            post_source_mean = source_policy.act_inference(obs_batch)
            source_kl = diagonal_fixed_std_kl(
                post_mu,
                post_source_mean,
                source_policy.std.detach(),
            )
            incremental_kl = torch.sum(
                torch.log(post_sigma / old_sigma_batch)
                + (
                    torch.square(old_sigma_batch)
                    + torch.square(old_mu_batch - post_mu)
                )
                / (2.0 * torch.square(post_sigma))
                - 0.5,
                dim=-1,
            )
            action_drift = torch.abs(post_mu - post_source_mean)
            accepted = bool(
                _all_finite((source_kl, incremental_kl, action_drift))
                and float(source_kl.mean()) <= limits.source_kl_mean_max
                and float(source_kl.max()) <= limits.source_kl_max
                and float(incremental_kl.mean()) <= limits.incremental_kl_mean_max
                and float(incremental_kl.max()) <= limits.incremental_kl_max
                and float(action_drift.max()) <= limits.action_drift_max
            )
        if not accepted:
            with torch.no_grad():
                for parameter, snapshot in zip(
                    trainable, parameter_snapshot, strict=True
                ):
                    parameter.copy_(snapshot)
            algorithm.optimizer.load_state_dict(optimizer_snapshot)
            raise RuntimeError(
                "Phase77p minibatch trust-region violation; Adam step rolled back: "
                f"source_kl_mean={float(source_kl.mean()):.9g}, "
                f"source_kl_max={float(source_kl.max()):.9g}, "
                f"incremental_kl_mean={float(incremental_kl.mean()):.9g}, "
                f"incremental_kl_max={float(incremental_kl.max()):.9g}, "
                f"action_drift_max={float(action_drift.max()):.9g}"
            )
        optimizer_steps += 1

        totals["value_function"] += float(value_loss.detach())
        totals["surrogate"] += float(surrogate_loss.detach())
        totals["entropy"] += float(entropy_batch.mean().detach())
        totals["source_kl"] += float(source_kl.mean().detach())
        totals["incremental_kl"] += float(incremental_kl.mean().detach())
        source_kl_peak = max(source_kl_peak, float(source_kl.max().detach()))
        incremental_kl_peak = max(
            incremental_kl_peak,
            float(incremental_kl.max().detach()),
        )
        action_drift_peak = max(
            action_drift_peak,
            float(action_drift.max().detach()),
        )

    expected_steps = algorithm.num_learning_epochs * algorithm.num_mini_batches
    if optimizer_steps != expected_steps:
        raise RuntimeError(
            f"expected {expected_steps} optimizer steps, observed {optimizer_steps}"
        )
    algorithm.storage.clear()
    for name in totals:
        totals[name] /= optimizer_steps
    return {
        **totals,
        "source_kl_max": source_kl_peak,
        "incremental_kl_max": incremental_kl_peak,
        "action_drift_max": action_drift_peak,
        "optimizer_steps": optimizer_steps,
        "finite": all(math.isfinite(value) for value in totals.values()),
    }


def source_retention(
    candidate: nn.Module,
    source: nn.Module,
    observations,
) -> dict[str, float | bool]:
    """Evaluate deterministic actor retention on a frozen observation batch."""

    with torch.no_grad():
        candidate_action = candidate.act_inference(observations)
        source_action = source.act_inference(observations)
        kl = diagonal_fixed_std_kl(
            candidate_action,
            source_action,
            source.std.detach(),
        )
        drift = torch.abs(candidate_action - source_action)
    return {
        "source_kl_mean": float(kl.mean()),
        "source_kl_max": float(kl.max()),
        "action_drift_max": float(drift.max()),
        "finite": bool(torch.isfinite(kl).all() and torch.isfinite(drift).all()),
    }


def clone_frozen_source(policy: nn.Module) -> nn.Module:
    """Create the immutable actor/critic reference used for KL anchoring."""

    return freeze_module(copy.deepcopy(policy))


__all__ = [
    "AnchoredUpdateLimits",
    "anchored_ppo_update",
    "clone_frozen_source",
    "diagonal_fixed_std_kl",
    "inject_actor_lora_train_critic",
    "posture_reward_weight",
    "source_retention",
    "state_hash",
    "tensor_hash",
    "trainable_parameter_groups",
]
