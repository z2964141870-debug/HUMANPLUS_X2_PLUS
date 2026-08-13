"""Pure training contracts for the next X2 native full-actor pilot.

This module intentionally has no IsaacLab dependency.  It defines the parts
of the pilot that must remain testable before a simulator is launched:

* the eight balanced command roles;
* the posture/support curriculum;
* full-actor plus freshly initialized critic parameter ownership; and
* fixed, disjoint optimizer groups.

The source actor is a warm start, not a frozen base or an additive residual.
The source critic is never retained because its value target predates the new
posture and support objectives.
"""

from __future__ import annotations

import hashlib
import math
import copy
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from torch import nn


ROLE_NAMES = (
    "straight_vx_0p20",
    "straight_vx_0p35",
    "straight_vx_0p50",
    "turn_left_vx_0p35_wz_0p15",
    "turn_right_vx_0p35_wz_0p15",
    "decelerate_from_vx_0p20",
    "decelerate_from_vx_0p35",
    "decelerate_from_vx_0p50",
)


@dataclass(frozen=True)
class PilotSpec:
    """Frozen numerical contract for the capability pilot."""

    num_envs: int = 512
    steps_per_env: int = 48
    updates: int = 25
    learning_epochs: int = 2
    mini_batches: int = 4
    actor_learning_rate: float = 3.0e-5
    critic_learning_rate: float = 2.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    max_grad_norm: float = 1.0
    source_kl_mean_max: float = 0.02
    source_kl_max: float = 0.20
    action_drift_max: float = 0.10
    template_scale: float = 0.15

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.num_envs % len(ROLE_NAMES) != 0:
            raise ValueError("num_envs must be positive and divisible by eight roles")
        if self.steps_per_env <= 0 or self.updates <= 0:
            raise ValueError("rollout dimensions must be positive")
        if self.learning_epochs <= 0 or self.mini_batches <= 0:
            raise ValueError("PPO epoch/minibatch counts must be positive")
        positive = (
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.gamma,
            self.lam,
            self.clip_param,
            self.max_grad_norm,
            self.source_kl_mean_max,
            self.source_kl_max,
            self.action_drift_max,
            self.template_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("pilot scalar parameters must be finite and positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.lam <= 1.0:
            raise ValueError("gamma and lambda must lie in (0, 1]")
        if self.source_kl_mean_max > self.source_kl_max:
            raise ValueError("mean KL gate cannot exceed the maximum KL gate")

    @property
    def transitions(self) -> int:
        return self.num_envs * self.steps_per_env * self.updates

    @property
    def optimizer_steps(self) -> int:
        return self.updates * self.learning_epochs * self.mini_batches

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveWeights:
    """Additional posture objectives for an update range.

    Existing Stage219 locomotion terms remain present.  These two weights are
    negative because the corresponding reward terms are non-negative costs.
    """

    signed_backward_pitch: float
    actual_support_com: float


@dataclass(frozen=True)
class FullActorUpdateLimits:
    """Post-minibatch acceptance limits for the warm-started full actor."""

    source_kl_mean_max: float = 0.02
    source_kl_max: float = 0.20
    incremental_kl_mean_max: float = 0.01
    incremental_kl_max: float = 0.10
    action_drift_max: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.source_kl_mean_max,
            self.source_kl_max,
            self.incremental_kl_mean_max,
            self.incremental_kl_max,
            self.action_drift_max,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("update limits must be finite and positive")
        if self.source_kl_mean_max > self.source_kl_max:
            raise ValueError("source mean KL cannot exceed source max KL")
        if self.incremental_kl_mean_max > self.incremental_kl_max:
            raise ValueError("incremental mean KL cannot exceed incremental max KL")


def objective_weights(update_index: int) -> ObjectiveWeights:
    """Return the preregistered three-stage posture/support curriculum."""

    if not 1 <= update_index <= 25:
        raise ValueError("update_index must lie in [1, 25]")
    if update_index <= 5:
        magnitude = 0.5
    elif update_index <= 15:
        magnitude = 1.0
    else:
        magnitude = 1.5
    return ObjectiveWeights(-magnitude, -magnitude)


def role_ids(num_envs: int, *, paired: bool = False) -> torch.Tensor:
    """Return balanced, scene-index-interleaved command roles.

    Training assigns roles by ``env_id % 8``.  Paired evaluation assigns the
    same role to adjacent A/B lanes with ``(env_id // 2) % 8``.
    """

    divisor = 2 * len(ROLE_NAMES) if paired else len(ROLE_NAMES)
    if num_envs <= 0 or num_envs % divisor != 0:
        raise ValueError(f"num_envs must be positive and divisible by {divisor}")
    ids = torch.arange(num_envs, dtype=torch.long)
    if paired:
        ids = torch.div(ids, 2, rounding_mode="floor")
    return ids.remainder(len(ROLE_NAMES))


def role_count_dict(num_envs: int, *, paired: bool = False) -> dict[str, int]:
    ids = role_ids(num_envs, paired=paired)
    return {
        name: int((ids == index).sum())
        for index, name in enumerate(ROLE_NAMES)
    }


def module_state_hash(module: nn.Module) -> str:
    """Hash a module state including names, dtype and shape."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not layers:
        raise RuntimeError("network contains no Linear layers")
    return layers


def _reset_critic(critic: nn.Module, seed: int) -> None:
    """Deterministically reset all critic Linear layers without actor drift."""

    if seed < 0:
        raise ValueError("critic seed must be non-negative")
    devices = sorted(
        {
            parameter.device.index
            for parameter in critic.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        for layer in _linear_layers(critic):
            layer.reset_parameters()


def configure_full_actor_fresh_critic(
    policy: nn.Module,
    *,
    critic_seed: int,
) -> dict[str, object]:
    """Open the full actor, reset the critic, and freeze exploration std."""

    for required in ("actor", "critic", "std"):
        if not hasattr(policy, required):
            raise TypeError(f"policy is missing {required!r}")
    actor_before = module_state_hash(policy.actor)
    critic_before = module_state_hash(policy.critic)
    source_std = policy.std.detach().clone()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(True)
    _reset_critic(policy.critic, critic_seed)
    for parameter in policy.critic.parameters():
        parameter.requires_grad_(True)
    policy.std.requires_grad_(False)

    actor_after = module_state_hash(policy.actor)
    critic_after = module_state_hash(policy.critic)
    if actor_before != actor_after:
        raise RuntimeError("warm-start actor changed while resetting the critic")
    if critic_before == critic_after:
        raise RuntimeError("critic reset did not change its state")
    if not torch.equal(policy.std.detach(), source_std):
        raise RuntimeError("exploration std changed during configuration")

    trainable = sorted(
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    )
    expected = sorted(
        [f"actor.{name}" for name, _ in policy.actor.named_parameters()]
        + [f"critic.{name}" for name, _ in policy.critic.named_parameters()]
    )
    if trainable != expected:
        raise RuntimeError(f"unexpected trainable parameter set: {trainable}")
    return {
        "initialization": "stage219_actor_warm_start_fresh_critic",
        "critic_seed": int(critic_seed),
        "actor_state_hash": actor_after,
        "source_critic_state_hash": critic_before,
        "fresh_critic_state_hash": critic_after,
        "std_frozen": True,
        "std_min": float(source_std.min()),
        "std_max": float(source_std.max()),
        "trainable_names": trainable,
        "actor_parameters": sum(parameter.numel() for parameter in policy.actor.parameters()),
        "critic_parameters": sum(parameter.numel() for parameter in policy.critic.parameters()),
    }


def optimizer_parameter_groups(
    policy: nn.Module,
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> list[dict[str, object]]:
    """Return disjoint fixed-rate parameter groups for actor and critic."""

    for value in (actor_learning_rate, critic_learning_rate):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("learning rates must be finite and positive")
    actor = [parameter for parameter in policy.actor.parameters() if parameter.requires_grad]
    critic = [parameter for parameter in policy.critic.parameters() if parameter.requires_grad]
    if not actor or not critic:
        raise RuntimeError("actor and critic parameter groups must be non-empty")
    identifiers = [id(parameter) for parameter in actor + critic]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("actor and critic optimizer groups overlap")
    return [
        {"params": actor, "lr": float(actor_learning_rate), "name": "actor_full"},
        {"params": critic, "lr": float(critic_learning_rate), "name": "critic_fresh"},
    ]


def diagonal_fixed_std_kl(
    mean: torch.Tensor,
    source_mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """KL between equal-diagonal-std Gaussians."""

    if mean.shape != source_mean.shape:
        raise ValueError("candidate and source means must have identical shapes")
    if std.ndim == 1:
        std = std.reshape(*([1] * (mean.ndim - 1)), -1)
    if std.shape[-1] != mean.shape[-1] or torch.any(std <= 0.0):
        raise ValueError("std must be positive and match the action width")
    return 0.5 * torch.square((mean - source_mean) / std).sum(dim=-1)


def _all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def source_retention(
    candidate: nn.Module,
    source: nn.Module,
    observations,
) -> dict[str, float | bool]:
    """Measure full-actor drift without consulting either critic."""

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


def full_actor_ppo_update(
    algorithm,
    source_policy: nn.Module,
    *,
    limits: FullActorUpdateLimits = FullActorUpdateLimits(),
) -> dict[str, float | int | bool]:
    """Run one fixed-rate PPO update with post-step rollback gates.

    The PPO objective contains no source KL penalty.  PPO clipping supplies
    the local optimizer geometry; the immutable source comparison is an
    acceptance gate applied after every Adam step.  This leaves substantially
    more capacity than the failed LoRA campaign while remaining fail-closed.
    """

    if algorithm.policy.is_recurrent:
        raise RuntimeError("the full-actor pilot supports only feed-forward policies")
    if getattr(algorithm, "rnd", None) is not None:
        raise RuntimeError("the full-actor pilot does not support RND")
    if getattr(algorithm, "symmetry", None) is not None:
        raise RuntimeError("the full-actor pilot does not use RSL symmetry")
    if getattr(algorithm, "schedule", "fixed") != "fixed":
        raise RuntimeError("separate actor/critic rates require a fixed PPO schedule")

    totals = {"value_function": 0.0, "surrogate": 0.0, "entropy": 0.0}
    source_kl_total = 0.0
    incremental_kl_total = 0.0
    source_kl_peak = 0.0
    incremental_kl_peak = 0.0
    action_drift_peak = 0.0
    gradient_norm_peak = 0.0
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
        hidden_states_batch,
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
            hidden_states=hidden_states_batch[0],
        )
        log_prob = algorithm.policy.get_actions_log_prob(actions_batch)
        value = algorithm.policy.evaluate(
            obs_batch,
            masks=masks_batch,
            hidden_states=hidden_states_batch[1],
        )
        entropy = algorithm.policy.entropy
        ratio = torch.exp(log_prob - old_actions_log_prob_batch.squeeze(-1))
        surrogate = -advantages_batch.squeeze(-1) * ratio
        surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
            ratio,
            1.0 - algorithm.clip_param,
            1.0 + algorithm.clip_param,
        )
        surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
        if algorithm.use_clipped_value_loss:
            value_clipped = target_values_batch + (value - target_values_batch).clamp(
                -algorithm.clip_param,
                algorithm.clip_param,
            )
            value_loss = torch.maximum(
                torch.square(value - returns_batch),
                torch.square(value_clipped - returns_batch),
            ).mean()
        else:
            value_loss = torch.square(returns_batch - value).mean()
        loss = (
            surrogate_loss
            + algorithm.value_loss_coef * value_loss
            - algorithm.entropy_coef * entropy.mean()
        )
        if not _all_finite((loss, surrogate_loss, value_loss, ratio)):
            raise FloatingPointError("non-finite full-actor PPO loss")

        trainable = [
            parameter for parameter in algorithm.policy.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("the full-actor policy has no trainable parameters")
        parameter_snapshot = [parameter.detach().clone() for parameter in trainable]
        optimizer_snapshot = copy.deepcopy(algorithm.optimizer.state_dict())
        algorithm.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in trainable]
        if any(gradient is None for gradient in gradients) or not _all_finite(
            gradient for gradient in gradients if gradient is not None
        ):
            raise FloatingPointError("missing or non-finite full-actor gradient")
        gradient_norm = nn.utils.clip_grad_norm_(trainable, algorithm.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite full-actor gradient norm")
        algorithm.optimizer.step()

        with torch.no_grad():
            if hasattr(algorithm.policy, "get_actor_obs"):
                actor_obs = algorithm.policy.get_actor_obs(obs_batch)
                actor_obs = algorithm.policy.actor_obs_normalizer(actor_obs)
            else:
                actor_obs = obs_batch
            algorithm.policy.update_distribution(actor_obs)
            post_mean = algorithm.policy.action_mean
            post_std = algorithm.policy.action_std
            source_mean = source_policy.act_inference(obs_batch)
            source_kl = diagonal_fixed_std_kl(
                post_mean,
                source_mean,
                source_policy.std.detach(),
            )
            incremental_kl = torch.sum(
                torch.log(post_std / old_sigma_batch)
                + (
                    torch.square(old_sigma_batch)
                    + torch.square(old_mu_batch - post_mean)
                )
                / (2.0 * torch.square(post_std))
                - 0.5,
                dim=-1,
            )
            drift = torch.abs(post_mean - source_mean)
            accepted = bool(
                _all_finite((source_kl, incremental_kl, drift))
                and float(source_kl.mean()) <= limits.source_kl_mean_max
                and float(source_kl.max()) <= limits.source_kl_max
                and float(incremental_kl.mean()) <= limits.incremental_kl_mean_max
                and float(incremental_kl.max()) <= limits.incremental_kl_max
                and float(drift.max()) <= limits.action_drift_max
            )
        if not accepted:
            with torch.no_grad():
                for parameter, snapshot in zip(trainable, parameter_snapshot, strict=True):
                    parameter.copy_(snapshot)
            algorithm.optimizer.load_state_dict(optimizer_snapshot)
            raise RuntimeError(
                "full-actor trust gate failed; minibatch was rolled back: "
                f"source_kl_mean={float(source_kl.mean()):.9g}, "
                f"source_kl_max={float(source_kl.max()):.9g}, "
                f"incremental_kl_mean={float(incremental_kl.mean()):.9g}, "
                f"incremental_kl_max={float(incremental_kl.max()):.9g}, "
                f"action_drift_max={float(drift.max()):.9g}"
            )

        optimizer_steps += 1
        totals["value_function"] += float(value_loss.detach())
        totals["surrogate"] += float(surrogate_loss.detach())
        totals["entropy"] += float(entropy.mean().detach())
        source_kl_total += float(source_kl.mean())
        incremental_kl_total += float(incremental_kl.mean())
        source_kl_peak = max(source_kl_peak, float(source_kl.max()))
        incremental_kl_peak = max(incremental_kl_peak, float(incremental_kl.max()))
        action_drift_peak = max(action_drift_peak, float(drift.max()))
        gradient_norm_peak = max(gradient_norm_peak, float(gradient_norm))

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
        "source_kl_mean": source_kl_total / optimizer_steps,
        "source_kl_max": source_kl_peak,
        "incremental_kl_mean": incremental_kl_total / optimizer_steps,
        "incremental_kl_max": incremental_kl_peak,
        "action_drift_max": action_drift_peak,
        "gradient_norm_max": gradient_norm_peak,
        "optimizer_steps": optimizer_steps,
        "finite": all(math.isfinite(value) for value in totals.values()),
    }


__all__ = [
    "FullActorUpdateLimits",
    "ObjectiveWeights",
    "PilotSpec",
    "ROLE_NAMES",
    "configure_full_actor_fresh_critic",
    "diagonal_fixed_std_kl",
    "full_actor_ppo_update",
    "module_state_hash",
    "objective_weights",
    "optimizer_parameter_groups",
    "role_count_dict",
    "role_ids",
    "source_retention",
]
