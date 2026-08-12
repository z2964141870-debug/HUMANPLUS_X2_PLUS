"""Pure reward-gradient attribution helpers for the Phase69 X2 diagnostic.

The Phase68 residual policy starts with a zero output head.  At that point the
encoder features are fixed and every first-step policy-gradient contribution
can be represented exactly by the 66 head coordinates.  These helpers keep the
reward decomposition, gait-phase decomposition, and environment bootstrap
independent of IsaacLab so they can be tested without starting the simulator.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


PHASE_NAMES = (
    "double_support_zero",
    "right_swing_left_support",
    "double_support_half",
    "left_swing_right_support",
)


def discounted_returns(
    reward: torch.Tensor,
    gamma: float = 0.99,
    terminal_bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return zero-bootstrap discounted returns for a ``[T, N]`` reward."""

    if reward.ndim != 2:
        raise ValueError("reward must have shape [time, env]")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    result = torch.zeros_like(reward)
    if terminal_bootstrap is None:
        running = torch.zeros_like(reward[0])
    else:
        if terminal_bootstrap.shape != reward.shape[1:]:
            raise ValueError("terminal bootstrap must have shape [env]")
        running = terminal_bootstrap.to(device=reward.device, dtype=reward.dtype)
    for step in range(reward.shape[0] - 1, -1, -1):
        running = reward[step] + gamma * running
        result[step] = running
    return result


def additive_normalized_credits(
    reward_components: Mapping[str, torch.Tensor],
    values: torch.Tensor,
    *,
    gamma: float = 0.99,
    terminal_bootstrap: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Decompose a normalized Monte-Carlo advantage without losing closure.

    The common denominator is the standard deviation of the total raw
    advantage.  Every component is centered separately, including the finite
    sample critic-baseline term, so their sum exactly reconstructs the centered
    and normalized total advantage used by Phase68.
    """

    if not reward_components:
        raise ValueError("at least one reward component is required")
    shapes = {tuple(value.shape) for value in reward_components.values()}
    if len(shapes) != 1 or values.ndim != 2 or tuple(values.shape) not in shapes:
        raise ValueError("reward components and values must share shape [time, env]")
    returns = {
        name: discounted_returns(reward, gamma=gamma)
        for name, reward in reward_components.items()
    }
    bootstrap_trace = discounted_returns(
        torch.zeros_like(values),
        gamma=gamma,
        terminal_bootstrap=terminal_bootstrap,
    )
    raw_advantage = sum(returns.values()) + bootstrap_trace - values
    denominator = raw_advantage.std() + 1.0e-8
    credits = {
        name: (value - value.mean()) / denominator
        for name, value in returns.items()
    }
    baseline = bootstrap_trace - values
    credits["baseline"] = (baseline - baseline.mean()) / denominator
    normalized_total = (raw_advantage - raw_advantage.mean()) / denominator
    return credits, raw_advantage, normalized_total


def standalone_credit(reward: torch.Tensor, *, gamma: float = 0.99) -> torch.Tensor:
    """Center and normalize one counterfactual reward without a critic."""

    returns = discounted_returns(reward, gamma=gamma)
    return (returns - returns.mean()) / (returns.std() + 1.0e-8)


def semantic_phase_ids(policy_observation: torch.Tensor) -> torch.Tensor:
    """Map the frozen gait suffix to four deployable action-time phases."""

    if policy_observation.shape[-1] != 93:
        raise ValueError("Phase69 requires the frozen 93D policy observation")
    suffix = policy_observation[..., -4:]
    cosine = suffix[..., 1]
    contacts = suffix[..., 2:]
    finite = torch.isfinite(suffix).all(dim=-1)
    zero = torch.isclose(contacts, torch.zeros_like(contacts), rtol=0.0, atol=1.0e-6)
    one = torch.isclose(contacts, torch.ones_like(contacts), rtol=0.0, atol=1.0e-6)
    left, right = one.unbind(dim=-1)
    binary = (zero | one).all(dim=-1)
    result = torch.full(cosine.shape, -1, dtype=torch.int64, device=cosine.device)
    valid = finite & binary & (left | right)
    result[valid & left & right & (cosine >= 0.0)] = 0
    result[valid & left & ~right] = 1
    result[valid & left & right & (cosine < 0.0)] = 2
    result[valid & ~left & right] = 3
    return result


def analytical_head_ascent_per_env(
    encoded_features: torch.Tensor,
    latent_actions: torch.Tensor,
    credit: torch.Tensor,
    *,
    latent_std: float = 0.35,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-env first-step ascent vectors in head weight/bias order.

    The output order is ``head.weight.flatten()`` followed by ``head.bias``.
    Per-environment vectors are averaged over the full time horizon.  A phase
    mask zeroes samples but keeps that same denominator, making phase vectors
    exactly additive.
    """

    if encoded_features.ndim != 3 or latent_actions.ndim != 3 or credit.ndim != 2:
        raise ValueError("expected features/actions [T,N,D] and credit [T,N]")
    if encoded_features.shape[:2] != latent_actions.shape[:2] or credit.shape != encoded_features.shape[:2]:
        raise ValueError("feature/action/credit time and environment shapes differ")
    if latent_actions.shape[-1] != 2 or latent_std <= 0.0:
        raise ValueError("Phase69 requires two latent actions and positive std")
    weighted_credit = credit
    if sample_mask is not None:
        if sample_mask.shape != credit.shape:
            raise ValueError("sample mask shape differs from credit")
        weighted_credit = weighted_credit * sample_mask.to(weighted_credit.dtype)
    score_mu = latent_actions / (latent_std * latent_std)
    weighted_score = weighted_credit.unsqueeze(-1) * score_mu
    weight = torch.einsum("tna,tnh->nah", weighted_score, encoded_features)
    weight = weight / encoded_features.shape[0]
    bias = weighted_score.mean(dim=0)
    return torch.cat((weight.flatten(start_dim=1), bias), dim=-1)


def vector_summary(vector: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Summarize one ascent vector relative to a nonzero reference."""

    vector = vector.flatten().to(torch.float64)
    reference = reference.flatten().to(torch.float64)
    reference_norm_sq = torch.dot(reference, reference)
    if reference_norm_sq <= 0.0:
        raise ValueError("reference vector must be nonzero")
    vector_norm = torch.linalg.vector_norm(vector)
    reference_norm = torch.sqrt(reference_norm_sq)
    dot = torch.dot(vector, reference)
    return {
        "norm": float(vector_norm),
        "dot_with_total": float(dot),
        "projection_on_total": float(dot / reference_norm_sq),
        "cosine_with_total": float(
            dot / (vector_norm * reference_norm)
            if vector_norm > 0.0
            else torch.tensor(0.0, dtype=torch.float64)
        ),
    }


def bootstrap_projection(
    component_per_env: torch.Tensor,
    total_per_env: torch.Tensor,
    *,
    seed: int = 690042,
    draws: int = 2048,
) -> dict[str, object]:
    """Bootstrap an environment-level component projection onto total ascent."""

    if component_per_env.shape != total_per_env.shape or component_per_env.ndim != 2:
        raise ValueError("component and total must share shape [env, coordinate]")
    if draws < 1 or component_per_env.shape[0] < 2:
        raise ValueError("bootstrap needs positive draws and at least two envs")
    component = component_per_env.detach().cpu().to(torch.float64)
    total = total_per_env.detach().cpu().to(torch.float64)
    full_component = component.mean(dim=0)
    full_total = total.mean(dim=0)
    full_denominator = torch.dot(full_total, full_total)
    if full_denominator <= 0.0:
        raise ValueError("total ascent is zero")
    full = torch.dot(full_component, full_total) / full_denominator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        component.shape[0], (draws, component.shape[0]), generator=generator
    )
    component_draw = component[indices].mean(dim=1)
    total_draw = total[indices].mean(dim=1)
    denominator = (total_draw * total_draw).sum(dim=-1)
    projection = (component_draw * total_draw).sum(dim=-1) / denominator.clamp_min(1.0e-30)
    loo_values = []
    for held_out in range(component.shape[0]):
        keep = torch.arange(component.shape[0]) != held_out
        component_loo = component[keep].mean(dim=0)
        total_loo = total[keep].mean(dim=0)
        loo_values.append(
            torch.dot(component_loo, total_loo)
            / torch.dot(total_loo, total_loo).clamp_min(1.0e-30)
        )
    loo = torch.stack(loo_values)
    sign = 1.0 if full >= 0.0 else -1.0
    return {
        "point": float(full),
        "bootstrap_draws": draws,
        "bootstrap_p025": float(torch.quantile(projection, 0.025)),
        "bootstrap_p50": float(torch.quantile(projection, 0.50)),
        "bootstrap_p975": float(torch.quantile(projection, 0.975)),
        "bootstrap_same_sign_fraction": float((projection * sign > 0.0).double().mean()),
        "leave_one_env_out_same_sign_fraction": float((loo * sign > 0.0).double().mean()),
    }
