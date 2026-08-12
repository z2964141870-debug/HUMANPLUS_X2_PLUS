"""Pure helpers for the zero-optimizer Phase70 long-lookahead diagnostic."""

from __future__ import annotations

import torch


def anchor_returns(
    reward: torch.Tensor,
    *,
    anchor_steps: int = 200,
    gamma: float = 0.99,
    terminal_bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return first-anchor score returns using the full observed horizon.

    For a reward with shape ``[T, N]``, row ``t`` is
    ``sum_{k=t}^{T-1} gamma**(k-t) reward[k] + gamma**(T-t) B(s_T)``.
    The returned score anchor is restricted to rows ``[0, anchor_steps)``.
    """

    if reward.ndim != 2:
        raise ValueError("reward must have shape [time, env]")
    if not 0 < anchor_steps <= reward.shape[0]:
        raise ValueError("anchor_steps must lie within the reward horizon")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if terminal_bootstrap is None:
        running = torch.zeros_like(reward[0])
    else:
        if terminal_bootstrap.shape != reward.shape[1:]:
            raise ValueError("terminal bootstrap must have shape [env]")
        running = terminal_bootstrap.to(device=reward.device, dtype=reward.dtype)
    output = torch.zeros_like(reward)
    for step in range(reward.shape[0] - 1, -1, -1):
        running = reward[step] + gamma * running
        output[step] = running
    return output[:anchor_steps]


def normalized_advantage(
    reward: torch.Tensor,
    values: torch.Tensor,
    *,
    anchor_steps: int = 200,
    gamma: float = 0.99,
    terminal_bootstrap: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and globally normalized anchor advantage."""

    if values.ndim != 2 or values.shape[1] != reward.shape[1]:
        raise ValueError("values must have shape [time, env]")
    if values.shape[0] < anchor_steps:
        raise ValueError("values do not cover the anchor")
    raw = anchor_returns(
        reward,
        anchor_steps=anchor_steps,
        gamma=gamma,
        terminal_bootstrap=terminal_bootstrap,
    ) - values[:anchor_steps]
    return raw, (raw - raw.mean()) / (raw.std() + 1.0e-8)


def standalone_credit(
    reward: torch.Tensor,
    *,
    anchor_steps: int = 200,
    gamma: float = 0.99,
    terminal_bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize one reward/metric lookahead without a value baseline."""

    value = anchor_returns(
        reward,
        anchor_steps=anchor_steps,
        gamma=gamma,
        terminal_bootstrap=terminal_bootstrap,
    )
    return (value - value.mean()) / (value.std() + 1.0e-8)


def vector_comparison(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compare two nonzero gradient vectors."""

    value = value.detach().cpu().flatten().to(torch.float64)
    reference = reference.detach().cpu().flatten().to(torch.float64)
    value_norm = torch.linalg.vector_norm(value)
    reference_norm = torch.linalg.vector_norm(reference)
    if value_norm <= 0.0 or reference_norm <= 0.0:
        raise ValueError("gradient vectors must be nonzero")
    dot = torch.dot(value, reference)
    return {
        "cosine": float(dot / (value_norm * reference_norm)),
        "relative_l2": float(torch.linalg.vector_norm(value - reference) / reference_norm),
        "projection_on_reference": float(dot / torch.dot(reference, reference)),
    }


def environment_influence(per_env: torch.Tensor) -> dict[str, float]:
    """Summarize cluster influence using environment trajectories as units."""

    if per_env.ndim != 2 or per_env.shape[0] < 2:
        raise ValueError("per_env must have shape [env, coordinate]")
    total = per_env.mean(dim=0)
    scalar = torch.einsum("nc,c->n", per_env.to(torch.float64), total.to(torch.float64))
    absolute = scalar.abs()
    denominator = absolute.sum().clamp_min(1.0e-30)
    return {
        "kish_effective_sample_size": float(
            denominator.square() / absolute.square().sum().clamp_min(1.0e-30)
        ),
        "maximum_absolute_env_contribution_fraction": float(absolute.max() / denominator),
    }
