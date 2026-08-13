"""Pure helpers for the zero-optimizer Phase72 antithetic diagnostic.

Phase72 pairs two closed-loop trajectories that start from the same simulator
state and receive opposite latent Gaussian innovations.  The helpers here are
CPU-testable and deliberately contain no IsaacLab imports.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch


ANCHOR_STEPS = 200
HORIZON_STEPS = 400
NUM_ENVS = 64
LATENT_DIM = 2
LATENT_STD = 0.35
HEAD_DIM = 66


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""

    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(f"{value.dtype}:{value.shape}".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def generate_epsilon(seed: int) -> torch.Tensor:
    """Generate one immutable CPU antithetic innovation schedule."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(
        HORIZON_STEPS,
        NUM_ENVS,
        LATENT_DIM,
        generator=generator,
        dtype=torch.float32,
    )


def anchor_returns64(
    reward: torch.Tensor,
    *,
    gamma: float = 0.99,
    anchor_steps: int = ANCHOR_STEPS,
) -> torch.Tensor:
    """Compute first-anchor/full-horizon returns in float64."""

    if reward.ndim != 2:
        raise ValueError("reward must have shape [time, env]")
    if reward.shape[0] < anchor_steps or not 0.0 < gamma <= 1.0:
        raise ValueError("invalid anchor or gamma")
    reward64 = reward.detach().cpu().to(torch.float64)
    running = torch.zeros_like(reward64[0])
    result = torch.empty_like(reward64)
    for step in range(reward64.shape[0] - 1, -1, -1):
        running = reward64[step] + gamma * running
        result[step] = running
    return result[:anchor_steps]


def paired_standardized_credit(
    plus_raw: torch.Tensor,
    minus_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Center/scale a pair with one denominator shared by both signs."""

    if plus_raw.shape != minus_raw.shape or plus_raw.ndim != 2:
        raise ValueError("paired raw credits must share shape [anchor, env]")
    plus = plus_raw.detach().cpu().to(torch.float64)
    minus = minus_raw.detach().cpu().to(torch.float64)
    joined = torch.cat((plus.reshape(-1), minus.reshape(-1)))
    mean = joined.mean()
    std = joined.std()
    if not torch.isfinite(std) or std <= 1.0e-12:
        raise ValueError("paired credit has zero or non-finite scale")
    return (
        (plus - mean) / (std + 1.0e-12),
        (minus - mean) / (std + 1.0e-12),
        {"paired_mean": float(mean), "paired_std": float(std)},
    )


def analytical_head_ascent64(
    encoded_features: torch.Tensor,
    latent_actions: torch.Tensor,
    credit: torch.Tensor,
    *,
    latent_std: float = LATENT_STD,
) -> torch.Tensor:
    """Return one 66-D score ascent per environment in float64."""

    if encoded_features.ndim != 3 or latent_actions.ndim != 3 or credit.ndim != 2:
        raise ValueError("features/actions/credit must be [T,N,D], [T,N,2], [T,N]")
    if encoded_features.shape[:2] != latent_actions.shape[:2] or credit.shape != encoded_features.shape[:2]:
        raise ValueError("feature/action/credit shapes differ")
    if latent_actions.shape[-1] != LATENT_DIM or encoded_features.shape[-1] != 32:
        raise ValueError("Phase72 requires latent width 2 and feature width 32")
    features = encoded_features.detach().cpu().to(torch.float64)
    actions = latent_actions.detach().cpu().to(torch.float64)
    credit64 = credit.detach().cpu().to(torch.float64)
    score_mu = actions / (float(latent_std) ** 2)
    weighted = credit64.unsqueeze(-1) * score_mu
    weight = torch.einsum("tna,tnh->nah", weighted, features) / features.shape[0]
    bias = weighted.mean(dim=0)
    result = torch.cat((weight.flatten(start_dim=1), bias), dim=-1)
    if result.shape != (features.shape[1], HEAD_DIM):
        raise RuntimeError("Phase72 head ascent shape changed")
    return result


def paired_direction(
    plus_features: torch.Tensor,
    plus_latent: torch.Tensor,
    plus_raw: torch.Tensor,
    minus_features: torch.Tensor,
    minus_latent: torch.Tensor,
    minus_raw: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average the two standardized score ascents of one antithetic pair."""

    plus_credit, minus_credit, scale = paired_standardized_credit(plus_raw, minus_raw)
    plus = analytical_head_ascent64(plus_features, plus_latent, plus_credit)
    minus = analytical_head_ascent64(minus_features, minus_latent, minus_credit)
    return 0.5 * (plus + minus), scale


def projection(component: torch.Tensor, metric: torch.Tensor) -> float:
    """Project one mean direction onto a nonzero metric direction."""

    component = component.detach().cpu().flatten().to(torch.float64)
    metric = metric.detach().cpu().flatten().to(torch.float64)
    denominator = torch.dot(metric, metric)
    if denominator <= 1.0e-30:
        raise ValueError("metric direction is zero")
    return float(torch.dot(component, metric) / denominator)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Cosine between two nonzero vectors."""

    left = left.detach().cpu().flatten().to(torch.float64)
    right = right.detach().cpu().flatten().to(torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= 1.0e-30:
        raise ValueError("direction is zero")
    return float(torch.dot(left, right) / denominator)


def environment_influence(per_cluster: torch.Tensor) -> dict[str, float]:
    """Kish ESS and maximum absolute influence over trajectory clusters."""

    if per_cluster.ndim != 2 or per_cluster.shape[0] < 2:
        raise ValueError("per_cluster must have shape [cluster, coordinate]")
    value = per_cluster.detach().cpu().to(torch.float64)
    total = value.mean(dim=0)
    scalar = torch.einsum("nc,c->n", value, total)
    absolute = scalar.abs()
    denominator = absolute.sum().clamp_min(1.0e-30)
    return {
        "kish_effective_sample_size": float(
            denominator.square() / absolute.square().sum().clamp_min(1.0e-30)
        ),
        "maximum_absolute_cluster_contribution_fraction": float(
            absolute.max() / denominator
        ),
    }


def stratified_bootstrap_projection(
    component: torch.Tensor,
    metric: torch.Tensor,
    *,
    seed: int,
    draws: int = 4096,
) -> dict[str, float]:
    """Two-level bootstrap over reset/latent seeds and env clusters."""

    if component.shape != metric.shape or component.ndim != 3:
        raise ValueError("component and metric must share [seed, env, coordinate]")
    if component.shape[1] != NUM_ENVS or component.shape[0] < 2:
        raise ValueError("unexpected Phase72 seed/env shape")
    component = component.detach().cpu().to(torch.float64)
    metric = metric.detach().cpu().to(torch.float64)
    full = projection(component.mean(dim=(0, 1)), metric.mean(dim=(0, 1)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = []
    for _ in range(draws):
        seed_index = torch.randint(
            component.shape[0], (component.shape[0],), generator=generator
        )
        env_index = torch.randint(
            NUM_ENVS,
            (component.shape[0], NUM_ENVS),
            generator=generator,
        )
        selected_seed = seed_index.unsqueeze(1).expand_as(env_index)
        c = component[selected_seed, env_index].mean(dim=(0, 1))
        m = metric[selected_seed, env_index].mean(dim=(0, 1))
        values.append(torch.dot(c, m) / torch.dot(m, m).clamp_min(1.0e-30))
    samples = torch.stack(values)
    return {
        "point": full,
        "draws": draws,
        "p025": float(torch.quantile(samples, 0.025)),
        "p50": float(torch.quantile(samples, 0.50)),
        "p975": float(torch.quantile(samples, 0.975)),
    }


def build_raw_signals(bundle: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Build the seven preregistered raw anchor signals from one bundle."""

    reward = torch.as_tensor(bundle["total_reward"])
    terms = torch.as_tensor(bundle["reward_by_term"])
    names = tuple(bundle["reward_term_names"])
    values = torch.as_tensor(bundle["value"])
    pitch_metric = torch.as_tensor(bundle["signed_pitch_rad"])
    support_metric = torch.as_tensor(bundle["support_outside_m"])
    step_dt = float(bundle["step_dt_s"])
    pitch = terms[..., names.index("signed_backward_pitch")]
    support = terms[..., names.index("actual_support_com")]
    locomotion = reward - pitch - support
    total_return = anchor_returns64(reward)
    return {
        "primary_total": total_return - values[:ANCHOR_STEPS].to(torch.float64),
        "reward_only": total_return,
        "locomotion": anchor_returns64(locomotion),
        "pitch_reward": anchor_returns64(pitch),
        "support_reward": anchor_returns64(support),
        "positive_pitch_metric": anchor_returns64(pitch_metric * step_dt),
        "lower_support_metric": anchor_returns64(-support_metric * step_dt),
    }
