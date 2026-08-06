"""Pure helpers for DC-PEFT actor-objective gradient diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch


def clipped_head_policy_losses(
    pg_losses: torch.Tensor,
    pg_losses_clipped: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], float, int]:
    """Recover the exact per-head PPO policy losses before the head sum.

    ``pg_losses`` already contains the configured positive advantage-head weights.
    Entropy is deliberately excluded: it is a shared regularizer rather than a
    semantic objective and would artificially make all head gradients align.
    """
    if pg_losses.shape != pg_losses_clipped.shape or pg_losses.ndim != 3:
        raise ValueError("policy loss tensors must share shape (batch, time, heads)")
    if padding_mask.shape != pg_losses.shape[:-1] or padding_mask.dtype is not torch.bool:
        raise ValueError("padding_mask must be boolean with shape (batch, time)")
    valid = ~padding_mask
    valid_steps = int(valid.sum().item())
    if valid_steps <= 0:
        raise ValueError("actor-gradient probe received no valid rollout steps")
    clipped = torch.maximum(pg_losses, pg_losses_clipped)
    head_losses = [clipped[..., index][valid].mean() for index in range(clipped.shape[-1])]
    reconstructed = torch.stack(head_losses).sum()
    combined = clipped.sum(dim=-1)[valid].mean()
    error = float(torch.abs(reconstructed.detach() - combined.detach()).item())
    return head_losses, error, valid_steps


def gradient_geometry(vectors: Mapping[str, torch.Tensor]) -> dict:
    """Return JSON-safe norm/cosine statistics for named flat vectors."""
    if len(vectors) < 2:
        raise ValueError("gradient geometry requires at least two objective vectors")
    names = list(vectors)
    flattened = []
    expected_size = None
    for name in names:
        vector = torch.as_tensor(vectors[name]).detach().double().reshape(-1).cpu()
        if expected_size is None:
            expected_size = int(vector.numel())
        elif vector.numel() != expected_size:
            raise ValueError("gradient vectors must have equal length")
        if vector.numel() == 0 or not torch.isfinite(vector).all():
            raise ValueError(f"gradient vector {name!r} is empty or non-finite")
        flattened.append(vector)
    matrix = torch.stack(flattened)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    denominator = torch.outer(norms, norms)
    cosine = torch.zeros_like(denominator)
    active = denominator > 0
    cosine[active] = (matrix @ matrix.T)[active] / denominator[active]

    pairwise = {}
    negative = 0
    pair_count = 0
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            value = float(cosine[first, second].item())
            pairwise[f"{names[first]}__{names[second]}"] = value
            negative += int(value < 0.0)
            pair_count += 1
    return {
        "group_names": names,
        "vector_size": int(expected_size),
        "gradient_norms": {
            name: float(norms[index].item()) for index, name in enumerate(names)
        },
        "cosine_matrix": cosine.tolist(),
        "pairwise_cosines": pairwise,
        "pairwise_negative_fraction": float(negative / pair_count),
    }


def finite_summary(values: Sequence[float]) -> dict:
    """Summarize a non-empty finite scalar sequence."""
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "negative_fraction": float((tensor < 0).double().mean().item()),
        "near_orthogonal_fraction": float((torch.abs(tensor) < 0.05).double().mean().item()),
    }


def assert_finite_record(record: Mapping) -> None:
    """Recursively reject NaN/Inf before a diagnostic record is persisted."""
    for value in record.values():
        if isinstance(value, Mapping):
            assert_finite_record(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    assert_finite_record(item)
                elif isinstance(item, (float, int)) and not math.isfinite(float(item)):
                    raise ValueError("diagnostic record contains a non-finite value")
        elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise ValueError("diagnostic record contains a non-finite value")
