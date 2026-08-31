"""Zero-preserving residual adapters for the X2 six-link goal contract.

This module is deliberately framework-agnostic.  It does not replace the
SONIC actor/critic backbone and it does not change action dimensions.  A
frozen legacy module receives its original observation, while a new goal
branch learns an additive residual.  The final goal projection is initialized
to zero, so a newly created adapter exactly reproduces the legacy module even
when the garment goal is non-zero.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class ZeroPreservingResidualAdapter(nn.Module):
    """Add a zero-initialized future-goal residual to a legacy module output.

    Args:
        legacy_module: Existing actor or critic.  It receives ``base_obs``.
        base_obs_dim: Expected width of the legacy observation.
        goal_dim: Width of the flattened six-link goal feature (36 * H).
        output_dim: Output width of ``legacy_module``.
        hidden_dim: Width of the new goal encoder.
        goal_activation: Activation constructor used in the goal branch.
        freeze_legacy: If true, legacy parameters are frozen by default.

    The legacy module may return ``(batch, output_dim)`` or a tensor with a
    final output axis of ``output_dim``.  The adapter never concatenates the
    goal into the old input, which keeps the old first-layer state dict
    compatible.
    """

    def __init__(
        self,
        legacy_module: nn.Module,
        *,
        base_obs_dim: int,
        goal_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        goal_activation: Callable[[], nn.Module] = nn.Tanh,
        freeze_legacy: bool = True,
    ) -> None:
        super().__init__()
        if base_obs_dim <= 0 or goal_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.legacy_module = legacy_module
        self.base_obs_dim = int(base_obs_dim)
        self.goal_dim = int(goal_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, self.hidden_dim),
            goal_activation(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        # Critical compatibility invariant: the new branch is initially a
        # no-op for every goal, not only for an all-zero goal sample.
        final = self.goal_encoder[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover - constructor contract
            raise TypeError("goal encoder final layer must be nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if freeze_legacy:
            for parameter in self.legacy_module.parameters():
                parameter.requires_grad_(False)

    def forward(self, base_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if base_obs.ndim < 2 or base_obs.shape[-1] != self.base_obs_dim:
            raise ValueError(
                f"base_obs must end in {self.base_obs_dim}, got {tuple(base_obs.shape)}"
            )
        if goal.ndim < 2 or goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"goal must end in {self.goal_dim}, got {tuple(goal.shape)}"
            )
        if base_obs.shape[:-1] != goal.shape[:-1]:
            raise ValueError(
                "base_obs and goal batch dimensions differ: "
                f"{tuple(base_obs.shape)} vs {tuple(goal.shape)}"
            )
        legacy_output = self.legacy_module(base_obs)
        if legacy_output.shape[:-1] != base_obs.shape[:-1] or legacy_output.shape[-1] != self.output_dim:
            raise ValueError(
                "legacy module output must match batch dimensions and output_dim: "
                f"got {tuple(legacy_output.shape)}, expected *{self.output_dim}"
            )
        residual = self.goal_encoder(goal)
        return legacy_output + residual


def build_x2_single_frame_adapter(
    legacy_module: nn.Module,
    *,
    base_obs_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
) -> ZeroPreservingResidualAdapter:
    """Build the H=1 adapter for the 36-D six-link contract."""

    return ZeroPreservingResidualAdapter(
        legacy_module,
        base_obs_dim=base_obs_dim,
        goal_dim=36,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
