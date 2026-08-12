"""Bounded deployable knee-target residual for the frozen X2 locomotion actor."""

from __future__ import annotations

import torch
from torch import nn


class PhaseConditionedKneeTargetResidual(nn.Module):
    """Map the existing 93D policy observation to two bounded target offsets.

    The final projection is exactly zero initialized, so inserting this module
    cannot change the frozen controller before an explicitly authorized update.
    Its moving mask uses only the deployed gait-clock suffix; standing output is
    therefore zero by construction rather than learned from simulator state.
    """

    def __init__(
        self,
        observation_dim: int = 93,
        hidden_dim: int = 32,
        maximum_target_offset_rad: float = 0.003,
    ) -> None:
        super().__init__()
        if observation_dim < 4 or hidden_dim <= 0 or maximum_target_offset_rad <= 0.0:
            raise ValueError("invalid phase-conditioned residual dimensions or bound")
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_target_offset_rad = float(maximum_target_offset_rad)
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(self.hidden_dim, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, policy_observation: torch.Tensor) -> torch.Tensor:
        if policy_observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation dim {self.observation_dim}, got {policy_observation.shape[-1]}"
            )
        gait_suffix = policy_observation[..., -4:]
        clock = gait_suffix[..., :2]
        desired_contacts = gait_suffix[..., 2:]
        moving = clock.square().sum(dim=-1) > 0.25
        valid_contacts = (desired_contacts > 0.5).any(dim=-1)
        moving = moving & valid_contacts
        raw = self.head(self.encoder(policy_observation))
        bounded = self.maximum_target_offset_rad * torch.tanh(raw)
        return bounded * moving.to(dtype=bounded.dtype).unsqueeze(-1)

    def target_offset_15d(self, policy_observation: torch.Tensor) -> torch.Tensor:
        """Return physical-radian offsets in the frozen 15D lower action order."""

        knee = self(policy_observation)
        result = knee.new_zeros((*knee.shape[:-1], 15))
        result[..., 3] = knee[..., 0]
        result[..., 9] = knee[..., 1]
        return result

    def trainable_manifest(self) -> dict[str, object]:
        return {
            "observation_dim": self.observation_dim,
            "hidden_dim": self.hidden_dim,
            "maximum_target_offset_rad": self.maximum_target_offset_rad,
            "trainable_names": sorted(
                name for name, parameter in self.named_parameters() if parameter.requires_grad
            ),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
