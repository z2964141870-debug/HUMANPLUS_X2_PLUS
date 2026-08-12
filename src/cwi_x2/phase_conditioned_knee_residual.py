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
        # The deployed gait generator emits only the three supported contact
        # codes [1, 1], [1, 0], and [0, 1].  Treat fractional, all-zero, and
        # non-finite suffixes as invalid rather than thresholding them into a
        # valid phase.  This is a safety gate, not a learned classifier.
        finite_suffix = torch.isfinite(gait_suffix).all(dim=-1)
        contact_is_zero = torch.isclose(
            desired_contacts,
            torch.zeros_like(desired_contacts),
            rtol=0.0,
            atol=1.0e-6,
        )
        contact_is_one = torch.isclose(
            desired_contacts,
            torch.ones_like(desired_contacts),
            rtol=0.0,
            atol=1.0e-6,
        )
        binary_contacts = (contact_is_zero | contact_is_one).all(dim=-1)
        at_least_one_contact = contact_is_one.any(dim=-1)
        valid_contacts = finite_suffix & binary_contacts & at_least_one_contact
        moving = (clock.square().sum(dim=-1) > 0.25) & valid_contacts
        raw = self.head(self.encoder(policy_observation))
        bounded = self.maximum_target_offset_rad * torch.tanh(raw)
        # torch.where keeps an invalid row exactly zero even if a malformed
        # suffix made the unselected network branch non-finite.
        return torch.where(moving.unsqueeze(-1), bounded, torch.zeros_like(bounded))

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
