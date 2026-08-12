"""Auditable two-dimensional PPO interface for the X2 physical knee residual.

The frozen Stage219 actor remains the 15-D controller seen by IsaacLab.  RSL-RL
sees only a two-dimensional latent Gaussian whose deterministic tanh transform
is inserted after the actor/template/plant composition in physical radians.
This keeps the sampled action, stored action, and PPO likelihood identical.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from rsl_rl.env import VecEnv
from torch import nn
from torch.distributions import Normal

from cwi_x2.phase_conditioned_knee_residual import (
    PhaseConditionedKneeTargetResidual,
)


KNEE_ACTION_INDICES = (3, 9)
LATENT_ACTION_DIM = 2
LATENT_STD = 0.35
RESIDUAL_INITIALIZATION_SEED = 680042


def build_seeded_residual(
    seed: int = RESIDUAL_INITIALIZATION_SEED,
) -> PhaseConditionedKneeTargetResidual:
    """Construct the Phase68 residual without consuming caller RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return PhaseConditionedKneeTargetResidual()


def deployable_moving_mask(policy_observation: torch.Tensor) -> torch.Tensor:
    """Return the exact fail-closed moving/contact mask used at deployment."""

    if policy_observation.shape[-1] != 93:
        raise ValueError("Phase68 requires the frozen 93D policy observation")
    gait_suffix = policy_observation[..., -4:]
    clock = gait_suffix[..., :2]
    desired_contacts = gait_suffix[..., 2:]
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
    return (
        finite_suffix
        & binary_contacts
        & at_least_one_contact
        & (clock.square().sum(dim=-1) > 0.25)
    )


class ResidualActorCritic(nn.Module):
    """Frozen 15-D source actor/critic plus a trainable 2-D latent policy."""

    is_recurrent = False

    def __init__(
        self,
        source_policy: nn.Module,
        residual: PhaseConditionedKneeTargetResidual,
        *,
        latent_std: float = LATENT_STD,
    ) -> None:
        super().__init__()
        if latent_std <= 0.0:
            raise ValueError("latent std must be positive")
        self.source_policy = source_policy.eval()
        for parameter in self.source_policy.parameters():
            parameter.requires_grad_(False)
        self.residual = residual
        self.register_buffer(
            "latent_std",
            torch.full((LATENT_ACTION_DIM,), float(latent_std)),
        )
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _policy_observation(obs) -> torch.Tensor:
        value = obs["policy"]
        if value.shape[-1] != 93:
            raise ValueError("Phase68 policy observation width changed")
        return value

    def latent_mean(self, obs) -> torch.Tensor:
        policy_observation = self._policy_observation(obs)
        raw = self.residual.head(self.residual.encoder(policy_observation))
        active = deployable_moving_mask(policy_observation)
        return torch.where(active.unsqueeze(-1), raw, torch.zeros_like(raw))

    def physical_target_offset_from_latent(self, latent: torch.Tensor, obs) -> torch.Tensor:
        if latent.shape[-1] != LATENT_ACTION_DIM:
            raise ValueError("Phase68 latent action width changed")
        policy_observation = self._policy_observation(obs)
        active = deployable_moving_mask(policy_observation)
        bounded = self.residual.maximum_target_offset_rad * torch.tanh(latent)
        return torch.where(active.unsqueeze(-1), bounded, torch.zeros_like(bounded))

    def source_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return torch.clamp(self.source_policy.act_inference(obs), -1.0, 1.0)

    def update_distribution(self, obs) -> None:
        mean = self.latent_mean(obs)
        self.distribution = Normal(mean, self.latent_std.expand_as(mean))

    def act(self, obs, **_kwargs) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        return self.latent_mean(obs)

    def evaluate(self, obs, **_kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self.source_policy.evaluate(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Phase68 distribution is not initialized")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Phase68 distribution is not initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Phase68 distribution is not initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Phase68 distribution is not initialized")
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, _obs) -> None:
        return None

    def reset(self, _dones=None) -> None:
        return None

    def get_hidden_states(self):
        return None

    def trainable_manifest(self) -> dict[str, object]:
        return {
            "names": sorted(
                f"residual.{name}"
                for name, parameter in self.residual.named_parameters()
                if parameter.requires_grad
            ),
            "parameters": sum(
                parameter.numel()
                for parameter in self.residual.parameters()
                if parameter.requires_grad
            ),
            "latent_action_dim": LATENT_ACTION_DIM,
            "latent_std": self.latent_std.detach().cpu().tolist(),
            "physical_bound_rad": self.residual.maximum_target_offset_rad,
            "physical_indices": list(KNEE_ACTION_INDICES),
        }


class PhysicalKneeResidualVecEnv(VecEnv):
    """Expose the physical residual as the only two-dimensional MDP action."""

    def __init__(
        self,
        inner,
        policy: ResidualActorCritic,
        *,
        term_name: str = "joint_pos",
        step_callback: Callable[[dict[str, torch.Tensor]], None] | None = None,
    ) -> None:
        self.inner = inner
        self.policy = policy
        self.num_envs = inner.num_envs
        self.num_actions = LATENT_ACTION_DIM
        self.device = inner.device
        self.max_episode_length = inner.max_episode_length
        self.cfg = inner.cfg
        self._observations = inner.get_observations()
        self._term = inner.unwrapped.action_manager._terms[term_name]
        self._original_process_actions = self._term.process_actions
        self._step_callback = step_callback
        self._pending_offset = torch.zeros(
            self.num_envs, 15, device=self.device, dtype=torch.float32
        )
        self._hook_calls = 0
        self._current_source_target = torch.zeros_like(self._pending_offset)
        self._current_final_target = torch.zeros_like(self._pending_offset)
        self._current_effective_offset = torch.zeros_like(self._pending_offset)

        def process_actions_with_residual(actions: torch.Tensor) -> None:
            self._original_process_actions(actions)
            self._hook_calls += 1
            self._current_source_target = self._term._processed_actions.clone()
            proposed = self._current_source_target + self._pending_offset
            if self._term.cfg.clip is not None:
                proposed = torch.clamp(
                    proposed,
                    min=self._term._clip[:, :, 0],
                    max=self._term._clip[:, :, 1],
                )
            self._current_final_target = proposed
            self._current_effective_offset = proposed - self._current_source_target
            self._term._processed_actions[:] = proposed

        self._term.process_actions = process_actions_with_residual

    @property
    def unwrapped(self):
        return self.inner.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.inner.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.inner.episode_length_buf = value

    def get_observations(self):
        return self._observations

    def step(self, latent_actions: torch.Tensor):
        if latent_actions.shape != (self.num_envs, LATENT_ACTION_DIM):
            raise RuntimeError(
                f"Phase68 expected latent shape {(self.num_envs, LATENT_ACTION_DIM)}, "
                f"got {tuple(latent_actions.shape)}"
            )
        if not torch.isfinite(latent_actions).all():
            raise RuntimeError("Phase68 received a non-finite latent action")
        with torch.no_grad():
            base_action = self.policy.source_action(self._observations)
            knee_offset = self.policy.physical_target_offset_from_latent(
                latent_actions, self._observations
            )
            active_mask = deployable_moving_mask(self._observations["policy"])
        self._pending_offset.zero_()
        self._pending_offset[:, KNEE_ACTION_INDICES[0]] = knee_offset[:, 0]
        self._pending_offset[:, KNEE_ACTION_INDICES[1]] = knee_offset[:, 1]
        calls_before = self._hook_calls
        next_obs, reward, done, extras = self.inner.step(base_action)
        if self._hook_calls != calls_before + 1:
            raise RuntimeError("Phase68 physical-target hook was not called exactly once")
        non_knee = self._current_effective_offset.clone()
        non_knee[:, list(KNEE_ACTION_INDICES)] = 0.0
        record = {
            "latent": latent_actions.detach().clone(),
            "base_action": base_action.detach().clone(),
            "requested_knee_offset": knee_offset.detach().clone(),
            "effective_knee_offset": self._current_effective_offset[
                :, list(KNEE_ACTION_INDICES)
            ].detach().clone(),
            "source_target": self._current_source_target.detach().clone(),
            "final_target": self._current_final_target.detach().clone(),
            "non_knee_effective": non_knee.detach().clone(),
            "active_mask": active_mask.detach().clone(),
            "reward": reward.detach().clone(),
            "done": done.detach().clone(),
        }
        if self._step_callback is not None:
            self._step_callback(record)
        self._observations = next_obs
        return next_obs, reward, done, extras

    def close(self):
        self._term.process_actions = self._original_process_actions
        return self.inner.close()
