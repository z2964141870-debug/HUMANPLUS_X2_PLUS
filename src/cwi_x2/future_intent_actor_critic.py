"""Torch-only future-intent adapter for a frozen Stage208 actor."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


BASE_ACTOR_OBS_DIM = 93
UPPER_INTENT_DIM = 28
GAIT_PHASE_DIM = 4
NUM_LOWER_ACTIONS = 15
NUM_COORDINATION_MODES = 8
COMMAND_OBS_START = 9


def build_coordination_basis(
    *, device: torch.device | str | None = None, dtype=torch.float32
) -> torch.Tensor:
    """Eight unit modes spanning hip pitch/roll/yaw and waist yaw/roll."""
    basis = torch.zeros(
        NUM_COORDINATION_MODES,
        NUM_LOWER_ACTIONS,
        device=device,
        dtype=dtype,
    )
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    for row, left, right in (
        (0, 0, 6),  # hip pitch common/differential
        (2, 1, 7),  # hip roll common/differential
        (4, 2, 8),  # hip yaw common/differential
    ):
        basis[row, left] = inv_sqrt_2
        basis[row, right] = inv_sqrt_2
        basis[row + 1, left] = inv_sqrt_2
        basis[row + 1, right] = -inv_sqrt_2
    basis[6, 12] = 1.0  # waist yaw
    basis[7, 14] = 1.0  # waist roll
    return basis


class FutureIntentActorCritic(ActorCritic):
    """Frozen Stage208 actor plus a zero-initialized low-dimensional residual."""

    is_recurrent = False
    VALID_MODES = {"disabled", "current", "future", "future_no_phase"}

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=(256, 128, 128),
        critic_hidden_dims=(256, 128, 128),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        base_actor_obs_dim=BASE_ACTOR_OBS_DIM,
        upper_intent_dim=UPPER_INTENT_DIM,
        gait_phase_dim=GAIT_PHASE_DIM,
        coordination_hidden_dim=32,
        coordination_output_scale=0.10,
        coordination_blend=1.0,
        intent_gate_scale_rad=0.02,
        locomotion_gate_scale=0.10,
        adapter_mode="future",
        **kwargs,
    ):
        if actor_obs_normalization:
            raise ValueError("future-intent adapter requires unnormalized observations")
        if num_actions != NUM_LOWER_ACTIONS:
            raise ValueError(f"future-intent adapter requires {NUM_LOWER_ACTIONS} actions")
        if adapter_mode not in self.VALID_MODES:
            raise ValueError(
                f"unknown adapter mode {adapter_mode!r}; expected {sorted(self.VALID_MODES)}"
            )
        if not 0.0 < coordination_output_scale <= 1.0:
            raise ValueError("coordination output scale must lie in (0, 1]")
        if not 0.0 <= coordination_blend <= 1.0:
            raise ValueError("coordination blend must lie in [0, 1]")
        if intent_gate_scale_rad <= 0.0:
            raise ValueError("intent gate scale must be positive")
        if locomotion_gate_scale <= 0.0:
            raise ValueError("locomotion gate scale must be positive")

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            **kwargs,
        )

        self.base_actor_obs_dim = int(base_actor_obs_dim)
        self.upper_intent_dim = int(upper_intent_dim)
        self.gait_phase_dim = int(gait_phase_dim)
        self.coordination_output_scale = float(coordination_output_scale)
        self.coordination_blend = float(coordination_blend)
        self.intent_gate_scale_rad = float(intent_gate_scale_rad)
        self.locomotion_gate_scale = float(locomotion_gate_scale)
        self.adapter_mode = str(adapter_mode)
        expected_actor_dim = self.base_actor_obs_dim + self.upper_intent_dim
        actual_actor_dim = sum(
            obs[name].shape[-1] for name in obs_groups["policy"]
        )
        if actual_actor_dim != expected_actor_dim:
            raise ValueError(
                "future-intent actor observation contract mismatch: "
                f"expected={expected_actor_dim} actual={actual_actor_dim}"
            )
        if self.gait_phase_dim > self.base_actor_obs_dim:
            raise ValueError("gait phase cannot exceed the base observation")

        self.actor = MLP(
            self.base_actor_obs_dim,
            num_actions,
            list(actor_hidden_dims),
            activation,
        )
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

        adapter_input_dim = self.upper_intent_dim + self.gait_phase_dim
        self.coordination_adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, int(coordination_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(coordination_hidden_dim), NUM_COORDINATION_MODES),
        )
        nn.init.zeros_(self.coordination_adapter[-1].weight)
        nn.init.zeros_(self.coordination_adapter[-1].bias)
        self.register_buffer(
            "coordination_basis",
            build_coordination_basis(),
            persistent=False,
        )
        self._last_coordination_residual: torch.Tensor | None = None
        print(f"Frozen Stage208 actor: {self.actor}")
        print(
            "Future-intent coordination adapter: "
            f"mode={self.adapter_mode} module={self.coordination_adapter}"
        )

    def _mean_from_actor_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        expected = self.base_actor_obs_dim + self.upper_intent_dim
        if actor_obs.shape[-1] != expected:
            raise RuntimeError(
                f"unexpected future-intent observation shape: {tuple(actor_obs.shape)}"
            )
        base_obs = actor_obs[..., : self.base_actor_obs_dim]
        intent = actor_obs[..., self.base_actor_obs_dim :]
        if self.adapter_mode in {"disabled", "current"}:
            intent = torch.cat(
                (
                    intent[..., : self.upper_intent_dim // 2],
                    torch.zeros_like(intent[..., self.upper_intent_dim // 2 :]),
                ),
                dim=-1,
            )
        phase = base_obs[..., -self.gait_phase_dim :]
        if self.adapter_mode in {"disabled", "future_no_phase"}:
            phase = torch.zeros_like(phase)

        adapter_input = torch.cat((intent, phase), dim=-1)
        mode_coefficients = torch.tanh(self.coordination_adapter(adapter_input))
        residual = mode_coefficients @ self.coordination_basis
        residual = torch.clamp(
            residual,
            min=-self.coordination_output_scale,
            max=self.coordination_output_scale,
        )
        intent_gate = torch.clamp(
            torch.amax(torch.abs(intent), dim=-1, keepdim=True)
            / self.intent_gate_scale_rad,
            min=0.0,
            max=1.0,
        )
        if self.adapter_mode == "disabled":
            intent_gate = torch.zeros_like(intent_gate)
        forward_command = base_obs[..., COMMAND_OBS_START : COMMAND_OBS_START + 1]
        locomotion_gate = torch.clamp(
            torch.abs(forward_command) / self.locomotion_gate_scale,
            min=0.0,
            max=1.0,
        )
        residual = (
            self.coordination_blend * intent_gate * locomotion_gate * residual
        )
        self._last_coordination_residual = residual
        return self.actor(base_obs) + residual

    def update_distribution(self, obs):
        mean = self._mean_from_actor_observation(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"unknown action noise type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self._mean_from_actor_observation(actor_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load native checkpoints or adapter-free Stage208 checkpoints."""
        incompatible = nn.Module.load_state_dict(self, state_dict, strict=False)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("coordination_adapter.")
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "future-intent checkpoint contract mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        return True
