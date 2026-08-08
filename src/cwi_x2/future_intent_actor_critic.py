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
LOCOMOTION_INTENT_DIM = 2
DYNAMIC_INTENT_DIM = UPPER_INTENT_DIM + LOCOMOTION_INTENT_DIM
GAIT_PHASE_DIM = 4
NUM_LOWER_ACTIONS = 15
NUM_COORDINATION_MODES = 8
COMMAND_OBS_START = 9
JOINT_POS_OBS_START = 12
JOINT_VEL_OBS_START = 43
LAST_ACTION_OBS_START = 74
LOWER_ISAAC_INDICES = (0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8)
LOWER_ACTION_SCALES = (
    0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
    0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
    0.4, 0.16, 0.16,
)
RESPONSE_CONTEXT_DIM = 3 * NUM_LOWER_ACTIONS
RESPONSE_MODE_MASK = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)


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


def lower_response_context(base_obs: torch.Tensor) -> torch.Tensor:
    """Return deployable normalized lower q/dq/last-action response context."""
    if base_obs.shape[-1] != BASE_ACTOR_OBS_DIM:
        raise ValueError(f"expected {BASE_ACTOR_OBS_DIM}D base observation")
    indices = torch.as_tensor(
        LOWER_ISAAC_INDICES,
        device=base_obs.device,
        dtype=torch.long,
    )
    scales = torch.as_tensor(
        LOWER_ACTION_SCALES,
        device=base_obs.device,
        dtype=base_obs.dtype,
    )
    q = torch.index_select(base_obs, -1, indices + JOINT_POS_OBS_START)
    dq = torch.index_select(base_obs, -1, indices + JOINT_VEL_OBS_START)
    last_action = base_obs[
        ...,
        LAST_ACTION_OBS_START : LAST_ACTION_OBS_START + NUM_LOWER_ACTIONS,
    ]
    q_normalized = 0.5 * torch.clamp(q / scales, min=-2.0, max=2.0)
    dq_normalized = torch.clamp(dq / 5.0, min=-1.0, max=1.0)
    return torch.cat((q_normalized, dq_normalized, last_action), dim=-1)


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
        upper_intent_dim=DYNAMIC_INTENT_DIM,
        gait_phase_dim=GAIT_PHASE_DIM,
        coordination_hidden_dim=32,
        coordination_output_scale=0.10,
        coordination_blend=1.0,
        intent_gate_scale_rad=0.02,
        locomotion_gate_scale=0.10,
        response_adapter_enabled=False,
        locomotion_intent_only=False,
        transition_adapter_enabled=False,
        transition_output_scale=0.03,
        response_output_scale=0.05,
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
        if not 0.0 < response_output_scale <= coordination_output_scale:
            raise ValueError(
                "response output scale must lie in (0, coordination output scale]"
            )

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
        self.response_adapter_enabled = bool(response_adapter_enabled)
        self.locomotion_intent_only = bool(locomotion_intent_only)
        self.transition_adapter_enabled = bool(transition_adapter_enabled)
        self.transition_output_scale = float(transition_output_scale)
        if not 0.0 < self.transition_output_scale <= coordination_output_scale:
            raise ValueError(
                "transition output scale must lie in (0, coordination output scale]"
            )
        self.response_output_scale = float(response_output_scale)
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
        if self.response_adapter_enabled:
            for parameter in self.coordination_adapter.parameters():
                parameter.requires_grad_(False)
            self.response_adapter = nn.Sequential(
                nn.Linear(
                    RESPONSE_CONTEXT_DIM
                    + LOCOMOTION_INTENT_DIM
                    + self.gait_phase_dim,
                    int(coordination_hidden_dim),
                ),
                nn.ELU(),
                nn.Linear(int(coordination_hidden_dim), NUM_COORDINATION_MODES),
            )
            nn.init.zeros_(self.response_adapter[-1].weight)
            nn.init.zeros_(self.response_adapter[-1].bias)
            if self.locomotion_intent_only:
                for parameter in self.response_adapter.parameters():
                    parameter.requires_grad_(False)
                first_weight = self.response_adapter[0].weight
                first_weight.requires_grad_(True)
                gradient_mask = torch.zeros_like(first_weight)
                start = RESPONSE_CONTEXT_DIM
                gradient_mask[:, start : start + LOCOMOTION_INTENT_DIM] = 1.0
                self.register_buffer(
                    "_response_gradient_mask",
                    gradient_mask,
                    persistent=False,
                )
                first_weight.register_hook(
                    lambda gradient: gradient * self._response_gradient_mask
                )
        else:
            self.response_adapter = None
        if self.transition_adapter_enabled:
            if self.response_adapter is None:
                raise ValueError("transition adapter requires the response adapter")
            for parameter in self.response_adapter.parameters():
                parameter.requires_grad_(False)
            self.transition_adapter = nn.Sequential(
                nn.Linear(LOCOMOTION_INTENT_DIM + self.gait_phase_dim, 16),
                nn.ELU(),
                nn.Linear(16, NUM_COORDINATION_MODES),
            )
            nn.init.zeros_(self.transition_adapter[-1].weight)
            nn.init.zeros_(self.transition_adapter[-1].bias)
        else:
            self.transition_adapter = None
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
        if self.locomotion_intent_only:
            print("Response adapter update mask: locomotion-intent input columns only")
        if self.transition_adapter_enabled:
            print("Transition adapter: locomotion intent + gait phase, no yaw modes")

    def _mean_from_actor_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        expected = self.base_actor_obs_dim + self.upper_intent_dim
        if actor_obs.shape[-1] != expected:
            raise RuntimeError(
                f"unexpected future-intent observation shape: {tuple(actor_obs.shape)}"
            )
        base_obs = actor_obs[..., : self.base_actor_obs_dim]
        intent = actor_obs[..., self.base_actor_obs_dim :]
        upper_intent = intent[..., :UPPER_INTENT_DIM]
        locomotion_intent = intent[..., UPPER_INTENT_DIM:]
        if self.adapter_mode in {"disabled", "current"}:
            upper_intent = torch.cat(
                (
                    upper_intent[..., : UPPER_INTENT_DIM // 2],
                    torch.zeros_like(upper_intent[..., UPPER_INTENT_DIM // 2 :]),
                ),
                dim=-1,
            )
        intent = torch.cat((upper_intent, locomotion_intent), dim=-1)
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
            torch.amax(torch.abs(upper_intent), dim=-1, keepdim=True)
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
        future_residual = (
            self.coordination_blend * intent_gate * locomotion_gate * residual
        )
        response_residual = torch.zeros_like(future_residual)
        if self.response_adapter is not None:
            response_input = torch.cat(
                (lower_response_context(base_obs), locomotion_intent, phase),
                dim=-1,
            )
            response_coefficients = torch.tanh(self.response_adapter(response_input))
            response_mode_mask = torch.as_tensor(
                RESPONSE_MODE_MASK,
                device=response_coefficients.device,
                dtype=response_coefficients.dtype,
            )
            response_coefficients = response_coefficients * response_mode_mask
            response_residual = torch.clamp(
                response_coefficients @ self.coordination_basis,
                min=-self.response_output_scale,
                max=self.response_output_scale,
            ) * locomotion_gate
        transition_residual = torch.zeros_like(future_residual)
        if self.transition_adapter is not None:
            transition_coefficients = torch.tanh(
                self.transition_adapter(torch.cat((locomotion_intent, phase), dim=-1))
            )
            transition_mode_mask = torch.as_tensor(
                RESPONSE_MODE_MASK,
                device=transition_coefficients.device,
                dtype=transition_coefficients.dtype,
            )
            transition_coefficients = transition_coefficients * transition_mode_mask
            transition_gate = torch.clamp(
                torch.abs(locomotion_intent[..., 1:2]) / 0.10,
                min=0.0,
                max=1.0,
            )
            transition_residual = torch.clamp(
                transition_coefficients @ self.coordination_basis,
                min=-self.transition_output_scale,
                max=self.transition_output_scale,
            ) * transition_gate
        combined_residual = torch.clamp(
            future_residual + response_residual + transition_residual,
            min=-self.coordination_output_scale,
            max=self.coordination_output_scale,
        )
        self._last_coordination_residual = combined_residual
        return self.actor(base_obs) + combined_residual

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
        state_dict = dict(state_dict)
        migrations = (
            ("coordination_adapter.0.weight", UPPER_INTENT_DIM),
            ("response_adapter.0.weight", RESPONSE_CONTEXT_DIM),
        )
        current = nn.Module.state_dict(self)
        for name, prefix_dim in migrations:
            if name not in state_dict or name not in current:
                continue
            old_weight = state_dict[name]
            new_weight = current[name]
            if old_weight.shape == new_weight.shape:
                continue
            if (
                old_weight.shape[0] != new_weight.shape[0]
                or old_weight.shape[1] + LOCOMOTION_INTENT_DIM
                != new_weight.shape[1]
            ):
                continue
            migrated = torch.zeros_like(new_weight)
            migrated[:, :prefix_dim] = old_weight[:, :prefix_dim]
            migrated[:, prefix_dim + LOCOMOTION_INTENT_DIM :] = old_weight[
                :, prefix_dim:
            ]
            state_dict[name] = migrated
        incompatible = nn.Module.load_state_dict(self, state_dict, strict=False)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith(
                ("coordination_adapter.", "response_adapter.", "transition_adapter.")
            )
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "future-intent checkpoint contract mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        return True
