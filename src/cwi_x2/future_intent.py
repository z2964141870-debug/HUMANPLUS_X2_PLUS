"""Isaac environment contract for the Stage6 future-intent adapter."""

from __future__ import annotations

import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from gear_sonic.envs.x2_velocity.flat_env_cfg import (
    X2LowerTeacherPhaseObservationsCfg,
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg,
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg_PLAY,
)

from .future_intent_actor_critic import (
    BASE_ACTOR_OBS_DIM,
    DYNAMIC_INTENT_DIM,
    GAIT_PHASE_DIM,
    LOCOMOTION_INTENT_DIM,
    UPPER_INTENT_DIM,
    FutureIntentActorCritic,
)


def dynamic_intent_observation(
    env,
    upper_horizon_s: float = 0.6,
    locomotion_horizon_s: float = 1.0,
) -> torch.Tensor:
    """Return upper-body intent plus current/future locomotion intent."""
    action_term = env.action_manager._terms["joint_pos"]
    method = getattr(action_term, "_cwi_upper_intent_features", None)
    if method is None:
        raise RuntimeError(
            "future-intent observation requires the opt-in CWI upper-motion hook"
        )
    upper = method(upper_horizon_s)
    if upper.shape != (env.num_envs, UPPER_INTENT_DIM):
        raise RuntimeError(
            f"upper-intent observation shape mismatch: {tuple(upper.shape)}"
        )
    command_term = env.command_manager._terms["base_velocity"]
    locomotion_method = getattr(command_term, "locomotion_intent_features", None)
    if locomotion_method is None:
        command = env.command_manager.get_command("base_velocity")[:, :1]
        locomotion = torch.cat((command / 0.5, torch.zeros_like(command)), dim=-1)
    else:
        locomotion = locomotion_method(locomotion_horizon_s)
    if locomotion.shape != (env.num_envs, LOCOMOTION_INTENT_DIM):
        raise RuntimeError(
            f"locomotion-intent observation shape mismatch: {tuple(locomotion.shape)}"
        )
    return torch.cat((upper, locomotion), dim=-1)


@configclass
class X2FutureIntentObservationsCfg:
    """Stage208 observations with a deployable 28-D intent suffix."""

    @configclass
    class PolicyCfg(X2LowerTeacherPhaseObservationsCfg.PolicyCfg):
        dynamic_intent = ObsTerm(
            func=dynamic_intent_observation,
            params={"upper_horizon_s": 0.6, "locomotion_horizon_s": 1.0},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: X2LowerTeacherPhaseObservationsCfg.CriticCfg = (
        X2LowerTeacherPhaseObservationsCfg.CriticCfg()
    )


@configclass
class X2FutureIntentFlatEnvCfg(X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg):
    """Training environment for the future-intent coordination adapter."""

    observations: X2FutureIntentObservationsCfg = X2FutureIntentObservationsCfg()


@configclass
class X2FutureIntentFlatEnvCfg_PLAY(
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg_PLAY
):
    """Deterministic playback with the same future-intent observation suffix."""

    observations: X2FutureIntentObservationsCfg = X2FutureIntentObservationsCfg()


@configclass
class X2FutureIntentActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for :class:`FutureIntentActorCritic`."""

    class_name = "FutureIntentActorCritic"
    init_noise_std = 0.4
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [256, 128, 128]
    critic_hidden_dims = [256, 128, 128]
    activation = "elu"
    base_actor_obs_dim: int = BASE_ACTOR_OBS_DIM
    upper_intent_dim: int = DYNAMIC_INTENT_DIM
    gait_phase_dim: int = GAIT_PHASE_DIM
    coordination_hidden_dim: int = 32
    coordination_output_scale: float = 0.10
    coordination_blend: float = 1.0
    intent_gate_scale_rad: float = 0.02
    response_adapter_enabled: bool = False
    locomotion_intent_only: bool = False
    transition_adapter_enabled: bool = False
    transition_output_scale: float = 0.03
    response_output_scale: float = 0.05
    adapter_mode: str = "future"
