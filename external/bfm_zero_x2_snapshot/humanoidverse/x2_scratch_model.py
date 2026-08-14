"""A 3090-sized BFM-Zero model factory for X2 scratch experiments."""

from __future__ import annotations

import gymnasium
import numpy as np

from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.fb_cpr_aux.agent import (
    FBcprAuxAgentConfig,
    FBcprAuxAgentTrainConfig,
)
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from humanoidverse.x2_scratch import (
    ACTION_DIM,
    COMMAND_DIM,
    GAIT_PHASE_DIM,
    STATE_DIM,
    X2_AUX_REWARD_SCALES,
    X2ScratchDataConfig,
)


def x2_scratch_observation_space(
    config: X2ScratchDataConfig = X2ScratchDataConfig(),
    *,
    include_command_phase: bool = False,
) -> gymnasium.spaces.Dict:
    infinite = lambda dimension: gymnasium.spaces.Box(  # noqa: E731
        low=-np.inf,
        high=np.inf,
        shape=(dimension,),
        dtype=np.float32,
    )
    spaces = {
        "state": infinite(STATE_DIM),
        "privileged_state": infinite(STATE_DIM),
        "last_action": infinite(ACTION_DIM),
        "history_actor": infinite(config.history_actor_dim),
    }
    if include_command_phase:
        spaces["command"] = infinite(COMMAND_DIM)
        spaces["gait_phase"] = infinite(GAIT_PHASE_DIM)
    return gymnasium.spaces.Dict(spaces)


def x2_scratch_model_config(
    *,
    device: str = "cpu",
    hidden_dim: int = 256,
    hidden_layers: int = 3,
    z_dim: int = 64,
    history_length: int = 4,
    include_command_phase: bool = False,
) -> FBcprAuxModelConfig:
    """Return a reduced BFM architecture without changing the algorithm.

    The official default uses width 2048 and six residual blocks.  This
    factory keeps all BFM components and objectives but makes the first X2
    capacity smoke possible on one RTX 3090.
    """

    if device not in ("cpu", "cuda"):
        raise ValueError("BFM model device must be 'cpu' or 'cuda'")
    # Upstream residual_embedding also consumes ``hidden_layers`` and requires
    # at least two blocks, independently of embedding_layers.
    if hidden_dim < 64 or hidden_layers < 2 or z_dim < 16:
        raise ValueError("scratch network capacity is below the audited floor")
    config = X2ScratchDataConfig(history_length=history_length)
    deployable_context = ("command", "gait_phase") if include_command_phase else ()
    actor_keys = ("state", "last_action", "history_actor", *deployable_context)
    critic_keys = (
        "state", "privileged_state", "last_action", "history_actor", *deployable_context
    )
    backward_keys = ("state", "privileged_state")
    architecture = FBcprAuxModelArchiConfig(
        z_dim=z_dim,
        norm_z=True,
        f=ForwardArchiConfig(
            hidden_dim=hidden_dim,
            model="residual",
            hidden_layers=hidden_layers,
            embedding_layers=2,
            num_parallel=2,
            ensemble_mode="batch",
            input_filter=DictInputFilterConfig(key=critic_keys),
        ),
        b=BackwardArchiConfig(
            hidden_dim=hidden_dim,
            hidden_layers=2,
            norm=True,
            input_filter=DictInputFilterConfig(key=backward_keys),
        ),
        actor=ActorArchiConfig(
            model="residual",
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            embedding_layers=2,
            input_filter=DictInputFilterConfig(key=actor_keys),
        ),
        critic=ForwardArchiConfig(
            hidden_dim=hidden_dim,
            model="residual",
            hidden_layers=hidden_layers,
            embedding_layers=2,
            num_parallel=2,
            ensemble_mode="batch",
            input_filter=DictInputFilterConfig(key=critic_keys),
        ),
        discriminator=DiscriminatorArchiConfig(
            hidden_dim=hidden_dim,
            hidden_layers=2,
            input_filter=DictInputFilterConfig(key=backward_keys),
        ),
        aux_critic=ForwardArchiConfig(
            hidden_dim=hidden_dim,
            model="residual",
            hidden_layers=hidden_layers,
            embedding_layers=2,
            num_parallel=2,
            ensemble_mode="batch",
            input_filter=DictInputFilterConfig(key=critic_keys),
        ),
    )
    normalizers = {
        key: BatchNormNormalizerConfig(momentum=0.01)
        for key in x2_scratch_observation_space(
            config, include_command_phase=include_command_phase
        ).spaces
    }
    return FBcprAuxModelConfig(
        device=device,
        archi=architecture,
        obs_normalizer=ObsNormalizerConfig(
            normalizers=normalizers,
            allow_mismatching_keys=False,
        ),
        inference_batch_size=8192,
        seq_length=8,
        actor_std=0.10,
        amp=False,
        norm_aux_reward=RewardNormalizerConfig(translate=False, scale=True),
    )


def build_x2_scratch_model(**kwargs):
    config = x2_scratch_model_config(**kwargs)
    observation_space = x2_scratch_observation_space(
        X2ScratchDataConfig(history_length=kwargs.get("history_length", 4)),
        include_command_phase=kwargs.get("include_command_phase", False),
    )
    return config.build(observation_space, ACTION_DIM)


def x2_scratch_agent_config(
    *,
    device: str = "cpu",
    hidden_dim: int = 256,
    hidden_layers: int = 3,
    z_dim: int = 64,
    history_length: int = 4,
    include_command_phase: bool = False,
    batch_size: int = 32,
    use_x2_aux_rewards: bool = False,
    rollout_expert_trajectories: bool = False,
    rollout_expert_trajectories_length: int = 128,
    rollout_expert_trajectories_percentage: float = 0.5,
) -> FBcprAuxAgentConfig:
    """Return a full randomly initialized BFM agent for bounded smokes."""

    if batch_size < 8 or batch_size % 8:
        raise ValueError("scratch batch size must be a positive multiple of 8")
    if rollout_expert_trajectories_length < 8:
        raise ValueError("expert rollout length must cover at least one BFM sequence")
    if not 0.0 < rollout_expert_trajectories_percentage <= 1.0:
        raise ValueError("expert rollout percentage must be in (0, 1]")
    return FBcprAuxAgentConfig(
        model=x2_scratch_model_config(
            device=device,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            z_dim=z_dim,
            history_length=history_length,
            include_command_phase=include_command_phase,
        ),
        train=FBcprAuxAgentTrainConfig(
            lr_f=3.0e-4,
            lr_b=1.0e-5,
            lr_actor=3.0e-4,
            lr_critic=3.0e-4,
            lr_discriminator=1.0e-5,
            lr_aux_critic=3.0e-4,
            weight_decay=0.0,
            clip_grad_norm=0.0,
            fb_target_tau=0.01,
            critic_target_tau=0.005,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            expert_asm_ratio=0.6,
            relabel_ratio=0.8,
            fb_pessimism_penalty=0.0,
            critic_pessimism_penalty=0.5,
            actor_pessimism_penalty=0.5,
            aux_critic_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=batch_size,
            discount=0.98,
            use_mix_rollout=True,
            update_z_every_step=100,
            z_buffer_size=1024,
            rollout_expert_trajectories=rollout_expert_trajectories,
            rollout_expert_trajectories_length=rollout_expert_trajectories_length,
            rollout_expert_trajectories_percentage=rollout_expert_trajectories_percentage,
            reg_coeff=0.05,
            reg_coeff_aux=0.02,
            scale_reg=True,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
        ),
        aux_rewards=list(X2_AUX_REWARD_SCALES) if use_x2_aux_rewards else [],
        aux_rewards_scaling=dict(X2_AUX_REWARD_SCALES) if use_x2_aux_rewards else {},
        cudagraphs=False,
        compile=False,
    )


def parameter_summary(model) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "actor": sum(parameter.numel() for parameter in model._actor.parameters()),
        "forward": sum(parameter.numel() for parameter in model._forward_map.parameters()),
        "backward": sum(parameter.numel() for parameter in model._backward_map.parameters()),
        "critic": sum(parameter.numel() for parameter in model._critic.parameters()),
        "aux_critic": sum(parameter.numel() for parameter in model._aux_critic.parameters()),
        "discriminator": sum(parameter.numel() for parameter in model._discriminator.parameters()),
    }
