#!/usr/bin/env python3
"""Run legacy Stage172 PPO with the local Stage6 environment and adapter."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def gain_randomization_range() -> tuple[float, float] | None:
    """Read the opt-in Stage6 lower-body PD curriculum contract."""
    lower = os.environ.get("CWI_STAGE6_GAIN_MIN")
    upper = os.environ.get("CWI_STAGE6_GAIN_MAX")
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError("both CWI_STAGE6_GAIN_MIN and CWI_STAGE6_GAIN_MAX are required")
    bounds = (float(lower), float(upper))
    if not 0.0 < bounds[0] <= bounds[1]:
        raise ValueError("Stage6 gain range must satisfy 0 < min <= max")
    return bounds


def optional_positive_float(name: str) -> float | None:
    """Read an optional strictly-positive scalar override."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = float(raw)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


def optional_positive_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


ROOT = Path(__file__).resolve().parents[1]
OLD_SCRIPT = Path(
    os.environ.get(
        "CWI_STAGE6_BASE_TRAIN",
        "/home/humanplus/x2_teleop_final/x2_sonic/scripts/"
        "train_x2_stage172_lower_velocity.py",
    )
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("_cwi_stage6_base_train", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy trainer: {OLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from cwi_x2.future_intent import (
        X2FutureIntentActorCriticCfg,
        X2FutureIntentFlatEnvCfg,
    )
    from cwi_x2.future_intent_actor_critic import FutureIntentActorCritic
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    mode = os.environ.get("CWI_STAGE6_ADAPTER_MODE", "future")
    coordination_blend = float(
        os.environ.get("CWI_STAGE6_COORDINATION_BLEND", "1.0")
    )
    velocity_min = float(os.environ.get("CWI_STAGE6_VELOCITY_MIN", "0.20"))
    velocity_max = float(os.environ.get("CWI_STAGE6_VELOCITY_MAX", "0.45"))
    gain_range = gain_randomization_range()
    learning_rate = optional_positive_float("CWI_STAGE6_LEARNING_RATE")
    desired_kl = optional_positive_float("CWI_STAGE6_DESIRED_KL")
    response_adapter_enabled = env_flag("CWI_STAGE6_RESPONSE_ADAPTER")
    locomotion_intent_only = env_flag("CWI_STAGE6_LOCOMOTION_INTENT_ONLY")
    transition_adapter_enabled = env_flag("CWI_STAGE6_TRANSITION_ADAPTER")
    transition_curriculum_enabled = env_flag(
        "CWI_STAGE6_TRANSITION_CURRICULUM"
    )
    steps_per_env = optional_positive_int("CWI_STAGE6_STEPS_PER_ENV")
    transition_phase_offset_max_s = float(
        os.environ.get("CWI_STAGE6_TRANSITION_PHASE_OFFSET_MAX_S", "0.0")
    )
    if transition_phase_offset_max_s < 0.0:
        raise ValueError("transition phase offset maximum must be non-negative")
    if not 0.0 <= velocity_min <= velocity_max:
        raise ValueError(
            "CWI Stage6 velocity range must satisfy 0 <= min <= max"
        )
    if not 0.0 <= coordination_blend <= 1.0:
        raise ValueError("CWI Stage6 coordination blend must lie in [0, 1]")
    original_runner_cfg = module.X2LowerVelocityFlatPPORunnerCfg
    original_manager_env = module.ManagerBasedRLEnv

    def runner_cfg_factory():
        cfg = original_runner_cfg()
        cfg.save_interval = int(os.environ.get("CWI_STAGE6_SAVE_INTERVAL", "1"))
        if steps_per_env is not None:
            cfg.num_steps_per_env = steps_per_env
        if learning_rate is not None:
            cfg.algorithm.learning_rate = learning_rate
        if desired_kl is not None:
            cfg.algorithm.desired_kl = desired_kl
        return cfg

    def policy_cfg_factory():
        cfg = X2FutureIntentActorCriticCfg()
        cfg.class_name = "ResponseHistoryActorCritic"
        cfg.adapter_mode = mode
        cfg.coordination_blend = coordination_blend
        cfg.response_adapter_enabled = response_adapter_enabled
        cfg.locomotion_intent_only = locomotion_intent_only
        cfg.transition_adapter_enabled = transition_adapter_enabled
        return cfg

    def manager_env_factory(*args, **kwargs):
        cfg = kwargs.get("cfg")
        if cfg is None:
            raise RuntimeError("Stage6 requires ManagerBasedRLEnv(cfg=...)")
        cfg.commands.base_velocity.ranges.lin_vel_x = (
            velocity_min,
            velocity_max,
        )
        if transition_curriculum_enabled:
            from cwi_x2.transition_command import (
                stopped_base_speed_l2,
                transition_velocity_cfg,
            )

            source_command = cfg.commands.base_velocity
            cfg.commands.base_velocity = transition_velocity_cfg(
                source_command,
                ideal_env_fraction=float(
                    getattr(source_command, "ideal_env_fraction", 0.0)
                ),
                ideal_heading_control_stiffness=float(
                    getattr(
                        source_command,
                        "ideal_heading_control_stiffness",
                        source_command.heading_control_stiffness,
                    )
                ),
                response_heading_control_stiffness=float(
                    getattr(
                        source_command,
                        "response_heading_control_stiffness",
                        source_command.heading_control_stiffness,
                    )
                ),
                maximum_phase_offset_s=transition_phase_offset_max_s,
            )
            cfg.episode_length_s = 12.0
            cfg.rewards.stand_lin_vel_xy_l2.func = stopped_base_speed_l2
            cfg.rewards.stand_lin_vel_xy_l2.weight = -3.0
            cfg.rewards.stand_lin_vel_xy_l2.params = {
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot"),
            }
        if gain_range is not None:
            cfg.events.randomize_actuator_gains = EventTermCfg(
                func=mdp.randomize_actuator_gains,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=[
                            ".*_hip_.*_joint",
                            ".*_knee_joint",
                            ".*_ankle_.*_joint",
                            "waist_.*_joint",
                        ],
                    ),
                    "stiffness_distribution_params": gain_range,
                    "damping_distribution_params": gain_range,
                    "operation": "scale",
                    "distribution": "uniform",
                },
            )
        return original_manager_env(*args, **kwargs)

    module.X2LowerVelocityFlatPPORunnerCfg = runner_cfg_factory
    module.X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg = (
        X2FutureIntentFlatEnvCfg
    )
    module.X2ResponseHistoryActorCriticCfg = policy_cfg_factory
    module.ResponseHistoryActorCritic = FutureIntentActorCritic
    module.ManagerBasedRLEnv = manager_env_factory
    # Keep all generated logs/checkpoints inside this new project.
    module.__file__ = str(Path(__file__).resolve())
    try:
        module.main()
    finally:
        module.simulation_app.close()


if __name__ == "__main__":
    main()
