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
        return cfg

    def manager_env_factory(*args, **kwargs):
        cfg = kwargs.get("cfg")
        if cfg is None:
            raise RuntimeError("Stage6 requires ManagerBasedRLEnv(cfg=...)")
        cfg.commands.base_velocity.ranges.lin_vel_x = (
            velocity_min,
            velocity_max,
        )
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
