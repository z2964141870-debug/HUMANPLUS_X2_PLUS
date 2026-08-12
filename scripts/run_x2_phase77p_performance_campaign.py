#!/usr/bin/env python3
"""Train or evaluate the outcome-first X2 posture performance pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/home/yu/x2_teleop_final/x2_sonic")
SOURCE_RUN = OLD / (
    "logs/rsl_rl/x2_lower_velocity_flat/"
    "2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_"
    "selfoff_resume2550_to2650_v1"
)
ENV_YAML = SOURCE_RUN / "params/env.yaml"
AGENT_YAML = SOURCE_RUN / "params/agent.yaml"
TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER = ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
CANONICAL_ONNX = ROOT.parent / "x2_official_rl_deploy_v1/models/stage219_s2600_actor.onnx"
SOURCE_SHA256 = "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("train", "eval"), required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--candidate", type=Path)
parser.add_argument("--resume", type=Path)
parser.add_argument("--resume-report", type=Path)
parser.add_argument("--resume-gate", type=Path)
parser.add_argument("--checkpoint-output", type=Path)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--start-update", type=int, default=0)
parser.add_argument("--updates", type=int, default=5)
parser.add_argument("--steps-per-env", type=int, default=48)
parser.add_argument("--eval-steps", type=int, default=512)
parser.add_argument("--lane", choices=("A", "B"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.report.exists():
    raise FileExistsError(f"refusing to overwrite report: {args.report}")
if args.mode == "train":
    if args.candidate is not None or args.checkpoint_output is None:
        raise ValueError("train mode requires --checkpoint-output and forbids --candidate")
    if args.checkpoint_output.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint: {args.checkpoint_output}"
        )
    if args.num_envs != 256:
        raise ValueError("the confirmatory pilot is frozen to 256 training envs")
    if args.start_update not in (0, 5) or args.updates != 5:
        raise ValueError("the pilot is frozen to segments 0->5 and 5->10")
    if args.steps_per_env != 48:
        raise ValueError("the pilot is frozen to 48 steps per environment")
    continuation = (args.resume, args.resume_report, args.resume_gate)
    if args.start_update == 0 and any(value is not None for value in continuation):
        raise ValueError("the 0->5 segment forbids all resume inputs")
    if args.start_update == 5 and any(value is None for value in continuation):
        raise ValueError("the 5->10 segment requires checkpoint, report, and gate")
else:
    if (
        args.candidate is None
        or args.resume is not None
        or args.resume_report is not None
        or args.resume_gate is not None
        or args.checkpoint_output is not None
    ):
        raise ValueError("eval mode requires --candidate and forbids resume/output checkpoint")
    if args.num_envs != 128 or args.eval_steps != 512 or args.lane is None:
        raise ValueError("evaluation is frozen to 128 envs and 512 control steps")


EXPECTED_PRELAUNCH = {
    args.source.resolve(): SOURCE_SHA256,
    ENV_YAML: "f702a358bdbc1df94ac2a54b83aa4f6d7c98c76b091ac05a65fd074c44e6f9d7",
    AGENT_YAML: "38d462ad726e0e74d797f8a0ce3799aaadc02737e6e14a5cdf3443f7da0a8368",
    TEMPLATE: "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d",
    UPPER: "71db36d0206c44da05640f6e3616f918945524e891a5df8051533fcbeb2ab2ef",
    CANONICAL_ONNX: "b95bad3680658c7c25be50f236f070c80b7ff7ba8992355cec2ddfb1ee53c0f9",
}


def prelaunch_guard() -> None:
    sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    if not args.prereg.is_file() or not sidecar.is_file():
        raise FileNotFoundError(args.prereg)
    actual = sha256(args.prereg)
    if sidecar.read_text(encoding="utf-8") != f"{actual}  {args.prereg.name}\n":
        raise RuntimeError("prereg sidecar mismatch")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_phase77p_performance_campaign_prereg_v1":
        raise RuntimeError("prereg schema mismatch")
    for section in ("immutable_code", "immutable_evidence"):
        for relative, expected in prereg[section].items():
            target = ROOT / relative
            if not target.is_file() or sha256(target) != expected:
                raise RuntimeError(f"{section} mismatch: {relative}")
    for absolute, expected in prereg["immutable_inputs"].items():
        target = Path(absolute)
        if not target.is_file() or sha256(target) != expected:
            raise RuntimeError(f"immutable input mismatch: {absolute}")
    for target in (args.resume, args.resume_report, args.resume_gate, args.candidate):
        if target is not None:
            if not target.is_file():
                raise FileNotFoundError(target)
            target_sidecar = target.with_suffix(target.suffix + ".sha256")
            if not target_sidecar.is_file() or target_sidecar.read_text(
                encoding="utf-8"
            ) != f"{sha256(target)}  {target.name}\n":
                raise RuntimeError(f"dynamic input sidecar mismatch: {target}")


prelaunch_guard()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)
from gear_sonic.envs.x2_velocity import (  # noqa: E402
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg,
)
from gear_sonic.envs.x2_velocity.rsl_rl_ppo_cfg import (  # noqa: E402
    X2LowerVelocityFlatPPORunnerCfg,
)
from cwi_x2.transition_command import (  # noqa: E402
    TransitionVelocityCommand,
    transition_velocity_cfg,
)
from cwi_x2.phase77p_performance_campaign import (  # noqa: E402
    AnchoredUpdateLimits,
    anchored_ppo_update,
    clone_frozen_source,
    inject_actor_lora_train_critic,
    posture_reward_weight,
    source_retention,
    state_hash,
    tensor_hash,
    trainable_parameter_groups,
)
from x2_native_locomotion_posture_phase60 import (  # noqa: E402
    actual_support_com_outside_distance,
    actual_support_com_penalty,
    signed_backward_pitch_penalty,
    signed_root_pitch_rad,
)


EXPECTED = EXPECTED_PRELAUNCH
LOWER15 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)
LIMITS = AnchoredUpdateLimits()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(path)
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists() or path.exists():
        raise FileExistsError(f"checkpoint target or temporary exists: {path}")
    torch.save(payload, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    metadata_keys = (
        "schema",
        "source_checkpoint_sha256",
        "source_iteration",
        "update_index",
        "train_seed",
        "campaign_config",
        "lora_manifest",
    )
    metadata_ok = all(reloaded.get(key) == payload.get(key) for key in metadata_keys)
    model_ok = (
        reloaded.get("model_state_dict", {}).keys()
        == payload.get("model_state_dict", {}).keys()
        and all(
            torch.equal(
                reloaded["model_state_dict"][name],
                value.detach().cpu(),
            )
            for name, value in payload["model_state_dict"].items()
        )
    )
    optimizer_ok = len(reloaded.get("optimizer_state_dict", {}).get("param_groups", [])) == 2
    if not (metadata_ok and model_ok and optimizer_ok):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("strict checkpoint reload failed")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_inputs() -> None:
    mismatches = []
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(str(path))
    if mismatches:
        raise RuntimeError(f"immutable input mismatch: {mismatches}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.candidate is not None and not args.candidate.is_file():
        raise FileNotFoundError(args.candidate)


def _configure_fixed_upper_and_ideal_actuator(cfg) -> None:
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = str(TEMPLATE)
    # Current source defaults have drifted to 0.20.  Source identity is 0.15.
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    _apply_x2_actuator_response(
        cfg.scene.robot,
        {
            "enabled": True,
            "profile": "session03_session04_group",
            "randomize": False,
            "strength": 1.0,
            "filter_strength": 1.0,
            "delay_strength": 1.0,
            "include_ideal_endpoint": False,
            "ideal_env_fraction": 1.0,
            "filter_only_env_fraction": 0.0,
        },
        physics_dt_sec=cfg.sim.dt,
    )


def _configure_reset_panel(cfg, *, evaluation: bool) -> None:
    magnitude = 0.006 if evaluation else 0.010
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "roll": (-magnitude, magnitude),
        "pitch": (-magnitude, magnitude),
        "yaw": (-0.02, 0.02),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "z": (0.0, 0.0),
        "roll": (-0.01, 0.01),
        "pitch": (-0.01, 0.01),
        "yaw": (-0.01, 0.01),
    }
    cfg.events.reset_robot_joints.params["position_range"] = (0.9975, 1.0025)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-0.01, 0.01)


class PerformanceTransitionVelocityCommand(TransitionVelocityCommand):
    """Pair-balanced fixed command panel with one transition stratum.

    Adjacent environments always share a command role.  This makes the eval
    A/B lane swap orthogonal to speed/turn/transition allocation.  Training
    transition pairs receive fixed phase offsets on an eight-second cycle so
    every rollout sees the same phase mixture; evaluation transition pairs run
    one aligned start/cruise/stop event with a terminal hold.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        self._performance_pair_id = torch.div(ids, 2, rounding_mode="floor")
        self._performance_role = self._performance_pair_id.remainder(8)
        self._performance_turn_rate = torch.where(
            torch.div(self._performance_pair_id, 8, rounding_mode="floor")
            .remainder(2)
            .bool(),
            torch.full((self.num_envs,), 0.15, device=self.device),
            torch.full((self.num_envs,), -0.15, device=self.device),
        )
        transition_block = torch.div(
            self._performance_pair_id, 8, rounding_mode="floor"
        )
        block_count = max(1, self.num_envs // 16)
        self._performance_transition_offset = (
            (transition_block.to(torch.float32) + 0.5)
            * (8.0 / float(block_count))
        )
        self._freeze_sampled_commands(torch.arange(self.num_envs, device=self.device))

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if hasattr(self, "_performance_role"):
            self._freeze_sampled_commands(self._resolved_env_ids(env_ids))

    def _freeze_sampled_commands(self, env_ids):
        role = self._performance_role[env_ids]
        speed = torch.full_like(role, 0.35, dtype=torch.float32)
        speed = torch.where(role <= 1, torch.full_like(speed, 0.20), speed)
        speed = torch.where(
            (role >= 4) & (role <= 5), torch.full_like(speed, 0.50), speed
        )
        self.cruise_speed[env_ids] = speed
        self.vel_command_b[env_ids, 0] = speed
        self.vel_command_b[env_ids, 1:] = 0.0

    def _transition_elapsed(self):
        elapsed = (
            self._env.episode_length_buf.to(torch.float32)
            * float(self._env.step_dt)
        )
        if args.mode == "train":
            return torch.remainder(
                elapsed + self._performance_transition_offset,
                8.0,
            )
        return elapsed

    def _update_command(self):
        super()._update_command()
        if not hasattr(self, "_performance_role"):
            return
        role = self._performance_role
        speed = self.cruise_speed.clone()
        transition = role == 7
        speed[transition] = self._scheduled_speed(
            self._transition_elapsed()
        )[transition]
        self.vel_command_b[:, 0] = speed
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = torch.where(
            (role == 6) & (speed.abs() > 0.05),
            self._performance_turn_rate,
            torch.zeros_like(self._performance_turn_rate),
        )

    def terminal_stop_mask(self):
        return (self._performance_role == 7) & (self._transition_elapsed() >= 7.2)

    def command_role(self):
        return self._performance_role


def build_env_cfg(*, evaluation: bool):
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    _configure_fixed_upper_and_ideal_actuator(cfg)
    _configure_reset_panel(cfg, evaluation=evaluation)

    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    transition = transition_velocity_cfg(
        cfg.commands.base_velocity,
        ideal_env_fraction=1.0,
        ideal_heading_control_stiffness=1.0,
        response_heading_control_stiffness=1.0,
        stand_s=0.0,
        accelerate_s=1.0,
        cruise_s=4.2,
        decelerate_s=2.0,
        maximum_phase_offset_s=0.0,
    )
    transition.class_type = PerformanceTransitionVelocityCommand
    cfg.commands.base_velocity = transition
    cfg.episode_length_s = 12.0

    cfg.rewards.track_lin_vel_xy_exp.weight = 4.0
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
    cfg.rewards.track_ang_vel_z_exp.weight = 1.0
    cfg.rewards.track_ang_vel_z_exp.params["std"] = 1.0
    cfg.rewards.yaw_rate_l2.weight = -0.5
    cfg.rewards.feet_air_time.weight = 1.0
    cfg.rewards.feet_slide.weight = -0.20
    cfg.rewards.contact_dwell.weight = -1.0
    cfg.rewards.contact_phase.weight = -1.0
    cfg.rewards.heading_error_l2.weight = 0.0
    cfg.rewards.signed_backward_pitch = RewTerm(
        func=signed_backward_pitch_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "tolerance_rad": 0.05,
            "normalization_rad": 0.15,
            "command_threshold_mps": 0.10,
            "asset_name": "robot",
        },
    )
    cfg.rewards.actual_support_com = RewTerm(
        func=actual_support_com_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "normalization_m": 0.10,
            "force_threshold_n": 10.0,
            "command_threshold_mps": 0.10,
        },
    )
    return cfg


def build_dense_model(obs, device):
    return ActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=15,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        init_noise_std=0.4,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    ).to(device)


def validate_live_contract(env, wrapped) -> dict:
    obs = wrapped.get_observations()
    if tuple(obs["policy"].shape) != (args.num_envs, 93):
        raise RuntimeError("policy observation contract is not [num_envs,93]")
    if tuple(obs["critic"].shape) != (args.num_envs, 93):
        raise RuntimeError("critic observation contract is not [num_envs,93]")
    if wrapped.num_actions != 15:
        raise RuntimeError("action contract is not 15D")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15:
        raise RuntimeError("lower12+waist3 action order changed")
    if float(env.cfg.actions.joint_pos.template_scale) != 0.15:
        raise RuntimeError("gait template scale drifted from Stage219 identity")
    zero_mask = term._cwi_upper_zero_mask
    if not bool(zero_mask.all()):
        raise RuntimeError("performance pilot requires fixed upper body")
    actuator = env.scene["robot"].actuators["legs"]
    alpha = actuator._position_alpha.reshape(args.num_envs, -1)[:, 0]
    lag = actuator.positions_delay_buffer.time_lags.reshape(args.num_envs)
    ideal = torch.isclose(alpha, torch.ones_like(alpha)) & (lag == 0)
    if not bool(ideal.all()):
        raise RuntimeError("posture pilot requires the ideal actuator domain")
    roles = env.command_manager.get_term("base_velocity").command_role()
    expected_per_role = args.num_envs // 8
    if torch.bincount(roles, minlength=8).tolist() != [expected_per_role] * 8:
        raise RuntimeError("command strata are not balanced across all eight roles")
    return obs


def load_dense_source(obs, device):
    payload = torch.load(args.source, map_location=device, weights_only=False)
    if int(payload.get("iter", -1)) != 2600:
        raise RuntimeError("source checkpoint iteration is not 2600")
    model = build_dense_model(obs, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.std.requires_grad_(False)
    return model, payload


def set_posture_weights(env, update_index: int) -> float:
    weight = posture_reward_weight(update_index)
    for name in ("signed_backward_pitch", "actual_support_com"):
        cfg = copy.deepcopy(env.reward_manager.get_term_cfg(name))
        cfg.weight = weight
        env.reward_manager.set_term_cfg(name, cfg)
    realized = {
        name: float(env.reward_manager.get_term_cfg(name).weight)
        for name in ("signed_backward_pitch", "actual_support_com")
    }
    if set(realized.values()) != {weight}:
        raise RuntimeError(f"posture reward ramp failed: {realized}")
    return weight


def _quantiles(values: list[torch.Tensor]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    tensor = torch.cat([value.reshape(-1).cpu() for value in values])
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(tensor.mean()),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
    }


def training_rollout_metrics(records: dict, env_steps: int) -> dict:
    return {
        "signed_pitch_rad": _quantiles(records["pitch"]),
        "support_outside_m": _quantiles(records["support"]),
        "velocity_tracking_rmse_mps": math.sqrt(
            max(0.0, float(torch.cat(records["velocity_sq"]).mean()))
        ),
        "yaw_tracking_rmse_radps": math.sqrt(
            max(0.0, float(torch.cat(records["yaw_sq"]).mean()))
        ),
        "termination_sample_fraction": records["done_count"]
        / float(args.num_envs * env_steps),
        "termination_count": records["done_count"],
        "time_outs_count": records["time_outs_count"],
        "root_height_min_m": records["root_height_min"],
        "root_tilt_max_rad": records["root_tilt_max"],
    }


def campaign_config() -> dict:
    return {
        "num_envs": 256,
        "steps_per_env": 48,
        "updates_total": 10,
        "ppo_epochs": 1,
        "ppo_minibatches": 4,
        "ppo_clip": 0.10,
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_coefficient": 0.0,
        "actor_lr": 1.0e-5,
        "critic_lr": 1.0e-4,
        "source_kl_coefficient": 0.10,
        "source_kl_mean_max": LIMITS.source_kl_mean_max,
        "source_kl_max": LIMITS.source_kl_max,
        "incremental_kl_mean_max": LIMITS.incremental_kl_mean_max,
        "incremental_kl_max": LIMITS.incremental_kl_max,
        "action_drift_max": LIMITS.action_drift_max,
        "posture_reward_weight": -0.5,
        "source_std_tensor_hash": "2d49dce85d503d6ec9cc69351cb1285f54e600bbdaed1afacf6d69d69f711c51",
        "source_std_min": 0.08823904395103455,
        "source_std_max": 0.7206376791000366,
        "command_allocation": {
            "vx_0p20": 64,
            "vx_0p35": 64,
            "vx_0p50": 64,
            "turn_vx_0p35_yaw_pm_0p15": 32,
            "transition_vx_0p35": 32,
        },
        "template_scale": 0.15,
        "fixed_upper": True,
        "ideal_actuator": True,
    }


def train() -> dict:
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(evaluation=False))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = validate_live_contract(env, wrapped)
        agent_cfg = X2LowerVelocityFlatPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        agent_cfg.num_steps_per_env = args.steps_per_env
        agent_cfg.policy.init_noise_std = 0.4
        agent_cfg.algorithm.num_learning_epochs = 1
        agent_cfg.algorithm.num_mini_batches = 4
        agent_cfg.algorithm.learning_rate = 1.0e-5
        agent_cfg.algorithm.clip_param = 0.10
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.desired_kl = None
        agent_cfg.algorithm.entropy_coef = 0.0
        agent_cfg.algorithm.gamma = 0.99
        agent_cfg.algorithm.lam = 0.95
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=None,
            device=args.device,
        )
        dense_source, source_payload = load_dense_source(obs, env.device)
        runner.alg.policy.load_state_dict(
            source_payload["model_state_dict"], strict=True
        )
        source_policy = clone_frozen_source(dense_source)
        source_state_before = state_hash(source_policy)
        source_std_before = source_policy.std.detach().clone()
        if (
            source_policy.std.numel() != 15
            or tensor_hash(source_policy.std)
            != campaign_config()["source_std_tensor_hash"]
            or float(source_policy.std.min()) != campaign_config()["source_std_min"]
            or float(source_policy.std.max()) != campaign_config()["source_std_max"]
        ):
            raise RuntimeError("source policy/std identity changed")
        lora_manifest = inject_actor_lora_train_critic(
            runner.alg.policy,
            rank=4,
            alpha=4.0,
        )
        runner.alg.policy.std.requires_grad_(False)
        runner.alg.optimizer = torch.optim.Adam(
            trainable_parameter_groups(
                runner.alg.policy,
                actor_lr=1.0e-5,
                critic_lr=1.0e-4,
            )
        )
        resume_sha = resume_report_sha = resume_gate_sha = None
        if args.resume is not None:
            resume = torch.load(args.resume, map_location=env.device, weights_only=False)
            if resume.get("schema") != "x2_phase77p_performance_checkpoint_v1":
                raise RuntimeError("resume checkpoint schema changed")
            if resume.get("source_checkpoint_sha256") != EXPECTED[args.source.resolve()]:
                raise RuntimeError("resume checkpoint source identity changed")
            if int(resume.get("update_index", -1)) != args.start_update:
                raise RuntimeError("resume checkpoint update index changed")
            if int(resume.get("train_seed", -1)) != args.seed:
                raise RuntimeError("resume checkpoint train seed changed")
            if resume.get("campaign_config") != campaign_config():
                raise RuntimeError("resume checkpoint campaign configuration changed")
            resume_report = json.loads(args.resume_report.read_text(encoding="utf-8"))
            resume_gate = json.loads(args.resume_gate.read_text(encoding="utf-8"))
            resume_sha = sha256(args.resume)
            resume_report_sha = sha256(args.resume_report)
            resume_gate_sha = sha256(args.resume_gate)
            if not (
                resume_report.get("schema") == "x2_phase77p_performance_train_v1"
                and resume_report.get("decision") == "SEGMENT_VALID"
                and int(resume_report.get("seed", -1)) == args.seed
                and int(resume_report.get("end_update", -1)) == 5
                and resume_report.get("checkpoint_sha256") == resume_sha
                and resume_report.get("campaign_config") == campaign_config()
            ):
                raise RuntimeError("resume train report does not bind the checkpoint/config")
            gate_reports = {
                item["sha256"] for item in resume_gate.get("train_reports", [])
            }
            if not (
                resume_gate.get("decision") == "PASS_UPDATE5_CONTINUE_TO_UPDATE10"
                and resume_gate.get("update10_training_unlocked") is True
                and resume_report_sha in gate_reports
                and str(args.seed) in resume_gate.get("seed_summaries", {})
                and resume_gate["seed_summaries"][str(args.seed)].get("passed") is True
            ):
                raise RuntimeError("resume local gate did not authorize this seed")
            runner.alg.policy.load_state_dict(resume["model_state_dict"], strict=True)
            runner.alg.optimizer.load_state_dict(resume["optimizer_state_dict"])

        fixed_obs = {
            "policy": obs["policy"].detach().clone(),
            "critic": obs["critic"].detach().clone(),
        }
        obs_train = wrapped.get_observations().to(args.device)
        update_records = []
        optimizer_steps_total = 0
        runner.train_mode()
        for update_index in range(
            args.start_update + 1,
            args.start_update + args.updates + 1,
        ):
            weight = set_posture_weights(env, update_index)
            records = {
                "pitch": [],
                "support": [],
                "velocity_sq": [],
                "yaw_sq": [],
                "done_count": 0,
                "time_outs_count": 0,
                "root_height_min": float("inf"),
                "root_tilt_max": 0.0,
                "policy_obs": [],
                "critic_obs": [],
            }
            with torch.inference_mode():
                for _ in range(args.steps_per_env):
                    records["policy_obs"].append(obs_train["policy"].detach().clone())
                    records["critic_obs"].append(obs_train["critic"].detach().clone())
                    action = runner.alg.act(obs_train)
                    next_obs, reward, done, extras = wrapped.step(
                        action.to(wrapped.device)
                    )
                    next_obs = next_obs.to(args.device)
                    reward = reward.to(args.device)
                    done = done.to(args.device)
                    runner.alg.process_env_step(next_obs, reward, done, extras)
                    robot = env.scene["robot"]
                    command = env.command_manager.get_command("base_velocity")
                    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
                    pitch = signed_root_pitch_rad(robot)
                    support, _ = actual_support_com_outside_distance(
                        env,
                        force_threshold_n=10.0,
                    )
                    records["pitch"].append(pitch[moving].detach().cpu())
                    records["support"].append(support[moving].detach().cpu())
                    records["velocity_sq"].append(
                        torch.square(
                            robot.data.root_lin_vel_b[:, :2] - command[:, :2]
                        )
                        .sum(-1)
                        .detach()
                        .cpu()
                    )
                    records["yaw_sq"].append(
                        torch.square(
                            robot.data.root_ang_vel_b[:, 2] - command[:, 2]
                        )
                        .detach()
                        .cpu()
                    )
                    records["done_count"] += int(done.bool().sum())
                    time_outs = extras.get("time_outs")
                    if time_outs is not None:
                        records["time_outs_count"] += int(time_outs.bool().sum())
                    records["root_height_min"] = min(
                        records["root_height_min"],
                        float(robot.data.root_pos_w[:, 2].min()),
                    )
                    quat = robot.data.root_quat_w
                    up_z = 1.0 - 2.0 * (
                        torch.square(quat[:, 1]) + torch.square(quat[:, 2])
                    )
                    records["root_tilt_max"] = max(
                        records["root_tilt_max"],
                        float(torch.acos(torch.clamp(up_z, -1.0, 1.0)).max()),
                    )
                    obs_train = next_obs
                # Installed RSL-RL evaluates V(s_T); never zero-bootstrap here.
                runner.alg.compute_returns(obs_train)

            losses = anchored_ppo_update(
                runner.alg,
                source_policy,
                source_kl_coefficient=0.10,
                limits=LIMITS,
            )
            optimizer_steps_total += int(losses["optimizer_steps"])
            fixed_retention = source_retention(
                runner.alg.policy,
                source_policy,
                fixed_obs,
            )
            rollout_obs = {
                "policy": torch.cat(records["policy_obs"], dim=0),
                "critic": torch.cat(records["critic_obs"], dim=0),
            }
            rollout_retention = source_retention(
                runner.alg.policy,
                source_policy,
                rollout_obs,
            )
            metrics = training_rollout_metrics(records, args.steps_per_env)
            gates = {
                "finite": bool(
                    losses["finite"]
                    and fixed_retention["finite"]
                    and rollout_retention["finite"]
                ),
                "post_minibatch_source_kl_mean": losses["source_kl"]
                <= LIMITS.source_kl_mean_max,
                "post_minibatch_source_kl_max": losses["source_kl_max"]
                <= LIMITS.source_kl_max,
                "post_minibatch_incremental_kl_mean": losses["incremental_kl"]
                <= LIMITS.incremental_kl_mean_max,
                "post_minibatch_incremental_kl_max": losses["incremental_kl_max"]
                <= LIMITS.incremental_kl_max,
                "rollout_source_kl_mean": rollout_retention["source_kl_mean"]
                <= LIMITS.source_kl_mean_max,
                "rollout_source_kl_max": rollout_retention["source_kl_max"]
                <= LIMITS.source_kl_max,
                "rollout_action_drift": rollout_retention["action_drift_max"]
                <= LIMITS.action_drift_max,
                "fixed_action_drift": fixed_retention["action_drift_max"]
                <= LIMITS.action_drift_max,
                "termination": metrics["termination_sample_fraction"] <= 0.02,
                "no_time_outs": metrics["time_outs_count"] == 0,
                "root_height": metrics["root_height_min_m"] >= 0.60,
                "root_tilt": metrics["root_tilt_max_rad"] <= 0.35,
                "source_immutable": state_hash(source_policy) == source_state_before,
                "candidate_std_immutable": torch.equal(
                    runner.alg.policy.std.detach(), source_std_before
                ),
            }
            update_records.append(
                {
                    "update_index": update_index,
                    "posture_reward_weight": weight,
                    "losses": losses,
                    "fixed_source_retention": fixed_retention,
                    "rollout_source_retention": rollout_retention,
                    "rollout_metrics": metrics,
                    "gates": gates,
                }
            )
            if not all(gates.values()):
                raise RuntimeError(
                    f"Phase77p update {update_index} failed trust/safety gates: {gates}"
                )

        checkpoint_payload = {
            "schema": "x2_phase77p_performance_checkpoint_v1",
            "source_checkpoint_sha256": EXPECTED[args.source.resolve()],
            "source_iteration": 2600,
            "update_index": args.start_update + args.updates,
            "train_seed": args.seed,
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "lora_manifest": lora_manifest,
            "campaign_config": campaign_config(),
            "resume_checkpoint_sha256": resume_sha,
            "resume_train_report_sha256": resume_report_sha,
            "resume_gate_sha256": resume_gate_sha,
        }
        atomic_checkpoint(args.checkpoint_output, checkpoint_payload)
        report = {
            "schema": "x2_phase77p_performance_train_v1",
            "decision": "SEGMENT_VALID",
            "source_checkpoint": str(args.source),
            "source_checkpoint_sha256": EXPECTED[args.source.resolve()],
            "resume_checkpoint": str(args.resume) if args.resume else None,
            "resume_checkpoint_sha256": resume_sha,
            "resume_train_report": str(args.resume_report) if args.resume_report else None,
            "resume_train_report_sha256": resume_report_sha,
            "resume_gate": str(args.resume_gate) if args.resume_gate else None,
            "resume_gate_sha256": resume_gate_sha,
            "checkpoint": str(args.checkpoint_output),
            "checkpoint_sha256": sha256(args.checkpoint_output),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "start_update": args.start_update,
            "end_update": args.start_update + args.updates,
            "transitions": args.num_envs * args.steps_per_env * args.updates,
            "optimizer_steps": optimizer_steps_total,
            "campaign_config": campaign_config(),
            "terminal_bootstrap": "installed RSL-RL policy.evaluate(s_T)",
            "action_contract": "93D observation -> coordinated 15D lower12+waist3 actor",
            "adapter": lora_manifest,
            "updates": update_records,
            "source_model_hash": source_state_before,
            "source_std_tensor_hash": campaign_config()["source_std_tensor_hash"],
            "source_std_min": float(source_std_before.min()),
            "source_std_max": float(source_std_before.max()),
            "candidate_std_equal_source": torch.equal(
                runner.alg.policy.std.detach(), source_std_before
            ),
            "candidate_model_hash": state_hash(runner.alg.policy),
            "checkpoint_count": 1,
        }
        return report
    finally:
        # The outer supervisor owns process lifetime.  Calling Kit close here
        # has previously hung after complete evidence was written.
        pass


def load_candidate(obs, device):
    payload = torch.load(args.candidate, map_location=device, weights_only=False)
    if payload.get("schema") != "x2_phase77p_performance_checkpoint_v1":
        raise RuntimeError("candidate checkpoint schema changed")
    if payload.get("source_checkpoint_sha256") != EXPECTED[args.source.resolve()]:
        raise RuntimeError("candidate source identity changed")
    model = build_dense_model(obs, device)
    inject_actor_lora_train_critic(model, rank=4, alpha=4.0)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.std.requires_grad_(False)
    return model, payload


def wrapped_yaw(quat):
    return torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (torch.square(quat[:, 2]) + torch.square(quat[:, 3])),
    )


ROLE_NAMES = {
    0: "vx_0p20_a",
    1: "vx_0p20_b",
    2: "vx_0p35_a",
    3: "vx_0p35_b",
    4: "vx_0p50_a",
    5: "vx_0p50_b",
    6: "turn_vx_0p35_yaw_pm_0p15",
    7: "transition_vx_0p35",
}


def _finite_values(rows: list[dict], field: str) -> list[float]:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"missing/non-finite per-env metric: {field}")
    return values


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def summarize_eval_group(rows: list[dict]) -> dict:
    terminal_rows = [row for row in rows if row["role_id"] == 7]
    velocity_mse = _finite_values(rows, "velocity_mse")
    yaw_mse = _finite_values(rows, "yaw_mse")
    return {
        "env_count": len(rows),
        "role_counts": {
            ROLE_NAMES[role]: sum(row["role_id"] == role for row in rows)
            for role in range(8)
        },
        "signed_pitch_rad": _summary(_finite_values(rows, "signed_pitch_mean_rad")),
        "support_outside_m": _summary(_finite_values(rows, "support_mean_m")),
        "stance_slip_p95_mps": _summary(_finite_values(rows, "stance_slip_p95_mps")),
        "terminal_base_speed_mps": _summary(
            _finite_values(terminal_rows, "terminal_speed_mean_mps")
        ),
        "terminal_double_support": _summary(
            _finite_values(terminal_rows, "terminal_double_support_mean")
        ),
        "velocity_tracking_rmse_mps": math.sqrt(sum(velocity_mse) / len(velocity_mse)),
        "yaw_tracking_rmse_radps": math.sqrt(sum(yaw_mse) / len(yaw_mse)),
        "survival_s_mean": sum(_finite_values(rows, "survival_s")) / len(rows),
        "survival_s_min": min(_finite_values(rows, "survival_s")),
        "termination_rate": sum(bool(row["terminated"]) for row in rows) / len(rows),
        "time_outs_count": sum(int(row["time_out"]) for row in rows),
        "root_height_min_mean": sum(_finite_values(rows, "root_height_min_m")) / len(rows),
        "root_height_min_global": min(_finite_values(rows, "root_height_min_m")),
        "root_tilt_max_mean": sum(_finite_values(rows, "root_tilt_max_rad")) / len(rows),
        "root_tilt_max_global": max(_finite_values(rows, "root_tilt_max_rad")),
    }


def evaluate() -> dict:
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(evaluation=True))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = validate_live_contract(env, wrapped)
        source, source_payload = load_dense_source(obs, env.device)
        source = clone_frozen_source(source)
        candidate, candidate_payload = load_candidate(obs, env.device)
        even = torch.arange(args.num_envs, device=env.device).remainder(2) == 0
        source_mask = even if args.lane == "A" else ~even
        candidate_mask = ~source_mask
        robot = env.scene["robot"]
        foot_ids = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )[0]
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        survival = torch.full(
            (args.num_envs,),
            args.eval_steps * env.step_dt,
            device=env.device,
        )
        terminated = torch.zeros_like(alive)
        timed_out = torch.zeros_like(alive)
        root_height_min = robot.data.root_pos_w[:, 2].clone()
        tilt_max = torch.zeros(args.num_envs, device=env.device)
        samples = {
            name: []
            for name in (
                "pitch",
                "support",
                "slip",
                "velocity_sq",
                "yaw_sq",
                "terminal_speed",
                "terminal_double_support",
            )
        }
        action_drift_max = 0.0
        with torch.inference_mode():
            for step in range(args.eval_steps):
                source_action = source.act_inference(obs)
                candidate_action = candidate.act_inference(obs)
                action_drift_max = max(
                    action_drift_max,
                    float(torch.abs(candidate_action - source_action).max()),
                )
                action = torch.where(
                    candidate_mask.unsqueeze(-1),
                    candidate_action,
                    source_action,
                ).clamp(-1.0, 1.0)
                next_obs, _, done, extras = wrapped.step(action)
                done = done.reshape(-1).bool()
                step_time_out = extras.get("time_outs")
                if step_time_out is None:
                    step_time_out = torch.zeros_like(done)
                else:
                    step_time_out = step_time_out.reshape(-1).bool()
                command = env.command_manager.get_command("base_velocity")
                moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
                valid_moving = alive & moving
                pitch = signed_root_pitch_rad(robot)
                support, contact_count = actual_support_com_outside_distance(
                    env,
                    force_threshold_n=10.0,
                )
                forces = torch.stack(
                    tuple(
                        env.scene[name]
                        .data.force_matrix_w[..., 2]
                        .abs()
                        .reshape(args.num_envs, -1)
                        .amax(dim=-1)
                        for name in (
                            "left_foot_ground_contact",
                            "right_foot_ground_contact",
                        )
                    ),
                    dim=-1,
                )
                contact = forces > 10.0
                foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
                slip = (foot_speed * contact).sum(-1) / contact_count.clamp_min(1)
                slip = torch.where(
                    contact_count > 0,
                    slip,
                    torch.full_like(slip, torch.nan),
                )
                velocity_sq = torch.square(
                    robot.data.root_lin_vel_b[:, :2] - command[:, :2]
                ).sum(-1)
                yaw_sq = torch.square(
                    robot.data.root_ang_vel_b[:, 2] - command[:, 2]
                )
                terminal = env.command_manager.get_term(
                    "base_velocity"
                ).terminal_stop_mask()

                def masked(value, mask):
                    return torch.where(
                        mask,
                        value,
                        torch.full_like(value, torch.nan),
                    ).detach().cpu()

                samples["pitch"].append(masked(pitch, valid_moving))
                samples["support"].append(masked(support, valid_moving))
                samples["slip"].append(masked(slip, alive))
                samples["velocity_sq"].append(masked(velocity_sq, alive))
                samples["yaw_sq"].append(masked(yaw_sq, alive))
                samples["terminal_speed"].append(
                    masked(
                        torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1),
                        alive & terminal,
                    )
                )
                samples["terminal_double_support"].append(
                    masked((contact_count == 2).float(), alive & terminal)
                )
                quat = robot.data.root_quat_w
                up_z = 1.0 - 2.0 * (
                    torch.square(quat[:, 1]) + torch.square(quat[:, 2])
                )
                tilt = torch.acos(torch.clamp(up_z, -1.0, 1.0))
                root_height_min = torch.where(
                    alive,
                    torch.minimum(root_height_min, robot.data.root_pos_w[:, 2]),
                    root_height_min,
                )
                tilt_max = torch.where(alive, torch.maximum(tilt_max, tilt), tilt_max)
                newly_done = alive & done
                survival[newly_done] = (step + 1) * env.step_dt
                terminated |= newly_done
                timed_out |= alive & step_time_out
                alive &= ~done
                obs = next_obs

        stacked = {name: torch.stack(values, dim=0) for name, values in samples.items()}
        roles = env.command_manager.get_term("base_velocity").command_role().cpu()

        def finite_mean(value: torch.Tensor) -> float | None:
            selected = value[torch.isfinite(value)]
            return float(selected.mean()) if selected.numel() else None

        def finite_quantile(value: torch.Tensor, q: float) -> float | None:
            selected = value[torch.isfinite(value)]
            return float(torch.quantile(selected, q)) if selected.numel() else None

        per_env = []
        for env_id in range(args.num_envs):
            role_id = int(roles[env_id])
            treatment = "source" if bool(source_mask[env_id]) else "candidate"
            per_env.append(
                {
                    "env_id": env_id,
                    "pair_id": env_id // 2,
                    "lane": args.lane,
                    "treatment": treatment,
                    "role_id": role_id,
                    "role": ROLE_NAMES[role_id],
                    "signed_pitch_mean_rad": finite_mean(stacked["pitch"][:, env_id]),
                    "support_mean_m": finite_mean(stacked["support"][:, env_id]),
                    "stance_slip_p95_mps": finite_quantile(
                        stacked["slip"][:, env_id], 0.95
                    ),
                    "velocity_mse": finite_mean(stacked["velocity_sq"][:, env_id]),
                    "yaw_mse": finite_mean(stacked["yaw_sq"][:, env_id]),
                    "terminal_speed_mean_mps": finite_mean(
                        stacked["terminal_speed"][:, env_id]
                    ),
                    "terminal_double_support_mean": finite_mean(
                        stacked["terminal_double_support"][:, env_id]
                    ),
                    "moving_sample_count": int(
                        torch.isfinite(stacked["pitch"][:, env_id]).sum()
                    ),
                    "terminal_sample_count": int(
                        torch.isfinite(stacked["terminal_speed"][:, env_id]).sum()
                    ),
                    "survival_s": float(survival[env_id]),
                    "terminated": bool(terminated[env_id]),
                    "time_out": bool(timed_out[env_id]),
                    "root_height_min_m": float(root_height_min[env_id]),
                    "root_tilt_max_rad": float(tilt_max[env_id]),
                }
            )
        source_rows = [row for row in per_env if row["treatment"] == "source"]
        candidate_rows = [row for row in per_env if row["treatment"] == "candidate"]
        groups = {
            "source": summarize_eval_group(source_rows),
            "candidate": summarize_eval_group(candidate_rows),
        }
        source_group, candidate_group = groups["source"], groups["candidate"]
        deltas = {
            "signed_pitch_mean_rad": candidate_group["signed_pitch_rad"]["mean"]
            - source_group["signed_pitch_rad"]["mean"],
            "signed_pitch_p05_rad": candidate_group["signed_pitch_rad"]["p05"]
            - source_group["signed_pitch_rad"]["p05"],
            "support_outside_mean_m": candidate_group["support_outside_m"]["mean"]
            - source_group["support_outside_m"]["mean"],
            "velocity_rmse_mps": candidate_group["velocity_tracking_rmse_mps"]
            - source_group["velocity_tracking_rmse_mps"],
            "yaw_rmse_radps": candidate_group["yaw_tracking_rmse_radps"]
            - source_group["yaw_tracking_rmse_radps"],
            "termination_rate": candidate_group["termination_rate"]
            - source_group["termination_rate"],
            "stance_slip_p95_mps": candidate_group["stance_slip_p95_mps"]["p95"]
            - source_group["stance_slip_p95_mps"]["p95"],
            "terminal_speed_mean_mps": candidate_group["terminal_base_speed_mps"]["mean"]
            - source_group["terminal_base_speed_mps"]["mean"],
            "terminal_double_support_mean": candidate_group["terminal_double_support"]["mean"]
            - source_group["terminal_double_support"]["mean"],
        }
        finite = all(
            math.isfinite(value)
            for value in deltas.values()
        ) and all(
            metric is not None and math.isfinite(metric)
            for group in groups.values()
            for summary in (
                group["signed_pitch_rad"],
                group["support_outside_m"],
            )
            for metric in summary.values()
        )
        per_env_complete = all(
            row["moving_sample_count"] > 0
            and all(
                row[field] is not None and math.isfinite(row[field])
                for field in (
                    "signed_pitch_mean_rad",
                    "support_mean_m",
                    "stance_slip_p95_mps",
                    "velocity_mse",
                    "yaw_mse",
                    "survival_s",
                    "root_height_min_m",
                    "root_tilt_max_rad",
                )
            )
            and (
                row["role_id"] != 7
                or (
                    row["terminal_sample_count"] > 0
                    and row["terminal_speed_mean_mps"] is not None
                    and row["terminal_double_support_mean"] is not None
                    and math.isfinite(row["terminal_speed_mean_mps"])
                    and math.isfinite(row["terminal_double_support_mean"])
                )
            )
            for row in per_env
        )
        finite = bool(finite and per_env_complete)
        return {
            "schema": "x2_phase77p_performance_eval_v1",
            "decision": "EVAL_FINITE" if finite else "EVAL_INVALID",
            "source_checkpoint": str(args.source),
            "source_checkpoint_sha256": EXPECTED[args.source.resolve()],
            "source_iteration": int(source_payload.get("iter", -1)),
            "candidate_checkpoint": str(args.candidate),
            "candidate_checkpoint_sha256": sha256(args.candidate),
            "candidate_update_index": int(candidate_payload["update_index"]),
            "candidate_train_seed": int(candidate_payload["train_seed"]),
            "eval_seed": args.seed,
            "lane": args.lane,
            "num_envs": args.num_envs,
            "eval_steps": args.eval_steps,
            "horizon_s": args.eval_steps * float(env.step_dt),
            "allocation": (
                "A: even source / odd candidate" if args.lane == "A"
                else "B: even candidate / odd source"
            ),
            "event": {
                "stand_s": 0.0,
                "accelerate_s": 1.0,
                "cruise_s": 4.2,
                "decelerate_s": 2.0,
                "terminal_hold_observed_s": args.eval_steps * float(env.step_dt) - 7.2,
            },
            "groups": groups,
            "per_env": per_env,
            "candidate_minus_source": deltas,
            "candidate_source_action_drift_max": action_drift_max,
            "finite": finite,
            "per_env_complete": per_env_complete,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
        }
    finally:
        pass


def main() -> None:
    validate_inputs()
    try:
        report = train() if args.mode == "train" else evaluate()
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        failure = {
            "schema": "x2_phase77p_performance_failure_v1",
            "mode": args.mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if not args.report.exists():
            atomic_json(args.report, failure)
        raise


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        # Kit shutdown has hung in earlier technical phases after evidence was
        # already complete.  The external process-group supervisor owns cleanup.
        os._exit(0)
