#!/usr/bin/env python3
"""Run one initial-only, single-process 128-env paired-lane Phase74 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/home/yu/x2_teleop_final/x2_sonic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp = path.with_name(f".{path.name}.tmp")
    if path.exists() or sidecar.exists() or temp.exists():
        raise FileExistsError(f"refusing to overwrite immutable Phase74 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    digest = sha256(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--residual-checkpoint", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--seed-index", type=int, required=True)
parser.add_argument("--env-seed", type=int, required=True)
parser.add_argument("--commit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--failure", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def prereg_guard() -> dict[str, object]:
    if args.num_envs != 128 or args.seed_index not in range(3):
        raise ValueError("Phase74 requires 128 envs and technical seed-index 0..2")
    for path in (args.commit, args.output, args.failure):
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"Phase74 output already exists: {path}")
    prereg = json.loads(args.prereg.read_text())
    sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    if sidecar.read_text() != f"{sha256(args.prereg)}  {args.prereg.name}\n":
        raise RuntimeError("Phase74 prereg sidecar mismatch")
    if prereg.get("schema") != "x2_phase74_pairing_preflight_prereg_v1":
        raise RuntimeError("Phase74 prereg schema changed")
    seed = prereg["technical_seeds"][args.seed_index]
    if int(seed["env_seed"]) != args.env_seed:
        raise RuntimeError("Phase74 technical env seed changed")
    input_paths = {
        "base_checkpoint_sha256": args.checkpoint,
        "zero_residual_checkpoint_sha256": args.residual_checkpoint,
        "environment_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/env.yaml",
        "agent_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/agent.yaml",
        "gait_template_sha256": OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz",
        "upper_motion_sha256": ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz",
    }
    mismatched_inputs = [
        name for name, path in input_paths.items()
        if not path.is_file() or sha256(path) != prereg["immutable_inputs"].get(name)
    ]
    if mismatched_inputs:
        raise RuntimeError(f"Phase74 immutable input drift: {mismatched_inputs}")
    code_paths = {
        "runner_sha256": Path(__file__).resolve(),
        "helper_sha256": ROOT / "src/cwi_x2/phase74_pairing_preflight.py",
        "run_script_sha256": ROOT / "scripts/run_x2_phase74_pairing_preflight.sh",
        "test_sha256": ROOT / "tests/test_phase74_pairing_preflight.py",
        "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase74_pairing_preflight.py",
        "finalizer_test_sha256": ROOT / "tests/test_phase74_pairing_finalizer.py",
        "phase60_posture_module_sha256": ROOT / "src/x2_native_locomotion_posture_phase60.py",
        "x2_flat_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/flat_env_cfg.py",
        "x2_reward_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/rewards.py",
        "x2_gait_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/gait.py",
        "x2_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actions.py",
        "x2_gait_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/actions.py",
        "upper_hook_sha256": ROOT / "hooks/sitecustomize.py",
        "heading_command_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/heading_command.py",
        "modular_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
        "x2_robot_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py",
        "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "ledger_sha256": ROOT / "tools/retarget/run_with_gpu_ledger.py",
    }
    mismatched_code = [
        name for name, path in code_paths.items()
        if not path.is_file() or sha256(path) != prereg["immutable_code"].get(name)
    ]
    if mismatched_code:
        raise RuntimeError(f"Phase74 immutable code drift: {mismatched_code}")
    return prereg


PREREG = prereg_guard()


def startup_failure(exc: BaseException) -> None:
    failure = {
        "schema": "x2_phase74_pairing_preflight_failure_v1",
        "decision": "FAIL_TECHNICAL_STOP",
        "seed_index": args.seed_index,
        "env_seed": args.env_seed,
        "failure_stage": "app_launcher_or_post_launcher_import",
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "screen_exists": args.output.exists(),
        "commit_exists": args.commit.exists(),
        "scientific_metrics_present": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "remaining_launches_forbidden": True,
    }
    try:
        atomic_json(args.failure, failure)
    except BaseException:
        traceback.print_exc()
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


try:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
except BaseException as exc:
    startup_failure(exc)


def phase74_excepthook(exc_type, exc, tb) -> None:
    del exc_type, tb
    startup_failure(exc)


sys.excepthook = phase74_excepthook

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.utils.math import convert_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.heading_command import gain_scheduled_velocity_cfg  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from x2_native_locomotion_posture_phase60 import (  # noqa: E402
    actual_support_com_penalty,
    signed_backward_pitch_penalty,
)
from cwi_x2.phase68_residual_ppo import ResidualActorCritic, build_seeded_residual  # noqa: E402
from cwi_x2.phase69_reward_attribution import semantic_phase_ids  # noqa: E402
from cwi_x2.phase74_pairing_preflight import (  # noqa: E402
    copy_pair_rows,
    pair_diagnostics,
    paired_lane_ids,
    tensor_hash,
)


TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER = ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
LOWER15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
ROBOT_DATA_STATIC_PAIR_INVARIANTS = (
    "GRAVITY_VEC_W", "FORWARD_VEC_B", "default_root_state", "default_joint_pos", "default_joint_vel",
    "default_joint_pos_limits", "default_joint_stiffness", "default_joint_damping", "default_joint_armature",
    "default_joint_friction_coeff", "default_joint_dynamic_friction_coeff", "default_joint_viscous_friction_coeff",
    "joint_pos_limits", "joint_vel_limits", "joint_effort_limits", "joint_stiffness", "joint_damping",
    "joint_armature", "joint_friction_coeff", "joint_dynamic_friction_coeff", "joint_viscous_friction_coeff",
    "default_mass", "default_inertia", "soft_joint_pos_limits", "soft_joint_vel_limits", "gear_ratio",
    "default_fixed_tendon_stiffness", "default_fixed_tendon_damping", "default_fixed_tendon_limit_stiffness",
    "default_fixed_tendon_pos_limits", "default_fixed_tendon_rest_length", "default_fixed_tendon_offset",
    "fixed_tendon_stiffness", "fixed_tendon_damping", "fixed_tendon_limit_stiffness", "fixed_tendon_pos_limits",
    "fixed_tendon_rest_length", "fixed_tendon_offset", "default_spatial_tendon_stiffness",
    "default_spatial_tendon_damping", "default_spatial_tendon_limit_stiffness", "default_spatial_tendon_offset",
    "spatial_tendon_stiffness", "spatial_tendon_damping", "spatial_tendon_limit_stiffness",
    "spatial_tendon_offset",
)
ROBOT_DATA_TIMESTAMPED_BUFFERS = (
    "_root_link_pose_w", "_root_link_vel_w", "_body_link_pose_w", "_body_link_vel_w", "_body_com_pose_b",
    "_root_com_pose_w", "_root_com_vel_w", "_body_com_pose_w", "_body_com_vel_w", "_body_com_acc_w",
    "_root_state_w", "_root_link_state_w", "_root_com_state_w", "_body_state_w", "_body_link_state_w",
    "_body_com_state_w", "_joint_pos", "_joint_vel", "_joint_acc", "_body_incoming_joint_wrench_b",
)


def build_env_cfg():
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = 128
    cfg.seed = args.env_seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = str(TEMPLATE)
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()}
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity = gain_scheduled_velocity_cfg(
        cfg.commands.base_velocity,
        ideal_env_fraction=1.0,
        ideal_heading_control_stiffness=1.0,
        response_heading_control_stiffness=0.05,
    )
    reset = PREREG["reset_perturbation"]
    cfg.events.reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": tuple(reset["root_pose_roll_rad"]),
            "pitch": tuple(reset["root_pose_pitch_rad"]), "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": tuple(reset["root_linear_x_mps"]),
            "y": tuple(reset["root_linear_y_mps"]), "z": (0.0, 0.0),
            "roll": tuple(reset["root_angular_roll_radps"]),
            "pitch": tuple(reset["root_angular_pitch_radps"]),
            "yaw": tuple(reset["root_angular_yaw_radps"]),
        },
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = False
    cfg.rewards.signed_backward_pitch = RewTerm(
        func=signed_backward_pitch_penalty, weight=-0.5,
        params={
            "command_name": "base_velocity", "tolerance_rad": 0.05,
            "normalization_rad": 0.15, "command_threshold_mps": 0.10,
            "asset_name": "robot",
        },
    )
    cfg.rewards.actual_support_com = RewTerm(
        func=actual_support_com_penalty, weight=-0.5,
        params={
            "command_name": "base_velocity", "normalization_m": 0.10,
            "force_threshold_n": 10.0, "command_threshold_mps": 0.10,
        },
    )
    _apply_x2_actuator_response(
        cfg.scene.robot,
        {
            "enabled": True, "profile": "session03_session04_group",
            "randomize": False, "strength": 1.0, "filter_strength": 1.0,
            "delay_strength": 1.0, "include_ideal_endpoint": False,
            "ideal_env_fraction": 1.0, "filter_only_env_fraction": 0.0,
        },
        physics_dt_sec=cfg.sim.dt,
    )
    return cfg


def build_model(obs, device):
    return ActorCritic(
        obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]}, num_actions=15,
        actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128], activation="elu",
        init_noise_std=0.4, noise_std_type="scalar", actor_obs_normalization=False,
        critic_obs_normalization=False,
    ).to(device)


def dynamic_state_entries(env) -> tuple[dict[str, tuple[torch.Tensor, int, bool]], list[str], dict[str, object]]:
    """Build the explicit per-env state manifest and fail on unknown mutable tensors."""

    entries: dict[str, tuple[torch.Tensor, int, bool]] = {}
    owners: list[tuple[str, object, int, set[str]]] = []
    metadata: dict[str, object] = {}

    def add(name: str, value, *, axis: int = 0, copy: bool = True) -> None:
        if isinstance(value, torch.Tensor) and value.ndim > axis and value.shape[axis] == 128:
            entries[name] = (value, axis, copy)

    def owner(prefix: str, value: object, axis: int = 0, excluded: tuple[str, ...] = ()) -> None:
        owners.append((prefix, value, axis, set(excluded)))

    for name in ("episode_length_buf", "reset_buf", "reset_terminated", "reset_time_outs", "reward_buf"):
        add(f"env.{name}", getattr(env, name, None))
    owner("env", env, excluded=("obs_buf",))
    action_manager = env.action_manager
    for name in ("_action", "_prev_action"):
        add(f"action_manager.{name}", getattr(action_manager, name, None))
    owner("action_manager", action_manager)
    action_term = action_manager._terms["joint_pos"]
    for name in (
        "_raw_actions", "_processed_actions", "_preemphasis_delta", "_combined_normalized_actions",
        "_normalized_template_bias", "_preclip_combined_actions", "_plant_bias_env_mask",
        "_cwi_upper_clip_ids", "_cwi_upper_zero_mask", "_cwi_upper_baseline",
        "_cwi_prev_upper_target", "_cwi_upper_fallback_active", "_cwi_initial_yaw",
        "_cwi_initial_yaw_valid", "_cwi_upper_heading_error",
    ):
        add(f"action_term.{name}", getattr(action_term, name, None))
    for name in ("_scale", "_offset", "_clip"):
        add(f"action_term.{name}", getattr(action_term, name, None), copy=False)
    owner(
        "action_term", action_term,
        excluded=("_normalized_plant_bias", "_template_q"),
    )
    command = env.command_manager._terms["base_velocity"]
    for name in (
        "time_left", "command_counter", "vel_command_b", "heading_target", "is_heading_env",
        "is_standing_env", "heading_error_integral", "heading_integral_contribution", "heading_error",
        "heading_control_stiffness", "heading_integral_gain",
    ):
        add(f"command.{name}", getattr(command, name, None))
    for name, value in command.metrics.items():
        add(f"command.metrics.{name}", value)
    owner("command", command)
    termination = env.termination_manager
    for name in ("_term_dones", "_last_episode_dones", "_truncated_buf", "_terminated_buf"):
        add(f"termination.{name}", getattr(termination, name, None))
    owner("termination", termination)
    reward = env.reward_manager
    for name in ("_reward_buf", "_step_reward"):
        add(f"reward.{name}", getattr(reward, name, None))
    for name, value in reward._episode_sums.items():
        add(f"reward._episode_sums.{name}", value)
    owner("reward", reward)
    for term_name, term_cfg in zip(reward._term_names, reward._term_cfgs, strict=True):
        term_owner = term_cfg.func
        for name in ("_air_time", "_contact_time", "_contact", "_mode_time", "_initialized"):
            add(f"reward_term.{term_name}.{name}", getattr(term_owner, name, None))
        if hasattr(term_owner, "__dict__"):
            owner(f"reward_term.{term_name}", term_owner)
    for sensor_name, sensor in sorted(env.scene.sensors.items()):
        for name in ("_timestamp", "_timestamp_last_update", "_is_outdated"):
            add(f"sensor.{sensor_name}.{name}", getattr(sensor, name, None))
        owner(f"sensor.{sensor_name}", sensor)
        data = sensor._data
        for name, value in vars(data).items():
            add(f"sensor.{sensor_name}.data.{name}", value)
        owner(f"sensor.{sensor_name}.data", data)
    robot = env.scene["robot"]
    for name in ("_joint_pos_target_sim", "_joint_vel_target_sim", "_joint_effort_target_sim"):
        add(f"robot.{name}", getattr(robot, name, None))
    owner("robot", robot, excluded=("_ALL_INDICES",))
    for name in (
        "joint_pos_target", "joint_vel_target", "joint_effort_target", "computed_torque", "applied_torque",
    ):
        add(f"robot.data.{name}", getattr(robot._data, name, None))
    add("robot.data._previous_joint_vel", robot._data._previous_joint_vel)
    for name in ROBOT_DATA_STATIC_PAIR_INVARIANTS:
        add(f"robot.data.{name}", getattr(robot._data, name, None), copy=False)
    owner(
        "robot.data",
        robot._data,
        excluded=("_root_physx_view", "_physics_sim_view", *ROBOT_DATA_TIMESTAMPED_BUFFERS),
    )
    metadata["robot.data._sim_timestamp"] = float(robot._data._sim_timestamp)
    metadata["robot.data.timestamped_buffers"] = {
        name: {
            "timestamp": float(getattr(robot._data, name).timestamp),
            "data_present": getattr(robot._data, name).data is not None,
        }
        for name in ROBOT_DATA_TIMESTAMPED_BUFFERS
    }
    for composer_name in ("_instantaneous_wrench_composer", "_permanent_wrench_composer"):
        composer = getattr(robot, composer_name)
        prefix = f"robot.{composer_name}"
        add(f"{prefix}._composed_force_b_torch", composer._composed_force_b_torch)
        add(f"{prefix}._composed_torque_b_torch", composer._composed_torque_b_torch)
        owner(
            prefix,
            composer,
            excluded=("_ALL_ENV_INDICES_TORCH", "_ALL_BODY_INDICES_TORCH"),
        )
        metadata[f"{prefix}._active"] = bool(composer._active)
        metadata[f"{prefix}._link_poses_updated"] = bool(composer._link_poses_updated)
    for group_name, actuator in robot.actuators.items():
        for name in (
            "computed_effort", "applied_effort", "_position_alpha", "_filtered_joint_positions",
            "_position_filter_initialized", "_ideal_env_mask",
        ):
            add(f"actuator.{group_name}.{name}", getattr(actuator, name, None))
        for name in (
            "stiffness", "damping", "armature", "friction", "dynamic_friction", "viscous_friction",
            "velocity_limit", "effort_limit", "velocity_limit_sim", "effort_limit_sim",
        ):
            add(f"actuator.{group_name}.{name}", getattr(actuator, name, None), copy=False)
        owner(f"actuator.{group_name}", actuator, excluded=("joint_indices", "_ALL_INDICES"))
        for buffer_name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
            buffer = getattr(actuator, buffer_name, None)
            if buffer is None:
                continue
            prefix = f"actuator.{group_name}.{buffer_name}"
            add(f"{prefix}._time_lags", getattr(buffer, "_time_lags", None))
            owner(prefix, buffer)
            circular = getattr(buffer, "_circular_buffer", None)
            if circular is None:
                continue
            circular_prefix = f"{prefix}._circular_buffer"
            add(f"{circular_prefix}._num_pushes", getattr(circular, "_num_pushes", None))
            add(f"{circular_prefix}._max_len", getattr(circular, "_max_len", None), copy=False)
            storage = getattr(circular, "_buffer", None)
            add(f"{circular_prefix}._buffer", storage, axis=1)
            owner(circular_prefix, circular, axis=1, excluded=("_ALL_INDICES",))
            metadata[f"{circular_prefix}._pointer"] = int(circular._pointer)
    histories = env.observation_manager._group_obs_term_history_buffer
    for group_name, terms in histories.items():
        for term_name, circular in terms.items():
            prefix = f"observation_history.{group_name}.{term_name}"
            add(f"{prefix}._num_pushes", circular._num_pushes)
            add(f"{prefix}._max_len", circular._max_len, copy=False)
            add(f"{prefix}._buffer", circular._buffer, axis=1)
            owner(prefix, circular, axis=1, excluded=("_ALL_INDICES",))
            metadata[f"{prefix}._pointer"] = int(circular._pointer)
    for index, time_left in enumerate(env.event_manager._interval_term_time_left):
        add(f"event_manager.interval_time_left.{index}", time_left)
    metadata["env.common_step_counter"] = int(env.common_step_counter)
    metadata["env.sim_step_counter"] = int(env._sim_step_counter)

    def discover_container(prefix: str, value, axis: int) -> list[str]:
        if isinstance(value, torch.Tensor):
            if value.ndim > axis and value.shape[axis] == 128:
                return [prefix]
            return []
        if isinstance(value, dict):
            found: list[str] = []
            for key, child in value.items():
                found.extend(discover_container(f"{prefix}.{key}", child, axis))
            return found
        if isinstance(value, (list, tuple)):
            found = []
            for index, child in enumerate(value):
                found.extend(discover_container(f"{prefix}.{index}", child, axis))
            return found
        return []

    unknown: list[str] = []
    for prefix, owner_value, axis, excluded in owners:
        if not hasattr(owner_value, "__dict__"):
            continue
        for attr, value in vars(owner_value).items():
            if attr in excluded:
                continue
            for path in discover_container(f"{prefix}.{attr}", value, axis):
                if path not in entries:
                    unknown.append(path)
    return entries, sorted(set(unknown)), metadata


def copy_dynamic_state(env, donors, recipients) -> tuple[list[str], list[str], dict[str, object]]:
    entries, unknown, metadata = dynamic_state_entries(env)
    copied: list[str] = []
    for name, (value, axis, should_copy) in entries.items():
        if not should_copy:
            continue
        if axis == 0:
            copy_pair_rows(value, donors, recipients)
        elif axis == 1:
            value[:, recipients] = value[:, donors].clone()
        else:
            raise RuntimeError(f"unsupported Phase74 batch axis: {name} axis={axis}")
        copied.append(name)
    return sorted(copied), unknown, metadata


def invalidate_articulation_lazy_caches(env) -> list[str]:
    """Force derived ArticulationData to be recomputed from the cloned PhysX state."""

    data = env.scene["robot"]._data
    invalidated = []
    for name in ROBOT_DATA_TIMESTAMPED_BUFFERS:
        buffer = getattr(data, name)
        buffer.timestamp = -1.0
        invalidated.append(name)
    return invalidated


def state_tensors(env, policy, residual) -> tuple[dict[str, torch.Tensor], list[str], dict[str, object]]:
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    env.sim.forward()
    env.obs_buf = env.observation_manager.compute(update_history=False)
    obs = env.obs_buf
    source_action = policy.source_action(obs)
    with torch.inference_mode():
        encoded = residual.encoder(obs["policy"])
        latent_mean = residual.head(encoded)
    physx_root = robot.root_physx_view.get_root_transforms().clone()
    physx_root[:, :3] -= origins
    values: dict[str, torch.Tensor] = {
        "physx.root_transform_local_xyzw": physx_root,
        "physx.root_velocity_w": robot.root_physx_view.get_root_velocities().clone(),
        "physx.joint_position": robot.root_physx_view.get_dof_positions().clone(),
        "physx.joint_velocity": robot.root_physx_view.get_dof_velocities().clone(),
        "policy_observation": obs["policy"],
        "critic_observation": obs["critic"],
        "source_action": source_action,
        "residual_encoded_feature": encoded,
        "residual_latent_mean": latent_mean,
        "phase_id": semantic_phase_ids(obs["policy"]),
        "active_mask": torch.linalg.vector_norm(env.command_manager.get_command("base_velocity")[:, :2], dim=-1) > 0.1,
    }
    entries, unknown, metadata = dynamic_state_entries(env)
    for name, (value, axis, _) in entries.items():
        if axis == 0:
            values[f"dynamic.{name}"] = value
        elif axis == 1:
            values[f"dynamic.{name}"] = value.transpose(0, 1)
    return ({name: value.detach().clone() for name, value in values.items()}, unknown, metadata)


def source_initial_summary(env, donors) -> dict[str, object]:
    """Hash the untouched donor panel and prove the technical reset is non-degenerate."""

    robot = env.scene["robot"]
    origins = env.scene.env_origins
    root = robot.root_physx_view.get_root_transforms().clone()
    root[:, :3] -= origins
    velocity = robot.root_physx_view.get_root_velocities().clone()
    joint_position = robot.root_physx_view.get_dof_positions().clone()
    joint_velocity = robot.root_physx_view.get_dof_velocities().clone()
    command = env.command_manager.get_command("base_velocity").clone()
    fields = {
        "root_transform_local_xyzw": root[donors],
        "root_velocity_w": velocity[donors],
        "joint_position": joint_position[donors],
        "joint_velocity": joint_velocity[donors],
        "command": command[donors],
    }
    field_hashes = {name: tensor_hash(value) for name, value in fields.items()}
    combined = hashlib.sha256(json.dumps(field_hashes, sort_keys=True).encode()).hexdigest()
    randomized = torch.cat((root[donors, 3:7], velocity[donors]), dim=-1)
    max_range = float(
        torch.max(torch.max(randomized, dim=0).values - torch.min(randomized, dim=0).values).item()
    )
    return {
        "field_sha256": field_hashes,
        "combined_sha256": combined,
        "randomized_state_max_range": max_range,
        "nondegenerate": bool(max_range > 1.0e-6),
    }


def run() -> None:
    env = ManagerBasedRLEnv(cfg=build_env_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    obs = wrapped.get_observations()
    if list(obs["policy"].shape) != [128, 93] or wrapped.num_actions != 15:
        raise RuntimeError("Phase74 observation/action shape changed")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15:
        raise RuntimeError("Phase74 lower joint order changed")
    source = build_model(obs, env.device).eval()
    source_payload = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    source.load_state_dict(source_payload["model_state_dict"], strict=True)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    residual = build_seeded_residual().to(env.device).eval()
    residual_payload = torch.load(args.residual_checkpoint, map_location=env.device, weights_only=False)
    residual.load_state_dict(residual_payload["residual_state_dict"], strict=True)
    for parameter in residual.parameters():
        parameter.requires_grad_(False)
    policy = ResidualActorCritic(source, residual).to(env.device).eval()
    donors, recipients = paired_lane_ids(args.seed_index, 64, device=env.device)
    source_initial = source_initial_summary(env, donors)

    robot = env.scene["robot"]
    origins = env.scene.env_origins
    donor_root_xyzw = robot.root_physx_view.get_root_transforms()[donors].clone()
    donor_root_pose = donor_root_xyzw.clone()
    donor_root_pose[:, 3:7] = convert_quat(donor_root_pose[:, 3:7], to="wxyz")
    donor_root_pose[:, :3] = donor_root_pose[:, :3] - origins[donors] + origins[recipients]
    donor_root_state = torch.cat(
        (donor_root_pose, robot.root_physx_view.get_root_velocities()[donors].clone()), dim=-1
    )
    robot.write_root_state_to_sim(donor_root_state, env_ids=recipients)
    robot.write_joint_state_to_sim(
        robot.root_physx_view.get_dof_positions()[donors].clone(),
        robot.root_physx_view.get_dof_velocities()[donors].clone(),
        env_ids=recipients,
    )
    copied_fields, unknown_before, metadata_before = copy_dynamic_state(env, donors, recipients)
    invalidated_caches = invalidate_articulation_lazy_caches(env)
    tensors, unknown_after, metadata_after = state_tensors(env, policy, residual)
    unknown_fields = sorted(set(unknown_before + unknown_after))
    state_entries, _, _ = dynamic_state_entries(env)
    state_manifest = {
        name: {
            "batch_axis": axis,
            "copied": should_copy,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for name, (value, axis, should_copy) in sorted(state_entries.items())
    }
    commit = {
        "schema": "x2_phase74_pairing_commit_v1",
        "seed_index": args.seed_index,
        "env_seed": args.env_seed,
        "pair_count": 64,
        "donor_ids": donors.cpu().tolist(),
        "recipient_ids": recipients.cpu().tolist(),
        "copied_dynamic_fields": copied_fields,
        "state_manifest": state_manifest,
        "unknown_mutable_tensor_fields": unknown_fields,
        "metadata_before_copy": metadata_before,
        "metadata_after_observation_recompute": metadata_after,
        "invalidated_articulation_lazy_caches": invalidated_caches,
        "source_donor_initial": source_initial,
        "tensor_sides": {
            name: {
                "dtype": str(value.dtype), "shape": list(value.shape),
                "donor_sha256": tensor_hash(value[donors]),
                "recipient_sha256": tensor_hash(value[recipients]),
            }
            for name, value in sorted(tensors.items())
        },
        "scientific_metrics_present": False,
        "physics_rollout_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_count": 0,
    }
    commit_sha = atomic_json(args.commit, commit)
    tolerances = PREREG["derived_tolerances"]
    diagnostics = {}
    for name, value in sorted(tensors.items()):
        if name.startswith("dynamic.sensor.") and "force" in name:
            threshold = float(tolerances["contact_force_max_abs_n"])
        elif name in {
            "policy_observation", "critic_observation", "source_action", "residual_encoded_feature",
            "residual_latent_mean", "physx.root_transform_local_xyzw", "physx.root_velocity_w",
            "physx.joint_position", "physx.joint_velocity",
        }:
            threshold = float(tolerances["derived_float_max_abs"])
        else:
            threshold = 0.0
        diagnostics[name] = pair_diagnostics(value, donors, recipients, atol=threshold)
        finite_required = not name.endswith(".data.contact_pos_w")
        diagnostics[name]["finite_required"] = finite_required
        diagnostics[name]["all_finite"] = bool(torch.isfinite(value).all()) if value.dtype.is_floating_point else True
        if finite_required:
            diagnostics[name]["passed"] = bool(
                diagnostics[name]["passed"] and diagnostics[name]["all_finite"]
            )
        if name.startswith("dynamic.sensor.") and "force" in name:
            diagnostics[name]["relative_l2_threshold"] = float(tolerances["contact_force_relative_l2"])
            diagnostics[name]["passed"] = bool(
                diagnostics[name]["passed"]
                and diagnostics[name]["relative_l2"] <= float(tolerances["contact_force_relative_l2"])
            )
    valid = bool(
        diagnostics
        and not unknown_fields
        and source_initial["nondegenerate"] is True
        and all(bool(row["passed"]) for row in diagnostics.values())
    )
    report = {
        "schema": "x2_phase74_pairing_preflight_screen_v1",
        "decision": "PASS_INITIAL_PAIRING_LAUNCH" if valid else "FAIL_TECHNICAL_STOP",
        "seed_index": args.seed_index,
        "env_seed": args.env_seed,
        "commit": str(args.commit),
        "commit_sha256": commit_sha,
        "pair_count": 64,
        "field_count": len(diagnostics),
        "diagnostics": diagnostics,
        "state_manifest_field_count": len(state_manifest),
        "copied_dynamic_field_count": len(copied_fields),
        "unknown_mutable_tensor_fields": unknown_fields,
        "unknown_mutable_tensor_fields_empty": not unknown_fields,
        "source_donor_initial": source_initial,
        "observation_recomputed_after_clone": True,
        "physx_low_level_readback_present": True,
        "all_pairs_pass": valid,
        "scientific_metrics_present": False,
        "physics_rollout_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "phase75_shadow_preregistration_unlocked": False,
        "phase75_scientific_preregistration_unlocked": False,
        "phase75_launch_unlocked": False,
        "training_unlocked": False,
        "deployment_unlocked": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not valid:
        raise RuntimeError("Phase74 paired-lane technical gates failed")


try:
    run()
except BaseException as exc:
    failure = {
        "schema": "x2_phase74_pairing_preflight_failure_v1",
        "decision": "FAIL_TECHNICAL_STOP",
        "seed_index": args.seed_index,
        "env_seed": args.env_seed,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "screen_exists": args.output.exists(),
        "commit_exists": args.commit.exists(),
        "scientific_metrics_present": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "remaining_launches_forbidden": True,
    }
    try:
        atomic_json(args.failure, failure)
    except BaseException:
        traceback.print_exc()
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    simulation_app.close()
