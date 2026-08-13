#!/usr/bin/env python3
"""Run one immutable Phase72 antithetic, zero-optimizer X2 trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    digest = sha256(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--residual-checkpoint", type=Path, required=True)
parser.add_argument("--schedule", type=Path, required=True)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--seed-index", type=int, required=True)
parser.add_argument("--env-seed", type=int, required=True)
parser.add_argument("--sign", choices=("plus", "minus"), required=True)
parser.add_argument("--init-mode", choices=("write", "verify"), required=True)
parser.add_argument("--init-commit", type=Path, required=True)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def prereg_guard() -> dict[str, object]:
    """Validate all preregistered bytes before AppLauncher constructs physics."""

    if args.num_envs != 64 or args.steps != 400 or args.seed_index not in range(5):
        raise ValueError("Phase72 requires seed-index 0..4, 64 envs, and 400 steps")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        raise FileExistsError("Phase72 screen already exists")
    if args.bundle.exists() or args.bundle.with_suffix(args.bundle.suffix + ".sha256").exists():
        raise FileExistsError("Phase72 bundle already exists")
    prereg = json.loads(args.prereg.read_text())
    prereg_sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    expected_sidecar = f"{sha256(args.prereg)}  {args.prereg.name}\n"
    if not prereg_sidecar.is_file() or prereg_sidecar.read_text() != expected_sidecar:
        raise RuntimeError("Phase72 prereg sidecar mismatch")
    if prereg.get("schema") != "x2_phase72_antithetic_prereg_v1":
        raise RuntimeError("Phase72 prereg schema changed")
    seed_record = prereg["seed_pairs"][args.seed_index]
    if int(seed_record["env_seed"]) != args.env_seed:
        raise RuntimeError("Phase72 environment seed differs from preregistration")
    if Path(seed_record["schedule_path"]).resolve() != args.schedule.resolve():
        raise RuntimeError("Phase72 schedule path differs from preregistration")
    if sha256(args.schedule) != seed_record["schedule_sha256"]:
        raise RuntimeError("Phase72 schedule bytes changed")
    if sha256(args.checkpoint) != prereg["immutable_inputs"]["base_checkpoint_sha256"]:
        raise RuntimeError("Phase72 base checkpoint changed")
    if sha256(args.residual_checkpoint) != prereg["immutable_inputs"]["zero_residual_checkpoint_sha256"]:
        raise RuntimeError("Phase72 zero residual checkpoint changed")
    input_paths = {
        "environment_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/env.yaml",
        "agent_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/agent.yaml",
        "gait_template_sha256": OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz",
        "upper_motion_sha256": ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz",
        "schedule_inventory_sha256": ROOT / "reports/retarget/x2_phase72_schedule_inventory.json",
    }
    input_mismatched = [
        name
        for name, path in input_paths.items()
        if not path.is_file()
        or sha256(path) != prereg["immutable_inputs"].get(name)
    ]
    if input_mismatched:
        raise RuntimeError(f"Phase72 immutable input drift: {input_mismatched}")
    code_paths = {
        "runner_sha256": Path(__file__).resolve(),
        "helper_sha256": ROOT / "src/cwi_x2/phase72_antithetic.py",
        "run_script_sha256": ROOT / "scripts/run_x2_phase72_antithetic.sh",
        "pair_validator_sha256": ROOT / "tools/retarget/validate_x2_phase72_pair.py",
        "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase72_antithetic.py",
        "runner_test_sha256": ROOT / "tests/test_phase72_antithetic_runner.py",
        "helper_test_sha256": ROOT / "tests/test_phase72_antithetic.py",
        "finalizer_test_sha256": ROOT / "tests/test_phase72_antithetic_finalizer.py",
        "schedule_generator_sha256": ROOT / "tools/retarget/prepare_x2_phase72_schedules.py",
        "phase60_posture_module_sha256": ROOT / "src/x2_native_locomotion_posture_phase60.py",
        "x2_flat_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/flat_env_cfg.py",
        "x2_reward_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/rewards.py",
        "x2_gait_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/gait.py",
        "x2_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actions.py",
        "heading_command_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/heading_command.py",
        "modular_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
        "x2_robot_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py",
        "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "phase70_helper_sha256": ROOT / "src/cwi_x2/phase70_long_lookahead.py",
        "ledger_sha256": ROOT / "tools/retarget/run_with_gpu_ledger.py",
    }
    expected_code = prereg["immutable_code"]
    mismatched = [
        name
        for name, path in code_paths.items()
        if not path.is_file() or sha256(path) != expected_code.get(name)
    ]
    if mismatched:
        raise RuntimeError(f"Phase72 immutable code drift: {mismatched}")
    if args.init_mode == "write":
        if args.init_commit.exists() or args.init_commit.with_suffix(args.init_commit.suffix + ".sha256").exists():
            raise FileExistsError("Phase72 pair init commit already exists")
    else:
        sidecar = args.init_commit.with_suffix(args.init_commit.suffix + ".sha256")
        if not args.init_commit.is_file() or not sidecar.is_file():
            raise FileNotFoundError("Phase72 reference init commit is absent")
        if sidecar.read_text() != f"{sha256(args.init_commit)}  {args.init_commit.name}\n":
            raise RuntimeError("Phase72 reference init commit sidecar mismatch")
    return prereg


PREREG = prereg_guard()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.heading_command import gain_scheduled_velocity_cfg  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from x2_native_locomotion_posture_phase60 import (  # noqa: E402
    actual_support_com_outside_distance,
    actual_support_com_penalty,
    signed_backward_pitch_penalty,
    signed_root_pitch_rad,
)
from cwi_x2.phase68_residual_ppo import (  # noqa: E402
    PhysicalKneeResidualVecEnv,
    ResidualActorCritic,
    build_seeded_residual,
)
from cwi_x2.phase69_reward_attribution import semantic_phase_ids  # noqa: E402
from cwi_x2.phase72_antithetic import tensor_hash  # noqa: E402


ORIGINAL = OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
ENV_YAML = ORIGINAL.parent / "params/env.yaml"
AGENT_YAML = ORIGINAL.parent / "params/agent.yaml"
TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER = ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
LOWER15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
REWARD_TERMS = (
    ("track_lin_vel_xy_exp", 4.0), ("track_ang_vel_z_exp", 1.0),
    ("lin_vel_z_l2", -0.2), ("ang_vel_xy_l2", -0.05),
    ("dof_torques_l2", -2.0e-6), ("dof_acc_l2", -1.0e-7),
    ("action_rate_l2", -0.005), ("feet_air_time", 1.0),
    ("flat_orientation_l2", -1.0), ("dof_pos_limits", -1.0),
    ("termination_penalty", -200.0), ("feet_slide", -0.2),
    ("joint_deviation_hip", -0.1), ("joint_deviation_arms", -0.1),
    ("joint_deviation_torso", -0.1), ("yaw_rate_l2", -0.5),
    ("stand_lin_vel_xy_l2", 0.0), ("action_magnitude_l2", 0.0),
    ("heading_error_l2", 0.0), ("contact_dwell", -1.0),
    ("contact_phase", -1.0), ("signed_backward_pitch", -0.5),
    ("actual_support_com", -0.5),
)


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_env_cfg():
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = 64
    cfg.seed = args.env_seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = str(TEMPLATE)
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
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
            "pitch": tuple(reset["root_pose_pitch_rad"]),
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": tuple(reset["root_linear_x_mps"]),
            "y": tuple(reset["root_linear_y_mps"]),
            "z": (0.0, 0.0),
            "roll": tuple(reset["root_angular_roll_radps"]),
            "pitch": tuple(reset["root_angular_pitch_radps"]),
            "yaw": tuple(reset["root_angular_yaw_radps"]),
        },
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
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
            "command_name": "base_velocity", "tolerance_rad": 0.05,
            "normalization_rad": 0.15, "command_threshold_mps": 0.10,
            "asset_name": "robot",
        },
    )
    cfg.rewards.actual_support_com = RewTerm(
        func=actual_support_com_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity", "normalization_m": 0.10,
            "force_threshold_n": 10.0, "command_threshold_mps": 0.10,
        },
    )
    cfg.observations.policy.enable_corruption = False
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


def initial_tensors(env, wrapped, policy) -> dict[str, torch.Tensor]:
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    term = env.action_manager._terms["joint_pos"]
    actuator = robot.actuators["legs"]
    tensors = {
        "policy_observation": obs["policy"],
        "critic_observation": obs["critic"],
        "root_pose_w": robot.data.root_pose_w,
        "root_velocity_w": robot.data.root_vel_w,
        "joint_position": robot.data.joint_pos,
        "joint_velocity": robot.data.joint_vel,
        "command": env.command_manager.get_command("base_velocity"),
        "episode_length": env.episode_length_buf,
        "source_action": policy.source_action(obs),
        "actuator_alpha": actuator._position_alpha,
        "actuator_lag": actuator.positions_delay_buffer.time_lags,
    }
    for name in ("left_foot_ground_contact", "right_foot_ground_contact"):
        sensor = env.scene[name]
        for attr in ("net_forces_w", "force_matrix_w"):
            value = getattr(sensor.data, attr, None)
            if isinstance(value, torch.Tensor):
                tensors[f"{name}_{attr}"] = value
    for term_name, term_cfg in zip(
        env.reward_manager._term_names,
        env.reward_manager._term_cfgs,
        strict=True,
    ):
        term_func = term_cfg.func
        for attr, value in vars(term_func).items() if hasattr(term_func, "__dict__") else ():
            if isinstance(value, torch.Tensor):
                tensors[f"reward_term_{term_name}_{attr}"] = value
    for prefix, owner in (
        ("actuator", actuator),
        ("actuator_delay", actuator.positions_delay_buffer),
        (
            "actuator_delay_circular",
            getattr(actuator.positions_delay_buffer, "_circular_buffer", None),
        ),
    ):
        if owner is None:
            continue
        for attr, value in vars(owner).items():
            if isinstance(value, torch.Tensor):
                tensors[f"{prefix}_{attr}"] = value
    for attr in ("raw_actions", "processed_actions", "_raw_actions", "_processed_actions"):
        value = getattr(term, attr, None)
        if isinstance(value, torch.Tensor):
            tensors[f"action_term_{attr}"] = value
    return {name: value.detach().clone() for name, value in tensors.items()}


def initial_commit(env, wrapped, policy) -> dict[str, object]:
    tensors = initial_tensors(env, wrapped, policy)
    hashes = {name: tensor_hash(value) for name, value in tensors.items()}
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(f"{name}:{value}".encode())
    root_velocity = tensors["root_velocity_w"].to(torch.float64)
    root_quat = tensors["root_pose_w"][:, 3:7].to(torch.float64)
    return {
        "schema": "x2_phase72_pair_initial_commit_v1",
        "seed_index": args.seed_index,
        "env_seed": args.env_seed,
        "tensor_hashes": hashes,
        "combined_sha256": digest.hexdigest(),
        "root_velocity_std": root_velocity.std(dim=0).tolist(),
        "root_velocity_max_abs": root_velocity.abs().max(dim=0).values.tolist(),
        "root_quaternion_xyz_std": root_quat[:, 1:4].std(dim=0).tolist(),
        "nondegenerate_perturbation": bool(
            root_velocity[:, [0, 1, 3, 4, 5]].std(dim=0).min() > 1.0e-5
            and root_quat[:, 1:3].std(dim=0).min() > 1.0e-5
        ),
    }


def atomic_bundle(path: Path, payload: dict[str, object]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp = path.with_name(f".{path.name}.tmp")
    if path.exists() or sidecar.exists() or temp.exists():
        raise FileExistsError(f"refusing to overwrite immutable bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temp)
    restored = torch.load(temp, map_location="cpu", weights_only=False)
    for key in (
        "policy_observation", "latent_action", "total_reward", "reward_by_term",
        "encoded_feature", "requested_knee_offset_rad", "effective_knee_offset_rad",
    ):
        if not torch.equal(restored[key], payload[key].detach().cpu()):
            raise RuntimeError(f"Phase72 strict bundle reload failed: {key}")
    os.replace(temp, path)
    digest = sha256(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def run() -> None:
    env = wrapped = residual_env = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg())
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        obs = wrapped.get_observations()
        if list(obs["policy"].shape) != [64, 93] or list(obs["critic"].shape) != [64, 93]:
            raise RuntimeError("Phase72 live observation shape changed")
        if wrapped.num_actions != 15:
            raise RuntimeError("Phase72 inner action width changed")
        term = env.action_manager._terms["joint_pos"]
        if tuple(term._joint_names) != LOWER15:
            raise RuntimeError("Phase72 lower joint order changed")
        if not torch.equal(term._cwi_upper_zero_mask, torch.ones(64, dtype=torch.bool, device=env.device)):
            raise RuntimeError("Phase72 fixed upper-body contract changed")
        actuator = env.scene["robot"].actuators["legs"]
        alpha = actuator._position_alpha.reshape(64, -1)[:, 0]
        lag = actuator.positions_delay_buffer.time_lags.reshape(64)
        if not bool(torch.all(torch.isclose(alpha, torch.ones_like(alpha)) & (lag == 0))):
            raise RuntimeError("Phase72 requires 64 ideal actuator environments")

        source = build_model(obs, env.device).eval()
        source_payload = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        source.load_state_dict(source_payload["model_state_dict"], strict=True)
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        residual = build_seeded_residual().to(env.device).eval()
        residual_payload = torch.load(args.residual_checkpoint, map_location=env.device, weights_only=False)
        if residual_payload.get("schema") != "x2_phase68_residual_checkpoint_v1":
            raise RuntimeError("Phase72 zero residual schema changed")
        residual.load_state_dict(residual_payload["residual_state_dict"], strict=True)
        for parameter in residual.parameters():
            parameter.requires_grad_(False)
        policy = ResidualActorCritic(source, residual).to(env.device).eval()
        source_hash_before = state_hash(source)
        residual_hash_before = state_hash(residual)

        commit = initial_commit(env, wrapped, policy)
        if args.init_mode == "write":
            atomic_json(args.init_commit, commit)
            pair_init_exact = True
        else:
            reference = json.loads(args.init_commit.read_text())
            pair_init_exact = (
                reference.get("schema") == commit["schema"]
                and reference.get("seed_index") == commit["seed_index"]
                and reference.get("env_seed") == commit["env_seed"]
                and reference.get("tensor_hashes") == commit["tensor_hashes"]
                and reference.get("combined_sha256") == commit["combined_sha256"]
            )
            if not pair_init_exact:
                raise RuntimeError("Phase72 pair initial tensors are not bitwise identical")

        schedule_payload = torch.load(args.schedule, map_location="cpu", weights_only=False)
        epsilon = schedule_payload["epsilon"]
        seed_record = PREREG["seed_pairs"][args.seed_index]
        if (
            schedule_payload.get("schema") != "x2_phase72_epsilon_schedule_v1"
            or int(schedule_payload.get("latent_seed", -1)) != int(seed_record["latent_seed"])
            or tensor_hash(epsilon) != seed_record["epsilon_tensor_sha256"]
            or list(epsilon.shape) != [400, 64, 2]
        ):
            raise RuntimeError("Phase72 epsilon schedule content changed")

        names = tuple(env.reward_manager._term_names)
        weights = tuple(float(env.reward_manager.get_term_cfg(name).weight) for name in names)
        if names != tuple(name for name, _ in REWARD_TERMS) or weights != tuple(weight for _, weight in REWARD_TERMS):
            raise RuntimeError("Phase72 reward names/order/weights changed")
        records: dict[str, object] = {
            "policy_observation": [], "critic_observation": [], "latent": [],
            "mu": [], "sigma": [], "log_prob": [], "value": [], "phase_id": [],
            "reward": [], "term_reward": [], "signed_pitch": [], "support_outside": [],
            "requested": [], "effective": [], "done_count": 0, "active_count": 0,
            "non_knee_max_abs": 0.0,
        }
        step_dt = float(env.step_dt)

        def record_step(row: dict[str, torch.Tensor]) -> None:
            records["reward"].append(row["reward"].detach().clone())
            records["term_reward"].append(env.reward_manager._step_reward.detach().clone() * step_dt)
            records["requested"].append(row["requested_knee_offset"].detach().clone())
            records["effective"].append(row["effective_knee_offset"].detach().clone())
            records["signed_pitch"].append(signed_root_pitch_rad(env.scene["robot"]).detach().clone())
            outside, _ = actual_support_com_outside_distance(env, force_threshold_n=10.0)
            records["support_outside"].append(outside.detach().clone())
            records["done_count"] = int(records["done_count"]) + int(row["done"].bool().sum())
            records["active_count"] = int(records["active_count"]) + int(row["active_mask"].sum())
            records["non_knee_max_abs"] = max(
                float(records["non_knee_max_abs"]), float(row["non_knee_effective"].abs().max())
            )

        residual_env = PhysicalKneeResidualVecEnv(wrapped, policy, step_callback=record_step)
        sign_value = 1.0 if args.sign == "plus" else -1.0
        current = residual_env.get_observations()
        with torch.inference_mode():
            for step in range(400):
                records["policy_observation"].append(current["policy"].detach().clone())
                records["critic_observation"].append(current["critic"].detach().clone())
                records["phase_id"].append(semantic_phase_ids(current["policy"]).detach().clone())
                mu = policy.latent_mean(current)
                sigma = policy.latent_std.expand_as(mu)
                innovation = epsilon[step].to(device=env.device)
                latent = mu + sign_value * sigma * innovation
                policy.update_distribution(current)
                log_prob = policy.get_actions_log_prob(latent)
                value = policy.evaluate(current).squeeze(-1)
                records["latent"].append(latent.detach().clone())
                records["mu"].append(mu.detach().clone())
                records["sigma"].append(sigma.detach().clone())
                records["log_prob"].append(log_prob.detach().clone())
                records["value"].append(value.detach().clone())
                current, _, _, _ = residual_env.step(latent)

        tensors = {
            name: torch.stack(records[name], dim=0)
            for name in (
                "policy_observation", "critic_observation", "latent", "mu", "sigma",
                "log_prob", "value", "phase_id", "reward", "term_reward",
                "signed_pitch", "support_outside", "requested", "effective",
            )
        }
        with torch.inference_mode():
            encoded = residual.encoder(tensors["policy_observation"][:200].flatten(0, 1)).reshape(200, 64, 32)
        reward_closure = tensors["reward"] - tensors["term_reward"].sum(dim=-1)
        recovered_epsilon = (tensors["latent"] - tensors["mu"]) / (sign_value * tensors["sigma"])
        requested_abs = float(tensors["requested"].abs().sum())
        effective_abs = float(tensors["effective"].abs().sum())
        phase = tensors["phase_id"]
        technical = {
            "pair_initial_state_exact": pair_init_exact,
            "nondegenerate_reset_perturbation": bool(commit["nondegenerate_perturbation"]),
            "zero_residual_mean": float(tensors["mu"].abs().max()) == 0.0,
            "epsilon_replay_exact": float((recovered_epsilon.cpu() - epsilon).abs().max()) <= 1.0e-6,
            "all_25600_active": int(records["active_count"]) == 400 * 64,
            "no_done_or_timeout": int(records["done_count"]) == 0,
            "all_phase_samples_classified": int((phase >= 0).sum()) == 400 * 64,
            "every_anchor_env_covers_four_phases": all(
                bool(torch.all((phase[:200] == index).sum(dim=0) > 0)) for index in range(4)
            ),
            "reward_closure": float(reward_closure.abs().max()) <= 1.0e-7,
            "non_knee_exact_zero": float(records["non_knee_max_abs"]) == 0.0,
            "physical_bound": float(tensors["requested"].abs().max()) <= 0.003 + 1.0e-8,
            "effective_exact": requested_abs > 0.0 and effective_abs / requested_abs >= 0.999999,
            "source_state_unchanged": state_hash(source) == source_hash_before,
            "residual_state_unchanged": state_hash(residual) == residual_hash_before,
            "source_and_residual_grad_none": all(
                parameter.grad is None for parameter in tuple(source.parameters()) + tuple(residual.parameters())
            ),
            "finite": all(torch.isfinite(value).all() for value in tensors.values()) and torch.isfinite(encoded).all(),
            "optimizer_steps_zero": True,
            "optimizer_state_entries_zero": True,
            "backward_calls_zero": True,
            "checkpoint_count_zero": True,
        }
        valid = all(technical.values())
        bundle = {
            "schema": "x2_phase72_antithetic_evidence_v1",
            "artifact_role": "raw rollout evidence; not a checkpoint",
            "seed_index": args.seed_index,
            "env_seed": args.env_seed,
            "latent_seed": int(seed_record["latent_seed"]),
            "sign": args.sign,
            "sign_value": sign_value,
            "schedule_sha256": sha256(args.schedule),
            "epsilon_tensor_sha256": tensor_hash(epsilon),
            "initial_commit_sha256": sha256(args.init_commit),
            "initial_combined_sha256": commit["combined_sha256"],
            "step_dt_s": step_dt,
            "policy_observation": tensors["policy_observation"].cpu(),
            "critic_observation": tensors["critic_observation"].cpu(),
            "encoded_feature": encoded.cpu(),
            "latent_action": tensors["latent"].cpu(),
            "latent_mean": tensors["mu"].cpu(),
            "latent_sigma": tensors["sigma"].cpu(),
            "old_log_prob": tensors["log_prob"].cpu(),
            "value": tensors["value"].cpu(),
            "total_reward": tensors["reward"].cpu(),
            "reward_term_names": names,
            "reward_term_weights": weights,
            "reward_by_term": tensors["term_reward"].cpu(),
            "signed_pitch_rad": tensors["signed_pitch"].cpu(),
            "support_outside_m": tensors["support_outside"].cpu(),
            "phase_id": tensors["phase_id"].cpu(),
            "requested_knee_offset_rad": tensors["requested"].cpu(),
            "effective_knee_offset_rad": tensors["effective"].cpu(),
        }
        bundle_sha = atomic_bundle(args.bundle, bundle)
        report = {
            "schema": "x2_phase72_antithetic_screen_v1",
            "decision": "VALID_PENDING_PAIR" if valid else "FAIL_INVALID_STOP",
            "seed_index": args.seed_index,
            "env_seed": args.env_seed,
            "latent_seed": int(seed_record["latent_seed"]),
            "sign": args.sign,
            "init_mode": args.init_mode,
            "initial_commit": str(args.init_commit),
            "initial_commit_sha256": sha256(args.init_commit),
            "initial_combined_sha256": commit["combined_sha256"],
            "bundle": str(args.bundle),
            "bundle_sha256": bundle_sha,
            "bundle_bytes": args.bundle.stat().st_size,
            "epsilon_tensor_sha256": tensor_hash(epsilon),
            "requested_max_abs_rad": float(tensors["requested"].abs().max()),
            "effective_requested_abs_ratio": effective_abs / requested_abs,
            "reward_closure_max_abs": float(reward_closure.abs().max()),
            "phase_counts": {str(index): int((phase == index).sum()) for index in range(4)},
            "technical_checks": technical,
            "physics_steps": 400 * 64,
            "optimizer_steps": 0,
            "optimizer_state_entries": 0,
            "backward_calls": 0,
            "checkpoint_count": 0,
            "optimizer_unlocked": False,
            "long_training_unlocked": False,
            "deployment_unlocked": False,
            "task2_complete": False,
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
        if not valid:
            raise RuntimeError("Phase72 launch technical gates failed")
    finally:
        if residual_env is not None:
            residual_env.close()
        elif wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


failure = None
try:
    run()
except Exception as exc:
    failure = exc
    traceback.print_exc()
finally:
    simulation_app.close()
if failure is not None:
    raise failure
