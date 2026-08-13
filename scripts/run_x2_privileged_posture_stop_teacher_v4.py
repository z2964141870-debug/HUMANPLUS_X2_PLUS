#!/usr/bin/env python3
"""Search and validate a state-feedback X2 posture/deceleration/hold teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    with sidecar.open("r+b") as stream:
        os.fsync(stream.fileno())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--stationary", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--failure", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=820)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def prelaunch_guard() -> tuple[dict, str]:
    for output in (
        args.report,
        args.report.with_name(args.report.name + ".sha256"),
        args.failure,
        args.failure.with_name(args.failure.name + ".sha256"),
    ):
        if output.exists():
            raise FileExistsError(output)
    prereg_sidecar = args.prereg.with_name(args.prereg.name + ".sha256")
    prereg_sha = sha256(args.prereg)
    if prereg_sidecar.read_text(encoding="utf-8") != f"{prereg_sha}  {args.prereg.name}\n":
        raise RuntimeError("preregistration sidecar mismatch")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_privileged_posture_stop_teacher_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    runtime = prereg["runtime"]
    if args.num_envs != runtime["num_envs"] or args.steps != runtime["steps"] or args.seed != runtime["seed"]:
        raise RuntimeError("runtime differs from preregistration")
    expected_paths = {
        "source_checkpoint": args.source,
        "stationary_actor": args.stationary,
    }
    for name, path in expected_paths.items():
        if path.resolve() != Path(prereg["immutable_inputs"][name]["path"]).resolve():
            raise RuntimeError(f"runtime input path changed: {name}")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"{section} mismatch: {name}")
    outputs = prereg["outputs"]
    for name, actual in (("screen", args.report), ("failure", args.failure)):
        if Path(outputs[name]).resolve() != actual.resolve():
            raise RuntimeError(f"output path changed: {name}")
    return prereg, prereg_sha


PREREG, PREREG_SHA = prelaunch_guard()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from cwi_x2.privileged_posture_stop_teacher import (  # noqa: E402
    ACTION_DIM,
    FEATURE_DIM,
    HandoffState,
    SEGMENTS,
    advance_handoff,
    feasibility_gates,
    hierarchical_candidate_key,
    state_feedback_residual,
    summarize_segment,
    teacher_features,
)
from cwi_x2.transition_command import TransitionVelocityCommand, transition_velocity_cfg  # noqa: E402
from x2_native_locomotion_posture_phase60 import (  # noqa: E402
    actual_support_com_outside_distance,
    signed_root_pitch_rad,
)


LOWER15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
CRUISE_START_S = 2.0
DECEL_START_S = 6.2
HOLD_START_S = 8.2


class PostureStopVelocityCommand(TransitionVelocityCommand):
    """One matched event with a stratified schedule-to-gait phase offset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        self._teacher_offset_s = ids.remainder(16).to(torch.float32) / 15.0 * 0.60
        self.cruise_speed.fill_(0.35)
        self.vel_command_b[:, 0] = 0.35
        self.vel_command_b[:, 1:] = 0.0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if hasattr(self, "_teacher_offset_s"):
            resolved = self._resolved_env_ids(env_ids)
            self.cruise_speed[resolved] = 0.35
            self.vel_command_b[resolved, 0] = 0.35
            self.vel_command_b[resolved, 1:] = 0.0

    def _elapsed_s(self):
        elapsed = self._env.episode_length_buf.to(torch.float32) * float(self._env.step_dt)
        offset = getattr(self, "_teacher_offset_s", torch.zeros_like(elapsed))
        return elapsed + offset


def build_cfg():
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = PREREG["immutable_inputs"]["template"]["path"]
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    reset = PREREG["runtime"]["reset_panel"]
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0),
        "roll": (-reset["roll_pitch_rad"], reset["roll_pitch_rad"]),
        "pitch": (-reset["roll_pitch_rad"], reset["roll_pitch_rad"]),
        "yaw": (-reset["yaw_rad"], reset["yaw_rad"]),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-reset["linear_velocity_mps"], reset["linear_velocity_mps"]),
        "y": (-reset["linear_velocity_mps"], reset["linear_velocity_mps"]),
        "z": (0.0, 0.0),
        "roll": (-reset["angular_velocity_radps"], reset["angular_velocity_radps"]),
        "pitch": (-reset["angular_velocity_radps"], reset["angular_velocity_radps"]),
        "yaw": (-reset["angular_velocity_radps"], reset["angular_velocity_radps"]),
    }
    cfg.events.reset_robot_joints.params["position_range"] = tuple(reset["joint_position_scale"])
    cfg.events.reset_robot_joints.params["velocity_range"] = (
        -reset["joint_velocity_radps"], reset["joint_velocity_radps"]
    )
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
        stand_s=1.0,
        accelerate_s=1.0,
        cruise_s=4.2,
        decelerate_s=2.0,
        maximum_phase_offset_s=0.0,
    )
    transition.class_type = PostureStopVelocityCommand
    cfg.commands.base_velocity = transition
    cfg.episode_length_s = 20.0
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
    return cfg


def build_source(obs, device):
    payload = torch.load(args.source, map_location=device, weights_only=False)
    if int(payload.get("iter", -1)) != 2600:
        raise RuntimeError("source iteration changed")
    model = ActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        init_noise_std=0.4,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}:{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_hash(model) -> str:
    return tensor_hash(torch.cat([value.reshape(-1) for value in model.state_dict().values()]))


def roll_angle(robot) -> torch.Tensor:
    gravity = robot.data.projected_gravity_b
    return torch.atan2(-gravity[:, 1], -gravity[:, 2])


def forced_gait_suffix(elapsed_s: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    phase = torch.remainder(elapsed_s / 0.8, 1.0)
    angle = 2.0 * math.pi * phase
    right_swing = (phase >= 0.075) & (phase < 0.425)
    left_swing = (phase >= 0.575) & (phase < 0.925)
    contacts = torch.stack((~left_swing, ~right_swing), dim=-1)
    contacts = torch.where(moving.unsqueeze(-1), contacts, torch.ones_like(contacts))
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    clock = clock * moving.to(clock.dtype).unsqueeze(-1)
    return torch.cat((clock, contacts.to(clock.dtype)), dim=-1)


def actor_input(observation, command, previous_action, elapsed_s, force_moving):
    policy = observation["policy"].clone()
    policy[:, 9:12] = command
    policy[:, 74:89] = previous_action
    moving = force_moving | (torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10)
    policy[:, 89:93] = forced_gait_suffix(elapsed_s, moving)
    return {"policy": policy, "critic": observation["critic"]}


def stationary_action(session, policy_observation, previous_action, device):
    stand_obs = policy_observation.clone()
    stand_obs[:, 9:12] = 0.0
    stand_obs[:, 74:89] = previous_action
    stand_obs[:, 89:93] = stand_obs.new_tensor([0.0, 0.0, 1.0, 1.0])
    output = session.run(["actions"], {"obs": stand_obs.detach().cpu().numpy()})[0]
    result = torch.as_tensor(output, device=device, dtype=torch.float32)
    if result.shape != (policy_observation.shape[0], ACTION_DIM) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("stationary actor output contract changed")
    return result.clamp(-1.0, 1.0)


def segment_id(elapsed_s: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(elapsed_s, -1, dtype=torch.int64)
    result = torch.where((elapsed_s >= CRUISE_START_S) & (elapsed_s < DECEL_START_S), 0, result)
    result = torch.where((elapsed_s >= DECEL_START_S) & (elapsed_s < HOLD_START_S), 1, result)
    result = torch.where(elapsed_s >= HOLD_START_S, 2, result)
    return result


def initialize_matrix(direction: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(ACTION_DIM, FEATURE_DIM, dtype=torch.float32)
    scale = math.sqrt(FEATURE_DIM / 5.0) * (0.10 / 0.12)
    for source_column, feature_column in enumerate((0, 10, 11, 12, 13)):
        matrix[:, feature_column] = direction[:, source_column].cpu() * scale
    return matrix


def rows_to_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        segment: summarize_segment([row for row in rows if row["segment"] == segment])
        for segment in SEGMENTS
    }


def paired_bootstrap(source_rows, candidate_rows, field, *, seed: int, draws: int = 4096):
    source = {(row["env_id"], row["segment"]): row for row in source_rows}
    candidate = {(row["env_id"], row["segment"]): row for row in candidate_rows}
    keys = sorted(set(source) & set(candidate))
    values = np.asarray([candidate[key][field] - source[key][field] for key in keys], dtype=np.float64)
    if len(values) != args.num_envs or not np.isfinite(values).all():
        raise RuntimeError("paired bootstrap inputs are incomplete")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    sample = values[indices].mean(axis=1)
    return {
        "point": float(values.mean()),
        "ci95": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
        "environment_pairs": len(values),
        "draws": draws,
    }


def run() -> dict:
    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    observation = wrapped.get_observations()
    if tuple(observation["policy"].shape) != (args.num_envs, 93):
        raise RuntimeError("policy observation contract changed")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15 or not bool(term._cwi_upper_zero_mask.all()):
        raise RuntimeError("action/fixed-upper contract changed")
    source = build_source(observation, env.device)
    source_before = model_hash(source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    stationary = ort.InferenceSession(str(args.stationary), sess_options=options, providers=["CPUExecutionProvider"])
    if stationary.get_inputs()[0].name != "obs" or stationary.get_outputs()[0].name != "actions":
        raise RuntimeError("stationary ONNX input/output names changed")
    direction_payload = json.loads(Path(PREREG["immutable_evidence"]["v2b_result"]["path"]).read_text())
    direction = torch.as_tensor(direction_payload["search"]["best_parameters_15x5"], dtype=torch.float32)
    initial_mean = initialize_matrix(direction)
    robot = env.scene["robot"]
    foot_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    group_ids = torch.div(torch.arange(args.num_envs, device=env.device), 16, rounding_mode="floor")
    if torch.bincount(group_ids, minlength=16).tolist() != [16] * 16:
        raise RuntimeError("search group allocation changed")
    limits = PREREG["gates"]
    fingerprints: list[dict] = []

    def rollout(mode: str, matrices: torch.Tensor, replay_seed: int, label: str):
        torch.manual_seed(replay_seed)
        torch.cuda.manual_seed_all(replay_seed)
        observation, _ = wrapped.reset()
        initial = {
            "label": label,
            "seed": replay_seed,
            "policy": tensor_hash(observation["policy"]),
            "root": tensor_hash(torch.cat((robot.data.root_pos_w, robot.data.root_quat_w, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)),
            "joint": tensor_hash(torch.cat((robot.data.joint_pos, robot.data.joint_vel), dim=-1)),
        }
        fingerprints.append(initial)
        count = args.num_envs
        main_previous = torch.zeros(count, ACTION_DIM, device=env.device)
        stand_previous = torch.zeros_like(main_previous)
        previous_residual = torch.zeros_like(main_previous)
        last_executed = torch.zeros_like(main_previous)
        handoff = HandoffState(
            dwell_steps=torch.zeros(count, device=env.device, dtype=torch.int64),
            latched=torch.zeros(count, device=env.device, dtype=torch.bool),
            blend=torch.zeros(count, device=env.device),
        )
        alive = torch.ones(count, device=env.device, dtype=torch.bool)
        terminated = torch.zeros_like(alive)
        timed_out = torch.zeros_like(alive)
        termination_segment = torch.full((count,), -1, device=env.device, dtype=torch.int64)
        root_height_min = robot.data.root_pos_w[:, 2].clone()
        tilt_max = torch.zeros(count, device=env.device)
        residual_abs_max = torch.zeros(count, device=env.device)
        residual_step_abs_max = torch.zeros(count, device=env.device)
        sample_lists = {name: [] for name in (
            "segment", "pitch", "velocity_sq", "lateral_sq", "yaw_sq", "support",
            "slip", "flight", "root_height", "tilt", "speed", "double_support",
            "residual", "residual_step", "action_clip",
            "effectiveness", "wrong_sign",
        )}
        with torch.no_grad():
            for step in range(args.steps):
                elapsed = env.command_manager.get_term("base_velocity")._elapsed_s()
                scheduled_speed = env.command_manager.get_term("base_velocity")._scheduled_speed(elapsed)
                terminal = elapsed >= HOLD_START_S
                segment = segment_id(elapsed)
                gait_elapsed = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
                pitch = signed_root_pitch_rad(robot)
                roll = roll_angle(robot)
                support, contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
                forces = torch.stack(tuple(
                    env.scene[name].data.force_matrix_w[..., 2].abs().reshape(count, -1).amax(dim=-1)
                    for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                ), dim=-1)
                contact = forces > 10.0
                speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                quat = robot.data.root_quat_w
                tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0))

                feedback_mask = torch.ones(count, device=env.device, dtype=torch.bool)
                fixed_mask = torch.zeros_like(feedback_mask)
                if mode == "search":
                    feedback_mask = group_ids != 0
                elif mode == "source_direct":
                    feedback_mask.zero_()
                elif mode == "fixed_half_direct":
                    feedback_mask.zero_()
                    fixed_mask.fill_(True)
                elif mode != "feedback_gated":
                    raise ValueError(f"unknown rollout mode: {mode}")

                handoff_next = advance_handoff(
                    handoff,
                    speed_mps=speed,
                    contact_count=contact_count,
                    tilt_rad=tilt,
                    support_outside_m=support,
                    terminal=terminal & feedback_mask,
                    dwell_required=PREREG["teacher"]["handoff_dwell_steps"],
                    blend_steps=PREREG["teacher"]["handoff_blend_steps"],
                    speed_max_mps=PREREG["teacher"]["handoff_speed_max_mps"],
                    tilt_max_rad=PREREG["teacher"]["handoff_tilt_max_rad"],
                    support_max_m=PREREG["teacher"]["handoff_support_max_m"],
                )
                # Direct controls enter stationary authority immediately at the
                # terminal boundary.  Feedback lanes keep the gait alive until
                # the physical handoff gate and then blend over one second.
                direct_mask = ~feedback_mask
                blend = torch.where(direct_mask & terminal, torch.ones_like(handoff_next.blend), handoff_next.blend)
                pre_move = elapsed < 1.0
                blend = torch.where(pre_move, torch.ones_like(blend), blend)
                handoff = handoff_next

                scheduled_command = torch.zeros(count, 3, device=env.device)
                scheduled_command[:, 0] = scheduled_speed
                brake = torch.clamp(-0.5 * robot.data.root_lin_vel_b[:, :2], -0.20, 0.20)
                brake_norm = torch.linalg.vector_norm(brake, dim=-1)
                need_floor = terminal & feedback_mask & (speed > 0.10) & (brake_norm < 0.11)
                direction_xy = -robot.data.root_lin_vel_b[:, :2] / speed.clamp_min(1.0e-6).unsqueeze(-1)
                brake = torch.where(need_floor.unsqueeze(-1), 0.11 * direction_xy, brake)
                control_command = scheduled_command.clone()
                braking = terminal & feedback_mask & (blend < 1.0)
                control_command[:, :2] = torch.where(braking.unsqueeze(-1), brake, control_command[:, :2])
                control_command = torch.where((blend >= 1.0).unsqueeze(-1), torch.zeros_like(control_command), control_command)
                force_moving = braking

                main_input = actor_input(observation, control_command, main_previous, gait_elapsed, force_moving)
                main_action = source.act_inference(main_input).clamp(-1.0, 1.0)
                stand_raw = stationary_action(stationary, observation["policy"], stand_previous, env.device)
                main_stand_input = actor_input(
                    observation,
                    torch.zeros_like(control_command),
                    main_previous,
                    gait_elapsed,
                    torch.zeros_like(force_moving),
                )
                main_stand = source.act_inference(main_stand_input).clamp(-1.0, 1.0)
                stand_action = torch.clamp(0.5 * main_stand + 0.5 * stand_raw, -1.0, 1.0)
                base_action = (1.0 - blend).unsqueeze(-1) * main_action + blend.unsqueeze(-1) * stand_action

                residual = torch.zeros_like(base_action)
                if bool(fixed_mask.any()):
                    gait = observation["policy"][:, 89:93]
                    phase_features = torch.stack(
                        (torch.ones(count, device=env.device), gait[:, 0], gait[:, 1], gait[:, 2] - 0.5, gait[:, 3] - 0.5),
                        dim=-1,
                    )
                    signal = torch.einsum("af,nf->na", direction.to(env.device), phase_features) / math.sqrt(5.0)
                    fixed = 0.10 * torch.tanh(signal)
                    moving = scheduled_speed > 0.10
                    residual = torch.where((fixed_mask & moving).unsqueeze(-1), fixed, residual)
                if bool(feedback_mask.any()):
                    gait = forced_gait_suffix(gait_elapsed, torch.linalg.vector_norm(control_command[:, :2], dim=-1) > 0.10)
                    features = teacher_features(
                        pitch_rad=pitch,
                        pitch_rate_radps=robot.data.root_ang_vel_b[:, 1],
                        roll_rad=roll,
                        roll_rate_radps=robot.data.root_ang_vel_b[:, 0],
                        body_velocity_xy_mps=robot.data.root_lin_vel_b[:, :2],
                        command_velocity_xy_mps=scheduled_command[:, :2],
                        yaw_rate_error_radps=robot.data.root_ang_vel_b[:, 2] - scheduled_command[:, 2],
                        support_outside_m=support,
                        contact=contact,
                        gait_suffix=gait,
                        decelerating=(elapsed >= DECEL_START_S) & (elapsed < HOLD_START_S),
                        holding=terminal & handoff.latched,
                    )
                    selected_matrix = matrices[group_ids] if mode == "search" else matrices[0].expand(count, -1, -1)
                    feedback = state_feedback_residual(
                        selected_matrix,
                        features,
                        previous_residual,
                        contact_count=contact_count,
                        tilt_rad=tilt,
                        support_outside_m=support,
                        holding=terminal & handoff.latched,
                        maximum_abs=PREREG["teacher"]["residual_abs_max"],
                        hold_maximum_abs=PREREG["teacher"]["hold_residual_abs_max"],
                        maximum_step=PREREG["teacher"]["residual_step_abs_max"],
                    )
                    feedback_active = feedback_mask & (elapsed >= 1.0)
                    residual = torch.where(feedback_active.unsqueeze(-1), feedback, residual)

                proposed = base_action + residual
                action = proposed.clamp(-1.0, 1.0)
                realized = action - base_action
                residual_step = (realized - previous_residual).abs().amax(dim=-1)
                residual_abs = realized.abs().amax(dim=-1)
                command_term = env.command_manager.get_term("base_velocity")
                command_term.vel_command_b[:] = control_command
                next_observation, _reward, done, extras = wrapped.step(action)
                done = done.reshape(-1).bool()
                time_out = extras.get("time_outs", torch.zeros_like(done)).reshape(-1).bool()

                # Reconstruct the same action/template/plant/final-limit path
                # for the no-teacher base action.  This distinguishes baseline
                # saturation from intervention-induced clipping and proves the
                # requested residual actually reaches the physical PD target.
                candidate_preclip = term.preclip_combined_actions
                source_preclip = candidate_preclip - action + base_action
                normalized_clip = (
                    (candidate_preclip.abs() > 1.0)
                    & (source_preclip.abs() <= 1.0)
                ).any(dim=-1)
                source_combined = source_preclip.clamp(-1.0, 1.0)
                source_target = source_combined * term._scale + term._offset
                candidate_unclipped_target = candidate_preclip.clamp(-1.0, 1.0) * term._scale + term._offset
                source_final_clip = torch.zeros(count, device=env.device, dtype=torch.bool)
                candidate_final_clip = torch.zeros_like(source_final_clip)
                if term.cfg.clip is not None:
                    source_final_clip = (
                        (source_target < term._clip[:, :, 0])
                        | (source_target > term._clip[:, :, 1])
                    ).any(dim=-1)
                    candidate_final_clip = (
                        (candidate_unclipped_target < term._clip[:, :, 0])
                        | (candidate_unclipped_target > term._clip[:, :, 1])
                    ).any(dim=-1)
                    source_target = torch.clamp(
                        source_target,
                        min=term._clip[:, :, 0],
                        max=term._clip[:, :, 1],
                    )
                intervention_clip = (
                    normalized_clip | (candidate_final_clip & ~source_final_clip)
                ).to(torch.float32)
                requested_target = (action - base_action) * term._scale
                effective_target = term._processed_actions - source_target
                requested_norm = torch.linalg.vector_norm(requested_target, dim=-1)
                effective_norm = torch.linalg.vector_norm(effective_target, dim=-1)
                active_request = requested_norm > 1.0e-7
                effectiveness = torch.where(
                    active_request,
                    effective_norm / requested_norm.clamp_min(1.0e-12),
                    torch.ones_like(requested_norm),
                )
                wrong_sign = (
                    (effective_target * requested_target < -1.0e-10)
                    & (requested_target.abs() > 1.0e-7)
                ).any(dim=-1).to(torch.float32)

                post_pitch = signed_root_pitch_rad(robot)
                post_support, post_contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
                post_forces = torch.stack(tuple(
                    env.scene[name].data.force_matrix_w[..., 2].abs().reshape(count, -1).amax(dim=-1)
                    for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                ), dim=-1)
                post_contact = post_forces > 10.0
                foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
                slip = (foot_speed * post_contact).sum(-1) / post_contact_count.clamp_min(1)
                post_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                post_quat = robot.data.root_quat_w
                post_tilt = torch.acos(torch.clamp(1.0 - 2.0 * (post_quat[:, 1].square() + post_quat[:, 2].square()), -1.0, 1.0))
                velocity_sq = torch.square(robot.data.root_lin_vel_b[:, :2] - scheduled_command[:, :2]).sum(dim=-1)
                lateral_sq = torch.square(robot.data.root_lin_vel_b[:, 1])
                yaw_sq = torch.square(robot.data.root_ang_vel_b[:, 2] - scheduled_command[:, 2])

                valid = alive & (segment >= 0)
                def masked(value):
                    return torch.where(valid, value, torch.full_like(value, torch.nan)).detach().cpu()

                sample_lists["segment"].append(segment.detach().cpu())
                sample_lists["pitch"].append(masked(post_pitch))
                sample_lists["velocity_sq"].append(masked(velocity_sq))
                sample_lists["lateral_sq"].append(masked(lateral_sq))
                sample_lists["yaw_sq"].append(masked(yaw_sq))
                sample_lists["support"].append(masked(post_support))
                sample_lists["slip"].append(masked(slip))
                sample_lists["flight"].append(masked((post_contact_count == 0).to(torch.float32)))
                sample_lists["root_height"].append(masked(robot.data.root_pos_w[:, 2]))
                sample_lists["tilt"].append(masked(post_tilt))
                sample_lists["speed"].append(masked(post_speed))
                sample_lists["double_support"].append(masked((post_contact_count == 2).to(torch.float32)))
                sample_lists["residual"].append(masked(residual_abs))
                sample_lists["residual_step"].append(masked(residual_step))
                sample_lists["action_clip"].append(masked(intervention_clip))
                sample_lists["effectiveness"].append(masked(effectiveness))
                sample_lists["wrong_sign"].append(masked(wrong_sign))

                residual_abs_max = torch.maximum(residual_abs_max, residual_abs)
                residual_step_abs_max = torch.maximum(residual_step_abs_max, residual_step)
                root_height_min = torch.where(alive, torch.minimum(root_height_min, robot.data.root_pos_w[:, 2]), root_height_min)
                tilt_max = torch.where(alive, torch.maximum(tilt_max, post_tilt), tilt_max)
                newly_done = alive & done
                termination_segment = torch.where(newly_done, segment, termination_segment)
                terminated |= newly_done & ~time_out
                timed_out |= newly_done & time_out
                alive &= ~done
                previous_residual = realized
                last_executed = action
                main_previous = last_executed
                stand_previous = last_executed
                observation = next_observation

        stacked = {name: torch.stack(values, dim=0) for name, values in sample_lists.items()}
        rows = []
        treatment = mode
        for env_id in range(count):
            if mode == "search":
                treatment = "source_direct" if int(group_ids[env_id]) == 0 else "feedback_gated"
            for index, name in enumerate(SEGMENTS):
                mask = (stacked["segment"][:, env_id] == index) & torch.isfinite(stacked["pitch"][:, env_id])
                sample_count = int(mask.sum())
                if sample_count:
                    def selected(metric):
                        return stacked[metric][:, env_id][mask]
                    pitch_values = selected("pitch")
                    row = {
                        "pitch_mean_rad": float(pitch_values.mean()),
                        "pitch_p05_rad": float(torch.quantile(pitch_values, 0.05)),
                        "velocity_mse": float(selected("velocity_sq").mean()),
                        "lateral_mse": float(selected("lateral_sq").mean()),
                        "yaw_mse": float(selected("yaw_sq").mean()),
                        "support_mean_m": float(selected("support").mean()),
                        "slip_p95_mps": float(torch.quantile(selected("slip"), 0.95)),
                        "flight_fraction": float(selected("flight").mean()),
                        "root_height_min_m": float(selected("root_height").min()),
                        "tilt_max_rad": float(selected("tilt").max()),
                        "speed_p95_mps": float(torch.quantile(selected("speed"), 0.95)),
                        "double_support_fraction": float(selected("double_support").mean()),
                        "residual_abs_max": float(selected("residual").max()),
                        "residual_step_abs_max": float(selected("residual_step").max()),
                        "action_clip_fraction": float(selected("action_clip").mean()),
                        "effectiveness_ratio_min": float(selected("effectiveness").min()),
                        "wrong_sign_fraction": float(selected("wrong_sign").mean()),
                    }
                else:
                    row = {
                        "pitch_mean_rad": -math.pi, "pitch_p05_rad": -math.pi,
                        "velocity_mse": 1.0e6, "lateral_mse": 1.0e6, "yaw_mse": 1.0e6,
                        "support_mean_m": 1.0, "slip_p95_mps": 10.0, "flight_fraction": 1.0,
                        "root_height_min_m": float(root_height_min[env_id]),
                        "tilt_max_rad": max(float(tilt_max[env_id]), math.pi),
                        "speed_p95_mps": 10.0, "double_support_fraction": 0.0,
                        "residual_abs_max": float(residual_abs_max[env_id]),
                        "residual_step_abs_max": float(residual_step_abs_max[env_id]),
                        "action_clip_fraction": 1.0,
                        "effectiveness_ratio_min": 0.0,
                        "wrong_sign_fraction": 1.0,
                    }
                row.update({
                    "env_id": env_id,
                    "search_group": int(group_ids[env_id]),
                    "treatment": treatment,
                    "segment": name,
                    "sample_count": sample_count,
                    "terminated": bool(
                        terminated[env_id]
                        and index == max(0, int(termination_segment[env_id]))
                    ),
                    "time_out": bool(
                        timed_out[env_id]
                        and index == max(0, int(termination_segment[env_id]))
                    ),
                })
                rows.append(row)
        return rows, initial

    search = PREREG["search"]
    generator = torch.Generator(device="cpu").manual_seed(search["parameter_seed"])
    mean = initial_mean.clone()
    std = torch.full_like(mean, search["initial_std"])
    std[:, :1] = search["initial_bias_std"]
    best = None
    iterations = []
    for iteration in range(search["iterations"]):
        samples = []
        for candidate in range(15):
            if candidate == 0:
                sample = mean.clone()
            else:
                sample = mean + std * torch.randn(mean.shape, generator=generator)
            samples.append(sample.clamp(search["lower_bound"], search["upper_bound"]))
        matrices = torch.stack((torch.zeros_like(mean), *samples), dim=0).to(env.device)
        rows, _ = rollout("search", matrices, search["reset_seeds"][iteration], f"search_{iteration}")
        source_rows = [row for row in rows if row["search_group"] == 0]
        source_summary = rows_to_summary(source_rows)
        candidates = []
        for candidate in range(15):
            group = candidate + 1
            candidate_rows = [row for row in rows if row["search_group"] == group]
            summary = rows_to_summary(candidate_rows)
            key = hierarchical_candidate_key(source_summary, summary, limits)
            record = {
                "candidate": candidate,
                "group": group,
                "key": list(key),
                "summary": summary,
            }
            candidates.append(record)
            if best is None or key < tuple(best["key"]):
                best = {"key": list(key), "matrix": samples[candidate].clone(), "iteration": iteration, **record}
        ordered = sorted(candidates, key=lambda row: tuple(row["key"]))
        elite_indices = [row["candidate"] for row in ordered[: search["elites"]]]
        elite = torch.stack([samples[index] for index in elite_indices])
        mean = elite.mean(dim=0)
        std = torch.maximum(elite.std(dim=0, unbiased=False), torch.full_like(std, search["std_floor"]))
        iterations.append({
            "iteration": iteration,
            "reset_seed": search["reset_seeds"][iteration],
            "best_key": ordered[0]["key"],
            "elite_candidates": elite_indices,
            "candidates": candidates,
        })
    if best is None:
        raise RuntimeError("search produced no candidate")

    best_matrix = best["matrix"].to(env.device).unsqueeze(0)
    validation_seed = PREREG["validation"]["seed"]
    source_rows, source_fp = rollout("source_direct", best_matrix, validation_seed, "validation_source")
    fixed_rows, fixed_fp = rollout("fixed_half_direct", best_matrix, validation_seed, "validation_fixed_half")
    feedback_rows, feedback_fp = rollout("feedback_gated", best_matrix, validation_seed, "validation_feedback")
    fingerprint_exact = (
        source_fp["policy"] == fixed_fp["policy"] == feedback_fp["policy"]
        and source_fp["root"] == fixed_fp["root"] == feedback_fp["root"]
        and source_fp["joint"] == fixed_fp["joint"] == feedback_fp["joint"]
    )
    summaries = {
        "source_direct": rows_to_summary(source_rows),
        "fixed_half_direct": rows_to_summary(fixed_rows),
        "feedback_gated": rows_to_summary(feedback_rows),
    }
    gates = feasibility_gates(summaries["source_direct"], summaries["feedback_gated"], limits)
    cruise_bootstrap = paired_bootstrap(
        [row for row in source_rows if row["segment"] == "cruise"],
        [row for row in feedback_rows if row["segment"] == "cruise"],
        "pitch_mean_rad", seed=PREREG["validation"]["bootstrap_seed"],
    )
    decel_bootstrap = paired_bootstrap(
        [row for row in source_rows if row["segment"] == "decelerate"],
        [row for row in feedback_rows if row["segment"] == "decelerate"],
        "pitch_mean_rad", seed=PREREG["validation"]["bootstrap_seed"] + 1,
    )
    gates["cruise_pitch_ci"] = cruise_bootstrap["ci95"][0] >= limits["cruise_pitch_ci_lower_min"]
    gates["decel_pitch_ci"] = decel_bootstrap["ci95"][0] >= limits["decel_pitch_ci_lower_min"]
    gates["reset_fingerprint_exact"] = fingerprint_exact
    source_after = model_hash(source)
    technical = {
        "finite_rows": all(
            all(not isinstance(value, float) or math.isfinite(value) for value in row.values())
            for row in source_rows + fixed_rows + feedback_rows
        ),
        "all_segments_present": all(
            sum(row["segment"] == segment for row in rows) == args.num_envs
            for rows in (source_rows, fixed_rows, feedback_rows)
            for segment in SEGMENTS
        ),
        "reset_fingerprint_exact": fingerprint_exact,
        "source_model_immutable": source_before == source_after,
        "source_hash_matches_prereg": sha256(args.source) == PREREG["immutable_inputs"]["source_checkpoint"]["sha256"],
        "stationary_hash_matches_prereg": sha256(args.stationary) == PREREG["immutable_inputs"]["stationary_actor"]["sha256"],
        "optimizer_steps_zero": True,
        "backward_calls_zero": True,
        "checkpoint_writes_zero": True,
    }
    valid = all(technical.values())
    full_pass = valid and all(gates.values())
    cruise_names = [
        "cruise_pitch_mean", "cruise_pitch_p05", "decel_pitch_mean", "decel_pitch_p05",
        "cruise_zero_termination", "cruise_zero_timeout", "decel_zero_termination", "decel_zero_timeout",
        "cruise_velocity", "cruise_lateral", "cruise_yaw", "cruise_support", "cruise_slip", "cruise_flight",
        "cruise_root_height", "cruise_tilt",
        "decel_velocity", "decel_lateral", "decel_yaw", "decel_support", "decel_slip", "decel_flight",
        "decel_root_height", "decel_tilt",
        "cruise_pitch_ci", "decel_pitch_ci", "residual_bound", "residual_slew", "no_action_clip",
        "physical_effectiveness", "physical_sign",
    ]
    cruise_pass = valid and all(gates[name] for name in cruise_names)
    hold_names = [name for name in gates if name.startswith("hold_") or name.startswith("candidat")]
    hold_pass = valid and all(gates[name] for name in hold_names)
    if not valid:
        decision = "FAIL_IMPLEMENTATION_STOP"
    elif full_pass:
        decision = "PASS_STATE_FEEDBACK_TEACHER_BC_PREREG_ONLY"
    elif cruise_pass and not hold_pass:
        decision = "CRUISE_FEASIBLE_BRAKE_HOLD_SKILL_PREREG_ONLY"
    else:
        decision = "FAIL_STATE_FEEDBACK_TEACHER_NEW_ACTOR_PREREG_ONLY"
    return {
        "schema": "x2_privileged_posture_stop_teacher_result_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": decision,
        "runtime": {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "duration_s": args.steps * float(env.step_dt),
            "control_steps": (search["iterations"] + 3) * args.steps,
            "wall_time_s": time.monotonic() - started,
        },
        "search": {
            "iterations": iterations,
            "best_iteration": best["iteration"],
            "best_candidate": best["candidate"],
            "best_lexicographic_key": best["key"],
            "best_matrix_15x17": best["matrix"].tolist(),
        },
        "validation": {
            "summaries": summaries,
            "gates": gates,
            "cruise_pitch_paired_bootstrap": cruise_bootstrap,
            "decelerate_pitch_paired_bootstrap": decel_bootstrap,
            "initial_fingerprints": [source_fp, fixed_fp, feedback_fp],
            "per_env": {
                "source_direct": source_rows,
                "fixed_half_direct": fixed_rows,
                "feedback_gated": feedback_rows,
            },
        },
        "technical_checks": technical,
        "evidence_boundary": {
            "privileged_state_feedback": True,
            "fixed_direction_is_diagnostic_only": True,
            "scalar_reward_used": False,
            "critic_used": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "bc_dagger_preregistration_unlocked": decision == "PASS_STATE_FEEDBACK_TEACHER_BC_PREREG_ONLY",
            "brake_hold_skill_preregistration_unlocked": decision == "CRUISE_FEASIBLE_BRAKE_HOLD_SKILL_PREREG_ONLY",
            "new_actor_preregistration_unlocked": decision == "FAIL_STATE_FEEDBACK_TEACHER_NEW_ACTOR_PREREG_ONLY",
            "training_unlocked": False,
            "deployment_unlocked": False,
            "whole_body_training_unlocked": False,
        },
    }


def fail(error: BaseException) -> None:
    payload = {
        "schema": "x2_privileged_posture_stop_teacher_failure_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "FAIL_IMPLEMENTATION_STOP",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
    }
    try:
        atomic_json(args.failure, payload)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(1)


try:
    result = run()
    atomic_json(args.report, result)
    print(json.dumps({"decision": result["decision"], "technical": result["technical_checks"]}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result["decision"] != "FAIL_IMPLEMENTATION_STOP" else 1)
except BaseException as error:
    fail(error)
