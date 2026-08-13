#!/usr/bin/env python3
"""Run one fresh-seed direct-vs-locomotion-zero hold confirmation segment."""

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
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--stationary", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--failure", type=Path, required=True)
parser.add_argument("--seed-index", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=820)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def prelaunch_guard() -> tuple[dict, str]:
    for path in (args.report, args.report.with_name(args.report.name + ".sha256"),
                 args.failure, args.failure.with_name(args.failure.name + ".sha256")):
        if path.exists():
            raise FileExistsError(path)
    prereg_sidecar = args.prereg.with_name(args.prereg.name + ".sha256")
    prereg_sha = sha256(args.prereg)
    if prereg_sidecar.read_text(encoding="utf-8") != f"{prereg_sha}  {args.prereg.name}\n":
        raise RuntimeError("preregistration sidecar mismatch")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_locomotion_zero_hold_confirmation_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    runtime = prereg["runtime"]
    if args.seed_index not in (0, 1, 2) or args.seed != runtime["seeds"][args.seed_index]:
        raise RuntimeError("seed/index differs from preregistration")
    if args.num_envs != runtime["num_envs"] or args.steps != runtime["steps"]:
        raise RuntimeError("runtime shape differs from preregistration")
    for name, actual in (("source_checkpoint", args.source), ("stationary_actor", args.stationary)):
        if actual.resolve() != Path(prereg["immutable_inputs"][name]["path"]).resolve():
            raise RuntimeError(f"runtime input path changed: {name}")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"{section} mismatch: {name}")
    run = prereg["runs"][args.seed_index]
    if Path(run["screen"]).resolve() != args.report.resolve() or Path(run["failure"]).resolve() != args.failure.resolve():
        raise RuntimeError("output namespace changed")
    return prereg, prereg_sha


PREREG, PREREG_SHA = prelaunch_guard()


def fail_fast(exc: BaseException, stage: str) -> None:
    payload = {
        "schema": "x2_locomotion_zero_hold_confirmation_failure_v1",
        "prereg_sha256": PREREG_SHA,
        "seed_index": args.seed_index,
        "seed": args.seed,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
    }
    try:
        atomic_json(args.failure, payload)
    except BaseException:
        pass
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


try:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
except BaseException as exc:
    fail_fast(exc, "app_launcher")

try:
    import onnxruntime as ort
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.modules import ActorCritic
    from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response
    from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE
    from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg
    from cwi_x2.locomotion_zero_confirmation import (
        SEGMENTS, TREATMENTS, checkerboard_assignment, seed_gates,
        summarize_treatment, validate_assignment,
    )
    from cwi_x2.transition_command import TransitionVelocityCommand, transition_velocity_cfg
    from x2_native_locomotion_posture_phase60 import (
        actual_support_com_outside_distance, signed_root_pitch_rad,
    )
except BaseException as exc:
    fail_fast(exc, "post_app_import")


ACTION_DIM = 15
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


class ConfirmationVelocityCommand(TransitionVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        self._confirmation_offset_s = ids.remainder(16).to(torch.float32) / 15.0 * 0.60
        self.cruise_speed.fill_(0.35)
        self.vel_command_b[:, 0] = 0.35
        self.vel_command_b[:, 1:] = 0.0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if hasattr(self, "_confirmation_offset_s"):
            resolved = self._resolved_env_ids(env_ids)
            self.cruise_speed[resolved] = 0.35
            self.vel_command_b[resolved, 0] = 0.35
            self.vel_command_b[resolved, 1:] = 0.0

    def _elapsed_s(self):
        elapsed = self._env.episode_length_buf.to(torch.float32) * float(self._env.step_dt)
        return elapsed + getattr(self, "_confirmation_offset_s", torch.zeros_like(elapsed))


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
        "y": (-reset["linear_velocity_mps"], reset["linear_velocity_mps"]), "z": (0.0, 0.0),
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
        cfg.commands.base_velocity, ideal_env_fraction=1.0,
        ideal_heading_control_stiffness=1.0, response_heading_control_stiffness=1.0,
        stand_s=1.0, accelerate_s=1.0, cruise_s=4.2, decelerate_s=2.0,
        maximum_phase_offset_s=0.0,
    )
    transition.class_type = ConfirmationVelocityCommand
    cfg.commands.base_velocity = transition
    cfg.episode_length_s = 20.0
    _apply_x2_actuator_response(
        cfg.scene.robot,
        {"enabled": True, "profile": "session03_session04_group", "randomize": False,
         "strength": 1.0, "filter_strength": 1.0, "delay_strength": 1.0,
         "include_ideal_endpoint": False, "ideal_env_fraction": 1.0,
         "filter_only_env_fraction": 0.0},
        physics_dt_sec=cfg.sim.dt,
    )
    return cfg


def build_source(obs, device):
    payload = torch.load(args.source, map_location=device, weights_only=False)
    if int(payload.get("iter", -1)) != 2600:
        raise RuntimeError("source iteration changed")
    model = ActorCritic(
        obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM, actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128], activation="elu", init_noise_std=0.4,
        noise_std_type="scalar", actor_obs_normalization=False, critic_obs_normalization=False,
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


def actor_input(observation, command, previous_action, elapsed_s):
    policy = observation["policy"].clone()
    policy[:, 9:12] = command
    policy[:, 74:89] = previous_action
    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
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
    return torch.where(elapsed_s >= HOLD_START_S, 2, result)


def run() -> dict:
    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    observation, _ = wrapped.reset()
    term = env.action_manager._terms["joint_pos"]
    if tuple(observation["policy"].shape) != (args.num_envs, 93):
        raise RuntimeError("policy observation contract changed")
    if tuple(term._joint_names) != LOWER15 or not bool(term._cwi_upper_zero_mask.all()):
        raise RuntimeError("action/fixed-upper contract changed")
    source = build_source(observation, env.device)
    source_before = model_hash(source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    stationary = ort.InferenceSession(
        str(args.stationary), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if stationary.get_inputs()[0].name != "obs" or stationary.get_outputs()[0].name != "actions":
        raise RuntimeError("stationary ONNX contract changed")
    assignments = checkerboard_assignment(args.num_envs, args.seed_index)
    if not validate_assignment(assignments, args.seed_index):
        raise RuntimeError("treatment assignment is not balanced")
    candidate = torch.tensor(
        [name == "locomotion_zero" for name in assignments], device=env.device
    )
    robot = env.scene["robot"]
    foot_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    initial = {
        "policy": tensor_hash(observation["policy"]),
        "root": tensor_hash(torch.cat((robot.data.root_pos_w, robot.data.root_quat_w,
                                        robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)),
        "joint": tensor_hash(torch.cat((robot.data.joint_pos, robot.data.joint_vel), dim=-1)),
    }
    count = args.num_envs
    previous = torch.zeros(count, ACTION_DIM, device=env.device)
    alive = torch.ones(count, device=env.device, dtype=torch.bool)
    terminated = torch.zeros_like(alive)
    timed_out = torch.zeros_like(alive)
    termination_segment = torch.full((count,), -1, device=env.device, dtype=torch.int64)
    expected_counts = torch.zeros(count, 3, device=env.device, dtype=torch.int64)
    sample_lists = {name: [] for name in (
        "segment", "pitch", "velocity_sq", "lateral_sq", "yaw_sq", "support", "slip",
        "flight", "root_height", "tilt", "speed", "double_support", "action_slew",
        "normalized_clip",
    )}
    with torch.no_grad():
        for _step in range(args.steps):
            command_term = env.command_manager.get_term("base_velocity")
            elapsed = command_term._elapsed_s()
            scheduled_speed = command_term._scheduled_speed(elapsed)
            segment = segment_id(elapsed)
            for index in range(3):
                expected_counts[:, index] += segment == index
            gait_elapsed = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
            command = torch.zeros(count, 3, device=env.device)
            command[:, 0] = scheduled_speed
            main_action = source.act_inference(actor_input(
                observation, command, previous, gait_elapsed
            )).clamp(-1.0, 1.0)
            stand_raw = stationary_action(stationary, observation["policy"], previous, env.device)
            zero = torch.zeros_like(command)
            main_stand = source.act_inference(actor_input(
                observation, zero, previous, gait_elapsed
            )).clamp(-1.0, 1.0)
            direct_stand = (0.5 * main_stand + 0.5 * stand_raw).clamp(-1.0, 1.0)
            terminal = elapsed >= HOLD_START_S
            direct_terminal = terminal & ~candidate
            action = torch.where(direct_terminal.unsqueeze(-1), direct_stand, main_action)
            premove = elapsed < 1.0
            action = torch.where(premove.unsqueeze(-1), direct_stand, action)
            action_slew = (action - previous).abs().amax(dim=-1)
            command_term.vel_command_b[:] = command
            next_observation, _reward_ignored, done, extras = wrapped.step(action)
            done = done.reshape(-1).bool()
            time_out = extras.get("time_outs", torch.zeros_like(done)).reshape(-1).bool()

            post_pitch = signed_root_pitch_rad(robot)
            post_support, post_contact_count = actual_support_com_outside_distance(
                env, force_threshold_n=10.0
            )
            post_forces = torch.stack(tuple(
                env.scene[name].data.force_matrix_w[..., 2].abs().reshape(count, -1).amax(dim=-1)
                for name in ("left_foot_ground_contact", "right_foot_ground_contact")
            ), dim=-1)
            post_contact = post_forces > 10.0
            foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
            slip = (foot_speed * post_contact).sum(-1) / post_contact_count.clamp_min(1)
            post_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
            post_quat = robot.data.root_quat_w
            post_tilt = torch.acos(torch.clamp(
                1.0 - 2.0 * (post_quat[:, 1].square() + post_quat[:, 2].square()), -1.0, 1.0
            ))
            target_xy = torch.stack((scheduled_speed, torch.zeros_like(scheduled_speed)), dim=-1)
            velocity_sq = torch.square(robot.data.root_lin_vel_b[:, :2] - target_xy).sum(dim=-1)
            normalized_clip = (term.preclip_combined_actions.abs() > 1.0).any(dim=-1).to(torch.float32)
            valid = alive & (segment >= 0)

            def masked(value):
                return torch.where(valid, value, torch.full_like(value, torch.nan)).detach().cpu()

            sample_lists["segment"].append(segment.detach().cpu())
            sample_lists["pitch"].append(masked(post_pitch))
            sample_lists["velocity_sq"].append(masked(velocity_sq))
            sample_lists["lateral_sq"].append(masked(robot.data.root_lin_vel_b[:, 1].square()))
            sample_lists["yaw_sq"].append(masked(robot.data.root_ang_vel_b[:, 2].square()))
            sample_lists["support"].append(masked(post_support))
            sample_lists["slip"].append(masked(slip))
            sample_lists["flight"].append(masked((post_contact_count == 0).to(torch.float32)))
            sample_lists["root_height"].append(masked(robot.data.root_pos_w[:, 2]))
            sample_lists["tilt"].append(masked(post_tilt))
            sample_lists["speed"].append(masked(post_speed))
            sample_lists["double_support"].append(masked((post_contact_count == 2).to(torch.float32)))
            sample_lists["action_slew"].append(masked(action_slew))
            sample_lists["normalized_clip"].append(masked(normalized_clip))
            newly_done = alive & done
            termination_segment = torch.where(newly_done, segment, termination_segment)
            terminated |= newly_done & ~time_out
            timed_out |= newly_done & time_out
            alive &= ~done
            previous = action
            observation = next_observation

    stacked = {name: torch.stack(values, dim=0) for name, values in sample_lists.items()}
    rows: list[dict] = []
    for env_id in range(count):
        treatment = assignments[env_id]
        for index, segment_name in enumerate(SEGMENTS):
            mask = (stacked["segment"][:, env_id] == index) & torch.isfinite(stacked["pitch"][:, env_id])
            sample_count = int(mask.sum())
            expected = int(expected_counts[env_id, index])
            row = {
                "env_id": env_id, "treatment": treatment, "segment": segment_name,
                "sample_count": sample_count, "expected_sample_count": expected,
                "sample_fraction": sample_count / max(expected, 1),
                "terminated": bool(terminated[env_id] and index == max(0, int(termination_segment[env_id]))),
                "time_out": bool(timed_out[env_id] and index == max(0, int(termination_segment[env_id]))),
            }
            if sample_count:
                def selected(metric):
                    return stacked[metric][:, env_id][mask]
                pitch = selected("pitch")
                row.update({
                    "pitch_mean_rad": float(pitch.mean()),
                    "pitch_p05_rad": float(torch.quantile(pitch, 0.05)),
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
                    "action_slew_max": float(selected("action_slew").max()),
                    "normalized_clip_fraction": float(selected("normalized_clip").mean()),
                })
            else:
                row.update({name: None for name in (
                    "pitch_mean_rad", "pitch_p05_rad", "velocity_mse", "lateral_mse", "yaw_mse",
                    "support_mean_m", "slip_p95_mps", "flight_fraction", "root_height_min_m",
                    "tilt_max_rad", "speed_p95_mps", "double_support_fraction", "action_slew_max",
                    "normalized_clip_fraction",
                )})
            rows.append(row)

    summaries = {treatment: summarize_treatment(rows, treatment) for treatment in TREATMENTS}
    gates = seed_gates(summaries["direct_mix"], summaries["locomotion_zero"])
    technical = {
        "assignment_exact": validate_assignment(assignments, args.seed_index),
        "row_count_exact": len(rows) == args.num_envs * 3,
        "all_treatments_segments_present": all(
            sum(row["treatment"] == treatment and row["segment"] == segment for row in rows) == 128
            for treatment in TREATMENTS for segment in SEGMENTS
        ),
        "present_metrics_finite": all(
            value is None or not isinstance(value, float) or math.isfinite(value)
            for row in rows for value in row.values()
        ),
        "source_model_immutable": source_before == model_hash(source),
        "source_hash_matches_prereg": sha256(args.source) == PREREG["immutable_inputs"]["source_checkpoint"]["sha256"],
        "stationary_hash_matches_prereg": sha256(args.stationary) == PREREG["immutable_inputs"]["stationary_actor"]["sha256"],
        "optimizer_steps_zero": True,
        "backward_calls_zero": True,
        "checkpoint_writes_zero": True,
    }
    if not all(technical.values()):
        raise RuntimeError(f"technical checks failed: {technical}")
    report = {
        "schema": "x2_locomotion_zero_hold_confirmation_segment_v1",
        "prereg_sha256": PREREG_SHA,
        "seed_index": args.seed_index,
        "seed": args.seed,
        "runtime": {"num_envs": args.num_envs, "steps": args.steps,
                    "elapsed_s": time.monotonic() - started},
        "assignment": {"method": "checkerboard", "direct_count": assignments.count("direct_mix"),
                       "candidate_count": assignments.count("locomotion_zero"),
                       "sha256": hashlib.sha256("\n".join(assignments).encode()).hexdigest()},
        "initial_fingerprint": initial,
        "summaries": summaries,
        "gates": gates,
        "per_env": rows,
        "technical_checks": technical,
        "segment_status": "SEGMENT_VALID",
        "reward_inspected": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
    }
    atomic_json(args.report, report)
    print(json.dumps({"segment_status": "SEGMENT_VALID", "seed": args.seed,
                      "candidate_hold": summaries["locomotion_zero"]["hold"],
                      "direct_hold": summaries["direct_mix"]["hold"],
                      "gates": gates}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


try:
    run()
except BaseException as exc:
    fail_fast(exc, "run")
