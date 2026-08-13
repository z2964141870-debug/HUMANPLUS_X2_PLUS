#!/usr/bin/env python3
"""Evaluate the frozen safe teacher dose on the eight-role command panel."""

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
    if path.exists():
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
    temporary.replace(path)


def sidecar(path: Path) -> None:
    target = path.with_name(path.name + ".sha256")
    if target.exists():
        raise FileExistsError(target)
    target.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    with target.open("r+b") as stream:
        os.fsync(stream.fileno())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--failure", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--lane", choices=("A", "B"), required=True)
parser.add_argument("--steps", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def prelaunch_guard() -> tuple[dict, str, str]:
    for output in (
        args.report, args.report.with_name(args.report.name + ".sha256"),
        args.failure, args.failure.with_name(args.failure.name + ".sha256"),
    ):
        if output.exists():
            raise FileExistsError(output)
    prereg_sidecar = args.prereg.with_name(args.prereg.name + ".sha256")
    prereg_sha = sha256(args.prereg)
    if prereg_sidecar.read_text(encoding="utf-8") != f"{prereg_sha}  {args.prereg.name}\n":
        raise RuntimeError("preregistration sidecar mismatch")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_privileged_teacher_command_panel_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    if args.num_envs != 128 or args.steps != 512 or args.seed not in prereg["panel"]["seeds"]:
        raise RuntimeError("runtime shape/seed differs from preregistration")
    if args.source.resolve() != Path(prereg["immutable_inputs"]["source_checkpoint"]["path"]).resolve():
        raise RuntimeError("source path differs from preregistration")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"{section} mismatch: {name}")
    launch_id = f"s{args.seed}_{args.lane}"
    expected = prereg["outputs"]["launches"][launch_id]
    if Path(expected["report"]).resolve() != args.report.resolve():
        raise RuntimeError("report path differs from preregistration")
    if Path(expected["failure"]).resolve() != args.failure.resolve():
        raise RuntimeError("failure path differs from preregistration")
    return prereg, prereg_sha, launch_id


PREREG, PREREG_SHA, LAUNCH_ID = prelaunch_guard()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from cwi_x2.privileged_teacher_command_panel import ROLE_NAMES  # noqa: E402
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


class TeacherPanelVelocityCommand(TransitionVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        self._panel_pair_id = torch.div(ids, 2, rounding_mode="floor")
        self._panel_role = self._panel_pair_id.remainder(8)
        self._panel_turn_rate = torch.where(
            torch.div(self._panel_pair_id, 8, rounding_mode="floor").remainder(2).bool(),
            torch.full((self.num_envs,), 0.15, device=self.device),
            torch.full((self.num_envs,), -0.15, device=self.device),
        )
        self._freeze(torch.arange(self.num_envs, device=self.device))

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if hasattr(self, "_panel_role"):
            self._freeze(self._resolved_env_ids(env_ids))

    def _freeze(self, env_ids):
        role = self._panel_role[env_ids]
        speed = torch.full_like(role, 0.35, dtype=torch.float32)
        speed = torch.where(role <= 1, torch.full_like(speed, 0.20), speed)
        speed = torch.where((role >= 4) & (role <= 5), torch.full_like(speed, 0.50), speed)
        self.cruise_speed[env_ids] = speed
        self.vel_command_b[env_ids, 0] = speed
        self.vel_command_b[env_ids, 1:] = 0.0

    def _update_command(self):
        super()._update_command()
        if not hasattr(self, "_panel_role"):
            return
        role = self._panel_role
        speed = self.cruise_speed.clone()
        transition = role == 7
        speed[transition] = self._scheduled_speed(self._elapsed_s())[transition]
        self.vel_command_b[:, 0] = speed
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = torch.where(
            (role == 6) & (speed.abs() > 0.05), self._panel_turn_rate,
            torch.zeros_like(self._panel_turn_rate),
        )

    def command_role(self):
        return self._panel_role


def build_cfg():
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = PREREG["immutable_inputs"]["template"]["path"]
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()}
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    magnitude = 0.006
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "roll": (-magnitude, magnitude),
        "pitch": (-magnitude, magnitude), "yaw": (-0.02, 0.02),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0),
        "roll": (-0.01, 0.01), "pitch": (-0.01, 0.01), "yaw": (-0.01, 0.01),
    }
    cfg.events.reset_robot_joints.params["position_range"] = (0.9975, 1.0025)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-0.01, 0.01)
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
        stand_s=0.0, accelerate_s=1.0, cruise_s=4.2, decelerate_s=2.0,
        maximum_phase_offset_s=0.0,
    )
    transition.class_type = TeacherPanelVelocityCommand
    cfg.commands.base_velocity = transition
    cfg.episode_length_s = 12.0
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
        num_actions=15, actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128],
        activation="elu", init_noise_std=0.4, noise_std_type="scalar",
        actor_obs_normalization=False, critic_obs_normalization=False,
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


def finite_mean(value: torch.Tensor) -> float:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        raise RuntimeError("empty metric")
    return float(finite.mean())


def finite_quantile(value: torch.Tensor, q: float) -> float:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        raise RuntimeError("empty metric")
    return float(torch.quantile(finite, q))


def run() -> dict:
    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    obs = wrapped.get_observations()
    if tuple(obs["policy"].shape) != (128, 93):
        raise RuntimeError("policy observation contract changed")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15 or not bool(term._cwi_upper_zero_mask.all()):
        raise RuntimeError("action/fixed-upper contract changed")
    source = build_source(obs, env.device)
    source_sha = tensor_hash(torch.cat([value.reshape(-1) for value in source.state_dict().values()]))
    direction_result = json.loads(Path(PREREG["immutable_evidence"]["v2b_result"]["path"]).read_text())
    parameters = torch.as_tensor(direction_result["search"]["best_parameters_15x5"], device=env.device, dtype=torch.float32)
    scale = float(PREREG["panel"]["teacher_scale"])
    even = torch.arange(128, device=env.device).remainder(2) == 0
    source_mask = even if args.lane == "A" else ~even
    candidate_mask = ~source_mask
    initial_policy = obs["policy"].detach().clone()
    robot = env.scene["robot"]
    initial_root = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1).detach().clone()
    initial_joint = torch.cat((robot.data.joint_pos, robot.data.joint_vel), dim=-1).detach().clone()
    foot_ids = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
    alive = torch.ones(128, dtype=torch.bool, device=env.device)
    survival = torch.full((128,), args.steps * env.step_dt, device=env.device)
    terminated = torch.zeros_like(alive)
    timed_out = torch.zeros_like(alive)
    root_height_min = robot.data.root_pos_w[:, 2].clone()
    tilt_max = torch.zeros(128, device=env.device)
    residual_abs_max = torch.zeros(128, device=env.device)
    residual_step_abs_max = torch.zeros(128, device=env.device)
    previous_residual = torch.zeros((128, 15), device=env.device)
    samples = {name: [] for name in (
        "pitch", "support", "slip", "velocity_sq", "lateral_sq", "yaw_sq",
        "flight", "terminal_speed", "terminal_double_support",
    )}
    with torch.no_grad():
        for step in range(args.steps):
            source_action = source.act_inference(obs).clamp(-1.0, 1.0)
            policy_obs = obs["policy"]
            gait = policy_obs[:, 89:93]
            features = torch.stack((torch.ones(128, device=env.device), gait[:, 0], gait[:, 1], gait[:, 2] - 0.5, gait[:, 3] - 0.5), dim=-1)
            signal = torch.einsum("af,nf->na", parameters, features) / math.sqrt(5.0)
            command = env.command_manager.get_command("base_velocity")
            moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
            requested = scale * 0.20 * torch.tanh(signal) * moving.unsqueeze(-1)
            candidate_action = torch.clamp(source_action + requested, -1.0, 1.0)
            realized = candidate_action - source_action
            action = torch.where(candidate_mask.unsqueeze(-1), candidate_action, source_action)
            next_obs, _reward, done, extras = wrapped.step(action)
            done = done.reshape(-1).bool()
            time_out = extras.get("time_outs", torch.zeros_like(done)).reshape(-1).bool()
            pitch = signed_root_pitch_rad(robot)
            support, contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
            forces = torch.stack(tuple(
                env.scene[name].data.force_matrix_w[..., 2].abs().reshape(128, -1).amax(dim=-1)
                for name in ("left_foot_ground_contact", "right_foot_ground_contact")
            ), dim=-1)
            contact = forces > 10.0
            foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
            slip = (foot_speed * contact).sum(-1) / contact_count.clamp_min(1)
            velocity_sq = torch.square(robot.data.root_lin_vel_b[:, :2] - command[:, :2]).sum(-1)
            lateral_sq = torch.square(robot.data.root_lin_vel_b[:, 1])
            yaw_sq = torch.square(robot.data.root_ang_vel_b[:, 2] - command[:, 2])
            terminal = env.command_manager.get_term("base_velocity").terminal_stop_mask()
            quat = robot.data.root_quat_w
            tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0))
            valid_moving = alive & moving

            def masked(value, mask):
                return torch.where(mask, value, torch.full_like(value, torch.nan)).detach().cpu()

            samples["pitch"].append(masked(pitch, valid_moving))
            samples["support"].append(masked(support, valid_moving))
            samples["slip"].append(masked(slip, alive))
            samples["velocity_sq"].append(masked(velocity_sq, alive))
            samples["lateral_sq"].append(masked(lateral_sq, alive))
            samples["yaw_sq"].append(masked(yaw_sq, alive))
            samples["flight"].append(masked((contact_count == 0).float(), alive))
            samples["terminal_speed"].append(masked(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1), alive & terminal))
            samples["terminal_double_support"].append(masked((contact_count == 2).float(), alive & terminal))
            active_residual = torch.where(candidate_mask.unsqueeze(-1), realized, torch.zeros_like(realized))
            residual_abs_max = torch.maximum(residual_abs_max, active_residual.abs().amax(dim=-1))
            residual_step_abs_max = torch.maximum(residual_step_abs_max, (active_residual - previous_residual).abs().amax(dim=-1))
            previous_residual = active_residual
            root_height_min = torch.where(alive, torch.minimum(root_height_min, robot.data.root_pos_w[:, 2]), root_height_min)
            tilt_max = torch.where(alive, torch.maximum(tilt_max, tilt), tilt_max)
            newly_done = alive & done
            survival[newly_done] = (step + 1) * env.step_dt
            terminated |= newly_done & ~time_out
            timed_out |= newly_done & time_out
            alive &= ~done
            obs = next_obs

    stacked = {name: torch.stack(values, dim=0) for name, values in samples.items()}
    roles = env.command_manager.get_term("base_velocity").command_role().cpu()
    per_env = []
    for env_id in range(128):
        role_id = int(roles[env_id])
        treatment = "source" if bool(source_mask[env_id]) else "candidate"
        per_env.append({
            "env_id": env_id, "pair_id": env_id // 2, "lane": args.lane,
            "treatment": treatment, "role_id": role_id, "role": ROLE_NAMES[role_id],
            "signed_pitch_mean_rad": finite_mean(stacked["pitch"][:, env_id]),
            "signed_pitch_p05_rad": finite_quantile(stacked["pitch"][:, env_id], 0.05),
            "support_mean_m": finite_mean(stacked["support"][:, env_id]),
            "stance_slip_p95_mps": finite_quantile(stacked["slip"][:, env_id], 0.95),
            "velocity_mse": finite_mean(stacked["velocity_sq"][:, env_id]),
            "lateral_mse": finite_mean(stacked["lateral_sq"][:, env_id]),
            "yaw_mse": finite_mean(stacked["yaw_sq"][:, env_id]),
            "flight_fraction": finite_mean(stacked["flight"][:, env_id]),
            "terminal_speed_mean_mps": finite_mean(stacked["terminal_speed"][:, env_id]) if role_id == 7 else 0.0,
            "terminal_double_support_mean": finite_mean(stacked["terminal_double_support"][:, env_id]) if role_id == 7 else 1.0,
            "moving_sample_count": int(torch.isfinite(stacked["pitch"][:, env_id]).sum()),
            "terminal_sample_count": int(torch.isfinite(stacked["terminal_speed"][:, env_id]).sum()),
            "survival_s": float(survival[env_id]), "terminated": bool(terminated[env_id]),
            "time_out": bool(timed_out[env_id]), "root_height_min_m": float(root_height_min[env_id]),
            "root_tilt_max_rad": float(tilt_max[env_id]),
            "teacher_residual_abs_max": float(residual_abs_max[env_id]),
            "teacher_residual_step_abs_max": float(residual_step_abs_max[env_id]),
        })
    source_after = tensor_hash(torch.cat([value.reshape(-1) for value in source.state_dict().values()]))
    technical = {
        "finite_per_env": all(
            all(not isinstance(value, float) or math.isfinite(value) for value in row.values())
            for row in per_env
        ),
        "role_balance": torch.bincount(roles, minlength=8).tolist() == [16] * 8,
        "source_model_immutable": source_sha == source_after,
        "optimizer_steps_zero": True,
        "checkpoint_writes_zero": True,
    }
    return {
        "schema": "x2_privileged_teacher_panel_launch_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "PANEL_LAUNCH_FINITE" if all(technical.values()) else "FAIL_INVALID_STOP",
        "launch_id": LAUNCH_ID, "seed": args.seed, "lane": args.lane,
        "initial": {
            "policy_observation_sha256": tensor_hash(initial_policy),
            "root_state_sha256": tensor_hash(initial_root),
            "joint_state_sha256": tensor_hash(initial_joint),
        },
        "technical_checks": technical, "per_env": per_env,
        "evidence_boundary": {
            "teacher_scale": scale, "optimizer_steps": 0, "backward_calls": 0,
            "checkpoint_writes": 0, "reward_inspected": False,
            "bc_or_dagger_unlocked": False, "training_unlocked": False,
            "deployment_unlocked": False,
        },
        "resource": {"wall_time_s": time.monotonic() - started, "control_steps": args.steps},
    }


def fail(error: BaseException) -> None:
    payload = {
        "schema": "x2_privileged_teacher_panel_failure_v1",
        "preregistration_sha256": PREREG_SHA, "launch_id": LAUNCH_ID,
        "decision": "FAIL_IMPLEMENTATION_STOP", "error_type": type(error).__name__,
        "error": str(error), "traceback": traceback.format_exc(),
        "optimizer_steps": 0, "checkpoint_writes": 0,
    }
    try:
        atomic_json(args.failure, payload)
        sidecar(args.failure)
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(1)


try:
    result = run()
    atomic_json(args.report, result)
    sidecar(args.report)
    print(json.dumps({"decision": result["decision"], "launch_id": LAUNCH_ID, "technical": result["technical_checks"]}, indent=2))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if result["decision"] == "PANEL_LAUNCH_FINITE" else 1)
except BaseException as error:
    fail(error)
