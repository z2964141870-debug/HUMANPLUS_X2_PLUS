#!/usr/bin/env python3
"""Batched zero-training privileged-teacher reachability in validated IsaacLab."""

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
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=780042)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def prelaunch_guard() -> tuple[dict, str]:
    for output in (args.report, args.report.with_name(args.report.name + ".sha256"), args.failure, args.failure.with_name(args.failure.name + ".sha256")):
        if output.exists():
            raise FileExistsError(output)
    prereg_sidecar = args.prereg.with_name(args.prereg.name + ".sha256")
    prereg_sha = sha256(args.prereg)
    if prereg_sidecar.read_text(encoding="utf-8") != f"{prereg_sha}  {args.prereg.name}\n":
        raise RuntimeError("preregistration sidecar mismatch")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_privileged_teacher_isaac_reachability_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    if (args.num_envs, args.seed, args.steps) != (64, 780042, 200):
        raise RuntimeError("runtime shape/seed differs from preregistration")
    if args.source.resolve() != Path(prereg["immutable_inputs"]["source_checkpoint"]["path"]).resolve():
        raise RuntimeError("source path differs from preregistration")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"{section} mismatch: {name}")
    if Path(prereg["outputs"]["result"]).resolve() != args.report.resolve():
        raise RuntimeError("result path differs from preregistration")
    if Path(prereg["outputs"]["failure"]).resolve() != args.failure.resolve():
        raise RuntimeError("failure path differs from preregistration")
    return prereg, prereg_sha


PREREG, PREREG_SHA = prelaunch_guard()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.heading_command import gain_scheduled_velocity_cfg  # noqa: E402
from cwi_x2.privileged_teacher_isaac_reachability import (  # noqa: E402
    reachability_cost,
    strict_gates,
    summarize_arrays,
)
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
        response_heading_control_stiffness=1.0,
    )
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = 10.0
    _apply_x2_actuator_response(
        cfg.scene.robot,
        {
            "enabled": True, "profile": "session03_session04_group", "randomize": False,
            "strength": 1.0, "filter_strength": 1.0, "delay_strength": 1.0,
            "include_ideal_endpoint": False, "ideal_env_fraction": 1.0,
            "filter_only_env_fraction": 0.0,
        },
        physics_dt_sec=cfg.sim.dt,
    )
    return cfg


def build_source(obs, device):
    payload = torch.load(args.source, map_location=device, weights_only=False)
    if int(payload.get("iter", -1)) != 2600:
        raise RuntimeError("source checkpoint iteration changed")
    model = ActorCritic(
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


def reset_observation(wrapped, env):
    observation, _ = wrapped.reset()
    policy = observation["policy"]
    if tuple(policy.shape) != (64, 93):
        raise RuntimeError("policy observation is not [64,93]")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15 or float(env.cfg.actions.joint_pos.template_scale) != 0.15:
        raise RuntimeError("Stage219 action contract drifted")
    if not bool(term._cwi_upper_zero_mask.all()):
        raise RuntimeError("teacher reachability requires fixed upper body")
    actuator = env.scene["robot"].actuators["legs"]
    alpha = actuator._position_alpha.reshape(64, -1)[:, 0]
    lag = actuator.positions_delay_buffer.time_lags.reshape(64)
    if not bool((torch.isclose(alpha, torch.ones_like(alpha)) & (lag == 0)).all()):
        raise RuntimeError("teacher reachability requires ideal actuator")
    return observation


def run_population(wrapped, env, source, parameters: torch.Tensor) -> tuple[dict[str, np.ndarray], dict]:
    obs = reset_observation(wrapped, env)
    initial = obs["policy"].detach().clone()
    initial_spread = float(torch.max(torch.abs(initial - initial[-1:])))
    robot = env.scene["robot"]
    initial_root = torch.cat(
        (
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
        ),
        dim=-1,
    ).detach().clone()
    foot_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    names = (
        "pitch", "velocity_sq", "lateral_sq", "yaw_sq", "support", "slip",
        "root_height", "tilt", "residual", "done", "contact_count",
    )
    storage = {name: [] for name in names}
    with torch.inference_mode():
        for _ in range(args.steps):
            policy_obs = obs["policy"]
            source_action = source.act_inference(obs).clamp(-1.0, 1.0)
            gait = policy_obs[:, 89:93]
            features = torch.stack(
                (
                    torch.ones(64, device=env.device), gait[:, 0], gait[:, 1],
                    gait[:, 2] - 0.5, gait[:, 3] - 0.5,
                ), dim=-1,
            )
            signal = torch.einsum("naf,nf->na", parameters, features) / math.sqrt(5.0)
            requested = float(PREREG["experiment"]["teacher_residual_abs_max"]) * torch.tanh(signal)
            action = torch.clamp(source_action + requested, -1.0, 1.0)
            realized = action - source_action
            next_obs, _reward, done, _extras = wrapped.step(action)
            command = env.command_manager.get_command("base_velocity")
            pitch = signed_root_pitch_rad(robot)
            support, contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
            forces = torch.stack(
                tuple(
                    env.scene[name].data.force_matrix_w[..., 2].abs().reshape(64, -1).amax(dim=-1)
                    for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                ), dim=-1,
            )
            contact = forces > 10.0
            foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
            slip = (foot_speed * contact).sum(-1) / contact_count.clamp_min(1)
            quat = robot.data.root_quat_w
            tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0))
            values = {
                "pitch": pitch,
                "velocity_sq": torch.square(robot.data.root_lin_vel_b[:, :2] - command[:, :2]).sum(-1),
                "lateral_sq": torch.square(robot.data.root_lin_vel_b[:, 1]),
                "yaw_sq": torch.square(robot.data.root_ang_vel_b[:, 2] - command[:, 2]),
                "support": support,
                "slip": slip,
                "root_height": robot.data.root_pos_w[:, 2],
                "tilt": tilt,
                "residual": realized,
                "done": done.reshape(-1).bool(),
                "contact_count": contact_count,
            }
            for name, value in values.items():
                storage[name].append(value.detach().cpu())
            obs = next_obs
    arrays = {name: torch.stack(values, dim=0).numpy() for name, values in storage.items()}
    return arrays, {
        "initial_policy_observation_sha256": tensor_hash(initial),
        "initial_env_spread_max_abs": initial_spread,
        "initial_root_state_sha256": tensor_hash(initial_root),
    }


def concatenate_records(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.concatenate((left[name], right[name]), axis=1) for name in left}


def run() -> dict:
    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    initial_obs = wrapped.get_observations()
    source = build_source(initial_obs, env.device)
    source_state_sha = hashlib.sha256(
        b"".join(
            name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            for name, value in sorted(source.state_dict().items())
        )
    ).hexdigest()

    search = PREREG["search"]
    rng = np.random.default_rng(int(search["seed"]))
    shape = (15, 5)
    mean = np.zeros(shape, dtype=np.float64)
    std = np.full(shape, float(search["initial_std"]), dtype=np.float64)
    lower, upper = map(float, search["coefficient_bounds"])
    best = {"cost": float("inf"), "parameters": np.zeros(shape), "summary": None}
    iterations = []
    reset_fingerprints = []
    source_summaries = []
    for iteration in range(int(search["iterations"])):
        population = np.clip(
            rng.normal(mean, std, size=(63,) + shape), lower, upper
        )
        population[0] = mean
        if iteration == 0:
            population[0] = 0.0
        full = np.concatenate((population, np.zeros((1,) + shape)), axis=0)
        records, fingerprint = run_population(
            wrapped, env, source,
            torch.as_tensor(full, device=env.device, dtype=torch.float32),
        )
        reset_fingerprints.append(fingerprint)
        source_summary = summarize_arrays(records, np.arange(64) == 63)
        source_summaries.append(source_summary)
        scored = []
        for env_id in range(63):
            summary = summarize_arrays(records, np.arange(64) == env_id)
            cost = reachability_cost(summary)
            scored.append((cost, population[env_id].copy(), summary, env_id))
            if cost < best["cost"]:
                best = {"cost": float(cost), "parameters": population[env_id].copy(), "summary": summary}
        scored.sort(key=lambda item: item[0])
        elite = np.stack([item[1] for item in scored[: int(search["elites"])]])
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(float(search["std_floor"]), 0.25 * std + 0.75 * elite.std(axis=0))
        iterations.append(
            {
                "iteration": iteration,
                "best_cost": float(scored[0][0]),
                "best_env_id": int(scored[0][3]),
                "best_survived": bool(scored[0][2]["survived_full_horizon"]),
                "source": source_summary,
            }
        )

    # Two deterministic reset passes swap source/candidate on every env index.
    ids = np.arange(64)
    validation_records = []
    validation_fingerprints = []
    lane_summaries = {}
    for lane in ("A", "B"):
        candidate_mask = (ids % 2 == 1) if lane == "A" else (ids % 2 == 0)
        parameter_batch = np.zeros((64,) + shape, dtype=np.float32)
        parameter_batch[candidate_mask] = np.asarray(best["parameters"], dtype=np.float32)
        records, fingerprint = run_population(
            wrapped, env, source,
            torch.as_tensor(parameter_batch, device=env.device),
        )
        validation_records.append(records)
        validation_fingerprints.append(fingerprint)
        lane_summaries[lane] = {
            "source": summarize_arrays(records, ~candidate_mask),
            "candidate": summarize_arrays(records, candidate_mask),
        }
    combined = concatenate_records(*validation_records)
    source_mask = np.concatenate((ids % 2 == 0, ids % 2 == 1))
    candidate_mask = ~source_mask
    source_summary = summarize_arrays(combined, source_mask)
    candidate_summary = summarize_arrays(combined, candidate_mask)
    gates = strict_gates(source_summary, candidate_summary, PREREG["gates"])
    lane_gates = {
        lane: strict_gates(values["source"], values["candidate"], PREREG["gates"])
        for lane, values in lane_summaries.items()
    }
    technical = {
        "initial_env_spread": all(item["initial_env_spread_max_abs"] <= 1.0e-6 for item in reset_fingerprints + validation_fingerprints),
        "reset_observation_replay": len({item["initial_policy_observation_sha256"] for item in reset_fingerprints + validation_fingerprints}) == 1,
        "reset_root_replay": len({item["initial_root_state_sha256"] for item in reset_fingerprints + validation_fingerprints}) == 1,
        "source_survival_every_search_iteration": all(item["survived_full_horizon"] for item in source_summaries),
        "source_model_immutable": source_state_sha == hashlib.sha256(
            b"".join(
                name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
                for name, value in sorted(source.state_dict().items())
            )
        ).hexdigest(),
        "finite": all(math.isfinite(float(item["best_cost"])) for item in iterations),
    }
    lane_pitch = {
        lane: {
            "mean_delta_rad": values["candidate"]["signed_pitch_rad"]["mean"] - values["source"]["signed_pitch_rad"]["mean"],
            "p05_delta_rad": values["candidate"]["signed_pitch_rad"]["p05"] - values["source"]["signed_pitch_rad"]["p05"],
        }
        for lane, values in lane_summaries.items()
    }
    lane_consistency = all(
        values["mean_delta_rad"] >= PREREG["gates"]["lane_pitch_mean_delta_rad_min"]
        and values["p05_delta_rad"] >= PREREG["gates"]["lane_pitch_p05_delta_rad_min"]
        and all(
            passed
            for name, passed in lane_gates[lane].items()
            if name not in {"pitch_mean", "pitch_p05"}
        )
        for lane, values in lane_pitch.items()
    )
    passed = bool(all(technical.values()) and all(gates.values()) and lane_consistency)
    return {
        "schema": "x2_privileged_teacher_isaac_reachability_result_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "PASS_TEACHER_REACHABILITY_LOCAL_ONLY" if passed else "FAIL_NO_TEACHER_REACHABILITY_STOP",
        "technical_checks": technical,
        "search": {
            "config": search,
            "iterations": iterations,
            "best_cost": best["cost"],
            "best_parameters_15x5": best["parameters"].tolist(),
        },
        "validation": {
            "lane_summaries": lane_summaries,
            "lane_gates": lane_gates,
            "lane_pitch": lane_pitch,
            "lane_consistency": lane_consistency,
            "source": source_summary,
            "candidate": candidate_summary,
            "deltas": {
                "signed_pitch_mean_rad": candidate_summary["signed_pitch_rad"]["mean"] - source_summary["signed_pitch_rad"]["mean"],
                "signed_pitch_p05_rad": candidate_summary["signed_pitch_rad"]["p05"] - source_summary["signed_pitch_rad"]["p05"],
                "velocity_rmse_mps": candidate_summary["velocity_rmse_mps"] - source_summary["velocity_rmse_mps"],
                "lateral_rms_mps": candidate_summary["lateral_rms_mps"] - source_summary["lateral_rms_mps"],
                "yaw_rmse_radps": candidate_summary["yaw_rmse_radps"] - source_summary["yaw_rmse_radps"],
                "support_outside_mean_m": candidate_summary["support_outside_mean_m"] - source_summary["support_outside_mean_m"],
                "stance_slip_p95_mps": candidate_summary["stance_slip_p95_mps"] - source_summary["stance_slip_p95_mps"],
            },
            "gates": gates,
        },
        "evidence_boundary": {
            "privileged_oracle_not_deployable": True,
            "torch_optimizer_objects": 0,
            "policy_optimizer_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "cem_distribution_updates": int(search["iterations"]),
            "teacher_panel_preregistration_unlocked": passed,
            "bc_or_dagger_unlocked": False,
            "ppo_or_long_training_unlocked": False,
            "deployment_unlocked": False,
        },
        "resource": {
            "wall_time_s": time.monotonic() - started,
            "isaac_launches": 1,
            "control_steps": (int(search["iterations"]) + 2) * args.steps,
            "physics_substeps_per_environment": (
                (int(search["iterations"]) + 2) * args.steps * int(env.cfg.decimation)
            ),
        },
    }


def fail(error: BaseException) -> None:
    payload = {
        "schema": "x2_privileged_teacher_isaac_reachability_failure_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "FAIL_IMPLEMENTATION_STOP",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
    }
    try:
        atomic_json(args.failure, payload)
        sidecar(args.failure)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(1)


try:
    result = run()
    atomic_json(args.report, result)
    sidecar(args.report)
    print(json.dumps({
        "decision": result["decision"],
        "technical_checks": result["technical_checks"],
        "deltas": result["validation"]["deltas"],
        "gates": result["validation"]["gates"],
        "wall_time_s": result["resource"]["wall_time_s"],
    }, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
except BaseException as error:
    fail(error)
