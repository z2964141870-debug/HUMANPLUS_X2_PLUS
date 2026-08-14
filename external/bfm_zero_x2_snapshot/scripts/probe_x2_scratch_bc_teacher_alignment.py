#!/usr/bin/env python3
"""Probe corrected scratch BC actions on live Stage219 teacher states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402
from gear_sonic.envs.x2_velocity.gait import gait_phase_observation  # noqa: E402

from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.agents.load_utils import load_agent_from_checkpoint_dir  # noqa: E402
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
)
NUM_ENVS = 64
STEPS = 64
SEED = 770161
STAGE219 = Path(
    "/home/yu/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/"
    "2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_"
    "resume2550_to2650_v1/model_2600.pt"
)
STAGE219_SHA256 = "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb"
TEMPLATE = Path(
    "/home/yu/x2_teleop_final/x2_sonic/data/processed/"
    "x2_official_forward_gait_phase_template_15dof.npz"
)
TEMPLATE_SHA256 = "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d"
STUDENT = Path("artifacts/x2_scratch_closed_loop_bc_joint_v4_seed770191")
STUDENT_SHA256 = "8c16bf2c240162d1b7d571370011f7ea93312b235b035b6cc64945c0ebfedc02"
STUDENT_REPORT = Path("reports/x2_scratch_closed_loop_bc_joint_v4.json")
STUDENT_REPORT_SHA256 = "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc"
REPORT = Path("reports/x2_scratch_closed_loop_bc_joint_v4_alignment_probe.json")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def build_cfg():
    cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.seed = SEED
    cfg.sim.device = args.device
    cfg.scene.num_envs = NUM_ENVS
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.actions.joint_pos.scale = {
        name: float(scale)
        for name, scale in zip(X2_LOWER_JOINTS_15, X2_SCRATCH_ACTION_SCALE_15)
    }
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


def teacher_observation(env, previous_raw):
    data = env.scene["robot"].data
    gait = gait_phase_observation(
        env,
        command_name="base_velocity",
        cycle_time_s=0.8,
        double_support_fraction=0.30,
    )
    value = torch.cat(
        (
            data.root_lin_vel_b,
            data.root_ang_vel_b,
            data.projected_gravity_b,
            env.command_manager.get_command("base_velocity"),
            data.joint_pos - data.default_joint_pos,
            data.joint_vel,
            previous_raw,
            gait,
        ),
        dim=-1,
    )
    return {"policy": value, "critic": value}


def build_teacher(obs):
    teacher = ActorCritic(
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
    ).cuda()
    teacher.load_state_dict(
        torch.load(STAGE219, map_location="cuda", weights_only=False)["model_state_dict"],
        strict=True,
    )
    return teacher.eval()


def main() -> dict:
    if (
        file_hash(STAGE219) != STAGE219_SHA256
        or file_hash(TEMPLATE) != TEMPLATE_SHA256
        or file_hash(STUDENT_REPORT) != STUDENT_REPORT_SHA256
        or tree_hash(STUDENT) != STUDENT_SHA256
    ):
        raise RuntimeError("alignment probe immutable input guard failed")
    if REPORT.exists() or REPORT.with_name(f"{REPORT.name}.sha256").exists():
        raise FileExistsError("refusing to overwrite alignment probe")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(env, wrapped, history_length=4, to_numpy=False)
    observation, _ = adapter.reset(seed=SEED)
    student = load_agent_from_checkpoint_dir(STUDENT, device="cuda")
    canonical = torch.as_tensor(
        json.loads(STUDENT_REPORT.read_text())["canonical_latent"],
        device="cuda",
    ).reshape(1, 64).expand(NUM_ENVS, -1)
    previous_raw = torch.zeros(NUM_ENVS, ACTION_DIM, device="cuda")
    teacher = build_teacher(teacher_observation(env, previous_raw))
    template = np.load(TEMPLATE, allow_pickle=False)
    cycle = torch.as_tensor(template["q_cycle_zero_mean_rad"], device="cuda")
    stage_scale = torch.as_tensor(template["action_scale_rad"], device="cuda")
    scratch_scale = torch.as_tensor(X2_SCRATCH_ACTION_SCALE_15, device="cuda")
    period_s = float(template["period_s"])
    trace = []
    for step in range(STEPS):
        current = {key: value for key, value in observation.items() if key != "time"}
        with torch.inference_mode():
            student_action = student.act(current, canonical, mean=True).clamp(-1.0, 1.0)
            teacher_raw = teacher.act_inference(
                teacher_observation(env, previous_raw)
            ).clamp(-1.0, 1.0)
            phase = torch.remainder(
                env.episode_length_buf.float() * env.step_dt / period_s, 1.0
            )
            position = phase * cycle.shape[0] - 0.5
            lower_unwrapped = torch.floor(position)
            blend = position - lower_unwrapped
            lower = lower_unwrapped.long() % cycle.shape[0]
            upper = (lower + 1) % cycle.shape[0]
            q_bias = (
                (1.0 - blend).unsqueeze(-1) * cycle[lower]
                + blend.unsqueeze(-1) * cycle[upper]
            )
            combined = (teacher_raw + 0.15 * q_bias / stage_scale).clamp(-1.0, 1.0)
            teacher_action = (combined * stage_scale / scratch_scale).clamp(-1.0, 1.0)
        error = student_action - teacher_action
        trace.append(
            {
                "step": step,
                "mae": float(error.abs().mean()),
                "max_abs": float(error.abs().max()),
                "student_action_mean_by_joint": student_action.mean(0).tolist(),
                "teacher_action_mean_by_joint": teacher_action.mean(0).tolist(),
            }
        )
        observation, _, terminated, truncated, _ = adapter.step(teacher_action)
        done = terminated | truncated
        previous_raw.copy_(teacher_raw)
        previous_raw[done] = 0.0
    return {
        "schema": "x2_scratch_bc_live_teacher_alignment_probe_joint_v4",
        "steps": STEPS,
        "num_envs": NUM_ENVS,
        "teacher_executed": True,
        "student_executed": False,
        "student_weights_scratch_lineage": True,
        "stage219_weights_loaded_into_student": False,
        "action_mae_mean": float(np.mean([item["mae"] for item in trace])),
        "action_mae_step0": trace[0]["mae"],
        "action_mae_last": trace[-1]["mae"],
        "trace": trace,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "performance_claim": False,
    }


try:
    report = main()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = REPORT.with_name(f".{REPORT.name}.tmp")
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    temporary_sidecar = REPORT.with_name(f".{REPORT.name}.sha256.tmp")
    temporary.write_text(serialized)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    temporary_sidecar.write_text(f"{digest}  {REPORT.name}\n")
    os.replace(temporary, REPORT)
    os.replace(temporary_sidecar, sidecar)
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}), flush=True)
    os._exit(0)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
