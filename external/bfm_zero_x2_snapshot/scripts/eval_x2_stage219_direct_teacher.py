#!/usr/bin/env python3
"""Audit reconstructed Stage219 combined actions in the direct BFM plant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--response-domain", action="store_true")
parser.add_argument(
    "--command-panel",
    action="store_true",
    help="screen fixed response-domain speed and turn roles without training",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)
from gear_sonic.envs.x2_velocity.gait import gait_phase_observation  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)

from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
)


NUM_ENVS = 512
STEPS = 400
SEED = 770141
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
REPORT = Path(
    "reports/x2_stage219_direct_teacher_response_command_panel.json"
    if args.command_panel
    else "reports/x2_stage219_direct_teacher_response_domain_audit.json"
    if args.response_domain
    else "reports/x2_stage219_direct_teacher_source_domain_audit.json"
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
    # This audit supplies the already-combined Stage219 residual + 0.15 gait
    # target.  The plant must therefore be genuinely direct/template-free;
    # leaving the native action term's current 0.20 default enabled would add
    # the gait template a second time.
    cfg.actions.joint_pos.template_scale = 0.0
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
            "ideal_env_fraction": 0.0 if (args.response_domain or args.command_panel) else 1.0,
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
    if value.shape != (NUM_ENVS, 93) or not torch.isfinite(value).all():
        raise RuntimeError("teacher observation reconstruction failed")
    return {"policy": value, "critic": value}


def main() -> dict:
    if file_hash(STAGE219) != STAGE219_SHA256 or file_hash(TEMPLATE) != TEMPLATE_SHA256:
        raise RuntimeError("teacher audit immutable input guard failed")
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    if REPORT.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite teacher audit")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(env, wrapped, history_length=4, to_numpy=False)
    adapter.reset(seed=SEED)
    role_specs = (
        (
            ("vx_0p20", 0, 128, (0.20, 0.0, 0.0)),
            ("vx_0p35", 128, 256, (0.35, 0.0, 0.0)),
            ("vx_0p50", 256, 384, (0.50, 0.0, 0.0)),
            ("turn_left", 384, 448, (0.35, 0.0, 0.30)),
            ("turn_right", 448, 512, (0.35, 0.0, -0.30)),
        )
        if args.command_panel
        else (("vx_0p35", 0, NUM_ENVS, (0.35, 0.0, 0.0)),)
    )
    command_panel = torch.empty(NUM_ENVS, 3, device="cuda")
    for _name, start, stop, command in role_specs:
        command_panel[start:stop] = torch.tensor(command, device="cuda")
    env.command_manager.get_command("base_velocity").copy_(command_panel)
    previous_raw = torch.zeros(NUM_ENVS, ACTION_DIM, device="cuda")
    obs = teacher_observation(env, previous_raw)
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
    payload = torch.load(STAGE219, map_location="cuda", weights_only=False)
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    teacher.eval()
    template = np.load(TEMPLATE, allow_pickle=False)
    template_q = torch.as_tensor(template["q_cycle_zero_mean_rad"], device="cuda")
    action_scale = torch.as_tensor(template["action_scale_rad"], device="cuda")
    period_s = float(template["period_s"])

    ever_done = torch.zeros(NUM_ENVS, dtype=torch.bool, device="cuda")
    first_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device="cuda")
    first_velocity_sum = torch.zeros(NUM_ENVS, device="cuda")
    first_yaw_rate_sum = torch.zeros(NUM_ENVS, device="cuda")
    first_velocity_count = torch.zeros(NUM_ENVS, device="cuda")
    terminated_count = 0
    clip_count = 0
    for step in range(1, STEPS + 1):
        # Keep the panel immutable even if a lane resets unexpectedly.  The
        # rollout is shorter than the native 10 s command-resampling period.
        env.command_manager.get_command("base_velocity").copy_(command_panel)
        obs = teacher_observation(env, previous_raw)
        with torch.inference_mode():
            raw = teacher.act_inference(obs).clamp(-1.0, 1.0)
            phase = torch.remainder(
                env.episode_length_buf.float() * env.step_dt / period_s, 1.0
            )
            phase_position = phase * template_q.shape[0] - 0.5
            lower_unwrapped = torch.floor(phase_position)
            blend = phase_position - lower_unwrapped
            lower = lower_unwrapped.long() % template_q.shape[0]
            upper = (lower + 1) % template_q.shape[0]
            q_bias = (
                (1.0 - blend).unsqueeze(-1) * template_q[lower]
                + blend.unsqueeze(-1) * template_q[upper]
            )
            preclip = raw + 0.15 * q_bias / action_scale
            stage219_combined = preclip.clamp(-1.0, 1.0)
            scratch_scale = torch.as_tensor(
                X2_SCRATCH_ACTION_SCALE_15, device="cuda"
            )
            action = (
                stage219_combined * action_scale / scratch_scale
            ).clamp(-1.0, 1.0)
        alive = ~ever_done
        first_velocity_sum[alive] += env.scene["robot"].data.root_lin_vel_b[alive, 0]
        first_yaw_rate_sum[alive] += env.scene["robot"].data.root_ang_vel_b[alive, 2]
        first_velocity_count[alive] += 1
        clip_count += int((preclip.abs() >= 1.0).sum())
        _, _, terminated, truncated, _ = adapter.step(action)
        done = terminated | truncated
        new_done = done & ~ever_done
        first_steps[new_done] = step
        ever_done |= done
        terminated_count += int(terminated.sum())
        previous_raw.copy_(raw)
        previous_raw[done] = 0.0
    first_steps[~ever_done] = STEPS
    survival = ~ever_done
    groups = {}
    for name, start, stop, command in role_specs:
        lane_count = stop - start
        vx = first_velocity_sum[start:stop] / first_velocity_count[start:stop].clamp_min(1)
        yaw_rate = first_yaw_rate_sum[start:stop] / first_velocity_count[start:stop].clamp_min(1)
        role_survival = survival[start:stop]
        velocity_error = torch.abs(vx - command[0])
        yaw_error = torch.abs(yaw_rate - command[2])
        groups[name] = {
            "lanes": lane_count,
            "command": list(command),
            "survived_lanes": int(role_survival.sum()),
            "survival_steps_mean": float(first_steps[start:stop].float().mean()),
            "forward_velocity_mean_mps": float(vx.mean()),
            "yaw_rate_mean_radps": float(yaw_rate.mean()),
            "velocity_error_mean_mps": float(velocity_error.mean()),
            "yaw_rate_error_mean_radps": float(yaw_error.mean()),
            "tracking_gate": bool(
                role_survival.all()
                and velocity_error.mean() <= 0.15
                and yaw_error.mean() <= 0.20
            ),
        }
    valid = (
        bool(survival.all())
        and terminated_count == 0
        and all(group["tracking_gate"] for group in groups.values())
    )
    return {
        "schema": (
            "x2_stage219_response_command_panel_teacher_audit_v1"
            if args.command_panel
            else "x2_stage219_response_domain_direct_teacher_audit_v1"
            if args.response_domain
            else "x2_stage219_source_domain_direct_teacher_audit_v1"
        ),
        "decision": (
            "PASS_RESPONSE_COMMAND_TEACHER_PANEL"
            if valid and args.command_panel
            else "PASS_DIRECT_TEACHER_EQUIVALENCE"
            if valid
            else "FAIL_RESPONSE_COMMAND_TEACHER_PANEL"
            if args.command_panel
            else "FAIL_DIRECT_TEACHER_EQUIVALENCE"
        ),
        "teacher_checkpoint_sha256": STAGE219_SHA256,
        "teacher_iteration": int(payload["iter"]),
        "template_sha256": TEMPLATE_SHA256,
        "template_scale": 0.15,
        "internal_template_scale": 0.0,
        "physical_action_scale_conversion": True,
        "collision_profile": "sole12",
        "self_collisions": False,
        "actuator_domain": "response_only" if (args.response_domain or args.command_panel) else "ideal_only",
        "command_panel": bool(args.command_panel),
        "groups": groups,
        "num_envs": NUM_ENVS,
        "control_steps": STEPS,
        "survived_lanes": int(survival.sum()),
        "terminated_events": terminated_count,
        "first_episode_survival_steps_mean": float(first_steps.float().mean()),
        "first_episode_survival_steps_min": int(first_steps.min()),
        "first_episode_forward_velocity_mean_mps": float(
            (first_velocity_sum / first_velocity_count.clamp_min(1)).mean()
        ),
        "combined_action_clip_fraction": clip_count / (NUM_ENVS * STEPS * ACTION_DIM),
        "teacher_weights_loaded": True,
        "scratch_weights_loaded": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "performance_claim": False,
    }


try:
    report = main()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = REPORT.with_name(f".{REPORT.name}.tmp")
    temporary_sidecar = REPORT.with_name(f".{REPORT.name}.sha256.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    temporary_sidecar.write_text(f"{digest}  {REPORT.name}\n", encoding="utf-8")
    os.replace(temporary, REPORT)
    os.replace(temporary_sidecar, REPORT.with_name(f"{REPORT.name}.sha256"))
    print(json.dumps(report, sort_keys=True), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
