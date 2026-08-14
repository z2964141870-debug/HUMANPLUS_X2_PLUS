#!/usr/bin/env python3
"""Screen bounded state-feedback commands for the two failed X2 teacher roles.

This is a teacher-feasibility experiment, not BFM student training.  Stage219
provides direct physical target proposals, while the frozen grid changes only
its command observation from live body velocity/yaw feedback.  No scratch
checkpoint is loaded or written.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher-checkpoint", type=Path, required=True)
parser.add_argument("--template", type=Path, required=True)
parser.add_argument("--baseline-report", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
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
from humanoidverse.x2_teacher_feedback import (  # noqa: E402
    low_speed_candidates,
    right_turn_candidates,
    state_feedback_command,
)


NUM_ENVS = 512
STEPS = 400
SEED = 770381
LANES_PER_CANDIDATE = 16
STAGE219 = args.teacher_checkpoint.expanduser().resolve()
STAGE219_SHA256 = "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb"
TEMPLATE = args.template.expanduser().resolve()
TEMPLATE_SHA256 = "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d"
BASELINE_REPORT = args.baseline_report.expanduser().resolve()
BASELINE_SHA256 = "e82c7365ccbc9008181bb962ef5af3cdb0df3b7be96121559daecd7ec1379de3"
REPORT = args.report.expanduser().resolve()


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
            "ideal_env_fraction": 0.0,
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
    immutable = {
        STAGE219: STAGE219_SHA256,
        TEMPLATE: TEMPLATE_SHA256,
        BASELINE_REPORT: BASELINE_SHA256,
    }
    if any(not path.is_file() or file_hash(path) != digest for path, digest in immutable.items()):
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
    candidates = (*low_speed_candidates(), *right_turn_candidates())
    if len(candidates) * LANES_PER_CANDIDATE != NUM_ENVS:
        raise RuntimeError("candidate grid no longer fills the frozen 512 lanes")
    desired_panel = torch.empty(NUM_ENVS, 3, device="cuda")
    ramp_s = torch.empty(NUM_ENVS, device="cuda")
    kp_vx = torch.empty(NUM_ENVS, device="cuda")
    kp_yaw = torch.empty(NUM_ENVS, device="cuda")
    max_vx = torch.empty(NUM_ENVS, device="cuda")
    max_yaw = torch.empty(NUM_ENVS, device="cuda")
    candidate_slices = []
    for index, candidate in enumerate(candidates):
        start, stop = index * LANES_PER_CANDIDATE, (index + 1) * LANES_PER_CANDIDATE
        desired = (0.20, 0.0, 0.0) if candidate.role == "vx_0p20" else (0.35, 0.0, -0.30)
        desired_panel[start:stop] = torch.tensor(desired, device="cuda")
        ramp_s[start:stop] = candidate.ramp_s
        kp_vx[start:stop] = candidate.kp_vx
        kp_yaw[start:stop] = candidate.kp_yaw
        max_vx[start:stop] = candidate.max_vx_mps
        max_yaw[start:stop] = candidate.max_abs_yaw_radps
        candidate_slices.append((candidate, start, stop, desired))
    env.command_manager.get_command("base_velocity").zero_()
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
    root_height_min = torch.full((NUM_ENVS,), torch.inf, device="cuda")
    root_tilt_max = torch.zeros(NUM_ENVS, device="cuda")
    clip_count_by_lane = torch.zeros(NUM_ENVS, dtype=torch.long, device="cuda")
    terminated_count = 0
    for step in range(1, STEPS + 1):
        data = env.scene["robot"].data
        measured = torch.stack(
            (
                data.root_lin_vel_b[:, 0],
                data.root_lin_vel_b[:, 1],
                data.root_ang_vel_b[:, 2],
            ),
            dim=-1,
        )
        elapsed_s = torch.full(
            (NUM_ENVS,), (step - 1) * float(env.step_dt), device="cuda"
        )
        effective_command = state_feedback_command(
            desired_panel,
            measured,
            elapsed_s,
            ramp_s,
            kp_vx,
            kp_yaw,
            max_vx,
            max_yaw,
        )
        env.command_manager.get_command("base_velocity").copy_(effective_command)
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
        first_velocity_sum[alive] += data.root_lin_vel_b[alive, 0]
        first_yaw_rate_sum[alive] += data.root_ang_vel_b[alive, 2]
        first_velocity_count[alive] += 1
        root_height_min[alive] = torch.minimum(root_height_min[alive], data.root_pos_w[alive, 2])
        tilt = torch.acos(torch.clamp(-data.projected_gravity_b[:, 2], -1.0, 1.0))
        root_tilt_max[alive] = torch.maximum(root_tilt_max[alive], tilt[alive])
        clip_count_by_lane += (preclip.abs() >= 1.0).sum(dim=-1)
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
    qualified_by_role = {"vx_0p20": [], "turn_right": []}
    for candidate, start, stop, desired in candidate_slices:
        lane_count = stop - start
        vx = first_velocity_sum[start:stop] / first_velocity_count[start:stop].clamp_min(1)
        yaw_rate = first_yaw_rate_sum[start:stop] / first_velocity_count[start:stop].clamp_min(1)
        role_survival = survival[start:stop]
        velocity_error = torch.abs(vx - desired[0])
        yaw_error = torch.abs(yaw_rate - desired[2])
        clip_fraction = float(clip_count_by_lane[start:stop].sum()) / (
            lane_count * STEPS * ACTION_DIM
        )
        gates = {
            "all_lanes_survive": bool(role_survival.all()),
            "velocity_error": bool(velocity_error.mean() <= 0.05),
            "yaw_error": bool(yaw_error.mean() <= (0.08 if candidate.role == "vx_0p20" else 0.20)),
            "root_height": bool(root_height_min[start:stop].min() >= 0.55),
            "root_tilt": bool(root_tilt_max[start:stop].max() <= 0.70),
            "action_clip": clip_fraction <= 0.01,
        }
        teacher_gate = all(gates.values())
        record = {
            "lanes": lane_count,
            "desired_command": list(desired),
            "feedback": candidate.record(),
            "survived_lanes": int(role_survival.sum()),
            "survival_steps_mean": float(first_steps[start:stop].float().mean()),
            "forward_velocity_mean_mps": float(vx.mean()),
            "yaw_rate_mean_radps": float(yaw_rate.mean()),
            "velocity_error_mean_mps": float(velocity_error.mean()),
            "yaw_rate_error_mean_radps": float(yaw_error.mean()),
            "root_height_min_m": float(root_height_min[start:stop].min()),
            "root_tilt_max_rad": float(root_tilt_max[start:stop].max()),
            "combined_action_clip_fraction": clip_fraction,
            "gates": gates,
            "teacher_gate": teacher_gate,
        }
        groups[candidate.name] = record
        if teacher_gate:
            qualified_by_role[candidate.role].append(candidate.name)
    selected = {}
    for role, names in qualified_by_role.items():
        if not names:
            selected[role] = None
            continue
        selected[role] = min(
            names,
            key=lambda name: (
                groups[name]["velocity_error_mean_mps"]
                + groups[name]["yaw_rate_error_mean_radps"],
                groups[name]["combined_action_clip_fraction"],
                name,
            ),
        )
    valid = all(selected.values())
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    return {
        "schema": "x2_state_feedback_command_teacher_v8_screen_v1",
        "decision": (
            "PASS_COMMAND_FEEDBACK_TEACHER_FEASIBILITY"
            if valid
            else "FAIL_COMMAND_FEEDBACK_ESCALATE_TO_DIRECT_15D_TEACHER"
        ),
        "preregistered_before_rollout": True,
        "teacher_checkpoint_sha256": STAGE219_SHA256,
        "teacher_iteration": int(payload["iter"]),
        "template_sha256": TEMPLATE_SHA256,
        "baseline_report_sha256": BASELINE_SHA256,
        "baseline_failed_roles": sorted(
            name
            for name, record in baseline["groups"].items()
            if not record["tracking_gate"]
        ),
        "template_scale": 0.15,
        "internal_template_scale": 0.0,
        "physical_action_scale_conversion": True,
        "collision_profile": "sole12",
        "self_collisions": False,
        "actuator_domain": "response_only",
        "groups": groups,
        "qualified_candidates_by_role": qualified_by_role,
        "selected_candidate_by_role": selected,
        "num_envs": NUM_ENVS,
        "control_steps": STEPS,
        "survived_lanes": int(survival.sum()),
        "terminated_events": terminated_count,
        "first_episode_survival_steps_mean": float(first_steps.float().mean()),
        "first_episode_survival_steps_min": int(first_steps.min()),
        "teacher_weights_loaded": True,
        "scratch_weights_loaded": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "performance_claim": False,
        "permissions": {
            "multi_command_teacher_data_unlocked": valid,
            "direct_15d_teacher_required": not valid,
            "student_training_unlocked": False,
            "deployment_unlocked": False,
        },
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
