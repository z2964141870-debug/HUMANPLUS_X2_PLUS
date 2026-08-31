#!/usr/bin/env python3
"""Search and verify a direct 15-D X2 gait-feedback teacher without policy weights."""

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
parser.add_argument("--template", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)

from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.x2_direct_teacher import (  # noqa: E402
    INITIAL_MEAN,
    INITIAL_STD,
    LOWER_BOUNDS,
    PARAMETER_NAMES,
    UPPER_BOUNDS,
    clamp_parameters,
    direct_teacher_action,
    interpolate_cycle,
    update_cem,
)
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
)


NUM_ENVS = 512
CANDIDATES = 64
LANES_PER_CANDIDATE = 8
ELITES = 8
ITERATIONS = 8
SEARCH_STEPS = 300
VERIFY_STEPS = 400
SEED = 770501
CONTROL_DT = 0.02
TEMPLATE = args.template.expanduser().resolve()
TEMPLATE_SHA256 = "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d"
REPORT = args.report.expanduser().resolve()
ROLES = (
    ("vx_0p20", (0.20, 0.0, 0.0)),
    ("turn_right", (0.35, 0.0, -0.30)),
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
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.actions.joint_pos.scale = {
        name: float(scale)
        for name, scale in zip(X2_LOWER_JOINTS_15, X2_SCRATCH_ACTION_SCALE_15)
    }
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


def parameter_dict(vector: torch.Tensor) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAMETER_NAMES, vector)}


def rollout(
    adapter,
    env,
    cycle,
    action_scale,
    parameters,
    command,
    *,
    steps,
    seed,
    candidate_ids=None,
):
    adapter.reset(seed=seed)
    env.command_manager.get_command("base_velocity").copy_(command)
    robot = env.scene["robot"]
    phase = parameters[:, 21].clone()
    previous_action = torch.zeros(NUM_ENVS, ACTION_DIM, device=env.device)
    ever_done = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
    first_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.device)
    score = torch.zeros(NUM_ENVS, device=env.device)
    vx_sum = torch.zeros(NUM_ENVS, device=env.device)
    yaw_sum = torch.zeros(NUM_ENVS, device=env.device)
    metric_count = torch.zeros(NUM_ENVS, device=env.device)
    min_height = torch.full((NUM_ENVS,), torch.inf, device=env.device)
    max_tilt = torch.zeros(NUM_ENVS, device=env.device)
    clip_count = torch.zeros(NUM_ENVS, device=env.device)
    action_count = torch.zeros(NUM_ENVS, device=env.device)

    for step in range(1, steps + 1):
        env.command_manager.get_command("base_velocity").copy_(command)
        data = robot.data
        phase = torch.remainder(phase + CONTROL_DT / parameters[:, 1], 1.0)
        q_cycle = interpolate_cycle(cycle, phase)
        ramp = torch.full(
            (NUM_ENVS,), min(1.0, step * CONTROL_DT / 0.50), device=env.device
        )
        action, preclip = direct_teacher_action(
            parameters,
            q_cycle,
            action_scale,
            data.root_lin_vel_b,
            data.root_ang_vel_b,
            data.projected_gravity_b,
            command,
            previous_action,
            ramp,
        )
        action[ever_done] = 0.0
        _, _, terminated, truncated, _ = adapter.step(action)
        done = terminated | truncated
        new_done = done & ~ever_done
        first_steps[new_done] = step
        alive = ~ever_done & ~done
        data = robot.data
        vx_error = data.root_lin_vel_b[:, 0] - command[:, 0]
        yaw_error = data.root_ang_vel_b[:, 2] - command[:, 2]
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        height = data.root_pos_w[:, 2]
        rate = (action - previous_action).square().mean(-1)
        if not all(
            torch.isfinite(value).all()
            for value in (action, preclip, vx_error, yaw_error, tilt, height)
        ):
            raise RuntimeError("direct CEM rollout produced a non-finite value")
        # Survival is deliberately dominant.  The remaining normalized terms
        # rank stable candidates by command tracking and smooth posture.
        step_score = (
            2.0
            - (vx_error / 0.12).square()
            - (yaw_error / 0.20).square()
            - 1.5 * tilt.square()
            - 0.20 * rate
            - 0.05 * action.square().mean(-1)
        ).clamp(min=-20.0, max=3.0)
        score[alive] += step_score[alive]
        # A terminated trajectory must rank below every full-horizon survivor,
        # even if it briefly tracked the command perfectly before falling.
        score[new_done] -= 10000.0
        if step > 50:
            vx_sum[alive] += data.root_lin_vel_b[alive, 0]
            yaw_sum[alive] += data.root_ang_vel_b[alive, 2]
            metric_count[alive] += 1
        min_height[alive] = torch.minimum(min_height[alive], height[alive])
        max_tilt[alive] = torch.maximum(max_tilt[alive], tilt[alive])
        clip_count[alive] += (preclip[alive].abs() >= 1.0).sum(-1)
        action_count[alive] += ACTION_DIM
        ever_done |= done
        previous_action.copy_(action)
        previous_action[done] = 0.0
    first_steps[~ever_done] = steps
    metrics = {
        "score": score,
        "survived": ~ever_done,
        "first_steps": first_steps,
        "vx": vx_sum / metric_count.clamp_min(1),
        "yaw": yaw_sum / metric_count.clamp_min(1),
        "min_height": min_height,
        "max_tilt": max_tilt,
        "clip_fraction": clip_count / action_count.clamp_min(1),
    }
    if candidate_ids is not None:
        candidate_score = torch.zeros(CANDIDATES, device=env.device)
        for candidate in range(CANDIDATES):
            candidate_score[candidate] = score[candidate_ids == candidate].mean()
        metrics["candidate_score"] = candidate_score
    return metrics


def main() -> dict:
    if not TEMPLATE.is_file() or file_hash(TEMPLATE) != TEMPLATE_SHA256:
        raise RuntimeError("direct teacher immutable template guard failed")
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    if REPORT.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite direct teacher result")
    if CANDIDATES * LANES_PER_CANDIDATE != NUM_ENVS:
        raise RuntimeError("candidate lane partition mismatch")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(env, wrapped, history_length=4, to_numpy=False)
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != X2_LOWER_JOINTS_15:
        raise RuntimeError("direct teacher action order mismatch")
    template = np.load(TEMPLATE, allow_pickle=False)
    cycle = torch.as_tensor(template["q_cycle_zero_mean_rad"], device=env.device)
    action_scale = torch.as_tensor(X2_SCRATCH_ACTION_SCALE_15, device=env.device)
    candidate_ids = torch.div(
        torch.arange(NUM_ENVS, device=env.device),
        LANES_PER_CANDIDATE,
        rounding_mode="floor",
    )
    search_records = {}
    selected = {}
    for role_index, (role, command_tuple) in enumerate(ROLES):
        mean = INITIAL_MEAN.to(env.device).clone()
        std = INITIAL_STD.to(env.device).clone()
        best_score = -torch.inf
        best = None
        iterations = []
        command = torch.tensor(command_tuple, device=env.device).repeat(NUM_ENVS, 1)
        for iteration in range(ITERATIONS):
            samples = clamp_parameters(
                mean[None] + std[None] * torch.randn(CANDIDATES, len(PARAMETER_NAMES), device=env.device)
            )
            # Strict elitism keeps the best safe controller in the population;
            # candidate one separately evaluates the updated distribution mean.
            samples[0] = mean if best is None else best
            samples[1] = mean
            parameters = samples[candidate_ids]
            metrics = rollout(
                adapter,
                env,
                cycle,
                action_scale,
                parameters,
                command,
                steps=SEARCH_STEPS,
                seed=SEED + role_index * 100 + iteration,
                candidate_ids=candidate_ids,
            )
            scores = metrics["candidate_score"]
            iteration_best = int(torch.argmax(scores))
            if scores[iteration_best] > best_score:
                best_score = scores[iteration_best].clone()
                best = samples[iteration_best].clone()
            mean, std, elite_ids = update_cem(
                mean, std, samples, scores, elite_count=ELITES
            )
            iterations.append(
                {
                    "iteration": iteration,
                    "best_score": float(scores[iteration_best]),
                    "mean_score": float(scores.mean()),
                    "best_survived_lanes": int(
                        metrics["survived"][candidate_ids == iteration_best].sum()
                    ),
                    "best_vx_mps": float(
                        metrics["vx"][candidate_ids == iteration_best].mean()
                    ),
                    "best_yaw_radps": float(
                        metrics["yaw"][candidate_ids == iteration_best].mean()
                    ),
                    "elite_ids": [int(value) for value in elite_ids],
                }
            )
            print(json.dumps({"role": role, **iterations[-1]}, sort_keys=True), flush=True)
        if best is None:
            raise RuntimeError(f"CEM did not select a candidate for {role}")
        selected[role] = best
        search_records[role] = {
            "best_score": float(best_score),
            "parameters": parameter_dict(best),
            "iterations": iterations,
        }

    half = NUM_ENVS // 2
    parameters = torch.empty(NUM_ENVS, len(PARAMETER_NAMES), device=env.device)
    command = torch.empty(NUM_ENVS, 3, device=env.device)
    parameters[:half] = selected[ROLES[0][0]]
    parameters[half:] = selected[ROLES[1][0]]
    command[:half] = torch.tensor(ROLES[0][1], device=env.device)
    command[half:] = torch.tensor(ROLES[1][1], device=env.device)
    verification = rollout(
        adapter,
        env,
        cycle,
        action_scale,
        parameters,
        command,
        steps=VERIFY_STEPS,
        seed=SEED + 999,
    )
    groups = {}
    for index, (role, command_tuple) in enumerate(ROLES):
        sl = slice(index * half, (index + 1) * half)
        vx_error = torch.abs(verification["vx"][sl] - command_tuple[0])
        yaw_error = torch.abs(verification["yaw"][sl] - command_tuple[2])
        gates = {
            "all_lanes_survive": bool(verification["survived"][sl].all()),
            "velocity_error": bool(vx_error.mean() <= (0.07 if role == "vx_0p20" else 0.10)),
            "yaw_error": bool(yaw_error.mean() <= (0.10 if role == "vx_0p20" else 0.15)),
            "root_height": bool(verification["min_height"][sl].min() >= 0.55),
            "root_tilt": bool(verification["max_tilt"][sl].max() <= 0.70),
            "action_clip": bool(verification["clip_fraction"][sl].mean() <= 0.01),
        }
        groups[role] = {
            "lanes": half,
            "command": list(command_tuple),
            "survived_lanes": int(verification["survived"][sl].sum()),
            "survival_steps_mean": float(verification["first_steps"][sl].float().mean()),
            "forward_velocity_mean_mps": float(verification["vx"][sl].mean()),
            "yaw_rate_mean_radps": float(verification["yaw"][sl].mean()),
            "velocity_error_mean_mps": float(vx_error.mean()),
            "yaw_rate_error_mean_radps": float(yaw_error.mean()),
            "root_height_min_m": float(verification["min_height"][sl].min()),
            "root_tilt_max_rad": float(verification["max_tilt"][sl].max()),
            "action_clip_fraction": float(verification["clip_fraction"][sl].mean()),
            "gates": gates,
            "teacher_gate": all(gates.values()),
        }
    passed = all(group["teacher_gate"] for group in groups.values())
    return {
        "schema": "x2_direct_cem_teacher_v9_result_v1",
        "decision": (
            "PASS_DIRECT_15D_GAIT_FEEDBACK_TEACHER"
            if passed
            else "FAIL_PARAMETRIC_TEACHER_REQUIRE_CLONE_STATE_MPC_OR_RL"
        ),
        "preregistered_before_rollout": True,
        "teacher_policy_weights_loaded": False,
        "scratch_policy_weights_loaded": False,
        "stage219_weights_loaded": False,
        "template_sha256": TEMPLATE_SHA256,
        "template_role": "zero_mean_periodic_prior_only",
        "action_contract": "direct_normalized_15d_lower_joint_position_targets",
        "actuator_domain": "identified_session03_session04_response_only",
        "collision_profile": "sole12",
        "self_collisions": False,
        "num_envs": NUM_ENVS,
        "cem_candidates": CANDIDATES,
        "lanes_per_candidate": LANES_PER_CANDIDATE,
        "cem_iterations": ITERATIONS,
        "search_steps": SEARCH_STEPS,
        "verification_steps": VERIFY_STEPS,
        "search": search_records,
        "verification": groups,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "performance_claim": False,
        "permissions": {
            "teacher_data_collection_unlocked": passed,
            "scratch_student_training_unlocked": passed,
            "clone_state_mpc_or_rl_teacher_required": not passed,
            "deployment_unlocked": False,
        },
    }


def write_report(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, REPORT)
    digest = file_hash(REPORT)
    REPORT.with_name(f"{REPORT.name}.sha256").write_text(f"{digest}  {REPORT.name}\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


try:
    write_report(main())
except Exception as error:  # pragma: no cover - evidence path
    traceback.print_exc()
    print(json.dumps({"error": type(error).__name__, "message": str(error)}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
