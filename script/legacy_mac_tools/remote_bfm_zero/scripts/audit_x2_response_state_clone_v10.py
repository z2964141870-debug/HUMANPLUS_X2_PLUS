#!/usr/bin/env python3
"""Audit full X2 response-state cloning before implementing parallel MPC."""

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
from humanoidverse.x2_response_clone import (  # noqa: E402
    clone_action_manager_rows,
    clone_robot_actuator_rows,
    clone_tensor_rows,
    max_row_deviation,
)
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
)


NUM_ENVS = 128
WARMUP_STEPS = 25
AUDIT_STEPS = 50
SEED = 770601
IMMEDIATE_GATE = 2.0e-5
ROLLOUT_GATE = 2.0e-4
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
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 1.0
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


def relative_root_state(env, robot) -> torch.Tensor:
    root = robot.data.root_state_w.clone()
    root[:, :3] -= env.scene.env_origins
    return root


def clone_live_state(env, adapter, source_id: int = 0) -> dict:
    robot = env.scene["robot"]
    target_ids = torch.arange(1, NUM_ENVS, device=env.device, dtype=torch.long)
    root = robot.data.root_state_w[source_id].repeat(NUM_ENVS, 1)
    relative_position = robot.data.root_pos_w[source_id] - env.scene.env_origins[source_id]
    root[:, :3] = relative_position + env.scene.env_origins
    joint_pos = robot.data.joint_pos[source_id].repeat(NUM_ENVS, 1)
    joint_vel = robot.data.joint_vel[source_id].repeat(NUM_ENVS, 1)
    all_ids = torch.arange(NUM_ENVS, device=env.device, dtype=torch.long)
    robot.write_root_state_to_sim(root, env_ids=all_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=all_ids)
    actuator_fields = clone_robot_actuator_rows(robot, source_id, target_ids)
    action_fields = clone_action_manager_rows(env.action_manager, source_id, target_ids)
    for value in (
        env.episode_length_buf,
        adapter._last_action,
        adapter._history,
    ):
        clone_tensor_rows(value, source_id, target_ids)
    command = env.command_manager.get_command("base_velocity")
    clone_tensor_rows(command, source_id, target_ids)
    env.sim.forward()
    env.scene.update(0.0)
    return {
        "actuator_fields": actuator_fields,
        "action_fields": action_fields,
    }


def state_deviation(env, robot) -> dict[str, float]:
    return {
        "root_state_relative": max_row_deviation(relative_root_state(env, robot)),
        "joint_pos": max_row_deviation(robot.data.joint_pos),
        "joint_vel": max_row_deviation(robot.data.joint_vel),
        "projected_gravity": max_row_deviation(robot.data.projected_gravity_b),
        "root_lin_vel_body": max_row_deviation(robot.data.root_lin_vel_b),
        "root_ang_vel_body": max_row_deviation(robot.data.root_ang_vel_b),
    }


def actuator_deviation(robot) -> dict[str, float]:
    output = {}
    for group, actuator in robot.actuators.items():
        for field in ("_filtered_joint_positions", "_position_alpha"):
            value = getattr(actuator, field, None)
            if isinstance(value, torch.Tensor):
                output[f"{group}.{field}"] = max_row_deviation(value)
        for field in (
            "positions_delay_buffer",
            "velocities_delay_buffer",
            "efforts_delay_buffer",
        ):
            value = getattr(actuator, field, None)
            if value is not None and value._circular_buffer._buffer is not None:
                buffer = value._circular_buffer._buffer.transpose(0, 1).flatten(1)
                output[f"{group}.{field}"] = max_row_deviation(buffer)
    return output


def audit_action(step: int, device) -> torch.Tensor:
    phase = torch.tensor(2.0 * torch.pi * step / 40.0, device=device)
    base = torch.zeros(ACTION_DIM, device=device)
    base[0] = 0.06 * torch.sin(phase)
    base[6] = 0.06 * torch.sin(phase + torch.pi)
    base[3] = -0.05 * torch.sin(phase)
    base[9] = -0.05 * torch.sin(phase + torch.pi)
    base[4] = 0.04 * torch.sin(phase)
    base[10] = 0.04 * torch.sin(phase + torch.pi)
    base[12] = 0.02 * torch.sin(0.5 * phase)
    return base.repeat(NUM_ENVS, 1)


def main() -> dict:
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    if REPORT.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite clone audit evidence")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(env, wrapped, history_length=4, to_numpy=False)
    adapter.reset(seed=SEED)
    robot = env.scene["robot"]
    for step in range(WARMUP_STEPS):
        adapter.step(audit_action(step, env.device))
    cloned = clone_live_state(env, adapter)
    immediate_state = state_deviation(env, robot)
    immediate_actuator = actuator_deviation(robot)
    immediate_max = max((*immediate_state.values(), *immediate_actuator.values()))

    maximum_state = dict(immediate_state)
    maximum_actuator = dict(immediate_actuator)
    terminated_lanes = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
    for step in range(AUDIT_STEPS):
        _, _, terminated, truncated, _ = adapter.step(
            audit_action(WARMUP_STEPS + step, env.device)
        )
        terminated_lanes |= terminated | truncated
        for name, value in state_deviation(env, robot).items():
            maximum_state[name] = max(maximum_state[name], value)
        for name, value in actuator_deviation(robot).items():
            maximum_actuator[name] = max(maximum_actuator[name], value)
    rollout_max = max((*maximum_state.values(), *maximum_actuator.values()))
    passed = (
        immediate_max <= IMMEDIATE_GATE
        and rollout_max <= ROLLOUT_GATE
        and not bool(terminated_lanes.any())
    )
    return {
        "schema": "x2_response_state_clone_audit_v10",
        "decision": (
            "PASS_FULL_RESPONSE_STATE_CLONE_FOR_PARALLEL_MPC"
            if passed
            else "FAIL_RESPONSE_STATE_CLONE_BLOCK_PARALLEL_MPC"
        ),
        "preregistered_before_rollout": True,
        "num_envs": NUM_ENVS,
        "warmup_steps": WARMUP_STEPS,
        "audit_steps": AUDIT_STEPS,
        "immediate_gate": IMMEDIATE_GATE,
        "rollout_gate": ROLLOUT_GATE,
        "immediate_max_abs": immediate_max,
        "rollout_max_abs": rollout_max,
        "immediate_state_deviation": immediate_state,
        "immediate_actuator_deviation": immediate_actuator,
        "rollout_state_deviation": maximum_state,
        "rollout_actuator_deviation": maximum_actuator,
        "terminated_lanes": int(terminated_lanes.sum()),
        "cloned_fields": cloned,
        "actuator_domain": "identified_session03_session04_response_only",
        "action_contract": "direct_normalized_15d_lower_joint_position_targets",
        "policy_weights_loaded": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "performance_claim": False,
        "permissions": {
            "parallel_mpc_teacher_unlocked": passed,
            "student_training_unlocked": False,
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
