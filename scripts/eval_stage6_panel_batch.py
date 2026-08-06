#!/usr/bin/env python3
"""Evaluate one Stage6 branch on a 6-motion by 2-speed physical panel.

The simulator runs all cases concurrently, but policy inference is deliberately
performed one environment at a time.  This preserves the batch-size-one GEMM
path used by the validated legacy evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--adapter-mode",
    choices=("disabled", "current", "future", "future_no_phase"),
    required=True,
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--speeds", type=float, nargs="+", default=(0.20, 0.30))
parser.add_argument("--gait-template", type=Path, required=True)
parser.add_argument("--stage-label", type=str, required=True)
parser.add_argument("--coordination-blend", type=float, default=1.0)
parser.add_argument("--initial-roll-pitch-range-rad", type=float, default=0.0)
parser.add_argument("--initial-yaw-range-rad", type=float, default=0.0)
parser.add_argument(
    "--initial-lateral-velocity-range-mps",
    type=float,
    default=0.0,
)
parser.add_argument("--initial-yaw-rate-range-radps", type=float, default=0.0)
parser.add_argument("--record-diagnostic-traces", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from cwi_x2.future_intent import (  # noqa: E402
    X2FutureIntentActorCriticCfg,
    X2FutureIntentFlatEnvCfg_PLAY,
)
from cwi_x2.future_intent_actor_critic import FutureIntentActorCritic  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)
from gear_sonic.envs.x2_velocity.heading_command import (  # noqa: E402
    heading_pi_velocity_cfg,
)
from gear_sonic.envs.x2_velocity.rsl_rl_ppo_cfg import (  # noqa: E402
    X2LowerVelocityFlatPPORunnerCfg,
)


def _motion_paths() -> list[str]:
    raw_list = os.environ.get("CWI_UPPER_MOTION_LIST")
    if raw_list:
        result = [path for path in raw_list.split(";") if path]
    elif os.environ.get("CWI_UPPER_MOTION"):
        result = [os.environ["CWI_UPPER_MOTION"]]
    else:
        raise RuntimeError("Stage6 panel requires CWI_UPPER_MOTION_LIST")
    if not result:
        raise RuntimeError("Stage6 panel motion list is empty")
    return result


def _motion_tags(count: int) -> list[str]:
    raw = os.environ.get("CWI_STAGE6_PANEL_TAGS", "")
    tags = [tag for tag in raw.split(";") if tag]
    if len(tags) != count:
        raise RuntimeError(
            f"CWI_STAGE6_PANEL_TAGS expected {count} entries, got {len(tags)}"
        )
    return tags


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _ground_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    matrix = env.scene[sensor_name].data.force_matrix_w
    if matrix is None:
        raise RuntimeError(f"{sensor_name} has no force matrix")
    return matrix[..., 2].abs().reshape(matrix.shape[0], -1).amax(dim=-1)


def _policy_batch_one(policy, observations, count: int) -> torch.Tensor:
    return torch.cat(
        [policy(observations[index : index + 1]) for index in range(count)],
        dim=0,
    )


def _percentile_or_nan(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    template = args.gait_template.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not template.is_file():
        raise FileNotFoundError(template)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not args.speeds or any(speed < 0.0 for speed in args.speeds):
        raise ValueError("--speeds must contain non-negative values")
    if not 0.0 <= args.coordination_blend <= 1.0:
        raise ValueError("--coordination-blend must lie in [0, 1]")
    perturbation_ranges = {
        "roll_pitch_rad": args.initial_roll_pitch_range_rad,
        "yaw_rad": args.initial_yaw_range_rad,
        "lateral_velocity_mps": args.initial_lateral_velocity_range_mps,
        "yaw_rate_radps": args.initial_yaw_rate_range_radps,
    }
    if any(value < 0.0 for value in perturbation_ranges.values()):
        raise ValueError("initial perturbation half-ranges must be non-negative")

    paths = _motion_paths()
    tags = _motion_tags(len(paths))
    case_count = len(paths) * len(args.speeds)
    speed_by_env = torch.tensor(
        [
            speed
            for speed in args.speeds
            for _ in range(len(paths))
        ],
        device=args.device,
        dtype=torch.float32,
    )

    cfg = X2FutureIntentFlatEnvCfg_PLAY()
    cfg.scene.num_envs = case_count
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.episode_length_s = args.steps * cfg.decimation * cfg.sim.dt + 2.0
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale
        for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    cfg.actions.joint_pos.template_path = str(template)
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (
        min(args.speeds),
        max(args.speeds),
    )
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = True
    cfg.commands.base_velocity.rel_heading_envs = 1.0
    cfg.commands.base_velocity.heading_control_stiffness = 0.05
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
    cfg.commands.base_velocity = heading_pi_velocity_cfg(
        cfg.commands.base_velocity,
        heading_control_stiffness=0.05,
        heading_integral_gain=0.0,
        heading_integral_rate_limit=0.05,
    )
    cfg.observations.policy.enable_corruption = False
    cfg.events.reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (
                -args.initial_roll_pitch_range_rad,
                args.initial_roll_pitch_range_rad,
            ),
            "pitch": (
                -args.initial_roll_pitch_range_rad,
                args.initial_roll_pitch_range_rad,
            ),
            "yaw": (
                -args.initial_yaw_range_rad,
                args.initial_yaw_range_rad,
            ),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (
                -args.initial_lateral_velocity_range_mps,
                args.initial_lateral_velocity_range_mps,
            ),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (
                -args.initial_yaw_rate_range_radps,
                args.initial_yaw_rate_range_radps,
            ),
        },
    }
    actuator_report = _apply_x2_actuator_response(
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
        },
        physics_dt_sec=cfg.sim.dt,
    )

    agent_cfg = X2LowerVelocityFlatPPORunnerCfg()
    agent_cfg.device = args.device
    policy_cfg = X2FutureIntentActorCriticCfg()
    policy_cfg.class_name = "ResponseHistoryActorCritic"
    policy_cfg.adapter_mode = args.adapter_mode
    policy_cfg.coordination_blend = args.coordination_blend
    agent_cfg.policy = policy_cfg
    import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

    rsl_on_policy_runner.ResponseHistoryActorCritic = FutureIntentActorCritic

    env = None
    wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        policy_module = runner.alg.policy
        robot = env.scene["robot"]
        action_term = env.action_manager._terms["joint_pos"]
        command_term = env.command_manager.get_term("base_velocity")
        expected_clip_ids = torch.arange(
            case_count,
            device=env.device,
            dtype=torch.int64,
        ) % len(paths)
        if not torch.equal(action_term._cwi_upper_clip_ids, expected_clip_ids):
            raise RuntimeError("upper-motion clip assignment is not deterministic")
        if tuple(action_term._cwi_upper_clip_paths) != tuple(paths):
            raise RuntimeError("upper-motion path order changed inside the action term")

        command_term.command[:, 0] = speed_by_env
        command_term.command[:, 1] = 0.0
        observations = wrapped.get_observations()
        foot_ids, foot_names = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        upper_ids = action_term._cwi_upper_joint_ids
        initial_root = robot.data.root_pos_w.clone()
        initial_root_quat = robot.data.root_quat_w.clone()
        initial_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        initial_root_lin_vel_w = robot.data.root_lin_vel_w.clone()
        initial_root_ang_vel_w = robot.data.root_ang_vel_w.clone()
        initial_tilt = torch.acos(
            torch.clamp(
                -robot.data.projected_gravity_b[:, 2],
                -1.0,
                1.0,
            )
        ).clone()
        initial_foot_z = robot.data.body_pos_w[:, foot_ids, 2].clone()
        previous_upper_target = robot.data.joint_pos_target[:, upper_ids].clone()

        active = torch.ones(case_count, dtype=torch.bool, device=env.device)
        steps_completed = torch.zeros(
            case_count,
            dtype=torch.int64,
            device=env.device,
        )
        first_done = torch.full(
            (case_count,),
            -1,
            dtype=torch.int64,
            device=env.device,
        )
        samples = torch.zeros(case_count, dtype=torch.int64, device=env.device)
        root_z_min = torch.full(
            (case_count,),
            float("inf"),
            device=env.device,
        )
        root_z_max = torch.full(
            (case_count,),
            -float("inf"),
            device=env.device,
        )
        tilt_max = torch.zeros(case_count, device=env.device)
        lateral_max = torch.zeros(case_count, device=env.device)
        forward_final = torch.zeros(case_count, device=env.device)
        heading_max = torch.zeros(case_count, device=env.device)
        heading_final = torch.zeros(case_count, device=env.device)
        world_heading_abs_max = torch.abs(initial_yaw).clone()
        world_heading_final = initial_yaw.clone()
        body_vx_sq_error_sum = torch.zeros(case_count, device=env.device)
        yaw_rate_abs_sum = torch.zeros(case_count, device=env.device)
        foot_lift_max = torch.zeros(case_count, 2, device=env.device)
        foot_contact_count = torch.zeros(case_count, 2, device=env.device)
        single_support_count = torch.zeros(case_count, device=env.device)
        double_support_count = torch.zeros(case_count, device=env.device)
        flight_count = torch.zeros(case_count, device=env.device)
        residual_sq_sum = torch.zeros(case_count, device=env.device)
        residual_count = torch.zeros(case_count, device=env.device)
        residual_abs_max = torch.zeros(case_count, device=env.device)
        upper_excursion_max = torch.zeros(case_count, device=env.device)
        upper_velocity_max = torch.zeros(case_count, device=env.device)
        hazard_count = torch.zeros(case_count, device=env.device)
        upper_tracking_values: list[list[float]] = [
            [] for _ in range(case_count)
        ]
        lateral_traces: list[list[float]] = [
            [] for _ in range(case_count)
        ]
        world_heading_traces: list[list[float]] = [
            [] for _ in range(case_count)
        ]
        tilt_traces: list[list[float]] = [
            [] for _ in range(case_count)
        ]
        root_z_traces: list[list[float]] = [
            [] for _ in range(case_count)
        ]

        for step in range(args.steps):
            with torch.inference_mode():
                actions = _policy_batch_one(policy, observations, case_count)
                actor_obs = policy_module.get_actor_obs(observations)
                actor_obs = policy_module.actor_obs_normalizer(actor_obs)
                base_actions = torch.cat(
                    [
                        policy_module.actor(
                            actor_obs[
                                index : index + 1,
                                : policy_module.base_actor_obs_dim,
                            ]
                        )
                        for index in range(case_count)
                    ],
                    dim=0,
                )
                residual = actions - base_actions
                observations, _, dones, _ = wrapped.step(actions)

            newly_done = active & dones
            first_done[newly_done] = step
            valid = active & ~dones
            if torch.any(valid):
                index = torch.nonzero(valid, as_tuple=False).flatten()
                steps_completed[index] += 1
                samples[index] += 1
                root = robot.data.root_pos_w[index]
                tilt = torch.acos(
                    torch.clamp(
                        -robot.data.projected_gravity_b[index, 2],
                        -1.0,
                        1.0,
                    )
                )
                yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[index])
                yaw_delta = torch.atan2(
                    torch.sin(yaw - initial_yaw[index]),
                    torch.cos(yaw - initial_yaw[index]),
                )
                world_heading_error = torch.abs(
                    torch.atan2(torch.sin(yaw), torch.cos(yaw))
                )
                lateral = torch.abs(root[:, 1] - initial_root[index, 1])
                forward = root[:, 0] - initial_root[index, 0]
                foot_lift = (
                    robot.data.body_pos_w[index][:, foot_ids, 2]
                    - initial_foot_z[index]
                )
                left_force = _ground_force(
                    env,
                    "left_foot_ground_contact",
                )[index]
                right_force = _ground_force(
                    env,
                    "right_foot_ground_contact",
                )[index]
                contact = torch.stack(
                    (left_force > 10.0, right_force > 10.0),
                    dim=-1,
                )
                contact_count = contact.to(torch.int64).sum(dim=-1)
                upper_target = robot.data.joint_pos_target[index][:, upper_ids]
                upper_default = robot.data.default_joint_pos[index][:, upper_ids]
                upper_actual = robot.data.joint_pos[index][:, upper_ids]
                upper_velocity = torch.abs(
                    (
                        upper_target
                        - previous_upper_target[index]
                    )
                    / env.step_dt
                )
                upper_error = torch.abs(upper_actual - upper_target)

                root_z_min[index] = torch.minimum(root_z_min[index], root[:, 2])
                root_z_max[index] = torch.maximum(root_z_max[index], root[:, 2])
                tilt_max[index] = torch.maximum(tilt_max[index], tilt)
                lateral_max[index] = torch.maximum(lateral_max[index], lateral)
                forward_final[index] = forward
                heading_max[index] = torch.maximum(
                    heading_max[index],
                    torch.abs(yaw_delta),
                )
                heading_final[index] = yaw_delta
                world_heading_abs_max[index] = torch.maximum(
                    world_heading_abs_max[index],
                    world_heading_error,
                )
                world_heading_final[index] = torch.atan2(
                    torch.sin(yaw),
                    torch.cos(yaw),
                )
                body_vx_error = (
                    robot.data.root_lin_vel_b[index, 0]
                    - speed_by_env[index]
                )
                body_vx_sq_error_sum[index] += body_vx_error.square()
                yaw_rate_abs_sum[index] += torch.abs(
                    robot.data.root_ang_vel_w[index, 2]
                )
                foot_lift_max[index] = torch.maximum(
                    foot_lift_max[index],
                    foot_lift,
                )
                foot_contact_count[index] += contact.to(torch.float32)
                single_support_count[index] += (contact_count == 1)
                double_support_count[index] += (contact_count == 2)
                flight_count[index] += (contact_count == 0)
                residual_sq_sum[index] += residual[index].square().sum(dim=-1)
                residual_count[index] += residual[index].shape[-1]
                residual_abs_max[index] = torch.maximum(
                    residual_abs_max[index],
                    residual[index].abs().amax(dim=-1),
                )
                upper_excursion_max[index] = torch.maximum(
                    upper_excursion_max[index],
                    torch.abs(upper_target - upper_default).amax(dim=-1),
                )
                upper_velocity_max[index] = torch.maximum(
                    upper_velocity_max[index],
                    upper_velocity.amax(dim=-1),
                )
                hazard_count[index] += (
                    (tilt > 0.35) | (root[:, 2] < 0.58)
                )
                for row, environment_index in enumerate(index.tolist()):
                    upper_tracking_values[environment_index].extend(
                        upper_error[row].detach().cpu().tolist()
                    )
                    if args.record_diagnostic_traces:
                        lateral_traces[environment_index].append(
                            float(lateral[row].item())
                        )
                        world_heading_traces[environment_index].append(
                            float(world_heading_error[row].item())
                        )
                        tilt_traces[environment_index].append(
                            float(tilt[row].item())
                        )
                        root_z_traces[environment_index].append(
                            float(root[row, 2].item())
                        )
                previous_upper_target[index] = upper_target

            active = valid
            policy_module.reset(dones)
            if not torch.any(active):
                break

        cases: list[dict[str, object]] = []
        for environment_index in range(case_count):
            clip_id = environment_index % len(paths)
            speed_index = environment_index // len(paths)
            count = int(samples[environment_index].item())
            duration = count * env.step_dt
            speed = float(args.speeds[speed_index])
            forward = float(forward_final[environment_index].item())
            expected_forward = speed * duration
            denominator = max(count, 1)
            case = {
                    "case_id": f"{tags[clip_id]}__vx{speed:.2f}",
                    "motion_tag": tags[clip_id],
                    "motion_path": paths[clip_id],
                    "speed_mps": speed,
                    "environment_index": environment_index,
                    "initial_root_quat_wxyz": [
                        float(value)
                        for value in initial_root_quat[
                            environment_index
                        ].tolist()
                    ],
                    "initial_heading_rad": float(
                        initial_yaw[environment_index].item()
                    ),
                    "initial_base_tilt_rad": float(
                        initial_tilt[environment_index].item()
                    ),
                    "initial_root_lin_vel_w_xyz_mps": [
                        float(value)
                        for value in initial_root_lin_vel_w[
                            environment_index
                        ].tolist()
                    ],
                    "initial_root_ang_vel_w_xyz_radps": [
                        float(value)
                        for value in initial_root_ang_vel_w[
                            environment_index
                        ].tolist()
                    ],
                    "steps_requested": args.steps,
                    "steps_completed": int(
                        steps_completed[environment_index].item()
                    ),
                    "first_done_step": (
                        None
                        if int(first_done[environment_index].item()) < 0
                        else int(first_done[environment_index].item())
                    ),
                    "survived_full_horizon": (
                        int(steps_completed[environment_index].item())
                        == args.steps
                    ),
                    "duration_s": duration,
                    "forward_displacement_m": forward,
                    "expected_forward_displacement_m": expected_forward,
                    "progress_ratio": (
                        forward / expected_forward
                        if expected_forward > 1.0e-9
                        else float("nan")
                    ),
                    "lateral_drift_max_m": float(
                        lateral_max[environment_index].item()
                    ),
                    "body_vx_rmse_mps": math.sqrt(
                        float(
                            body_vx_sq_error_sum[environment_index].item()
                        )
                        / denominator
                    ),
                    "yaw_rate_abs_mean_radps": float(
                        yaw_rate_abs_sum[environment_index].item()
                    )
                    / denominator,
                    "heading_deviation_abs_max_rad": float(
                        heading_max[environment_index].item()
                    ),
                    "heading_deviation_final_rad": float(
                        heading_final[environment_index].item()
                    ),
                    "world_heading_error_abs_max_rad": float(
                        world_heading_abs_max[environment_index].item()
                    ),
                    "world_heading_error_final_rad": float(
                        world_heading_final[environment_index].item()
                    ),
                    "root_z_min_m": float(
                        root_z_min[environment_index].item()
                    ),
                    "root_z_max_m": float(
                        root_z_max[environment_index].item()
                    ),
                    "base_tilt_max_rad": float(
                        tilt_max[environment_index].item()
                    ),
                    "left_foot_lift_max_m": float(
                        foot_lift_max[environment_index, 0].item()
                    ),
                    "right_foot_lift_max_m": float(
                        foot_lift_max[environment_index, 1].item()
                    ),
                    "left_ground_contact_fraction": float(
                        foot_contact_count[environment_index, 0].item()
                    )
                    / denominator,
                    "right_ground_contact_fraction": float(
                        foot_contact_count[environment_index, 1].item()
                    )
                    / denominator,
                    "single_support_fraction": float(
                        single_support_count[environment_index].item()
                    )
                    / denominator,
                    "double_support_fraction": float(
                        double_support_count[environment_index].item()
                    )
                    / denominator,
                    "flight_fraction": float(
                        flight_count[environment_index].item()
                    )
                    / denominator,
                    "coordination_residual_rms": math.sqrt(
                        float(residual_sq_sum[environment_index].item())
                        / max(
                            float(residual_count[environment_index].item()),
                            1.0,
                        )
                    ),
                    "coordination_residual_abs_max": float(
                        residual_abs_max[environment_index].item()
                    ),
                    "upper_target_excursion_abs_max_rad": float(
                        upper_excursion_max[environment_index].item()
                    ),
                    "upper_target_velocity_abs_max_radps": float(
                        upper_velocity_max[environment_index].item()
                    ),
                    "upper_tracking_abs_p95_rad": _percentile_or_nan(
                        upper_tracking_values[environment_index],
                        95.0,
                    ),
                    "hazard_fraction": float(
                        hazard_count[environment_index].item()
                    )
                    / denominator,
                }
            if args.record_diagnostic_traces:
                case["diagnostic_trace"] = {
                    "lateral_drift_m": lateral_traces[environment_index],
                    "world_heading_error_abs_rad": (
                        world_heading_traces[environment_index]
                    ),
                    "base_tilt_rad": tilt_traces[environment_index],
                    "root_z_m": root_z_traces[environment_index],
                }
            cases.append(case)

        report = {
            "schema_version": 1,
            "stage": args.stage_label,
            "checkpoint": str(checkpoint),
            "adapter_mode": args.adapter_mode,
            "coordination_blend": args.coordination_blend,
            "seed": args.seed,
            "initial_perturbation_half_ranges": perturbation_ranges,
            "diagnostic_traces_recorded": args.record_diagnostic_traces,
            "device": args.device,
            "inference_batch_size": 1,
            "simulator_environment_count": case_count,
            "steps": args.steps,
            "speeds_mps": list(args.speeds),
            "motion_tags": tags,
            "gait_template": str(template),
            "gait_template_scale": 0.15,
            "upper_contract": {
                "scale": float(os.environ.get("CWI_UPPER_SCALE", "nan")),
                "time_scale": float(
                    os.environ.get("CWI_UPPER_TIME_SCALE", "nan")
                ),
                "max_excursion_rad": float(
                    os.environ.get("CWI_UPPER_MAX_EXCURSION_RAD", "nan")
                ),
                "max_velocity_radps": float(
                    os.environ.get("CWI_UPPER_MAX_VELOCITY_RADPS", "nan")
                ),
                "future_horizon_s": 0.6,
            },
            "actuator_domain": "nominal_delay",
            "actuator_response_report": actuator_report,
            "foot_body_names": foot_names,
            "cases": cases,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "stage": args.stage_label,
            "adapter_mode": args.adapter_mode,
            "coordination_blend": args.coordination_blend,
            "checkpoint": str(checkpoint),
            "survived": sum(
                bool(case["survived_full_horizon"]) for case in cases
            ),
            "case_count": len(cases),
            "output": str(output),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(
            f"[Stage6 panel] FATAL {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
