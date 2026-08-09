#!/usr/bin/env python3
"""16-env zero-update injection probe for one Phase20 state role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--expected-manifest-sha256", required=True)
parser.add_argument("--state-role", choices=("success_safe", "critical_from_failure"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=47)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402

from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from official_x2.outcome_aware_state_role_v2 import (  # noqa: E402
    fixed_batch_indices,
    load_manifest,
    reset_from_outcome_aware_v2,
    resolve_rows,
)
from official_x2.recovery_reset_curriculum import joint_reorder_indices  # noqa: E402
from official_x2.recovery_suffix_aggregation import canonical_sha256, sha256_file  # noqa: E402
from official_x2.stateful_recovery_isaac import (  # noqa: E402
    StatefulRecoveryRLEnv,
    configure_stateful_recovery_cfg,
)


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual - expected)).item())


def main() -> None:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path, args.expected_manifest_sha256)
    fixed_indices = fixed_batch_indices(manifest, args.state_role, args.num_envs)

    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    template = Path(
        "/home/humanplus/x2_teleop_final/x2_sonic/data/processed/"
        "x2_official_forward_gait_phase_template_15dof.npz"
    )
    cfg.actions.joint_pos.template_path = str(template)
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    configure_stateful_recovery_cfg(cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.scene.robot.soft_joint_pos_limit_factor = 1.0
    cfg.scene.robot.actuators["feet"].stiffness = 40.0
    cfg.scene.robot.actuators["feet"].damping = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.events.recovery_state_reset = EventTermCfg(
        func=reset_from_outcome_aware_v2,
        mode="reset",
        params={
            "manifest_path": str(manifest_path),
            "reset_fraction": 1.0,
            "state_role_name": args.state_role,
            "expected_manifest_sha256": args.expected_manifest_sha256,
            "fixed_manifest_indices": fixed_indices,
            "asset_name": "robot",
        },
    )

    report: dict = {
        "stage": "BASE Phase20 outcome-aware state-role zero-update injection",
        "state_role": args.state_role,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "fixed_manifest_indices": fixed_indices,
        "ppo_updates": 0,
        "optimizer_steps": 0,
        "failure_actions_are_expert_labels": False,
    }
    env = None
    try:
        env = StatefulRecoveryRLEnv(cfg=cfg)
        observation, _ = env.reset(seed=args.seed)
        debug = env._x2_phase20_reset_last
        if not debug.get("stateful_finalized"):
            raise RuntimeError("Phase20 post-manager finalizer did not run")
        selected = debug["selected_env_ids"]
        indices = debug["manifest_indices"].detach().cpu().tolist()
        if indices != fixed_indices or len(selected) != args.num_envs:
            raise RuntimeError("Phase20 fixed batch identity changed")
        rows = resolve_rows(manifest, indices)
        robot = env.scene["robot"]
        source_names = rows[0]["physical_state"]["joint_names"]
        reorder = torch.as_tensor(
            joint_reorder_indices(source_names, robot.joint_names),
            device=robot.device, dtype=torch.long,
        )
        expected_q = torch.as_tensor(
            [row["physical_state"]["joint_position_rad"] for row in rows],
            device=robot.device, dtype=robot.data.joint_pos.dtype,
        )[:, reorder]
        expected_dq = torch.as_tensor(
            [row["physical_state"]["joint_velocity_radps"] for row in rows],
            device=robot.device, dtype=robot.data.joint_vel.dtype,
        )[:, reorder]
        expected_position = env.scene.env_origins[selected] + torch.as_tensor(
            [row["physical_state"]["root_position_m"] for row in rows],
            device=robot.device, dtype=robot.data.root_pos_w.dtype,
        )
        xyzw = torch.as_tensor(
            [row["physical_state"]["root_quaternion_xyzw"] for row in rows],
            device=robot.device, dtype=robot.data.root_quat_w.dtype,
        )
        expected_quat = xyzw[:, [3, 0, 1, 2]]
        expected_root_velocity = torch.as_tensor(
            [
                row["physical_state"]["root_linear_velocity_world_mps"]
                + row["physical_state"]["root_angular_velocity_world_radps"]
                for row in rows
            ], device=robot.device, dtype=robot.data.root_vel_w.dtype,
        )
        expected_obs = torch.as_tensor(
            [row["observation_93d"] for row in rows], device=env.device, dtype=torch.float32
        )
        expected_previous = torch.as_tensor(
            [row["actual_previous_action_input"] for row in rows], device=env.device, dtype=torch.float32
        )
        expected_issued = torch.as_tensor(
            [row["actual_issued_action"] for row in rows], device=env.device, dtype=torch.float32
        )
        expected_clock = torch.as_tensor(
            [round(float(row["stop_elapsed_s"]) / 0.02) for row in rows],
            device=env.episode_length_buf.device, dtype=env.episode_length_buf.dtype,
        )
        expected_command = torch.as_tensor(
            [row["command_velocity_mps_radps"] for row in rows], device=env.device, dtype=torch.float32
        )
        action_term = env.action_manager.get_term("joint_pos")
        policy_obs = observation["policy"]
        quat_alignment = torch.abs(
            torch.sum(robot.data.root_quat_w[selected] * expected_quat, dim=-1)
        )
        errors = {
            "joint_position_rad": max_error(robot.data.joint_pos[selected], expected_q),
            "joint_velocity_radps": max_error(robot.data.joint_vel[selected], expected_dq),
            "root_position_m": max_error(robot.data.root_pos_w[selected], expected_position),
            "root_quaternion_one_minus_abs_dot": float(torch.max(1.0 - quat_alignment).item()),
            "root_linear_velocity_mps": max_error(robot.data.root_lin_vel_w[selected], expected_root_velocity[:, :3]),
            "root_angular_velocity_radps": max_error(robot.data.root_ang_vel_w[selected], expected_root_velocity[:, 3:]),
            "policy_observation_93d": max_error(policy_obs[selected], expected_obs),
            "policy_previous_action": max_error(policy_obs[selected, 74:89], expected_previous),
            "manager_action": max_error(env.action_manager.action[selected], expected_previous),
            "manager_previous_action": max_error(env.action_manager.prev_action[selected], expected_previous),
            "term_effective_issued_action": max_error(action_term.combined_normalized_actions[selected], expected_issued),
            "episode_clock_steps": float(torch.max(torch.abs(env.episode_length_buf[selected] - expected_clock)).item()),
            "policy_command": max_error(policy_obs[selected, 9:12], expected_command),
        }
        controller_exact = actor_source_exact = source_ref_exact = True
        hash_errors = []
        for env_id, manifest_index, row in zip(
            selected.detach().cpu().tolist(), indices, rows
        ):
            controller = env._x2_phase20_controller_state_by_env[int(env_id)]
            actor_source = env._x2_phase20_actor_source_by_env[int(env_id)]
            if canonical_sha256(controller) != row["controller_state_sha256"]:
                controller_exact = False
                hash_errors.append({"env_id": env_id, "field": "controller"})
            if canonical_sha256(actor_source) != row["actor_observation_state_sha256"]:
                actor_source_exact = False
                hash_errors.append({"env_id": env_id, "field": "actor_source"})
            expected_ref = {
                "manifest_index": manifest_index,
                "snapshot_sha256": row["snapshot_sha256"],
                "controller_state_sha256": row["controller_state_sha256"],
                "actor_observation_state_sha256": row["actor_observation_state_sha256"],
            }
            if env._x2_phase20_source_ref_by_env[int(env_id)] != expected_ref:
                source_ref_exact = False
        physical_keys = (
            "joint_position_rad", "joint_velocity_radps", "root_position_m",
            "root_quaternion_one_minus_abs_dot", "root_linear_velocity_mps",
            "root_angular_velocity_radps",
        )
        logical_keys = (
            "policy_observation_93d", "policy_previous_action", "manager_action",
            "manager_previous_action", "term_effective_issued_action",
            "episode_clock_steps", "policy_command",
        )
        finite = bool(torch.isfinite(policy_obs[selected]).all())
        passed = bool(
            policy_obs[selected].shape == (args.num_envs, 93)
            and all(errors[key] <= 1.0e-5 for key in physical_keys)
            and all(errors[key] <= 1.0e-6 for key in logical_keys)
            and controller_exact and actor_source_exact and source_ref_exact and finite
        )
        report.update({
            "result": {
                "selected_env_count": int(len(selected)),
                "snapshot_max_errors": errors,
                "physical_tolerance": 1.0e-5,
                "actor_and_logical_tolerance": 1.0e-6,
                "controller_registry_hash_exact": controller_exact,
                "actor_source_registry_hash_exact": actor_source_exact,
                "source_reference_exact": source_ref_exact,
                "hash_errors": hash_errors,
                "finite": finite,
                "source_snapshot_sha256": [row["snapshot_sha256"] for row in rows],
            },
            "passed": passed,
            "continuation_boundary": (
                "Exact sensor/gait overrides are valid only for reset-return and expire after the first simulator step. "
                "This is not a proof of a vendor ROS mid-event suffix or future closed-loop success."
            ),
        })
        if not passed:
            raise RuntimeError("Phase20 zero-update state-role injection failed")
    except BaseException as exc:
        report.setdefault("passed", False)
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
