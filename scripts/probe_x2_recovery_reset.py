#!/usr/bin/env python3
"""Zero-update IsaacLab smoke for the isolated X2 recovery reset event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--expected-sha256", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=47)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402

from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from official_x2.recovery_reset_curriculum import (  # noqa: E402
    audit_recovery_dataset,
    joint_reorder_indices,
    load_recovery_dataset,
    reset_from_recovery_dataset,
)


def main() -> None:
    if args.num_envs < 2:
        raise ValueError("--num-envs must be at least 2 for balanced labels")
    dataset = args.dataset.expanduser().resolve()
    audit = audit_recovery_dataset(dataset, args.expected_sha256)
    arrays = load_recovery_dataset(str(dataset), args.expected_sha256)

    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    gait_template = Path(
        "/home/humanplus/x2_teleop_final/x2_sonic/data/processed/"
        "x2_official_forward_gait_phase_template_15dof.npz"
    )
    if not gait_template.is_file():
        raise FileNotFoundError(gait_template)
    cfg.actions.joint_pos.template_path = str(gait_template)
    # Match the stand_backend authority contract used by the source actor and
    # by the template file (twice the static foundation action scale).
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
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
        func=reset_from_recovery_dataset,
        mode="reset",
        params={
            "dataset_path": str(dataset),
            "recovery_fraction": 1.0,
            "sampling_mode": "balanced",
            "expected_sha256": args.expected_sha256,
            "asset_name": "robot",
        },
    )

    env = None
    report = {
        "stage": "stage336_recovery_reset_zero_update_smoke",
        "hypothesis": (
            "Stage335 official-MuJoCo stop-boundary physical states can initialize the isolated "
            "IsaacLab stand backend without changing the actor, optimizer, or reference."
        ),
        "intervention": "final reset event; recovery_fraction=1; balanced eventual-pass/fail sampling",
        "control": "the inherited reset_base/reset_robot_joints remain unchanged and run first",
        "training_updates": 0,
        "dataset_audit": audit.__dict__,
    }
    try:
        env = ManagerBasedRLEnv(cfg=cfg)
        obs, _ = env.reset(seed=args.seed)
        debug = env._x2_recovery_reset_last
        selected = debug["selected_env_ids"]
        sample_ids = debug["sample_indices"]
        robot = env.scene["robot"]
        sample_cpu = sample_ids.detach().cpu().numpy()
        reorder = joint_reorder_indices(arrays["joint_names"].tolist(), robot.joint_names)
        reorder_t = torch.as_tensor(reorder, device=robot.device, dtype=torch.long)
        expected_q = robot.data.default_joint_pos[selected] + torch.as_tensor(
            arrays["joint_pos_rel_rad"][sample_cpu],
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )[:, reorder_t]
        limits = robot.data.soft_joint_pos_limits[selected]
        expected_q = torch.clamp(expected_q, limits[..., 0], limits[..., 1])
        expected_z = torch.as_tensor(
            arrays["root_z_m"][sample_cpu],
            device=robot.device,
            dtype=robot.data.root_pos_w.dtype,
        ) + env.scene.env_origins[selected, 2]
        labels = arrays["eventual_pass"][sample_cpu]
        policy_obs = obs["policy"]

        q_error = float(torch.max(torch.abs(robot.data.joint_pos[selected] - expected_q)).item())
        z_error = float(torch.max(torch.abs(robot.data.root_pos_w[selected, 2] - expected_z)).item())
        finite = bool(
            torch.isfinite(robot.data.joint_pos[selected]).all()
            and torch.isfinite(robot.data.root_state_w[selected]).all()
            and torch.isfinite(policy_obs).all()
        )

        zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        _, _, terminated, truncated, _ = env.step(zero_action)
        immediate_survivors = int((~(terminated | truncated)).sum().item())
        result = {
            "selected_env_count": int(len(selected)),
            "sample_indices": sample_ids.detach().cpu().tolist(),
            "eventual_pass_samples": int(labels.sum()),
            "eventual_fail_samples": int((~labels).sum()),
            "policy_observation_shape": list(policy_obs.shape),
            "action_dim": int(env.action_manager.total_action_dim),
            "joint_state_max_abs_error_rad": q_error,
            "root_z_max_abs_error_m": z_error,
            "finite_state_and_observation": finite,
            "one_zero_action_step_survivors": immediate_survivors,
            "one_zero_action_step_total": int(env.num_envs),
        }
        passed = bool(
            len(selected) == env.num_envs
            and abs(int(labels.sum()) - int((~labels).sum())) <= 1
            and policy_obs.shape == (env.num_envs, 93)
            and env.action_manager.total_action_dim == 15
            and q_error <= 1.0e-5
            and z_error <= 1.0e-5
            and finite
            and immediate_survivors == env.num_envs
        )
        report.update(
            {
                "result": result,
                "passed": passed,
                "conclusion": (
                    "physical recovery-reset injection is ready for a <=5-update isolated smoke"
                    if passed
                    else "reset injection is not ready for PPO; stop before training"
                ),
            }
        )
        if not passed:
            raise RuntimeError(report["conclusion"])
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
