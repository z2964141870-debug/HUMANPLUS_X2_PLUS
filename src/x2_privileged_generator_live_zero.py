"""Zero-update trainer proving the native reset survives live manager reset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


class PrivilegedGeneratorLiveZeroTrainer:
    """Trainer-shaped probe: one reset, zero env steps, zero optimizer."""

    def __init__(self, *, env: Any, log_dir: str | Path, **_: Any) -> None:
        self.env = env
        self.log_dir = Path(log_dir)
        self.optimizer = None

    def train(self) -> None:
        observations = self.env.reset(flatten_dict_obs=True)
        raw_env = self.env.env
        debug = getattr(raw_env, "_x2_native_generator_reset_last", None)
        if not debug or debug.get("strict_no_op"):
            raise RuntimeError("native generator live reset did not execute")
        selected = debug["selected_env_ids"]
        if selected.numel() != raw_env.num_envs:
            raise RuntimeError("live zero requires every environment to use the native seed")
        robot = raw_env.scene["robot"]
        # Consecutive resets happen at the same simulation timestamp.  Isaac's
        # lazy ``robot.data`` cache can therefore retain the previous reset,
        # while the authoritative PhysX tensors already contain the write.
        # Gate on PhysX and keep cache disagreement as a diagnostic only.
        physx_pose_xyzw = robot.root_physx_view.get_root_transforms()[selected]
        physx_pose_wxyz = torch.cat(
            (physx_pose_xyzw[:, :3], physx_pose_xyzw[:, 6:7], physx_pose_xyzw[:, 3:6]),
            dim=-1,
        )
        physx_velocity = robot.root_physx_view.get_root_velocities()[selected]
        physx_joint_pos = robot.root_physx_view.get_dof_positions()[selected]
        physx_joint_vel = robot.root_physx_view.get_dof_velocities()[selected]
        errors = {
            "physx_root_pose_max_abs": float((physx_pose_wxyz - debug["root_pose"]).abs().max()),
            "physx_root_velocity_max_abs": float(
                (physx_velocity - debug["root_velocity"]).abs().max()
            ),
            "physx_joint_pos_max_abs": float(
                (physx_joint_pos - debug["joint_pos"]).abs().max()
            ),
            "physx_joint_vel_max_abs": float(
                (physx_joint_vel - debug["joint_vel"]).abs().max()
            ),
            "cache_root_pose_max_abs": float((
                torch.cat((robot.data.root_pos_w[selected], robot.data.root_quat_w[selected]), dim=-1)
                - debug["root_pose"]
            ).abs().max()),
            "cache_root_velocity_max_abs": float((
                torch.cat((robot.data.root_lin_vel_w[selected], robot.data.root_ang_vel_w[selected]), dim=-1)
                - debug["root_velocity"]
            ).abs().max()),
            "cache_joint_pos_max_abs": float(
                (robot.data.joint_pos[selected] - debug["joint_pos"]).abs().max()
            ),
            "cache_joint_vel_max_abs": float(
                (robot.data.joint_vel[selected] - debug["joint_vel"]).abs().max()
            ),
        }
        checks = {
            "physx_root_pose": errors["physx_root_pose_max_abs"] <= 1.0e-6,
            "physx_root_velocity": errors["physx_root_velocity_max_abs"] <= 1.0e-6,
            "physx_joint_pos": errors["physx_joint_pos_max_abs"] <= 1.0e-6,
            "physx_joint_vel": errors["physx_joint_vel_max_abs"] <= 1.0e-6,
            "no_contact_labels_written": debug["contact_labels_written"] is False,
            "post_manager_finalizer_applied": debug.get("finalizer_applied") is True,
            "observations_finite": all(
                bool(torch.isfinite(value).all())
                for value in observations.values() if torch.is_tensor(value)
            ),
        }
        action_term = raw_env.action_manager.get_term("joint_pos")
        tokenizer = raw_env.observation_manager.compute_group("tokenizer", update_history=False)
        future = tokenizer.get("command_multi_future_nonflat") if isinstance(tokenizer, dict) else None
        checks.update({
            "action_is_wbt29": len(action_term._joint_names) == 29,
            "future_is_10x58": torch.is_tensor(future) and list(future.shape[-2:]) == [10, 58],
        })
        passed = all(bool(value) for value in checks.values())
        report = {
            "schema": "x2_privileged_generator_live_zero_v1",
            "checks": checks,
            "errors": errors,
            "runtime": {
                "env_instances": int(raw_env.num_envs),
                "environment_resets": 1,
                "environment_control_steps": 0,
                "optimizer_instances": 0,
                "optimizer_steps": 0,
                "checkpoints_created": 0,
                "seed_path": debug["seed_path"],
                "seed_sha256": debug["seed_sha256"],
                "frame_indices": debug["frame_indices"].detach().cpu().tolist(),
                "soft_position_overshoot_max_rad": debug[
                    "soft_position_overshoot_max_rad"
                ],
                "hard_position_projection_applied": debug[
                    "hard_position_projection_applied"
                ],
                "post_manager_finalizer_applied": debug.get("finalizer_applied", False),
            },
            "decision": {
                "live_native_reset_passed": passed,
                "training_unlocked": False,
                "result": "PASS_LIVE_ZERO_ONLY" if passed else "FAIL_LIVE_ZERO",
            },
            "boundary": "One Isaac reset and fixed observations only; no physics step, reward, PPO, or optimizer claim.",
        }
        output = Path(os.environ["X2_PRIVILEGED_GENERATOR_LIVE_ZERO_OUTPUT"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "decision": report["decision"]}))
        if not passed:
            raise RuntimeError("privileged generator live zero failed")
