#!/usr/bin/env python3
"""Zero-update IsaacLab probe for the full Stage335 snapshot contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--expected-sha256", required=True)
parser.add_argument("--source-report", type=Path, required=True)
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
    audit_stateful_recovery_sidecar,
    joint_reorder_indices,
    load_recovery_dataset,
    load_stateful_recovery_sidecar,
    finalize_stateful_recovery,
    reset_from_recovery_dataset,
)
from official_x2.stateful_recovery_isaac import (  # noqa: E402
    StatefulRecoveryRLEnv,
    configure_stateful_recovery_cfg,
)


_RESET_BOUNDARY_CAPTURE: dict[str, dict[str, torch.Tensor]] = {}


def _capture_joint_velocity_boundary(env, label: str) -> None:
    """Capture both Isaac's cache and the PhysX tensor at one reset boundary."""
    debug = getattr(env, "_x2_recovery_reset_last", None)
    if not debug or "selected_env_ids" not in debug:
        return
    selected = debug["selected_env_ids"]
    if len(selected) == 0:
        return
    robot = env.scene["robot"]
    _RESET_BOUNDARY_CAPTURE[label] = {
        "selected_env_ids": selected.detach().cpu().clone(),
        "sample_indices": debug["sample_indices"].detach().cpu().clone(),
        "cached_joint_vel": robot.data.joint_vel[selected].detach().cpu().clone(),
        "physx_joint_vel": robot.root_physx_view.get_dof_velocities()[selected]
        .detach()
        .cpu()
        .clone(),
        "requested_joint_velocity_abs_max_radps": torch.tensor(
            float(debug.get("requested_joint_velocity_abs_max_radps", float("nan")))
        ),
    }


def _diagnostic_reset_from_recovery_dataset(
    env,
    env_ids,
    dataset_path: str,
    recovery_fraction: float,
    sampling_mode: str = "balanced",
    expected_sha256: str | None = None,
    asset_name: str = "robot",
    stateful_source_report: str | None = None,
    stateful_source_report_sha256: str | None = None,
):
    """Production reset plus a read-only capture immediately after the write."""
    reset_from_recovery_dataset(
        env,
        env_ids,
        dataset_path=dataset_path,
        recovery_fraction=recovery_fraction,
        sampling_mode=sampling_mode,
        expected_sha256=expected_sha256,
        asset_name=asset_name,
        stateful_source_report=stateful_source_report,
        stateful_source_report_sha256=stateful_source_report_sha256,
    )
    _capture_joint_velocity_boundary(env, "event_after_write")


class _BoundaryDiagnosticStatefulRecoveryRLEnv(StatefulRecoveryRLEnv):
    """Expose reset boundaries without changing their production ordering."""

    def _reset_idx(self, env_ids):
        # Spell out StatefulRecoveryRLEnv's two calls so the diagnostic can
        # distinguish manager-reset effects from the later reset forward().
        ManagerBasedRLEnv._reset_idx(self, env_ids)
        _capture_joint_velocity_boundary(self, "after_managers_before_finalizer")
        finalize_stateful_recovery(self)
        _capture_joint_velocity_boundary(self, "after_finalizer_before_forward")


def main() -> None:
    dataset = args.dataset.expanduser().resolve()
    source_report = args.source_report.expanduser().resolve()
    physical_audit = audit_recovery_dataset(dataset, args.expected_sha256)
    stateful_audit = audit_stateful_recovery_sidecar(
        dataset, source_report, args.expected_sha256
    )
    arrays = load_recovery_dataset(str(dataset), args.expected_sha256)
    sidecar = load_stateful_recovery_sidecar(
        str(dataset), str(source_report), args.expected_sha256
    )

    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    gait_template = Path(
        "/home/humanplus/x2_teleop_final/x2_sonic/data/processed/"
        "x2_official_forward_gait_phase_template_15dof.npz"
    )
    if not gait_template.is_file():
        raise FileNotFoundError(gait_template)
    cfg.actions.joint_pos.template_path = str(gait_template)
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    configure_stateful_recovery_cfg(cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    # The generic 0.9 soft-limit contraction is invalid for X2's asymmetric
    # shoulder-roll ranges: it excludes the official/default zero pose itself
    # (left lower becomes +0.0917, right upper -0.0917).  Stateful replay uses
    # the official URDF/MJCF hard range without that artificial contraction.
    cfg.scene.robot.soft_joint_pos_limit_factor = 1.0
    cfg.scene.robot.actuators["feet"].stiffness = 40.0
    cfg.scene.robot.actuators["feet"].damping = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # These baseline values are overwritten only for selected RSI envs by the
    # post-manager finalizer.  Non-selected/fraction=0 behavior stays exact.
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.events.recovery_state_reset = EventTermCfg(
        func=_diagnostic_reset_from_recovery_dataset,
        mode="reset",
        params={
            "dataset_path": str(dataset),
            "recovery_fraction": 1.0,
            "sampling_mode": "balanced",
            "expected_sha256": args.expected_sha256,
            "asset_name": "robot",
            "stateful_source_report": str(source_report),
        },
    )

    env = None
    report = {
        "stage": "x2_recovery_phase4_stateful_zero_update",
        "hypothesis": (
            "restoring the physical snapshot alone is insufficient; the post-manager reset hook "
            "must also reconstruct clock/action/phase/command snapshot buffers"
        ),
        "intervention": "fraction=1 balanced stateful Stage335 reset; no policy update",
        "control": "fraction=0 pure tests are strict no-op; inherited reset path remains unchanged",
        "training_updates": 0,
        "physical_audit": physical_audit.__dict__,
        "stateful_audit": stateful_audit.__dict__,
    }
    try:
        env = _BoundaryDiagnosticStatefulRecoveryRLEnv(cfg=cfg)
        obs, _ = env.reset(seed=args.seed)
        debug = env._x2_recovery_reset_last
        if not debug.get("stateful_finalized", False):
            raise RuntimeError("post-manager stateful finalizer did not run")
        selected = debug["selected_env_ids"]
        sample_ids = debug["sample_indices"]
        sample_cpu = sample_ids.detach().cpu().numpy()
        robot = env.scene["robot"]
        reorder = joint_reorder_indices(arrays["joint_names"].tolist(), robot.joint_names)
        reorder_t = torch.as_tensor(reorder, device=robot.device, dtype=torch.long)
        inverse_reorder_t = torch.as_tensor(
            np.argsort(reorder), device=robot.device, dtype=torch.long
        )
        raw_expected_q = robot.data.default_joint_pos[selected] + torch.as_tensor(
            arrays["joint_pos_rel_rad"][sample_cpu],
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )[:, reorder_t]
        limits = robot.data.soft_joint_pos_limits[selected]
        expected_q = torch.clamp(raw_expected_q, limits[..., 0], limits[..., 1])
        raw_expected_dq = torch.as_tensor(
            arrays["joint_vel_radps"][sample_cpu],
            device=robot.device,
            dtype=robot.data.joint_vel.dtype,
        )[:, reorder_t]
        velocity_limits = robot.data.soft_joint_vel_limits[selected]
        expected_dq = torch.clamp(raw_expected_dq, -velocity_limits, velocity_limits)

        def boundary_velocity_errors() -> dict[str, dict[str, float | bool]]:
            result: dict[str, dict[str, float | bool]] = {}
            for label, capture in _RESET_BOUNDARY_CAPTURE.items():
                same_samples = bool(
                    torch.equal(capture["sample_indices"], sample_ids.detach().cpu())
                )
                if not same_samples:
                    result[label] = {"same_samples": False}
                    continue
                expected_cpu = expected_dq.detach().cpu()
                result[label] = {
                    "same_samples": True,
                    "requested_abs_max_radps": float(
                        capture["requested_joint_velocity_abs_max_radps"].item()
                    ),
                    "expected_abs_max_radps": float(torch.max(torch.abs(expected_cpu)).item()),
                    "cached_abs_max_radps": float(
                        torch.max(torch.abs(capture["cached_joint_vel"])).item()
                    ),
                    "physx_abs_max_radps": float(
                        torch.max(torch.abs(capture["physx_joint_vel"])).item()
                    ),
                    "cached_vs_expected_max_radps": float(
                        torch.max(torch.abs(capture["cached_joint_vel"] - expected_cpu)).item()
                    ),
                    "physx_vs_expected_max_radps": float(
                        torch.max(torch.abs(capture["physx_joint_vel"] - expected_cpu)).item()
                    ),
                    "cache_vs_physx_max_radps": float(
                        torch.max(
                            torch.abs(
                                capture["cached_joint_vel"] - capture["physx_joint_vel"]
                            )
                        ).item()
                    ),
                }
            return result
        expected_root_pos = env.scene.env_origins[selected].clone()
        expected_root_pos[:, 2] += torch.as_tensor(
            arrays["root_z_m"][sample_cpu],
            device=robot.device,
            dtype=robot.data.root_pos_w.dtype,
        )
        roll = torch.as_tensor(
            arrays["root_roll_rad"][sample_cpu], device=robot.device, dtype=torch.float32
        )
        pitch = torch.as_tensor(
            arrays["root_pitch_rad"][sample_cpu], device=robot.device, dtype=torch.float32
        )
        yaw = torch.as_tensor(
            arrays["root_yaw_rad"][sample_cpu], device=robot.device, dtype=torch.float32
        )
        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        expected_root_quat = torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=-1,
        )

        def body_to_world(v_body):
            q_xyz = expected_root_quat[:, 1:4]
            cross = 2.0 * torch.linalg.cross(q_xyz, v_body, dim=-1)
            return (
                v_body
                + expected_root_quat[:, 0:1] * cross
                + torch.linalg.cross(q_xyz, cross, dim=-1)
            )

        expected_root_lin_vel = body_to_world(
            torch.as_tensor(
                arrays["base_lin_vel_body_mps"][sample_cpu],
                device=robot.device,
                dtype=torch.float32,
            )
        )
        expected_root_ang_vel = body_to_world(
            torch.as_tensor(
                arrays["base_ang_vel_body_radps"][sample_cpu],
                device=robot.device,
                dtype=torch.float32,
            )
        )
        expected_previous = torch.as_tensor(
            arrays["previous_action"][sample_cpu], device=env.device, dtype=torch.float32
        )
        expected_gait = torch.as_tensor(
            arrays["gait_phase"][sample_cpu], device=env.device, dtype=torch.float32
        )
        expected_command = torch.as_tensor(
            sidecar["command_velocity_mps_radps"][sample_cpu],
            device=env.device,
            dtype=torch.float32,
        )
        expected_clock = torch.as_tensor(
            sidecar["episode_clock_steps"][sample_cpu],
            device=env.device,
            dtype=env.episode_length_buf.dtype,
        )
        expected_issued = torch.as_tensor(
            sidecar["previous_issued_action"][sample_cpu],
            device=env.device,
            dtype=torch.float32,
        )
        policy_obs = obs["policy"]
        action_term = env.action_manager.get_term("joint_pos")

        # Audit all immutable source rows against the target simulator's
        # limits.  Runtime equality after clamping is not source-snapshot
        # equality, so any nonzero projection keeps the training gate locked.
        all_joint_rel = torch.as_tensor(
            arrays["joint_pos_rel_rad"], device=robot.device, dtype=torch.float32
        )[:, reorder_t]
        all_joint_vel = torch.as_tensor(
            arrays["joint_vel_radps"], device=robot.device, dtype=torch.float32
        )[:, reorder_t]
        default0 = robot.data.default_joint_pos[0].unsqueeze(0)
        limits0 = robot.data.soft_joint_pos_limits[0].unsqueeze(0)
        velocity_limits0 = robot.data.soft_joint_vel_limits[0].unsqueeze(0)
        all_q_raw = default0 + all_joint_rel
        all_q_projected = torch.clamp(all_q_raw, limits0[..., 0], limits0[..., 1])
        all_dq_projected = torch.clamp(
            all_joint_vel, -velocity_limits0, velocity_limits0
        )
        all_q_projection = torch.abs(all_q_projected - all_q_raw)
        all_dq_projection = torch.abs(all_dq_projected - all_joint_vel)
        joint_projection_by_name = {
            name: {
                "position_changed_state_count": int(
                    (all_q_projection[:, index] > 1.0e-7).sum().item()
                ),
                "position_max_change_rad": float(
                    all_q_projection[:, index].max().item()
                ),
                "isaac_default_rad": float(default0[0, index].item()),
                "isaac_soft_limit_rad": [
                    float(limits0[0, index, 0].item()),
                    float(limits0[0, index, 1].item()),
                ],
                "source_absolute_range_rad": [
                    float(all_q_raw[:, index].min().item()),
                    float(all_q_raw[:, index].max().item()),
                ],
                "velocity_changed_state_count": int(
                    (all_dq_projection[:, index] > 1.0e-7).sum().item()
                ),
                "velocity_max_change_radps": float(
                    all_dq_projection[:, index].max().item()
                ),
            }
            for index, name in enumerate(robot.joint_names)
            if bool(
                torch.any(all_q_projection[:, index] > 1.0e-7)
                or torch.any(all_dq_projection[:, index] > 1.0e-7)
            )
        }

        def max_error(actual, expected):
            return float(torch.max(torch.abs(actual - expected)).item())

        quaternion_alignment = torch.abs(
            torch.sum(robot.data.root_quat_w[selected] * expected_root_quat, dim=-1)
        )
        errors = {
            "joint_pos_rad": max_error(robot.data.joint_pos[selected], expected_q),
            "joint_vel_radps": max_error(robot.data.joint_vel[selected], expected_dq),
            "root_position_m": max_error(robot.data.root_pos_w[selected], expected_root_pos),
            "root_quaternion_one_minus_abs_dot": float(
                torch.max(1.0 - quaternion_alignment).item()
            ),
            "root_linear_velocity_mps": max_error(
                robot.data.root_lin_vel_w[selected], expected_root_lin_vel
            ),
            "root_angular_velocity_radps": max_error(
                robot.data.root_ang_vel_w[selected], expected_root_ang_vel
            ),
            "policy_base_lin_vel": max_error(
                policy_obs[selected, 0:3],
                torch.as_tensor(
                    arrays["base_lin_vel_body_mps"][sample_cpu],
                    device=env.device,
                    dtype=torch.float32,
                ),
            ),
            "policy_base_ang_vel": max_error(
                policy_obs[selected, 3:6],
                torch.as_tensor(
                    arrays["base_ang_vel_body_radps"][sample_cpu],
                    device=env.device,
                    dtype=torch.float32,
                ),
            ),
            "policy_projected_gravity": max_error(
                policy_obs[selected, 6:9],
                torch.as_tensor(
                    arrays["projected_gravity"][sample_cpu],
                    device=env.device,
                    dtype=torch.float32,
                ),
            ),
            "episode_clock_steps": int(
                torch.max(torch.abs(env.episode_length_buf[selected] - expected_clock)).item()
            ),
            "policy_command": max_error(policy_obs[selected, 9:12], expected_command),
            "policy_joint_pos": max_error(
                policy_obs[selected, 12:43],
                torch.as_tensor(
                    arrays["joint_pos_rel_rad"][sample_cpu],
                    device=env.device,
                    dtype=torch.float32,
                ),
            ),
            "policy_joint_vel": max_error(
                policy_obs[selected, 43:74],
                torch.as_tensor(
                    arrays["joint_vel_radps"][sample_cpu],
                    device=env.device,
                    dtype=torch.float32,
                ),
            ),
            "policy_previous_action": max_error(policy_obs[selected, 74:89], expected_previous),
            "policy_gait_phase": max_error(policy_obs[selected, 89:93], expected_gait),
            "manager_action": max_error(env.action_manager.action[selected], expected_previous),
            "manager_prev_action": max_error(env.action_manager.prev_action[selected], expected_previous),
            "term_raw_action": max_error(action_term.raw_actions[selected], expected_previous),
            "term_effective_action": max_error(
                action_term.combined_normalized_actions[selected], expected_issued
            ),
            "policy_joint_pos_after_projection": max_error(
                policy_obs[selected, 12:43],
                (expected_q - robot.data.default_joint_pos[selected])[:, inverse_reorder_t],
            ),
            "policy_joint_vel_after_projection": max_error(
                policy_obs[selected, 43:74], expected_dq[:, inverse_reorder_t]
            ),
        }
        runtime_dq_error_by_joint = {
            name: float(
                torch.max(
                    torch.abs(
                        robot.data.joint_vel[selected, index] - expected_dq[:, index]
                    )
                ).item()
            )
            for index, name in enumerate(robot.joint_names)
            if float(
                torch.max(
                    torch.abs(
                        robot.data.joint_vel[selected, index] - expected_dq[:, index]
                    )
                ).item()
            )
            > 1.0e-5
        }
        finite = bool(torch.isfinite(policy_obs).all())
        zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        _, _, terminated, truncated, _ = env.step(zero_action)
        survivors = int((~(terminated | truncated)).sum().item())
        passed = bool(
            policy_obs.shape == (env.num_envs, 93)
            and env.action_manager.total_action_dim == 15
            and all(value <= 1.0e-5 for value in errors.values())
            and not bool(torch.any(all_q_projection > 1.0e-7))
            and not bool(torch.any(all_dq_projection > 1.0e-7))
            and finite
            and survivors == env.num_envs
        )
        report.update({
            "result": {
                "selected_env_count": int(len(selected)),
                "policy_observation_shape": list(policy_obs.shape),
                "action_dim": int(env.action_manager.total_action_dim),
                "snapshot_max_errors": errors,
                "source_to_isaac_projection": {
                    "all_state_count": int(len(arrays["root_z_m"])),
                    "joint_position_changed_state_count": int(
                        torch.any(all_q_projection > 1.0e-7, dim=1).sum().item()
                    ),
                    "joint_velocity_changed_state_count": int(
                        torch.any(all_dq_projection > 1.0e-7, dim=1).sum().item()
                    ),
                    "joint_position_max_change_rad": float(all_q_projection.max().item()),
                    "joint_velocity_max_change_radps": float(all_dq_projection.max().item()),
                    "selected_joint_position_max_change_rad": max_error(
                        expected_q, raw_expected_q
                    ),
                    "selected_joint_velocity_max_change_radps": max_error(
                        expected_dq, raw_expected_dq
                    ),
                    "affected_joints": joint_projection_by_name,
                    "runtime_joint_velocity_error_by_joint_radps": runtime_dq_error_by_joint,
                },
                "joint_velocity_reset_boundaries": boundary_velocity_errors(),
                "finite": finite,
                "low_command_force_moving_samples": int(
                    np.sum(
                        sidecar["force_moving"][sample_cpu]
                        & (
                            np.linalg.norm(
                                sidecar["command_velocity_mps_radps"][sample_cpu, :2], axis=1
                            ) <= 0.1
                        )
                    )
                ),
                "one_zero_action_step_survivors": survivors,
                "one_zero_action_step_total": int(env.num_envs),
            },
            "passed": passed,
            "continuation_boundary": (
                "logical buffers and projected physical state are exact, but any nonzero "
                "source-to-Isaac limit projection invalidates source-snapshot equivalence; "
                "future brake feedback is also not encoded, so this cannot authorize PPO"
            ),
        })
        if not passed:
            raise RuntimeError("stateful reset snapshot probe failed")
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
