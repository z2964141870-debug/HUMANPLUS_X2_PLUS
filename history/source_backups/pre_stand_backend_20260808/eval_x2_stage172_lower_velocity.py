#!/usr/bin/env python3
"""Evaluate a Stage172 checkpoint with fixed planar velocity commands."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--command_vx", type=float, default=0.30)
parser.add_argument(
    "--command_wz",
    type=float,
    default=0.0,
    help="Fixed yaw-rate command used when --heading_hold is disabled.",
)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stage_label", type=str, default="stage172")
parser.add_argument(
    "--action_scale_multiplier",
    type=float,
    default=1.0,
    help="Evaluation-only multiplier for the configured 15-DOF action scales.",
)
parser.add_argument(
    "--sagittal_scale_multiplier",
    type=float,
    default=1.0,
    help="Additional evaluation-only multiplier for hip/knee/ankle pitch scales.",
)
parser.add_argument(
    "--observation_profile",
    choices=(
        "deployable",
        "teacher",
        "teacher_phase",
        "teacher_phase_template",
        "teacher_phase_template_response_history",
    ),
    default="deployable",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("docs/reports/x2_stage172_fixed_vx_eval.json"),
)
parser.add_argument(
    "--trace_output",
    type=Path,
    default=None,
    help=(
        "Optional compressed per-step diagnostic trace. This records only "
        "simulator/controller quantities already used by the evaluator and "
        "does not change policy inputs or physics."
    ),
)
parser.add_argument(
    "--gait_template",
    type=Path,
    default=None,
    help="Optional zero-mean X2-native 15-DOF phase template (.npz).",
)
parser.add_argument(
    "--gait_template_scale",
    type=float,
    default=0.0,
    help="Training-free template amplitude used only for Stage182 feasibility scans.",
)
parser.add_argument(
    "--heading_hold",
    action="store_true",
    help="Convert heading error to a deployable yaw-rate command during fixed-forward playback.",
)
parser.add_argument("--heading_stiffness", type=float, default=1.0)
parser.add_argument("--heading_rate_limit", type=float, default=0.5)
parser.add_argument(
    "--heading_integral_gain",
    type=float,
    default=0.0,
    help="Optional bounded integral gain for slow heading-bias rejection.",
)
parser.add_argument(
    "--heading_integral_rate_limit",
    type=float,
    default=0.05,
    help="Maximum absolute yaw-rate contribution from the integral branch.",
)
parser.add_argument(
    "--yaw_basis_checkpoint",
    type=Path,
    default=None,
    help=(
        "Optional frozen yaw-specialist checkpoint. Its central action difference "
        "at +/- yaw_basis_command_abs is added as a zero-at-zero command residual."
    ),
)
parser.add_argument("--yaw_basis_command_abs", type=float, default=0.2)
parser.add_argument("--yaw_basis_scale", type=float, default=1.0)
parser.add_argument(
    "--actuator_domain",
    choices=("ideal", "filter", "delay", "noise"),
    default="ideal",
    help="Cumulative actuator/observation attribution domain.",
)
parser.add_argument(
    "--self_collisions",
    choices=("on", "off"),
    default="on",
    help="Evaluation collision flag; must match training unless used for attribution.",
)
parser.add_argument(
    "--collision_profile",
    choices=("mesh", "sole_spheres", "sole12", "twist_training"),
    default="mesh",
    help="X2 URDF collision substrate used for the physical rollout.",
)
parser.add_argument("--actuator_response_strength", type=float, default=1.0)
parser.add_argument(
    "--actuator_filter_strength",
    type=float,
    default=None,
    help="Optional independent low-pass strength; defaults to actuator_response_strength.",
)
parser.add_argument(
    "--actuator_delay_strength",
    type=float,
    default=None,
    help="Optional independent delay strength; defaults to 0 for filter domain and response strength otherwise.",
)
parser.add_argument(
    "--report_symmetry",
    action="store_true",
    help="Report actor left/right equivariance on the visited phase-teacher states.",
)
parser.add_argument(
    "--actor_symmetry_projection_alpha",
    type=float,
    default=0.0,
    help=(
        "Training-free blend toward the actor's left/right equivariant projection. "
        "Zero preserves the checkpoint exactly."
    ),
)
parser.add_argument(
    "--actor_symmetry_projection_mask",
    choices=("yaw_roll", "all"),
    default="yaw_roll",
    help="Anatomical actor channels eligible for symmetry projection.",
)
parser.add_argument(
    "--response_adapter_mask",
    choices=("all", "no_yaw", "sagittal"),
    default="all",
    help="Evaluation/training-matched anatomical mask for a response-history checkpoint.",
)
parser.add_argument(
    "--zero_delay_groups",
    type=str,
    default="",
    help=(
        "Comma-separated actuator groups whose explicit DelayBuffer lag is set "
        "to zero after reset. Diagnostic attribution only; filtering is unchanged."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import numpy as np  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from gear_sonic.envs.x2_velocity import (  # noqa: E402
    X2LowerVelocityFlatEnvCfg_PLAY,
    X2LowerVelocityTeacherFlatEnvCfg_PLAY,
    X2LowerVelocityTeacherPhaseFlatEnvCfg_PLAY,
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg_PLAY,
    X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg_PLAY,
)
from gear_sonic.envs.x2_velocity.flat_env_cfg import (  # noqa: E402
    X2_GAIT_CYCLE_TIME_S,
    X2_GAIT_DOUBLE_SUPPORT_FRACTION,
)
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)
from gear_sonic.envs.x2_velocity.gait import (  # noqa: E402
    gait_phase_and_desired_contacts,
    gait_transition_grace_mask,
)
from gear_sonic.envs.x2_velocity.rsl_rl_ppo_cfg import (  # noqa: E402
    X2LowerVelocityFlatPPORunnerCfg,
    X2ResponseHistoryActorCriticCfg,
)
from gear_sonic.envs.x2_velocity.response_history_actor_critic import (  # noqa: E402
    ResponseHistoryActorCritic,
    response_adapter_output_mask,
)
from gear_sonic.envs.x2_velocity.symmetry import (  # noqa: E402
    compute_x2_left_right_symmetric_states,
    mirror_x2_lower_actions_left_right,
)
from gear_sonic.envs.x2_velocity.heading_command import (  # noqa: E402
    heading_pi_velocity_cfg,
)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(-1)[0].item())


def _ground_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    matrix = env.scene[sensor_name].data.force_matrix_w
    if matrix is None:
        raise RuntimeError(f"{sensor_name} has no ground-filtered force matrix")
    return matrix[..., 2].abs().reshape(matrix.shape[0], -1).amax(dim=-1)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return wrapped yaw for an IsaacLab wxyz quaternion tensor."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.command_vx < 0.0:
        raise ValueError("Stage172 forward gate requires --command_vx >= 0")
    if args.action_scale_multiplier <= 0.0:
        raise ValueError("--action_scale_multiplier must be positive")
    if args.sagittal_scale_multiplier <= 0.0:
        raise ValueError("--sagittal_scale_multiplier must be positive")
    if args.gait_template_scale < 0.0:
        raise ValueError("--gait_template_scale must be non-negative")
    if args.heading_stiffness <= 0.0 or args.heading_rate_limit <= 0.0:
        raise ValueError("heading stiffness and rate limit must be positive")
    if args.heading_integral_gain < 0.0:
        raise ValueError("heading integral gain must be non-negative")
    if args.heading_integral_gain > 0.0 and args.heading_integral_rate_limit <= 0.0:
        raise ValueError("positive heading integral gain requires a positive rate limit")
    if args.heading_integral_gain > 0.0 and not args.heading_hold:
        raise ValueError("heading integral gain requires --heading_hold")
    if not 0.0 <= args.actuator_response_strength <= 1.0:
        raise ValueError("--actuator_response_strength must be in [0, 1]")
    if args.actuator_filter_strength is not None:
        if not 0.0 <= args.actuator_filter_strength <= 1.0:
            raise ValueError("--actuator_filter_strength must be in [0, 1]")
        if args.actuator_domain == "ideal":
            raise ValueError("--actuator_filter_strength requires a non-ideal actuator domain")
    if args.actuator_delay_strength is not None:
        if not 0.0 <= args.actuator_delay_strength <= 1.0:
            raise ValueError("--actuator_delay_strength must be in [0, 1]")
        if args.actuator_domain == "ideal":
            raise ValueError("--actuator_delay_strength requires a non-ideal actuator domain")
    if args.gait_template_scale > 0.0 and args.gait_template is None:
        raise ValueError("positive --gait_template_scale requires --gait_template")
    if not 0.0 <= args.actor_symmetry_projection_alpha <= 1.0:
        raise ValueError("actor symmetry projection alpha must be in [0, 1]")
    if args.yaw_basis_checkpoint is not None:
        if args.observation_profile not in {
            "teacher",
            "teacher_phase",
            "teacher_phase_template",
        }:
            raise ValueError("yaw-basis adapter requires a teacher observation profile")
        if args.yaw_basis_command_abs <= 0.0:
            raise ValueError("--yaw_basis_command_abs must be positive")
        if args.yaw_basis_scale < 0.0:
            raise ValueError("--yaw_basis_scale must be non-negative")
        if args.actor_symmetry_projection_alpha > 0.0:
            raise ValueError("yaw-basis adapter cannot be combined with actor symmetry projection")

    response_history_profile = (
        args.observation_profile == "teacher_phase_template_response_history"
    )
    internal_template = args.observation_profile in {
        "teacher_phase_template",
        "teacher_phase_template_response_history",
    }
    if args.report_symmetry and not internal_template:
        raise ValueError("--report_symmetry requires --observation_profile teacher_phase_template")
    if (
        args.actor_symmetry_projection_alpha > 0.0
        and args.observation_profile != "teacher_phase_template"
    ):
        raise ValueError(
            "actor symmetry projection is defined only for the 93-D teacher_phase_template profile"
        )
    if response_history_profile:
        cfg = X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg_PLAY()
        if args.gait_template is not None:
            cfg.actions.joint_pos.template_path = str(args.gait_template.expanduser().resolve())
        cfg.actions.joint_pos.template_scale = args.gait_template_scale
    elif internal_template:
        cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg_PLAY()
        if args.gait_template is not None:
            cfg.actions.joint_pos.template_path = str(args.gait_template.expanduser().resolve())
        cfg.actions.joint_pos.template_scale = args.gait_template_scale
    elif args.observation_profile == "teacher_phase":
        cfg = X2LowerVelocityTeacherPhaseFlatEnvCfg_PLAY()
    elif args.observation_profile == "teacher":
        cfg = X2LowerVelocityTeacherFlatEnvCfg_PLAY()
    else:
        cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.actions.joint_pos.scale = {
        pattern: scale * args.action_scale_multiplier
        for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    for pattern in (".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"):
        cfg.actions.joint_pos.scale[pattern] *= args.sagittal_scale_multiplier
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE[
        args.collision_profile
    ]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = (
        args.self_collisions == "on"
    )
    cfg.episode_length_s = args.steps * cfg.decimation * cfg.sim.dt + 2.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.command_vx, args.command_vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    if args.heading_hold:
        cfg.commands.base_velocity.heading_command = True
        cfg.commands.base_velocity.rel_heading_envs = 1.0
        cfg.commands.base_velocity.heading_control_stiffness = args.heading_stiffness
        cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (
            -args.heading_rate_limit,
            args.heading_rate_limit,
        )
        # Always use the X2 wrapper, even for P-only evaluation.  Besides the
        # optional integral branch it synchronizes the heading-derived yaw
        # command during reset, before the first policy observation.  Stock
        # UniformVelocityCommand exposes one sampled direct-yaw command for a
        # single first action, making an otherwise inactive rate range change
        # the entire sensitive closed-loop trajectory.
        cfg.commands.base_velocity = heading_pi_velocity_cfg(
            cfg.commands.base_velocity,
            heading_control_stiffness=args.heading_stiffness,
            heading_integral_gain=args.heading_integral_gain,
            heading_integral_rate_limit=args.heading_integral_rate_limit,
        )
    else:
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_heading_envs = 0.0
        cfg.commands.base_velocity.ranges.ang_vel_z = (args.command_wz, args.command_wz)

    actuator_response_cfg = None
    if args.actuator_domain != "ideal":
        filter_strength = (
            args.actuator_response_strength
            if args.actuator_filter_strength is None
            else args.actuator_filter_strength
        )
        delay_strength = (
            (0.0 if args.actuator_domain == "filter" else args.actuator_response_strength)
            if args.actuator_delay_strength is None
            else args.actuator_delay_strength
        )
        actuator_response_cfg = {
            "enabled": True,
            "profile": "session03_session04_group",
            "randomize": False,
            "strength": args.actuator_response_strength,
            "filter_strength": filter_strength,
            "delay_strength": delay_strength,
            "include_ideal_endpoint": False,
            "ideal_env_fraction": 0.0,
        }
    actuator_response_report = _apply_x2_actuator_response(
        cfg.scene.robot,
        actuator_response_cfg,
        physics_dt_sec=cfg.sim.dt,
    )
    cfg.observations.policy.enable_corruption = args.actuator_domain == "noise"

    agent_cfg = X2LowerVelocityFlatPPORunnerCfg()
    if response_history_profile:
        import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

        rsl_on_policy_runner.ResponseHistoryActorCritic = ResponseHistoryActorCritic
        agent_cfg.policy = X2ResponseHistoryActorCriticCfg()
        agent_cfg.policy.response_adapter_output_mask = response_adapter_output_mask(
            args.response_adapter_mask
        )
    agent_cfg.device = args.device

    zero_delay_groups = tuple(
        group.strip() for group in args.zero_delay_groups.split(",") if group.strip()
    )
    unknown_zero_delay_groups = sorted(
        set(zero_delay_groups) - {"legs", "feet", "waist", "arms", "head"}
    )
    if unknown_zero_delay_groups:
        raise ValueError(f"unknown zero-delay actuator groups: {unknown_zero_delay_groups}")

    env = None
    wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=cfg)
        robot = env.scene["robot"]
        velocity_command_term = env.command_manager.get_term("base_velocity")
        for group_name in zero_delay_groups:
            actuator = robot.actuators.get(group_name)
            if actuator is None:
                raise ValueError(f"actuator group not found for zero-delay attribution: {group_name}")
            buffers = [
                getattr(actuator, "positions_delay_buffer", None),
                getattr(actuator, "velocities_delay_buffer", None),
                getattr(actuator, "efforts_delay_buffer", None),
            ]
            if any(buffer is None for buffer in buffers):
                raise ValueError(f"actuator group has no delay buffers: {group_name}")
            zero_lags = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)
            for buffer in buffers:
                buffer.set_time_lag(zero_lags)
                buffer.reset()
            # DelayedImplicitActuator deliberately bypasses DelayBuffer only
            # when the configured capacity is zero.  Merely assigning a
            # zero-lag row in a nonzero-capacity buffer is not guaranteed to
            # reproduce the stock direct path, so the attribution must switch
            # that branch as well.
            actuator.cfg.min_delay = 0
            actuator.cfg.max_delay = 0
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # Evaluation never consumes optimizer state.  This also permits the
        # zero-initialized response adapter to load an adapter-free Stage192
        # checkpoint for an exact inheritance check.
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        policy_module = runner.alg.policy
        yaw_basis_checkpoint = None
        yaw_basis_runner = None
        yaw_basis_policy = None
        if args.yaw_basis_checkpoint is not None:
            yaw_basis_checkpoint = args.yaw_basis_checkpoint.expanduser().resolve()
            if not yaw_basis_checkpoint.is_file():
                raise FileNotFoundError(yaw_basis_checkpoint)
            yaw_basis_runner = OnPolicyRunner(
                wrapped,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
            yaw_basis_runner.load(str(yaw_basis_checkpoint), load_optimizer=False)
            yaw_basis_policy = yaw_basis_runner.get_inference_policy(device=env.device)

        obs = wrapped.get_observations()
        foot_ids, foot_names = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        action_term = env.action_manager._terms["joint_pos"]
        selected_joint_names = [robot.joint_names[index] for index in action_term._joint_ids]
        if args.actor_symmetry_projection_mask == "all":
            actor_symmetry_projection_mask = torch.ones(
                len(selected_joint_names), device=env.device
            )
        else:
            actor_symmetry_projection_mask = torch.tensor(
                [
                    float("_roll_" in name or "_yaw_" in name)
                    for name in selected_joint_names
                ],
                device=env.device,
            )
        template_q = None
        template_action_scale = None
        template_period_s = None
        template_path = None
        if args.gait_template is not None:
            template_path = args.gait_template.expanduser().resolve()
            if not template_path.is_file():
                raise FileNotFoundError(template_path)
            template_npz = np.load(template_path, allow_pickle=False)
            template_names = template_npz["joint_names_15"].tolist()
            if template_names != selected_joint_names:
                raise RuntimeError(
                    f"gait template joint order mismatch: {template_names} != {selected_joint_names}"
                )
            template_q = torch.as_tensor(
                template_npz["q_cycle_zero_mean_rad"], device=env.device, dtype=torch.float32
            )
            template_action_scale = torch.as_tensor(
                template_npz["action_scale_rad"], device=env.device, dtype=torch.float32
            )
            template_period_s = float(np.asarray(template_npz["period_s"]).item())
            if template_q.ndim != 2 or template_q.shape[1] != 15:
                raise RuntimeError(f"invalid gait template shape: {tuple(template_q.shape)}")
            if template_period_s <= 0.0 or torch.any(template_action_scale <= 0.0):
                raise RuntimeError("invalid gait template period/action scale")
        initial_root = robot.data.root_pos_w[0].clone()
        initial_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0]).clone()
        initial_foot_z = robot.data.body_pos_w[0, foot_ids, 2].clone()

        completed_steps = 0
        first_done_step = None
        reward_sum = 0.0
        root_z_min = float("inf")
        root_z_max = -float("inf")
        tilt_max = 0.0
        lateral_drift_max = 0.0
        forward_displacement = 0.0
        body_vx_sum = 0.0
        body_vx_sq_error_sum = 0.0
        yaw_rate_abs_sum = 0.0
        yaw_rate_sum = 0.0
        yaw_rate_sq_sum = 0.0
        heading_deviation_max = 0.0
        heading_deviation_final = 0.0
        heading_command_abs_max = 0.0
        heading_integral_contribution_abs_max = 0.0
        action_abs_max = 0.0
        action_saturated_count = 0
        action_element_count = 0
        action_abs_max_by_joint = torch.zeros(15, device=env.device)
        action_abs_sum_by_joint = torch.zeros(15, device=env.device)
        action_saturated_count_by_joint = torch.zeros(15, dtype=torch.int64, device=env.device)
        template_bias_abs_max = 0.0
        template_induced_clip_count = 0
        template_element_count = 0
        joint_excursion_max_by_joint = torch.zeros(15, device=env.device)
        held_target_error_max = 0.0
        left_contact_count = 0
        right_contact_count = 0
        single_support_count = 0
        double_support_count = 0
        flight_count = 0
        contact_switches = [0, 0]
        previous_contact = None
        hysteresis_contact = None
        hysteresis_switches = [0, 0]
        hysteresis_mode_run_steps = [0, 0]
        hysteresis_mode_durations = [[], []]
        phase_valid_samples = 0
        phase_contact_match_count = [0, 0]
        phase_both_contact_match_count = 0
        foot_lift_max = [0.0, 0.0]
        left_force_min = float("inf")
        right_force_min = float("inf")
        left_force_max = 0.0
        right_force_max = 0.0
        symmetry_squared_error_sum = 0.0
        symmetry_component_count = 0
        symmetry_max_abs_error = 0.0
        symmetry_squared_error_by_joint = torch.zeros(15, device=env.device)
        symmetry_sample_count = 0
        symmetry_projection_squared_sum = 0.0
        symmetry_projection_component_count = 0
        symmetry_projection_max_abs = 0.0
        yaw_basis_residual_squared_sum = 0.0
        yaw_basis_residual_component_count = 0
        yaw_basis_residual_max_abs = 0.0
        trace = {
            "time_s": [],
            "root_pos_w_m": [],
            "root_quat_wxyz": [],
            "root_lin_vel_b_mps": [],
            "root_ang_vel_w_radps": [],
            "heading_delta_rad": [],
            "base_tilt_rad": [],
            "policy_action": [],
            "unprojected_policy_action": [],
            "actor_symmetry_projection_delta": [],
            "yaw_basis_residual": [],
            "base_actor_action": [],
            "response_adapter_residual": [],
            "combined_action": [],
            "joint_pos_rad": [],
            "joint_vel_radps": [],
            "joint_target_rad": [],
            "all_joint_pos_rad": [],
            "all_joint_target_rad": [],
            "foot_z_w_m": [],
            "foot_lift_m": [],
            "foot_force_n": [],
            "foot_contact": [],
            "desired_contact": [],
            "gait_phase": [],
            "velocity_command_b": [],
            "heading_error_rad": [],
            "heading_error_integral_rad_s": [],
            "heading_integral_contribution_radps": [],
        }
        traced_actuator_groups = tuple(
            group_name
            for group_name in ("legs", "feet", "waist", "arms")
            if hasattr(robot.actuators.get(group_name), "_filtered_joint_positions")
        )
        for group_name in traced_actuator_groups:
            trace[f"{group_name}_filtered_target_rad"] = []

        for step in range(args.steps):
            with torch.inference_mode():
                unprojected_actions = None
                projection_delta = None
                symmetry_diagnostics = (
                    args.report_symmetry
                    or args.actor_symmetry_projection_alpha > 0.0
                )
                if symmetry_diagnostics:
                    symmetric_obs, _ = compute_x2_left_right_symmetric_states(
                        env=wrapped, obs=obs, actions=None
                    )
                    batch_size = obs.batch_size[0]
                    # Keep the executed action on the exact same batch-size-1
                    # inference path as the physical baseline.  Concatenating
                    # original+mirror changes GEMM numerics enough to select a
                    # different gait attractor in this highly sensitive policy.
                    actions = policy(obs)
                    unprojected_actions = actions.clone()
                    expected_mirrored_actions = mirror_x2_lower_actions_left_right(actions)
                    mirrored_prediction = policy(symmetric_obs[batch_size:])
                    symmetry_error = mirrored_prediction - expected_mirrored_actions
                    if args.report_symmetry:
                        symmetry_squared_error_sum += float(symmetry_error.square().sum().item())
                        symmetry_component_count += int(symmetry_error.numel())
                        symmetry_max_abs_error = max(
                            symmetry_max_abs_error, _scalar(symmetry_error.abs().max())
                        )
                        symmetry_squared_error_by_joint += symmetry_error.square().sum(dim=0)
                        symmetry_sample_count += int(symmetry_error.shape[0])
                    if args.actor_symmetry_projection_alpha > 0.0:
                        mirrored_back = mirror_x2_lower_actions_left_right(
                            mirrored_prediction
                        )
                        equivariant_projection = 0.5 * (actions + mirrored_back)
                        projection_delta = (
                            args.actor_symmetry_projection_alpha
                            * actor_symmetry_projection_mask.unsqueeze(0)
                            * (equivariant_projection - actions)
                        )
                        actions = actions + projection_delta
                        symmetry_projection_squared_sum += float(
                            projection_delta.square().sum().item()
                        )
                        symmetry_projection_component_count += int(
                            projection_delta.numel()
                        )
                        symmetry_projection_max_abs = max(
                            symmetry_projection_max_abs,
                            _scalar(projection_delta.abs().max()),
                        )
                else:
                    actions = policy(obs)
                    unprojected_actions = actions.clone()
                yaw_basis_residual = torch.zeros_like(actions)
                if yaw_basis_policy is not None:
                    positive_obs = obs.clone()
                    negative_obs = obs.clone()
                    positive_obs["policy"][..., 11] = args.yaw_basis_command_abs
                    negative_obs["policy"][..., 11] = -args.yaw_basis_command_abs
                    positive_action = yaw_basis_policy(positive_obs)
                    negative_action = yaw_basis_policy(negative_obs)
                    central_yaw_basis = 0.5 * (positive_action - negative_action)
                    command_fraction = torch.clamp(
                        velocity_command_term.command[:, 2:3]
                        / args.yaw_basis_command_abs,
                        min=-1.0,
                        max=1.0,
                    )
                    yaw_basis_residual = (
                        args.yaw_basis_scale * command_fraction * central_yaw_basis
                    )
                    actions = actions + yaw_basis_residual
                    yaw_basis_residual_squared_sum += float(
                        yaw_basis_residual.square().sum().item()
                    )
                    yaw_basis_residual_component_count += int(yaw_basis_residual.numel())
                    yaw_basis_residual_max_abs = max(
                        yaw_basis_residual_max_abs,
                        _scalar(yaw_basis_residual.abs().max()),
                    )
                if projection_delta is None:
                    projection_delta = torch.zeros_like(actions)
                policy_action_for_trace = actions[0].detach().clone()
                if response_history_profile:
                    actor_obs = policy_module.get_actor_obs(obs)
                    actor_obs = policy_module.actor_obs_normalizer(actor_obs)
                    base_action_for_trace = policy_module.actor(
                        actor_obs[..., : policy_module.base_actor_obs_dim]
                    )[0].detach().clone()
                else:
                    base_action_for_trace = policy_action_for_trace.clone()
                response_residual_for_trace = (
                    policy_action_for_trace - base_action_for_trace
                )
                if template_q is not None and args.gait_template_scale > 0.0 and not internal_template:
                    phase_value = torch.remainder(
                        env.episode_length_buf.to(torch.float32) * env.step_dt / template_period_s,
                        1.0,
                    )
                    bins = template_q.shape[0]
                    phase_position = phase_value * bins - 0.5
                    lower_unwrapped = torch.floor(phase_position)
                    blend = phase_position - lower_unwrapped
                    lower = lower_unwrapped.to(torch.int64) % bins
                    upper = (lower + 1) % bins
                    q_bias = (
                        (1.0 - blend).unsqueeze(-1) * template_q[lower]
                        + blend.unsqueeze(-1) * template_q[upper]
                    )
                    normalized_bias = args.gait_template_scale * q_bias / template_action_scale
                    wrapper_clipped_actions = torch.clamp(actions, -1.0, 1.0)
                    policy_preexisting_clip = wrapper_clipped_actions.abs() >= 0.999
                    preclip_actions = wrapper_clipped_actions + normalized_bias
                    template_bias_abs_max = max(
                        template_bias_abs_max, _scalar(normalized_bias.abs().max())
                    )
                    template_induced_clip_count += int(
                        ((preclip_actions.abs() > 1.0) & ~policy_preexisting_clip).sum().item()
                    )
                    template_element_count += int(preclip_actions.numel())
                    actions = torch.clamp(preclip_actions, -1.0, 1.0)
                obs, reward, dones, _ = wrapped.step(actions)

            metric_actions = actions
            if internal_template:
                metric_actions = action_term.combined_normalized_actions
                normalized_bias = action_term.normalized_template_bias
                preclip_actions = action_term.preclip_combined_actions
                policy_preexisting_clip = action_term.raw_actions.abs() >= 0.999
                template_bias_abs_max = max(
                    template_bias_abs_max, _scalar(normalized_bias.abs().max())
                )
                template_induced_clip_count += int(
                    ((preclip_actions.abs() > 1.0) & ~policy_preexisting_clip).sum().item()
                )
                template_element_count += int(preclip_actions.numel())

            # IsaacLab resets done environments inside step(); exclude the
            # reset pose from trajectory metrics and stop this episode.
            if bool(dones[0].item()):
                first_done_step = step
                policy_module.reset(dones)
                break

            completed_steps = step + 1
            reward_sum += _scalar(reward)
            root = robot.data.root_pos_w[0]
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[0, 2], -1.0, 1.0))
            left_force = _ground_force(env, "left_foot_ground_contact")
            right_force = _ground_force(env, "right_foot_ground_contact")
            contact = torch.stack((left_force > 10.0, right_force > 10.0), dim=-1)[0]
            foot_lift = robot.data.body_pos_w[0, foot_ids, 2] - initial_foot_z

            root_z_min = min(root_z_min, _scalar(root[2]))
            root_z_max = max(root_z_max, _scalar(root[2]))
            tilt_max = max(tilt_max, _scalar(tilt))
            forward_displacement = _scalar(root[0] - initial_root[0])
            lateral_drift_max = max(lateral_drift_max, abs(_scalar(root[1] - initial_root[1])))
            body_vx = _scalar(robot.data.root_lin_vel_b[0, 0])
            body_vx_sum += body_vx
            body_vx_sq_error_sum += (body_vx - args.command_vx) ** 2
            yaw_rate = _scalar(robot.data.root_ang_vel_w[0, 2])
            yaw_rate_sum += yaw_rate
            yaw_rate_abs_sum += abs(yaw_rate)
            yaw_rate_sq_sum += yaw_rate**2
            yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
            yaw_delta = torch.atan2(torch.sin(yaw - initial_yaw), torch.cos(yaw - initial_yaw))
            heading_deviation_final = _scalar(yaw_delta)
            heading_deviation_max = max(heading_deviation_max, abs(heading_deviation_final))
            heading_command_abs_max = max(
                heading_command_abs_max,
                abs(_scalar(velocity_command_term.command[0, 2])),
            )
            integral_contribution = getattr(
                velocity_command_term, "heading_integral_contribution", None
            )
            if integral_contribution is not None:
                heading_integral_contribution_abs_max = max(
                    heading_integral_contribution_abs_max,
                    abs(_scalar(integral_contribution[0])),
                )
            action_abs_max = max(action_abs_max, _scalar(metric_actions.abs().max()))
            action_saturated_count += int((metric_actions.abs() >= 0.999).sum().item())
            action_element_count += int(metric_actions.numel())
            action_abs = metric_actions[0].abs()
            action_abs_max_by_joint = torch.maximum(action_abs_max_by_joint, action_abs)
            action_abs_sum_by_joint += action_abs
            action_saturated_count_by_joint += (action_abs >= 0.999).to(torch.int64)
            selected_ids = action_term._joint_ids
            joint_excursion = torch.abs(
                robot.data.joint_pos[0, selected_ids]
                - robot.data.default_joint_pos[0, selected_ids]
            )
            joint_excursion_max_by_joint = torch.maximum(joint_excursion_max_by_joint, joint_excursion)
            held_ids = action_term._held_joint_ids
            held_error = torch.abs(
                robot.data.joint_pos_target[:, held_ids]
                - robot.data.default_joint_pos[:, held_ids]
            )
            held_target_error_max = max(held_target_error_max, _scalar(held_error.max()))
            left_contact_count += int(contact[0].item())
            right_contact_count += int(contact[1].item())
            contact_count = int(contact.to(torch.int32).sum().item())
            single_support_count += int(contact_count == 1)
            double_support_count += int(contact_count == 2)
            flight_count += int(contact_count == 0)
            if previous_contact is not None:
                changed = contact != previous_contact
                contact_switches[0] += int(changed[0].item())
                contact_switches[1] += int(changed[1].item())
            previous_contact = contact.clone()
            force_pair = torch.stack((left_force, right_force), dim=-1)[0]
            if hysteresis_contact is None:
                hysteresis_contact = force_pair > 30.0
                hysteresis_mode_run_steps = [1, 1]
            else:
                next_hysteresis = torch.where(
                    hysteresis_contact,
                    force_pair >= 5.0,
                    force_pair > 30.0,
                )
                changed_hysteresis = next_hysteresis != hysteresis_contact
                for foot_index in range(2):
                    if bool(changed_hysteresis[foot_index].item()):
                        hysteresis_switches[foot_index] += 1
                        hysteresis_mode_durations[foot_index].append(
                            hysteresis_mode_run_steps[foot_index] * env.step_dt
                        )
                        hysteresis_mode_run_steps[foot_index] = 1
                    else:
                        hysteresis_mode_run_steps[foot_index] += 1
                hysteresis_contact = next_hysteresis
            phase, desired_contact, moving = gait_phase_and_desired_contacts(
                env,
                command_name="base_velocity",
                cycle_time_s=X2_GAIT_CYCLE_TIME_S,
                double_support_fraction=X2_GAIT_DOUBLE_SUPPORT_FRACTION,
            )
            phase_outside_grace = gait_transition_grace_mask(
                phase,
                cycle_time_s=X2_GAIT_CYCLE_TIME_S,
                double_support_fraction=X2_GAIT_DOUBLE_SUPPORT_FRACTION,
                transition_grace_s=0.04,
            )
            if bool((moving[0] & phase_outside_grace[0]).item()):
                phase_match = hysteresis_contact == desired_contact[0]
                phase_valid_samples += 1
                phase_contact_match_count[0] += int(phase_match[0].item())
                phase_contact_match_count[1] += int(phase_match[1].item())
                phase_both_contact_match_count += int(phase_match.all().item())
            foot_lift_max[0] = max(foot_lift_max[0], _scalar(foot_lift[0]))
            foot_lift_max[1] = max(foot_lift_max[1], _scalar(foot_lift[1]))
            left_force_min = min(left_force_min, _scalar(left_force))
            right_force_min = min(right_force_min, _scalar(right_force))
            left_force_max = max(left_force_max, _scalar(left_force))
            right_force_max = max(right_force_max, _scalar(right_force))
            if args.trace_output is not None:
                trace["time_s"].append((step + 1) * env.step_dt)
                trace["root_pos_w_m"].append(root.detach().cpu().numpy())
                trace["root_quat_wxyz"].append(
                    robot.data.root_quat_w[0].detach().cpu().numpy()
                )
                trace["root_lin_vel_b_mps"].append(
                    robot.data.root_lin_vel_b[0].detach().cpu().numpy()
                )
                trace["root_ang_vel_w_radps"].append(
                    robot.data.root_ang_vel_w[0].detach().cpu().numpy()
                )
                trace["heading_delta_rad"].append(heading_deviation_final)
                trace["base_tilt_rad"].append(_scalar(tilt))
                trace["policy_action"].append(policy_action_for_trace.cpu().numpy())
                trace["unprojected_policy_action"].append(
                    unprojected_actions[0].detach().cpu().numpy()
                )
                trace["actor_symmetry_projection_delta"].append(
                    projection_delta[0].detach().cpu().numpy()
                )
                trace["yaw_basis_residual"].append(
                    yaw_basis_residual[0].detach().cpu().numpy()
                )
                trace["base_actor_action"].append(base_action_for_trace.cpu().numpy())
                trace["response_adapter_residual"].append(
                    response_residual_for_trace.cpu().numpy()
                )
                trace["combined_action"].append(metric_actions[0].detach().cpu().numpy())
                trace["joint_pos_rad"].append(
                    robot.data.joint_pos[0, selected_ids].detach().cpu().numpy()
                )
                trace["joint_vel_radps"].append(
                    robot.data.joint_vel[0, selected_ids].detach().cpu().numpy()
                )
                trace["joint_target_rad"].append(
                    robot.data.joint_pos_target[0, selected_ids].detach().cpu().numpy()
                )
                trace["all_joint_pos_rad"].append(
                    robot.data.joint_pos[0].detach().cpu().numpy()
                )
                trace["all_joint_target_rad"].append(
                    robot.data.joint_pos_target[0].detach().cpu().numpy()
                )
                trace["foot_z_w_m"].append(
                    robot.data.body_pos_w[0, foot_ids, 2].detach().cpu().numpy()
                )
                trace["foot_lift_m"].append(foot_lift.detach().cpu().numpy())
                trace["foot_force_n"].append(force_pair.detach().cpu().numpy())
                trace["foot_contact"].append(hysteresis_contact.detach().cpu().numpy())
                trace["desired_contact"].append(desired_contact[0].detach().cpu().numpy())
                trace["gait_phase"].append(_scalar(phase[0]))
                trace["velocity_command_b"].append(
                    velocity_command_term.command[0].detach().cpu().numpy()
                )
                heading_error_state = getattr(
                    velocity_command_term, "heading_error", None
                )
                heading_integral_state = getattr(
                    velocity_command_term, "heading_error_integral", None
                )
                heading_integral_output = getattr(
                    velocity_command_term, "heading_integral_contribution", None
                )
                trace["heading_error_rad"].append(
                    _scalar(heading_error_state[0])
                    if heading_error_state is not None
                    else -heading_deviation_final
                )
                trace["heading_error_integral_rad_s"].append(
                    _scalar(heading_integral_state[0])
                    if heading_integral_state is not None
                    else 0.0
                )
                trace["heading_integral_contribution_radps"].append(
                    _scalar(heading_integral_output[0])
                    if heading_integral_output is not None
                    else 0.0
                )
                for group_name in traced_actuator_groups:
                    trace[f"{group_name}_filtered_target_rad"].append(
                        robot.actuators[group_name]
                        ._filtered_joint_positions[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

        samples = max(completed_steps, 1)
        for foot_index in range(2):
            if hysteresis_mode_run_steps[foot_index] > 0:
                hysteresis_mode_durations[foot_index].append(
                    hysteresis_mode_run_steps[foot_index] * env.step_dt
                )
        hysteresis_duration_median = []
        hysteresis_short_fraction = []
        for durations in hysteresis_mode_durations:
            if durations:
                duration_tensor = torch.tensor(durations)
                hysteresis_duration_median.append(float(duration_tensor.median().item()))
                hysteresis_short_fraction.append(
                    float((duration_tensor < 0.10).to(torch.float32).mean().item())
                )
            else:
                hysteresis_duration_median.append(0.0)
                hysteresis_short_fraction.append(0.0)
        duration_s = completed_steps * env.step_dt
        expected_displacement = args.command_vx * duration_s
        progress_ratio = (
            forward_displacement / expected_displacement
            if expected_displacement > 1.0e-6
            else None
        )
        trace_path = None
        if args.trace_output is not None:
            trace_path = args.trace_output.expanduser().resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_arrays = {
                name: np.asarray(values)
                for name, values in trace.items()
            }
            trace_arrays["action_joint_order"] = np.asarray(selected_joint_names)
            trace_arrays["all_joint_order"] = np.asarray(robot.joint_names)
            for group_name in traced_actuator_groups:
                trace_arrays[f"{group_name}_joint_order"] = np.asarray(
                    robot.actuators[group_name].joint_names
                )
            np.savez_compressed(trace_path, **trace_arrays)
        result = {
            "stage": args.stage_label,
            "probe": "fixed_forward_checkpoint_physics",
            "seed": args.seed,
            "checkpoint": str(checkpoint),
            "trace_output": str(trace_path) if trace_path is not None else None,
            "actor_observation_profile": args.observation_profile,
            "response_adapter_mask": args.response_adapter_mask if response_history_profile else None,
            "action_scale_multiplier": args.action_scale_multiplier,
            "sagittal_scale_multiplier": args.sagittal_scale_multiplier,
            "gait_template": str(template_path) if template_path is not None else None,
            "gait_template_scale": args.gait_template_scale,
            "heading_hold": args.heading_hold,
            "heading_reset_command_synchronized": args.heading_hold,
            "heading_stiffness": args.heading_stiffness if args.heading_hold else None,
            "heading_rate_limit_radps": args.heading_rate_limit if args.heading_hold else None,
            "heading_integral_gain": args.heading_integral_gain if args.heading_hold else None,
            "heading_integral_rate_limit_radps": (
                args.heading_integral_rate_limit if args.heading_integral_gain > 0.0 else None
            ),
            "actuator_domain": args.actuator_domain,
            "self_collisions": args.self_collisions,
            "collision_profile": args.collision_profile,
            "collision_asset_path": cfg.scene.robot.spawn.asset_path,
            "actuator_response_strength": args.actuator_response_strength,
            "actuator_filter_strength": (
                args.actuator_response_strength
                if args.actuator_filter_strength is None
                else args.actuator_filter_strength
            ),
            "actuator_delay_strength": (
                (0.0 if args.actuator_domain == "filter" else args.actuator_response_strength)
                if args.actuator_delay_strength is None
                else args.actuator_delay_strength
            ) if args.actuator_domain != "ideal" else 0.0,
            "actuator_response_report": actuator_response_report,
            "counterfactual_zero_delay_groups": list(zero_delay_groups),
            "policy_observation_corruption": cfg.observations.policy.enable_corruption,
            "symmetry_report_enabled": args.report_symmetry,
            "actor_symmetry_projection_alpha": args.actor_symmetry_projection_alpha,
            "actor_symmetry_projection_mask": (
                args.actor_symmetry_projection_mask
                if args.actor_symmetry_projection_alpha > 0.0
                else None
            ),
            "actor_symmetry_projection_rmse": (
                (
                    symmetry_projection_squared_sum
                    / max(symmetry_projection_component_count, 1)
                )
                ** 0.5
                if args.actor_symmetry_projection_alpha > 0.0
                else 0.0
            ),
            "actor_symmetry_projection_max_abs": symmetry_projection_max_abs,
            "yaw_basis_checkpoint": (
                str(yaw_basis_checkpoint) if yaw_basis_checkpoint is not None else None
            ),
            "yaw_basis_command_abs_radps": (
                args.yaw_basis_command_abs if yaw_basis_checkpoint is not None else None
            ),
            "yaw_basis_scale": (
                args.yaw_basis_scale if yaw_basis_checkpoint is not None else None
            ),
            "yaw_basis_residual_rmse": (
                yaw_basis_residual_squared_sum
                / max(yaw_basis_residual_component_count, 1)
            ) ** 0.5,
            "yaw_basis_residual_max_abs": yaw_basis_residual_max_abs,
            "actor_lr_mirror_rmse": (
                (symmetry_squared_error_sum / max(symmetry_component_count, 1)) ** 0.5
                if args.report_symmetry
                else None
            ),
            "actor_lr_mirror_max_abs_error": (
                symmetry_max_abs_error if args.report_symmetry else None
            ),
            "actor_lr_mirror_rmse_by_joint": (
                {
                    name: float(value)
                    for name, value in zip(
                        selected_joint_names,
                        torch.sqrt(
                            symmetry_squared_error_by_joint / max(symmetry_sample_count, 1)
                        ).detach().cpu().tolist(),
                    )
                }
                if args.report_symmetry
                else None
            ),
            "gait_template_bias_abs_max": template_bias_abs_max,
            "gait_template_induced_clip_fraction": template_induced_clip_count / max(template_element_count, 1),
            "command_vx_mps": args.command_vx,
            "command_wz_radps": args.command_wz if not args.heading_hold else None,
            "control_dt_s": env.step_dt,
            "steps_requested": args.steps,
            "steps_completed": completed_steps,
            "duration_s": duration_s,
            "first_done_step": first_done_step,
            "survived_full_horizon": first_done_step is None and completed_steps == args.steps,
            "forward_displacement_m": forward_displacement,
            "expected_forward_displacement_m": expected_displacement,
            "progress_ratio": progress_ratio,
            "lateral_drift_max_m": lateral_drift_max,
            "body_vx_mean_mps": body_vx_sum / samples,
            "body_vx_rmse_mps": (body_vx_sq_error_sum / samples) ** 0.5,
            "yaw_rate_abs_mean_radps": yaw_rate_abs_sum / samples,
            "yaw_rate_mean_radps": yaw_rate_sum / samples,
            "yaw_rate_rms_radps": (yaw_rate_sq_sum / samples) ** 0.5,
            "heading_deviation_final_rad": heading_deviation_final,
            "heading_deviation_abs_max_rad": heading_deviation_max,
            "heading_command_abs_max_radps": heading_command_abs_max,
            "heading_integral_contribution_abs_max_radps": (
                heading_integral_contribution_abs_max
            ),
            "root_z_min_m": root_z_min,
            "root_z_max_m": root_z_max,
            "base_tilt_max_rad": tilt_max,
            "reward_sum": reward_sum,
            "action_abs_max": action_abs_max,
            "action_saturation_fraction": action_saturated_count / max(action_element_count, 1),
            "action_joint_order": selected_joint_names,
            "action_abs_max_by_joint": {
                name: float(value)
                for name, value in zip(selected_joint_names, action_abs_max_by_joint.detach().cpu().tolist())
            },
            "action_abs_mean_by_joint": {
                name: float(value)
                for name, value in zip(
                    selected_joint_names,
                    (action_abs_sum_by_joint / samples).detach().cpu().tolist(),
                )
            },
            "action_saturation_fraction_by_joint": {
                name: float(value)
                for name, value in zip(
                    selected_joint_names,
                    (action_saturated_count_by_joint / samples).detach().cpu().tolist(),
                )
            },
            "joint_excursion_max_rad_by_joint": {
                name: float(value)
                for name, value in zip(selected_joint_names, joint_excursion_max_by_joint.detach().cpu().tolist())
            },
            "held_target_default_error_max_rad": held_target_error_max,
            "foot_body_names": foot_names,
            "left_ground_contact_fraction": left_contact_count / samples,
            "right_ground_contact_fraction": right_contact_count / samples,
            "single_support_fraction": single_support_count / samples,
            "double_support_fraction": double_support_count / samples,
            "flight_fraction": flight_count / samples,
            "left_contact_switch_count": contact_switches[0],
            "right_contact_switch_count": contact_switches[1],
            "contact_hysteresis_enter_n": 30.0,
            "contact_hysteresis_exit_n": 5.0,
            "left_hysteresis_contact_switch_count": hysteresis_switches[0],
            "right_hysteresis_contact_switch_count": hysteresis_switches[1],
            "left_hysteresis_mode_duration_median_s": hysteresis_duration_median[0],
            "right_hysteresis_mode_duration_median_s": hysteresis_duration_median[1],
            "left_hysteresis_short_bout_fraction_lt_0p1s": hysteresis_short_fraction[0],
            "right_hysteresis_short_bout_fraction_lt_0p1s": hysteresis_short_fraction[1],
            "gait_cycle_time_s": X2_GAIT_CYCLE_TIME_S,
            "gait_double_support_fraction": X2_GAIT_DOUBLE_SUPPORT_FRACTION,
            "phase_contact_valid_samples": phase_valid_samples,
            "left_phase_contact_match_fraction": phase_contact_match_count[0] / max(phase_valid_samples, 1),
            "right_phase_contact_match_fraction": phase_contact_match_count[1] / max(phase_valid_samples, 1),
            "both_phase_contact_match_fraction": phase_both_contact_match_count / max(phase_valid_samples, 1),
            "left_foot_lift_max_m": foot_lift_max[0],
            "right_foot_lift_max_m": foot_lift_max[1],
            "left_ground_force_min_n": left_force_min,
            "left_ground_force_max_n": left_force_max,
            "right_ground_force_min_n": right_force_min,
            "right_ground_force_max_n": right_force_max,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        markdown = args.output.with_suffix(".md")
        markdown.write_text(
            "\n".join(
                [
                    f"# {args.stage_label} 固定前进命令物理评估",
                    "",
                    f"- checkpoint：`{checkpoint}`",
                    f"- 生存：`{completed_steps}/{args.steps}` steps，first done=`{first_done_step}`",
                    f"- 前进位移/期望：`{forward_displacement:.4f}/{expected_displacement:.4f} m`，ratio=`{progress_ratio}`",
                    f"- body vx mean/RMSE：`{result['body_vx_mean_mps']:.4f}/{result['body_vx_rmse_mps']:.4f} m/s`",
                    f"- yaw rate mean-abs/RMS：`{result['yaw_rate_abs_mean_radps']:.4f}/{result['yaw_rate_rms_radps']:.4f} rad/s`",
                    f"- heading final/max deviation：`{result['heading_deviation_final_rad']:.4f}/{result['heading_deviation_abs_max_rad']:.4f} rad`",
                    f"- 最大横漂/倾斜：`{lateral_drift_max:.4f} m` / `{tilt_max:.4f} rad`",
                    f"- 单支撑/双支撑/腾空比例：`{result['single_support_fraction']:.3f}/{result['double_support_fraction']:.3f}/{result['flight_fraction']:.3f}`",
                    f"- 左右脚接触切换：`{contact_switches[0]}/{contact_switches[1]}`",
                    f"- 迟滞左右脚切换：`{hysteresis_switches[0]}/{hysteresis_switches[1]}`（30 N on / 5 N off）",
                    f"- 迟滞模式中位持续时间：`{hysteresis_duration_median[0]:.3f}/{hysteresis_duration_median[1]:.3f} s`",
                    f"- phase 左/右/同时接触匹配：`{result['left_phase_contact_match_fraction']:.3f}/{result['right_phase_contact_match_fraction']:.3f}/{result['both_phase_contact_match_fraction']:.3f}`",
                    f"- 左右脚最大抬升：`{foot_lift_max[0]:.4f}/{foot_lift_max[1]:.4f} m`",
                    f"- action 饱和率：`{result['action_saturation_fraction']:.5f}`",
                    f"- action scale 评估倍率：`{args.action_scale_multiplier:.3f}`",
                    f"- sagittal scale 评估倍率：`{args.sagittal_scale_multiplier:.3f}`",
                    f"- gait template/scale：`{template_path}` / `{args.gait_template_scale:.3f}`",
                    f"- template bias max / induced clip：`{template_bias_abs_max:.4f}` / `{result['gait_template_induced_clip_fraction']:.5f}`",
                    f"- actor 左右镜像 RMSE/max：`{result['actor_lr_mirror_rmse']}` / `{result['actor_lr_mirror_max_abs_error']}`",
                    "",
                    "该报告只陈述物理轨迹，不以训练 reward 单独判定会走。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as exc:
        print(f"[Stage172 eval] FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
