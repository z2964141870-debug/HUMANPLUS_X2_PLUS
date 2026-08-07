#!/usr/bin/env python3
"""Train an X2 lower-body branch with an explicit official ankle-PD contract."""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="stage172_smoke")
parser.add_argument(
    "--resume_checkpoint",
    type=Path,
    default=None,
    help="Optional RSL-RL checkpoint whose policy and optimizer are resumed.",
)
parser.add_argument(
    "--target_iteration",
    type=int,
    default=None,
    help="Absolute final iteration index when resuming (for example 1000).",
)
parser.add_argument(
    "--weights_only_resume",
    action="store_true",
    help="Load model weights/iteration but reset optimizer state after an observation-contract expansion.",
)
parser.add_argument(
    "--gait_template",
    type=Path,
    default=None,
    help="Optional X2-native gait template used by the phase-template residual profile.",
)
parser.add_argument(
    "--gait_template_scale",
    type=float,
    default=None,
    help="Override the template amplitude for the phase-template residual profile.",
)
parser.add_argument(
    "--plant_ankle_roll_bias",
    type=float,
    default=0.0,
    help=(
        "Hidden normalized bias applied to both ankle-roll targets in the matched "
        "training plant. Negative values train the actor to learn the positive "
        "official-MuJoCo compensation."
    ),
)
parser.add_argument(
    "--plant_bias_env_fraction",
    type=float,
    default=0.5,
    help="Fraction of environments receiving --plant_ankle_roll_bias; the rest are matched-PD controls.",
)
parser.add_argument(
    "--actuator_domain",
    choices=("ideal", "filter", "delay", "noise"),
    default="ideal",
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
parser.add_argument("--actuator_ideal_fraction", type=float, default=0.0)
parser.add_argument(
    "--official_ankle_pd",
    action="store_true",
    help="Match the official MuJoCo deployment branch at ankle Kp/Kd=40/20.",
)
parser.add_argument(
    "--self_collisions",
    choices=("on", "off"),
    default="on",
    help="Enable or disable articulation self-collisions for the selected collision profile.",
)
parser.add_argument(
    "--collision_profile",
    choices=("mesh", "sole_spheres", "sole12", "twist_training"),
    default="mesh",
    help="X2 URDF collision geometry used consistently during training.",
)
parser.add_argument(
    "--actuator_filter_only_fraction",
    type=float,
    default=0.0,
    help="Fixed response-model fraction using nominal filtering with zero delay.",
)
parser.add_argument("--ideal_heading_stiffness", type=float, default=1.0)
parser.add_argument(
    "--heading_env_fraction",
    type=float,
    default=1.0,
    help="Fraction of envs using heading targets; the remainder sample direct yaw-rate commands.",
)
parser.add_argument(
    "--yaw_command_abs_max",
    type=float,
    default=0.5,
    help="Symmetric yaw-rate command range and heading-controller rate limit.",
)
parser.add_argument(
    "--lateral_command_abs_max",
    type=float,
    default=0.0,
    help="Symmetric body-frame lateral velocity command range used to learn deployable path authority.",
)
parser.add_argument(
    "--heading_error_weight",
    type=float,
    default=0.0,
    help="Non-negative weight magnitude for heading-target-only world heading error.",
)
parser.add_argument(
    "--response_heading_stiffness",
    type=float,
    default=None,
    help="Optional lower outer-loop gain for response-model envs in a mixed-domain run.",
)
parser.add_argument(
    "--mirror_loss_coeff",
    type=float,
    default=0.0,
    help="Optional soft left/right residual-policy mirror loss (phase-template profile only).",
)
parser.add_argument(
    "--actor_anchor_coeff",
    type=float,
    default=0.0,
    help=(
        "Quadratic gradient anchor around the resumed actor. This protects a fixed "
        "known-good policy from cumulative PPO drift while leaving critic/std adaptive."
    ),
)
parser.add_argument(
    "--response_adapter_mask",
    choices=("all", "no_yaw", "sagittal"),
    default="all",
    help="Anatomical output mask for the causal response-history adapter.",
)
parser.add_argument(
    "--profile",
    choices=(
        "foundation",
        "break_standstill",
        "privileged_teacher",
        "privileged_teacher_authority2x",
        "privileged_teacher_authority2x_noise_matched",
        "privileged_teacher_heading_stabilized",
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
        "stand_backend",
    ),
    default="foundation",
    help="Pre-registered environment/reward profile.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg, RslRlVecEnvWrapper  # noqa: E402

from gear_sonic.envs.x2_velocity import (  # noqa: E402
    X2LowerVelocityFlatEnvCfg,
    X2LowerVelocityTeacherFlatEnvCfg,
    X2LowerVelocityTeacherPhaseFlatEnvCfg,
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg,
    X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg,
)
from gear_sonic.envs.x2_velocity.rsl_rl_ppo_cfg import (  # noqa: E402
    X2LowerVelocityFlatPPORunnerCfg,
    X2ResponseHistoryActorCriticCfg,
)
from gear_sonic.envs.x2_velocity.response_history_actor_critic import (  # noqa: E402
    ResponseHistoryActorCritic,
    response_adapter_output_mask,
)
from gear_sonic.envs.x2_velocity.heading_command import (  # noqa: E402
    gain_scheduled_velocity_cfg,
)
from gear_sonic.envs.x2_velocity.symmetry import (  # noqa: E402
    compute_x2_left_right_symmetric_states,
)
from gear_sonic.envs.x2_velocity.actor_anchor import FixedParameterAnchor  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)


def main() -> None:
    response_history_profile = args.profile == "privileged_teacher_phase_template_response_history"
    if response_history_profile:
        env_cfg = X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg()
    elif args.profile in {
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "stand_backend",
    }:
        env_cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    elif args.profile == "privileged_teacher_phase_contact":
        env_cfg = X2LowerVelocityTeacherPhaseFlatEnvCfg()
    elif args.profile in {
        "privileged_teacher",
        "privileged_teacher_authority2x",
        "privileged_teacher_authority2x_noise_matched",
        "privileged_teacher_heading_stabilized",
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
    }:
        env_cfg = X2LowerVelocityTeacherFlatEnvCfg()
    else:
        env_cfg = X2LowerVelocityFlatEnvCfg()
    if args.gait_template is not None or args.gait_template_scale is not None:
        if args.profile not in {
            "privileged_teacher_phase_template_residual",
            "privileged_teacher_phase_template_heading_hold",
            "privileged_teacher_phase_template_response_history",
            "stand_backend",
        }:
            raise ValueError("gait-template overrides require the phase-template residual profile")
        if args.gait_template is not None:
            env_cfg.actions.joint_pos.template_path = str(args.gait_template.expanduser().resolve())
        if args.gait_template_scale is not None:
            if args.gait_template_scale < 0.0:
                raise ValueError("--gait_template_scale must be non-negative")
            env_cfg.actions.joint_pos.template_scale = args.gait_template_scale
    if not 0.0 <= args.plant_bias_env_fraction <= 1.0:
        raise ValueError("--plant_bias_env_fraction must be in [0, 1]")
    if abs(args.plant_ankle_roll_bias) > 1.0:
        raise ValueError("--plant_ankle_roll_bias must be in [-1, 1]")
    if args.plant_ankle_roll_bias != 0.0:
        if args.profile not in {
            "privileged_teacher_phase_template_residual",
            "privileged_teacher_phase_template_heading_hold",
            "privileged_teacher_phase_template_response_history",
        }:
            raise ValueError("plant ankle-roll bias requires a phase-template training profile")
        plant_bias = [0.0] * 15
        plant_bias[5] = args.plant_ankle_roll_bias
        plant_bias[11] = args.plant_ankle_roll_bias
        env_cfg.actions.joint_pos.normalized_plant_bias = tuple(plant_bias)
        env_cfg.actions.joint_pos.plant_bias_env_fraction = args.plant_bias_env_fraction
        # This matched branch targets the official control contract that first
        # made Stage208 survive: proximal joints retain their profile while the
        # two ankle actuator groups use Kp/Kd 40/20.
        env_cfg.scene.robot.actuators["feet"].stiffness = 40.0
        env_cfg.scene.robot.actuators["feet"].damping = 20.0
    if args.official_ankle_pd:
        # This is the isolated Stage221 intervention.  Unlike the legacy
        # plant-bias switch it changes no target position and therefore keeps
        # the experiment strictly about the deployment PD contract.
        env_cfg.scene.robot.actuators["feet"].stiffness = 40.0
        env_cfg.scene.robot.actuators["feet"].damping = 20.0
    agent_cfg = X2LowerVelocityFlatPPORunnerCfg()
    if response_history_profile:
        # OnPolicyRunner resolves custom classes from its own module globals.
        # Register the project class explicitly instead of modifying RSL-RL.
        import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

        rsl_on_policy_runner.ResponseHistoryActorCritic = ResponseHistoryActorCritic
        agent_cfg.policy = X2ResponseHistoryActorCriticCfg()
        agent_cfg.policy.response_adapter_output_mask = response_adapter_output_mask(
            args.response_adapter_mask
        )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE[
        args.collision_profile
    ]
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = (
        args.self_collisions == "on"
    )
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = args.run_name

    if args.target_iteration is not None and args.resume_checkpoint is None:
        raise ValueError("--target_iteration requires --resume_checkpoint")
    if args.weights_only_resume and args.resume_checkpoint is None:
        raise ValueError("--weights_only_resume requires --resume_checkpoint")
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
    if not 0.0 <= args.actuator_ideal_fraction <= 1.0:
        raise ValueError("--actuator_ideal_fraction must be in [0, 1]")
    if not 0.0 <= args.actuator_filter_only_fraction <= 1.0:
        raise ValueError("--actuator_filter_only_fraction must be in [0, 1]")
    if args.actuator_ideal_fraction + args.actuator_filter_only_fraction > 1.0 + 1.0e-9:
        raise ValueError("ideal and filter-only actuator fractions must sum to <= 1")
    if args.actuator_filter_only_fraction > 0.0 and args.actuator_domain not in ("delay", "noise"):
        raise ValueError("filter-only actuator partition requires delay or noise domain")
    if args.ideal_heading_stiffness <= 0.0:
        raise ValueError("--ideal_heading_stiffness must be positive")
    if not 0.0 <= args.heading_env_fraction <= 1.0:
        raise ValueError("--heading_env_fraction must be in [0, 1]")
    if args.yaw_command_abs_max <= 0.0:
        raise ValueError("--yaw_command_abs_max must be positive")
    if args.lateral_command_abs_max < 0.0:
        raise ValueError("--lateral_command_abs_max must be non-negative")
    if args.heading_error_weight < 0.0:
        raise ValueError("--heading_error_weight must be non-negative")
    if args.response_heading_stiffness is not None:
        if args.response_heading_stiffness <= 0.0:
            raise ValueError("--response_heading_stiffness must be positive")
        if args.actuator_domain == "ideal":
            raise ValueError("response heading gain requires a non-ideal actuator domain")
        if not 0.0 < args.actuator_ideal_fraction < 1.0:
            raise ValueError("gain scheduling requires a mixed ideal/response environment split")
    if args.mirror_loss_coeff < 0.0:
        raise ValueError("--mirror_loss_coeff must be non-negative")
    if args.mirror_loss_coeff > 0.0:
        if args.profile != "privileged_teacher_phase_template_heading_hold":
            raise ValueError(
                "mirror loss is pre-registered only for the phase-template heading-hold profile"
            )
        agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_x2_left_right_symmetric_states,
            mirror_loss_coeff=args.mirror_loss_coeff,
        )
    if args.actor_anchor_coeff < 0.0:
        raise ValueError("--actor_anchor_coeff must be non-negative")
    if args.actor_anchor_coeff > 0.0 and args.resume_checkpoint is None:
        raise ValueError("--actor_anchor_coeff requires --resume_checkpoint")

    if args.profile in {
        "break_standstill",
        "privileged_teacher",
        "privileged_teacher_authority2x",
        "privileged_teacher_authority2x_noise_matched",
        "privileged_teacher_heading_stabilized",
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
        "stand_backend",
    }:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.60)
        env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
        env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
        env_cfg.rewards.track_ang_vel_z_exp.weight = 0.20
        env_cfg.rewards.feet_air_time.weight = 1.0
        env_cfg.rewards.feet_slide.weight = -0.20

    if args.profile in {
        "privileged_teacher_authority2x",
        "privileged_teacher_authority2x_noise_matched",
        "privileged_teacher_heading_stabilized",
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
        "stand_backend",
    }:
        # The foundation action contract uses half of each actuator's effort
        # limit at |action|=1. Stage175 showed persistent clipping on the
        # roll/sagittal support chain, so this matched-training branch uses
        # the full effort-normalized target range without exceeding the
        # configured simulator effort limits.
        env_cfg.actions.joint_pos.scale = {
            pattern: 2.0 * scale
            for pattern, scale in env_cfg.actions.joint_pos.scale.items()
        }

    if args.profile in {
        "privileged_teacher_authority2x_noise_matched",
        "privileged_teacher_heading_stabilized",
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
        "stand_backend",
    }:
        # Doubling action scale also doubles the physical joint-target noise
        # produced by a fixed normalized Gaussian.  Stage176 therefore tested
        # more authority and twice the initial physical exploration at once.
        # Halve both normalized std and its scalar-std entropy drive so the
        # initial target-space noise and its first-order entropy gradient match
        # the original-scale teacher while retaining the full target range.
        agent_cfg.policy.init_noise_std = 0.4
        agent_cfg.algorithm.entropy_coef = 0.004

    if args.profile in {
        "privileged_teacher_heading_stabilized",
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
        "stand_backend",
    }:
        # Stage177 learned accurate body-forward speed but increasingly spun
        # its heading.  Restore the upstream G1 flat yaw reward strength, use
        # a wider kernel so the signal is not saturated at the observed rates,
        # and add a direct yaw-rate penalty.  No contact/gait term changes here.
        env_cfg.rewards.track_ang_vel_z_exp.weight = 1.0
        env_cfg.rewards.track_ang_vel_z_exp.params["std"] = 1.0
        env_cfg.rewards.yaw_rate_l2.weight = -0.5

    if args.profile in {
        "privileged_teacher_contact_dwell",
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
    }:
        env_cfg.rewards.contact_dwell.weight = -1.0

    if args.profile in {
        "privileged_teacher_contact_dwell_velocity4",
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
    }:
        env_cfg.rewards.track_lin_vel_xy_exp.weight = 4.0

    if args.profile in {
        "privileged_teacher_phase_contact",
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
    }:
        env_cfg.rewards.contact_phase.weight = -1.0

    if args.profile == "stand_backend":
        # Stand/stop is deliberately a separate backend skill.  The Stage208
        # locomotion actor keeps height in official MuJoCo but walks under a
        # zero command; train this branch only on the missing attractor instead
        # of forcing one policy update to compromise walking and standing.
        env_cfg.commands.base_velocity.rel_standing_envs = 1.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        env_cfg.rewards.track_lin_vel_xy_exp.weight = 4.0
        env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        env_cfg.rewards.stand_lin_vel_xy_l2.weight = -3.0
        env_cfg.rewards.track_ang_vel_z_exp.weight = 1.0
        env_cfg.rewards.yaw_rate_l2.weight = -0.5
        env_cfg.rewards.ang_vel_xy_l2.weight = -0.5
        env_cfg.rewards.flat_orientation_l2.weight = -6.0
        env_cfg.rewards.feet_air_time.weight = 0.0
        env_cfg.rewards.feet_slide.weight = -0.5
        env_cfg.rewards.contact_dwell.weight = -1.0
        env_cfg.rewards.contact_phase.weight = -2.0
        env_cfg.rewards.action_rate_l2.weight = -0.05
        env_cfg.rewards.action_magnitude_l2.weight = -0.01
        env_cfg.rewards.joint_deviation_hip.weight = -0.2
        env_cfg.rewards.joint_deviation_torso.weight = -0.5

        # Match the only official-domain PD intervention that converted
        # Stage208 from an immediate fall to an 8-second locomotion survivor:
        # keep proximal/waist 300/20 and lower ankle stiffness to 40/20.
        env_cfg.scene.robot.actuators["feet"].stiffness = 40.0
        env_cfg.scene.robot.actuators["feet"].damping = 20.0

    if args.profile in {
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
    }:
        # Stage183 learned the desired contact phase but could retain a nearly
        # constant heading error: yaw-rate tracking alone cannot remove an
        # already accumulated heading offset.  Feed a bounded heading-error
        # correction through the existing deployable yaw-command channel.
        env_cfg.commands.base_velocity.heading_command = True
        env_cfg.commands.base_velocity.rel_heading_envs = args.heading_env_fraction
        env_cfg.commands.base_velocity.heading_control_stiffness = args.ideal_heading_stiffness
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (
            -args.yaw_command_abs_max,
            args.yaw_command_abs_max,
        )
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (
            -args.lateral_command_abs_max,
            args.lateral_command_abs_max,
        )
        env_cfg.rewards.heading_error_l2.weight = -args.heading_error_weight
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        if args.response_heading_stiffness is not None:
            env_cfg.commands.base_velocity = gain_scheduled_velocity_cfg(
                env_cfg.commands.base_velocity,
                ideal_env_fraction=args.actuator_ideal_fraction,
                ideal_heading_control_stiffness=args.ideal_heading_stiffness,
                response_heading_control_stiffness=args.response_heading_stiffness,
            )

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
            "ideal_env_fraction": args.actuator_ideal_fraction,
            "filter_only_env_fraction": args.actuator_filter_only_fraction,
        }
        env_cfg.observations.policy.enable_corruption = args.actuator_domain == "noise"
    actuator_response_report = _apply_x2_actuator_response(
        env_cfg.scene.robot,
        actuator_response_cfg,
        physics_dt_sec=env_cfg.sim.dt,
    )

    root = Path(__file__).resolve().parents[1]
    log_root = root / "logs" / "rsl_rl" / agent_cfg.experiment_name
    log_dir = log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)

    print(f"[Stage172] log_dir={log_dir}")
    print(
        f"[Stage172] num_envs={args.num_envs} max_iterations={args.max_iterations} "
        f"device={args.device} profile={args.profile}"
    )
    print(
        f"[Stage172] actuator_domain={args.actuator_domain} "
        f"strength={args.actuator_response_strength} "
        f"ideal_fraction={args.actuator_ideal_fraction} "
        f"heading_env_fraction={args.heading_env_fraction} "
        f"yaw_command_abs_max={args.yaw_command_abs_max} "
        f"lateral_command_abs_max={args.lateral_command_abs_max} "
        f"heading_error_weight={args.heading_error_weight} "
        f"heading_gain_ideal={args.ideal_heading_stiffness} "
        f"heading_gain_response={args.response_heading_stiffness} "
        f"plant_ankle_roll_bias={args.plant_ankle_roll_bias} "
        f"plant_bias_env_fraction={args.plant_bias_env_fraction} "
        f"mirror_loss_coeff={args.mirror_loss_coeff} "
        f"actor_anchor_coeff={args.actor_anchor_coeff} "
        f"response_adapter_mask={args.response_adapter_mask} "
        f"collision_profile={args.collision_profile} "
        f"self_collisions={args.self_collisions} "
        f"report={actuator_response_report}",
        flush=True,
    )
    if args.profile in {
        "privileged_teacher_phase_template_residual",
        "privileged_teacher_phase_template_heading_hold",
        "privileged_teacher_phase_template_response_history",
        "stand_backend",
    }:
        print(
            "[Stage172] gait_template="
            f"{env_cfg.actions.joint_pos.template_path} "
            f"scale={env_cfg.actions.joint_pos.template_scale}",
            flush=True,
        )
    env = None
    wrapped = None
    actor_anchor = None
    try:
        print("[Stage172] creating ManagerBasedRLEnv", flush=True)
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print("[Stage172] environment ready", flush=True)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print("[Stage172] RSL wrapper ready", flush=True)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        print("[Stage172] OnPolicyRunner ready", flush=True)
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.add_git_repo_to_log(__file__)
        learning_iterations = agent_cfg.max_iterations
        if args.resume_checkpoint is not None:
            checkpoint = args.resume_checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            runner.load(
                str(checkpoint),
                load_optimizer=not args.weights_only_resume,
                map_location=agent_cfg.device,
            )
            saved_iteration = int(runner.current_learning_iteration)
            # RSL-RL saves after completing ``iter`` but resumes its loop at
            # that same index. Advance explicitly so one PPO update is not
            # silently repeated and checkpoint numbering stays absolute.
            runner.current_learning_iteration = saved_iteration + 1
            if args.target_iteration is not None:
                learning_iterations = args.target_iteration - runner.current_learning_iteration
                if learning_iterations <= 0:
                    raise ValueError(
                        f"target iteration {args.target_iteration} must exceed resumed "
                        f"iteration {runner.current_learning_iteration}"
                    )
            print(
                f"[Stage172] resumed checkpoint={checkpoint} saved_iter={saved_iteration} "
                f"start_iter={runner.current_learning_iteration} updates={learning_iterations} "
                f"optimizer={'reset' if args.weights_only_resume else 'restored'}",
                flush=True,
            )
            if args.actor_anchor_coeff > 0.0:
                actor_anchor = FixedParameterAnchor(
                    runner.alg.policy.actor,
                    coefficient=args.actor_anchor_coeff,
                )
                print(f"[Stage172] actor_anchor_start={actor_anchor.metrics()}", flush=True)
        print("[Stage172] starting PPO learn", flush=True)
        runner.learn(num_learning_iterations=learning_iterations, init_at_random_ep_len=True)
        if actor_anchor is not None:
            print(f"[Stage172] actor_anchor_end={actor_anchor.metrics()}", flush=True)
        print("[Stage172] PPO learn complete", flush=True)
    except BaseException as exc:
        print(f"[Stage172] FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if actor_anchor is not None:
            actor_anchor.close()
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        main()
    finally:
        simulation_app.close()
