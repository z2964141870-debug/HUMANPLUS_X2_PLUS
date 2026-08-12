#!/usr/bin/env python3
"""Phase56: one standard-93D PPO update or deterministic A/B evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("train", "eval", "screen"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-output", type=Path)
parser.add_argument("--final-output", type=Path)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-steps", type=int, default=200)
parser.add_argument("--update-index", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
_phase60_variant = os.environ.get("CWI_PHASE60_POSTURE_VARIANT")
_phase61_transition = os.environ.get("CWI_PHASE61_TRANSITION_POSTURE", "0") == "1"
_phase62_screen = os.environ.get("CWI_PHASE62_ACTION_SENSITIVITY", "0") == "1"
_phase63_screen = os.environ.get("CWI_PHASE63_HIP_DOSE_SCREEN", "0") == "1"
_phase64_screen = os.environ.get("CWI_PHASE64_KNEE_DOSE_SCREEN", "0") == "1"
_phase65_pass_raw = os.environ.get("CWI_PHASE65_CROSSOVER_PASS")
_phase65_screen = _phase65_pass_raw is not None
_phase65_pass = int(_phase65_pass_raw) if _phase65_screen else None
_phase66_pass_raw = os.environ.get("CWI_PHASE66_SIDE_PHASE_PASS")
_phase66_screen = _phase66_pass_raw is not None
_phase66_pass = int(_phase66_pass_raw) if _phase66_screen else None
_phase67_live_zero = os.environ.get("CWI_PHASE67_RESIDUAL_LIVE_ZERO", "0") == "1"
if sum(
    (
        _phase60_variant is not None,
        _phase61_transition,
        _phase62_screen,
        _phase63_screen,
        _phase64_screen,
        _phase65_screen,
        _phase66_screen,
        _phase67_live_zero,
    )
) > 1:
    raise ValueError("Phase60 through Phase67 modes are mutually exclusive")
if _phase62_screen or _phase63_screen or _phase64_screen or _phase65_screen or _phase66_screen or _phase67_live_zero:
    if args.mode != "screen" or args.num_envs != 64 or args.seed != 42 or args.eval_steps != 200:
        raise ValueError("Phase62 through Phase67 screens require screen mode, 64 envs, seed 42, and 200 steps")
    if _phase65_screen and _phase65_pass not in (0, 1):
        raise ValueError("Phase65 crossover pass must be 0 or 1")
    if _phase66_screen and _phase66_pass not in (0, 1):
        raise ValueError("Phase66 side/phase pass must be 0 or 1")
elif args.mode == "screen":
    raise ValueError("screen mode requires a Phase62 through Phase67 screen flag")
if _phase61_transition:
    if args.num_envs != 64 or args.seed != 42:
        raise ValueError("Phase61 requires 64 envs and seed 42")
    if args.mode == "eval" and args.eval_steps != 512:
        raise ValueError("Phase61 evaluation requires exactly 512 steps")
elif _phase60_variant is not None:
    if args.num_envs != 64 or args.seed not in ({40, 41, 42} if args.mode == "eval" else {42}):
        raise ValueError("Phase60 requires 64 envs, train seed 42, and eval seed 40/41/42")
elif (args.num_envs, args.seed) != (64, 42):
    raise ValueError("Phase56 is frozen to 64 envs and seed 42")
if args.mode == "train" and (args.source_output is None or args.final_output is None):
    raise ValueError("train mode requires --source-output and --final-output")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.rsl_rl_ppo_cfg import X2LowerVelocityFlatPPORunnerCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.heading_command import gain_scheduled_velocity_cfg  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from x2_upper_robust_lora_phase57 import (  # noqa: E402
    dense_tensor_map,
    inject_standard93d_lora,
    tensor_map_hash,
)
from x2_native_locomotion_posture_phase60 import (  # noqa: E402
    actual_support_com_outside_distance,
    actual_support_com_penalty,
    signed_backward_pitch_penalty,
    signed_root_pitch_rad,
)
from cwi_x2.transition_command import (  # noqa: E402
    stopped_base_speed_l2,
    terminal_double_support_penalty,
    transition_velocity_cfg,
)
from cwi_x2.transition_schedule import audit_phase_consistent_event  # noqa: E402
from cwi_x2.phase_conditioned_knee_residual import (  # noqa: E402
    PhaseConditionedKneeTargetResidual,
)

PEFT_PHASE58 = os.environ.get("CWI_PHASE58_PEFT", "0") == "1"
PEFT_PHASE59 = os.environ.get("CWI_PHASE59_PEFT", "0") == "1"
POSTURE_VARIANT = os.environ.get("CWI_PHASE60_POSTURE_VARIANT")
if POSTURE_VARIANT is not None and POSTURE_VARIANT not in {"A", "B", "C"}:
    raise ValueError("CWI_PHASE60_POSTURE_VARIANT must be A, B, or C")
PEFT_PHASE60 = POSTURE_VARIANT is not None
PEFT_PHASE61 = _phase61_transition
ACTION_SCREEN_PHASE62 = _phase62_screen
ACTION_SCREEN_PHASE63 = _phase63_screen
ACTION_SCREEN_PHASE64 = _phase64_screen
ACTION_SCREEN_PHASE65 = _phase65_screen
ACTION_SCREEN_PHASE66 = _phase66_screen
ACTION_SCREEN_PHASE67 = _phase67_live_zero
ACTION_SCREEN = (
    ACTION_SCREEN_PHASE62
    or ACTION_SCREEN_PHASE63
    or ACTION_SCREEN_PHASE64
    or ACTION_SCREEN_PHASE65
    or ACTION_SCREEN_PHASE66
    or ACTION_SCREEN_PHASE67
)
PEFT_POSTURE = PEFT_PHASE60 or PEFT_PHASE61
POSTURE_METRICS = PEFT_POSTURE or ACTION_SCREEN
PEFT_PROTECTED = PEFT_PHASE58 or PEFT_PHASE59 or PEFT_POSTURE
PHASE = 67 if ACTION_SCREEN_PHASE67 else (66 if ACTION_SCREEN_PHASE66 else (65 if ACTION_SCREEN_PHASE65 else (64 if ACTION_SCREEN_PHASE64 else (63 if ACTION_SCREEN_PHASE63 else (62 if ACTION_SCREEN_PHASE62 else (61 if PEFT_PHASE61 else (60 if PEFT_PHASE60 else (59 if PEFT_PHASE59 else (58 if PEFT_PHASE58 else 56)))))))))
TRAIN_STEPS = 512 if PEFT_PHASE61 else 24
POSTURE_REWARD_WEIGHTS = {
    "A": (0.0, 0.0),
    "B": (-0.5, 0.0),
    "C": (-0.5, -0.5),
}

REPO = Path(__file__).resolve().parents[1]
OLD = Path("/home/humanplus/x2_teleop_final/x2_sonic")
RUN = OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1"
ORIGINAL = RUN / "model_2600.pt"
ENV_YAML = RUN / "params/env.yaml"
AGENT_YAML = RUN / "params/agent.yaml"
TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
EXPECTED = {
    ORIGINAL: "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb",
    ENV_YAML: "f702a358bdbc1df94ac2a54b83aa4f6d7c98c76b091ac05a65fd074c44e6f9d7",
    AGENT_YAML: "38d462ad726e0e74d797f8a0ce3799aaadc02737e6e14a5cdf3443f7da0a8368",
    TEMPLATE: "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d",
    UPPER: "71db36d0206c44da05640f6e3616f918945524e891a5df8051533fcbeb2ab2ef",
}
LOWER15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
UPPER14 = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(f"{value.dtype}:{value.shape}".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_env_cfg(*, evaluation: bool):
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = 64
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = str(TEMPLATE)
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
    cfg.commands.base_velocity.ranges.lin_vel_x = (
        (0.30, 0.30)
        if PEFT_PHASE61
        else ((0.35, 0.35) if evaluation else (0.25, 0.60))
    )
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) if evaluation else (-0.20, 0.20)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = not evaluation and not PEFT_PHASE61
    cfg.commands.base_velocity.rel_heading_envs = 0.0 if (evaluation or PEFT_PHASE61) else 0.75
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity = gain_scheduled_velocity_cfg(
        cfg.commands.base_velocity,
        ideal_env_fraction=1.0 if ACTION_SCREEN else 0.75,
        ideal_heading_control_stiffness=1.0,
        response_heading_control_stiffness=0.05,
    )
    if PEFT_PHASE61:
        cfg.commands.base_velocity = transition_velocity_cfg(
            cfg.commands.base_velocity,
            ideal_env_fraction=0.75,
            ideal_heading_control_stiffness=1.0,
            response_heading_control_stiffness=0.05,
            stand_s=0.0,
            accelerate_s=1.0,
            cruise_s=4.2,
            decelerate_s=2.0,
            maximum_phase_offset_s=0.0,
        )
        cfg.episode_length_s = 12.0
        audit_phase_consistent_event(
            stand_s=0.0,
            accelerate_s=1.0,
            cruise_s=4.2,
            decelerate_s=2.0,
            terminal_hold_s=2.0,
            gait_cycle_s=0.8,
            double_support_fraction=0.30,
            rollout_s=512 * float(cfg.sim.dt) * int(cfg.decimation),
            random_episode_phase=False,
            maximum_phase_offset_s=0.0,
        )
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
    }
    if evaluation or PEFT_PHASE61:
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
    cfg.rewards.track_lin_vel_xy_exp.weight = 4.0
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
    cfg.rewards.track_ang_vel_z_exp.weight = 1.0
    cfg.rewards.track_ang_vel_z_exp.params["std"] = 1.0
    cfg.rewards.yaw_rate_l2.weight = -0.5
    cfg.rewards.feet_air_time.weight = 1.0
    cfg.rewards.feet_slide.weight = -0.20
    cfg.rewards.contact_dwell.weight = -1.0
    cfg.rewards.contact_phase.weight = -1.0
    cfg.rewards.heading_error_l2.weight = 0.0
    if PEFT_POSTURE:
        pitch_weight, support_weight = (
            (-0.5, -0.5)
            if PEFT_PHASE61
            else POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT]
        )
        # Candidate evaluation uses a common reward contract.  The posture and
        # support terms remain diagnostics there and influence training only.
        if evaluation:
            pitch_weight = support_weight = 0.0
        cfg.rewards.signed_backward_pitch = RewTerm(
            func=signed_backward_pitch_penalty,
            weight=pitch_weight,
            params={
                "command_name": "base_velocity",
                "tolerance_rad": 0.05,
                "normalization_rad": 0.15,
                "command_threshold_mps": 0.10,
                "asset_name": "robot",
            },
        )
        cfg.rewards.actual_support_com = RewTerm(
            func=actual_support_com_penalty,
            weight=support_weight,
            params={
                "command_name": "base_velocity",
                "normalization_m": 0.10,
                "force_threshold_n": 10.0,
                "command_threshold_mps": 0.10,
            },
        )
    if PEFT_PHASE61:
        cfg.rewards.stand_lin_vel_xy_l2 = RewTerm(
            func=stopped_base_speed_l2,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        cfg.rewards.transition_terminal_double_support = RewTerm(
            func=terminal_double_support_penalty,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "enter_force_n": 30.0,
                "left_sensor_name": "left_foot_ground_contact",
                "right_sensor_name": "right_foot_ground_contact",
            },
        )
    cfg.observations.policy.enable_corruption = False
    _apply_x2_actuator_response(
        cfg.scene.robot,
        {"enabled": True, "profile": "session03_session04_group", "randomize": False,
         "strength": 1.0, "filter_strength": 1.0, "delay_strength": 1.0,
         "include_ideal_endpoint": False,
         "ideal_env_fraction": 1.0 if ACTION_SCREEN else 0.75,
         "filter_only_env_fraction": 0.0},
        physics_dt_sec=cfg.sim.dt,
    )
    return cfg


def build_model(obs, device):
    return ActorCritic(
        obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=15, actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128], activation="elu",
        init_noise_std=0.4, noise_std_type="scalar",
        actor_obs_normalization=False, critic_obs_normalization=False,
    ).to(device)


def load_state(model, checkpoint: Path):
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    state = payload["model_state_dict"]
    has_lora = any("lora_" in name for name in state)
    if PEFT_PROTECTED:
        if has_lora:
            inject_standard93d_lora(model, rank=4, alpha=4.0)
            model.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
            inject_standard93d_lora(model, rank=4, alpha=4.0)
    else:
        if has_lora:
            raise RuntimeError("Phase56 dense runner cannot load a LoRA checkpoint")
        model.load_state_dict(state, strict=True)
    return payload


def validate_live_contract(env, wrapped):
    obs = wrapped.get_observations()
    if list(obs["policy"].shape) != [64, 93] or list(obs["critic"].shape) != [64, 93]:
        raise RuntimeError("Phase56 requires live 64x93 actor/critic observations")
    if wrapped.num_actions != 15:
        raise RuntimeError("Phase56 requires live 15D action")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15:
        raise RuntimeError("Phase56 lower12+waist3 order changed")
    zero_mask = term._cwi_upper_zero_mask
    expected_mask = (
        torch.ones(64, dtype=torch.bool, device=zero_mask.device)
        if POSTURE_METRICS
        else torch.arange(64, device=zero_mask.device) % 2 == 0
    )
    if not torch.equal(zero_mask, expected_mask):
        raise RuntimeError(f"Phase{PHASE} upper split differs from its frozen contract")
    robot = env.scene["robot"]
    if len(robot.joint_names) != 31:
        raise RuntimeError("Phase56 articulation is not 31DoF")
    actuator = robot.actuators["legs"]
    alpha = actuator._position_alpha.reshape(64, -1)[:, 0]
    lag = actuator.positions_delay_buffer.time_lags.reshape(64)
    ideal = torch.isclose(alpha, torch.ones_like(alpha)) & (lag == 0)
    counts = {
        "none_ideal": int((zero_mask & ideal).sum()),
        "none_response": int((zero_mask & ~ideal).sum()),
        "bounded_ideal": int((~zero_mask & ideal).sum()),
        "bounded_response": int((~zero_mask & ~ideal).sum()),
    }
    expected_counts = (
        {"none_ideal": 64, "none_response": 0, "bounded_ideal": 0, "bounded_response": 0}
        if ACTION_SCREEN
        else
        {"none_ideal": 48, "none_response": 16, "bounded_ideal": 0, "bounded_response": 0}
        if PEFT_POSTURE
        else {"none_ideal": 24, "none_response": 8, "bounded_ideal": 24, "bounded_response": 8}
    )
    if counts != expected_counts:
        raise RuntimeError(f"Phase56 upper/domain split changed: {counts}")
    return obs, term, zero_mask, ideal, counts


def wrapped_yaw(quat):
    return torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
    )


def aggregate_group(mask, sums, counts, survival, terminal_rate, root_min, tilt_max):
    result = {
        name: float(sums[name][mask].sum() / torch.clamp(counts[mask].sum(), min=1))
        for name in sums
    }
    result["velocity_tracking_rmse"] = math.sqrt(max(0.0, result.pop("velocity_tracking_sq")))
    result["yaw_tracking_rmse"] = math.sqrt(max(0.0, result.pop("yaw_tracking_sq")))
    result["upper_tracking_rmse"] = math.sqrt(max(0.0, result.pop("upper_tracking_sq")))
    result["survival_s_mean"] = float(survival[mask].mean())
    result["termination_rate"] = float(terminal_rate[mask].float().mean())
    result["root_height_min_mean"] = float(root_min[mask].mean())
    result["root_tilt_max_mean"] = float(tilt_max[mask].mean())
    result["env_count"] = int(mask.sum())
    return result


def _sample_summary(samples: list[torch.Tensor], mask: torch.Tensor) -> dict[str, float | None]:
    values = torch.stack(samples, dim=0)[:, mask.detach().cpu()].reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(values.mean()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _sample_summary_with_count(
    samples: list[torch.Tensor], mask: torch.Tensor
) -> dict[str, float | int | None]:
    result = _sample_summary(samples, mask)
    values = torch.stack(samples, dim=0)[:, mask.detach().cpu()].reshape(-1)
    result["finite_count"] = int(torch.isfinite(values).sum())
    return result


def aggregate_phase64_semantics(
    groups: dict[str, torch.Tensor],
    phase_samples: dict[str, dict[str, list[torch.Tensor]]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Summarize deployable gait-clock semantic regions for each mirrored group."""

    result = {}
    for group_name, group_mask in groups.items():
        group_result = {}
        for phase_name, metrics in phase_samples.items():
            pitch = _sample_summary_with_count(metrics["signed_pitch_rad"], group_mask)
            support = _sample_summary_with_count(metrics["com_support_outside_m"], group_mask)
            slip = _sample_summary_with_count(metrics["stance_slip_mps"], group_mask)
            flight = _sample_summary_with_count(metrics["flight_fraction"], group_mask)
            velocity_sq = torch.stack(metrics["velocity_tracking_sq"], dim=0)[
                :, group_mask.detach().cpu()
            ].reshape(-1)
            velocity_sq = velocity_sq[torch.isfinite(velocity_sq)]
            group_result[phase_name] = {
                "sample_count": int(velocity_sq.numel()),
                "signed_pitch_rad": pitch,
                "com_support_outside_m": support,
                "stance_slip_mps": slip,
                "flight_fraction": flight,
                "velocity_tracking_rmse_mps": (
                    math.sqrt(max(0.0, float(velocity_sq.mean())))
                    if velocity_sq.numel()
                    else None
                ),
            }
        result[group_name] = group_result
    return result


def aggregate_phase60_group(
    mask, sums, counts, survival, terminal_rate, root_min, tilt_max,
    samples, knee_min, knee_max,
):
    result = aggregate_group(mask, sums, counts, survival, terminal_rate, root_min, tilt_max)
    result["signed_pitch_rad"] = _sample_summary(samples["signed_pitch_rad"], mask)
    result["com_support_outside_m"] = _sample_summary(samples["com_support_outside_m"], mask)
    result["stance_slip_mps"] = _sample_summary(samples["stance_slip_mps"], mask)
    result["swing_sole_clearance_m"] = _sample_summary(samples["swing_sole_clearance_m"], mask)
    for name in (
        "terminal_base_speed_mps",
        "terminal_double_support",
        "terminal_root_height_m",
        "terminal_root_tilt_rad",
    ):
        if name in samples:
            result[name] = _sample_summary(samples[name], mask)
    knee_excursion = knee_max[mask] - knee_min[mask]
    result["knee_excursion_rad_mean"] = float(knee_excursion.mean())
    result["knee_joint_range"] = {
        side: {
            "minimum_rad_mean": float(knee_min[mask, index].mean()),
            "minimum_rad_min": float(knee_min[mask, index].min()),
            "maximum_rad_mean": float(knee_max[mask, index].mean()),
            "maximum_rad_max": float(knee_max[mask, index].max()),
            "excursion_rad_mean": float(knee_excursion[:, index].mean()),
            "excursion_rad_max": float(knee_excursion[:, index].max()),
        }
        for side, index in (("left", 0), ("right", 1))
    }
    result["knee_excursion_left_right_abs_diff_rad_mean"] = float(
        (knee_excursion[:, 0] - knee_excursion[:, 1]).abs().mean()
    )
    return result


def phase62_action_biases(device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assign one small sagittal finite-difference intervention per env group."""

    epsilon = 0.02
    specifications = (
        ("base", (), 0.0),
        ("hip_pitch_neg", (0, 6), -epsilon),
        ("hip_pitch_pos", (0, 6), epsilon),
        ("knee_neg", (3, 9), -epsilon),
        ("knee_pos", (3, 9), epsilon),
        ("ankle_pitch_neg", (4, 10), -epsilon),
        ("ankle_pitch_pos", (4, 10), epsilon),
        ("waist_pitch_neg", (13,), -epsilon),
        ("waist_pitch_pos", (13,), epsilon),
    )
    sizes = (8,) + (7,) * 8
    biases = torch.zeros((64, 15), device=device)
    groups: dict[str, torch.Tensor] = {}
    start = 0
    for (name, indices, value), size in zip(specifications, sizes, strict=True):
        stop = start + size
        mask = torch.zeros(64, dtype=torch.bool, device=device)
        mask[start:stop] = True
        groups[name] = mask
        if indices:
            biases[start:stop, list(indices)] = value
        start = stop
    if start != 64 or not torch.allclose(biases[groups["base"]], torch.zeros_like(biases[groups["base"]])):
        raise RuntimeError("Phase62 action-bias partition is invalid")
    return biases, groups


def phase63_hip_pitch_doses(device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assign preregistered bilateral hip-pitch doses to eight equal groups."""

    specifications = (
        ("base", 0.0),
        ("hip_pitch_m004", -0.004),
        ("hip_pitch_m006", -0.006),
        ("hip_pitch_m008", -0.008),
        ("hip_pitch_m010", -0.010),
        ("hip_pitch_m012", -0.012),
        ("hip_pitch_m016", -0.016),
        ("hip_pitch_m020", -0.020),
    )
    biases = torch.zeros((64, 15), device=device)
    groups: dict[str, torch.Tensor] = {}
    for group_index, (name, value) in enumerate(specifications):
        start = group_index * 8
        stop = start + 8
        mask = torch.zeros(64, dtype=torch.bool, device=device)
        mask[start:stop] = True
        groups[name] = mask
        biases[start:stop, [0, 6]] = value
    if not torch.allclose(biases[groups["base"]], torch.zeros_like(biases[groups["base"]])):
        raise RuntimeError("Phase63 action-bias partition is invalid")
    return biases, groups


def phase64_knee_pitch_mirrored_doses(
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assign knee doses to two mirrored round-robin lanes plus pooled groups."""

    slot_specs = (
        ("A_base", 0.0),
        ("A_knee_m008", -0.008),
        ("A_knee_m010", -0.010),
        ("A_knee_m012", -0.012),
        ("B_base", 0.0),
        ("B_knee_m012", -0.012),
        ("B_knee_m010", -0.010),
        ("B_knee_m008", -0.008),
    )
    env_ids = torch.arange(64, device=device)
    row = torch.div(env_ids, 8, rounding_mode="floor")
    column = env_ids.remainder(8)
    latin_slot = (row + column).remainder(8)
    biases = torch.zeros((64, 15), device=device)
    groups: dict[str, torch.Tensor] = {}
    for slot, (name, value) in enumerate(slot_specs):
        mask = latin_slot == slot
        groups[name] = mask
        biases[mask, 3] = value
        biases[mask, 9] = value
    groups["base"] = groups["A_base"] | groups["B_base"]
    for suffix in ("m008", "m010", "m012"):
        groups[f"knee_{suffix}"] = groups[f"A_knee_{suffix}"] | groups[f"B_knee_{suffix}"]
    if any(int(groups[name].sum()) != 8 for name, _ in slot_specs):
        raise RuntimeError("Phase64 mirrored lane partition is invalid")
    if any(int(groups[name].sum()) != 16 for name in ("base", "knee_m008", "knee_m010", "knee_m012")):
        raise RuntimeError("Phase64 pooled partition is invalid")
    return biases, groups


def phase65_crossover_partition(
    device: torch.device, pass_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the balanced checkerboard treatment mask for one crossover pass."""

    env_ids = torch.arange(64, device=device)
    row = torch.div(env_ids, 8, rounding_mode="floor")
    column = env_ids.remainder(8)
    sequence_tc = (row + column).remainder(2) == 0
    sequence_ct = ~sequence_tc
    treatment = sequence_tc if pass_index == 0 else sequence_ct
    groups = {
        "sequence_TC": sequence_tc,
        "sequence_CT": sequence_ct,
        "treatment": treatment,
        "control": ~treatment,
    }
    if any(int(mask.sum()) != 32 for mask in groups.values()):
        raise RuntimeError("Phase65 checkerboard crossover partition is invalid")
    return treatment, groups


def phase65_single_support_knee_bias(
    policy_observation: torch.Tensor,
    command: torch.Tensor,
    treatment_mask: torch.Tensor,
    *,
    dose: float = -0.010,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build a deployable bilateral bias from the actor's gait suffix only."""

    if policy_observation.ndim != 2 or policy_observation.shape[1] < 4:
        raise RuntimeError("Phase65 requires the deployable four-value gait suffix")
    gait = policy_observation[:, -4:]
    sine, cosine, desired_left, desired_right = gait.unbind(dim=-1)
    left = desired_left > 0.5
    right = desired_right > 0.5
    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
    double_support = left & right
    right_swing_left_support = left & ~right
    left_swing_right_support = ~left & right
    valid_contacts = double_support | right_swing_left_support | left_swing_right_support
    standing = ~moving
    active_single_support = (
        treatment_mask
        & moving
        & (right_swing_left_support | left_swing_right_support)
    )
    bias = torch.zeros((policy_observation.shape[0], 15), device=policy_observation.device)
    bias[active_single_support, 3] = dose
    bias[active_single_support, 9] = dose
    semantics = {
        "double_support_zero": moving & double_support & (cosine >= 0.0),
        "right_swing_left_support": moving & right_swing_left_support,
        "double_support_half": moving & double_support & (cosine < 0.0),
        "left_swing_right_support": moving & left_swing_right_support,
        "standing": standing,
        "invalid_contact_suffix": moving & ~valid_contacts,
        "active_single_support": active_single_support,
        "moving": moving,
    }
    return bias, semantics


def phase66_side_phase_partition(
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[str, ...]]:
    """Assign every scene row/column once to each side-by-phase cell."""

    names = (
        "DS0_left", "DS0_right",
        "right_swing_left_support_left", "right_swing_left_support_right",
        "DS_half_left", "DS_half_right",
        "left_swing_right_support_left", "left_swing_right_support_right",
    )
    env_ids = torch.arange(64, device=device)
    row = torch.div(env_ids, 8, rounding_mode="floor")
    column = env_ids.remainder(8)
    slots = (row + column).remainder(8)
    groups = {name: slots == index for index, name in enumerate(names)}
    if any(int(mask.sum()) != 8 for mask in groups.values()):
        raise RuntimeError("Phase66 side/phase Latin-square partition is invalid")
    return slots, groups, names


def phase66_physical_knee_target_offset(
    policy_observation: torch.Tensor,
    command: torch.Tensor,
    condition_slot: torch.Tensor,
    *,
    candidate_enabled: bool,
    target_offset_rad: float = -0.003,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return a final-target knee offset and its deployable pre-step phase."""

    gait = policy_observation[:, -4:]
    _, cosine, desired_left, desired_right = gait.unbind(dim=-1)
    left = desired_left > 0.5
    right = desired_right > 0.5
    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
    regions = {
        "double_support_zero": moving & left & right & (cosine >= 0.0),
        "right_swing_left_support": moving & left & ~right,
        "double_support_half": moving & left & right & (cosine < 0.0),
        "left_swing_right_support": moving & ~left & right,
        "standing": ~moving,
        "invalid_contact_suffix": moving & ~(left | right),
    }
    region_index = torch.full(
        (policy_observation.shape[0],), -1, dtype=torch.int64, device=policy_observation.device
    )
    for index, name in enumerate(
        (
            "double_support_zero",
            "right_swing_left_support",
            "double_support_half",
            "left_swing_right_support",
            "standing",
        )
    ):
        region_index[regions[name]] = index
    requested = torch.zeros((policy_observation.shape[0], 15), device=policy_observation.device)
    if candidate_enabled:
        target_region = torch.div(condition_slot, 2, rounding_mode="floor")
        target_side = condition_slot.remainder(2)
        active = region_index == target_region
        requested[active & (target_side == 0), 3] = target_offset_rad
        requested[active & (target_side == 1), 9] = target_offset_rad
    regions["active"] = requested.abs().sum(dim=-1) > 0.0
    return requested, region_index, regions


def evaluate() -> None:
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(evaluation=True))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs, term, zero_mask, ideal_mask, domain_counts = validate_live_contract(env, wrapped)
        model = build_model(obs, env.device).eval()
        payload = load_state(model, args.checkpoint)
        model.std.requires_grad_(False)
        phase67_residual = (
            PhaseConditionedKneeTargetResidual().to(env.device).eval()
            if ACTION_SCREEN_PHASE67
            else None
        )
        robot = env.scene["robot"]
        upper_ids = [robot.joint_names.index(name) for name in UPPER14]
        initial_pos = robot.data.root_pos_w.clone()
        initial_yaw = wrapped_yaw(robot.data.root_quat_w).clone()
        alive = torch.ones(64, dtype=torch.bool, device=env.device)
        survival = torch.full((64,), args.eval_steps * env.step_dt, device=env.device)
        terminal = torch.zeros(64, dtype=torch.bool, device=env.device)
        root_min = robot.data.root_pos_w[:, 2].clone()
        tilt_max = torch.zeros(64, device=env.device)
        prev_action = torch.zeros((64, 15), device=env.device)
        if ACTION_SCREEN_PHASE62:
            action_bias, screen_groups = phase62_action_biases(env.device)
        elif ACTION_SCREEN_PHASE63:
            action_bias, screen_groups = phase63_hip_pitch_doses(env.device)
        elif ACTION_SCREEN_PHASE64:
            action_bias, screen_groups = phase64_knee_pitch_mirrored_doses(env.device)
        elif ACTION_SCREEN_PHASE65:
            phase65_treatment_mask, screen_groups = phase65_crossover_partition(
                env.device, _phase65_pass
            )
            action_bias = torch.zeros((64, 15), device=env.device)
        elif ACTION_SCREEN_PHASE66:
            phase66_condition_slot, screen_groups, phase66_condition_names = (
                phase66_side_phase_partition(env.device)
            )
            action_bias = torch.zeros((64, 15), device=env.device)
        elif ACTION_SCREEN_PHASE67:
            screen_groups = {"all": torch.ones(64, dtype=torch.bool, device=env.device)}
            action_bias = torch.zeros((64, 15), device=env.device)
        else:
            action_bias, screen_groups = torch.zeros((64, 15), device=env.device), {}
        metric_names = (
            "velocity_tracking_sq", "yaw_tracking_sq", "lateral_abs",
            "yaw_abs", "upper_tracking_sq", "action_abs", "action_delta_abs", "reward",
        )
        if POSTURE_METRICS:
            metric_names += ("flight_fraction", "single_support_fraction", "double_support_fraction")
        if ACTION_SCREEN_PHASE64 or ACTION_SCREEN_PHASE65:
            metric_names += (
                "knee_requested_bias_mean",
                "knee_effective_bias_mean",
                "knee_requested_target_offset_rad_mean",
                "knee_effective_target_offset_rad_mean",
                "knee_source_clip_fraction",
                "knee_intervention_clip_fraction",
            )
        sums = {name: torch.zeros(64, device=env.device) for name in metric_names}
        counts = torch.zeros(64, device=env.device)
        samples = {
            "signed_pitch_rad": [],
            "com_support_outside_m": [],
            "stance_slip_mps": [],
            "swing_sole_clearance_m": [],
        }
        phase64_samples = {
            phase_name: {
                metric_name: []
                for metric_name in (
                    "signed_pitch_rad",
                    "com_support_outside_m",
                    "stance_slip_mps",
                    "velocity_tracking_sq",
                    "flight_fraction",
                )
            }
            for phase_name in (
                "double_support_zero",
                "right_swing_left_support",
                "double_support_half",
                "left_swing_right_support",
            )
        } if ACTION_SCREEN_PHASE64 else {}
        phase65_raw_samples = {
            name: []
            for name in (
                "signed_pitch_rad",
                "com_support_outside_m",
                "stance_slip_mps",
                "swing_sole_clearance_m",
                "velocity_tracking_sq",
                "flight_fraction",
                "knee_requested_bias_left",
                "knee_requested_bias_right",
                "knee_effective_bias_left",
                "knee_effective_bias_right",
                "knee_effective_target_offset_left_rad",
                "knee_effective_target_offset_right_rad",
                "knee_source_clip_left",
                "knee_source_clip_right",
                "knee_intervention_clip_left",
                "knee_intervention_clip_right",
            )
        } if ACTION_SCREEN_PHASE65 else {}
        phase65_pre_step_regions = []
        phase65_replay_hashes = []
        phase65_bias_contract = {
            "inactive_requested_max_abs": 0.0,
            "inactive_effective_max_abs": 0.0,
            "shadow_standing_requested_max_abs": 0.0,
            "invalid_contact_suffix_count": 0,
            "active_sample_count": 0,
        } if ACTION_SCREEN_PHASE65 else {}
        phase65_initial_fingerprints = (
            {
                "policy_observation": tensor_hash(obs["policy"]),
                "critic_observation": tensor_hash(obs["critic"]),
                "root_state": tensor_hash(robot.data.root_state_w),
                "joint_position": tensor_hash(robot.data.joint_pos),
                "joint_velocity": tensor_hash(robot.data.joint_vel),
                "command": tensor_hash(env.command_manager.get_command("base_velocity")),
            }
            if ACTION_SCREEN_PHASE65
            else {}
        )
        phase66_raw_samples = {
            name: []
            for name in (
                "signed_pitch_rad", "com_support_outside_m", "stance_slip_mps",
                "swing_sole_clearance_m", "velocity_tracking_sq", "flight_fraction",
                "requested_target_left_rad", "requested_target_right_rad",
                "effective_target_left_rad", "effective_target_right_rad",
                "source_target_left_rad", "source_target_right_rad",
                "intervention_target_left_rad", "intervention_target_right_rad",
            )
        } if ACTION_SCREEN_PHASE66 else {}
        phase66_pre_step_regions = []
        phase66_replay_hashes = []
        phase66_contract = {
            "invalid_contact_suffix_count": 0,
            "inactive_requested_max_abs": 0.0,
            "inactive_effective_max_abs": 0.0,
            "active_sample_count": 0,
        } if ACTION_SCREEN_PHASE66 else {}
        phase66_initial_fingerprints = (
            {
                "policy_observation": tensor_hash(obs["policy"]),
                "critic_observation": tensor_hash(obs["critic"]),
                "root_state": tensor_hash(robot.data.root_state_w),
                "joint_position": tensor_hash(robot.data.joint_pos),
                "joint_velocity": tensor_hash(robot.data.joint_vel),
                "command": tensor_hash(env.command_manager.get_command("base_velocity")),
            }
            if ACTION_SCREEN_PHASE66
            else {}
        )
        phase67_contract = {
            "residual_output_max_abs_rad": 0.0,
            "standing_shadow_output_max_abs_rad": 0.0,
            "invalid_contact_shadow_output_max_abs_rad": 0.0,
            "processed_target_delta_max_abs_rad": 0.0,
            "non_knee_output_max_abs_rad": 0.0,
            "finite": True,
        } if ACTION_SCREEN_PHASE67 else {}
        phase67_step_hashes = []
        phase67_initial_fingerprints = (
            {
                "policy_observation": tensor_hash(obs["policy"]),
                "root_state": tensor_hash(robot.data.root_state_w),
                "joint_position": tensor_hash(robot.data.joint_pos),
                "command": tensor_hash(env.command_manager.get_command("base_velocity")),
                "residual_state": state_hash(phase67_residual),
            }
            if ACTION_SCREEN_PHASE67
            else {}
        )
        if PEFT_PHASE61:
            samples.update(
                terminal_base_speed_mps=[],
                terminal_double_support=[],
                terminal_root_height_m=[],
                terminal_root_tilt_rad=[],
            )
        foot_ids = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]
        knee_ids = [robot.joint_names.index("left_knee_joint"), robot.joint_names.index("right_knee_joint")]
        knee_min = robot.data.joint_pos[:, knee_ids].clone()
        knee_max = robot.data.joint_pos[:, knee_ids].clone()
        phase66_requested_target = torch.zeros((64, 15), device=env.device)
        phase66_effective_target = torch.zeros((64, 15), device=env.device)
        phase66_source_target = torch.zeros((64, 15), device=env.device)
        phase66_intervention_target = torch.zeros((64, 15), device=env.device)
        phase66_original_process_actions = None
        if ACTION_SCREEN_PHASE66:
            phase66_original_process_actions = term.process_actions

            def phase66_process_actions(actions: torch.Tensor) -> None:
                nonlocal phase66_effective_target
                nonlocal phase66_source_target
                nonlocal phase66_intervention_target
                phase66_original_process_actions(actions)
                phase66_source_target = term._processed_actions.clone()
                proposed = phase66_source_target + phase66_requested_target
                if term.cfg.clip is not None:
                    proposed = torch.clamp(
                        proposed, min=term._clip[:, :, 0], max=term._clip[:, :, 1]
                    )
                phase66_intervention_target = proposed
                phase66_effective_target = proposed - phase66_source_target
                term._processed_actions[:] = proposed

            term.process_actions = phase66_process_actions
        phase67_requested_target = torch.zeros((64, 15), device=env.device)
        phase67_source_target = torch.zeros((64, 15), device=env.device)
        phase67_intervention_target = torch.zeros((64, 15), device=env.device)
        if ACTION_SCREEN_PHASE67:
            phase67_original_process_actions = term.process_actions

            def phase67_process_actions(actions: torch.Tensor) -> None:
                nonlocal phase67_source_target
                nonlocal phase67_intervention_target
                phase67_original_process_actions(actions)
                phase67_source_target = term._processed_actions.clone()
                phase67_intervention_target = phase67_source_target + phase67_requested_target
                term._processed_actions[:] = phase67_intervention_target

            term.process_actions = phase67_process_actions
        with torch.inference_mode():
            for step in range(args.eval_steps):
                phase65_pre_policy = obs["policy"] if ACTION_SCREEN_PHASE65 else None
                if ACTION_SCREEN_PHASE65:
                    pre_step_command = env.command_manager.get_command("base_velocity")
                    action_bias, phase65_semantics = phase65_single_support_knee_bias(
                        phase65_pre_policy,
                        pre_step_command,
                        phase65_treatment_mask,
                    )
                    shadow_standing_bias, _ = phase65_single_support_knee_bias(
                        phase65_pre_policy,
                        torch.zeros_like(pre_step_command),
                        phase65_treatment_mask,
                    )
                    phase65_bias_contract["shadow_standing_requested_max_abs"] = max(
                        phase65_bias_contract["shadow_standing_requested_max_abs"],
                        float(shadow_standing_bias.abs().max()),
                    )
                    phase65_bias_contract["invalid_contact_suffix_count"] += int(
                        phase65_semantics["invalid_contact_suffix"].sum()
                    )
                    active_phase65 = phase65_semantics["active_single_support"]
                    phase65_bias_contract["active_sample_count"] += int(active_phase65.sum())
                    inactive_phase65 = ~active_phase65
                    phase65_bias_contract["inactive_requested_max_abs"] = max(
                        phase65_bias_contract["inactive_requested_max_abs"],
                        float(action_bias[inactive_phase65].abs().max()),
                    )
                    region_index = torch.full(
                        (64,), -1, dtype=torch.int64, device=env.device
                    )
                    for index, name in enumerate(
                        (
                            "double_support_zero",
                            "right_swing_left_support",
                            "double_support_half",
                            "left_swing_right_support",
                            "standing",
                        )
                    ):
                        region_index[phase65_semantics[name]] = index
                    phase65_pre_step_regions.append(region_index.detach().cpu())
                phase66_pre_policy = obs["policy"] if ACTION_SCREEN_PHASE66 else None
                if ACTION_SCREEN_PHASE66:
                    phase66_requested_target, phase66_region_index, phase66_semantics = (
                        phase66_physical_knee_target_offset(
                            phase66_pre_policy,
                            env.command_manager.get_command("base_velocity"),
                            phase66_condition_slot,
                            candidate_enabled=_phase66_pass == 0,
                        )
                    )
                    phase66_pre_step_regions.append(phase66_region_index.detach().cpu())
                    phase66_contract["invalid_contact_suffix_count"] += int(
                        phase66_semantics["invalid_contact_suffix"].sum()
                    )
                    phase66_contract["active_sample_count"] += int(
                        phase66_semantics["active"].sum()
                    )
                if ACTION_SCREEN_PHASE67:
                    phase67_requested_target = phase67_residual.target_offset_15d(obs["policy"])
                    standing_shadow = obs["policy"].clone()
                    standing_shadow[:, -4:] = standing_shadow.new_tensor([0.0, 0.0, 1.0, 1.0])
                    shadow_output = phase67_residual.target_offset_15d(standing_shadow)
                    invalid_contact_shadow = obs["policy"].clone()
                    invalid_contact_shadow[:, -2:] = invalid_contact_shadow.new_tensor(
                        [0.6, 0.6]
                    )
                    invalid_contact_output = phase67_residual.target_offset_15d(
                        invalid_contact_shadow
                    )
                    phase67_contract["residual_output_max_abs_rad"] = max(
                        phase67_contract["residual_output_max_abs_rad"],
                        float(phase67_requested_target.abs().max()),
                    )
                    phase67_contract["standing_shadow_output_max_abs_rad"] = max(
                        phase67_contract["standing_shadow_output_max_abs_rad"],
                        float(shadow_output.abs().max()),
                    )
                    phase67_contract["invalid_contact_shadow_output_max_abs_rad"] = max(
                        phase67_contract["invalid_contact_shadow_output_max_abs_rad"],
                        float(invalid_contact_output.abs().max()),
                    )
                    non_knee = phase67_requested_target.clone()
                    non_knee[:, [3, 9]] = 0.0
                    phase67_contract["non_knee_output_max_abs_rad"] = max(
                        phase67_contract["non_knee_output_max_abs_rad"],
                        float(non_knee.abs().max()),
                    )
                    phase67_contract["finite"] &= bool(
                        torch.isfinite(phase67_requested_target).all()
                        and torch.isfinite(shadow_output).all()
                        and torch.isfinite(invalid_contact_output).all()
                    )
                raw_action = model.act_inference(obs)
                action = raw_action
                if ACTION_SCREEN:
                    action = torch.clamp(raw_action + action_bias, -1.0, 1.0)
                next_obs, reward, done, _ = wrapped.step(action)
                done = done.reshape(-1).bool()
                root_quat = robot.data.root_quat_w
                root_yaw = wrapped_yaw(root_quat)
                yaw_delta = torch.atan2(torch.sin(root_yaw - initial_yaw), torch.cos(root_yaw - initial_yaw))
                root_up_z = 1.0 - 2.0 * (root_quat[:, 1].square() + root_quat[:, 2].square())
                tilt = torch.acos(torch.clamp(root_up_z, -1.0, 1.0))
                command = env.command_manager.get_command("base_velocity")
                velocity_error = robot.data.root_lin_vel_b[:, :2] - command[:, :2]
                yaw_error = robot.data.root_ang_vel_b[:, 2] - command[:, 2]
                upper_q = robot.data.joint_pos[:, upper_ids]
                upper_target = term._cwi_prev_upper_target
                valid = alive.to(torch.float32)
                values = {
                    "velocity_tracking_sq": velocity_error.square().sum(-1),
                    "yaw_tracking_sq": yaw_error.square(),
                    "lateral_abs": (robot.data.root_pos_w[:, 1] - initial_pos[:, 1]).abs(),
                    "yaw_abs": yaw_delta.abs(),
                    "upper_tracking_sq": (upper_q - upper_target).square().mean(-1),
                    "action_abs": action.abs().mean(-1),
                    "action_delta_abs": (action - prev_action).abs().mean(-1),
                    "reward": reward.reshape(-1),
                }
                if ACTION_SCREEN_PHASE64 or ACTION_SCREEN_PHASE65:
                    knee_action_ids = [3, 9]
                    base_clipped_action = torch.clamp(raw_action, -1.0, 1.0)
                    requested_knee_bias = action_bias[:, knee_action_ids]
                    normalized_template = term._normalized_template_bias[:, knee_action_ids]
                    normalized_plant = (
                        term._normalized_plant_bias[knee_action_ids].unsqueeze(0)
                        * term._plant_bias_env_mask
                    )
                    source_preclip = (
                        base_clipped_action[:, knee_action_ids]
                        + normalized_template
                        + normalized_plant
                    )
                    intervention_preclip = (
                        action[:, knee_action_ids] + normalized_template + normalized_plant
                    )
                    effective_knee_bias = (
                        torch.clamp(intervention_preclip, -1.0, 1.0)
                        - torch.clamp(source_preclip, -1.0, 1.0)
                    )
                    knee_scale = term._scale[:, knee_action_ids]
                    values.update(
                        knee_requested_bias_mean=requested_knee_bias.mean(dim=-1),
                        knee_effective_bias_mean=effective_knee_bias.mean(dim=-1),
                        knee_requested_target_offset_rad_mean=(
                            requested_knee_bias * knee_scale
                        ).mean(dim=-1),
                        knee_effective_target_offset_rad_mean=(
                            effective_knee_bias * knee_scale
                        ).mean(dim=-1),
                        knee_source_clip_fraction=(
                            source_preclip.abs() > 1.0
                        ).any(dim=-1).to(torch.float32),
                        knee_intervention_clip_fraction=(
                            intervention_preclip.abs() > 1.0
                        ).any(dim=-1).to(torch.float32),
                    )
                    if ACTION_SCREEN_PHASE65:
                        phase65_bias_contract["inactive_effective_max_abs"] = max(
                            phase65_bias_contract["inactive_effective_max_abs"],
                            float(effective_knee_bias[inactive_phase65].abs().max()),
                        )
                        phase65_effective_bias = effective_knee_bias
                        phase65_requested_bias = requested_knee_bias
                        phase65_effective_target_offset = effective_knee_bias * knee_scale
                        phase65_source_clip = source_preclip.abs() > 1.0
                        phase65_intervention_clip = intervention_preclip.abs() > 1.0
                if POSTURE_METRICS:
                    pitch = signed_root_pitch_rad(robot)
                    outside, contact_count = actual_support_com_outside_distance(
                        env, force_threshold_n=10.0
                    )
                    forces = torch.stack(
                        tuple(
                            env.scene[name].data.force_matrix_w[..., 2]
                            .abs().reshape(64, -1).amax(dim=-1)
                            for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                        ),
                        dim=-1,
                    )
                    contact = forces > 10.0
                    foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
                    stance_slip = (foot_speed * contact).sum(dim=-1) / contact_count.clamp_min(1)
                    stance_slip = torch.where(
                        contact_count > 0, stance_slip, torch.full_like(stance_slip, torch.nan)
                    )
                    sole_height = robot.data.body_pos_w[:, foot_ids, 2] - 0.068
                    swing = ~contact
                    swing_clearance = torch.where(
                        swing,
                        sole_height,
                        torch.full_like(sole_height, torch.nan),
                    ).nanmean(dim=-1)
                    if ACTION_SCREEN_PHASE65:
                        phase65_sample_valid = alive.clone()
                        phase65_step_values = {
                            "signed_pitch_rad": pitch,
                            "com_support_outside_m": outside,
                            "stance_slip_mps": stance_slip,
                            "swing_sole_clearance_m": swing_clearance,
                            "velocity_tracking_sq": velocity_error.square().sum(-1),
                            "flight_fraction": (contact_count == 0).to(torch.float32),
                            "knee_requested_bias_left": phase65_requested_bias[:, 0],
                            "knee_requested_bias_right": phase65_requested_bias[:, 1],
                            "knee_effective_bias_left": phase65_effective_bias[:, 0],
                            "knee_effective_bias_right": phase65_effective_bias[:, 1],
                            "knee_effective_target_offset_left_rad": (
                                phase65_effective_target_offset[:, 0]
                            ),
                            "knee_effective_target_offset_right_rad": (
                                phase65_effective_target_offset[:, 1]
                            ),
                            "knee_source_clip_left": phase65_source_clip[:, 0].to(torch.float32),
                            "knee_source_clip_right": phase65_source_clip[:, 1].to(torch.float32),
                            "knee_intervention_clip_left": (
                                phase65_intervention_clip[:, 0].to(torch.float32)
                            ),
                            "knee_intervention_clip_right": (
                                phase65_intervention_clip[:, 1].to(torch.float32)
                            ),
                        }
                        for name, value in phase65_step_values.items():
                            phase65_raw_samples[name].append(
                                torch.where(
                                    phase65_sample_valid,
                                    value,
                                    torch.full_like(value, torch.nan),
                                ).detach().cpu()
                            )
                    if ACTION_SCREEN_PHASE66:
                        phase66_sample_valid = alive.clone()
                        phase66_active = phase66_semantics["active"]
                        phase66_inactive = ~phase66_active
                        phase66_contract["inactive_requested_max_abs"] = max(
                            phase66_contract["inactive_requested_max_abs"],
                            float(phase66_requested_target[phase66_inactive].abs().max()),
                        )
                        phase66_contract["inactive_effective_max_abs"] = max(
                            phase66_contract["inactive_effective_max_abs"],
                            float(phase66_effective_target[phase66_inactive].abs().max()),
                        )
                        phase66_values = {
                            "signed_pitch_rad": pitch,
                            "com_support_outside_m": outside,
                            "stance_slip_mps": stance_slip,
                            "swing_sole_clearance_m": swing_clearance,
                            "velocity_tracking_sq": velocity_error.square().sum(-1),
                            "flight_fraction": (contact_count == 0).to(torch.float32),
                            "requested_target_left_rad": phase66_requested_target[:, 3],
                            "requested_target_right_rad": phase66_requested_target[:, 9],
                            "effective_target_left_rad": phase66_effective_target[:, 3],
                            "effective_target_right_rad": phase66_effective_target[:, 9],
                            "source_target_left_rad": phase66_source_target[:, 3],
                            "source_target_right_rad": phase66_source_target[:, 9],
                            "intervention_target_left_rad": phase66_intervention_target[:, 3],
                            "intervention_target_right_rad": phase66_intervention_target[:, 9],
                        }
                        for name, value in phase66_values.items():
                            phase66_raw_samples[name].append(
                                torch.where(
                                    phase66_sample_valid,
                                    value,
                                    torch.full_like(value, torch.nan),
                                ).detach().cpu()
                            )
                    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
                    valid_bool = alive.clone()
                    if ACTION_SCREEN_PHASE64:
                        phase = torch.remainder(
                            env.episode_length_buf.to(torch.float32) * env.step_dt / 0.8,
                            1.0,
                        )
                        phase_masks = {
                            "double_support_zero": (phase < 0.075) | (phase >= 0.925),
                            "right_swing_left_support": (phase >= 0.075) & (phase < 0.425),
                            "double_support_half": (phase >= 0.425) & (phase < 0.575),
                            "left_swing_right_support": (phase >= 0.575) & (phase < 0.925),
                        }
                        phase_values = {
                            "signed_pitch_rad": pitch,
                            "com_support_outside_m": outside,
                            "stance_slip_mps": stance_slip,
                            "velocity_tracking_sq": velocity_error.square().sum(-1),
                            "flight_fraction": (contact_count == 0).to(torch.float32),
                        }
                        for phase_name, phase_mask in phase_masks.items():
                            phase_valid = valid_bool & moving & phase_mask
                            for metric_name, value in phase_values.items():
                                phase64_samples[phase_name][metric_name].append(
                                    torch.where(
                                        phase_valid,
                                        value,
                                        torch.full_like(value, torch.nan),
                                    ).detach().cpu()
                                )
                    for name, value in {
                        "signed_pitch_rad": pitch,
                        "com_support_outside_m": outside,
                        "stance_slip_mps": stance_slip,
                        "swing_sole_clearance_m": swing_clearance,
                    }.items():
                        sample_valid = (
                            valid_bool & moving
                            if PEFT_PHASE61
                            and name in {"signed_pitch_rad", "com_support_outside_m"}
                            else valid_bool
                        )
                        samples[name].append(
                            torch.where(
                                sample_valid,
                                value,
                                torch.full_like(value, torch.nan),
                            ).detach().cpu()
                        )
                    if PEFT_PHASE61:
                        terminal_mask = env.command_manager.get_term(
                            "base_velocity"
                        ).terminal_stop_mask()
                        terminal_valid = valid_bool & terminal_mask
                        terminal_values = {
                            "terminal_base_speed_mps": torch.linalg.vector_norm(
                                robot.data.root_lin_vel_b[:, :2], dim=-1
                            ),
                            "terminal_double_support": (contact_count == 2).to(torch.float32),
                            "terminal_root_height_m": robot.data.root_pos_w[:, 2],
                            "terminal_root_tilt_rad": tilt,
                        }
                        for name, value in terminal_values.items():
                            samples[name].append(
                                torch.where(
                                    terminal_valid,
                                    value,
                                    torch.full_like(value, torch.nan),
                                ).detach().cpu()
                            )
                    values.update(
                        flight_fraction=(contact_count == 0).to(torch.float32),
                        single_support_fraction=(contact_count == 1).to(torch.float32),
                        double_support_fraction=(contact_count == 2).to(torch.float32),
                    )
                    knee = robot.data.joint_pos[:, knee_ids]
                    knee_min = torch.where(
                        alive.unsqueeze(-1), torch.minimum(knee_min, knee), knee_min
                    )
                    knee_max = torch.where(
                        alive.unsqueeze(-1), torch.maximum(knee_max, knee), knee_max
                    )
                if ACTION_SCREEN_PHASE65 and step < 3:
                    phase65_replay_hashes.append(
                        {
                            "step": step,
                            "pre_policy_observation": tensor_hash(phase65_pre_policy),
                            "raw_actor_action": tensor_hash(raw_action),
                            "applied_outer_action": tensor_hash(action),
                            "post_root_state": tensor_hash(robot.data.root_state_w),
                            "post_joint_position": tensor_hash(robot.data.joint_pos),
                            "post_joint_velocity": tensor_hash(robot.data.joint_vel),
                        }
                    )
                if ACTION_SCREEN_PHASE66 and step < 3:
                    phase66_replay_hashes.append(
                        {
                            "step": step,
                            "per_env": {
                                name: [tensor_hash(value[index]) for index in range(64)]
                                for name, value in {
                                    "pre_policy_observation": phase66_pre_policy,
                                    "raw_actor_action": raw_action,
                                    "applied_outer_action": action,
                                    "post_root_state": robot.data.root_state_w,
                                    "post_joint_position": robot.data.joint_pos,
                                    "post_joint_velocity": robot.data.joint_vel,
                                }.items()
                            },
                        }
                    )
                if ACTION_SCREEN_PHASE67:
                    processed_delta = phase67_intervention_target - phase67_source_target
                    phase67_contract["processed_target_delta_max_abs_rad"] = max(
                        phase67_contract["processed_target_delta_max_abs_rad"],
                        float(processed_delta.abs().max()),
                    )
                    phase67_contract["finite"] &= bool(
                        torch.isfinite(phase67_source_target).all()
                        and torch.isfinite(phase67_intervention_target).all()
                    )
                    if step in (0, 1, 2, 29, 30, 99, 199):
                        phase67_step_hashes.append(
                            {
                                "step": step,
                                "residual": tensor_hash(phase67_requested_target),
                                "source_processed_target": tensor_hash(phase67_source_target),
                                "intervention_processed_target": tensor_hash(
                                    phase67_intervention_target
                                ),
                                "root_state": tensor_hash(robot.data.root_state_w),
                            }
                        )
                for name, value in values.items():
                    sums[name] += value * valid
                counts += valid
                root_min = torch.where(alive, torch.minimum(root_min, robot.data.root_pos_w[:, 2]), root_min)
                tilt_max = torch.where(alive, torch.maximum(tilt_max, tilt), tilt_max)
                newly_done = alive & done
                survival[newly_done] = (step + 1) * env.step_dt
                terminal |= newly_done
                alive &= ~done
                prev_action = action
                obs = next_obs
        if ACTION_SCREEN:
            groups = screen_groups
        elif PEFT_POSTURE:
            groups = {
                "all": torch.ones_like(zero_mask),
                "ideal": ideal_mask,
                "response": ~ideal_mask,
            }
        else:
            groups = {
                "A_none": zero_mask,
                "B_bounded": ~zero_mask,
                "A_none_ideal": zero_mask & ideal_mask,
                "A_none_response": zero_mask & ~ideal_mask,
                "B_bounded_ideal": ~zero_mask & ideal_mask,
                "B_bounded_response": ~zero_mask & ~ideal_mask,
            }
        report = {
            "phase": PHASE, "mode": args.mode, "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_iter": int(payload.get("iter", -1)),
            "seed": args.seed, "num_envs": 64, "eval_steps": args.eval_steps,
            "posture_variant": (
                "action_sensitivity"
                if ACTION_SCREEN_PHASE62
                else (
                    "hip_pitch_dose"
                    if ACTION_SCREEN_PHASE63
                    else (
                        "phase_conditioned_residual_live_zero"
                        if ACTION_SCREEN_PHASE67
                        else "side_phase_physical_knee_target"
                        if ACTION_SCREEN_PHASE66
                        else "single_support_knee_crossover"
                        if ACTION_SCREEN_PHASE65
                        else ("joint_transition" if PEFT_PHASE61 else POSTURE_VARIANT)
                        if not ACTION_SCREEN_PHASE64
                        else "knee_pitch_mirrored_dose"
                    )
                )
            ),
            "control_dt_s": float(env.step_dt), "horizon_s": float(args.eval_steps * env.step_dt),
            "domain_upper_counts": domain_counts,
            "groups": {
                name: (
                    aggregate_phase60_group(
                        mask, sums, counts, survival, terminal, root_min, tilt_max,
                        samples, knee_min, knee_max,
                    )
                    if POSTURE_METRICS
                    else aggregate_group(mask, sums, counts, survival, terminal, root_min, tilt_max)
                )
                for name, mask in groups.items()
            },
            "finite": bool(all(torch.isfinite(value).all() for value in sums.values())),
        }
        if PEFT_PHASE61:
            report["transition_event"] = {
                "stand_s": 0.0,
                "accelerate_s": 1.0,
                "cruise_s": 4.2,
                "decelerate_s": 2.0,
                "event_end_s": 7.2,
                "terminal_hold_observed_s": args.eval_steps * env.step_dt - 7.2,
                "phase_offset_s": 0.0,
            }
        if ACTION_SCREEN_PHASE62:
            report["action_sensitivity"] = {
                "epsilon_normalized_action": 0.02,
                "partition": {name: int(mask.sum()) for name, mask in screen_groups.items()},
                "dimensions": {
                    "hip_pitch": [0, 6],
                    "knee": [3, 9],
                    "ankle_pitch": [4, 10],
                    "waist_pitch": [13],
                },
                "actuator_domain": "ideal only",
                "checkpoint_modified": False,
            }
        if ACTION_SCREEN_PHASE63:
            report["hip_pitch_dose_screen"] = {
                "normalized_action_doses": [0.0, -0.004, -0.006, -0.008, -0.010, -0.012, -0.016, -0.020],
                "action_dimensions": [0, 6],
                "envs_per_dose": 8,
                "actuator_domain": "ideal only",
                "checkpoint_modified": False,
            }
        if ACTION_SCREEN_PHASE64:
            report["knee_pitch_dose_screen"] = {
                "normalized_action_doses": [0.0, -0.008, -0.010, -0.012],
                "action_dimensions": [3, 9],
                "grouping": "8x8_latin_square_slot_equals_row_plus_column_modulo_8",
                "lane_a_slots": ["base", "m008", "m010", "m012"],
                "lane_b_slots": ["base", "m012", "m010", "m008"],
                "envs_per_lane_group": 8,
                "envs_per_pooled_group": 16,
                "actuator_domain": "ideal only",
                "checkpoint_modified": False,
            }
            report["gait_phase_semantics"] = {
                "observation_suffix": ["sin_2pi_phase", "cos_2pi_phase", "desired_left_contact", "desired_right_contact"],
                "cycle_time_s": 0.8,
                "double_support_fraction": 0.30,
                "measurement_alignment": "post-step physical state grouped by the post-step deployable controller clock; constant dose is phase-invariant",
                "regions": {
                    "double_support_zero": "[0.000,0.075) union [0.925,1.000)",
                    "right_swing_left_support": "[0.075,0.425)",
                    "double_support_half": "[0.425,0.575)",
                    "left_swing_right_support": "[0.575,0.925)",
                },
                "metrics_by_group": aggregate_phase64_semantics(screen_groups, phase64_samples),
            }
        if ACTION_SCREEN_PHASE65:
            region_names = (
                "double_support_zero",
                "right_swing_left_support",
                "double_support_half",
                "left_swing_right_support",
                "standing",
            )
            region_tensor = torch.stack(phase65_pre_step_regions, dim=0)
            knee_excursion_per_env = knee_max - knee_min
            phase65_per_env_records = []
            for env_id in range(64):
                count = max(float(counts[env_id]), 1.0)
                phase65_per_env_records.append(
                    {
                        "env_id": env_id,
                        "row": env_id // 8,
                        "column": env_id % 8,
                        "sequence": "TC" if bool(screen_groups["sequence_TC"][env_id]) else "CT",
                        "received_treatment": bool(phase65_treatment_mask[env_id]),
                        "survival_s": float(survival[env_id]),
                        "terminated": bool(terminal[env_id]),
                        "lateral_abs_mean_m": float(sums["lateral_abs"][env_id] / count),
                        "yaw_abs_mean_rad": float(sums["yaw_abs"][env_id] / count),
                        "velocity_tracking_rmse_mps": math.sqrt(
                            max(0.0, float(sums["velocity_tracking_sq"][env_id] / count))
                        ),
                        "action_delta_abs_mean": float(
                            sums["action_delta_abs"][env_id] / count
                        ),
                        "root_height_min_m": float(root_min[env_id]),
                        "root_tilt_max_rad": float(tilt_max[env_id]),
                        "knee_left_excursion_rad": float(knee_excursion_per_env[env_id, 0]),
                        "knee_right_excursion_rad": float(knee_excursion_per_env[env_id, 1]),
                        "knee_left_right_excursion_abs_diff_rad": float(
                            abs(
                                knee_excursion_per_env[env_id, 0]
                                - knee_excursion_per_env[env_id, 1]
                            )
                        ),
                    }
                )
            report["single_support_knee_crossover"] = {
                "pass_index": _phase65_pass,
                "runner_sha256": sha256(Path(__file__)),
                "normalized_action_dose": -0.010,
                "action_dimensions": [3, 9],
                "configured_action_scale_rad": [
                    float(term._scale[0, 3]),
                    float(term._scale[0, 9]),
                ],
                "allocation": "8x8_checkerboard_sequence_equals_row_plus_column_modulo_2",
                "treatment_sequence": "TC" if _phase65_pass == 0 else "CT",
                "partition": {name: int(mask.sum()) for name, mask in screen_groups.items()},
                "phase_source": "pre-step deployable actor observation suffix and command",
                "measurement_alignment": "post-step physical outcome labeled by the pre-step semantic region that produced its action",
                "regions": list(region_names),
                "pre_step_region_index": region_tensor.tolist(),
                "pre_step_region_counts_per_env": {
                    name: (region_tensor == index).sum(dim=0).tolist()
                    for index, name in enumerate(region_names)
                },
                "initial_fingerprints": phase65_initial_fingerprints,
                "first_three_ds0_replay_hashes": phase65_replay_hashes,
                "bias_contract": phase65_bias_contract,
                "per_env_records": phase65_per_env_records,
                "raw_samples": {
                    name: torch.stack(values, dim=0).tolist()
                    for name, values in phase65_raw_samples.items()
                },
                "actuator_domain": "ideal only",
                "checkpoint_modified": False,
            }
        if ACTION_SCREEN_PHASE66:
            region_names = (
                "double_support_zero",
                "right_swing_left_support",
                "double_support_half",
                "left_swing_right_support",
                "standing",
            )
            region_tensor = torch.stack(phase66_pre_step_regions, dim=0)
            knee_excursion_per_env = knee_max - knee_min
            per_env_records = []
            for env_id in range(64):
                count = max(float(counts[env_id]), 1.0)
                per_env_records.append(
                    {
                        "env_id": env_id,
                        "row": env_id // 8,
                        "column": env_id % 8,
                        "condition": phase66_condition_names[int(phase66_condition_slot[env_id])],
                        "survival_s": float(survival[env_id]),
                        "terminated": bool(terminal[env_id]),
                        "lateral_abs_mean_m": float(sums["lateral_abs"][env_id] / count),
                        "yaw_abs_mean_rad": float(sums["yaw_abs"][env_id] / count),
                        "velocity_tracking_rmse_mps": math.sqrt(
                            max(0.0, float(sums["velocity_tracking_sq"][env_id] / count))
                        ),
                        "action_delta_abs_mean": float(sums["action_delta_abs"][env_id] / count),
                        "root_height_min_m": float(root_min[env_id]),
                        "root_tilt_max_rad": float(tilt_max[env_id]),
                        "knee_left_excursion_rad": float(knee_excursion_per_env[env_id, 0]),
                        "knee_right_excursion_rad": float(knee_excursion_per_env[env_id, 1]),
                        "knee_left_right_excursion_abs_diff_rad": float(
                            abs(knee_excursion_per_env[env_id, 0] - knee_excursion_per_env[env_id, 1])
                        ),
                    }
                )
            report["side_phase_physical_knee_target"] = {
                "pass_index": _phase66_pass,
                "runner_sha256": sha256(Path(__file__)),
                "candidate_enabled": _phase66_pass == 0,
                "requested_target_offset_rad": -0.003,
                "allocation": "8x8_latin_square_condition_equals_row_plus_column_modulo_8",
                "condition_names": list(phase66_condition_names),
                "condition_slot_by_env": phase66_condition_slot.tolist(),
                "phase_source": "pre-step deployable actor observation suffix and command",
                "measurement_alignment": "post-step physical outcome labeled by the pre-step semantic region that produced its target",
                "pre_step_region_index": region_tensor.tolist(),
                "pre_step_region_counts_per_env": {
                    name: (region_tensor == index).sum(dim=0).tolist()
                    for index, name in enumerate(region_names)
                },
                "initial_fingerprints": phase66_initial_fingerprints,
                "first_three_ds0_per_env_replay_hashes": phase66_replay_hashes,
                "contract": phase66_contract,
                "per_env_records": per_env_records,
                "raw_samples": {
                    name: torch.stack(values, dim=0).tolist()
                    for name, values in phase66_raw_samples.items()
                },
                "actuator_domain": "ideal only",
                "checkpoint_modified": False,
            }
        if ACTION_SCREEN_PHASE67:
            report["phase_conditioned_residual_live_zero"] = {
                "runner_sha256": sha256(Path(__file__)),
                "module_path": "src/cwi_x2/phase_conditioned_knee_residual.py",
                "module_sha256": sha256(
                    REPO / "src/cwi_x2/phase_conditioned_knee_residual.py"
                ),
                "trainable_manifest": phase67_residual.trainable_manifest(),
                "initial_fingerprints": phase67_initial_fingerprints,
                "contract": phase67_contract,
                "sampled_step_hashes": phase67_step_hashes,
                "environment_control_steps": args.eval_steps,
                "optimizer_steps": 0,
                "checkpoint_count": 0,
                "checkpoint_modified": False,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def save_weight_only(
    path: Path,
    model: torch.nn.Module,
    *,
    iteration: int,
    infos: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "iter": iteration, "infos": infos}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def save_or_validate_posture_source(
    path: Path,
    model: torch.nn.Module,
    *,
    iteration: int,
    infos: dict,
) -> None:
    """Keep one shared immutable posture source instead of duplicating it."""

    if not path.exists():
        save_weight_only(path, model, iteration=iteration, infos=infos)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    existing = payload.get("model_state_dict", {})
    current = model.state_dict()
    if existing.keys() != current.keys() or any(
        not torch.equal(existing[name].cpu(), value.detach().cpu())
        for name, value in current.items()
    ):
        raise RuntimeError("posture source exists with different tensor state")
    if int(payload.get("iter", -1)) != iteration:
        raise RuntimeError("posture source iteration differs")


def train() -> None:
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(evaluation=False))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs, _, _, _, domain_counts = validate_live_contract(env, wrapped)
        agent_cfg = X2LowerVelocityFlatPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        agent_cfg.num_steps_per_env = TRAIN_STEPS
        agent_cfg.policy.init_noise_std = 0.4
        agent_cfg.algorithm.entropy_coef = 0.004
        agent_cfg.algorithm.num_learning_epochs = 5
        agent_cfg.algorithm.num_mini_batches = 4
        agent_cfg.algorithm.learning_rate = 0.001
        agent_cfg.algorithm.schedule = "adaptive"
        agent_cfg.algorithm.desired_kl = 0.01
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        incoming_has_lora = any("lora_" in name for name in payload["model_state_dict"])
        if PEFT_PROTECTED and incoming_has_lora:
            lora_manifest = inject_standard93d_lora(runner.alg.policy, rank=4, alpha=4.0)
            runner.alg.policy.load_state_dict(payload["model_state_dict"], strict=True)
        else:
            runner.alg.policy.load_state_dict(payload["model_state_dict"], strict=True)
        lora_manifest = None
        if PEFT_PROTECTED:
            if not incoming_has_lora:
                lora_manifest = inject_standard93d_lora(runner.alg.policy, rank=4, alpha=4.0)
            else:
                lora_manifest = {
                    "rank": 4, "alpha": 4.0,
                    "actor_scopes": [f"actor.{index}" for index in (0, 2, 4, 6)],
                    "critic_scopes": [f"critic.{index}" for index in (0, 2, 4, 6)],
                    "trainable_names": sorted(name for name, parameter in runner.alg.policy.named_parameters() if parameter.requires_grad),
                    "trainable_parameters": sum(parameter.numel() for parameter in runner.alg.policy.parameters() if parameter.requires_grad),
                    "total_parameters": sum(parameter.numel() for parameter in runner.alg.policy.parameters()),
                }
            expected_trainable = sorted(
                f"{branch}.{index}.lora_{suffix}"
                for branch in ("actor", "critic")
                for index in (0, 2, 4, 6)
                for suffix in ("A", "B")
            )
            if lora_manifest["trainable_names"] != expected_trainable:
                raise RuntimeError("Phase58 trainable names differ from Phase57")
            runner.alg.learning_rate = 5.0e-5
            runner.alg.schedule = "fixed"
            runner.alg.optimizer = torch.optim.Adam(
                [parameter for parameter in runner.alg.policy.parameters() if parameter.requires_grad],
                lr=5.0e-5,
            )
            if incoming_has_lora:
                optimizer_state = payload.get("optimizer_state_dict")
                if optimizer_state is None:
                    raise RuntimeError("Phase59 cumulative update checkpoint lacks optimizer state")
                runner.alg.optimizer.load_state_dict(optimizer_state)
        else:
            runner.alg.policy.std.requires_grad_(False)
        source_model = copy.deepcopy(runner.alg.policy).eval()
        original_model = build_model(obs, env.device).eval()
        original_payload = torch.load(ORIGINAL, map_location=env.device, weights_only=False)
        original_model.load_state_dict(original_payload["model_state_dict"], strict=True)
        if PEFT_PROTECTED:
            inject_standard93d_lora(original_model, rank=4, alpha=4.0)
        source_hash = state_hash(source_model)
        dense_before = tensor_map_hash(dense_tensor_map(runner.alg.policy))
        std_before = tensor_hash(runner.alg.policy.std)
        trainable_names = sorted(name for name, p in runner.alg.policy.named_parameters() if p.requires_grad)
        frozen_names = sorted(name for name, p in runner.alg.policy.named_parameters() if not p.requires_grad)
        source_infos = {
            "phase": PHASE,
            "role": "immutable fresh source",
            "original_sha256": sha256(ORIGINAL),
        }
        if PEFT_POSTURE:
            save_or_validate_posture_source(
                args.source_output,
                source_model,
                iteration=int(payload.get("iter", 2600)),
                infos=source_infos,
            )
        else:
            save_weight_only(
                args.source_output, source_model, iteration=int(payload.get("iter", 2600)),
                infos=source_infos,
                optimizer=runner.alg.optimizer if PEFT_PHASE59 else None,
            )
        fixed_obs = {"policy": obs["policy"].clone(), "critic": obs["critic"].clone()}
        with torch.no_grad():
            source_action = source_model.act_inference(fixed_obs).clone()
            source_value = source_model.evaluate(fixed_obs).clone()
            original_action = original_model.act_inference(fixed_obs).clone()
            original_value = original_model.evaluate(fixed_obs).clone()
        step_counter = {"count": 0}
        original_step = runner.alg.optimizer.step

        def counted_step(*step_args, **step_kwargs):
            step_counter["count"] += 1
            return original_step(*step_args, **step_kwargs)

        runner.alg.optimizer.step = counted_step
        obs_train = wrapped.get_observations().to(args.device)
        runner.train_mode()
        command_vx_trace = []
        reset_events = 0

        def collect_transition() -> None:
            nonlocal obs_train, reset_events
            command_vx_trace.append(
                float(env.command_manager.get_command("base_velocity")[:, 0].mean())
            )
            action = runner.alg.act(obs_train)
            obs_train, reward, done, extras = wrapped.step(action.to(wrapped.device))
            obs_train, reward, done = (
                obs_train.to(args.device), reward.to(args.device), done.to(args.device)
            )
            runner.alg.process_env_step(obs_train, reward, done, extras)
            reset_events += int(done.reshape(-1).bool().sum())

        with torch.inference_mode():
            if PEFT_PHASE61:
                for _ in range(512):
                    collect_transition()
            else:
                for _ in range(24):
                    collect_transition()
            runner.alg.compute_returns(obs_train)
        loss_dict = runner.alg.update()
        final_model = runner.alg.policy.eval()
        dense_after = tensor_map_hash(dense_tensor_map(final_model))
        std_after = tensor_hash(final_model.std)
        with torch.no_grad():
            final_action = final_model.act_inference(fixed_obs).clone()
            final_value = final_model.evaluate(fixed_obs).clone()
        sigma = source_model.std.detach().reshape(1, -1)
        incremental_kl = 0.5 * ((final_action - source_action) / sigma).square().sum(-1)
        original_sigma = original_model.std.detach().reshape(1, -1)
        fixed_kl = 0.5 * ((final_action - original_action) / original_sigma).square().sum(-1)
        finite = (
            all(math.isfinite(float(value)) for value in loss_dict.values())
            and all(torch.isfinite(value).all() for value in final_model.state_dict().values())
            and all(p.grad is None or torch.isfinite(p.grad).all() for p in final_model.parameters())
            and torch.isfinite(fixed_kl).all() and torch.isfinite(incremental_kl).all()
        )
        if step_counter["count"] != 20:
            raise RuntimeError(f"Phase56 expected 20 optimizer minibatches, got {step_counter['count']}")
        if std_before != std_after:
            raise RuntimeError(f"Phase{PHASE} frozen std changed")
        if PEFT_PROTECTED and dense_before != dense_after:
            raise RuntimeError(f"Phase{PHASE} frozen dense hash changed")
        save_weight_only(
            args.final_output, final_model, iteration=int(payload.get("iter", 2600)) + 1,
            infos={"phase": PHASE, "role": "single-update final", "source_sha256": sha256(args.source_output)},
            optimizer=runner.alg.optimizer if PEFT_PHASE59 else None,
        )
        report = {
            "phase": PHASE, "mode": "train", "decision": "UPDATE_FINITE" if finite else "UPDATE_NONFINITE",
            "seed": args.seed, "num_envs": 64, "steps_per_env": TRAIN_STEPS,
            "posture_variant": "joint_transition" if PEFT_PHASE61 else POSTURE_VARIANT,
            "posture_reward_weights": (
                {
                    "signed_backward_pitch": (
                        -0.5 if PEFT_PHASE61 else POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT][0]
                    ),
                    "actual_support_com": (
                        -0.5 if PEFT_PHASE61 else POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT][1]
                    ),
                }
                if PEFT_POSTURE else None
            ),
            "transitions": 64 * TRAIN_STEPS, "learning_epochs": 5, "mini_batches": 4,
            "optimizer_steps": step_counter["count"], "std_frozen": True,
            "source_checkpoint": str(args.source_output), "source_checkpoint_sha256": sha256(args.source_output),
            "final_checkpoint": str(args.final_output), "final_checkpoint_sha256": sha256(args.final_output),
            "original_checkpoint": str(ORIGINAL), "original_checkpoint_sha256": sha256(ORIGINAL),
            "source_model_hash": source_hash, "final_model_hash": state_hash(final_model),
            "std_hash_before": std_before, "std_hash_after": std_after,
            "dense_hash_before": dense_before, "dense_hash_after": dense_after,
            "trainable_names": trainable_names, "frozen_names": frozen_names,
            "lora_manifest": lora_manifest,
            "learning_rate": 5.0e-5 if PEFT_PROTECTED else 0.001,
            "learning_rate_schedule": "fixed" if PEFT_PROTECTED else "adaptive",
            "domain_upper_counts": domain_counts, "losses": {name: float(value) for name, value in loss_dict.items()},
            "fixed_source_retention": {
                "action_max_abs": float((final_action - original_action).abs().max()),
                "value_max_abs": float((final_value - original_value).abs().max()),
                "kl_mean": float(fixed_kl.mean()), "kl_max": float(fixed_kl.max()),
            },
            "incremental_retention": {
                "action_max_abs": float((final_action - source_action).abs().max()),
                "value_max_abs": float((final_value - source_value).abs().max()),
                "kl_mean": float(incremental_kl.mean()), "kl_max": float(incremental_kl.max()),
            },
            "finite": bool(finite), "checkpoint_count": 2,
            "environment_control_steps": TRAIN_STEPS,
            "update_index": args.update_index,
        }
        if PEFT_PHASE61:
            sample_steps = (0, 25, 50, 260, 310, 360, 460, 511)
            report["transition_event_coverage"] = {
                "command_vx_mean_by_step": {
                    str(index): command_vx_trace[index] for index in sample_steps
                },
                "command_vx_max": max(command_vx_trace),
                "terminal_zero_steps": sum(
                    abs(value) <= 1.0e-6 for value in command_vx_trace
                ),
                "reset_events": reset_events,
                "static_audit_pass": True,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not finite:
            raise RuntimeError("Phase56 update produced non-finite state")
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def main() -> None:
    if any(not path.is_file() or sha256(path) != expected for path, expected in EXPECTED.items()):
        raise RuntimeError(f"Phase{PHASE} immutable artifact hash guard failed")
    required = {
        "CWI_UPPER_MOTION": str(UPPER),
        "CWI_UPPER_ZERO_FRACTION": "1.0" if PEFT_PHASE60 or PEFT_PHASE61 or ACTION_SCREEN else "0.50",
        "CWI_UPPER_DETERMINISTIC_SPLIT": "0" if PEFT_PHASE60 or PEFT_PHASE61 or ACTION_SCREEN else "1",
        "CWI_UPPER_SPLIT_MODE": "contiguous" if PEFT_PHASE60 or PEFT_PHASE61 or ACTION_SCREEN else "interleaved",
        "CWI_UPPER_SCALE": "0.25", "CWI_UPPER_TIME_SCALE": "1.0",
        "CWI_UPPER_LOOP": "1", "CWI_UPPER_MAX_EXCURSION_RAD": "0.12",
        "CWI_UPPER_MAX_VELOCITY_RADPS": "0.20",
    }
    for name, value in required.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Phase{PHASE} environment mismatch: {name}")
    if args.mode in {"eval", "screen"}:
        evaluate()
    else:
        if PEFT_PHASE60 and POSTURE_VARIANT == "A":
            raise RuntimeError("Phase60 group A is frozen evaluation-only")
        if args.checkpoint.resolve() != ORIGINAL.resolve() and not PEFT_PHASE59:
            raise RuntimeError("Phase56 train must fresh-start from original Stage219 PT")
        train()


if __name__ == "__main__":
    failure = None
    try:
        main()
    except Exception as exc:
        failure = exc
        traceback.print_exc()
    finally:
        simulation_app.close()
    if failure is not None:
        raise failure
