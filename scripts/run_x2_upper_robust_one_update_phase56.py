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
parser.add_argument("--mode", choices=("train", "eval"), required=True)
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
if _phase60_variant is not None:
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

PEFT_PHASE58 = os.environ.get("CWI_PHASE58_PEFT", "0") == "1"
PEFT_PHASE59 = os.environ.get("CWI_PHASE59_PEFT", "0") == "1"
POSTURE_VARIANT = os.environ.get("CWI_PHASE60_POSTURE_VARIANT")
if POSTURE_VARIANT is not None and POSTURE_VARIANT not in {"A", "B", "C"}:
    raise ValueError("CWI_PHASE60_POSTURE_VARIANT must be A, B, or C")
PEFT_PHASE60 = POSTURE_VARIANT is not None
PEFT_PROTECTED = PEFT_PHASE58 or PEFT_PHASE59 or PEFT_PHASE60
PHASE = 60 if PEFT_PHASE60 else (59 if PEFT_PHASE59 else (58 if PEFT_PHASE58 else 56))
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
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35) if evaluation else (0.25, 0.60)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) if evaluation else (-0.20, 0.20)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = not evaluation
    cfg.commands.base_velocity.rel_heading_envs = 0.0 if evaluation else 0.75
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity = gain_scheduled_velocity_cfg(
        cfg.commands.base_velocity,
        ideal_env_fraction=0.75,
        ideal_heading_control_stiffness=1.0,
        response_heading_control_stiffness=0.05,
    )
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
    }
    if evaluation:
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
    if PEFT_PHASE60:
        pitch_weight, support_weight = POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT]
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
    cfg.observations.policy.enable_corruption = False
    _apply_x2_actuator_response(
        cfg.scene.robot,
        {"enabled": True, "profile": "session03_session04_group", "randomize": False,
         "strength": 1.0, "filter_strength": 1.0, "delay_strength": 1.0,
         "include_ideal_endpoint": False, "ideal_env_fraction": 0.75,
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
        if PEFT_PHASE60
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
        {"none_ideal": 48, "none_response": 16, "bounded_ideal": 0, "bounded_response": 0}
        if PEFT_PHASE60
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


def aggregate_phase60_group(
    mask, sums, counts, survival, terminal_rate, root_min, tilt_max,
    samples, knee_min, knee_max,
):
    result = aggregate_group(mask, sums, counts, survival, terminal_rate, root_min, tilt_max)
    result["signed_pitch_rad"] = _sample_summary(samples["signed_pitch_rad"], mask)
    result["com_support_outside_m"] = _sample_summary(samples["com_support_outside_m"], mask)
    result["stance_slip_mps"] = _sample_summary(samples["stance_slip_mps"], mask)
    result["swing_sole_clearance_m"] = _sample_summary(samples["swing_sole_clearance_m"], mask)
    result["knee_excursion_rad_mean"] = float((knee_max[mask] - knee_min[mask]).mean())
    return result


def evaluate() -> None:
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(evaluation=True))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs, term, zero_mask, ideal_mask, domain_counts = validate_live_contract(env, wrapped)
        model = build_model(obs, env.device).eval()
        payload = load_state(model, args.checkpoint)
        model.std.requires_grad_(False)
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
        metric_names = (
            "velocity_tracking_sq", "yaw_tracking_sq", "lateral_abs",
            "yaw_abs", "upper_tracking_sq", "action_abs", "action_delta_abs", "reward",
        )
        if PEFT_PHASE60:
            metric_names += ("flight_fraction", "single_support_fraction", "double_support_fraction")
        sums = {name: torch.zeros(64, device=env.device) for name in metric_names}
        counts = torch.zeros(64, device=env.device)
        samples = {
            "signed_pitch_rad": [],
            "com_support_outside_m": [],
            "stance_slip_mps": [],
            "swing_sole_clearance_m": [],
        }
        foot_ids = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]
        knee_ids = [robot.joint_names.index("left_knee_joint"), robot.joint_names.index("right_knee_joint")]
        knee_min = robot.data.joint_pos[:, knee_ids].amin(dim=-1).clone()
        knee_max = robot.data.joint_pos[:, knee_ids].amax(dim=-1).clone()
        with torch.inference_mode():
            for step in range(args.eval_steps):
                action = model.act_inference(obs)
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
                if PEFT_PHASE60:
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
                    valid_bool = alive.clone()
                    for name, value in {
                        "signed_pitch_rad": pitch,
                        "com_support_outside_m": outside,
                        "stance_slip_mps": stance_slip,
                        "swing_sole_clearance_m": swing_clearance,
                    }.items():
                        samples[name].append(
                            torch.where(valid_bool, value, torch.full_like(value, torch.nan)).detach().cpu()
                        )
                    values.update(
                        flight_fraction=(contact_count == 0).to(torch.float32),
                        single_support_fraction=(contact_count == 1).to(torch.float32),
                        double_support_fraction=(contact_count == 2).to(torch.float32),
                    )
                    knee = robot.data.joint_pos[:, knee_ids]
                    knee_min = torch.where(alive, torch.minimum(knee_min, knee.amin(dim=-1)), knee_min)
                    knee_max = torch.where(alive, torch.maximum(knee_max, knee.amax(dim=-1)), knee_max)
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
        if PEFT_PHASE60:
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
            "phase": PHASE, "mode": "eval", "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_iter": int(payload.get("iter", -1)),
            "seed": args.seed, "num_envs": 64, "eval_steps": args.eval_steps,
            "posture_variant": POSTURE_VARIANT,
            "control_dt_s": float(env.step_dt), "horizon_s": float(args.eval_steps * env.step_dt),
            "domain_upper_counts": domain_counts,
            "groups": {
                name: (
                    aggregate_phase60_group(
                        mask, sums, counts, survival, terminal, root_min, tilt_max,
                        samples, knee_min, knee_max,
                    )
                    if PEFT_PHASE60
                    else aggregate_group(mask, sums, counts, survival, terminal, root_min, tilt_max)
                )
                for name, mask in groups.items()
            },
            "finite": bool(all(torch.isfinite(value).all() for value in sums.values())),
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


def save_or_validate_phase60_source(
    path: Path,
    model: torch.nn.Module,
    *,
    iteration: int,
    infos: dict,
) -> None:
    """Keep one shared immutable Phase60 source instead of duplicating it."""

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
        raise RuntimeError("Phase60 shared source exists with different tensor state")
    if int(payload.get("iter", -1)) != iteration:
        raise RuntimeError("Phase60 shared source iteration differs")


def train() -> None:
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(evaluation=False))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs, _, _, _, domain_counts = validate_live_contract(env, wrapped)
        agent_cfg = X2LowerVelocityFlatPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        agent_cfg.num_steps_per_env = 24
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
        if PEFT_PHASE60:
            save_or_validate_phase60_source(
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
        with torch.inference_mode():
            for _ in range(24):
                action = runner.alg.act(obs_train)
                obs_train, reward, done, extras = wrapped.step(action.to(wrapped.device))
                obs_train, reward, done = (
                    obs_train.to(args.device), reward.to(args.device), done.to(args.device)
                )
                runner.alg.process_env_step(obs_train, reward, done, extras)
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
            "seed": args.seed, "num_envs": 64, "steps_per_env": 24,
            "posture_variant": POSTURE_VARIANT,
            "posture_reward_weights": (
                {
                    "signed_backward_pitch": POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT][0],
                    "actual_support_com": POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT][1],
                }
                if PEFT_PHASE60 else None
            ),
            "transitions": 1536, "learning_epochs": 5, "mini_batches": 4,
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
            "environment_control_steps": 24,
            "update_index": args.update_index,
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
        "CWI_UPPER_ZERO_FRACTION": "1.0" if PEFT_PHASE60 else "0.50",
        "CWI_UPPER_DETERMINISTIC_SPLIT": "0" if PEFT_PHASE60 else "1",
        "CWI_UPPER_SPLIT_MODE": "contiguous" if PEFT_PHASE60 else "interleaved",
        "CWI_UPPER_SCALE": "0.25", "CWI_UPPER_TIME_SCALE": "1.0",
        "CWI_UPPER_LOOP": "1", "CWI_UPPER_MAX_EXCURSION_RAD": "0.12",
        "CWI_UPPER_MAX_VELOCITY_RADPS": "0.20",
    }
    for name, value in required.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Phase{PHASE} environment mismatch: {name}")
    if args.mode == "eval":
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
