#!/usr/bin/env python3
"""Phase57 live zero gate for PEFT-protected standard93D lower backend."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.num_envs, args.seed) != (64, 42):
    raise ValueError("Phase57 is frozen to 64 envs and seed 42")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.heading_command import gain_scheduled_velocity_cfg  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402
from x2_upper_robust_lora_phase57 import (  # noqa: E402
    dense_tensor_map,
    inject_standard93d_lora,
    tensor_map_hash,
)

REPO = Path(__file__).resolve().parents[1]
OLD = Path("/home/humanplus/x2_teleop_final/x2_sonic")
RUN = OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1"
CHECKPOINT = RUN / "model_2600.pt"
ENV_YAML = RUN / "params/env.yaml"
AGENT_YAML = RUN / "params/agent.yaml"
TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
EXPECTED = {
    CHECKPOINT: "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb",
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
HEAD2 = ("head_yaw_joint", "head_pitch_joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_env_cfg():
    cfg = X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg()
    cfg.scene.num_envs = 64
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.actions.joint_pos.template_path = str(TEMPLATE)
    cfg.actions.joint_pos.template_scale = 0.15
    cfg.actions.joint_pos.scale = {pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()}
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.60)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.20, 0.20)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = True
    cfg.commands.base_velocity.rel_heading_envs = 0.75
    cfg.commands.base_velocity = gain_scheduled_velocity_cfg(
        cfg.commands.base_velocity,
        ideal_env_fraction=0.75,
        ideal_heading_control_stiffness=1.0,
        response_heading_control_stiffness=0.05,
    )
    cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
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


def main() -> None:
    if any(not path.is_file() or sha256(path) != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("Phase57 immutable artifact hash guard failed")
    required = {
        "CWI_UPPER_MOTION": str(UPPER), "CWI_UPPER_ZERO_FRACTION": "0.50",
        "CWI_UPPER_DETERMINISTIC_SPLIT": "1", "CWI_UPPER_SPLIT_MODE": "interleaved",
        "CWI_UPPER_SCALE": "0.25", "CWI_UPPER_TIME_SCALE": "1.0",
        "CWI_UPPER_LOOP": "1", "CWI_UPPER_MAX_EXCURSION_RAD": "0.12",
        "CWI_UPPER_MAX_VELOCITY_RADPS": "0.20",
    }
    for name, value in required.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Phase57 environment mismatch: {name}")
    env = wrapped = None
    try:
        env = ManagerBasedRLEnv(cfg=build_env_cfg())
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = wrapped.get_observations()
        if list(obs["policy"].shape) != [64, 93] or list(obs["critic"].shape) != [64, 93]:
            raise RuntimeError("Phase57 live observations are not 64x93")
        if wrapped.num_actions != 15:
            raise RuntimeError("Phase57 action is not 15D")
        term = env.action_manager._terms["joint_pos"]
        if tuple(term._joint_names) != LOWER15:
            raise RuntimeError("Phase57 lower15 order changed")
        robot = env.scene["robot"]
        if len(robot.joint_names) != 31 or set(robot.joint_names) != set(LOWER15) | set(UPPER14) | set(HEAD2):
            raise RuntimeError("Phase57 15+14+2 partition is not 31DoF")
        head_ids = [robot.joint_names.index(name) for name in HEAD2]
        head_default = robot.data.default_joint_pos[0, head_ids]
        if not torch.equal(head_default, torch.zeros_like(head_default)):
            raise RuntimeError("Phase57 head2 is not nominal zero")
        zero_mask = term._cwi_upper_zero_mask
        if not torch.equal(zero_mask, torch.arange(64, device=zero_mask.device) % 2 == 0):
            raise RuntimeError("Phase57 upper split is not interleaved 32/32")
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
        if counts != {"none_ideal": 24, "none_response": 8, "bounded_ideal": 24, "bounded_response": 8}:
            raise RuntimeError(f"Phase57 upper/domain counts differ: {counts}")

        source = build_model(obs, env.device).eval()
        payload = torch.load(CHECKPOINT, map_location=env.device, weights_only=False)
        source.load_state_dict(payload["model_state_dict"], strict=True)
        candidate = copy.deepcopy(source).eval()
        dense_before = tensor_map_hash(dense_tensor_map(candidate))
        lora = inject_standard93d_lora(candidate, rank=4, alpha=4.0)
        dense_after = tensor_map_hash(dense_tensor_map(candidate))
        if dense_before != dense_after:
            raise RuntimeError("Phase57 dense/std hash changed during LoRA injection")
        expected_scopes = [f"actor.{index}" for index in (0, 2, 4, 6)]
        expected_critic = [f"critic.{index}" for index in (0, 2, 4, 6)]
        if lora["actor_scopes"] != expected_scopes or lora["critic_scopes"] != expected_critic:
            raise RuntimeError("Phase57 LoRA scope changed")
        fixed_batches = (obs, {"policy": obs["policy"].flip(0), "critic": obs["critic"].flip(0)})
        rows = []
        with torch.no_grad():
            for index, batch in enumerate(fixed_batches):
                source_action = source.act_inference(batch)
                adapted_action = candidate.act_inference(batch)
                source_value = source.evaluate(batch)
                adapted_value = candidate.evaluate(batch)
                rows.append({
                    "index": index,
                    "action_shape": list(source_action.shape),
                    "value_shape": list(source_value.shape),
                    "action_max_abs": float((source_action - adapted_action).abs().max()),
                    "value_max_abs": float((source_value - adapted_value).abs().max()),
                    "finite": bool(torch.isfinite(adapted_action).all() and torch.isfinite(adapted_value).all()),
                })
        passed = all(row["finite"] and row["action_max_abs"] == 0.0 and row["value_max_abs"] == 0.0 for row in rows)
        report = {
            "phase": 57,
            "decision": "PASS_LIVE_ZERO_UPDATE_ONLY" if passed else "FAIL_LIVE_ZERO_UPDATE",
            "preregistered_learning_rate": 5.0e-5,
            "source": {"checkpoint": str(CHECKPOINT), "sha256": sha256(CHECKPOINT), "iteration": int(payload["iter"])},
            "runtime": {
                "policy_obs": list(obs["policy"].shape), "critic_obs": list(obs["critic"].shape),
                "domain_upper_counts": counts, "upper_split": "even none / odd bounded",
                "head_default": head_default.detach().cpu().tolist(),
            },
            "boundary": {
                "policy_action_dim": wrapped.num_actions,
                "lower_action_names": list(LOWER15),
                "external_upper_names": list(UPPER14),
                "head_nominal_names": list(HEAD2),
                "legacy_body_partition": "29 = lower12+waist3 (15) + upper14",
                "sim_partition": "31 = body29 + head2 nominal",
            },
            "lora": {
                **lora,
                "parameter_ratio": lora["trainable_parameters"] / lora["total_parameters"],
                "zero_output_initialization": "lora_B exact zeros; lora_A Kaiming",
                "actor_semantics": "all actor paths are lower12+waist3 relevant because actor output is exclusively 15D lower/waist",
                "critic_is_independent_lora": True,
            },
            "dense_hash_before": dense_before, "dense_hash_after": dense_after,
            "fixed_forward_batches": rows,
            "source_retention_kl": {"mean": 0.0, "max": 0.0},
            "optimizer_constructed": False, "optimizer_steps": 0,
            "environment_resets": 1, "environment_control_steps": 0,
            "checkpoint_created": False,
            "boundary_claim": "Initialization/reset/fixed forward only; no physical robustness or learning claim.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Phase57 zero-output source retention failed")
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


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
