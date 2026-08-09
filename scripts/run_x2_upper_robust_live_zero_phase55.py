#!/usr/bin/env python3
"""Live zero-update gate for the standard Stage219 93D lower actor."""

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
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.num_envs, args.seed) != (64, 42):
    raise ValueError("Phase55 is frozen to 64 envs and seed 42")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg  # noqa: E402
from gear_sonic.envs.x2_velocity.heading_command import gain_scheduled_velocity_cfg  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import _apply_x2_actuator_response  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OLD = Path("/home/humanplus/x2_teleop_final/x2_sonic")
RUN = OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1"
CHECKPOINT = RUN / "model_2600.pt"
ENV_YAML = RUN / "params/env.yaml"
AGENT_YAML = RUN / "params/agent.yaml"
TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
LOWER15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
HEAD2 = ("head_yaw_joint", "head_pitch_joint")
EXPECTED = {
    CHECKPOINT: "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb",
    ENV_YAML: "f702a358bdbc1df94ac2a54b83aa4f6d7c98c76b091ac05a65fd074c44e6f9d7",
    AGENT_YAML: "38d462ad726e0e74d797f8a0ce3799aaadc02737e6e14a5cdf3443f7da0a8368",
    TEMPLATE: "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d",
    UPPER: "71db36d0206c44da05640f6e3616f918945524e891a5df8051533fcbeb2ab2ef",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
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
    cfg.actions.joint_pos.scale = {
        pattern: 2.0 * scale for pattern, scale in cfg.actions.joint_pos.scale.items()
    }
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


def main() -> None:
    print("[Phase55] entering immutable guards", flush=True)
    if any(not path.is_file() or sha256(path) != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("Phase55 immutable artifact hash guard failed")
    required_env = {
        "CWI_UPPER_MOTION": str(UPPER), "CWI_UPPER_ZERO_FRACTION": "0.50",
        "CWI_UPPER_DETERMINISTIC_SPLIT": "1",
        "CWI_UPPER_SPLIT_MODE": "interleaved",
        "CWI_UPPER_SCALE": "0.25", "CWI_UPPER_TIME_SCALE": "1.0",
        "CWI_UPPER_LOOP": "1", "CWI_UPPER_MAX_EXCURSION_RAD": "0.12",
        "CWI_UPPER_MAX_VELOCITY_RADPS": "0.20",
    }
    for name, value in required_env.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Phase55 environment mismatch: {name}")

    env = wrapped = None
    try:
        print("[Phase55] constructing live ManagerBasedRLEnv", flush=True)
        env = ManagerBasedRLEnv(cfg=build_env_cfg())
        print("[Phase55] live environment ready", flush=True)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)  # one reset, zero step
        print("[Phase55] wrapper reset ready", flush=True)
        obs = wrapped.get_observations()
        if list(obs["policy"].shape) != [64, 93] or list(obs["critic"].shape) != [64, 93]:
            raise RuntimeError("live actor/critic observation is not 64x93")
        expected_obs_terms = (
            "base_lin_vel", "base_ang_vel", "projected_gravity",
            "velocity_commands", "joint_pos", "joint_vel", "actions", "gait_phase",
        )
        policy_terms = tuple(env.observation_manager.active_terms["policy"])
        critic_terms = tuple(env.observation_manager.active_terms["critic"])
        policy_term_dims = [list(dim) for dim in env.observation_manager.group_obs_term_dim["policy"]]
        critic_term_dims = [list(dim) for dim in env.observation_manager.group_obs_term_dim["critic"]]
        if policy_terms != expected_obs_terms or critic_terms != expected_obs_terms:
            raise RuntimeError("live 93D observation term contract differs")
        if policy_term_dims[4:6] != [[31], [31]] or critic_term_dims[4:6] != [[31], [31]]:
            raise RuntimeError("realized 31DoF q/dq is not retained in live 93D observations")
        if wrapped.num_actions != 15:
            raise RuntimeError("live action dimension is not 15")
        action_term = env.action_manager._terms["joint_pos"]
        action_names = tuple(action_term._joint_names)
        if action_names != LOWER15:
            raise RuntimeError(f"live lower action order differs: {action_names}")
        robot = env.scene["robot"]
        if len(robot.joint_names) != 31:
            raise RuntimeError("live articulation is not 31 DoF")
        head_ids = [robot.joint_names.index(name) for name in HEAD2]
        head_default = robot.data.default_joint_pos[0, head_ids].detach().cpu()
        if not torch.equal(head_default, torch.zeros_like(head_default)):
            raise RuntimeError("head2 is not nominal zero")

        zero_mask = action_term._cwi_upper_zero_mask
        if (len(zero_mask), int(zero_mask.sum())) != (64, 32):
            raise RuntimeError("upper fixed/active sampler is not 32/32")
        if not torch.equal(zero_mask, torch.arange(64, device=zero_mask.device) % 2 == 0):
            raise RuntimeError("upper split is not the frozen even/odd interleave")
        response_actuator = robot.actuators["legs"]
        response_alpha = response_actuator._position_alpha.reshape(64, -1)[:, 0]
        response_lag = response_actuator.positions_delay_buffer.time_lags.reshape(64)
        ideal_mask = torch.isclose(response_alpha, torch.ones_like(response_alpha)) & (response_lag == 0)
        domain_upper_counts = {
            "none_ideal": int((zero_mask & ideal_mask).sum()),
            "none_response": int((zero_mask & ~ideal_mask).sum()),
            "bounded_ideal": int((~zero_mask & ideal_mask).sum()),
            "bounded_response": int((~zero_mask & ~ideal_mask).sum()),
        }
        if domain_upper_counts != {
            "none_ideal": 24, "none_response": 8,
            "bounded_ideal": 24, "bounded_response": 8,
        }:
            raise RuntimeError(f"upper x actuator-domain split is not balanced: {domain_upper_counts}")
        now = action_term._cwi_reference_time()
        reset_delta = action_term._cwi_bounded_intent_delta(now)
        future_delta = action_term._cwi_bounded_intent_delta(now + 1.0)
        if float(reset_delta.abs().max()) != 0.0:
            raise RuntimeError("reset upper delta is not exact zero")
        if float(future_delta[zero_mask].abs().max()) != 0.0:
            raise RuntimeError("fixed half has nonzero future upper delta")
        if float(future_delta[~zero_mask].abs().max()) > 0.12 + 1e-7:
            raise RuntimeError("active upper delta exceeds Phase50 bound")

        model = ActorCritic(
            obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=15, actor_hidden_dims=[256, 128, 128],
            critic_hidden_dims=[256, 128, 128], activation="elu",
            init_noise_std=0.4, noise_std_type="scalar",
            actor_obs_normalization=False, critic_obs_normalization=False,
        ).to(env.device)
        payload = torch.load(CHECKPOINT, map_location=env.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.std.requires_grad_(False)
        source, candidate = copy.deepcopy(model).eval(), copy.deepcopy(model).eval()
        trainable = sorted(name for name, p in candidate.named_parameters() if p.requires_grad)
        frozen = sorted(name for name, p in candidate.named_parameters() if not p.requires_grad)
        expected_trainable = sorted(
            f"{prefix}.{suffix}"
            for prefix in ("actor.0", "actor.2", "actor.4", "actor.6", "critic.0", "critic.2", "critic.4", "critic.6")
            for suffix in ("weight", "bias")
        )
        if trainable != expected_trainable or frozen != ["std"]:
            raise RuntimeError("trainable/frozen parameter contract differs")
        before = model_hash(candidate)
        batches = []
        paired_batches = (obs, {"policy": obs["policy"].flip(0), "critic": obs["critic"].flip(0)})
        with torch.no_grad():
            for index, batch in enumerate(paired_batches):
                sa, ca = source.act_inference(batch), candidate.act_inference(batch)
                sv, cv = source.evaluate(batch), candidate.evaluate(batch)
                batches.append({"index": index, "action_shape": list(sa.shape),
                                "value_shape": list(sv.shape),
                                "action_max_abs": float((sa - ca).abs().max()),
                                "value_max_abs": float((sv - cv).abs().max()),
                                "finite": bool(torch.isfinite(sa).all() and torch.isfinite(sv).all())})
        after = model_hash(candidate)
        passed = before == after and all(
            row["finite"] and row["action_max_abs"] == 0.0 and row["value_max_abs"] == 0.0
            for row in batches
        )
        report = {
            "phase": 55, "decision": "PASS_LIVE_ZERO_UPDATE_ONLY" if passed else "FAIL_LIVE_ZERO_UPDATE",
            "source": {"checkpoint": str(CHECKPOINT), "sha256": sha256(CHECKPOINT), "iteration": int(payload["iter"])},
            "runtime": {"policy_obs": list(obs["policy"].shape), "critic_obs": list(obs["critic"].shape),
                        "policy_terms": list(policy_terms), "critic_terms": list(critic_terms),
                        "policy_term_dims": policy_term_dims, "critic_term_dims": critic_term_dims,
                        "policy_action_dim": wrapped.num_actions, "sim_joint_count": len(robot.joint_names),
                        "action_joint_names": list(action_names), "head2": list(HEAD2),
                        "head_default": head_default.tolist(), "upper_artifact": str(UPPER),
                        "upper_artifact_sha256": sha256(UPPER), "fixed_upper_envs": int(zero_mask.sum()),
                        "active_upper_envs": int((~zero_mask).sum()),
                        "upper_sampler": "deterministic even/odd exact 32/32 paired split",
                        "domain_upper_counts": domain_upper_counts,
                        "reset_upper_delta_max": float(reset_delta.abs().max()),
                        "future_fixed_delta_max": float(future_delta[zero_mask].abs().max()),
                        "future_active_delta_max": float(future_delta[~zero_mask].abs().max()),
                        "upper_target_in_action_residual": False,
                        "realized_upper_q_dq_in_93d_proprioception": True},
            "parameters": {"trainable_names": trainable, "frozen_names": frozen,
                           "hash_before": before, "hash_after": after},
            "fixed_forward_batches": batches, "optimizer_constructed": False,
            "optimizer_steps": 0, "environment_resets": 1,
            "environment_control_steps": 0, "checkpoint_created": False,
            "boundary": "live initialization/reset and fixed forward only; no physics step, optimizer or robustness claim",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError("Phase55 live zero equivalence failed")
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
