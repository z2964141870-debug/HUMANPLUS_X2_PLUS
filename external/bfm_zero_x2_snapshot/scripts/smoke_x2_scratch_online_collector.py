#!/usr/bin/env python3
"""Collect a tiny online X2 replay from a randomly initialized BFM policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--seed", type=int, default=770004)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.num_envs, args.steps) != (4, 16):
    raise ValueError("the online collector smoke is fixed to 4 envs x 16 steps")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402

from humanoidverse.agents.buffers.transition import DictBuffer  # noqa: E402
from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
)
from humanoidverse.x2_scratch_model import (  # noqa: E402
    x2_scratch_agent_config,
    x2_scratch_observation_space,
)


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def milestone(stage: str, **details) -> None:
    """Emit progress that survives a native Isaac/Kit shutdown stall."""
    print(
        json.dumps(
            {
                "schema": "x2_bfm_zero_scratch_online_collector_milestone_v1",
                "stage": stage,
                **details,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def build_cfg():
    cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.actions.joint_pos.scale = {
        name: float(scale)
        for name, scale in zip(X2_LOWER_JOINTS_15, X2_SCRATCH_ACTION_SCALE_15)
    }
    return cfg


def without_time(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in observation.items() if key != "time"}


def main() -> dict:
    milestone("main_entered", num_envs=args.num_envs, steps=args.steps)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    milestone("environment_construct_begin")
    env = ManagerBasedRLEnv(cfg=build_cfg())
    milestone("environment_constructed")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(env, wrapped, history_length=4, to_numpy=False)
    observation, _ = adapter.reset(seed=args.seed)
    milestone("adapter_reset", state_shape=list(observation["state"].shape))

    config = x2_scratch_agent_config(
        device="cuda",
        hidden_dim=256,
        hidden_layers=3,
        z_dim=64,
        history_length=4,
        batch_size=32,
    )
    agent = config.build(x2_scratch_observation_space(), ACTION_DIM)
    milestone("scratch_model_built")
    model_hash_before = state_hash(agent._model)
    z = agent._model.sample_z(args.num_envs, device="cuda")
    replay = DictBuffer(capacity=args.num_envs * args.steps, device="cpu")

    terminated_count = 0
    truncated_count = 0
    action_abs_max = 0.0
    for step_index in range(args.steps):
        current = without_time(observation)
        with torch.inference_mode():
            action = agent.act(current, z, mean=False).clamp(-1.0, 1.0)
        action_abs_max = max(action_abs_max, float(action.abs().max()))
        next_observation, reward, terminated, truncated, _ = adapter.step(action)
        done = terminated | truncated
        if not torch.isfinite(torch.as_tensor(reward)).all():
            raise RuntimeError("online collector received a non-finite reward")
        keep = ~done
        if keep.any():
            replay.extend(
                {
                    "observation": tree_map(lambda value: value[keep], current),
                    "action": action[keep],
                    "z": z[keep],
                    "next": {
                        "observation": tree_map(
                            lambda value: value[keep], without_time(next_observation)
                        ),
                        "terminated": terminated[keep].reshape(-1, 1),
                    },
                }
            )
        terminated_count += int(terminated.sum())
        truncated_count += int(truncated.sum())
        if done.any():
            z[done] = agent._model.sample_z(int(done.sum()), device="cuda")
        observation = next_observation
        if step_index in {0, args.steps - 1}:
            milestone(
                "collector_step",
                step=step_index + 1,
                replay_size=len(replay),
                done=int(done.sum()),
            )

    sample = replay.sample(32)
    term = env.action_manager._terms["joint_pos"]
    optimizer_states_empty = all(
        len(optimizer.state) == 0
        for optimizer in (
            agent.actor_optimizer,
            agent.forward_optimizer,
            agent.backward_optimizer,
            agent.critic_optimizer,
            agent.aux_critic_optimizer,
            agent.discriminator_optimizer,
        )
    )
    model_unchanged = state_hash(agent._model) == model_hash_before
    peak_bytes = int(torch.cuda.max_memory_allocated())
    valid = (
        len(replay) >= 32
        and action_abs_max <= 1.0
        and tuple(sample["observation"]["state"].shape) == (32, STATE_DIM)
        and tuple(sample["observation"]["history_actor"].shape)
        == (32, 4 * (STATE_DIM + ACTION_DIM))
        and tuple(sample["action"].shape) == (32, 15)
        and tuple(sample["z"].shape) == (32, 64)
        and tuple(term._joint_names) == X2_LOWER_JOINTS_15
        and not hasattr(term, "_template_q")
        and optimizer_states_empty
        and model_unchanged
        and peak_bytes < 8 * 1024**3
    )
    return {
        "schema": "x2_bfm_zero_scratch_online_collector_smoke_v2",
        "decision": "PASS_ONLINE_COLLECTOR_SMOKE" if valid else "FAIL_ONLINE_COLLECTOR_SMOKE",
        "scratch_from_zero": True,
        "stage219_weights_loaded": False,
        "gait_template_used": False,
        "num_envs": args.num_envs,
        "control_steps_per_env": args.steps,
        "attempted_transitions": args.num_envs * args.steps,
        "stored_nonterminal_transitions": len(replay),
        "terminated": terminated_count,
        "truncated": truncated_count,
        "action_abs_max": action_abs_max,
        "sample_shapes": {
            "state": list(sample["observation"]["state"].shape),
            "history_actor": list(sample["observation"]["history_actor"].shape),
            "action": list(sample["action"].shape),
            "z": list(sample["z"].shape),
        },
        "optimizer_steps": 0,
        "optimizer_states_empty": optimizer_states_empty,
        "model_unchanged": model_unchanged,
        "checkpoint_writes": 0,
        "cuda_peak_allocated_bytes": peak_bytes,
        "long_training_unlocked": False,
    }


try:
    report = main()
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if report["decision"] != "PASS_ONLINE_COLLECTOR_SMOKE":
        raise RuntimeError(serialized)
except BaseException as error:
    print(
        json.dumps(
            {
                "schema": "x2_bfm_zero_scratch_online_collector_failure_v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    print(serialized, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
