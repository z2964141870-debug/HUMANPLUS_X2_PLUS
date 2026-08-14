#!/usr/bin/env python3
"""Run a bounded online BFM-Zero segment from random initialization on X2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--seed", type=int, default=770007)
parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=Path("artifacts/x2_scratch_short_segment_seed770007"),
)
parser.add_argument("--resume-checkpoint", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
EXPECTED_RUNS = {
    (4, 64, 770007): Path("artifacts/x2_scratch_short_segment_seed770007"),
    (512, 128, 770011): Path("artifacts/x2_scratch_512_pilot_seed770011"),
    (512, 1024, 770023): Path(
        "artifacts/x2_scratch_512_half_million_seed770023"
    ),
    (512, 1024, 770031): Path(
        "artifacts/x2_scratch_512_half_million_state71_seed770031"
    ),
    (512, 1024, 770041): Path(
        "artifacts/x2_scratch_512_half_million_expert_rollout_seed770041"
    ),
    (512, 2048, 770051): Path(
        "artifacts/x2_scratch_state71_cont1_seed770051"
    ),
    (512, 3072, 770061): Path(
        "artifacts/x2_scratch_terminal_aware_seed770061"
    ),
    (512, 512, 770071): Path(
        "artifacts/x2_scratch_state71_safety_ft_seed770071"
    ),
    (512, 512, 770081): Path(
        "artifacts/x2_scratch_state71_taskreward_ft_seed770081"
    ),
    (512, 3072, 770091): Path(
        "artifacts/x2_scratch_closed_loop_expert_seed770091"
    ),
}
REPORT_BY_RUN = {
    (512, 1024, 770023): Path("reports/x2_scratch_512_half_million.json"),
    (512, 1024, 770031): Path(
        "reports/x2_scratch_512_half_million_state71.json"
    ),
    (512, 1024, 770041): Path(
        "reports/x2_scratch_512_half_million_expert_rollout.json"
    ),
    (512, 2048, 770051): Path(
        "reports/x2_scratch_state71_cont1.json"
    ),
    (512, 3072, 770061): Path(
        "reports/x2_scratch_terminal_aware.json"
    ),
    (512, 512, 770071): Path(
        "reports/x2_scratch_state71_safety_ft.json"
    ),
    (512, 512, 770081): Path(
        "reports/x2_scratch_state71_taskreward_ft.json"
    ),
    (512, 3072, 770091): Path(
        "reports/x2_scratch_closed_loop_expert.json"
    ),
}
RESUME_BY_RUN = {
    (512, 2048, 770051): {
        "path": Path("artifacts/x2_scratch_512_half_million_state71_seed770031"),
        "tree_sha256": "130c4da4f7483ee1a0b514377a603b3108709f66e7a8ddf514aabddb2b6a9d2f",
    },
    (512, 512, 770071): {
        "path": Path("artifacts/x2_scratch_state71_cont1_seed770051"),
        "tree_sha256": "81e85ee3f2a52a2ea572874effbb64e5d67ef709aee448d019e3b392b868f6da",
    },
    (512, 512, 770081): {
        "path": Path("artifacts/x2_scratch_state71_cont1_seed770051"),
        "tree_sha256": "81e85ee3f2a52a2ea572874effbb64e5d67ef709aee448d019e3b392b868f6da",
    },
}
run_key = (args.num_envs, args.steps, args.seed)
if run_key not in EXPECTED_RUNS:
    raise ValueError("the bounded scratch runs are fixed to the registered run tuples")
if args.checkpoint_dir != EXPECTED_RUNS[run_key]:
    raise ValueError("the bounded scratch checkpoint path is immutable for this run")
resume_spec = RESUME_BY_RUN.get(run_key)
expected_resume = None if resume_spec is None else resume_spec["path"]
if args.resume_checkpoint != expected_resume:
    raise ValueError("the bounded scratch resume checkpoint is immutable for this run")
report_path = REPORT_BY_RUN.get(run_key)
if report_path is not None:
    report_temporary = report_path.with_name(f".{report_path.name}.tmp")
    report_sidecar = report_path.with_name(f"{report_path.name}.sha256")
    report_sidecar_temporary = report_sidecar.with_name(f".{report_sidecar.name}.tmp")
    if any(
        path.exists()
        for path in (
            report_path,
            report_temporary,
            report_sidecar,
            report_sidecar_temporary,
        )
    ):
        raise FileExistsError(f"refusing to overwrite scratch report: {report_path}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402

from humanoidverse.agents.buffers.transition import DictBuffer  # noqa: E402
from humanoidverse.agents.load_utils import load_agent_from_checkpoint_dir  # noqa: E402
from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    X2_ALL_AUX_REWARD_NAMES,
    X2_AUX_REWARD_SCALES,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
    X2_TASK_AUX_REWARD_SCALES,
    X2ScratchDataConfig,
    build_x2_scratch_replay_buffers,
    convert_stage219_critic_rollout,
    load_x2_motion_files,
)
from humanoidverse.x2_scratch_model import (  # noqa: E402
    x2_scratch_agent_config,
    x2_scratch_observation_space,
)


MOTION_ROOT = Path(
    "/home/yu/x2_teleop_final/x2_sonic/motion_lib_x2/"
    "stage72_official_true_forward4_v1"
)
CLOSED_LOOP_EXPERT_BUNDLE = Path(
    "/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim/artifacts/retarget/"
    "x2_phase70_long_lookahead/rollout_evidence_rerun1.pt"
)
CLOSED_LOOP_EXPERT_BUNDLE_SHA256 = (
    "0b7f5f4026eb62ea474a2204b11b67e7c500024e1f3b814add0dffc482a07d81"
)
UPDATE_STEPS = tuple(range(16, args.steps + 1, 16))
IS_512_PILOT = run_key == (512, 128, 770011)
IS_HALF_MILLION = run_key == (512, 1024, 770023)
IS_STATE71_HALF_MILLION = run_key == (512, 1024, 770031)
IS_EXPERT_ROLLOUT_HALF_MILLION = run_key == (512, 1024, 770041)
IS_STATE71_CONTINUATION = run_key == (512, 2048, 770051)
IS_TERMINAL_AWARE = run_key == (512, 3072, 770061)
IS_SAFETY_FINETUNE = run_key == (512, 512, 770071)
IS_TASK_REWARD_FINETUNE = run_key == (512, 512, 770081)
IS_CLOSED_LOOP_EXPERT_SCRATCH = run_key == (512, 3072, 770091)


def milestone(stage: str, **details) -> None:
    print(
        json.dumps(
            {
                "schema": "x2_bfm_zero_scratch_short_segment_milestone_v1",
                "stage": stage,
                **details,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def tensor_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(f"{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_hash(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode())
        digest.update(hashlib.sha256(payload).digest())
        total_bytes += len(payload)
    return digest.hexdigest(), total_bytes, len(files)


def component_modules(model) -> dict[str, torch.nn.Module]:
    return {
        "actor": model._actor,
        "forward": model._forward_map,
        "backward": model._backward_map,
        "critic": model._critic,
        "aux_critic": model._aux_critic,
        "discriminator": model._discriminator,
    }


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
    checkpoint = args.checkpoint_dir.resolve()
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    if checkpoint.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite scratch checkpoint: {checkpoint}")
    if resume_spec is not None:
        resume_checkpoint = resume_spec["path"].resolve()
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint is absent: {resume_checkpoint}")
        actual_resume_hash, _, _ = tree_hash(resume_checkpoint)
        if actual_resume_hash != resume_spec["tree_sha256"]:
            raise RuntimeError("resume checkpoint tree hash has drifted")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    milestone("main_entered")

    motions = sorted(
        path for path in MOTION_ROOT.glob("*.pkl") if path.name != "metadata.pkl"
    )
    episodes, dataset_audit = load_x2_motion_files(
        motions,
        config=X2ScratchDataConfig(history_length=4),
    )
    closed_loop_expert_audit = None
    if IS_CLOSED_LOOP_EXPERT_SCRATCH:
        if (
            not CLOSED_LOOP_EXPERT_BUNDLE.is_file()
            or hashlib.sha256(CLOSED_LOOP_EXPERT_BUNDLE.read_bytes()).hexdigest()
            != CLOSED_LOOP_EXPERT_BUNDLE_SHA256
        ):
            raise RuntimeError("closed-loop expert bundle is absent or has drifted")
        closed_loop_payload = torch.load(
            CLOSED_LOOP_EXPERT_BUNDLE,
            map_location="cpu",
            weights_only=False,
        )
        closed_loop_episodes, closed_loop_expert_audit = (
            convert_stage219_critic_rollout(
                closed_loop_payload["critic_observation"].numpy(),
                motion_id_offset=len(episodes),
                config=X2ScratchDataConfig(history_length=4),
            )
        )
        episodes.extend(closed_loop_episodes)
    expected_expert_frames = 962 + (
        25_600 if IS_CLOSED_LOOP_EXPERT_SCRATCH else 0
    )
    expert_replay = build_x2_scratch_replay_buffers(
        episodes,
        seq_length=8,
        z_dim=64,
        seed=args.seed,
        device="cuda" if IS_EXPERT_ROLLOUT_HALF_MILLION else "cpu",
    )

    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(env, wrapped, history_length=4, to_numpy=False)
    observation, _ = adapter.reset(seed=args.seed)
    milestone("environment_ready")

    if resume_spec is None:
        config = x2_scratch_agent_config(
            device="cuda",
            hidden_dim=256,
            hidden_layers=3,
            z_dim=64,
            history_length=4,
            batch_size=32,
            use_x2_aux_rewards=True,
            rollout_expert_trajectories=IS_EXPERT_ROLLOUT_HALF_MILLION,
            rollout_expert_trajectories_length=128,
            rollout_expert_trajectories_percentage=0.5,
        )
        agent = config.build(x2_scratch_observation_space(), ACTION_DIM)
    else:
        agent = load_agent_from_checkpoint_dir(resume_checkpoint, device="cuda")
        if agent.obs_space != x2_scratch_observation_space():
            raise RuntimeError("resume checkpoint observation contract has drifted")
        if IS_SAFETY_FINETUNE or IS_TASK_REWARD_FINETUNE:
            selected_aux_rewards = (
                X2_TASK_AUX_REWARD_SCALES
                if IS_TASK_REWARD_FINETUNE
                else X2_AUX_REWARD_SCALES
            )
            agent.cfg = agent.cfg.model_copy(
                update={
                    "aux_rewards": list(selected_aux_rewards),
                    "aux_rewards_scaling": dict(selected_aux_rewards),
                }
            )
            for group in agent.actor_optimizer.param_groups:
                group["lr"] = 1.0e-4
    model = agent._model
    components = component_modules(model)
    initial_hashes = {name: tensor_hash(module) for name, module in components.items()}
    z = model.sample_z(args.num_envs, device="cuda")
    probe_count = min(32, args.num_envs)
    probe_observation = tree_map(
        lambda value: value[:probe_count].detach().clone(), without_time(observation)
    )
    probe_z = z[:probe_count].detach().clone()
    with torch.inference_mode():
        probe_action_before = agent.act(
            probe_observation, probe_z, mean=True
        ).detach().clone()
    replay = DictBuffer(capacity=args.num_envs * args.steps, device="cpu")

    optimizers = {
        "actor": agent.actor_optimizer,
        "forward": agent.forward_optimizer,
        "backward": agent.backward_optimizer,
        "critic": agent.critic_optimizer,
        "aux_critic": agent.aux_critic_optimizer,
        "discriminator": agent.discriminator_optimizer,
    }
    optimizer_steps = {name: 0 for name in optimizers}
    for name, optimizer in optimizers.items():
        original_step = optimizer.step

        def counted_step(*step_args, _name=name, _step=original_step, **step_kwargs):
            optimizer_steps[_name] += 1
            return _step(*step_args, **step_kwargs)

        optimizer.step = counted_step

    terminated_count = 0
    truncated_count = 0
    metric_trace: list[dict[str, float]] = []
    for step_index in range(1, args.steps + 1):
        if IS_EXPERT_ROLLOUT_HALF_MILLION:
            z = agent.maybe_update_rollout_context(
                z,
                torch.full(
                    (args.num_envs, 1),
                    step_index - 1,
                    dtype=torch.long,
                    device="cuda",
                ),
                replay_buffer=expert_replay,
            )
        current = without_time(observation)
        with torch.inference_mode():
            action = agent.act(current, z, mean=False).clamp(-1.0, 1.0)
        next_observation, reward, terminated, truncated, info = adapter.step(action)
        done = terminated | truncated
        if not torch.isfinite(torch.as_tensor(reward)).all():
            raise RuntimeError("short segment received a non-finite environment reward")
        if set(info["aux_rewards"]) != set(X2_ALL_AUX_REWARD_NAMES):
            raise RuntimeError("short segment auxiliary reward keys differ from config")
        keep = (
            torch.ones_like(done, dtype=torch.bool)
            if (
                IS_TERMINAL_AWARE
                or IS_SAFETY_FINETUNE
                or IS_TASK_REWARD_FINETUNE
                or IS_CLOSED_LOOP_EXPERT_SCRATCH
            )
            else ~done
        )
        if keep.any():
            replay.extend(
                {
                    "observation": tree_map(lambda value: value[keep], current),
                    "action": action[keep],
                    "z": z[keep],
                    "aux_rewards": {
                        name: value[keep].reshape(-1, 1)
                        for name, value in info["aux_rewards"].items()
                    },
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
            z[done] = model.sample_z(int(done.sum()), device="cuda")
        observation = next_observation

        if step_index in UPDATE_STEPS:
            if len(replay) < 32:
                raise RuntimeError("short segment replay is too small for update")
            metrics = agent.update(
                {"expert_slicer": expert_replay["expert_slicer"], "train": replay},
                step=step_index,
            )
            selected = {
                name: float(torch.as_tensor(metrics[name]).detach().float().mean().cpu())
                for name in (
                    "actor_loss",
                    "fb_loss",
                    "critic_loss",
                    "aux_critic_loss",
                    "disc_loss",
                    "mean_aux_reward",
                )
            }
            if not all(np.isfinite(value) for value in selected.values()):
                raise RuntimeError("short segment update produced a non-finite metric")
            metric_trace.append({"control_step": step_index, **selected})
            milestone(
                "agent_update_complete",
                control_step=step_index,
                replay_size=len(replay),
                terminated=terminated_count,
            )

    final_hashes = {name: tensor_hash(module) for name, module in components.items()}
    changed = {name: initial_hashes[name] != final_hashes[name] for name in components}
    if not all(changed.values()) or any(count != len(UPDATE_STEPS) for count in optimizer_steps.values()):
        raise RuntimeError("short segment optimizer/component gate failed")
    with torch.inference_mode():
        probe_action_after = agent.act(probe_observation, probe_z, mean=True)
    probe_action_delta = (probe_action_after - probe_action_before).abs()
    if not torch.isfinite(probe_action_delta).all() or not bool(
        probe_action_delta.max() > 0
    ):
        raise RuntimeError("scratch segment fixed-probe action did not change finitely")

    agent.save(temporary)
    loaded = agent.__class__.load(temporary, device="cuda")
    reload_hash = tensor_hash(loaded._model)
    model_hash = tensor_hash(model)
    checkpoint_roundtrip = reload_hash == model_hash
    if not checkpoint_roundtrip:
        raise RuntimeError("scratch checkpoint strict round-trip failed")
    os.replace(temporary, checkpoint)
    checkpoint_hash, checkpoint_bytes, checkpoint_files = tree_hash(checkpoint)

    peak_bytes = int(torch.cuda.max_memory_allocated())
    valid = (
        dataset_audit["all_finite"]
        and expert_replay["audit"]["expert_frames"] == expected_expert_frames
        and len(metric_trace) == len(UPDATE_STEPS)
        and truncated_count == 0
        and checkpoint_roundtrip
        and checkpoint_bytes < 512 * 1024**2
        and peak_bytes < 8 * 1024**3
    )
    pass_decision = (
        "PASS_512_ENV_SCRATCH_PILOT"
        if IS_512_PILOT
        else (
            "PASS_512_ENV_CLOSED_LOOP_EXPERT_SCRATCH_SEGMENT"
            if IS_CLOSED_LOOP_EXPERT_SCRATCH
            else
            "PASS_512_ENV_STATE71_TASK_REWARD_FINETUNE_SEGMENT"
            if IS_TASK_REWARD_FINETUNE
            else
            "PASS_512_ENV_STATE71_SAFETY_FINETUNE_SEGMENT"
            if IS_SAFETY_FINETUNE
            else "PASS_512_ENV_TERMINAL_AWARE_SCRATCH_SEGMENT"
            if IS_TERMINAL_AWARE
            else "PASS_512_ENV_STATE71_CONTINUATION_SEGMENT"
            if IS_STATE71_CONTINUATION
            else "PASS_512_ENV_HALF_MILLION_EXPERT_ROLLOUT_SCRATCH_SEGMENT"
            if IS_EXPERT_ROLLOUT_HALF_MILLION
            else "PASS_512_ENV_HALF_MILLION_STATE71_SCRATCH_SEGMENT"
            if IS_STATE71_HALF_MILLION
            else "PASS_512_ENV_HALF_MILLION_SCRATCH_SEGMENT"
            if IS_HALF_MILLION
            else "PASS_SHORT_TRAINING_SEGMENT"
        )
    )
    fail_decision = (
        "FAIL_512_ENV_SCRATCH_PILOT"
        if IS_512_PILOT
        else (
            "FAIL_512_ENV_CLOSED_LOOP_EXPERT_SCRATCH_SEGMENT"
            if IS_CLOSED_LOOP_EXPERT_SCRATCH
            else
            "FAIL_512_ENV_STATE71_TASK_REWARD_FINETUNE_SEGMENT"
            if IS_TASK_REWARD_FINETUNE
            else
            "FAIL_512_ENV_STATE71_SAFETY_FINETUNE_SEGMENT"
            if IS_SAFETY_FINETUNE
            else "FAIL_512_ENV_TERMINAL_AWARE_SCRATCH_SEGMENT"
            if IS_TERMINAL_AWARE
            else "FAIL_512_ENV_STATE71_CONTINUATION_SEGMENT"
            if IS_STATE71_CONTINUATION
            else "FAIL_512_ENV_HALF_MILLION_EXPERT_ROLLOUT_SCRATCH_SEGMENT"
            if IS_EXPERT_ROLLOUT_HALF_MILLION
            else "FAIL_512_ENV_HALF_MILLION_STATE71_SCRATCH_SEGMENT"
            if IS_STATE71_HALF_MILLION
            else "FAIL_512_ENV_HALF_MILLION_SCRATCH_SEGMENT"
            if IS_HALF_MILLION
            else "FAIL_SHORT_TRAINING_SEGMENT"
        )
    )
    return {
        "schema": "x2_bfm_zero_scratch_bounded_segment_v2",
        "decision": pass_decision if valid else fail_decision,
        "scratch_from_zero": True,
        "stage219_weights_loaded": False,
        "stage219_rollout_data_used": IS_CLOSED_LOOP_EXPERT_SCRATCH,
        "closed_loop_expert_bundle": (
            {
                "path": str(CLOSED_LOOP_EXPERT_BUNDLE),
                "sha256": CLOSED_LOOP_EXPERT_BUNDLE_SHA256,
                "audit": closed_loop_expert_audit,
            }
            if IS_CLOSED_LOOP_EXPERT_SCRATCH
            else None
        ),
        "expert_episode_count": len(episodes),
        "expert_frames": expert_replay["audit"]["expert_frames"],
        "resumed_from_scratch_checkpoint": resume_spec is not None,
        "resume_checkpoint": (
            None
            if resume_spec is None
            else {
                "path": str(resume_checkpoint),
                "tree_sha256": resume_spec["tree_sha256"],
            }
        ),
        "gait_template_used": False,
        "state_dim": STATE_DIM,
        "base_linear_velocity_observed": STATE_DIM == 71,
        "expert_rollout_enabled": IS_EXPERT_ROLLOUT_HALF_MILLION,
        "expert_rollout_length": 128 if IS_EXPERT_ROLLOUT_HALF_MILLION else 0,
        "expert_rollout_fraction": 0.5 if IS_EXPERT_ROLLOUT_HALF_MILLION else 0.0,
        "terminal_transitions_retained": (
            IS_TERMINAL_AWARE
            or IS_SAFETY_FINETUNE
            or IS_TASK_REWARD_FINETUNE
            or IS_CLOSED_LOOP_EXPERT_SCRATCH
        ),
        "task_reward_auxiliary_scale": (
            X2_TASK_AUX_REWARD_SCALES["locomotion_total_reward"]
            if IS_TASK_REWARD_FINETUNE
            else None
        ),
        "termination_auxiliary_scale": (
            X2_AUX_REWARD_SCALES["termination"]
            if IS_TERMINAL_AWARE or IS_SAFETY_FINETUNE or IS_CLOSED_LOOP_EXPERT_SCRATCH
            else None
        ),
        "action_magnitude_auxiliary_scale": (
            (
                X2_TASK_AUX_REWARD_SCALES["action_magnitude_l2"]
                if IS_TASK_REWARD_FINETUNE
                else X2_AUX_REWARD_SCALES["action_magnitude_l2"]
            )
            if (
                IS_TERMINAL_AWARE
                or IS_SAFETY_FINETUNE
                or IS_TASK_REWARD_FINETUNE
                or IS_CLOSED_LOOP_EXPERT_SCRATCH
            )
            else None
        ),
        "actor_learning_rate": (
            1.0e-4 if IS_SAFETY_FINETUNE or IS_TASK_REWARD_FINETUNE else 3.0e-4
        ),
        "num_envs": args.num_envs,
        "control_steps_per_env": args.steps,
        "attempted_transitions": args.num_envs * args.steps,
        "stored_nonterminal_transitions": len(replay),
        "terminated": terminated_count,
        "truncated": truncated_count,
        "agent_update_calls": len(UPDATE_STEPS),
        "optimizer_steps": optimizer_steps,
        "total_optimizer_steps": sum(optimizer_steps.values()),
        "changed_components": changed,
        "fixed_probe": {
            "num_samples": probe_count,
            "action_delta_mean_abs": float(probe_action_delta.mean().cpu()),
            "action_delta_max_abs": float(probe_action_delta.max().cpu()),
            "all_finite": bool(torch.isfinite(probe_action_delta).all()),
        },
        "metric_trace": metric_trace,
        "checkpoint": {
            "path": str(checkpoint),
            "files": checkpoint_files,
            "bytes": checkpoint_bytes,
            "tree_sha256": checkpoint_hash,
            "strict_roundtrip": checkpoint_roundtrip,
        },
        "cuda_peak_allocated_bytes": peak_bytes,
        "performance_claim": False,
        "long_training_unlocked": False,
    }


try:
    report = main()
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    expected_pass = (
        "PASS_512_ENV_SCRATCH_PILOT"
        if IS_512_PILOT
        else (
            "PASS_512_ENV_CLOSED_LOOP_EXPERT_SCRATCH_SEGMENT"
            if IS_CLOSED_LOOP_EXPERT_SCRATCH
            else
            "PASS_512_ENV_STATE71_TASK_REWARD_FINETUNE_SEGMENT"
            if IS_TASK_REWARD_FINETUNE
            else
            "PASS_512_ENV_STATE71_SAFETY_FINETUNE_SEGMENT"
            if IS_SAFETY_FINETUNE
            else "PASS_512_ENV_TERMINAL_AWARE_SCRATCH_SEGMENT"
            if IS_TERMINAL_AWARE
            else "PASS_512_ENV_STATE71_CONTINUATION_SEGMENT"
            if IS_STATE71_CONTINUATION
            else "PASS_512_ENV_HALF_MILLION_EXPERT_ROLLOUT_SCRATCH_SEGMENT"
            if IS_EXPERT_ROLLOUT_HALF_MILLION
            else "PASS_512_ENV_HALF_MILLION_STATE71_SCRATCH_SEGMENT"
            if IS_STATE71_HALF_MILLION
            else "PASS_512_ENV_HALF_MILLION_SCRATCH_SEGMENT"
            if IS_HALF_MILLION
            else "PASS_SHORT_TRAINING_SEGMENT"
        )
    )
    if report["decision"] != expected_pass:
        raise RuntimeError(serialized)
except BaseException as error:
    print(
        json.dumps(
            {
                "schema": "x2_bfm_zero_scratch_short_segment_failure_v1",
                "error_type": type(error).__name__,
                "error": str(error),
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
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        report_temporary.write_bytes(payload)
        os.replace(report_temporary, report_path)
        sidecar_payload = f"{hashlib.sha256(payload).hexdigest()}  {report_path.name}\n"
        report_sidecar_temporary.write_text(sidecar_payload)
        os.replace(report_sidecar_temporary, report_sidecar)
    print(serialized, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
