#!/usr/bin/env python3
"""Train a fresh phase-conditioned X2 scratch actor on closed-loop labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from humanoidverse.x2_scratch import (
    ACTION_DIM,
    X2ScratchDataConfig,
    convert_stage219_critic_rollout,
    reconstruct_stage219_direct_action_labels,
)
from humanoidverse.x2_scratch_model import (
    x2_scratch_agent_config,
    x2_scratch_observation_space,
)
from scripts.train_x2_scratch_closed_loop_bc import (
    BUNDLE,
    BUNDLE_SHA256,
    TEMPLATE,
    TEMPLATE_SHA256,
    actor_mean,
    evaluate_mse,
    file_hash,
    model_hash,
    tree_hash,
)


OUTPUT = Path("artifacts/x2_scratch_phase_bc_v6_seed770301")
REPORT = Path("reports/x2_scratch_phase_bc_v6.json")
SEED = 770301
EPOCHS = 20
BATCH_SIZE = 512
TRAIN_ENVS = 48
HISTORY_LENGTH = 4
Z_DIM = 64


@torch.no_grad()
def build_dataset(critic_observation: torch.Tensor, labels: np.ndarray):
    episodes, audit = convert_stage219_critic_rollout(
        critic_observation.numpy(),
        direct_action_labels=labels,
        include_command_phase=True,
        config=X2ScratchDataConfig(history_length=HISTORY_LENGTH),
    )
    keys = tuple(episodes[0]["observation"])
    observation = {
        key: torch.as_tensor(
            np.stack(
                [episode["observation"][key] for episode in episodes], axis=1
            )
        )
        for key in keys
    }
    # Use complete eight-frame blocks so every label is aligned with the
    # action-time observation and no next-episode padding enters training.
    starts = tuple(range(0, critic_observation.shape[0] - 8, 8))
    current_parts = {key: [] for key in keys}
    label_parts = []
    env_parts = []
    for start in starts:
        for key, value in observation.items():
            current_parts[key].append(
                value[start : start + 8].reshape(-1, value.shape[-1])
            )
        label_parts.append(
            torch.as_tensor(labels[start : start + 8]).reshape(-1, ACTION_DIM)
        )
        env_parts.append(torch.arange(critic_observation.shape[1]).repeat(8))
    return (
        {key: torch.cat(parts) for key, parts in current_parts.items()},
        torch.cat(label_parts),
        torch.cat(env_parts),
        audit,
        len(starts),
    )


def main() -> dict:
    if file_hash(BUNDLE) != BUNDLE_SHA256 or file_hash(TEMPLATE) != TEMPLATE_SHA256:
        raise RuntimeError("phase-conditioned BC immutable input guard failed")
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    if any(path.exists() for path in (OUTPUT, temporary, REPORT, sidecar)):
        raise FileExistsError("refusing to overwrite phase-conditioned BC evidence")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    payload = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    critic_observation = payload["critic_observation"].contiguous()
    template = np.load(TEMPLATE, allow_pickle=False)
    labels = reconstruct_stage219_direct_action_labels(
        critic_observation.numpy(),
        template["q_cycle_zero_mean_rad"],
        template["action_scale_rad"],
        template_scale=0.15,
        period_s=float(template["period_s"]),
    )
    observation, action, env_id, rollout_audit, chunk_count = build_dataset(
        critic_observation, labels
    )

    config = x2_scratch_agent_config(
        device="cuda",
        hidden_dim=256,
        hidden_layers=3,
        z_dim=Z_DIM,
        history_length=HISTORY_LENGTH,
        include_command_phase=True,
        batch_size=32,
    )
    observation_space = x2_scratch_observation_space(
        X2ScratchDataConfig(history_length=HISTORY_LENGTH),
        include_command_phase=True,
    )
    agent = config.build(observation_space, ACTION_DIM)
    initial_model_hash = model_hash(agent._model)
    canonical = agent._model.sample_z(1, device="cuda").detach()
    if canonical.shape != (1, Z_DIM) or not math.isclose(
        float(canonical.norm()), math.sqrt(Z_DIM), abs_tol=1.0e-5
    ):
        raise RuntimeError("fresh phase-conditioned latent is invalid")

    train_indices = torch.nonzero(env_id < TRAIN_ENVS).reshape(-1)
    holdout_indices = torch.nonzero(env_id >= TRAIN_ENVS).reshape(-1)
    latent = canonical.cpu().expand(action.shape[0], -1).clone()
    before = {
        "train": evaluate_mse(agent, observation, latent, action, train_indices),
        "holdout": evaluate_mse(agent, observation, latent, action, holdout_indices),
    }

    agent._model.requires_grad_(False)
    agent._model._actor.requires_grad_(True)
    agent._model._actor.train(True)
    optimizer = torch.optim.Adam(agent._model._actor.parameters(), lr=1.0e-4)
    agent.actor_optimizer = optimizer
    trace = []
    optimizer_steps = 0
    for epoch in range(1, EPOCHS + 1):
        permutation = train_indices[torch.randperm(train_indices.numel())]
        losses = []
        for start in range(0, permutation.numel(), BATCH_SIZE):
            selection = permutation[start : start + BATCH_SIZE]
            batch_observation = {
                key: value[selection].cuda() for key, value in observation.items()
            }
            prediction = actor_mean(
                agent, batch_observation, canonical.expand(selection.numel(), -1)
            )
            target = action[selection].cuda()
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, target, beta=0.05
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                agent._model._actor.parameters(), 1.0
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("phase-conditioned BC gradient is non-finite")
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
        holdout = evaluate_mse(
            agent, observation, latent, action, holdout_indices
        )
        trace.append(
            {
                "epoch": epoch,
                "training_smooth_l1_mean": float(np.mean(losses)),
                "holdout_mse": holdout["mse"],
                "holdout_mae": holdout["mae"],
            }
        )

    after = {
        "train": evaluate_mse(agent, observation, latent, action, train_indices),
        "holdout": evaluate_mse(agent, observation, latent, action, holdout_indices),
    }
    final_model_hash = model_hash(agent._model)
    if initial_model_hash == final_model_hash:
        raise RuntimeError("phase-conditioned BC did not change the random model")

    agent.save(temporary)
    loaded = agent.__class__.load(temporary, device="cuda")
    strict_roundtrip = model_hash(loaded._model) == final_model_hash
    if not strict_roundtrip:
        raise RuntimeError("phase-conditioned checkpoint round-trip failed")
    os.replace(temporary, OUTPUT)
    checkpoint_hash, checkpoint_bytes, checkpoint_files = tree_hash(OUTPUT)
    label_audit = {
        "samples": int(np.prod(labels.shape[:2])),
        "abs_max": float(np.abs(labels).max()),
        "clip_fraction": float((np.abs(labels) >= 1.0).mean()),
        "finite": bool(np.isfinite(labels).all()),
    }
    valid = (
        rollout_audit["all_finite"]
        and rollout_audit["command_phase_observation_contract"]
        and label_audit["finite"]
        and after["holdout"]["mse"] < before["holdout"]["mse"] * 0.25
        and strict_roundtrip
    )
    return {
        "schema": "x2_bfm_zero_phase_conditioned_behavior_cloning_v6",
        "decision": "PASS_BC_FIT_ONLY" if valid else "FAIL_BC_FIT_STOP",
        "scratch_lineage": True,
        "random_student_initialization": True,
        "stage219_weights_loaded_into_student": False,
        "stage219_labels_used": True,
        "command_phase_observation_contract": True,
        "observation_keys": list(observation),
        "canonical_latent": canonical.cpu().reshape(-1).tolist(),
        "initial_model_hash": initial_model_hash,
        "final_model_hash": final_model_hash,
        "bundle_sha256": BUNDLE_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "rollout_audit": rollout_audit,
        "label_audit": label_audit,
        "chunk_count_per_env": chunk_count,
        "train_envs": TRAIN_ENVS,
        "holdout_envs": 64 - TRAIN_ENVS,
        "train_samples": int(train_indices.numel()),
        "holdout_samples": int(holdout_indices.numel()),
        "epochs": EPOCHS,
        "optimizer_steps": optimizer_steps,
        "before": before,
        "after": after,
        "trace": trace,
        "checkpoint": {
            "path": str(OUTPUT),
            "tree_sha256": checkpoint_hash,
            "bytes": checkpoint_bytes,
            "files": checkpoint_files,
            "strict_roundtrip": strict_roundtrip,
        },
        "performance_claim": False,
        "long_training_unlocked": False,
    }


if __name__ == "__main__":
    report = main()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_report = REPORT.with_name(f".{REPORT.name}.tmp")
    temporary_sidecar = REPORT.with_name(f".{REPORT.name}.sha256.tmp")
    temporary_report.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    temporary_sidecar.write_text(f"{digest}  {REPORT.name}\n", encoding="utf-8")
    os.replace(temporary_report, REPORT)
    os.replace(temporary_sidecar, REPORT.with_name(f"{REPORT.name}.sha256"))
    print(json.dumps(report, sort_keys=True))
    if report["decision"] != "PASS_BC_FIT_ONLY":
        raise SystemExit(1)
