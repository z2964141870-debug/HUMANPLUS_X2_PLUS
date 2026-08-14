#!/usr/bin/env python3
"""Behavior-clone executable Stage219 targets into the scratch BFM actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from humanoidverse.agents.load_utils import load_agent_from_checkpoint_dir
from humanoidverse.x2_scratch import (
    ACTION_DIM,
    X2ScratchDataConfig,
    convert_stage219_critic_rollout,
    reconstruct_stage219_direct_action_labels,
)


SOURCE = Path("artifacts/x2_scratch_closed_loop_expert_seed770091")
SOURCE_TREE_SHA256 = "9585582c2e497f9b9128f08a8c3f7bf22d54b2152da405d05443349f844bd963"
BUNDLE = Path(
    "/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim/artifacts/retarget/"
    "x2_phase70_long_lookahead/rollout_evidence_rerun1.pt"
)
BUNDLE_SHA256 = "0b7f5f4026eb62ea474a2204b11b67e7c500024e1f3b814add0dffc482a07d81"
TEMPLATE = Path(
    "/home/yu/x2_teleop_final/x2_sonic/data/processed/"
    "x2_official_forward_gait_phase_template_15dof.npz"
)
TEMPLATE_SHA256 = "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d"
OUTPUT = Path("artifacts/x2_scratch_closed_loop_bc_seed770101")
REPORT = Path("reports/x2_scratch_closed_loop_bc.json")
SEED = 770101
EPOCHS = 20
BATCH_SIZE = 512
TRAIN_ENVS = 48


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def build_dataset(
    agent,
    critic_observation: torch.Tensor,
    labels: np.ndarray,
    *,
    include_command_phase: bool = False,
):
    episodes, audit = convert_stage219_critic_rollout(
        critic_observation.numpy(),
        direct_action_labels=labels,
        include_command_phase=include_command_phase,
        config=X2ScratchDataConfig(history_length=4),
    )
    keys = tuple(episodes[0]["observation"])
    observation = {
        key: torch.as_tensor(
            np.stack([episode["observation"][key] for episode in episodes], axis=1)
        )
        for key in keys
    }
    starts = tuple(range(0, critic_observation.shape[0] - 8, 8))
    current_parts = {key: [] for key in keys}
    latent_parts = []
    label_parts = []
    env_parts = []
    for start in starts:
        next_obs = {
            key: value[start + 1 : start + 9].reshape(-1, value.shape[-1]).cuda()
            for key, value in observation.items()
        }
        backward = agent._model.backward_map(next_obs).reshape(8, 64, -1)
        latent = agent._model.project_z(backward.mean(dim=0)).cpu()
        for key, value in observation.items():
            current_parts[key].append(value[start : start + 8].reshape(-1, value.shape[-1]))
        latent_parts.append(latent.unsqueeze(0).expand(8, -1, -1).reshape(-1, 64))
        label_parts.append(
            torch.as_tensor(labels[start : start + 8]).reshape(-1, ACTION_DIM)
        )
        env_parts.append(torch.arange(64).repeat(8))
    current = {key: torch.cat(parts, dim=0) for key, parts in current_parts.items()}
    latent = torch.cat(latent_parts, dim=0)
    action = torch.cat(label_parts, dim=0)
    env_id = torch.cat(env_parts, dim=0)
    return current, latent, action, env_id, audit, len(starts)


def actor_mean(agent, observation, latent):
    normalized = agent._model._normalize(observation)
    return agent._model._actor(normalized, latent, agent._model.cfg.actor_std).mean.float()


@torch.no_grad()
def evaluate_mse(agent, observation, latent, action, indices) -> dict[str, float]:
    squared_error = []
    absolute_error = []
    for start in range(0, indices.numel(), 2048):
        selection = indices[start : start + 2048]
        batch_obs = {key: value[selection].cuda() for key, value in observation.items()}
        prediction = actor_mean(agent, batch_obs, latent[selection].cuda())
        target = action[selection].cuda()
        squared_error.append((prediction - target).square().cpu())
        absolute_error.append((prediction - target).abs().cpu())
    squared = torch.cat(squared_error)
    absolute = torch.cat(absolute_error)
    return {
        "mse": float(squared.mean()),
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
    }


def main(
    *,
    fixed_latent: bool = False,
    physical_contract_v2: bool = False,
    observation_contract_v3: bool = False,
    joint_contract_v4: bool = False,
) -> tuple[dict, Path]:
    if (physical_contract_v2 or observation_contract_v3 or joint_contract_v4) and not fixed_latent:
        raise ValueError("corrected contracts are frozen to the canonical latent")
    if sum((physical_contract_v2, observation_contract_v3, joint_contract_v4)) > 1:
        raise ValueError("select only one corrected contract version")
    if joint_contract_v4:
        output = Path(
            "artifacts/x2_scratch_closed_loop_bc_joint_v4_seed770191"
        )
        report_path = Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        )
        seed = 770191
    elif observation_contract_v3:
        output = Path(
            "artifacts/x2_scratch_closed_loop_bc_observation_v3_seed770171"
        )
        report_path = Path(
            "reports/x2_scratch_closed_loop_bc_observation_v3.json"
        )
        seed = 770171
    elif physical_contract_v2:
        output = Path(
            "artifacts/x2_scratch_closed_loop_bc_physical_v2_seed770151"
        )
        report_path = Path(
            "reports/x2_scratch_closed_loop_bc_physical_v2.json"
        )
        seed = 770151
    else:
        output = (
            Path("artifacts/x2_scratch_closed_loop_bc_fixed_z_seed770111")
            if fixed_latent
            else OUTPUT
        )
        report_path = (
            Path("reports/x2_scratch_closed_loop_bc_fixed_z.json")
            if fixed_latent
            else REPORT
        )
        seed = 770111 if fixed_latent else SEED
    if tree_hash(SOURCE)[0] != SOURCE_TREE_SHA256:
        raise RuntimeError("scratch source checkpoint has drifted")
    if file_hash(BUNDLE) != BUNDLE_SHA256 or file_hash(TEMPLATE) != TEMPLATE_SHA256:
        raise RuntimeError("closed-loop data/template hash guard failed")
    sidecar = report_path.with_name(f"{report_path.name}.sha256")
    temporary = output.with_name(f".{output.name}.tmp")
    if any(path.exists() for path in (output, temporary, report_path, sidecar)):
        raise FileExistsError("refusing to overwrite BC evidence")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    label_audit = {
        "samples": int(np.prod(labels.shape[:2])),
        "abs_max": float(np.abs(labels).max()),
        "clip_fraction": float((np.abs(labels) >= 1.0).mean()),
        "finite": bool(np.isfinite(labels).all()),
    }

    agent = load_agent_from_checkpoint_dir(SOURCE, device="cuda")
    source_model_hash = model_hash(agent._model)
    observation, latent, action, env_id, rollout_audit, chunk_count = build_dataset(
        agent, critic_observation, labels
    )
    canonical_latent = None
    if fixed_latent:
        canonical_latent = agent._model.project_z(
            latent.mean(dim=0, keepdim=True).cuda()
        ).cpu()
        latent = canonical_latent.expand_as(latent).clone()
    train_indices = torch.nonzero(env_id < TRAIN_ENVS).reshape(-1)
    holdout_indices = torch.nonzero(env_id >= TRAIN_ENVS).reshape(-1)
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
    for epoch in range(1, EPOCHS + 1):
        permutation = train_indices[torch.randperm(train_indices.numel())]
        losses = []
        for start in range(0, permutation.numel(), BATCH_SIZE):
            selection = permutation[start : start + BATCH_SIZE]
            batch_obs = {key: value[selection].cuda() for key, value in observation.items()}
            prediction = actor_mean(agent, batch_obs, latent[selection].cuda())
            target = action[selection].cuda()
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, target, beta=0.05
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent._model._actor.parameters(), 1.0)
            optimizer.step()
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
    if source_model_hash == final_model_hash:
        raise RuntimeError("BC did not change the scratch model")

    agent.save(temporary)
    loaded = agent.__class__.load(temporary, device="cuda")
    strict_roundtrip = model_hash(loaded._model) == final_model_hash
    if not strict_roundtrip:
        raise RuntimeError("BC checkpoint strict round-trip failed")
    os.replace(temporary, output)
    checkpoint_hash, checkpoint_bytes, checkpoint_files = tree_hash(output)
    valid = (
        label_audit["finite"]
        and rollout_audit["all_finite"]
        and after["holdout"]["mse"] < before["holdout"]["mse"] * 0.25
        and strict_roundtrip
    )
    return {
        "schema": (
            "x2_bfm_zero_closed_loop_behavior_cloning_joint_contract_v4"
            if joint_contract_v4
            else
            "x2_bfm_zero_closed_loop_behavior_cloning_observation_contract_v3"
            if observation_contract_v3
            else
            "x2_bfm_zero_closed_loop_behavior_cloning_physical_contract_v2"
            if physical_contract_v2
            else "x2_bfm_zero_closed_loop_behavior_cloning_v1"
        ),
        "decision": "PASS_BC_FIT_ONLY" if valid else "FAIL_BC_FIT_STOP",
        "scratch_lineage": True,
        "stage219_weights_loaded": False,
        "stage219_action_labels_used": True,
        "stage219_to_scratch_physical_action_scale_conversion": True,
        "direct_action_last_action_and_history_contract": True,
        "stage219_joint_state_reindexed_to_scratch_semantics": True,
        "latent_mode": "single_canonical" if fixed_latent else "per_expert_chunk",
        "canonical_latent": (
            canonical_latent.reshape(-1).tolist() if canonical_latent is not None else None
        ),
        "source_checkpoint": {
            "path": str(SOURCE),
            "tree_sha256": SOURCE_TREE_SHA256,
        },
        "source_model_hash": source_model_hash,
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
        "optimizer_steps": EPOCHS * int(np.ceil(train_indices.numel() / BATCH_SIZE)),
        "before": before,
        "after": after,
        "trace": trace,
        "checkpoint": {
            "path": str(output),
            "tree_sha256": checkpoint_hash,
            "bytes": checkpoint_bytes,
            "files": checkpoint_files,
            "strict_roundtrip": strict_roundtrip,
        },
        "performance_claim": False,
        "long_training_unlocked": False,
    }, report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-latent", action="store_true")
    parser.add_argument("--physical-contract-v2", action="store_true")
    parser.add_argument("--observation-contract-v3", action="store_true")
    parser.add_argument("--joint-contract-v4", action="store_true")
    args = parser.parse_args()
    report, report_path = main(
        fixed_latent=args.fixed_latent,
        physical_contract_v2=args.physical_contract_v2,
        observation_contract_v3=args.observation_contract_v3,
        joint_contract_v4=args.joint_contract_v4,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_sidecar = report_path.with_name(f".{report_path.name}.sha256.tmp")
    temporary_report.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    temporary_sidecar.write_text(f"{digest}  {report_path.name}\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    os.replace(temporary_sidecar, report_path.with_name(f"{report_path.name}.sha256"))
    print(json.dumps(report, sort_keys=True))
    if report["decision"] != "PASS_BC_FIT_ONLY":
        raise SystemExit(1)
