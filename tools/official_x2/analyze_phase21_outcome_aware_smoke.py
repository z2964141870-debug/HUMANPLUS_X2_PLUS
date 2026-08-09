#!/usr/bin/env python3
"""Finalize BASE Phase21 diagnostics without running physics or training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor(state: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    x = obs
    for index in (0, 2, 4):
        x = torch.nn.functional.elu(
            torch.nn.functional.linear(x, state[f"actor.{index}.weight"], state[f"actor.{index}.bias"])
        )
    return torch.nn.functional.linear(x, state["actor.6.weight"], state["actor.6.bias"])


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {key: value.detach().float() for key, value in payload["model_state_dict"].items()}


def support_observations(manifest_path: Path) -> torch.Tensor:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = [json.loads(Path(source["path"]).read_text(encoding="utf-8")) for source in manifest["sources"]]
    values = []
    for ref in manifest["rows"]:
        if not ref["eligible"]:
            continue
        row = sources[int(ref["source_index"])]["rows"][int(ref["source_row_index"])]
        values.append(row["observation_93d"])
    obs = torch.as_tensor(values, dtype=torch.float32)
    if obs.ndim != 2 or obs.shape[1] != 93 or not torch.isfinite(obs).all():
        raise ValueError("Phase21 support observations are not finite 93D rows")
    return obs


def policy_drift(source: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], obs: torch.Tensor) -> dict:
    with torch.inference_mode():
        old_mean = actor(source, obs)
        new_mean = actor(candidate, obs)
    old_std = source["std"].reshape(1, -1).clamp_min(1.0e-8)
    new_std = candidate["std"].reshape(1, -1).clamp_min(1.0e-8)
    kl = (
        torch.log(new_std / old_std)
        + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
        - 0.5
    ).sum(dim=-1)
    delta = new_mean - old_mean
    return {
        "definition": "post-hoc diagonal-Gaussian KL(source||final) on all eligible Phase20 93D rows; not PPO minibatch KL",
        "support_rows": int(obs.shape[0]),
        "kl_mean": float(kl.mean()),
        "kl_p95": float(torch.quantile(kl, 0.95)),
        "kl_max": float(kl.max()),
        "actor_mean_delta_rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "actor_mean_delta_abs_max": float(delta.abs().max()),
        "std_abs_max_delta": float((new_std - old_std).abs().max()),
    }


def scalar_series(run_dir: Path, tag: str) -> list[dict]:
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [{"iteration": int(item.step), "value": float(item.value)} for item in acc.Scalars(tag)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--f000-checkpoint", type=Path, required=True)
    parser.add_argument("--f005-checkpoint", type=Path, required=True)
    parser.add_argument("--f000-run", type=Path, required=True)
    parser.add_argument("--f005-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    source_state = load_state(args.source_checkpoint)
    obs = support_observations(args.manifest)
    training = {}
    for label, checkpoint, run_dir in (
        ("f000", args.f000_checkpoint, args.f000_run),
        ("f005", args.f005_checkpoint, args.f005_run),
    ):
        state = load_state(checkpoint)
        training[label] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "transitions": 7680,
            "reward": scalar_series(run_dir, "Train/mean_reward"),
            "episode_length": scalar_series(run_dir, "Train/mean_episode_length"),
            "surrogate_loss": scalar_series(run_dir, "Loss/surrogate"),
            "ppo_minibatch_kl_logged": False,
            "policy_drift_diagnostic": policy_drift(source_state, state, obs),
        }

    metrics = gate["groups"]
    lower_is_better = ("heading_max_rad", "lateral_displacement_m", "stop_drift_m", "stop_settle_time_s")
    continuous = {}
    for key in lower_is_better:
        means = {label: float(metrics[label]["aggregate"]["metrics"][key]["mean"]) for label in ("source", "f000", "f005")}
        continuous[key] = {
            "lower_is_better": True,
            "means": means,
            "candidate_no_greater_than_both_controls": means["f005"] <= min(means["source"], means["f000"]),
            "descriptive_only_no_preregistered_tolerance": True,
        }
    subgates = {
        label: metrics[label]["aggregate"]["subgates"] for label in ("source", "f000", "f005")
    }
    discrete_not_worse = all(
        subgates["f005"][key] >= max(subgates["source"][key], subgates["f000"][key])
        for key in ("stand", "startup", "move", "stop", "full")
    )
    result = {
        "stage": "BASE Phase21 outcome-aware state-role paired 5-update smoke",
        "status": "NOT_PROMOTED_25_UPDATES_LOCKED",
        "hypothesis": "A 5% balanced success-safe/critical-from-failure reset curriculum improves recovery without degrading the source or same-seed fraction-0 control.",
        "intervention": "same source, seed47, 64 env, 5 updates and optimizer/reward/obs/action/PD contract; only reset fraction 0.00 vs 0.05",
        "frozen_inputs": {
            "source_checkpoint": {"path": str(args.source_checkpoint.resolve()), "sha256": sha256(args.source_checkpoint)},
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
            "official_gate": {"path": str(args.gate.resolve()), "sha256": sha256(args.gate)},
        },
        "training_diagnostics_not_promotion_evidence": training,
        "official_gate": {
            "all_15_valid": bool(gate["decision"]["all_15_valid"]),
            "all_policy_slots_exact": bool(gate["decision"]["all_policy_slots_exact"]),
            "subgates": subgates,
            "continuous_metrics": continuous,
            "signed_pitch_reporting_only": True,
        },
        "decision": {
            "candidate_discrete_subgates_not_worse": discrete_not_worse,
            "candidate_improves_stop_or_full_count": subgates["f005"]["stop"] > max(subgates["source"]["stop"], subgates["f000"]["stop"]) or subgates["f005"]["full"] > max(subgates["source"]["full"], subgates["f000"]["full"]),
            "candidate_continuous_means_no_worse_than_both_controls": all(item["candidate_no_greater_than_both_controls"] for item in continuous.values()),
            "phase21_promoted": False,
            "updates25_unlocked": False,
            "reason": "candidate remained move/stop/full 0/5 and its heading/lateral means were worse than source; mixed stop-drift/settle improvements do not establish a recovery success",
        },
        "boundaries": [
            "PPO minibatch KL was not emitted to TensorBoard; support-set policy KL is explicitly post-hoc diagnostic only.",
            "Official MuJoCo/AimDK simulation is not a real-robot or hardware foot-force result.",
            "No fraction, seed, update, reward, PD, or handoff sweep was run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
