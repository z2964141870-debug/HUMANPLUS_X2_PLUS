#!/usr/bin/env python3
"""Export the deterministic actor MLP from an RSL-RL checkpoint to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F


ACTOR_INDICES = (0, 2, 4, 6)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_actor_state(
    state: dict[str, torch.Tensor], *, lora_alpha: float | None
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Return an exactly merged dense actor state from dense or LoRA weights."""

    actor_keys = {key for key in state if key.startswith("actor.")}
    dense_expected = {
        f"actor.{index}.{suffix}"
        for index in ACTOR_INDICES
        for suffix in ("weight", "bias")
    }
    if actor_keys == dense_expected:
        return (
            {key.removeprefix("actor."): state[key] for key in dense_expected},
            {"source_format": "dense", "lora_merged": False},
        )

    lora_expected = {
        f"actor.{index}.{suffix}"
        for index in ACTOR_INDICES
        for suffix in ("base.weight", "base.bias", "lora_A", "lora_B")
    }
    if actor_keys != lora_expected:
        raise RuntimeError(f"unexpected actor keys: {sorted(actor_keys ^ (dense_expected | lora_expected))}")
    if lora_alpha is None or not np.isfinite(lora_alpha) or lora_alpha <= 0.0:
        raise ValueError("a positive --lora-alpha is required for a LoRA checkpoint")

    merged: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    for index in ACTOR_INDICES:
        prefix = f"actor.{index}"
        base_weight = state[f"{prefix}.base.weight"]
        base_bias = state[f"{prefix}.base.bias"]
        lora_a = state[f"{prefix}.lora_A"]
        lora_b = state[f"{prefix}.lora_B"]
        rank = int(lora_a.shape[0])
        if rank <= 0 or tuple(lora_b.shape) != (base_weight.shape[0], rank):
            raise RuntimeError(f"invalid LoRA shape at {prefix}")
        if lora_a.shape[1] != base_weight.shape[1]:
            raise RuntimeError(f"LoRA input width mismatch at {prefix}")
        merged[f"{index}.weight"] = base_weight + (float(lora_alpha) / rank) * (lora_b @ lora_a)
        merged[f"{index}.bias"] = base_bias
        ranks[prefix] = rank
    return merged, {
        "source_format": "zero-output-lora",
        "lora_merged": True,
        "lora_alpha": float(lora_alpha),
        "lora_rank_by_layer": ranks,
    }


def checkpoint_actor_forward(
    state: dict[str, torch.Tensor], inputs: torch.Tensor, *, lora_alpha: float | None
) -> torch.Tensor:
    """Evaluate checkpoint actor tensors without first merging LoRA."""

    has_lora = any(key.startswith("actor.") and ".lora_" in key for key in state)
    value = inputs
    for layer_number, index in enumerate(ACTOR_INDICES):
        prefix = f"actor.{index}"
        if has_lora:
            if lora_alpha is None:
                raise ValueError("LoRA forward requires lora_alpha")
            base_weight = state[f"{prefix}.base.weight"]
            base_bias = state[f"{prefix}.base.bias"]
            lora_a = state[f"{prefix}.lora_A"]
            lora_b = state[f"{prefix}.lora_B"]
            value = F.linear(value, base_weight, base_bias) + (
                float(lora_alpha) / int(lora_a.shape[0])
            ) * F.linear(F.linear(value, lora_a), lora_b)
        else:
            value = F.linear(value, state[f"{prefix}.weight"], state[f"{prefix}.bias"])
        if layer_number < len(ACTOR_INDICES) - 1:
            value = F.elu(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="Required merge alpha for a LoRA checkpoint; dense checkpoints ignore it.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    actor_state, merge_manifest = canonical_actor_state(
        state, lora_alpha=args.lora_alpha
    )
    dims = [actor_state["0.weight"].shape[1], 256, 128, 128, actor_state["6.weight"].shape[0]]
    if dims != [93, 256, 128, 128, 15]:
        raise RuntimeError(f"unexpected actor architecture: {dims}")

    actor = torch.nn.Sequential(
        torch.nn.Linear(93, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 15),
    )
    actor.load_state_dict(actor_state)
    actor.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20260812)
    sample = torch.randn(257, 93, generator=generator, dtype=torch.float32)
    with torch.no_grad():
        checkpoint_output = checkpoint_actor_forward(
            state, sample, lora_alpha=args.lora_alpha
        )
        merged_output = actor(sample)
    merge_error = float(torch.max(torch.abs(checkpoint_output - merged_output)))
    if merge_error > 2.0e-6:
        raise RuntimeError(f"LoRA/dense merge mismatch: {merge_error}")
    torch.onnx.export(
        actor,
        sample,
        output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    with torch.no_grad():
        torch_output = actor(sample).numpy()
    ort_output = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
        ["actions"], {"obs": sample.numpy()}
    )[0]
    max_error = float(np.max(np.abs(torch_output - ort_output)))
    if max_error > 2.0e-6:
        raise RuntimeError(f"PyTorch/ONNX mismatch: {max_error}")
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_iteration": int(payload.get("iter", -1)),
        "onnx": str(output),
        "onnx_sha256": sha256(output),
        "input": {"name": "obs", "shape": ["batch", 93], "dtype": "float32"},
        "output": {"name": "actions", "shape": ["batch", 15], "dtype": "float32"},
        "opset": 17,
        "merge": {**merge_manifest, "checkpoint_vs_merged_max_abs_error": merge_error},
        "pytorch_onnx_max_abs_error": max_error,
    }
    manifest_path = output.with_suffix(".export.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
