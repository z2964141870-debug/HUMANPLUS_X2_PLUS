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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    expected = {
        "actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias",
        "actor.4.weight", "actor.4.bias", "actor.6.weight", "actor.6.bias",
    }
    actor_keys = {key for key in state if key.startswith("actor.")}
    if actor_keys != expected:
        raise RuntimeError(f"unexpected actor keys: {sorted(actor_keys ^ expected)}")
    dims = [state["actor.0.weight"].shape[1], 256, 128, 128, state["actor.6.weight"].shape[0]]
    if dims != [93, 256, 128, 128, 15]:
        raise RuntimeError(f"unexpected actor architecture: {dims}")

    actor = torch.nn.Sequential(
        torch.nn.Linear(93, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 15),
    )
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")})
    actor.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.linspace(-1.0, 1.0, 93, dtype=torch.float32).reshape(1, 93)
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
        "pytorch_onnx_max_abs_error": max_error,
    }
    manifest_path = output.with_suffix(".export.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
