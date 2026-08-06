#!/usr/bin/env python3
"""Create a compact, aggregate-preserving multi-critic warm start.

The source checkpoint is never modified. Policy tensors are copied bit-for-bit.
Only the final scalar value head is expanded:

    V_j(x) = V_scalar(x) / num_critics

Therefore ``sum_j V_j(x) == V_scalar(x)`` for every input before training.
Optimizer/scheduler state is intentionally omitted because its tensor shapes no
longer match and DC-PEFT uses this file as a pre-optimizer warm start.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


FINAL_PREFIX = "critic_module.module.12."
SCALED_OUTPUT_KEYS = (
    FINAL_PREFIX + "base_layer.weight",
    FINAL_PREFIX + "base_layer.bias",
    FINAL_PREFIX + "lora_B",
)
MASK_KEY = FINAL_PREFIX + "_lora_output_mask"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_value_state(
    state: OrderedDict[str, torch.Tensor], num_critics: int
) -> OrderedDict[str, torch.Tensor]:
    if num_critics < 2:
        raise ValueError("num_critics must be at least 2")
    missing = [key for key in (*SCALED_OUTPUT_KEYS, MASK_KEY) if key not in state]
    if missing:
        raise KeyError(f"missing final value-head tensors: {missing}")

    expanded: OrderedDict[str, torch.Tensor] = OrderedDict(
        (key, value.detach().clone()) for key, value in state.items()
    )
    for key in SCALED_OUTPUT_KEYS:
        value = state[key]
        if value.shape[0] != 1:
            raise ValueError(f"{key} is not a scalar-head tensor: {tuple(value.shape)}")
        repeat_shape = (num_critics,) + (1,) * (value.ndim - 1)
        expanded[key] = value.repeat(repeat_shape) / float(num_critics)

    mask = state[MASK_KEY]
    if mask.ndim != 1 or mask.shape[0] != 1:
        raise ValueError(f"{MASK_KEY} is not a scalar output mask: {tuple(mask.shape)}")
    expanded[MASK_KEY] = mask.repeat(num_critics)
    return expanded


def build_compact_checkpoint(
    source: dict[str, Any], source_path: Path, num_critics: int
) -> dict[str, Any]:
    policy_state = source.get("policy_state_dict")
    value_state = source.get("value_state_dict")
    if policy_state is None or value_state is None:
        raise KeyError("checkpoint must contain policy_state_dict and value_state_dict")

    return {
        "policy_state_dict": OrderedDict(
            (key, value.detach().clone()) for key, value in policy_state.items()
        ),
        "value_state_dict": expand_value_state(value_state, num_critics),
        "dcpeft_init_metadata": {
            "schema_version": 1,
            "mode": "aggregate_preserving_equal_split",
            "num_critics": num_critics,
            "source_path": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "optimizer_state_intentionally_omitted": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-critics", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must differ from input")
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    compact = build_compact_checkpoint(source, args.input, args.num_critics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    print(f"wrote={args.output}")
    print(f"sha256={sha256_file(args.output)}")
    print(f"num_critics={args.num_critics}")
    print("aggregate_value_contract=sum(heads)==source_scalar")


if __name__ == "__main__":
    main()
