"""Materialize an Any2Any LoRA checkpoint into a compact plain-weight base."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import torch


StateMerger = Callable[..., tuple[dict[str, torch.Tensor], dict[str, Any]]]
STATE_KEYS = ("policy_state_dict", "value_state_dict")


def _clone_state(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().clone()) for key, value in state.items())


def build_materialized_checkpoint(
    source: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    merge_state: StateMerger,
    fallback_alpha: float = 16.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fold policy/value LoRA weights and intentionally omit optimizer state."""

    output: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for state_key in STATE_KEYS:
        state = source.get(state_key)
        if not isinstance(state, dict):
            raise KeyError(f"checkpoint must contain mapping {state_key!r}")
        merged, report = merge_state(state, fallback_alpha=fallback_alpha)
        stale = sorted(
            key
            for key in merged
            if ".lora_" in key
            or "._lora_" in key
            or "._frozen_lora_delta_weight" in key
            or ".base_layer." in key
        )
        if stale:
            raise ValueError(f"materialized {state_key} retained LoRA keys: {stale[:5]}")
        nonfinite = sorted(
            key
            for key, value in merged.items()
            if isinstance(value, torch.Tensor)
            and (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all()
        )
        if nonfinite:
            raise ValueError(f"materialized {state_key} has non-finite tensors: {nonfinite}")
        output[state_key] = _clone_state(merged)
        reports[state_key] = report

    output["dcpeft_materialization_metadata"] = {
        "schema_version": 1,
        "mode": "lora_folded_plain_frozen_base",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "fallback_alpha": float(fallback_alpha),
        "optimizer_state_intentionally_omitted": True,
        "intended_use": "fresh_zero_lora_on_stage152b_base",
    }
    report = {
        "schema_version": 1,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "fallback_alpha": float(fallback_alpha),
        "states": reports,
        "optimizer_state_intentionally_omitted": True,
    }
    return output, report
