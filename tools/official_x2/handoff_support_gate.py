#!/usr/bin/env python3
"""Pure, hash-bound actor-history gate for BASE Phase23 handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "x2_phase23_previous_action_support_gate_v1"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_contract(previous_actions: np.ndarray, *, source_manifest_sha256: str) -> dict:
    reference = np.asarray(previous_actions, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 15 or not np.isfinite(reference).all():
        raise ValueError("Phase23 gate requires finite Nx15 previous-action rows")
    center = np.median(reference, axis=0)
    q25, q75 = np.percentile(reference, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    std = np.std(reference, axis=0)
    scale = np.where(scale > 1.0e-8, scale, std)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    delta = (reference[:, None, :] - reference[None, :, :]) / scale
    pair = np.sqrt(np.mean(delta * delta, axis=2))
    np.fill_diagonal(pair, np.inf)
    threshold = float(np.percentile(pair.min(axis=1), 95.0))
    payload = {
        "schema": SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "reference_semantics": "Phase19 recorder-v2 exact actor pre-inference previous_action, all Phase20 eligible rows",
        "distance": "nearest-neighbor RMS after per-feature median/IQR scale; fallback std then 1",
        "threshold": "reference leave-one-out nearest-neighbor p95",
        "threshold_value": threshold,
        "reference_count": int(reference.shape[0]),
        "feature_count": 15,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "reference_previous_action": reference.tolist(),
        "support_condition": "nearest_distance <= threshold_value",
        "contact_condition": "generator left_contact>0 and right_contact>0; not physical contact",
        "timeout": "end of existing stop horizon; never force handoff",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def load_contract(path: str | Path, expected_file_sha256: str | None = None) -> dict:
    resolved = Path(path)
    if expected_file_sha256 is not None and file_sha256(resolved) != expected_file_sha256:
        raise ValueError("Phase23 gate file hash mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or canonical_sha256(payload) != payload.get("content_sha256"):
        raise ValueError("Phase23 gate schema/content hash mismatch")
    reference = np.asarray(payload["reference_previous_action"], dtype=np.float64)
    if reference.shape != (int(payload["reference_count"]), 15):
        raise ValueError("Phase23 gate reference shape mismatch")
    return payload


def previous_action_distance(action: Sequence[float], contract: Mapping[str, Any]) -> float:
    value = np.asarray(action, dtype=np.float64)
    reference = np.asarray(contract["reference_previous_action"], dtype=np.float64)
    scale = np.asarray(contract["scale"], dtype=np.float64)
    if value.shape != (15,) or not np.isfinite(value).all():
        raise ValueError("Phase23 query action must be finite 15D")
    return float(np.sqrt(np.mean(((reference - value[None, :]) / scale[None, :]) ** 2, axis=1)).min())


def handoff_gate_decision(
    previous_action: Sequence[float], gait_phase: Sequence[float], contract: Mapping[str, Any]
) -> dict[str, Any]:
    phase = np.asarray(gait_phase, dtype=np.float64)
    if phase.shape != (4,):
        raise ValueError("Phase23 gait phase must be 4D")
    distance = previous_action_distance(previous_action, contract)
    in_support = distance <= float(contract["threshold_value"])
    generator_double_support = bool(phase[2] > 0.0 and phase[3] > 0.0)
    return {
        "distance": distance,
        "threshold": float(contract["threshold_value"]),
        "previous_action_in_support": in_support,
        "generator_double_support": generator_double_support,
        "allow_handoff": bool(in_support and generator_double_support),
    }
