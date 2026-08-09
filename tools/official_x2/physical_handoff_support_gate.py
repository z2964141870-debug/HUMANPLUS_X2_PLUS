#!/usr/bin/env python3
"""Hash-bound Phase24 support gate for physical actor-observation groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "x2_phase24_physical_handoff_support_gate_v1"
GROUPS = ("base_ang_vel", "projected_gravity")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _group_contract(values: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(values, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(reference).all():
        raise ValueError("Phase24 physical gate requires finite Nx3 group rows")
    center = np.median(reference, axis=0)
    q25, q75 = np.percentile(reference, [25.0, 75.0], axis=0)
    scale = (q75 - q25) / 1.349
    std = np.std(reference, axis=0)
    scale = np.where(scale > 1.0e-8, scale, std)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    pair = np.sqrt(np.mean(((reference[:, None, :] - reference[None, :, :]) / scale) ** 2, axis=2))
    np.fill_diagonal(pair, np.inf)
    loo = pair.min(axis=1)
    return {
        "feature_count": 3,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "reference": reference.tolist(),
        "threshold_method": "reference leave-one-out nearest-neighbor RMS p95",
        "threshold_value": float(np.percentile(loo, 95.0)),
        "loo_distance_quantiles": {
            "p50": float(np.percentile(loo, 50.0)),
            "p95": float(np.percentile(loo, 95.0)),
            "max": float(np.max(loo)),
        },
    }


def build_contract(
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    angular = np.asarray(base_ang_vel, dtype=np.float64)
    gravity = np.asarray(projected_gravity, dtype=np.float64)
    if angular.shape != gravity.shape or angular.ndim != 2 or angular.shape[1] != 3:
        raise ValueError("Phase24 groups must have matching Nx3 shapes")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "reference_semantics": "Phase19 recorder-v2 eligible rows, exact actor pre-inference observation_93d slices",
        "reference_count": int(angular.shape[0]),
        "groups": {
            "base_ang_vel": _group_contract(angular),
            "projected_gravity": _group_contract(gravity),
        },
        "support_condition": "both per-group robust nearest-neighbor distances <= their independently frozen LOO-p95 thresholds",
        "combination": "logical AND with Phase23 previous-action support and generator double-support",
        "timeout": "end of existing stop horizon; never force handoff",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def load_contract(path: str | Path, expected_file_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path)
    if expected_file_sha256 is not None and file_sha256(resolved) != expected_file_sha256:
        raise ValueError("Phase24 physical gate file hash mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or canonical_sha256(payload) != payload.get("content_sha256"):
        raise ValueError("Phase24 physical gate schema/content hash mismatch")
    for name in GROUPS:
        reference = np.asarray(payload["groups"][name]["reference"], dtype=np.float64)
        if reference.shape != (int(payload["reference_count"]), 3):
            raise ValueError(f"Phase24 {name} reference shape mismatch")
    return payload


def _distance(value: Sequence[float], group: Mapping[str, Any]) -> float:
    query = np.asarray(value, dtype=np.float64)
    reference = np.asarray(group["reference"], dtype=np.float64)
    scale = np.asarray(group["scale"], dtype=np.float64)
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("Phase24 physical query must be finite 3D")
    return float(np.sqrt(np.mean(((reference - query[None, :]) / scale[None, :]) ** 2, axis=1)).min())


def physical_gate_decision(
    base_ang_vel: Sequence[float],
    projected_gravity: Sequence[float],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for name, value in (("base_ang_vel", base_ang_vel), ("projected_gravity", projected_gravity)):
        group = contract["groups"][name]
        distance = _distance(value, group)
        threshold = float(group["threshold_value"])
        decisions[name] = {
            "distance": distance,
            "threshold": threshold,
            "in_support": bool(distance <= threshold),
        }
    decisions["allow_handoff"] = bool(all(decisions[name]["in_support"] for name in GROUPS))
    return decisions
