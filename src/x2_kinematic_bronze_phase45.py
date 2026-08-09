"""Immutable kinematic-Bronze sampler contract for faithful Any2Any smoke tests.

Bronze means a kinematically admissible X2 reference.  It does not imply
contact consistency, dynamic feasibility, hardware force truth, or Silver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BronzeSourceSelection:
    source_pkl: Path
    source_sha256: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class KinematicBronzeTrainSpec:
    sources: tuple[BronzeSourceSelection, ...]
    tier_report_paths: tuple[Path, ...]
    output_pkl: Path
    expected_keys: tuple[str, ...]


def _entry_frames(entry: Mapping[str, Any]) -> int:
    return int(len(entry["dof"]))


def build_bronze_train_artifact(spec: KinematicBronzeTrainSpec) -> dict[str, Any]:
    """Build a derived split without modifying any source cache.

    The caller remains responsible for proving each selected key is Bronze in
    the frozen tier reports.  This function only enforces source and sampler
    identity and marks the derived entries truthfully.
    """

    merged: dict[str, Any] = {}
    source_rows = []
    for source in spec.sources:
        actual_hash = sha256(source.source_pkl)
        if actual_hash != source.source_sha256:
            raise ValueError(f"Bronze source hash drift: {source.source_pkl}")
        payload = joblib.load(source.source_pkl)
        for key in source.keys:
            if key not in payload:
                raise KeyError(f"Bronze sampler key missing: {key}")
            if key in merged:
                raise ValueError(f"duplicate Bronze sampler key: {key}")
            entry = dict(payload[key])
            entry["split"] = "train"
            entry["phase45_tier"] = "kinematic_bronze"
            entry["phase45_truth_boundary"] = (
                "GMR kinematic reference only; not Silver, dynamic truth, "
                "hardware GRF/COP/wrench, or proof of free-root stability"
            )
            merged[key] = entry
        source_rows.append(
            {
                "path": str(source.source_pkl),
                "sha256": actual_hash,
                "keys": list(source.keys),
            }
        )
    if tuple(merged) != spec.expected_keys:
        raise ValueError(
            f"Bronze sampler order mismatch: {tuple(merged)} != {spec.expected_keys}"
        )
    spec.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, spec.output_pkl)
    return {
        "path": str(spec.output_pkl),
        "sha256": sha256(spec.output_pkl),
        "sampler_keys": list(merged),
        "frames": {key: _entry_frames(value) for key, value in merged.items()},
        "total_frames": sum(_entry_frames(value) for value in merged.values()),
        "fps": {key: float(value["fps"]) for key, value in merged.items()},
        "optimizer_eligible": True,
        "tier": "kinematic_bronze",
        "source_rows": source_rows,
        "tier_reports": [str(path) for path in spec.tier_report_paths],
    }


def validate_bronze_train_artifact(
    path: Path, expected_sha256: str, expected_keys: Sequence[str]
) -> dict[str, Any]:
    if sha256(path) != expected_sha256:
        raise ValueError("derived Bronze artifact hash drift")
    payload = joblib.load(path)
    if list(payload) != list(expected_keys):
        raise ValueError("derived Bronze sampler keys/order changed")
    for key, entry in payload.items():
        if entry.get("split") != "train":
            raise ValueError(f"{key}: held/non-train label in Bronze optimizer artifact")
        if entry.get("phase45_tier") != "kinematic_bronze":
            raise ValueError(f"{key}: missing explicit Bronze tier label")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "sampler_keys": list(payload),
        "frames": int(sum(_entry_frames(value) for value in payload.values())),
        "optimizer_eligible": True,
        "tier": "kinematic_bronze",
    }
