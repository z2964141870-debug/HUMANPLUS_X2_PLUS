#!/usr/bin/env python3
"""Validate the approved fixed-StandStill parent for live-reference arming."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


SCHEMA = "x2-fixed-standstill-parent-v1"
SCOPE = "supported-stationary-live-reference"
REQUIRED_ARTIFACT_ROLES = {
    "deploy_binary",
    "model",
    "powered_launcher",
    "profile_launcher",
}
EXPECTED_CONTRACT = {
    "reference": "StandStill",
    "profile": "neutral_damped",
    "policy_hz": 50,
    "writer_hz": 250,
    "imu_mode": "reconstructed_pelvis",
    "firmware": "test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7",
    "message_abi": "aimdk_msgs 0.8.18",
}


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_under_root(root: Path, relative_path: object, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ManifestError(f"{label} path is missing")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ManifestError(f"{label} path must be relative to deployment root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"{label} path escapes deployment root: {relative_path}") from exc
    return resolved


def _validate_evidence_item(root: Path, item: object, minimum_s: float, label: str) -> None:
    if not isinstance(item, dict):
        raise ManifestError(f"{label} must be an object")
    if item.get("result") != "pass":
        raise ManifestError(f"{label} result must be 'pass'")
    duration = item.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < minimum_s:
        raise ManifestError(f"{label} duration_s must be >= {minimum_s}")
    log_path = _resolved_under_root(root, item.get("log_path"), label)
    if not log_path.exists():
        raise ManifestError(f"{label} log path does not exist: {log_path}")
    summary_path = _resolved_under_root(root, item.get("summary_path"), label)
    if not summary_path.is_file():
        raise ManifestError(f"{label} summary does not exist: {summary_path}")
    expected = item.get("summary_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ManifestError(f"{label} summary_sha256 is invalid")
    actual = sha256_file(summary_path)
    if actual != expected.lower():
        raise ManifestError(
            f"{label} summary SHA-256 mismatch: expected={expected} actual={actual}"
        )


def validate_manifest(path: Path, root: Path) -> dict:
    root = root.resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"approved parent manifest is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")
    if payload.get("schema") != SCHEMA:
        raise ManifestError(f"manifest schema must be {SCHEMA!r}")
    if payload.get("approval_status") != "approved":
        raise ManifestError("approval_status is not 'approved'")
    if payload.get("approved_scope") != SCOPE:
        raise ManifestError(f"approved_scope must be {SCOPE!r}")
    for field in ("approved_by", "approved_at", "approval_note"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"{field} must be a non-empty string")
        if "TODO" in value.upper() or "REPLACE" in value.upper():
            raise ManifestError(f"{field} still contains a placeholder")

    contract = payload.get("fixed_contract")
    if not isinstance(contract, dict):
        raise ManifestError("fixed_contract must be an object")
    for key, expected in EXPECTED_CONTRACT.items():
        if contract.get(key) != expected:
            raise ManifestError(
                f"fixed_contract.{key} must be {expected!r}, got {contract.get(key)!r}"
            )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestError("artifacts must be a list")
    roles: set[str] = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ManifestError(f"artifact[{index}] must be an object")
        role = item.get("role")
        if role in roles:
            raise ManifestError(f"duplicate artifact role {role!r}")
        roles.add(role)
        artifact_path = _resolved_under_root(root, item.get("path"), f"artifact[{index}]")
        if not artifact_path.is_file():
            raise ManifestError(f"artifact[{index}] does not exist: {artifact_path}")
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ManifestError(f"artifact[{index}] sha256 is invalid")
        actual = sha256_file(artifact_path)
        if actual != expected.lower():
            raise ManifestError(
                f"artifact[{index}] SHA-256 mismatch: expected={expected} actual={actual}"
            )
    missing_roles = REQUIRED_ARTIFACT_ROLES - roles
    if missing_roles:
        raise ManifestError(f"manifest is missing artifact roles: {sorted(missing_roles)}")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ManifestError("evidence must be an object")
    short_passes = evidence.get("supported_30s_passes")
    if not isinstance(short_passes, list) or len(short_passes) < 3:
        raise ManifestError("evidence requires at least three supported_30s_passes")
    for index, item in enumerate(short_passes):
        _validate_evidence_item(root, item, 30.0, f"supported_30s_passes[{index}]")
    _validate_evidence_item(
        root, evidence.get("supported_300s_pass"), 300.0, "supported_300s_pass"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = validate_manifest(args.manifest.expanduser(), args.root.expanduser())
    except (OSError, ManifestError) as exc:
        print(f"PARENT_MANIFEST_BLOCKED: {exc}")
        return 4
    print(
        "PARENT_MANIFEST_OK "
        f"approved_by={payload['approved_by']} approved_at={payload['approved_at']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
