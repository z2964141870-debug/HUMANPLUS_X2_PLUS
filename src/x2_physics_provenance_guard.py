"""Fail-closed hash/runtime guard for the declared Phase25 X2 training domain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> tuple[str, int]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest(), len(files)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("immutable") is not True:
        raise ValueError("physics manifest is not immutable")
    return manifest


def check_file_contract(manifest: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, row in manifest["files"].items():
        path = Path(row["path"])
        if "sha256" in row:
            checks[name] = path.is_file() and sha256(path) == row["sha256"]
        elif "tree_sha256" in row:
            actual_hash, actual_count = tree_sha256(path)
            checks[name] = (
                path.is_dir()
                and actual_hash == row["tree_sha256"]
                and actual_count == int(row["file_count"])
            )
        else:
            checks[name] = False
    return checks


def check_runtime_snapshot(
    manifest: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    expected = manifest["declared_isaac_training_runtime"]
    missing = sorted(set(expected) - set(runtime))
    extra = sorted(set(runtime) - set(expected))
    different = {
        key: {"expected": expected[key], "actual": runtime[key]}
        for key in sorted(set(expected) & set(runtime))
        if runtime[key] != expected[key]
    }
    return {
        "exact": not missing and not extra and not different,
        "missing": missing,
        "extra": extra,
        "different": different,
    }


def assert_physics_contract(
    manifest_path: Path, runtime_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    files = check_file_contract(manifest)
    runtime = check_runtime_snapshot(manifest, runtime_snapshot)
    if not all(files.values()):
        raise RuntimeError(
            "B5 file hash guard failed: "
            + ", ".join(name for name, passed in files.items() if not passed)
        )
    if not runtime["exact"]:
        raise RuntimeError(f"B5 runtime contract differs: {runtime}")
    return {"file_checks": files, "runtime_check": runtime}


def load_runtime_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
