#!/usr/bin/env python3
"""Build a deterministic SHA256 inventory for BFM-Zero X2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SOURCE_COMMIT = "e355752c46c32271eaccad084876c61da068e8e5"
DEFAULT_SOURCE = Path("/home/yu/projects/BFM-Zero/artifacts")
DEFAULT_OUTPUT = Path(
    "/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim/"
    "external/bfm_zero_x2_snapshot/ARTIFACT_MANIFEST.json"
)
REMOTE_ROOT = (
    "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-14/"
    "x2_bfm_zero_scratch_artifacts_v7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--remote-checked-at")
    args = parser.parse_args()

    files = []
    for path in sorted(item for item in args.source.rglob("*") if item.is_file()):
        relative = path.relative_to(args.source).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "remote": f"{REMOTE_ROOT}/{relative}",
            }
        )

    document = {
        "schema": "bfm_zero_x2_artifact_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "source_root": str(args.source),
        "snapshot_path": "external/bfm_zero_x2_snapshot",
        "remote_root": REMOTE_ROOT,
        "remote_verified": args.remote_verified,
        "remote_checked_at": args.remote_checked_at,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(args.output, payload)
    sidecar = args.output.with_name(f"{args.output.name}.sha256")
    atomic_write(sidecar, f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode())


if __name__ == "__main__":
    main()
