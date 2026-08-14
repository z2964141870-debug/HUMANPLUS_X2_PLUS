#!/usr/bin/env python3
"""Audit and repair the Baidu backup for BFM-Zero X2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def list_remote(bdpan: Path, parent: str, retries: int = 4) -> dict[str, int]:
    for attempt in range(1, retries + 1):
        process = subprocess.run(
            [str(bdpan), "--no-check-update", "--json", "ls", parent, "--limit", "1000"],
            text=True,
            capture_output=True,
        )
        if process.returncode == 0:
            try:
                entries = json.loads(process.stdout)
                return {
                    item["server_filename"]: int(item["size"])
                    for item in entries
                    if not item["isdir"]
                }
            except (KeyError, TypeError, ValueError):
                pass
        if "目录不存在" in (process.stdout + process.stderr):
            return {}
        if attempt < retries:
            time.sleep(attempt * 2)
    raise RuntimeError(f"could not list remote directory after {retries} attempts: {parent}")


def audit_files(bdpan: Path, remote_root: str, files: list[dict]) -> list[dict]:
    by_parent: dict[str, list[dict]] = {}
    for item in files:
        relative = Path(item["path"])
        by_parent.setdefault(relative.parent.as_posix(), []).append(item)

    failures = []
    for relative_parent, expected in sorted(by_parent.items()):
        parent = remote_root if relative_parent == "." else f"{remote_root}/{relative_parent}"
        actual = list_remote(bdpan, parent)
        for item in expected:
            name = Path(item["path"]).name
            actual_bytes = actual.get(name)
            if actual_bytes != item["bytes"]:
                failures.append(
                    {
                        "path": item["path"],
                        "expected_bytes": item["bytes"],
                        "actual_bytes": actual_bytes,
                    }
                )
    return failures


def top_directory(path: str) -> str:
    return Path(path).parts[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bdpan", type=Path, default=Path("/home/yu/.local/bin/bdpan"))
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()

    document = json.loads(args.manifest.read_text())
    remote_root = document["remote_root"]
    files = document["files"]
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    failures = audit_files(args.bdpan, remote_root, files)
    print(f"AUDIT missing_or_mismatched={len(failures)}/{len(files)}", flush=True)

    if failures and args.repair:
        pending = sorted({top_directory(item["path"]) for item in failures})
        print(f"REPAIR directories={len(pending)}", flush=True)
        for index, name in enumerate(pending, 1):
            expected = [item for item in files if top_directory(item["path"]) == name]
            source = args.source / name
            remote = f"{remote_root}/{name}"
            repaired = False
            for attempt in range(1, args.attempts + 1):
                print(
                    f"UPLOAD {index}/{len(pending)} {name} attempt={attempt}/{args.attempts}",
                    flush=True,
                )
                process = subprocess.run(
                    [str(args.bdpan), "--no-check-update", "upload", str(source), remote]
                )
                remaining = audit_files(args.bdpan, remote_root, expected)
                if not remaining:
                    print(f"VERIFIED {name}", flush=True)
                    repaired = True
                    break
                print(
                    f"RETRY {name} cli_status={process.returncode} remaining={len(remaining)}",
                    flush=True,
                )
                time.sleep(attempt * 3)
            if not repaired:
                raise RuntimeError(f"failed to repair remote directory: {name}")
        failures = audit_files(args.bdpan, remote_root, files)

    result = {
        "schema": "bfm_zero_x2_baidu_sync_result_v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "remote_root": remote_root,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "missing_or_mismatched": failures,
        "remote_verified": not failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if args.result is not None:
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        atomic_write(args.result, payload)
        sidecar = args.result.with_name(f"{args.result.name}.sha256")
        atomic_write(
            sidecar,
            f"{hashlib.sha256(payload).hexdigest()}  {args.result.name}\n".encode(),
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
