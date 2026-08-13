#!/usr/bin/env python3
"""Build a reproducibility manifest for ignored, non-cache workspace evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def tracked_diff_clean(root: Path) -> bool:
    unstaged = subprocess.run(["git", "-C", str(root), "diff", "--quiet"], check=False).returncode == 0
    staged = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "--quiet"], check=False
    ).returncode == 0
    return unstaged and staged


def ignored_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "-i", "--exclude-standard", "-z"]
    )
    result: list[Path] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = Path(item.decode())
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.suffix in EXCLUDED_SUFFIXES:
            continue
        path = root / relative
        if path.is_file():
            result.append(path)
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    entries = []
    for path in ignored_files(root):
        relative = path.relative_to(root).as_posix()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "schema": "cwi_x2_local_complete_snapshot_manifest_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace_root": str(root),
        "git": {
            "head": git(root, "rev-parse", "HEAD"),
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "branch": git(root, "branch", "--show-current"),
            "tracked_diff_clean_before_manifest_write": tracked_diff_clean(root),
        },
        "external_evidence_policy": {
            "included": "git-ignored regular files except generated caches",
            "excluded_path_parts": sorted(EXCLUDED_PARTS),
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "external_evidence_count": len(entries),
        "external_evidence_bytes": sum(entry["bytes"] for entry in entries),
        "external_evidence": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
