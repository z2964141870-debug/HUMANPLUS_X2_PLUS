#!/usr/bin/env python3
"""Build the immutable local/remote inventory for the Phase62 backup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase62"
FILES = (
    "reports/retarget/x2_action_sensitivity_phase62_screen.json",
    "reports/retarget/x2_action_sensitivity_phase62_resource.json",
    "reports/retarget/x2_action_sensitivity_phase62_result.json",
    "reports/retarget/x2_action_sensitivity_phase62_result.json.sha256",
    "reports/retarget/x2_action_sensitivity_phase62.md",
    "EXPERIMENTS.md",
    "scripts/run_x2_upper_robust_one_update_phase56.py",
    "tools/retarget/finalize_x2_action_sensitivity_phase62.py",
    "tests/test_phase62_action_sensitivity.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    entries = []
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": relative,
                "remote": f"{REMOTE}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output = ROOT / "reports/retarget/x2_action_sensitivity_phase62_backup_manifest.json"
    report = {
        "schema": "x2_action_sensitivity_phase62_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "decision": "PASS_DIAGNOSTIC_DIRECTION_ONLY",
        "promotion": "none",
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "verification": "Each upload must be checked by remote path and byte size; SHA256 is preserved in this manifest because bdpan does not expose remote content hashes.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(entries), "bytes": report["total_bytes"], "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
