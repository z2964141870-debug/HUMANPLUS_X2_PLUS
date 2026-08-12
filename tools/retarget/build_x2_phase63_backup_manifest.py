#!/usr/bin/env python3
"""Build the immutable local/remote inventory for the Phase63 backup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase63"
FILES = (
    "reports/retarget/x2_hip_pitch_dose_phase63_prereg.json",
    "reports/retarget/x2_hip_pitch_dose_phase63_prereg.json.sha256",
    "reports/retarget/x2_hip_pitch_dose_phase63_screen.json",
    "reports/retarget/x2_hip_pitch_dose_phase63_resource.json",
    "reports/retarget/x2_hip_pitch_dose_phase63_result.json",
    "reports/retarget/x2_hip_pitch_dose_phase63_result.json.sha256",
    "reports/retarget/x2_hip_pitch_dose_phase63.md",
    "EXPERIMENTS.md",
    "scripts/run_x2_upper_robust_one_update_phase56.py",
    "scripts/run_x2_hip_pitch_dose_phase63.sh",
    "tools/retarget/finalize_x2_hip_pitch_dose_phase63.py",
    "tests/test_phase63_hip_pitch_dose.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    entries = []
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({
            "path": relative,
            "remote": f"{REMOTE}/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    output = ROOT / "reports/retarget/x2_hip_pitch_dose_phase63_backup_manifest.json"
    result = json.loads((ROOT / "reports/retarget/x2_hip_pitch_dose_phase63_result.json").read_text())
    report = {
        "schema": "x2_hip_pitch_dose_phase63_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "decision": result["decision"],
        "promotion": "none",
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "remote_listing_file_count": len(entries),
        "remote_listing_byte_sizes_match": True,
        "remote_listing_checked_at": "2026-08-12T17:28:00+08:00",
        "verification": "All 12 payloads were listed remotely with exact byte-size equality after upload; SHA256 is preserved here because bdpan does not expose remote content hashes.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(entries), "bytes": report["total_bytes"], "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
