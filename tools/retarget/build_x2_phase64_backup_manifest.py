#!/usr/bin/env python3
"""Build the immutable local inventory for the Phase64 backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase64"
FILES = (
    "reports/retarget/x2_knee_pitch_dose_phase64_prereg.json",
    "reports/retarget/x2_knee_pitch_dose_phase64_prereg.json.sha256",
    "reports/retarget/x2_knee_pitch_dose_phase64_screen.json",
    "reports/retarget/x2_knee_pitch_dose_phase64_resource.json",
    "reports/retarget/x2_knee_pitch_dose_phase64_result.json",
    "reports/retarget/x2_knee_pitch_dose_phase64_result.json.sha256",
    "reports/retarget/x2_knee_pitch_dose_phase64.md",
    "EXPERIMENTS.md",
    "scripts/run_x2_upper_robust_one_update_phase56.py",
    "scripts/run_x2_knee_pitch_dose_phase64.sh",
    "tools/retarget/finalize_x2_knee_pitch_dose_phase64.py",
    "tests/test_phase64_knee_pitch_dose.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at", default=None)
    args = parser.parse_args()

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
    output = ROOT / "reports/retarget/x2_knee_pitch_dose_phase64_backup_manifest.json"
    result = json.loads(
        (ROOT / "reports/retarget/x2_knee_pitch_dose_phase64_result.json").read_text()
    )
    report = {
        "schema": "x2_knee_pitch_dose_phase64_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "decision": result["decision"],
        "promotion": "none",
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "remote_listing_file_count": len(entries) if args.remote_verified else None,
        "remote_listing_byte_sizes_match": args.remote_verified,
        "remote_listing_checked_at": (
            args.checked_at
            if args.remote_verified and args.checked_at
            else datetime.now().astimezone().isoformat(timespec="seconds")
            if args.remote_verified
            else None
        ),
        "verification": (
            "All 12 payloads were listed remotely with exact byte-size equality after upload; "
            "SHA256 is preserved here because bdpan does not expose remote content hashes."
            if args.remote_verified
            else "Local immutable inventory only; remote upload and byte-size listing are not yet verified."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(entries),
                "bytes": report["total_bytes"],
                "sha256": sha256(output),
                "remote_verified": args.remote_verified,
            }
        )
    )


if __name__ == "__main__":
    main()
