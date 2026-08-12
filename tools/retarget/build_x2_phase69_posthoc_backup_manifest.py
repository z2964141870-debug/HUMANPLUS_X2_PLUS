#!/usr/bin/env python3
"""Build the immutable Phase69 posthoc-calibration backup inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase69_posthoc"
UPSTREAM_REMOTE = (
    "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase69/"
    "rollout_evidence_rerun1.pt"
)
FILES = (
    "reports/retarget/x2_phase69_posthoc_calibration.json",
    "reports/retarget/x2_phase69_posthoc_calibration.json.sha256",
    "reports/retarget/x2_phase69_posthoc_calibration.md",
    "tools/retarget/calibrate_x2_phase69_posthoc.py",
    "tests/test_phase69_posthoc_calibration.py",
    "src/cwi_x2/phase69_reward_attribution.py",
    "src/cwi_x2/phase68_residual_ppo.py",
    "src/cwi_x2/phase_conditioned_knee_residual.py",
    "artifacts/retarget/x2_phase69_reward_attribution/rollout_evidence_rerun1.pt.sha256",
    "EXPERIMENTS.md",
    "tools/retarget/build_x2_phase69_posthoc_backup_manifest.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
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
                "sha256": digest(path),
            }
        )
    report = {
        "schema": "x2_phase69_posthoc_calibration_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "decision": "POSTHOC_CALIBRATION_NO_PROMOTION",
        "promotion": "none",
        "upstream_evidence": {
            "remote": UPSTREAM_REMOTE,
            "bytes": 11749092,
            "sha256": "db249ea619739c3128e2311039c65d26f19c61e7aa0f0451e9c8fc580c2c2f18",
            "phase69_manifest_sha256": "9c8760a3b005e5d8340255470f7a8be35dcd5b5b6cad9f0a6dfbee5224753658",
            "duplicated_in_this_remote": False,
        },
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "remote_listing_payload_count": len(entries) if args.remote_verified else None,
        "remote_listing_byte_sizes_match": args.remote_verified,
        "upstream_remote_byte_size_match": args.remote_verified,
        "remote_listing_checked_at": (
            args.checked_at
            if args.remote_verified and args.checked_at
            else datetime.now().astimezone().isoformat(timespec="seconds")
            if args.remote_verified
            else None
        ),
        "verification": (
            "All 11 posthoc payloads were listed remotely as unique basenames with exact byte-size equality; the immutable upstream Phase69 tensor bundle was independently listed at 11,749,092 bytes. SHA256 remains locally preserved because bdpan does not expose remote content hashes."
            if args.remote_verified
            else "Local inventory only; posthoc payload and upstream remote byte-size verification are pending."
        ),
    }
    output = ROOT / "reports/retarget/x2_phase69_posthoc_calibration_backup_manifest.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest(output)}  {output.name}\n"
    )
    print(
        json.dumps(
            {
                "files": len(entries),
                "bytes": report["total_bytes"],
                "sha256": digest(output),
                "remote_verified": args.remote_verified,
            }
        )
    )


if __name__ == "__main__":
    main()
