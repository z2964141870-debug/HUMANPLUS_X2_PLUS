#!/usr/bin/env python3
"""Build the immutable inventory for all Phase69 attempts and evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase69"
FILES = (
    ("reports/retarget/x2_phase69_reward_attribution_prereg.json", None),
    ("reports/retarget/x2_phase69_reward_attribution_prereg.json.sha256", None),
    ("src/cwi_x2/phase69_reward_attribution.py", None),
    ("tools/retarget/finalize_x2_phase69_reward_attribution.py", None),
    ("tests/test_phase69_reward_attribution.py", None),
    ("tests/test_phase62_action_sensitivity.py", None),
    ("scripts/run_x2_upper_robust_one_update_phase56.py", "run_x2_upper_robust_one_update_phase56_rerun1.py"),
    ("scripts/run_x2_phase69_reward_attribution.sh", "run_x2_phase69_reward_attribution_rerun1.sh"),
    ("reports/retarget/x2_phase69_reward_attribution_attempt0_failure.json", None),
    ("reports/retarget/x2_phase69_reward_attribution_attempt0_failure.json.sha256", None),
    ("reports/retarget/x2_phase69_reward_attribution_resource.json", "x2_phase69_reward_attribution_attempt0_resource.json"),
    ("/tmp/x2_phase69_logs/attribution.log", "x2_phase69_reward_attribution_attempt0.log"),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1_prereg.json", None),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1_prereg.json.sha256", None),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1_screen.json", None),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1_resource.json", None),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1_result.json", None),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1_result.json.sha256", None),
    ("reports/retarget/x2_phase69_reward_attribution_rerun1.md", None),
    ("artifacts/retarget/x2_phase69_reward_attribution/rollout_evidence_rerun1.pt", None),
    ("artifacts/retarget/x2_phase69_reward_attribution/rollout_evidence_rerun1.pt.sha256", None),
    ("/tmp/x2_phase69_logs/attribution_rerun1.log", "x2_phase69_reward_attribution_rerun1.log"),
    ("EXPERIMENTS.md", None),
    ("tools/retarget/build_x2_phase69_backup_manifest.py", None),
)
SNAPSHOTS = (
    {
        "git_commit": "c8bd547",
        "git_path": "scripts/run_x2_upper_robust_one_update_phase56.py",
        "remote_name": "run_x2_upper_robust_one_update_phase56.py",
        "sha256": "228de1bf9067cc75059da2a3ba06eb2f178e9f18f5ca114c7a3e218facb37ade",
        "bytes": 156515,
    },
    {
        "git_commit": "c8bd547",
        "git_path": "scripts/run_x2_phase69_reward_attribution.sh",
        "remote_name": "run_x2_phase69_reward_attribution.sh",
        "sha256": "7fbc711f5ac8d7e3f881db6137ddfd4a0b2fb20ec222aadbbb7ea7ece596d3e5",
        "bytes": 2799,
    },
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
    args = parser.parse_args()
    entries = []
    for raw_path, remote_name in FILES:
        path = Path(raw_path) if raw_path.startswith("/") else ROOT / raw_path
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": raw_path,
                "remote": f"{REMOTE}/{remote_name or path.name}",
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    for snapshot in SNAPSHOTS:
        entries.append(
            {
                "path": f"git:{snapshot['git_commit']}:{snapshot['git_path']}",
                "remote": f"{REMOTE}/{snapshot['remote_name']}",
                "bytes": snapshot["bytes"],
                "sha256": snapshot["sha256"],
            }
        )
    report = {
        "schema": "x2_phase69_reward_attribution_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "decision": "FAIL_ATTRIBUTION_INVALID_STOP",
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
            "All 26 payloads were listed remotely as unique basenames with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256."
            if args.remote_verified
            else "Local inventory only; remote byte-size verification is pending."
        ),
    }
    output = ROOT / "reports/retarget/x2_phase69_reward_attribution_backup_manifest.json"
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
