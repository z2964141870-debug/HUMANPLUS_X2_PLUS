#!/usr/bin/env python3
"""Build the immutable local inventory for the Phase68 residual PPO backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase68"
FILES = (
    "reports/retarget/x2_phase_conditioned_residual_phase68_prereg.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_prereg.json.sha256",
    "reports/retarget/x2_phase_conditioned_residual_phase68_train.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_train_resource.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_source_eval.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_source_resource.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_candidate_eval.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_candidate_resource.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_result.json",
    "reports/retarget/x2_phase_conditioned_residual_phase68_result.json.sha256",
    "reports/retarget/x2_phase_conditioned_residual_phase68.md",
    "artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt",
    "artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt.sha256",
    "artifacts/retarget/x2_phase_conditioned_residual_phase68/candidate_phase_conditioned_residual_one_step.pt",
    "artifacts/retarget/x2_phase_conditioned_residual_phase68/candidate_phase_conditioned_residual_one_step.pt.sha256",
    "EXPERIMENTS.md",
    "src/cwi_x2/phase_conditioned_knee_residual.py",
    "src/cwi_x2/phase68_residual_ppo.py",
    "scripts/run_x2_upper_robust_one_update_phase56.py",
    "scripts/run_x2_phase_conditioned_residual_phase68.sh",
    "tools/retarget/finalize_x2_phase_conditioned_residual_phase68.py",
    "tests/test_phase68_residual_ppo.py",
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
    result = json.loads(
        (ROOT / "reports/retarget/x2_phase_conditioned_residual_phase68_result.json").read_text()
    )
    report = {
        "schema": "x2_phase_conditioned_residual_phase68_backup_manifest_v1",
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
            "All 22 payloads were listed remotely with exact byte-size equality; SHA256 is preserved here because bdpan does not expose remote content hashes."
            if args.remote_verified
            else "Local immutable inventory only; remote upload and byte-size listing are not yet verified."
        ),
    }
    output = ROOT / "reports/retarget/x2_phase_conditioned_residual_phase68_backup_manifest.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n"
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
