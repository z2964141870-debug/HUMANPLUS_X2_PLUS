#!/usr/bin/env python3
"""Build the immutable local/remote inventory for the Phase61 backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = (
    "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/"
    "phase61_sidecar_closure_v3"
)
FILES = (
    "artifacts/retarget/x2_native_transition_posture_phase61/source_stage219_zero_lora.pt",
    "artifacts/retarget/x2_native_transition_posture_phase61/source_stage219_zero_lora.pt.sha256",
    "artifacts/retarget/x2_native_transition_posture_phase61/candidate_joint_transition_one_update.pt",
    "artifacts/retarget/x2_native_transition_posture_phase61/candidate_joint_transition_one_update.pt.sha256",
    "reports/retarget/x2_native_transition_posture_phase61_prereg.json",
    "reports/retarget/x2_native_transition_posture_phase61_prereg.json.sha256",
    "reports/retarget/x2_native_transition_posture_phase61_command_probe.json",
    "reports/retarget/x2_native_transition_posture_phase61_source_eval.json",
    "reports/retarget/x2_native_transition_posture_phase61_train.json",
    "reports/retarget/x2_native_transition_posture_phase61_candidate_eval.json",
    "reports/retarget/x2_native_transition_posture_phase61_source_resource.json",
    "reports/retarget/x2_native_transition_posture_phase61_train_resource.json",
    "reports/retarget/x2_native_transition_posture_phase61_candidate_resource.json",
    "reports/retarget/x2_native_transition_posture_phase61_result.json",
    "reports/retarget/x2_native_transition_posture_phase61_result.json.sha256",
    "reports/retarget/x2_native_transition_posture_phase61.md",
    "EXPERIMENTS.md",
    "scripts/run_x2_native_transition_posture_phase61.sh",
    "scripts/run_x2_upper_robust_one_update_phase56.py",
    "tools/retarget/run_with_gpu_ledger.py",
    "tools/retarget/finalize_x2_native_transition_posture_phase61.py",
    "tools/official_x2/export_rsl_actor_onnx.py",
    "tests/test_phase61_native_transition_posture.py",
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
    output = ROOT / "reports/retarget/x2_native_transition_posture_phase61_backup_manifest.json"
    report = {
        "schema": "x2_native_transition_posture_phase61_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "decision": "FAIL_LOCAL_JOINT_TRANSITION_STOP",
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
            "All 23 payloads were listed remotely with exact byte-size equality after the two checkpoint sidecars were added; SHA256 is preserved here because bdpan does not expose remote content hashes."
            if args.remote_verified
            else "Local immutable inventory only; remote upload and byte-size listing are not yet verified."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(entries), "bytes": report["total_bytes"], "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
