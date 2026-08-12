#!/usr/bin/env python3
"""Build the immutable Phase70 rerun1 terminal-failure backup inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = (
    "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/"
    "task2_backward_pitch/phase70_rerun1_failure"
)
SNAPSHOT_DIR = ROOT / "artifacts/retarget/x2_phase70_rerun1_failure"
LOG_SOURCE = Path("/tmp/x2_phase70_logs/long_lookahead_rerun1.log")
LOG_SNAPSHOT = SNAPSHOT_DIR / "x2_phase70_long_lookahead_rerun1.log"
LOG_SHA256 = "7006499189a6266cee5836cf08818dd18b046857cd841f00972f8c8f65e52b3e"
MANIFEST = ROOT / "reports/retarget/x2_phase70_rerun1_failure_backup_manifest.json"
PREREG = ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg_v4.json"

EXPECTED_CODE = {
    "runner_sha256": "a6a50aecc64be5ca10a19a5e6411921707852bb12e9ed02bd42f96415b64dd85",
    "run_script_sha256": "270a44d5ab6402a7a1cd11091213ebd0670388174cfbcabaf4e73ebde00fd6ae",
    "lookahead_module_sha256": "9d36a8e6a37393374f740ed00b0292033d2a5d0148ba1cc9e1d626eb49f2331a",
    "finalizer_sha256": "4af1da0b9dd229d011a4a74281855b626c51b6bbe204152021662d5f5812f431",
    "lookahead_test_sha256": "32fed6c440f8b6445fd12e324bdfdddc6f08ee465bbd7245d038d2564fd27761",
    "finalizer_test_sha256": "8a0433a341f9ea0d888f8ddbb58ff2c2dbfa51e43aad6a1c6c81b53b6f64da6c",
    "phase69_module_sha256": "b3ebe7bc500cedc3ee9af23bafb5936bf92f9801c5ae2722fff009ec157d3295",
    "phase68_interface_sha256": "c67dd5f8bb8387ef9b1ec616cf0d885846944f19ed3e16155b4b607363f85157",
    "residual_module_sha256": "f365b1318be1023d4352c78d26657873f3dfc89803148ef7f61a5bd057019d0b",
}
CODE_PATHS = {
    "runner_sha256": ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py",
    "run_script_sha256": ROOT / "scripts/run_x2_phase70_long_lookahead.sh",
    "lookahead_module_sha256": ROOT / "src/cwi_x2/phase70_long_lookahead.py",
    "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase70_long_lookahead.py",
    "lookahead_test_sha256": ROOT / "tests/test_phase70_long_lookahead.py",
    "finalizer_test_sha256": ROOT / "tests/test_phase70_long_lookahead_finalizer.py",
    "phase69_module_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
    "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py",
    "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"immutable drift: {path}: expected {expected}, got {actual}")


def require_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text() != expected:
        raise RuntimeError(f"sidecar mismatch: {sidecar}")


def freeze_log() -> None:
    require_sha(LOG_SOURCE, LOG_SHA256)
    value = LOG_SOURCE.read_bytes()
    if LOG_SNAPSHOT.exists():
        require_sha(LOG_SNAPSHOT, LOG_SHA256)
        return
    LOG_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    LOG_SNAPSHOT.write_bytes(value)
    require_sha(LOG_SNAPSHOT, LOG_SHA256)


def item(path: Path, remote_name: str, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "remote": f"{REMOTE}/{remote_name}",
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
    args = parser.parse_args()

    freeze_log()
    require_sidecar(PREREG)
    failure = ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1_failure.json"
    raw_bundle = (
        ROOT
        / "artifacts/retarget/x2_phase70_long_lookahead/rollout_evidence_rerun1.pt"
    )
    require_sidecar(failure)
    require_sidecar(raw_bundle)
    prereg = json.loads(PREREG.read_text())
    if prereg.get("immutable_code") != EXPECTED_CODE:
        raise RuntimeError("v4 prereg immutable-code inventory changed")
    for name, path in CODE_PATHS.items():
        require_sha(path, EXPECTED_CODE[name])

    screen = ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1_screen.json"
    resource = ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1_resource.json"
    screen_data = json.loads(screen.read_text())
    failure_data = json.loads(failure.read_text())
    if screen_data.get("decision") != "FAIL_INVALID_STOP":
        raise RuntimeError("Phase70 rerun1 is no longer the frozen invalid screen")
    if failure_data.get("decision") != "FAIL_INVALID_STOP":
        raise RuntimeError("Phase70 rerun1 failure decision changed")
    if sha256(raw_bundle) != screen_data.get("rollout_bundle_sha256"):
        raise RuntimeError("raw bundle no longer matches the invalid screen")
    for absent in (
        ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1_result.json",
        ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1_result.json.sha256",
        ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1.md",
    ):
        if absent.exists():
            raise RuntimeError(f"unexpected Phase70 finalized result exists: {absent}")

    files = (
        (failure, failure.name, "authoritative_terminal_failure"),
        (failure.with_suffix(failure.suffix + ".sha256"), failure.name + ".sha256", "failure_sidecar"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_rerun1_failure.md", "x2_phase70_long_lookahead_rerun1_failure.md", "failure_summary"),
        (PREREG, PREREG.name, "v4_prereg"),
        (PREREG.with_suffix(PREREG.suffix + ".sha256"), PREREG.name + ".sha256", "v4_prereg_sidecar"),
        (screen, screen.name, "invalid_screen"),
        (resource, resource.name, "resource_ledger"),
        (LOG_SNAPSHOT, LOG_SNAPSHOT.name, "authoritative_traceback_log"),
        (raw_bundle, raw_bundle.name, "raw_rollout_evidence_not_checkpoint"),
        (raw_bundle.with_suffix(raw_bundle.suffix + ".sha256"), raw_bundle.name + ".sha256", "raw_bundle_sidecar"),
        (ROOT / "EXPERIMENTS.md", "EXPERIMENTS.md", "experiment_ledger"),
        (CODE_PATHS["runner_sha256"], "runner_v4.py", "frozen_runner"),
        (CODE_PATHS["run_script_sha256"], "run_phase70_rerun1_v4.sh", "frozen_run_script"),
        (CODE_PATHS["lookahead_module_sha256"], "phase70_long_lookahead_v4.py", "frozen_lookahead_module"),
        (CODE_PATHS["finalizer_sha256"], "finalize_phase70_long_lookahead_v4.py", "frozen_finalizer"),
        (CODE_PATHS["lookahead_test_sha256"], "test_phase70_long_lookahead_v4.py", "frozen_lookahead_test"),
        (CODE_PATHS["finalizer_test_sha256"], "test_phase70_long_lookahead_finalizer_v4.py", "frozen_finalizer_test"),
        (CODE_PATHS["phase69_module_sha256"], "phase69_reward_attribution_v4_dependency.py", "frozen_phase69_dependency"),
        (CODE_PATHS["phase68_interface_sha256"], "phase68_residual_ppo_v4_dependency.py", "frozen_phase68_dependency"),
        (CODE_PATHS["residual_module_sha256"], "phase_conditioned_knee_residual_v4_dependency.py", "frozen_residual_dependency"),
        (ROOT / "artifacts/retarget/x2_phase70_attempt0/repair_only.patch", "phase70_repair_only.patch", "attempt0_to_v4_repair_patch"),
        (ROOT / "tools/retarget/run_with_gpu_ledger.py", "run_with_gpu_ledger.py", "resource_ledger_tool"),
        (Path(__file__).resolve(), Path(__file__).name, "backup_manifest_builder"),
    )
    entries = [item(*row) for row in files]
    if len(entries) != 23 or len({Path(row["remote"]).name for row in entries}) != 23:
        raise RuntimeError("Phase70 rerun1 payload names are not exactly 23 unique files")

    checked_at = (
        args.checked_at
        if args.remote_verified and args.checked_at
        else datetime.now().astimezone().isoformat(timespec="seconds")
        if args.remote_verified
        else None
    )
    report = {
        "schema": "x2_phase70_rerun1_failure_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "attempt": "v4_repair_rerun1",
        "decision": "FAIL_INVALID_STOP",
        "terminal_governance_status": "FAIL_IMPLEMENTATION_STOP_NO_MORE_PHASE70_RERUN",
        "scientific_result": None,
        "formal_optimizer_steps": 0,
        "optimizer_state_entries": 0,
        "checkpoint_count": 0,
        "cumulative_phase70_launches": 2,
        "third_phase70_launch_forbidden": True,
        "ledger_child_exit_code_reliable": False,
        "normal_result_absent_by_design": True,
        "frozen_code_matches_v4_prereg": True,
        "raw_bundle": {
            "bytes": raw_bundle.stat().st_size,
            "sha256": sha256(raw_bundle),
            "artifact_role": "raw rollout evidence; not a model checkpoint",
        },
        "failed_technical_gates": list(failure_data["failed_technical_gates"]),
        "attempt0_backup_reference": {
            "remote_root": "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase70_attempt0",
            "manifest_sha256": "5fd585a032bcb62f47b289b82b6feabd1a8a3f330de6723a5e86159309351c9f",
            "remote_unique_files": 23,
            "remote_total_bytes": 307988,
        },
        "immutable_input_references": [
            {"role": "base_checkpoint", "bytes": 1809013, "sha256": "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb", "backed_up_previously": True},
            {"role": "source_zero_residual_checkpoint", "bytes": 20949, "sha256": "801b433da9e1c34569b590192d81dcddc26cc9b115d1903049729ad154d4eaf3", "backed_up_previously": True},
            {"role": "phase69_reference_bundle", "bytes": 11749092, "sha256": "db249ea619739c3128e2311039c65d26f19c61e7aa0f0451e9c8fc580c2c2f18", "backed_up_previously": True},
        ],
        "files": entries,
        "payload_count": len(entries),
        "payload_total_bytes": sum(int(row["bytes"]) for row in entries),
        "remote_expected_unique_file_count_including_controls": len(entries) + 2,
        "remote_listing_file_count": len(entries) + 2 if args.remote_verified else None,
        "remote_listing_byte_sizes_match": args.remote_verified,
        "remote_listing_checked_at": checked_at,
        "verification": (
            "All 25 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256."
            if args.remote_verified
            else "Local immutable inventory complete; remote byte-size verification pending."
        ),
    }
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n")
    MANIFEST.with_suffix(MANIFEST.suffix + ".sha256").write_text(
        f"{sha256(MANIFEST)}  {MANIFEST.name}\n"
    )
    print(
        json.dumps(
            {
                "payloads": len(entries),
                "remote_files_with_controls": len(entries) + 2,
                "bytes": report["payload_total_bytes"],
                "manifest_sha256": sha256(MANIFEST),
                "remote_verified": args.remote_verified,
            }
        )
    )


if __name__ == "__main__":
    main()
