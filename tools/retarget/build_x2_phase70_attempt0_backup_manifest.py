#!/usr/bin/env python3
"""Freeze and inventory the failed Phase70 v3 attempt0 without touching v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = ROOT / "artifacts/retarget/x2_phase70_attempt0"
REMOTE = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase70_attempt0"
SESSION = Path(
    "/home/yu/.codex/sessions/2026/08/12/"
    "rollout-2026-08-12T11-12-06-019ff3f4-adf0-7b53-877c-84af439f40b1.jsonl"
)
MANIFEST = ROOT / "reports/retarget/x2_phase70_attempt0_backup_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def session_payload(line_number: int) -> dict[str, object]:
    if not SESSION.is_file():
        raise FileNotFoundError(SESSION)
    with SESSION.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            if index == line_number:
                return json.loads(line)["payload"]
    raise RuntimeError(f"session line {line_number} is absent")


def session_change_content(line_number: int, repo_path: str) -> bytes:
    absolute = str(ROOT / repo_path)
    change = session_payload(line_number).get("changes", {}).get(absolute, {})
    content = change.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"full historical content absent at line {line_number}: {repo_path}")
    return content.encode()


def session_change_diff(line_number: int, repo_path: str) -> str:
    absolute = str(ROOT / repo_path)
    change = session_payload(line_number).get("changes", {}).get(absolute, {})
    unified_diff = change.get("unified_diff")
    if not isinstance(unified_diff, str):
        raise RuntimeError(f"historical diff absent at line {line_number}: {repo_path}")
    return unified_diff


def recover_attempt0_finalizer() -> bytes:
    repo_path = "tools/retarget/finalize_x2_phase70_long_lookahead.py"
    initial = session_change_content(6233, repo_path)
    with tempfile.TemporaryDirectory(prefix="phase70_attempt0_finalizer_") as raw_temp:
        temp = Path(raw_temp)
        target = temp / "finalizer.py"
        target.write_bytes(initial)
        for line_number in (6284, 6347, 6407):
            patch = (
                "--- finalizer.py\n+++ finalizer.py\n"
                + session_change_diff(line_number, repo_path)
            )
            completed = subprocess.run(
                ["patch", "--batch", "--fuzz=0", "-d", str(temp), "-p0"],
                input=patch.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout.decode(errors="replace"))
        return target.read_bytes()


def materialize(path: Path, value: bytes, expected_sha256: str) -> None:
    actual = sha256_bytes(value)
    if actual != expected_sha256:
        raise RuntimeError(
            f"reconstructed SHA mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    if path.exists():
        if not path.is_file() or sha256(path) != expected_sha256:
            raise RuntimeError(f"refusing to overwrite mismatched frozen snapshot: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def exact_source(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = path.read_bytes()
    if sha256_bytes(value) != expected_sha256:
        raise RuntimeError(f"source drift prevents attempt0 freeze: {path}")
    return value


def freeze_snapshots() -> None:
    runner = session_change_content(
        6655, "artifacts/retarget/x2_phase70_attempt0/runner_attempt0.py"
    )
    run_script = session_change_content(
        6655, "artifacts/retarget/x2_phase70_attempt0/run_phase70_attempt0.sh"
    )
    # The patch-event serializer retained one presentation-only blank line after
    # each added file. apply_patch did not write that extra line to disk.
    if runner.endswith(b"\n\n"):
        runner = runner[:-1]
    if run_script.endswith(b"\n\n"):
        run_script = run_script[:-1]
    materialize(
        SNAPSHOT_ROOT / "runner_attempt0.py",
        runner,
        "c62f8bdac7de750a41608e4a6bd38acb0c5366ca32fe7142f73dc13a0a1468dc",
    )
    materialize(
        SNAPSHOT_ROOT / "run_phase70_attempt0.sh",
        run_script,
        "aba512353a0d3070dad3e462fd47a613a0a8e4131e9a35d710d87990915e99cb",
    )
    materialize(
        SNAPSHOT_ROOT / "finalize_phase70_long_lookahead_attempt0.py",
        recover_attempt0_finalizer(),
        "6dd4bdc7bcc8312256f23c3053599604a93904f4da627f0fde48d95c6962b4f4",
    )
    materialize(
        SNAPSHOT_ROOT / "test_phase70_long_lookahead_attempt0.py",
        session_change_content(6199, "tests/test_phase70_long_lookahead.py"),
        "7a0bf0299041a225c3d2da8050cd14d1027c5c3cd052b1a8ddb085a64f05b94c",
    )
    sources = (
        (
            "src/cwi_x2/phase70_long_lookahead.py",
            "phase70_long_lookahead_attempt0.py",
            "9d36a8e6a37393374f740ed00b0292033d2a5d0148ba1cc9e1d626eb49f2331a",
        ),
        (
            "tests/test_phase70_long_lookahead_finalizer.py",
            "test_phase70_long_lookahead_finalizer_attempt0.py",
            "8a0433a341f9ea0d888f8ddbb58ff2c2dbfa51e43aad6a1c6c81b53b6f64da6c",
        ),
        (
            "src/cwi_x2/phase69_reward_attribution.py",
            "phase69_reward_attribution_attempt0_dependency.py",
            "b3ebe7bc500cedc3ee9af23bafb5936bf92f9801c5ae2722fff009ec157d3295",
        ),
        (
            "src/cwi_x2/phase68_residual_ppo.py",
            "phase68_residual_ppo_attempt0_dependency.py",
            "c67dd5f8bb8387ef9b1ec616cf0d885846944f19ed3e16155b4b607363f85157",
        ),
        (
            "src/cwi_x2/phase_conditioned_knee_residual.py",
            "phase_conditioned_knee_residual_attempt0_dependency.py",
            "f365b1318be1023d4352c78d26657873f3dfc89803148ef7f61a5bd057019d0b",
        ),
        (
            "tools/retarget/run_with_gpu_ledger.py",
            "run_with_gpu_ledger_attempt0.py",
            "d5a4f1a6b15cb827b5cba1bc7ff49cd7b794247259d23768686a7507bc45383e",
        ),
    )
    for source_name, remote_name, expected in sources:
        materialize(
            SNAPSHOT_ROOT / remote_name,
            exact_source(ROOT / source_name, expected),
            expected,
        )
    materialize(
        SNAPSHOT_ROOT / "x2_phase70_long_lookahead_attempt0.log",
        exact_source(
            Path("/tmp/x2_phase70_logs/long_lookahead.log"),
            "fbd24db721e0876c4fafd8b6c2b916239cdf5cdb9721eed3608c55db4b31dc0c",
        ),
        "fbd24db721e0876c4fafd8b6c2b916239cdf5cdb9721eed3608c55db4b31dc0c",
    )


def entry(path: Path, remote_name: str, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
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
    freeze_snapshots()
    source_files = (
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg.json", "x2_phase70_long_lookahead_prereg.json", "prereg_v1_no_launch"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg.json.sha256", "x2_phase70_long_lookahead_prereg.json.sha256", "prereg_v1_sidecar"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg_v2.json", "x2_phase70_long_lookahead_prereg_v2.json", "prereg_v2_no_launch"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg_v2.json.sha256", "x2_phase70_long_lookahead_prereg_v2.json.sha256", "prereg_v2_sidecar"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg_v3.json", "x2_phase70_long_lookahead_prereg_v3.json", "prereg_v3_attempt0"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg_v3.json.sha256", "x2_phase70_long_lookahead_prereg_v3.json.sha256", "prereg_v3_sidecar"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_attempt0_failure.json", "x2_phase70_long_lookahead_attempt0_failure.json", "authoritative_failure"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_attempt0_failure.json.sha256", "x2_phase70_long_lookahead_attempt0_failure.json.sha256", "failure_sidecar"),
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_resource.json", "x2_phase70_long_lookahead_attempt0_resource.json", "attempt0_resource_ledger"),
        (SNAPSHOT_ROOT / "x2_phase70_long_lookahead_attempt0.log", "x2_phase70_long_lookahead_attempt0.log", "authoritative_traceback_log"),
        (SNAPSHOT_ROOT / "runner_attempt0.py", "runner_attempt0.py", "frozen_runner"),
        (SNAPSHOT_ROOT / "run_phase70_attempt0.sh", "run_phase70_attempt0.sh", "frozen_run_script"),
        (SNAPSHOT_ROOT / "phase70_long_lookahead_attempt0.py", "phase70_long_lookahead_attempt0.py", "frozen_lookahead_module"),
        (SNAPSHOT_ROOT / "finalize_phase70_long_lookahead_attempt0.py", "finalize_phase70_long_lookahead_attempt0.py", "frozen_finalizer"),
        (SNAPSHOT_ROOT / "test_phase70_long_lookahead_attempt0.py", "test_phase70_long_lookahead_attempt0.py", "frozen_lookahead_test"),
        (SNAPSHOT_ROOT / "test_phase70_long_lookahead_finalizer_attempt0.py", "test_phase70_long_lookahead_finalizer_attempt0.py", "frozen_finalizer_test"),
        (SNAPSHOT_ROOT / "phase69_reward_attribution_attempt0_dependency.py", "phase69_reward_attribution_attempt0_dependency.py", "frozen_phase69_dependency"),
        (SNAPSHOT_ROOT / "phase68_residual_ppo_attempt0_dependency.py", "phase68_residual_ppo_attempt0_dependency.py", "frozen_phase68_dependency"),
        (SNAPSHOT_ROOT / "phase_conditioned_knee_residual_attempt0_dependency.py", "phase_conditioned_knee_residual_attempt0_dependency.py", "frozen_residual_dependency"),
        (SNAPSHOT_ROOT / "run_with_gpu_ledger_attempt0.py", "run_with_gpu_ledger_attempt0.py", "frozen_resource_ledger_tool"),
        (Path(__file__).resolve(), Path(__file__).name, "backup_manifest_builder"),
    )
    entries = [entry(*item) for item in source_files]
    prereg = json.loads(
        (ROOT / "reports/retarget/x2_phase70_long_lookahead_prereg_v3.json").read_text()
    )
    expected_code = prereg["immutable_code"]
    frozen_code = {
        "runner_sha256": sha256(SNAPSHOT_ROOT / "runner_attempt0.py"),
        "run_script_sha256": sha256(SNAPSHOT_ROOT / "run_phase70_attempt0.sh"),
        "lookahead_module_sha256": sha256(SNAPSHOT_ROOT / "phase70_long_lookahead_attempt0.py"),
        "finalizer_sha256": sha256(SNAPSHOT_ROOT / "finalize_phase70_long_lookahead_attempt0.py"),
        "lookahead_test_sha256": sha256(SNAPSHOT_ROOT / "test_phase70_long_lookahead_attempt0.py"),
        "finalizer_test_sha256": sha256(SNAPSHOT_ROOT / "test_phase70_long_lookahead_finalizer_attempt0.py"),
        "phase69_module_sha256": sha256(SNAPSHOT_ROOT / "phase69_reward_attribution_attempt0_dependency.py"),
        "phase68_interface_sha256": sha256(SNAPSHOT_ROOT / "phase68_residual_ppo_attempt0_dependency.py"),
        "residual_module_sha256": sha256(SNAPSHOT_ROOT / "phase_conditioned_knee_residual_attempt0_dependency.py"),
    }
    if frozen_code != expected_code:
        raise RuntimeError("frozen code inventory does not exactly match v3 prereg")
    absent_paths = (
        ROOT / "reports/retarget/x2_phase70_long_lookahead_screen.json",
        ROOT / "reports/retarget/x2_phase70_long_lookahead_result.json",
        ROOT / "reports/retarget/x2_phase70_long_lookahead_result.json.sha256",
        ROOT / "reports/retarget/x2_phase70_long_lookahead.md",
        ROOT / "artifacts/retarget/x2_phase70_long_lookahead/rollout_evidence.pt",
        ROOT / "artifacts/retarget/x2_phase70_long_lookahead/.rollout_evidence.pt.tmp",
    )
    if any(path.exists() for path in absent_paths):
        raise RuntimeError("an attempt0 output declared absent now exists")
    checked_at = (
        args.checked_at
        if args.remote_verified and args.checked_at
        else datetime.now().astimezone().isoformat(timespec="seconds")
        if args.remote_verified
        else None
    )
    report = {
        "schema": "x2_phase70_attempt0_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE,
        "attempt": "v3_attempt0",
        "decision": "FAIL_POST_ROLLOUT_DIAGNOSTIC_STOP",
        "scientific_result": None,
        "formal_optimizer_steps": 0,
        "optimizer_state_entries": 0,
        "model_checkpoint_count": 0,
        "v3_launch_consumed": True,
        "ledger_exit_code_reliable": False,
        "failure_stage": "post-rollout policy-logprob reconstruction before screen or raw bundle write",
        "frozen_code_matches_v3_prereg": True,
        "frozen_code_sha256": frozen_code,
        "attempt_outputs_absent": [str(path.relative_to(ROOT)) for path in absent_paths],
        "input_references": [
            {"role": "base_checkpoint", "bytes": 1809013, "sha256": "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb", "backed_up_in_prior_phase": True},
            {"role": "source_zero_residual_checkpoint", "bytes": 20949, "sha256": "801b433da9e1c34569b590192d81dcddc26cc9b115d1903049729ad154d4eaf3", "backed_up_in_prior_phase": True},
            {"role": "phase69_reference_bundle", "bytes": 11749092, "sha256": "db249ea619739c3128e2311039c65d26f19c61e7aa0f0451e9c8fc580c2c2f18", "backed_up_in_prior_phase": True},
        ],
        "attempt_evidence_payload_count": 20,
        "files": entries,
        "payload_count_including_builder": len(entries),
        "payload_total_bytes_including_builder": sum(item["bytes"] for item in entries),
        "remote_expected_unique_file_count_including_controls": len(entries) + 2,
        "remote_listing_file_count": len(entries) + 2 if args.remote_verified else None,
        "remote_listing_byte_sizes_match": args.remote_verified,
        "remote_listing_checked_at": checked_at,
        "verification": (
            "All 23 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256."
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
                "bytes": report["payload_total_bytes_including_builder"],
                "manifest_sha256": sha256(MANIFEST),
                "remote_verified": args.remote_verified,
            }
        )
    )


if __name__ == "__main__":
    main()
