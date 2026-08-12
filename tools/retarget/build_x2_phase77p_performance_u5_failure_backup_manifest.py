#!/usr/bin/env python3
"""Build and validate the immutable Phase77p update-5 failure backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim")
REPORT = ROOT / "reports/retarget"
ARTIFACT = ROOT / "artifacts/retarget/x2_phase77p_performance_campaign"
LOG = Path("/tmp/x2_phase77p_performance_logs")
REMOTE_ROOT = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase77p_performance_u5_failure"
MANIFEST = REPORT / "x2_phase77p_performance_u5_failure_backup_manifest.json"
FREEZE = REPORT / "x2_phase77p_performance_u5_failure_freeze.json"
FREEZE_MD = REPORT / "x2_phase77p_performance_u5_failure_freeze.md"

PREREG_HASHES = {
    "x2_phase77p_performance_campaign_prereg.json": "84ff45fcb10a4cfd0cacde594913590b4572868fbb74f86a8fb32e74e0f58334",
    "x2_phase77p_performance_campaign_prereg_v2.json": "9797181c60f072d06dfcb136c36ac381bd092c551b2c5c8a275d1cc082858481",
    "x2_phase77p_performance_campaign_prereg_v3.json": "3dc02e92e68695e4af446f82e6564ac0b4ec67ed2ffe9114f6c530aeec2c1f84",
}
CHECKPOINT_HASHES = {
    770101: "007314df98c54b92c65dbcc4149d49346629b619b4702eca83ffb9973474e4d8",
    770102: "d237d1d6891a577aad31beb41528520b42532a58a842ed2147dafd7533f46534",
}
CODE_HASHES = {
    "scripts/run_x2_phase77p_performance_campaign.py": "dfa46691ed87a62c935528847e2b930d0d91db55a13dd45495c037d9f914306f",
    "scripts/run_x2_phase77p_performance_campaign.sh": "48dacdb1d77bf2844e2b0dba9c4dd9e6fd55b9d3a44a89cb8dd91aca53ceffcf",
    "src/cwi_x2/phase77p_performance_campaign.py": "93b2d1c3370125c453a48fdce4a8292bedc13645e29ead407195655739ca54b0",
    "tests/test_phase77p_performance_campaign.py": "fcc2686a259ee891c33fd5ffda15edfba38ea0692e1d90ae8705ab7994a5b13d",
    "tests/test_phase77p_performance_finalizer.py": "3810819cf5f2c703e3407656a330088442da6cbd4d3095def4b5e50eb4866ba0",
    "tools/retarget/finalize_x2_phase77p_performance_campaign.py": "765789d3e00d4813edf041ae4dc7406a67ff56f11c627dfae6d9fbd30c20092d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_exact(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"immutable file absent or changed: {path}")


def require_sidecar(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"sidecar absent or mismatched: {sidecar}")
    return sidecar


def shown_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def entry(path: Path, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": shown_path(path),
        "remote": f"{REMOTE_ROOT}/{path.name}",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
    args = parser.parse_args()

    primaries: list[tuple[Path, str]] = []
    for name, digest in PREREG_HASHES.items():
        path = REPORT / name
        require_exact(path, digest)
        primaries.append((path, "preregistration"))

    resource_paths: list[Path] = []
    for train_seed, checkpoint_digest in CHECKPOINT_HASHES.items():
        checkpoint = ARTIFACT / f"seed{train_seed}_u5.pt"
        require_exact(checkpoint, checkpoint_digest)
        primaries.append((checkpoint, "update5_checkpoint"))

        train = REPORT / f"x2_phase77p_seed{train_seed}_u5_train.json"
        train_payload = json.loads(train.read_text(encoding="utf-8"))
        train_gates = {
            "schema": train_payload.get("schema") == "x2_phase77p_performance_train_v1",
            "decision": train_payload.get("decision") == "SEGMENT_VALID",
            "seed": train_payload.get("seed") == train_seed,
            "updates": [row.get("update_index") for row in train_payload.get("updates", [])] == [1, 2, 3, 4, 5],
            "optimizer_steps": train_payload.get("optimizer_steps") == 20,
            "checkpoint": train_payload.get("checkpoint_sha256") == checkpoint_digest,
        }
        if not all(train_gates.values()):
            raise RuntimeError(f"train provenance drift seed={train_seed}: {train_gates}")
        train_resource = REPORT / f"x2_phase77p_seed{train_seed}_u5_train_resource.json"
        train_log = LOG / f"seed{train_seed}_u5_train.log"
        primaries.extend(((train, "update5_train_report"), (train_resource, "train_resource_ledger"), (train_log, "immutable_train_log")))
        resource_paths.append(train_resource)

        for eval_seed in (771001, 771002, 771003):
            for lane in ("A", "B"):
                stem = f"x2_phase77p_seed{train_seed}_u5_eval{eval_seed}_lane{lane}"
                evaluation = REPORT / f"{stem}.json"
                eval_payload = json.loads(evaluation.read_text(encoding="utf-8"))
                eval_gates = {
                    "schema": eval_payload.get("schema") == "x2_phase77p_performance_eval_v1",
                    "decision": eval_payload.get("decision") == "EVAL_FINITE",
                    "train_seed": eval_payload.get("candidate_train_seed") == train_seed,
                    "update": eval_payload.get("candidate_update_index") == 5,
                    "eval_seed": eval_payload.get("eval_seed") == eval_seed,
                    "lane": eval_payload.get("lane") == lane,
                    "finite": eval_payload.get("finite") is True and eval_payload.get("per_env_complete") is True,
                    "no_optimizer": eval_payload.get("optimizer_steps") == 0 and eval_payload.get("checkpoint_writes") == 0,
                    "checkpoint": eval_payload.get("candidate_checkpoint_sha256") == checkpoint_digest,
                }
                if not all(eval_gates.values()):
                    raise RuntimeError(f"eval provenance drift {stem}: {eval_gates}")
                resource = REPORT / f"{stem}_resource.json"
                log = LOG / f"seed{train_seed}_u5_eval{eval_seed}_lane{lane}.log"
                primaries.extend(((evaluation, "update5_eval_report"), (resource, "eval_resource_ledger"), (log, "immutable_eval_log")))
                resource_paths.append(resource)

    gate = REPORT / "x2_phase77p_performance_u5_local_gate.json"
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    gate_checks = {
        "schema": gate_payload.get("schema") == "x2_phase77p_performance_local_gate_v2",
        "decision": gate_payload.get("decision") == "FAIL_UPDATE5_NO_SAFE_SIGNAL_STOP",
        "stage": gate_payload.get("stage") == 5,
        "no_selection": gate_payload.get("selected_train_seed") is None,
        "both_fail": all(not row.get("passed", True) for row in gate_payload.get("seed_summaries", {}).values()) and len(gate_payload.get("seed_summaries", {})) == 2,
        "update10_locked": gate_payload.get("update10_training_unlocked") is False,
        "long_locked": gate_payload.get("long_training_unlocked") is False,
        "panel_locked": gate_payload.get("official_panel_unlocked") is False,
        "deployment_locked": gate_payload.get("deployment_unlocked") is False,
        "train_inventory": len(gate_payload.get("train_reports", [])) == 2,
        "eval_inventory": len(gate_payload.get("eval_reports", [])) == 12,
    }
    if not all(gate_checks.values()):
        raise RuntimeError(f"local gate drift: {gate_checks}")
    primaries.append((gate, "update5_local_gate"))

    cumulative_positive_delta = 0
    elapsed_total = 0.0
    for resource in resource_paths:
        payload = json.loads(resource.read_text(encoding="utf-8"))
        log_path = Path(payload["log_path"])
        checks = {
            "schema": payload.get("schema") == "x2_gpu_deadline_ledger_phase76_v1",
            "exit": payload.get("exit_code") == 0 and payload.get("raw_returncode") == 0,
            "autonomous": payload.get("autonomous_exit") is True,
            "timeout": payload.get("timed_out") is False,
            "signals": payload.get("term_sent") is False and payload.get("kill_sent") is False,
            "cleanup": payload.get("forced_cleanup") is False,
            "gpu": payload.get("gpu") is not None and int(payload["gpu"]["memory_used_peak_mib"]) <= 20480,
            "disk_delta": int(payload.get("disk_used_delta_bytes", 2147483649)) <= 2147483648,
            "disk_free": int(payload.get("disk_after", {}).get("free_bytes", 0)) >= 268435456000,
            "log": log_path.is_file() and payload.get("log_sha256") == sha256(log_path),
        }
        if not all(checks.values()):
            raise RuntimeError(f"resource provenance drift {resource}: {checks}")
        cumulative_positive_delta += max(0, int(payload["disk_used_delta_bytes"]))
        elapsed_total += float(payload["elapsed_s"])
    if len(resource_paths) != 14 or cumulative_positive_delta != 1896235008:
        raise RuntimeError("resource inventory or cumulative disk delta drift")
    if abs(elapsed_total - 534.8600553129945) > 1e-9:
        raise RuntimeError("resource elapsed total drift")

    require_sidecar(gate)
    freeze_payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze_payload.get("decision") != "FAIL_UPDATE5_NO_SAFE_SIGNAL_STOP" or freeze_payload.get("update10", {}).get("train_launches") != 0:
        raise RuntimeError("failure freeze decision drift")
    primaries.extend(((FREEZE, "failure_freeze"), (FREEZE_MD, "failure_markdown")))

    for relative, digest in CODE_HASHES.items():
        path = ROOT / relative
        require_exact(path, digest)
        primaries.append((path, "immutable_campaign_code_or_test"))

    builder = Path(__file__).resolve()
    primaries.append((builder, "backup_builder"))

    files: list[dict[str, object]] = []
    for path, role in primaries:
        files.append(entry(path, role))
        if path.suffix in {".json", ".pt", ".log"} and path != MANIFEST:
            files.append(entry(require_sidecar(path), f"{role}_sidecar"))
    remote_names = [Path(str(row["remote"])).name for row in files]
    if len(remote_names) != len(set(remote_names)):
        raise RuntimeError("remote basename collision")
    if len(files) != 106:
        raise RuntimeError(f"unexpected payload count: {len(files)}")

    payload = {
        "schema": "x2_phase77p_performance_u5_failure_backup_manifest_v1",
        "date": "2026-08-13",
        "remote_root": REMOTE_ROOT,
        "decision": "FAIL_UPDATE5_NO_SAFE_SIGNAL_STOP",
        "completed_stage": 5,
        "update10_training_unlocked": False,
        "update10_launch_count": 0,
        "update10_checkpoint_count": 0,
        "update10_eval_count": 0,
        "train_launch_count": 2,
        "eval_launch_count": 12,
        "resource_ledger_count": 14,
        "immutable_log_count": 14,
        "checkpoint_count": 2,
        "immutable_campaign_code_or_test_count": 6,
        "files": files,
        "payload_count": len(files),
        "payload_total_bytes": sum(int(row["bytes"]) for row in files),
        "remote_expected_unique_file_count_including_controls": len(files) + 2,
        "remote_listing_file_count": len(files) + 2 if args.remote_verified else None,
        "remote_listing_byte_sizes_match": bool(args.remote_verified),
        "remote_listing_checked_at": args.checked_at if args.remote_verified else None,
        "verification": "All 108 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256." if args.remote_verified else "Local immutable inventory complete; remote byte-size verification pending.",
        "recovery_refs": [
            {
                "role": "phase76_lineage",
                "remote_root": "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase76_pairing_lifecycle_pass",
                "manifest_sha256": "530e5a19fa2b3aabbff2df90875dd37f658abd008f5a2c1b228864e6adf569af"
            }
        ]
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    Path(str(MANIFEST) + ".sha256").write_text(f"{sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "payload_count": len(files), "payload_total_bytes": payload["payload_total_bytes"], "manifest_sha256": sha256(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
