#!/usr/bin/env python3
"""Fail-closed aggregation of three Phase74 initial-only pairing launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp = path.with_name(f".{path.name}.tmp")
    if path.exists() or sidecar.exists() or temp.exists():
        raise FileExistsError(f"refusing to overwrite Phase74 final output: {path}")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)
    sidecar.write_text(f"{sha256(path)}  {path.name}\n")


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text() != f"{sha256(path)}  {path.name}\n":
        raise RuntimeError(f"Phase74 sidecar mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--resource", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if len(args.screen) != 3 or len(args.resource) != 3:
        raise ValueError("Phase74 finalizer requires exactly three screens and resources")
    verify_sidecar(args.prereg)
    prereg = json.loads(args.prereg.read_text())
    screens = []
    for path in args.screen:
        verify_sidecar(path)
        screens.append(json.loads(path.read_text()))
    resources = []
    for path in args.resource:
        verify_sidecar(path)
        resources.append(json.loads(path.read_text()))
    seeds = sorted(int(row["seed_index"]) for row in screens)
    labels = [row.get("label") for row in resources]
    expected_labels = {f"phase74_seed{seed}_single_process_pairing_preflight" for seed in range(3)}
    limits = prereg["resource_limits"]
    resource_valid = all(
        int(row.get("exit_code", -1)) == 0
        and isinstance(row.get("gpu"), dict)
        and float(row["gpu"].get("memory_used_peak_mib", float("inf"))) <= float(limits["gpu_peak_memory_mib_max"])
        and float(row.get("elapsed_s", float("inf"))) <= float(limits["wall_seconds_max"])
        and int(row.get("disk_used_delta_bytes", limits["per_launch_disk_delta_bytes_max"] + 1))
        <= int(limits["per_launch_disk_delta_bytes_max"])
        and int(row.get("disk_after", {}).get("free_bytes", 0)) >= int(limits["free_gib_min_after"]) * 1024**3
        for row in resources
    )
    cumulative_delta = sum(max(0, int(row["disk_used_delta_bytes"])) for row in resources)
    commit_valid = True
    source_initial_hashes = []
    for row in screens:
        commit = Path(row["commit"])
        try:
            verify_sidecar(commit)
            payload = json.loads(commit.read_text())
            source_hash = payload["source_donor_initial"]["combined_sha256"]
            source_initial_hashes.append(source_hash)
            commit_valid = bool(
                commit_valid
                and sha256(commit) == row["commit_sha256"]
                and payload.get("unknown_mutable_tensor_fields") == []
                and payload.get("source_donor_initial", {}).get("nondegenerate") is True
                and row.get("source_donor_initial") == payload.get("source_donor_initial")
                and len(payload.get("state_manifest", {})) == int(row.get("state_manifest_field_count", -1))
                and set(payload.get("copied_dynamic_fields", []))
                <= set(payload.get("state_manifest", {}))
            )
        except Exception:
            commit_valid = False
    valid = bool(
        prereg.get("schema") == "x2_phase74_pairing_preflight_prereg_v1"
        and seeds == [0, 1, 2]
        and set(labels) == expected_labels
        and all(row.get("decision") == "PASS_INITIAL_PAIRING_LAUNCH" for row in screens)
        and all(row.get("all_pairs_pass") is True for row in screens)
        and all(row.get("unknown_mutable_tensor_fields_empty") is True for row in screens)
        and all(row.get("observation_recomputed_after_clone") is True for row in screens)
        and all(row.get("physx_low_level_readback_present") is True for row in screens)
        and all(row.get("source_donor_initial", {}).get("nondegenerate") is True for row in screens)
        and len(set(source_initial_hashes)) == 3
        and all(row.get("scientific_metrics_present") is False for row in screens)
        and all(int(row.get("physics_rollout_steps", -1)) == 0 for row in screens)
        and all(int(row.get("optimizer_steps", -1)) == 0 for row in screens)
        and all(int(row.get("checkpoint_count", -1)) == 0 for row in screens)
        and all(all(bool(metric.get("passed")) for metric in row.get("diagnostics", {}).values()) for row in screens)
        and commit_valid
        and resource_valid
        and cumulative_delta <= int(limits["cumulative_disk_delta_bytes_max"])
    )
    decision = "PASS_INITIAL_PAIRING_PREFLIGHT_ONLY" if valid else "FAIL_TECHNICAL_STOP"
    payload = {
        "schema": "x2_phase74_pairing_preflight_result_v1",
        "decision": decision,
        "prereg_sha256": sha256(args.prereg),
        "seed_indices": seeds,
        "launch_count": 3,
        "pair_count_total": 192,
        "screens": [{"path": str(path), "sha256": sha256(path)} for path in args.screen],
        "resources": [{"path": str(path), "sha256": sha256(path)} for path in args.resource],
        "scientific_metrics_present": False,
        "physics_rollout_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "source_initial_hashes": source_initial_hashes,
        "source_initial_hashes_distinct": len(set(source_initial_hashes)) == 3,
        "phase75_shadow_preregistration_unlocked": valid,
        "phase75_scientific_preregistration_unlocked": False,
        "phase75_launch_unlocked": False,
        "training_unlocked": False,
        "deployment_unlocked": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    if args.markdown.exists():
        raise FileExistsError(f"refusing to overwrite Phase74 markdown: {args.markdown}")
    args.markdown.write_text(
        "# X2 Phase74 paired-lane technical preflight\n\n"
        f"Decision: `{decision}`.\n\n"
        "Three 128-env, initial-only launches were checked. No reward, posture metric, rollout, gradient, optimizer, checkpoint, or deployment evidence was produced. "
        "A pass only permits preregistration of a one-action shadow technical check; hidden PhysX contact/solver state remains outside this initial-only proof.\n"
    )
    if not valid:
        raise RuntimeError("Phase74 final technical gates failed")


if __name__ == "__main__":
    main()
