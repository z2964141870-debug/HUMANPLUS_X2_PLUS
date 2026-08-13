#!/usr/bin/env python3
"""Independently recompute the three-seed locomotion-zero confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from cwi_x2.locomotion_zero_confirmation import (
    SEGMENTS,
    TREATMENTS,
    aggregate_decision,
    checkerboard_assignment,
    seed_gates,
    summarize_treatment,
    validate_assignment,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"sidecar mismatch: {path}")


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_text(path.with_name(path.name + ".sha256"), f"{sha256(path)}  {path.name}\n")


def resource_checks(resource: dict, screen: Path, prereg: dict, expected_label: str) -> dict[str, bool]:
    limits = prereg["resources"]
    return {
        "schema": resource.get("schema") == "x2_gpu_deadline_ledger_phase76_v1",
        "label": resource.get("label") == expected_label,
        "child_exit_zero": resource.get("exit_code") == 0 and resource.get("raw_returncode") == 0,
        "autonomous_exit": resource.get("autonomous_exit") is True,
        "no_timeout_or_signal": not any(resource.get(name, False) for name in (
            "timed_out", "term_sent", "kill_sent", "forced_cleanup"
        )),
        "elapsed": float(resource["elapsed_s"]) <= limits["elapsed_max_s"],
        "gpu": resource.get("gpu") is not None
        and float(resource["gpu"]["memory_used_peak_mib"]) <= limits["gpu_peak_mib_max"],
        "free_space": int(resource["disk_after"]["free_bytes"]) >= limits["free_after_bytes_min"],
        "evidence_bytes": screen.stat().st_size <= limits["screen_bytes_max"],
    }


def validate_rows(rows: list[dict], assignments: list[str]) -> None:
    if len(rows) != 256 * len(SEGMENTS):
        raise RuntimeError("incomplete per-environment rows")
    observed = set()
    for row in rows:
        env_id = int(row["env_id"])
        key = (env_id, row["segment"])
        if env_id not in range(256) or row["segment"] not in SEGMENTS or key in observed:
            raise RuntimeError("invalid or duplicate per-environment row")
        if row["treatment"] != assignments[env_id]:
            raise RuntimeError("treatment assignment mismatch")
        observed.add(key)
    expected = {(env_id, segment) for env_id in range(256) for segment in SEGMENTS}
    if observed != expected:
        raise RuntimeError("per-environment row coverage mismatch")


def command_value(command: list[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise RuntimeError(f"resource command flag mismatch: {flag}")
    return command[indices[0] + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--resource", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if len(args.screen) != 3 or len(args.resource) != 3:
        raise RuntimeError("exactly three screens and three resource reports are required")
    for path in (args.result, args.result.with_name(args.result.name + ".sha256"),
                 args.markdown, args.markdown.with_name(args.markdown.name + ".sha256")):
        if path.exists():
            raise FileExistsError(path)
    for path in (args.prereg, *args.screen, *args.resource):
        verify_sidecar(path)

    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    prereg_sha = sha256(args.prereg)
    if prereg.get("schema") != "x2_locomotion_zero_hold_confirmation_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"immutable record drift: {section}.{name}")

    seed_records = []
    resource_results = []
    for seed_index, (screen_path, resource_path) in enumerate(zip(args.screen, args.resource)):
        expected_run = prereg["runs"][seed_index]
        if screen_path.resolve() != Path(expected_run["screen"]).resolve():
            raise RuntimeError("screen order/path mismatch")
        if resource_path.resolve() != Path(expected_run["resource"]).resolve():
            raise RuntimeError("resource order/path mismatch")
        screen = json.loads(screen_path.read_text(encoding="utf-8"))
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        log_path = Path(expected_run["log"])
        verify_sidecar(log_path)
        if Path(resource["log_path"]).resolve() != log_path.resolve():
            raise RuntimeError("resource log path mismatch")
        if resource["log_sha256"] != sha256(log_path):
            raise RuntimeError("resource log hash mismatch")
        command = resource.get("command", [])
        runner_path = Path(prereg["immutable_code"]["runner"]["path"])
        if not any(Path(value).resolve() == runner_path.resolve() for value in command if value.endswith(".py")):
            raise RuntimeError("resource command runner mismatch")
        expected_command_values = {
            "--prereg": str(args.prereg.resolve()),
            "--source": str(Path(prereg["immutable_inputs"]["source_checkpoint"]["path"]).resolve()),
            "--stationary": str(Path(prereg["immutable_inputs"]["stationary_actor"]["path"]).resolve()),
            "--report": str(screen_path.resolve()),
            "--failure": str(Path(expected_run["failure"]).resolve()),
            "--seed-index": str(seed_index),
            "--seed": str(expected_run["seed"]),
            "--num-envs": "256",
            "--steps": "820",
        }
        for flag, expected in expected_command_values.items():
            actual = command_value(command, flag)
            if flag in ("--prereg", "--source", "--stationary", "--report", "--failure"):
                actual = str(Path(actual).resolve())
            if actual != expected:
                raise RuntimeError(f"resource command value mismatch: {flag}")
        if screen.get("schema") != "x2_locomotion_zero_hold_confirmation_segment_v1":
            raise RuntimeError("unexpected screen schema")
        if screen.get("prereg_sha256") != prereg_sha:
            raise RuntimeError("screen does not bind preregistration")
        if screen.get("seed_index") != seed_index or screen.get("seed") != expected_run["seed"]:
            raise RuntimeError("screen seed mismatch")
        if screen.get("segment_status") != "SEGMENT_VALID":
            raise RuntimeError("invalid segment status")
        if not all(bool(value) for value in screen["technical_checks"].values()):
            raise RuntimeError("runner technical check failed")
        assignments = checkerboard_assignment(256, seed_index)
        if not validate_assignment(assignments, seed_index):
            raise RuntimeError("finalizer assignment construction failed")
        expected_assignment_sha = hashlib.sha256("\n".join(assignments).encode()).hexdigest()
        if screen["assignment"] != {
            "method": "checkerboard", "direct_count": 128, "candidate_count": 128,
            "sha256": expected_assignment_sha,
        }:
            raise RuntimeError("assignment evidence mismatch")
        rows = screen["per_env"]
        validate_rows(rows, assignments)
        summaries = {name: summarize_treatment(rows, name) for name in TREATMENTS}
        gates = seed_gates(summaries["direct_mix"], summaries["locomotion_zero"])
        if screen["summaries"] != summaries or screen["gates"] != gates:
            raise RuntimeError("screen summary/gate reconstruction mismatch")
        if any(screen.get(name) != 0 for name in (
            "optimizer_steps", "backward_calls", "checkpoint_writes"
        )) or screen.get("reward_inspected") is not False:
            raise RuntimeError("zero-training/evaluation boundary violated")
        checks = resource_checks(resource, screen_path, prereg, expected_run["label"])
        if not all(checks.values()):
            raise RuntimeError(f"resource check failed for seed {seed_index}: {checks}")
        seed_records.append({
            "seed_index": seed_index,
            "seed": expected_run["seed"],
            "screen_sha256": sha256(screen_path),
            "resource_sha256": sha256(resource_path),
            "summaries": summaries,
            "gates": gates,
            "initial_fingerprint": screen["initial_fingerprint"],
        })
        resource_results.append(checks)

    if len({record["initial_fingerprint"]["policy"] for record in seed_records}) != 3:
        raise RuntimeError("fresh reset seeds did not produce distinct initial observations")
    decision, pooled = aggregate_decision(seed_records)
    permissions = prereg["decision_routes"][decision]
    result = {
        "schema": "x2_locomotion_zero_hold_confirmation_result_v1",
        "decision": decision,
        "prereg_sha256": prereg_sha,
        "seed_records": seed_records,
        "pooled_gates": pooled,
        "resource_checks": resource_results,
        "permissions": permissions,
        "independent_train_or_eval_seed_count": 3,
        "environment_lanes_are_not_independent_seeds": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
        "baidu_access_or_upload": False,
    }
    atomic_json(args.result, result)
    candidate_terms = sum(
        row["summaries"]["locomotion_zero"]["hold"]["terminations"] for row in seed_records
    )
    direct_terms = sum(
        row["summaries"]["direct_mix"]["hold"]["terminations"] for row in seed_records
    )
    reaches = [row["summaries"]["locomotion_zero"]["hold"]["reach_fraction"] for row in seed_records]
    markdown = (
        "# X2 locomotion-zero hold confirmation\n\n"
        f"Decision: `{decision}`. Across three fresh reset seeds, direct-mix hold terminations="
        f"{direct_terms}; locomotion-zero hold terminations={candidate_terms}. Candidate hold reach "
        f"fractions={', '.join(f'{value:.6f}' for value in reaches)}.\n\n"
        "This was a zero-training balanced-lane confirmation. Environment lanes are balancing units, "
        "not independent experimental seeds. Passing can only unlock writing an official-panel "
        "preregistration; it does not unlock that launch, training, export, or deployment.\n"
    )
    atomic_text(args.markdown, markdown)
    atomic_text(args.markdown.with_name(args.markdown.name + ".sha256"),
                f"{sha256(args.markdown)}  {args.markdown.name}\n")
    print(json.dumps({"decision": decision, "pooled_gates": pooled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
