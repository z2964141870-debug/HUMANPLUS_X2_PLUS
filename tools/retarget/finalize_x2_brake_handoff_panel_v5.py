#!/usr/bin/env python3
"""Independently recompute the X2 brake/handoff panel decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from cwi_x2.brake_handoff_panel import (
    STRATEGIES, handoff_gates, strategy_by_id, strategy_key, summarize_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{sha256(path)}  {path.name}\n":
        raise RuntimeError(f"sidecar mismatch: {path}")


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_text(path, text)
    sidecar = path.with_name(path.name + ".sha256")
    atomic_text(sidecar, f"{sha256(path)}  {path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.result, args.result.with_name(args.result.name + ".sha256"),
                 args.markdown, args.markdown.with_name(args.markdown.name + ".sha256")):
        if path.exists():
            raise FileExistsError(path)
    for path in (args.prereg, args.screen, args.resource):
        verify_sidecar(path)
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    resource = json.loads(args.resource.read_text(encoding="utf-8"))
    if screen["prereg_sha256"] != sha256(args.prereg):
        raise RuntimeError("screen does not bind preregistration")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"immutable record drift: {section}.{name}")
    if [strategy.name for strategy in STRATEGIES] != prereg["strategy_names"]:
        raise RuntimeError("strategy inventory drift")

    panel_rows = screen["screening"]["per_env"]
    panel_summaries = {
        str(index): summarize_rows(panel_rows, index) for index in range(len(STRATEGIES))
    }
    ordered = sorted(range(len(STRATEGIES)), key=lambda index: strategy_key(panel_summaries[str(index)]))
    top_ids = ordered[: prereg["screening"]["validation_candidates"]]
    expected_validation_ids = list(dict.fromkeys((0, 1, *top_ids)))
    if screen["screening"]["ordered_strategy_ids"] != ordered:
        raise RuntimeError("screening order mismatch")
    if screen["screening"]["top_validation_ids"] != top_ids:
        raise RuntimeError("screening top set mismatch")
    if screen["validation"]["strategy_ids"] != expected_validation_ids:
        raise RuntimeError("validation strategy set mismatch")
    if screen["screening"]["summaries"] != panel_summaries:
        raise RuntimeError("screening summary mismatch")

    validation_summaries = {}
    validation_rows = screen["validation"]["per_env"]
    for strategy_id in expected_validation_ids:
        name = strategy_by_id(strategy_id).name
        rows = validation_rows[name]
        if len(rows) != prereg["runtime"]["num_envs"] * 3:
            raise RuntimeError(f"incomplete validation rows: {name}")
        validation_summaries[name] = summarize_rows(rows, strategy_id)
    if screen["validation"]["summaries"] != validation_summaries:
        raise RuntimeError("validation summary mismatch")
    best_id = min(
        expected_validation_ids,
        key=lambda index: strategy_key(validation_summaries[strategy_by_id(index).name]),
    )
    best_name = strategy_by_id(best_id).name
    gates = handoff_gates(validation_summaries["direct_mix"], validation_summaries[best_name])
    valid = all(bool(value) for value in screen["technical_checks"].values())
    resource_checks = {
        "child_exit_zero": resource.get("exit_code") == 0 and resource.get("raw_returncode") == 0,
        "autonomous_exit": resource.get("autonomous_exit") is True,
        "no_timeout_or_signal": not any(resource.get(name, False) for name in (
            "timed_out", "term_sent", "kill_sent", "forced_cleanup"
        )),
        "elapsed": float(resource["elapsed_s"]) <= prereg["resources"]["elapsed_max_s"],
        "gpu": resource.get("gpu") is not None
               and float(resource["gpu"]["memory_used_peak_mib"]) <= prereg["resources"]["gpu_peak_mib_max"],
        "free_space": int(resource["disk_after"]["free_bytes"]) >= prereg["resources"]["free_after_bytes_min"],
        "evidence_bytes": args.screen.stat().st_size + args.resource.stat().st_size <= prereg["resources"]["evidence_bytes_max"],
    }
    valid = valid and all(resource_checks.values())
    if valid and all(gates.values()):
        decision = "PASS_HANDOFF_CONTROLLER_OFFICIAL_PANEL_PREREG_ONLY"
    else:
        source_hold_terms = validation_summaries["direct_mix"]["hold"]["terminations"]
        best = validation_summaries[best_name]
        partial = (
            valid
            and best["cruise"]["terminations"] + best["decelerate"]["terminations"] == 0
            and best["hold"]["terminations"] <= 0.5 * source_hold_terms
            and sum(best[s]["timeouts"] for s in ("cruise", "decelerate", "hold")) == 0
        )
        decision = (
            "PARTIAL_HANDOFF_BRAKE_SKILL_PREREG_ONLY"
            if partial else "FAIL_HANDOFF_NEW_ACTOR_PREREG_ONLY"
        )
    if screen["validation"]["best_strategy_id"] != best_id or screen["validation"]["best_strategy"] != best_name:
        raise RuntimeError("best validation strategy mismatch")
    if screen["validation"]["gates"] != gates or screen["decision"] != decision:
        raise RuntimeError("screen decision mismatch")
    if screen["permissions"] != prereg["decision_routes"][decision]:
        raise RuntimeError("permission route mismatch")

    result = {
        "schema": "x2_brake_handoff_panel_result_v1",
        "decision": decision,
        "prereg_sha256": sha256(args.prereg),
        "screen_sha256": sha256(args.screen),
        "resource_sha256": sha256(args.resource),
        "best_strategy_id": best_id,
        "best_strategy": best_name,
        "validation_summaries": validation_summaries,
        "gates": gates,
        "technical_checks": screen["technical_checks"],
        "resource_checks": resource_checks,
        "global_filesystem_delta_bytes_informational_only": resource["disk_used_delta_bytes"],
        "permissions": prereg["decision_routes"][decision],
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
        "baidu_access_or_upload": False,
    }
    atomic_json(args.result, result)
    direct = validation_summaries["direct_mix"]
    best = validation_summaries[best_name]
    markdown = (
        "# X2 brake/handoff panel v5\n\n"
        f"Decision: `{decision}`. Best strategy: `{best_name}`.\n\n"
        f"Direct handoff terminations: cruise {direct['cruise']['terminations']}, "
        f"decelerate {direct['decelerate']['terminations']}, hold {direct['hold']['terminations']}. "
        f"Best strategy: {best['cruise']['terminations']}, {best['decelerate']['terminations']}, "
        f"{best['hold']['terminations']}. Hold speed p95={best['hold']['speed_p95']:.6f} m/s, "
        f"tilt max={best['hold']['tilt']:.6f} rad.\n\n"
        "This was a zero-training controller-feasibility panel. It does not unlock training or deployment. "
        "The filesystem-wide delta is retained only as provenance because unrelated local archive writes can occur concurrently.\n"
    )
    atomic_text(args.markdown, markdown)
    atomic_text(args.markdown.with_name(args.markdown.name + ".sha256"),
                f"{sha256(args.markdown)}  {args.markdown.name}\n")
    print(json.dumps({"decision": decision, "best_strategy": best_name,
                      "resource_checks": resource_checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
