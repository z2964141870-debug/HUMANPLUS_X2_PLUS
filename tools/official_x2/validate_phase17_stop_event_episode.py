#!/usr/bin/env python3
"""Fail-closed validator for one frozen Phase17 trace/sidecar pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_x2.recovery_suffix_aggregation import sha256_file
from official_x2.stop_event_suffix_contract import validate_stop_event_sidecar


EXPECTED_ADAPTER = "61c73512981775c3f7f08fe84e8e6d58e6a76fb7d11ccfb58e4f6665a20d1077"
EXPECTED_MOVING = "da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
EXPECTED_STATIONARY = "edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"
OUTCOME_KEYS = (
    "full_gate_pass", "stand_gate_pass", "startup_gate_pass",
    "move_gate_pass", "stop_gate_pass", "survived_stop_height_gate",
)


def classify_summary(summary: dict) -> str:
    """Fail closed: only the complete gate is an inventory success."""
    if summary.get("full_gate_pass"):
        return "success"
    if summary.get("survived_stop_height_gate"):
        return "critical"
    return "failure"


def validate_pair(trace_path: Path, sidecar_path: Path) -> str:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    validate_stop_event_sidecar(sidecar)
    summary = trace["summary"]
    if sidecar["source_trace"]["sha256"] != sha256_file(trace_path):
        raise ValueError("Phase17 source trace SHA mismatch")
    if sidecar["adapter"]["sha256"] != EXPECTED_ADAPTER:
        raise ValueError("Phase17 adapter SHA mismatch")
    if sidecar["model_assets"]["moving"]["sha256"] != EXPECTED_MOVING:
        raise ValueError("Phase17 moving model SHA mismatch")
    if sidecar["model_assets"]["stationary"]["sha256"] != EXPECTED_STATIONARY:
        raise ValueError("Phase17 stationary model SHA mismatch")
    if sidecar["row_count"] != 201:
        raise ValueError("Phase17 stop-event sidecar must contain 201 rows")
    if abs(float(sidecar["rows"][0]["stop_elapsed_s"])) > 1.0e-9:
        raise ValueError("Phase17 sidecar does not start at stop event")
    if abs(float(sidecar["rows"][-1]["stop_elapsed_s"]) - 4.0) > 1.0e-9:
        raise ValueError("Phase17 sidecar does not end at 4.0 seconds")
    for key in OUTCOME_KEYS:
        if bool(sidecar["episode_outcome"][key]) != bool(summary.get(key)):
            raise ValueError(f"Phase17 trace/sidecar outcome mismatch: {key}")
    return classify_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args()
    print(validate_pair(args.trace, args.sidecar))


if __name__ == "__main__":
    main()
