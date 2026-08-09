#!/usr/bin/env python3
"""Fail-closed validator for one frozen Phase19 trace/v2 sidecar pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_x2.recovery_suffix_aggregation import sha256_file
from official_x2.stop_event_suffix_contract_v2 import (
    SELF_CONSISTENCY_ATOL,
    validate_stop_event_sidecar_v2,
)


EXPECTED_ADAPTER = "6b7c6c353727c75877938372ef2240b8e87e45ad967abb361aacce2e5f66f9d9"
EXPECTED_V2_CONTRACT = "eae96b1e0ebd8b35379df8f2fb8d9b0a45aff4bb376141851af60ea083dbdbdb"
EXPECTED_MOVING = "da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
EXPECTED_STATIONARY = "edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"


def classify_summary(summary: dict) -> str:
    if summary.get("full_gate_pass"):
        return "success"
    if summary.get("survived_stop_height_gate"):
        return "critical"
    return "height_failure"


def validate_pair(trace_path: Path, sidecar_path: Path) -> dict[str, object]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    validate_stop_event_sidecar_v2(sidecar)
    summary = trace["summary"]
    if sidecar["source_trace"]["sha256"] != sha256_file(trace_path):
        raise ValueError("Phase19 source trace SHA mismatch")
    if sidecar["adapter"]["sha256"] != EXPECTED_ADAPTER:
        raise ValueError("Phase19 adapter SHA mismatch")
    if sidecar["model_assets"]["moving"]["sha256"] != EXPECTED_MOVING:
        raise ValueError("Phase19 moving model SHA mismatch")
    if sidecar["model_assets"]["stationary"]["sha256"] != EXPECTED_STATIONARY:
        raise ValueError("Phase19 stationary model SHA mismatch")
    if sidecar["row_count"] != 201:
        raise ValueError("Phase19 v2 sidecar must contain 201 rows")
    if abs(float(sidecar["rows"][0]["stop_elapsed_s"])) > 1.0e-9:
        raise ValueError("Phase19 sidecar does not start at stop event")
    if abs(float(sidecar["rows"][-1]["stop_elapsed_s"]) - 4.0) > 1.0e-9:
        raise ValueError("Phase19 sidecar does not end at 4.0 seconds")
    errors = [
        max(
            float(row["self_consistency"]["reconstruction_abs_max"]),
            float(row["self_consistency"]["model_prefix_abs_max"]),
        )
        for row in sidecar["rows"]
    ]
    max_error = max(errors)
    if max_error > SELF_CONSISTENCY_ATOL:
        raise ValueError(f"Phase19 actor-input self-consistency failed: {max_error}")
    for key in (
        "full_gate_pass", "stand_gate_pass", "startup_gate_pass",
        "move_gate_pass", "stop_gate_pass", "survived_stop_height_gate",
    ):
        if bool(sidecar["episode_outcome"].get(key)) != bool(summary.get(key)):
            raise ValueError(f"Phase19 trace/sidecar outcome mismatch: {key}")
    return {
        "classification": classify_summary(summary),
        "row_count": int(sidecar["row_count"]),
        "self_consistency_abs_max": max_error,
        "sidecar_content_sha256": sidecar["content_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_pair(args.trace, args.sidecar), sort_keys=True))


if __name__ == "__main__":
    main()
