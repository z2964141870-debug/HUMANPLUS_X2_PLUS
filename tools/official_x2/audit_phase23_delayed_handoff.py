#!/usr/bin/env python3
"""Aggregate the frozen BASE Phase23 training-free official A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


LABELS = ("fixed", "gated")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_elapsed(trace: list[dict], predicate) -> float | None:
    for row in trace:
        if row.get("stage") == "stop" and predicate(row):
            return float(row["elapsed_s"])
    return None


def read_episode(root: Path, label: str, repeat: int) -> dict:
    path = root / f"phase23_{label}_handoff_f005_stiff1p2_fixed_r{repeat}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    trace = payload["trace"]
    required = ("stand_gate_pass", "startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")
    counts = summary.get("policy_slot_inference_counts", {})
    valid = all(key in summary for key in required) and counts.get("stationary") == 100 and counts.get("main", 0) + counts.get("recovery", 0) == 660
    handoff = summary.get("curriculum_recovery_handoff_gate_handoff_s")
    collapse = first_elapsed(trace, lambda row: float(row.get("root_z_m", 9.0)) < 0.45)
    tilt = first_elapsed(trace, lambda row: float(row.get("root_tilt_rad", 0.0)) > 0.30)
    last_gate = summary.get("curriculum_recovery_handoff_gate_last")
    return {
        "path": str(path),
        "file_sha256": sha256(path),
        "valid": valid,
        "stand": bool(summary.get("stand_gate_pass")),
        "startup": bool(summary.get("startup_gate_pass")),
        "move": bool(summary.get("move_gate_pass")),
        "stop": bool(summary.get("stop_gate_pass")),
        "full": bool(summary.get("full_gate_pass")),
        "slot_counts": counts,
        "gate_enabled": bool(summary.get("curriculum_recovery_handoff_gate_enabled")),
        "gate_wait_s": float(summary.get("curriculum_recovery_handoff_gate_wait_s", 0.0)),
        "handoff_stop_elapsed_s": None if handoff is None else float(handoff),
        "gate_timed_out": bool(summary.get("curriculum_recovery_handoff_gate_timed_out")),
        "handoff_gate": last_gate,
        "first_tilt_gt_0p30_stop_s": tilt,
        "first_root_z_lt_0p45_stop_s": collapse,
        "collapse_latency_after_handoff_s": None if collapse is None or handoff is None else collapse - float(handoff),
        "heading_max_rad": float(summary["move_heading_max_deviation_rad"]),
        "lateral_displacement_m": float(summary["move_lateral_displacement_m"]),
        "stop_drift_m": float(summary["stop_root_xy_drift_m"]),
        "stop_settle_time_s": float(summary["stop_settle_time_s"]),
        "stop_root_z_min_m": float(summary["stop_root_z_min_m"]),
        "stop_root_pitch_mean_rad": float(summary["stop_root_pitch_mean_rad"]),
    }


def median(values: list[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None]
    return None if not clean else statistics.median(clean)


def aggregate(rows: list[dict]) -> dict:
    return {
        "valid": sum(row["valid"] for row in rows),
        "subgates": {key: sum(row[key] for row in rows) for key in ("stand", "startup", "move", "stop", "full")},
        "timeouts": sum(row["gate_timed_out"] for row in rows),
        "median_gate_wait_s": median([row["gate_wait_s"] for row in rows]),
        "median_handoff_stop_elapsed_s": median([row["handoff_stop_elapsed_s"] for row in rows]),
        "median_first_tilt_gt_0p30_stop_s": median([row["first_tilt_gt_0p30_stop_s"] for row in rows]),
        "median_first_root_z_lt_0p45_stop_s": median([row["first_root_z_lt_0p45_stop_s"] for row in rows]),
        "median_collapse_latency_after_handoff_s": median([row["collapse_latency_after_handoff_s"] for row in rows]),
        "continuous_medians": {
            key: median([row[key] for row in rows])
            for key in ("heading_max_rad", "lateral_displacement_m", "stop_drift_m", "stop_settle_time_s", "stop_root_z_min_m", "stop_root_pitch_mean_rad")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for label in LABELS:
        rows = [read_episode(args.result_root, label, repeat) for repeat in range(1, 6)]
        groups[label] = {"rows": rows, "aggregate": aggregate(rows)}
    fixed = groups["fixed"]["aggregate"]
    gated = groups["gated"]["aggregate"]
    result = {
        "stage": "BASE Phase23 training-free state-gated handoff A/B",
        "hypothesis": "Waiting for generator double support and Phase19-v2 previous-action support before recovery authority prevents the common post-handoff collapse.",
        "intervention": "after the same 2.0 s brake, gate recovery handoff on DS AND frozen previous-action support; timeout remains on Stage306 brake",
        "control": "fixed handoff at stop+2.0 s",
        "groups": groups,
        "decision": {
            "all_10_valid": all(group["aggregate"]["valid"] == 5 for group in groups.values()),
            "fixed_gate_off": all(not row["gate_enabled"] for row in groups["fixed"]["rows"]),
            "candidate_gate_on": all(row["gate_enabled"] for row in groups["gated"]["rows"]),
            "candidate_subgates_not_worse": {
                key: gated["subgates"][key] >= fixed["subgates"][key]
                for key in ("stand", "startup", "move", "stop", "full")
            },
            "gate_delays_height_collapse": bool(
                gated["median_first_root_z_lt_0p45_stop_s"] is not None
                and fixed["median_first_root_z_lt_0p45_stop_s"] is not None
                and gated["median_first_root_z_lt_0p45_stop_s"]
                > fixed["median_first_root_z_lt_0p45_stop_s"]
            ),
            "gate_prevents_common_collapse": gated["subgates"]["stop"] > fixed["subgates"]["stop"],
            "previous_action_plus_generator_ds_is_sufficient_falsified": bool(
                gated["subgates"]["stop"] == 0 and gated["subgates"]["full"] == 0
            ),
            "training_unlocked": False,
        },
        "evidence_boundary": "generator contact is a controller phase signal, not measured physical foot contact; relative timing is causal only for this single frozen gate intervention",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
