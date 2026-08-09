#!/usr/bin/env python3
"""Aggregate Phase24 against immutable Phase23 fixed/action-gated evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_stop_event(trace: list[dict], key: str, threshold: float, *, below: bool) -> float | None:
    for row in trace:
        if row.get("stage") != "stop":
            continue
        value = float(row.get(key, 9.0 if below else 0.0))
        if (value < threshold) if below else (value > threshold):
            return float(row["elapsed_s"])
    return None


def read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    s, trace = payload["summary"], payload["trace"]
    handoff = s.get("curriculum_recovery_handoff_gate_handoff_s")
    collapse = first_stop_event(trace, "root_z_m", 0.45, below=True)
    tilt = first_stop_event(trace, "root_tilt_rad", 0.30, below=False)
    counts = s.get("policy_slot_inference_counts", {})
    return {
        "path": str(path), "file_sha256": sha256(path),
        "valid": all(k in s for k in ("stand_gate_pass", "startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")),
        "stand": bool(s.get("stand_gate_pass")), "startup": bool(s.get("startup_gate_pass")),
        "move": bool(s.get("move_gate_pass")), "stop": bool(s.get("stop_gate_pass")), "full": bool(s.get("full_gate_pass")),
        "slot_counts": counts,
        "recovery_called": int(counts.get("recovery", 0)) > 0,
        "handoff_stop_elapsed_s": handoff,
        "gate_wait_s": float(s.get("curriculum_recovery_handoff_gate_wait_s", 0.0)),
        "gate_timed_out": bool(s.get("curriculum_recovery_handoff_gate_timed_out")),
        "last_gate": s.get("curriculum_recovery_handoff_gate_last"),
        "first_tilt_gt_0p30_stop_s": tilt,
        "first_root_z_lt_0p45_stop_s": collapse,
        "collapse_latency_after_handoff_s": (
            None if collapse is None or handoff is None else float(collapse) - float(handoff)
        ),
        "event_order": "no_collapse" if collapse is None else ("tilt_then_height" if tilt is not None and tilt <= collapse else "height_without_prior_tilt"),
        "heading_max_rad": float(s["move_heading_max_deviation_rad"]),
        "lateral_displacement_m": float(s["move_lateral_displacement_m"]),
        "stop_drift_m": float(s["stop_root_xy_drift_m"]),
        "stop_settle_time_s": float(s["stop_settle_time_s"]),
        "stop_root_z_min_m": float(s["stop_root_z_min_m"]),
    }


def median(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else statistics.median(values)


def aggregate(rows: list[dict]) -> dict:
    return {
        "valid": sum(row["valid"] for row in rows),
        "subgates": {key: sum(row[key] for row in rows) for key in ("stand", "startup", "move", "stop", "full")},
        "handoffs": sum(row["recovery_called"] for row in rows),
        "timeouts": sum(row["gate_timed_out"] for row in rows),
        "event_order_counts": {name: sum(row["event_order"] == name for row in rows) for name in ("tilt_then_height", "height_without_prior_tilt", "no_collapse")},
        "medians": {key: median(rows, key) for key in (
            "gate_wait_s", "handoff_stop_elapsed_s", "first_tilt_gt_0p30_stop_s", "first_root_z_lt_0p45_stop_s",
            "collapse_latency_after_handoff_s",
            "heading_max_rad", "lateral_displacement_m", "stop_drift_m", "stop_settle_time_s", "stop_root_z_min_m",
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patterns = {
        "phase23_fixed": "phase23_fixed_handoff_f005_stiff1p2_fixed_r{}.json",
        "phase23_action_gated": "phase23_gated_handoff_f005_stiff1p2_fixed_r{}.json",
        "phase24_physical_gated": "phase24_physical_gated_handoff_f005_stiff1p2_fixed_r{}.json",
    }
    groups = {}
    for label, pattern in patterns.items():
        rows = [read(args.result_root / pattern.format(i)) for i in range(1, 6)]
        groups[label] = {"rows": rows, "aggregate": aggregate(rows)}
    candidate = groups["phase24_physical_gated"]["aggregate"]
    result = {
        "stage": "BASE Phase24 training-free full physical-observation support gate",
        "hypothesis": "Phase23 still collapsed because action-history and generator DS omitted projected-gravity/base-angular-velocity support.",
        "intervention": "Phase23 gate AND independently frozen Phase19-v2 robust support for projected gravity and base angular velocity; timeout never forces handoff",
        "controls_reused": "immutable Phase23 5 fixed + 5 action-gated episodes; default-off pure tests isolate the added gate, so no extra physical control episodes were run",
        "groups": groups,
        "decision": {
            "all_candidate_valid": candidate["valid"] == 5,
            "candidate_handoff_count": candidate["handoffs"],
            "candidate_timeout_count": candidate["timeouts"],
            "candidate_stop_pass_count": candidate["subgates"]["stop"],
            "candidate_full_pass_count": candidate["subgates"]["full"],
            "physical_support_is_sufficient": candidate["subgates"]["stop"] > 0 and candidate["handoffs"] > 0,
            "no_handoff_survival_is_not_recovery_success": True,
            "training_unlocked": False,
        },
        "evidence_boundary": "generator DS is not measured contact; a timeout episode tests reachability of the support set under frozen Stage306 brake, not recovery actor quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
