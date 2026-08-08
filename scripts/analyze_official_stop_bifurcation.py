#!/usr/bin/env python3
"""Compare a passing and failing official-X2 stop rollout.

The official loop records one row per 20 ms control tick.  This utility keeps
the comparison in the measured state/action domain and reports when two runs
first cease to be practically identical, together with the state at the stop
handoff and the configured event latch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


STATE_KEYS = (
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "root_tilt_rad",
    "root_yaw_rad",
    "root_vx_b_mps",
    "root_vy_b_mps",
    "root_yaw_rate_radps",
)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload.get("summary"), dict) or not isinstance(payload.get("trace"), list):
        raise ValueError(f"{path} is not an official gate result")
    return payload


def vector(row: dict) -> np.ndarray:
    return np.asarray(
        [0.0 if row[key] is None else float(row[key]) for key in STATE_KEYS],
        dtype=np.float64,
    )


def action_delta(left: dict, right: dict) -> float:
    if left["action"] is None or right["action"] is None:
        return 0.0 if left["action"] is right["action"] else math.inf
    if not left["action"] or not right["action"]:
        return 0.0 if left["action"] == right["action"] else math.inf
    return float(
        np.max(
            np.abs(
                np.asarray(left["action"], dtype=np.float64)
                - np.asarray(right["action"], dtype=np.float64)
            )
        )
    )


def first_index(values: list[float], threshold: float) -> int | None:
    return next((index for index, value in enumerate(values) if value > threshold), None)


def row_digest(row: dict) -> dict:
    return {
        "stage": row["stage"],
        "elapsed_s": float(row["elapsed_s"]),
        **{
            key: None if row[key] is None else float(row[key])
            for key in STATE_KEYS
        },
    }


def stop_row(trace: list[dict], offset_s: float) -> dict:
    stop = [row for row in trace if row["stage"] == "stop"]
    if not stop:
        raise ValueError("trace has no stop stage")
    target = float(stop[0]["elapsed_s"]) + offset_s
    return min(stop, key=lambda row: abs(float(row["elapsed_s"]) - target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("passing", type=Path)
    parser.add_argument("failing", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-threshold", type=float, default=1.0e-3)
    parser.add_argument("--action-threshold", type=float, default=1.0e-3)
    args = parser.parse_args()

    passing = load(args.passing)
    failing = load(args.failing)
    left = passing["trace"]
    right = failing["trace"]
    if len(left) != len(right):
        raise ValueError(f"trace lengths differ: {len(left)} != {len(right)}")

    state_deltas = [float(np.max(np.abs(vector(a) - vector(b)))) for a, b in zip(left, right)]
    action_deltas = [action_delta(a, b) for a, b in zip(left, right)]
    first_state = first_index(state_deltas, args.state_threshold)
    first_action = first_index(action_deltas, args.action_threshold)

    pass_latch = float(passing["summary"].get("stop_hold_latch_s") or 0.0)
    fail_latch = float(failing["summary"].get("stop_hold_latch_s") or 0.0)
    checkpoints = {
        "stop_entry": 0.0,
        "event_latch": max(pass_latch, fail_latch),
        "post_latch_0p5s": max(pass_latch, fail_latch) + 0.5,
        "post_latch_1p0s": max(pass_latch, fail_latch) + 1.0,
    }

    result = {
        "passing": str(args.passing),
        "failing": str(args.failing),
        "pass_gate": bool(passing["summary"].get("full_gate_pass")),
        "fail_gate": bool(failing["summary"].get("full_gate_pass")),
        "trace_rows": len(left),
        "thresholds": {
            "state_max_abs": args.state_threshold,
            "action_max_abs": args.action_threshold,
        },
        "first_state_divergence": None
        if first_state is None
        else {
            "index": first_state,
            "max_abs": state_deltas[first_state],
            "passing": row_digest(left[first_state]),
            "failing": row_digest(right[first_state]),
        },
        "first_action_divergence": None
        if first_action is None
        else {
            "index": first_action,
            "max_abs": action_deltas[first_action],
            "passing": row_digest(left[first_action]),
            "failing": row_digest(right[first_action]),
        },
        "stop_hold_latch_s": {"passing": pass_latch, "failing": fail_latch},
        "checkpoints": {
            name: {
                "offset_s": offset,
                "passing": row_digest(stop_row(left, offset)),
                "failing": row_digest(stop_row(right, offset)),
                "state_max_abs": float(
                    np.max(
                        np.abs(vector(stop_row(left, offset)) - vector(stop_row(right, offset)))
                    )
                ),
                "action_max_abs": action_delta(stop_row(left, offset), stop_row(right, offset)),
            }
            for name, offset in checkpoints.items()
        },
        "outcomes": {
            "passing_stop_root_z_min_m": float(passing["summary"]["stop_root_z_min_m"]),
            "failing_stop_root_z_min_m": float(failing["summary"]["stop_root_z_min_m"]),
            "passing_stop_tilt_max_rad": float(passing["summary"]["stop_root_tilt_max_rad"]),
            "failing_stop_tilt_max_rad": float(failing["summary"]["stop_root_tilt_max_rad"]),
        },
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
