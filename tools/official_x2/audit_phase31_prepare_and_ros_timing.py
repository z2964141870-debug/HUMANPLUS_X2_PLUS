#!/usr/bin/env python3
"""BASE Phase31 read-only prepare-history and closed-ROS timing audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
OFFICIAL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
DEFAULT_HISTORICAL = OFFICIAL / "results/official_native_strict_20260807/stage250_video_straight.json"
DEFAULT_EVENT20 = OFFICIAL / "results/official_native_event_v2_20260809/official_native_event_v2_20s.npz"
DEFAULT_RESET = OFFICIAL / "results/official_native_reset_prefix_v2_20260809/official_native_event_v2_reset_prefix_24s.npz"
DEFAULT_ACTIVATION = OFFICIAL / "results/official_native_activation_prefix_phase21_20260809/official_native_event_v2_activation_prefix_phase21.npz"
DEFAULT_ADAPTER = ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py"
DEFAULT_OUTPUT = ROOT / "reports/official_x2/phase31_prepare_and_ros_timing_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats_ns(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": int(values.size),
        "p50_ms": float(np.quantile(values, 0.50) * 1e-6),
        "p95_ms": float(np.quantile(values, 0.95) * 1e-6),
        "max_ms": float(np.max(values) * 1e-6),
    }


def analyze_prepare(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload["trace"] if row.get("stage") == "prepare"]
    if not rows:
        raise RuntimeError("historical trace has no prepare rows")
    required = {
        "joint_q": lambda row: bool(row.get("joint_q_rad")),
        "joint_dq": lambda row: bool(row.get("joint_dq_radps")),
        "command_or_target": lambda row: bool(row.get("physical_lower_target_rad")) or bool(row.get("command_q_rad")),
        "ctrl_or_force": lambda row: bool(row.get("ctrl")) or bool(row.get("actuator_force")),
        "contact": lambda row: bool(row.get("contact")) or bool(row.get("contacts")),
        "obs93": lambda row: len(row.get("obs", [])) == 93,
        "action15": lambda row: len(row.get("action", [])) == 15,
    }
    presence = {key: sum(bool(test(row)) for row in rows) for key, test in required.items()}
    dt = np.asarray([row["control_wall_dt_s"] for row in rows if row.get("control_wall_dt_s") is not None])
    skew = np.asarray([row["source_meas_skew_s"] for row in rows if row.get("source_meas_skew_s") is not None])
    age = np.asarray([row["source_callback_age_max_s"] for row in rows if row.get("source_callback_age_max_s") is not None])
    return {
        "rows": len(rows),
        "elapsed_s": [float(row["elapsed_s"]) for row in rows],
        "field_presence_rows": presence,
        "root_only_dynamics_available": all(
            all(key in row for key in ("root_x_m", "root_y_m", "root_z_m", "root_tilt_rad", "root_vx_w_mps", "root_vy_w_mps"))
            for row in rows
        ),
        "timing": {
            "control_wall_dt": stats_ns(dt * 1e9),
            "source_measurement_skew": stats_ns(skew * 1e9),
            "source_callback_age_max": stats_ns(age * 1e9),
        },
        "known_current_code_formula": (
            "alpha=smoothstep(elapsed/prepare_seconds); target=start_q+alpha*(default-start_q); "
            "zero velocity/effort; configured Kp/Kd"
        ),
        "missing_formula_inputs": [
            "prepare_start_q for all 31 joints",
            "per-tick actual q/dq",
            "published per-group command receipt/application substep",
            "per-tick ctrl/actuator force",
            "contact/constraint/qacc_warmstart state",
            "cryptographic binding of the current adapter source to the historical capture",
        ],
        "uniquely_reconstructable": False,
        "reason": (
            "The interpolation formula is visible in current source, but its captured start_q and every physical/command "
            "state needed to reproduce the 0.2 s integration history were not recorded."
        ),
    }


def _latest_before(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(reference, query, side="right") - 1
    valid = indices >= 0
    return indices, valid


def _first_after(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(reference, query, side="left")
    valid = indices < reference.size
    return indices, valid


def analyze_event_v2(path: Path) -> dict[str, Any]:
    archive = np.load(path, allow_pickle=False)
    snapshots = np.asarray(archive["snapshot_receipt_monotonic_ns"], dtype=np.int64)
    group_order = [str(value) for value in archive["snapshot_state_group_order"].tolist()]
    state_receipts = np.asarray(archive["snapshot_state_group_receipt_monotonic_ns"], dtype=np.int64)
    command_receipts = np.asarray(archive["command_event_receipt_monotonic_ns"], dtype=np.int64)
    command_groups = np.asarray(archive["command_event_group"])

    state_age = {
        group: stats_ns(snapshots - state_receipts[:, index])
        for index, group in enumerate(group_order)
    }
    command_interval, state_command_bracket = {}, {}
    group_command_times: dict[str, np.ndarray] = {}
    for index, group in enumerate(group_order):
        times = np.sort(command_receipts[command_groups == group])
        group_command_times[group] = times
        command_interval[group] = stats_ns(np.diff(times))
        state_time = state_receipts[:, index]
        previous_index, has_previous = _latest_before(times, state_time)
        next_index, has_next = _first_after(times, state_time)
        previous_gap = state_time[has_previous] - times[previous_index[has_previous]]
        next_gap = times[next_index[has_next]] - state_time[has_next]
        state_command_bracket[group] = {
            "state_after_previous_recorder_command_receipt": stats_ns(previous_gap),
            "next_recorder_command_receipt_after_state": stats_ns(next_gap),
        }

    common = min(len(values) for values in group_command_times.values())
    aligned_command_times = np.stack([group_command_times[group][:common] for group in group_order], axis=1)
    command_group_skew = aligned_command_times.max(axis=1) - aligned_command_times.min(axis=1)
    state_group_skew = state_receipts.max(axis=1) - state_receipts.min(axis=1)
    header_populated = np.asarray(archive["command_event_header_populated"], dtype=bool)

    duration = float((snapshots[-1] - snapshots[0]) * 1e-9) if snapshots.size > 1 else 0.0
    return {
        "path": str(path),
        "sha256": sha256(path),
        "schema_version": str(archive["schema_version"].item()),
        "source": str(archive["source"].item()),
        "duration_s": duration,
        "snapshots": int(snapshots.size),
        "snapshot_interval": stats_ns(np.diff(snapshots)),
        "state_group_age_at_snapshot": state_age,
        "state_group_receipt_skew": stats_ns(state_group_skew),
        "imu_age_at_snapshot": stats_ns(snapshots - np.asarray(archive["snapshot_imu_receipt_monotonic_ns"])),
        "odom_age_at_snapshot": stats_ns(snapshots - np.asarray(archive["snapshot_odom_receipt_monotonic_ns"])),
        "command_events": int(command_receipts.size),
        "command_header_populated_fraction": float(np.mean(header_populated)) if header_populated.size else None,
        "command_interval_by_group": command_interval,
        "same_group_index_command_receipt_skew": stats_ns(command_group_skew),
        "state_receipt_bracketed_by_recorder_command_receipts": state_command_bracket,
        "semantic_boundary": (
            "Times are receipt timestamps in an independent recorder subscriber, not timestamps at which the closed "
            "simulator applied a command or integrated a physics substep."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    prepare = analyze_prepare(historical)
    events = {
        "official_native_20s": analyze_event_v2(args.event20),
        "official_reset_prefix": analyze_event_v2(args.reset),
        "official_activation_prefix": analyze_event_v2(args.activation),
    }
    return {
        "stage": "BASE Phase31",
        "scope": "read-only; no physics probe, training, WBT, Git, cloud, or hardware",
        "assets": {
            "stage250_historical": {"path": str(args.historical), "sha256": sha256(args.historical)},
            "current_adapter": {"path": str(args.adapter), "sha256": sha256(args.adapter)},
        },
        "stage250_prepare": prepare,
        "closed_ros_event_v2": events,
        "offline_replay_decision": {
            "exact_stage250_prepare_replay_possible": False,
            "exact_command_application_timing_replay_possible": False,
            "approximate_recorder_receipt_event_replay_possible": True,
            "new_physics_probe_executed": False,
            "status": "BLOCKED_BY_UNRECORDED_PREPARE_AND_APPLICATION_TIMING",
            "reason": (
                "Stage250 lacks prepare start_q, q/dq, targets, contacts and solver state. Event-v2 quantifies recorder-side "
                "receipt timing but was captured with the shipped native controller, not Stage250, and cannot reveal when "
                "the closed simulator consumed each command. Exact offline replay would fabricate missing state/timing."
            ),
            "minimum_future_capture": [
                "at state-ready: full mjData integration snapshot or qpos/qvel/qacc_warmstart/efc state",
                "every prepare control tick: 31D actual q/dq and final published target/Kp/Kd",
                "simulator-side command receipt and applied physics-step index for all four groups",
                "per-substep contact/constraint state or full integration-state snapshot at stand t=0",
                "one source hash binding adapter, scene, runtime config and capture",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--event20", type=Path, default=DEFAULT_EVENT20)
    parser.add_argument("--reset", type=Path, default=DEFAULT_RESET)
    parser.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["offline_replay_decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
