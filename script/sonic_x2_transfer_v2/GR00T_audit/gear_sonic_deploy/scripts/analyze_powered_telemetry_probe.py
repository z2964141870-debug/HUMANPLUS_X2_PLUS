#!/usr/bin/env python3
"""Analyze one fixed-reference X2 powered probe with the 16-file logger."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_supported_standstill import (
    pelvis_angles_deg,
    read_numeric_csv,
    read_tick_csv,
)


NUMERIC_TABLES = (
    "policy_target_pos",
    "target_pos",
    "joint_pos",
    "joint_vel",
    "tracking_error",
    "joint_effort",
    "stiffness",
    "damping",
    "pd_torque",
    "coil_temp",
    "motor_temp",
    "motor_voltage",
    "domain_state",
    "action_il",
    "imu",
)

SELECTED_JOINTS = (
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def first_last_abs_max(values: np.ndarray, elapsed: np.ndarray) -> tuple[float, float]:
    first = np.abs(values[elapsed <= min(1.0, elapsed[-1])])
    last = np.abs(values[elapsed >= max(0.0, elapsed[-1] - 1.0)])
    return float(np.max(first)), float(np.max(last))


def joint_names(header: list[str]) -> list[str]:
    names: list[str] = []
    for column in header[1:]:
        if column.startswith("q_"):
            names.append(column[2:])
        else:
            raise ValueError(f"unexpected joint_pos column: {column}")
    return names


def analyze(log_dir: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    tables = {
        name: read_numeric_csv(log_dir / f"{name}.csv")
        for name in NUMERIC_TABLES
    }
    ticks, tick_malformed = read_tick_csv(log_dir / "tick.csv")
    policy_keys = {
        key
        for key, record in ticks.items()
        if record.reason == "supported_policy_probe"
    }
    aligned_keys = sorted(
        policy_keys.intersection(*(set(table.rows) for table in tables.values()))
    )
    if len(aligned_keys) < 2:
        raise ValueError("fewer than two fully aligned supported_policy_probe rows")

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [tables[name].rows[key] for key in aligned_keys], dtype=np.float64
        )

    t = np.asarray(aligned_keys, dtype=np.float64) / 1_000_000.0
    elapsed = t - t[0]
    arrays = {name: values(name)[:, 1:] for name in NUMERIC_TABLES}
    roll, pitch, tilt = pelvis_angles_deg(arrays["imu"][:, :4])
    names = joint_names(tables["joint_pos"].header)
    if len(names) != arrays["joint_pos"].shape[1]:
        raise ValueError("joint name/count mismatch")

    malformed = {
        name: table.malformed_rows for name, table in tables.items()
    }
    malformed["tick"] = tick_malformed
    domains = arrays["domain_state"].astype(np.int64)
    domain_summary = {
        name: {
            "unique": sorted(int(value) for value in np.unique(domains[:, index])),
            "nonzero_samples": int(np.count_nonzero(domains[:, index])),
        }
        for index, name in enumerate(tables["domain_state"].header[1:])
    }

    per_joint: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        raw = arrays["policy_target_pos"][:, index]
        command = arrays["target_pos"][:, index]
        q = arrays["joint_pos"][:, index]
        dq = arrays["joint_vel"][:, index]
        error = arrays["tracking_error"][:, index]
        effort = arrays["joint_effort"][:, index]
        pd_torque = arrays["pd_torque"][:, index]
        first_error, last_error = first_last_abs_max(error, elapsed)
        per_joint[name] = {
            "policy_target_entry_rad": float(raw[0]),
            "policy_target_end_rad": float(raw[-1]),
            "policy_target_min_rad": float(np.min(raw)),
            "policy_target_max_rad": float(np.max(raw)),
            "hal_target_entry_rad": float(command[0]),
            "hal_target_end_rad": float(command[-1]),
            "hal_target_min_rad": float(np.min(command)),
            "hal_target_max_rad": float(np.max(command)),
            "policy_to_hal_backlog_abs_max_rad": float(np.max(np.abs(raw - command))),
            "q_entry_rad": float(q[0]),
            "q_end_rad": float(q[-1]),
            "q_min_rad": float(np.min(q)),
            "q_max_rad": float(np.max(q)),
            "dq_abs_max_rad_s": float(np.max(np.abs(dq))),
            "tracking_entry_rad": float(error[0]),
            "tracking_end_rad": float(error[-1]),
            "tracking_abs_max_rad": float(np.max(np.abs(error))),
            "tracking_rms_rad": float(np.sqrt(np.mean(error * error))),
            "tracking_first_1s_abs_max_rad": first_error,
            "tracking_last_1s_abs_max_rad": last_error,
            "measured_effort_abs_max_nm": float(np.max(np.abs(effort))),
            "measured_effort_rms_nm": float(np.sqrt(np.mean(effort * effort))),
            "estimated_pd_torque_abs_max_nm": float(np.max(np.abs(pd_torque))),
            "estimated_pd_torque_rms_nm": float(np.sqrt(np.mean(pd_torque * pd_torque))),
            "kp_min": float(np.min(arrays["stiffness"][:, index])),
            "kp_max": float(np.max(arrays["stiffness"][:, index])),
            "kd_min": float(np.min(arrays["damping"][:, index])),
            "kd_max": float(np.max(arrays["damping"][:, index])),
            "coil_temp_max_c": float(np.max(arrays["coil_temp"][:, index])),
            "motor_temp_max_c": float(np.max(arrays["motor_temp"][:, index])),
            "motor_voltage_min_v": float(np.min(arrays["motor_voltage"][:, index])),
            "motor_voltage_max_v": float(np.max(arrays["motor_voltage"][:, index])),
        }

    speed_rank = sorted(
        names,
        key=lambda name: per_joint[name]["dq_abs_max_rad_s"],
        reverse=True,
    )[:5]
    tracking_rank = sorted(
        names,
        key=lambda name: per_joint[name]["tracking_abs_max_rad"],
        reverse=True,
    )[:5]
    result: dict[str, object] = {
        "log_dir": str(log_dir),
        "aligned_policy_samples": len(aligned_keys),
        "policy_duration_s": float(elapsed[-1]),
        "sample_period_median_s": percentile(np.diff(t), 50),
        "sample_period_p95_s": percentile(np.diff(t), 95),
        "malformed_rows": malformed,
        "pelvis": {
            "entry_roll_deg": float(roll[0]),
            "entry_pitch_deg": float(pitch[0]),
            "entry_tilt_deg": float(tilt[0]),
            "end_roll_deg": float(roll[-1]),
            "end_pitch_deg": float(pitch[-1]),
            "end_tilt_deg": float(tilt[-1]),
            "peak_abs_roll_deg": float(np.max(np.abs(roll))),
            "peak_abs_pitch_deg": float(np.max(np.abs(pitch))),
            "peak_tilt_deg": float(np.max(tilt)),
        },
        "domain_state": domain_summary,
        "top_tracking_error": [
            {"joint": name, "rad": per_joint[name]["tracking_abs_max_rad"]}
            for name in tracking_rank
        ],
        "top_joint_speed": [
            {"joint": name, "rad_s": per_joint[name]["dq_abs_max_rad_s"]}
            for name in speed_rank
        ],
        "selected_joints": {
            name: per_joint[name] for name in SELECTED_JOINTS
        },
        "all_joints": per_joint,
    }
    trace = {
        "t": t,
        "policy_t": elapsed,
        "pelvis_roll_deg": roll,
        "pelvis_pitch_deg": pitch,
        "pelvis_tilt_deg": tilt,
    }
    for name in SELECTED_JOINTS:
        index = names.index(name)
        trace[f"{name}_policy_target"] = arrays["policy_target_pos"][:, index]
        trace[f"{name}_hal_target"] = arrays["target_pos"][:, index]
        trace[f"{name}_q"] = arrays["joint_pos"][:, index]
        trace[f"{name}_dq"] = arrays["joint_vel"][:, index]
        trace[f"{name}_tracking_error"] = arrays["tracking_error"][:, index]
        trace[f"{name}_measured_effort_nm"] = arrays["joint_effort"][:, index]
        trace[f"{name}_estimated_pd_torque_nm"] = arrays["pd_torque"][:, index]
    return result, trace


def render_markdown(result: dict[str, object]) -> str:
    pelvis = result["pelvis"]
    lines = [
        "# Powered StandStill Telemetry Probe",
        "",
        f"Source: `{result['log_dir']}`",
        "",
        "## Outcome",
        "",
        f"- Aligned policy data: `{result['aligned_policy_samples']}` samples / "
        f"`{result['policy_duration_s']:.3f} s`.",
        "- Pelvis roll/pitch/tilt: "
        f"`{pelvis['entry_roll_deg']:+.2f}/{pelvis['entry_pitch_deg']:+.2f}/"
        f"{pelvis['entry_tilt_deg']:.2f} deg` -> "
        f"`{pelvis['end_roll_deg']:+.2f}/{pelvis['end_pitch_deg']:+.2f}/"
        f"{pelvis['end_tilt_deg']:.2f} deg`.",
        "- Domain protection states: "
        + ", ".join(
            f"{name}={entry['unique']}"
            for name, entry in result["domain_state"].items()
        )
        + ".",
        "",
        "## Selected joints",
        "",
        "| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, joint in result["selected_joints"].items():
        lines.append(
            f"| {name} | {joint['policy_target_end_rad']:+.3f} -> "
            f"{joint['hal_target_end_rad']:+.3f} | "
            f"{joint['q_entry_rad']:+.3f} -> {joint['q_end_rad']:+.3f} | "
            f"{joint['tracking_entry_rad']:+.3f} / "
            f"{joint['tracking_abs_max_rad']:.3f} / "
            f"{joint['tracking_end_rad']:+.3f} | "
            f"{joint['dq_abs_max_rad_s']:.3f} | "
            f"{joint['measured_effort_abs_max_nm']:.2f} Nm | "
            f"{joint['estimated_pd_torque_abs_max_nm']:.2f} Nm | "
            f"{joint['kp_min']:.3f} / {joint['kd_min']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Largest tracking errors",
            "",
        ]
    )
    for entry in result["top_tracking_error"]:
        lines.append(f"- `{entry['joint']}`: `{entry['rad']:.3f} rad`.")
    lines.extend(["", "## Largest measured speeds", ""])
    for entry in result["top_joint_speed"]:
        lines.append(f"- `{entry['joint']}`: `{entry['rad_s']:.3f} rad/s`.")
    lines.extend(
        [
            "",
            "`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` "
            "is the final SafeCommand consumed unchanged by the 250 Hz writer. "
            "`pd_torque.csv` is requested PD torque before motor-side limits; "
            "`joint_effort.csv` is HAL-reported measured effort.",
            "",
        ]
    )
    return "\n".join(lines)


def write_trace(path: Path, trace: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(trace.keys())
        for row in zip(*trace.values()):
            writer.writerow(f"{float(value):.9f}" for value in row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result, trace = analyze(args.log_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    write_trace(args.output_dir / "selected_trace.csv", trace)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
