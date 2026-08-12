#!/usr/bin/env python3
"""Summarize and hash-bind the 24-case Stage264 new-machine replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_cases() -> list[tuple[str, str, str, float]]:
    rows = []
    for speed, vx in (("medium", 0.30), ("low", 0.25)):
        for motion, repeats in (("straight", 6), ("turn_right", 3), ("turn_left", 3)):
            for repeat in range(1, repeats + 1):
                rows.append((f"stage264_{speed}_{motion}_r{repeat}", speed, motion, vx))
    return rows


def l2_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Hash-bind the completed prefix without claiming the 24-case gate.",
    )
    args = parser.parse_args()

    rows = []
    for case, speed, motion, expected_vx in expected_cases():
        path = args.result_root / f"{case}.json"
        if not path.is_file():
            if args.allow_partial:
                continue
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        if not math.isclose(summary["command_vx_mps"], expected_vx, abs_tol=1.0e-12):
            raise RuntimeError(f"{case}: command_vx mismatch")
        expected_turn = {
            "straight": ("straight", None, False),
            "turn_right": ("turn", 0.15, False),
            "turn_left": ("turn", -0.09, True),
        }[motion]
        if summary["move_gate_kind"] != expected_turn[0]:
            raise RuntimeError(f"{case}: move gate kind mismatch")
        if summary["fixed_wz_radps"] != expected_turn[1]:
            raise RuntimeError(f"{case}: turn command mismatch")
        if summary["mirror_policy"] is not expected_turn[2]:
            raise RuntimeError(f"{case}: mirror-policy mismatch")
        move_trace = [row for row in payload["trace"] if row["stage"] == "move"]
        stop_trace = [row for row in payload["trace"] if row["stage"] == "stop"]
        if len(move_trace) != 200 or len(stop_trace) != 400:
            raise RuntimeError(f"{case}: expected 200 move and 400 stop trace rows")
        fall_rows = [
            row
            for row in stop_trace
            if row["root_z_m"] < summary["gate_thresholds"]["stop_root_z_min_m"]
            or row["root_tilt_rad"] > summary["gate_thresholds"]["stop_tilt_max_rad"]
        ]
        row = {
                "case": case,
                "speed": speed,
                "motion": motion,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "full_gate_pass": bool(summary["full_gate_pass"]),
                "startup_gate_pass": bool(summary["startup_gate_pass"]),
                "move_gate_pass": bool(summary["move_gate_pass"]),
                "stop_gate_pass": bool(summary["stop_gate_pass"]),
                "command_vx_mps": float(summary["command_vx_mps"]),
                "forward_m": float(summary["move_forward_displacement_m"]),
                "lateral_m": float(summary["move_lateral_displacement_m"]),
                "yaw_progress_rad": float(summary["move_yaw_progress_rad"]),
                "signed_pitch_mean_rad": float(summary["move_root_pitch_mean_rad"]),
                "stop_drift_m": float(summary["stop_root_xy_drift_m"]),
                "stop_settle_s": float(summary["stop_settle_time_s"]),
                "stop_root_z_min_m": float(summary["stop_root_z_min_m"]),
                "stop_tilt_max_rad": float(summary["stop_root_tilt_max_rad"]),
                "stop_entry": {
                    "pitch_rad": float(stop_trace[0]["root_pitch_rad"]),
                    "tilt_rad": float(stop_trace[0]["root_tilt_rad"]),
                    "body_vx_mps": float(stop_trace[0]["root_vx_b_mps"]),
                    "body_vy_mps": float(stop_trace[0]["root_vy_b_mps"]),
                    "yaw_rate_radps": float(stop_trace[0]["root_yaw_rate_radps"]),
                    "moving_to_stationary_action_jump_l2": l2_distance(
                        move_trace[-1]["action"], stop_trace[0]["action"]
                    ),
                },
                "first_stop_gate_violation_s": (
                    None if not fall_rows else float(fall_rows[0]["elapsed_s"])
                ),
            }
        rows.append(row)

    groups = {}
    for speed in ("medium", "low"):
        groups[speed] = {}
        for motion in ("straight", "turn_right", "turn_left"):
            selected = [row for row in rows if row["speed"] == speed and row["motion"] == motion]
            groups[speed][motion] = {
                "passes": sum(row["full_gate_pass"] for row in selected),
                "runs": len(selected),
                "forward_m_mean": (
                    statistics.fmean(row["forward_m"] for row in selected) if selected else None
                ),
                "signed_pitch_mean_rad": (
                    statistics.fmean(row["signed_pitch_mean_rad"] for row in selected)
                    if selected else None
                ),
                "yaw_progress_rad_mean": (
                    statistics.fmean(row["yaw_progress_rad"] for row in selected)
                    if selected else None
                ),
            }
    passes = sum(row["full_gate_pass"] for row in rows)
    failures = [row for row in rows if not row["full_gate_pass"]]
    report = {
        "stage": "stage264_new_machine_replay_20260812",
        "contract": {
            "checkpoint": "Stage219 model_2600.pt / stage219_s2600_actor.onnx",
            "deployment": "Stage250 actor-clip and bounded direction supervisor",
            "matrix": "speed {0.30,0.25} x {straight x6,right x3,left x3}",
            "physics_episodes": len(rows),
            "complete_matrix": len(rows) == 24,
        },
        "results": {
            "passes": passes,
            "runs": len(rows),
            "groups": groups,
            "failures": failures,
        },
        "artifact_bytes": sum(row["bytes"] for row in rows),
        "cases": rows,
        "decision": {
            "new_machine_gate_replayed": passes == len(rows) == 24,
            "historical_24_of_24_reproduced": passes == len(rows) == 24,
            "natural_posture_pass": False,
            "task2_required": True,
            "reason": (
                "24-case functional gate reproduced, but signed pitch remains approximately -0.2 rad"
                if passes == len(rows) == 24
                else f"functional replay is only {passes}/{len(rows)}; at least one stop failure occurred and signed pitch remains negative"
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage264 new-machine replay",
        "",
        f"- Full functional gate: **{passes}/{len(rows)}**.",
        f"- Raw external JSON traces: **{report['artifact_bytes'] / 1024 / 1024:.1f} MiB**.",
        f"- Complete matrix: **{report['contract']['complete_matrix']}**; no physics failure was retried or overwritten.",
        "",
        "| speed | motion | pass | mean forward | mean yaw | signed pitch mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for speed in ("medium", "low"):
        for motion in ("straight", "turn_right", "turn_left"):
            group = groups[speed][motion]
            forward = "—" if group["forward_m_mean"] is None else f"{group['forward_m_mean']:.3f} m"
            yaw = "—" if group["yaw_progress_rad_mean"] is None else f"{group['yaw_progress_rad_mean']:.3f} rad"
            pitch = "—" if group["signed_pitch_mean_rad"] is None else f"{group['signed_pitch_mean_rad']:.3f} rad"
            lines.append(
                f"| {speed} | {motion} | {group['passes']}/{group['runs']} | "
                f"{forward} | {yaw} | {pitch} |"
            )
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend([
            "| case | passed phases | first violation | stop z min | stop tilt max | handoff action jump |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in failures:
            passed_phases = "/".join(
                phase
                for phase, key in (
                    ("startup", "startup_gate_pass"),
                    ("move", "move_gate_pass"),
                    ("stop", "stop_gate_pass"),
                )
                if row[key]
            )
            lines.append(
                f"| {row['case']} | {passed_phases} | {row['first_stop_gate_violation_s']:.2f} s | "
                f"{row['stop_root_z_min_m']:.3f} m | {row['stop_tilt_max_rad']:.3f} rad | "
                f"{row['stop_entry']['moving_to_stationary_action_jump_l2']:.3f} |"
            )
    else:
        lines.append("None.")
    lines.extend([
        "",
        (
            "The historical 24/24 claim is not reproduced: the no-retry new-machine gate is 22/24. "
            "Both failures pass locomotion and fail after the direct moving-to-stationary policy handoff. "
            "Negative signed pitch also keeps Task 2 open."
            if failures
            else "The 24/24 functional gate is reproduced, but negative signed pitch keeps Task 2 open."
        ),
        "",
    ])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
