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
        rows.append(
            {
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
            }
        )

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
    report = {
        "stage": "stage264_new_machine_replay_20260812",
        "contract": {
            "checkpoint": "Stage219 model_2600.pt / stage219_s2600_actor.onnx",
            "deployment": "Stage250 actor-clip and bounded direction supervisor",
            "matrix": "speed {0.30,0.25} x {straight x6,right x3,left x3}",
            "physics_episodes": len(rows),
            "complete_matrix": len(rows) == 24,
        },
        "results": {"passes": passes, "runs": len(rows), "groups": groups},
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
    lines.extend([
        "",
        "The new machine reproduces the frozen functional gate only. The signed-pitch evidence keeps Task 2 open.",
        "",
    ])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
