#!/usr/bin/env python3
"""Extract healthy stop-boundary states from official X2 MuJoCo traces.

The output is a reset-state dataset, not a motion-reference dataset.  It keeps
the exact Stage208 93-D observation contract and enough named physical state to
initialize a dedicated stand/recovery policy in IsaacLab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ISAAC_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)

OBS_DIM = 93
JOINT_POS = slice(12, 43)
JOINT_VEL = slice(43, 74)
PREVIOUS_ACTION = slice(74, 89)
GAIT_PHASE = slice(89, 93)


def gravity_to_roll_pitch(gravity_body: np.ndarray) -> tuple[float, float]:
    """Recover ZYX roll/pitch from the body-frame world-down vector."""
    gravity = np.asarray(gravity_body, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("invalid projected gravity")
    gx, gy, gz = gravity / norm
    pitch = math.asin(float(np.clip(gx, -1.0, 1.0)))
    roll = math.atan2(-float(gy), -float(gz))
    return roll, pitch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_rows(
    paths: list[Path],
    *,
    sample_interval_s: float,
    max_stop_s: float,
    min_root_z_m: float,
    max_tilt_rad: float,
    max_speed_mps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for source_index, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        trace = payload["trace"]
        selected = 0
        next_sample_s = 0.0
        for trace_index, item in enumerate(trace):
            if item.get("stage") != "stop":
                continue
            elapsed_s = float(item["elapsed_s"])
            if elapsed_s + 1.0e-9 < next_sample_s:
                continue
            next_sample_s += sample_interval_s
            if elapsed_s > max_stop_s + 1.0e-9:
                break
            obs = np.asarray(item.get("obs", []), dtype=np.float32)
            if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
                continue
            root_z = float(item["root_z_m"])
            tilt = float(item["root_tilt_rad"])
            speed = math.hypot(float(item["root_vx_b_mps"]), float(item["root_vy_b_mps"]))
            if root_z < min_root_z_m or tilt > max_tilt_rad or speed > max_speed_mps:
                continue
            roll, pitch = gravity_to_roll_pitch(obs[6:9])
            rows.append({
                "source_index": source_index,
                "trace_index": trace_index,
                "elapsed_s": elapsed_s,
                "eventual_pass": bool(summary.get("full_gate_pass")),
                "root_z_m": root_z,
                "root_yaw_rad": float(item["root_yaw_rad"]),
                "root_roll_rad": roll,
                "root_pitch_rad": pitch,
                "base_lin_vel_body_mps": obs[0:3],
                "base_ang_vel_body_radps": obs[3:6],
                "projected_gravity": obs[6:9],
                "joint_pos_rel_rad": obs[JOINT_POS],
                "joint_vel_radps": obs[JOINT_VEL],
                "previous_action": obs[PREVIOUS_ACTION],
                "gait_phase": obs[GAIT_PHASE],
            })
            selected += 1
        sources.append({
            "path": str(path),
            "sha256": _sha256(path),
            "full_gate_pass": bool(summary.get("full_gate_pass")),
            "stop_gate_pass": bool(summary.get("stop_gate_pass")),
            "selected_states": selected,
        })
    return rows, sources


def write_dataset(output: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no states passed the extraction gate")
    scalar_keys = (
        "source_index", "trace_index", "elapsed_s", "eventual_pass", "root_z_m",
        "root_yaw_rad", "root_roll_rad", "root_pitch_rad",
    )
    vector_keys = (
        "base_lin_vel_body_mps", "base_ang_vel_body_radps", "projected_gravity",
        "joint_pos_rel_rad", "joint_vel_radps", "previous_action", "gait_phase",
    )
    arrays = {key: np.asarray([row[key] for row in rows]) for key in scalar_keys}
    arrays.update({key: np.stack([np.asarray(row[key]) for row in rows]) for key in vector_keys})
    arrays["joint_names"] = np.asarray(ISAAC_JOINTS)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--glob", default="stage326_s2652_*stiff1p2_fixed_r*.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-interval-s", type=float, default=0.20)
    parser.add_argument("--max-stop-s", type=float, default=4.0)
    parser.add_argument("--min-root-z-m", type=float, default=0.55)
    parser.add_argument("--max-tilt-rad", type=float, default=0.60)
    parser.add_argument("--max-speed-mps", type=float, default=0.60)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no input matched {args.input_dir / args.glob}")
    rows, sources = extract_rows(
        paths,
        sample_interval_s=args.sample_interval_s,
        max_stop_s=args.max_stop_s,
        min_root_z_m=args.min_root_z_m,
        max_tilt_rad=args.max_tilt_rad,
        max_speed_mps=args.max_speed_mps,
    )
    write_dataset(args.output, rows)
    pass_count = sum(bool(row["eventual_pass"]) for row in rows)
    report = {
        "stage": "stage335_official_stop_recovery_state_extraction",
        "purpose": "reset-state curriculum only; not a reference motion dataset",
        "observation_contract": "Stage208 deterministic 93D",
        "joint_names": list(ISAAC_JOINTS),
        "input_glob": str(args.input_dir / args.glob),
        "source_files": sources,
        "state_count": len(rows),
        "eventual_pass_state_count": pass_count,
        "eventual_fail_state_count": len(rows) - pass_count,
        "sample_interval_s": args.sample_interval_s,
        "max_stop_s": args.max_stop_s,
        "health_gate": {
            "min_root_z_m": args.min_root_z_m,
            "max_tilt_rad": args.max_tilt_rad,
            "max_speed_mps": args.max_speed_mps,
        },
        "dataset_path": str(args.output),
        "dataset_sha256": _sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
