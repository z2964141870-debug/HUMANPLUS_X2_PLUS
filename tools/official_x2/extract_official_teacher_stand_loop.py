#!/usr/bin/env python3
"""Extract a reproducible low-drift lower-body loop from the official teacher.

This does not claim that the dance clip is a locomotion reference.  It creates
one training-free diagnostic: can a short command trajectory already proven
stable in the official plant serve as a temporary stand attractor?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-seconds", type=float, default=1.24)
    parser.add_argument("--duration-seconds", type=float, default=0.80)
    parser.add_argument("--full-body", action="store_true")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    time_s = data["time_s"]
    names = data["joint_names"].tolist()
    command_q = data["command_q_rad"]
    actual_q = data["joint_q_rad"]
    begin = int(np.searchsorted(time_s, args.start_seconds, side="left"))
    sample_dt = float(np.median(np.diff(time_s)))
    count = max(2, int(round(args.duration_seconds / sample_dt)))
    end = begin + count
    if end > command_q.shape[0]:
        raise ValueError("requested teacher window exceeds input")
    selected_names = tuple(names) if args.full_body else LOWER_JOINTS
    indices = [names.index(name) for name in selected_names]
    targets = command_q[begin:end, indices].astype(np.float32)
    kp = data["command_kp"][begin, indices].astype(np.float32)
    kd = data["command_kd"][begin, indices].astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        action_joint_order=np.asarray(selected_names),
        joint_target_rad=targets,
        source_joint_q_rad=actual_q[begin, indices].astype(np.float32),
        joint_kp=kp,
        joint_kd=kd,
        source=np.asarray(str(Path(args.input).resolve())),
        truth_label=np.asarray("official_simulation_teacher_not_real_hardware_truth"),
        source_start_seconds=np.asarray(float(time_s[begin])),
        source_duration_seconds=np.asarray(float(time_s[end - 1] - time_s[begin] + sample_dt)),
        sample_dt_seconds=np.asarray(sample_dt),
        loop_seam_mean_abs_rad=np.asarray(float(np.mean(np.abs(targets[-1] - targets[0])))),
    )
    print(output)


if __name__ == "__main__":
    main()
