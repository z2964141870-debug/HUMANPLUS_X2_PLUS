#!/usr/bin/env python3
"""Replay an HMCP capture through the pure live-reference entry gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import gate_live_reference as gate
import publish_v51_reference as v51


def load_capture(path: Path, pose_scale: float) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("schema") != v51.HMCP_CAPTURE_SCHEMA or "qpos36" not in item:
                continue
            timestamp_s = item.get("velocity_timestamp_s")
            if timestamp_s is None:
                timestamp_s = item.get("send_timestamp_s")
            if timestamp_s is None and "received_monotonic_ns" in item:
                timestamp_s = int(item["received_monotonic_ns"]) * 1e-9
            if timestamp_s is None:
                raise ValueError(f"{path}:{line_number}: frame has no usable timestamp")
            joint_pos, quat_xyzw, _ = v51.hmcp_to_x2_reference(
                np.asarray(item["qpos36"], dtype=np.float64), pose_scale
            )
            records.append((int(item["sequence"]), float(timestamp_s), joint_pos, quat_xyzw))
    if not records:
        raise ValueError(f"capture contains no {v51.HMCP_CAPTURE_SCHEMA} frames: {path}")
    return records


def replay(path: Path, robot_yaw_deg: float, pose_scale: float) -> dict:
    records = load_capture(path, pose_scale)
    config = gate.GateConfig(
        source_stale_s=1.0,
        robot_stale_s=1.0,
        still_seconds=2.0,
        still_frames=50,
        blend_seconds=10.0,
        require_powered_debug=False,
    )
    machine = gate.LiveReferenceGate(config)
    robot_xyzw = gate.yaw_quat_xyzw(math.radians(robot_yaw_deg))
    t0 = records[0][1]
    previous_pos = None
    previous_timestamp_s = None
    transitions = []
    states_seen = set()
    for index, (sequence, timestamp_s, joint_pos, quat_xyzw) in enumerate(records):
        now_s = timestamp_s - t0
        dt_s = None if previous_timestamp_s is None else timestamp_s - previous_timestamp_s
        if previous_pos is None or dt_s is None or not 0.005 <= dt_s <= 0.2:
            joint_vel = np.zeros(v51.NUM_DOFS, dtype=np.float64)
        else:
            joint_vel = (joint_pos - previous_pos) / dt_s
        machine.on_robot(
            gate.RobotFrame(
                base_quat_wxyz=robot_xyzw[[3, 0, 1, 2]],
                control_tick=index,
                ros_timestamp_s=1000.0 + now_s,
                dry_run=False,
                recv_s=now_s,
            ),
            now_s,
        )
        machine.on_source(
            gate.PoseFrame(joint_pos, joint_vel, quat_xyzw, sequence, now_s), now_s
        )
        states_seen.add(machine.state.value)
        if not transitions or transitions[-1]["state"] != machine.state.value:
            transitions.append({
                "sequence": sequence,
                "time_s": now_s,
                "state": machine.state.value,
                "reason": machine.reason,
            })
        previous_pos = joint_pos
        previous_timestamp_s = timestamp_s
    final_status = machine.status(
        records[-1][1] - t0,
        session_id="offline-replay",
        arm_trigger_file="offline-replay",
    )
    return {
        "schema": "x2-live-reference-gate-replay-v1",
        "capture": str(path),
        "frames": len(records),
        "duration_s": records[-1][1] - records[0][1],
        "robot_yaw_deg_for_frame_check_only": robot_yaw_deg,
        "pose_scale": pose_scale,
        "states_seen": sorted(states_seen),
        "ever_ready": "STANDSTILL_READY" in states_seen,
        "ever_live": "LIVE" in states_seen,
        "transitions": transitions,
        "final_status": final_status,
        "interpretation": (
            "Offline source-entry audit only. Synthetic robot yaw cannot prove powered safety."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--robot-yaw-deg", type=float, default=32.62)
    parser.add_argument("--pose-scale", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-never-ready",
        action="store_true",
        help="exit nonzero if this capture reaches STANDSTILL_READY",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = replay(args.capture.expanduser(), args.robot_yaw_deg, args.pose_scale)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"GATE_REPLAY_ERROR: {exc}")
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(
            f"GATE_REPLAY frames={result['frames']} states={result['states_seen']} "
            f"ever_ready={result['ever_ready']} output={output}"
        )
    if args.require_never_ready and result["ever_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
