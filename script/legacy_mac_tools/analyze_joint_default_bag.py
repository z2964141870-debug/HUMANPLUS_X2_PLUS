#!/usr/bin/env python3
"""Analyze an X2 JOINT_DEFAULT rosbag without a ROS installation."""

from __future__ import annotations

import argparse
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev


def align(pos: int, base: int, size: int) -> int:
    return base + ((pos - base + size - 1) // size) * size


def read_u32(blob: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from("<I", blob, pos)[0], pos + 4


def read_string(blob: bytes, pos: int) -> tuple[str, int]:
    length, pos = read_u32(blob, pos)
    raw = blob[pos:pos + length]
    return raw.rstrip(b"\0").decode("utf-8"), pos + length


def parse_header(blob: bytes, pos: int, base: int) -> int:
    pos += 8  # stamp: int32 sec, uint32 nanosec
    _, pos = read_string(blob, pos)
    pos = align(pos, base, 4)
    pos += 4  # sequence
    pos += 8  # meas_stamp
    return pos


@dataclass
class JointState:
    position: float
    velocity: float
    effort: float
    coil_temp: int
    motor_temp: int
    motor_vol: int


@dataclass
class JointCommand:
    position: float
    velocity: float
    effort: float
    stiffness: float
    damping: float


def parse_joint_state_array(blob: bytes) -> dict[str, JointState]:
    base = 4
    pos = parse_header(blob, base, base)
    pos = align(pos, base, 4)
    count, pos = read_u32(blob, pos)
    # The robot's current JointStateArray wire schema carries one additional
    # uint32 status/reserved field after MessageHeader. The locally installed
    # .msg lacks it, which is why `ros2 topic echo` showed an empty array.
    if count == 0:
        possible_count, possible_pos = read_u32(blob, pos)
        if 0 < possible_count < 128:
            count, pos = possible_count, possible_pos
    joints: dict[str, JointState] = {}
    for _ in range(count):
        pos = align(pos, base, 4)
        name, pos = read_string(blob, pos)
        pos = align(pos, base, 8)
        position, velocity, effort = struct.unpack_from("<3d", blob, pos)
        pos += 24
        coil_temp, motor_temp, motor_vol = struct.unpack_from("<3B", blob, pos)
        pos += 3
        joints[name] = JointState(position, velocity, effort, coil_temp, motor_temp, motor_vol)
    return joints


def parse_joint_command_array(blob: bytes) -> dict[str, JointCommand]:
    base = 4
    pos = parse_header(blob, base, base)
    pos = align(pos, base, 4)
    count, pos = read_u32(blob, pos)
    joints: dict[str, JointCommand] = {}
    for _ in range(count):
        pos = align(pos, base, 4)
        name, pos = read_string(blob, pos)
        pos = align(pos, base, 8)
        values = struct.unpack_from("<5d", blob, pos)
        pos += 40
        joints[name] = JointCommand(*values)
    return joints


def parse_multi_array(blob: bytes, scalar: str) -> list[float]:
    base = 4
    pos = base
    dim_len, pos = read_u32(blob, pos)
    if dim_len != 0:
        raise ValueError(f"unsupported populated layout: {dim_len=}")
    pos += 4  # data_offset
    length, pos = read_u32(blob, pos)
    size = struct.calcsize(scalar)
    pos = align(pos, base, size)
    return list(struct.unpack_from(f"<{length}{scalar}", blob, pos))


@dataclass
class Imu:
    quat: tuple[float, float, float, float]
    omega: tuple[float, float, float]
    accel: tuple[float, float, float]


def parse_imu(blob: bytes) -> Imu:
    base = 4
    pos = base + 8
    _, pos = read_string(blob, pos)
    pos = align(pos, base, 8)
    values = struct.unpack_from("<37d", blob, pos)
    return Imu(tuple(values[0:4]), tuple(values[13:16]), tuple(values[25:28]))


def vector_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in values))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def nearest_pairs(left, right, max_delta_ns: int):
    pairs = []
    j = 0
    for t_left, value_left in left:
        while j + 1 < len(right) and abs(right[j + 1][0] - t_left) <= abs(right[j][0] - t_left):
            j += 1
        delta = abs(right[j][0] - t_left)
        if delta <= max_delta_ns:
            pairs.append((value_left, right[j][1], delta))
    return pairs


def rate(series) -> float:
    return (len(series) - 1) / ((series[-1][0] - series[0][0]) / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db3", type=Path)
    args = parser.parse_args()
    conn = sqlite3.connect(args.db3)
    topics = {name: ident for ident, name in conn.execute("SELECT id, name FROM topics")}

    def raw(name: str):
        return list(conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topics[name],),
        ))

    states = {}
    commands = {}
    for group in ("arm", "leg", "waist", "head"):
        states[group] = [(t, parse_joint_state_array(blob)) for t, blob in raw(f"/aima/hal/joint/{group}/state")]
        commands[group] = [(t, parse_joint_command_array(blob)) for t, blob in raw(f"/aima/hal/joint/{group}/command")]

    retarget = [(t, parse_multi_array(blob, "d")) for t, blob in raw("/aima/mc/joint/retargeting")]
    frames = [(t, parse_multi_array(blob, "f")) for t, blob in raw("/aima/mc/retarget_frame")]
    imus = {
        name: [(t, parse_imu(blob)) for t, blob in raw(f"/aima/hal/imu/{name}/state")]
        for name in ("torso", "chest")
    }

    all_times = [t for group in states.values() for t, _ in group]
    duration = (max(all_times) - min(all_times)) / 1e9
    print(f"duration_s={duration:.6f}")
    for group in ("arm", "leg", "waist", "head"):
        state_names = sorted(states[group][0][1])
        command_names = sorted(commands[group][0][1])
        print(
            f"{group}: state_count={len(states[group])} state_hz={rate(states[group]):.3f} "
            f"command_count={len(commands[group])} command_hz={rate(commands[group]):.3f} "
            f"joints={len(state_names)} names_match={state_names == command_names}"
        )

    print("\nCommand-to-state position tracking (nearest samples):")
    joint_rows = []
    for group in ("arm", "leg", "waist", "head"):
        pairs = nearest_pairs(commands[group], states[group], max_delta_ns=20_000_000)
        deltas = [delta / 1e6 for _, _, delta in pairs]
        for name in sorted(commands[group][0][1]):
            errors = [state[name].position - command[name].position for command, state, _ in pairs]
            velocities = [state[name].velocity for _, state, _ in pairs]
            positions = [state[name].position for _, state, _ in pairs]
            joint_rows.append((
                math.sqrt(fmean(e * e for e in errors)),
                max(abs(e) for e in errors),
                percentile([abs(e) for e in errors], 0.95),
                pstdev(positions),
                math.sqrt(fmean(v * v for v in velocities)),
                group,
                name,
            ))
        print(f"  {group}: pairs={len(pairs)} nearest_delta_ms_mean={fmean(deltas):.4f}")
    for rms, max_abs, p95, pos_std, vel_rms, group, name in sorted(joint_rows, reverse=True)[:15]:
        print(
            f"  {group:5s} {name:30s} rms={rms:.6f} p95={p95:.6f} "
            f"max={max_abs:.6f} position_std={pos_std:.6f} velocity_rms={vel_rms:.6f}"
        )
    all_rms = [row[0] for row in joint_rows]
    all_p95 = [row[2] for row in joint_rows]
    print(f"  all_joint_rms_mean={fmean(all_rms):.6f}")
    print(f"  worst_joint_rms={max(all_rms):.6f}")
    print(f"  worst_joint_p95={max(all_p95):.6f}")

    pairs = nearest_pairs(retarget, frames, max_delta_ns=30_000_000)
    target_diffs = [
        abs(a - b)
        for command, frame, _ in pairs
        for a, b in zip(command[7:62], frame[38:93])
    ]
    command_stds = [pstdev([sample[j] for _, sample in retarget]) for j in range(62)]
    print("\nRetarget streams:")
    print(f"  command_count={len(retarget)} command_hz={rate(retarget):.3f}")
    print(f"  frame_count={len(frames)} frame_hz={rate(frames):.3f}")
    print(f"  suffix_abs_diff_max={max(target_diffs):.9f}")
    print(f"  command_temporal_std_max={max(command_stds):.9f}")

    print("\nIMU stationary baseline:")
    for name, series in imus.items():
        quat_norms = [vector_norm(sample.quat) for _, sample in series]
        omega_norms = [vector_norm(sample.omega) for _, sample in series]
        accel_norms = [vector_norm(sample.accel) for _, sample in series]
        print(
            f"  {name}: count={len(series)} hz={rate(series):.3f} "
            f"quat_norm_mean={fmean(quat_norms):.8f} "
            f"omega_mean={fmean(omega_norms):.6f} omega_p95={percentile(omega_norms, 0.95):.6f} "
            f"accel_mean={fmean(accel_norms):.6f} accel_std={pstdev(accel_norms):.6f}"
        )


if __name__ == "__main__":
    main()
