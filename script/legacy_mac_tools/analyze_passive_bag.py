#!/usr/bin/env python3
"""Analyze the read-only X2 passive rosbag without requiring a ROS install."""

from __future__ import annotations

import argparse
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev


def _align(pos: int, base: int, size: int) -> int:
    return base + ((pos - base + size - 1) // size) * size


def parse_multi_array(blob: bytes, scalar: str) -> list[float]:
    # ROS 2 CDR encapsulation is four bytes. Alignment is relative to payload.
    base = 4
    pos = base
    (dim_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    if dim_len != 0:
        raise ValueError(f"only empty MultiArrayLayout is supported, got {dim_len=}")
    pos += 4  # data_offset
    (length,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    size = struct.calcsize(scalar)
    pos = _align(pos, base, size)
    return list(struct.unpack_from(f"<{length}{scalar}", blob, pos))


@dataclass
class Imu:
    quat_xyzw: tuple[float, float, float, float]
    omega: tuple[float, float, float]
    accel: tuple[float, float, float]


def parse_imu(blob: bytes) -> Imu:
    base = 4
    pos = base + 8  # header stamp
    (name_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4 + name_len
    pos = _align(pos, base, 8)
    values = struct.unpack_from("<37d", blob, pos)
    return Imu(
        quat_xyzw=tuple(values[0:4]),
        omega=tuple(values[13:16]),
        accel=tuple(values[25:28]),
    )


def norm(values: tuple[float, ...] | list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def describe(values: list[float]) -> tuple[float, float, float, float]:
    return min(values), max(values), fmean(values), pstdev(values)


def nearest_pairs(
    left: list[tuple[int, list[float]]],
    right: list[tuple[int, list[float]]],
    max_delta_ns: int = 30_000_000,
) -> list[tuple[list[float], list[float], int]]:
    pairs: list[tuple[list[float], list[float], int]] = []
    j = 0
    for t_left, value_left in left:
        while j + 1 < len(right) and abs(right[j + 1][0] - t_left) <= abs(right[j][0] - t_left):
            j += 1
        delta = abs(right[j][0] - t_left)
        if delta <= max_delta_ns:
            pairs.append((value_left, right[j][1], delta))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db3", type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db3)
    topics = {
        row[1]: (row[0], row[2])
        for row in conn.execute("SELECT id, name, type FROM topics")
    }

    def rows(name: str) -> list[tuple[int, bytes]]:
        topic_id = topics[name][0]
        return list(conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_id,),
        ))

    retarget = [(t, parse_multi_array(data, "d")) for t, data in rows("/aima/mc/joint/retargeting")]
    frame = [(t, parse_multi_array(data, "f")) for t, data in rows("/aima/mc/retarget_frame")]
    torso = [(t, parse_imu(data)) for t, data in rows("/aima/hal/imu/torso/state")]
    chest = [(t, parse_imu(data)) for t, data in rows("/aima/hal/imu/chest/state")]

    all_times = [t for series in (retarget, frame, torso, chest) for t, _ in series]
    duration = (max(all_times) - min(all_times)) / 1e9
    print(f"duration_s={duration:.6f}")
    for name, series in (
        ("retargeting", retarget), ("retarget_frame", frame),
        ("torso_imu", torso), ("chest_imu", chest),
    ):
        series_duration = (series[-1][0] - series[0][0]) / 1e9
        hz = (len(series) - 1) / series_duration
        print(f"{name}: count={len(series)} rate_hz={hz:.3f}")

    assert {len(v) for _, v in retarget} == {62}
    assert {len(v) for _, v in frame} == {93}

    pairs = nearest_pairs(retarget, frame)
    target_diffs: list[float] = []
    time_deltas_ms: list[float] = []
    for command, feedback, delta_ns in pairs:
        # Establish the observed slice relation without assigning undocumented
        # semantics to the remaining 38 values.
        target_diffs.extend(abs(a - b) for a, b in zip(command[7:62], feedback[38:93]))
        time_deltas_ms.append(delta_ns / 1e6)
    print("\n62/93 structure validation:")
    print(f"  paired_frames={len(pairs)}")
    print(f"  nearest_timestamp_delta_ms_mean={fmean(time_deltas_ms):.4f}")
    print(f"  target_slice_abs_diff_mean={fmean(target_diffs):.9f}")
    print(f"  target_slice_abs_diff_max={max(target_diffs):.9f}")
    command_stds = [pstdev([sample[j] for _, sample in retarget]) for j in range(62)]
    prefix_stds = [pstdev([sample[j] for _, sample in frame]) for j in range(38)]
    print(f"  command_temporal_std_max={max(command_stds):.9f}")
    print(f"  undocumented_prefix_temporal_std_max={max(prefix_stds):.9f}")

    for name, series in (("torso", torso), ("chest", chest)):
        quat_norms = [norm(sample.quat_xyzw) for _, sample in series]
        omega_norms = [norm(sample.omega) for _, sample in series]
        accel_norms = [norm(sample.accel) for _, sample in series]
        _, _, quat_mean, quat_std = describe(quat_norms)
        _, _, omega_mean, omega_std = describe(omega_norms)
        _, _, accel_mean, accel_std = describe(accel_norms)
        print(f"\n{name} IMU:")
        print(f"  quaternion_norm mean={quat_mean:.8f} std={quat_std:.8f}")
        print(f"  angular_speed_rad_s mean={omega_mean:.6f} std={omega_std:.6f}")
        print(f"  acceleration_norm_m_s2 mean={accel_mean:.6f} std={accel_std:.6f}")


if __name__ == "__main__":
    main()
