#!/usr/bin/env python3
"""Summarize live HMCP timing, velocity semantics, and Sonic action clipping."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

import publish_v51_reference as pub


CONTROL_RE = re.compile(
    r"CONTROL tick=(?P<tick>\d+).*?act_clip_ticks=(?P<clips>\d+) "
    r"max_pre_clip=(?P<peak>[0-9.eE+-]+)"
)


def _p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95)) if values.size else float("nan")


def summarize_capture(path: str | Path) -> dict[str, float | int | Counter[str]]:
    receive_timestamps: list[float] = []
    velocity_timestamps: list[float | None] = []
    positions: list[np.ndarray] = []
    timestamp_sources: Counter[str] = Counter()

    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("schema") != pub.HMCP_CAPTURE_SCHEMA or "qpos36" not in record:
                continue

            receive_ns = record.get("received_monotonic_ns")
            receive_s = None if receive_ns is None else int(receive_ns) * 1e-9
            if receive_s is not None:
                receive_timestamps.append(receive_s)

            velocity_s = record.get("velocity_timestamp_s")
            source = record.get("velocity_timestamp_source")
            if velocity_s is None:
                velocity_s = record.get("send_timestamp_s")
                if velocity_s is not None:
                    source = "hmcp-send"
            if velocity_s is None:
                velocity_s = receive_s
                if velocity_s is not None:
                    source = "receive-monotonic"
            velocity_timestamps.append(None if velocity_s is None else float(velocity_s))
            timestamp_sources[str(source or "missing")] += 1

            joint_pos, _, _ = pub.hmcp_to_x2_reference(
                np.asarray(record["qpos36"], dtype=np.float64),
                float(record.get("pose_scale", pub.POSE_SCALE)),
            )
            positions.append(joint_pos)

    frame_count = len(positions)
    receive_dt = np.diff(np.asarray(receive_timestamps, dtype=np.float64))
    source_hz = 0.0
    if len(receive_timestamps) >= 2:
        elapsed = receive_timestamps[-1] - receive_timestamps[0]
        if elapsed > 0.0:
            source_hz = (len(receive_timestamps) - 1) / elapsed

    derivative_resets = 0
    valid_dt: list[float] = []
    fixed_abs = np.empty(0, dtype=np.float64)
    timestamp_abs = np.empty(0, dtype=np.float64)
    if frame_count >= 2:
        delta = np.diff(np.stack(positions), axis=0)
        timestamp_velocity = np.zeros_like(delta)
        valid = np.zeros(frame_count - 1, dtype=bool)
        for index, (previous, current) in enumerate(
            zip(velocity_timestamps, velocity_timestamps[1:])
        ):
            if previous is None or current is None:
                derivative_resets += 1
                continue
            dt_s = current - previous
            if not pub.TIMESTAMP_MIN_DT_S <= dt_s <= pub.TIMESTAMP_RESET_GAP_S:
                derivative_resets += 1
                continue
            valid[index] = True
            valid_dt.append(dt_s)
            timestamp_velocity[index] = delta[index] / dt_s
        fixed_abs = np.abs(delta * pub.VELOCITY_DIFF_HZ).reshape(-1)
        timestamp_abs = np.abs(timestamp_velocity).reshape(-1)

    valid_dt_array = np.asarray(valid_dt, dtype=np.float64)
    ratio = valid_dt_array * pub.VELOCITY_DIFF_HZ
    return {
        "frames": frame_count,
        "source_hz": source_hz,
        "receive_gap_p95_ms": _p95(receive_dt) * 1000.0,
        "receive_gap_max_ms": float(np.max(receive_dt) * 1000.0) if receive_dt.size else float("nan"),
        "derivative_resets": derivative_resets,
        "timestamp_sources": timestamp_sources,
        "fixed_ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
        "fixed_ratio_p95": _p95(ratio),
        "fixed_velocity_abs_p95": _p95(fixed_abs),
        "timestamp_velocity_abs_p95": _p95(timestamp_abs),
        "fixed_velocity_abs_max": float(np.max(fixed_abs)) if fixed_abs.size else float("nan"),
        "timestamp_velocity_abs_max": (
            float(np.max(timestamp_abs)) if timestamp_abs.size else float("nan")
        ),
    }


def summarize_deploy_log(path: str | Path) -> dict[str, float | int]:
    control_ticks = 0
    clipped_ticks = 0
    max_pre_clip = 0.0
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = CONTROL_RE.search(line)
            if match is None:
                continue
            control_ticks = max(control_ticks, int(match.group("tick")))
            clipped_ticks = max(clipped_ticks, int(match.group("clips")))
            max_pre_clip = max(max_pre_clip, float(match.group("peak")))
    return {
        "control_ticks": control_ticks,
        "clipped_ticks": clipped_ticks,
        "max_pre_clip": max_pre_clip,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--deploy-log", required=True)
    args = parser.parse_args(argv)

    capture = summarize_capture(args.capture)
    policy = summarize_deploy_log(args.deploy_log)
    timestamp_sources = capture["timestamp_sources"]
    assert isinstance(timestamp_sources, Counter)
    print(
        "[live-summary] velocity: "
        f"frames={capture['frames']} mode=timestamp "
        f"timestamp_hmcp={timestamp_sources['hmcp-send']} "
        f"fallback={timestamp_sources['receive-monotonic']} "
        f"missing={timestamp_sources['missing']} "
        f"derivative_resets={capture['derivative_resets']}"
    )
    print(
        "[live-summary] source: "
        f"rate={capture['source_hz']:.2f}Hz "
        f"p95_gap={capture['receive_gap_p95_ms']:.1f}ms "
        f"max_gap={capture['receive_gap_max_ms']:.1f}ms"
    )
    print(
        "[live-summary] fixed/timestamp: "
        f"ratio_median={capture['fixed_ratio_median']:.3f} "
        f"ratio_p95={capture['fixed_ratio_p95']:.3f} "
        f"velocity_p95={capture['fixed_velocity_abs_p95']:.3f}/"
        f"{capture['timestamp_velocity_abs_p95']:.3f}rad/s "
        f"velocity_max={capture['fixed_velocity_abs_max']:.3f}/"
        f"{capture['timestamp_velocity_abs_max']:.3f}rad/s"
    )
    print(
        "[live-summary] policy: "
        f"control_ticks={policy['control_ticks']} "
        f"clipped_ticks={policy['clipped_ticks']} "
        f"max_pre_clip={policy['max_pre_clip']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
