#!/usr/bin/env python3
"""Analyze a live HMCP capture for a StandStill-to-garment policy entry.

This tool is intentionally offline. It does not import ROS, open sockets, or
publish commands. It applies the same named G1-to-X2 mapping used by
``publish_v51_reference.py`` and reports:

* the live-reference offset from Sonic's trained StandStill pose;
* reference velocity and frame-to-frame motion before policy entry;
* root roll/pitch/yaw and the uncorrected yaw gap to the robot;
* dry-run policy/action clipping evidence when deploy CSVs are available.

The output is evidence for configuring an entry gate and blend. It is not a
powered-stability result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import publish_v51_reference as reference  # noqa: E402


GROUPS: dict[str, tuple[int, ...]] = {
    "leg": tuple(range(0, 12)),
    "waist": tuple(range(12, 15)),
    "arm": tuple(range(15, 29)),
    "head": tuple(range(29, 31)),
}

# Staging envelope for finding an arms-down candidate in a capture. This is
# deliberately separate from the powered controller's command envelope. It is
# only used to reject calibration T-Poses and grossly non-neutral references
# while deriving observed low-motion statistics.
NEUTRAL_REFERENCE_OFFSET_LIMITS_RAD = {
    "leg": 0.60,
    "waist": 0.20,
    "arm": 0.65,
    "head": 0.08,
}

CONTROL_ENTRY_RE = re.compile(
    r"\[([0-9]+(?:\.[0-9]+)?)\].*?(?:Autostart elapsed.*?-> CONTROL|"
    r"GROUND_LOAD_HOLD -> SUPPORTED_POLICY)"
)
ROBOT_YAW_RE = re.compile(r"robot yaw\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*deg")


@dataclass(frozen=True)
class Capture:
    sequence: np.ndarray
    timestamp_s: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat_xyzw: np.ndarray
    root_rpy_deg: np.ndarray
    derivative_resets: int


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _quat_xyzw_to_rpy_deg(quat: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(float(x * x + y * y + z * z + w * w))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("root quaternion is degenerate")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def load_capture(path: Path) -> Capture:
    sequences: list[int] = []
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    last_sequence: int | None = None

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "metadata":
                continue
            if record.get("schema") != reference.HMCP_CAPTURE_SCHEMA:
                raise ValueError(f"{path}:{line_number}: unsupported record schema")
            sequence = int(record["sequence"])
            if not reference.is_newer_sequence(sequence, last_sequence):
                continue
            timestamp = record.get("velocity_timestamp_s", record.get("send_timestamp_s"))
            if timestamp is None and "received_monotonic_ns" in record:
                timestamp = int(record["received_monotonic_ns"]) * 1e-9
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                raise ValueError(f"{path}:{line_number}: invalid timestamp")
            joint_pos, quat_xyzw, _ = reference.hmcp_to_x2_reference(
                np.asarray(record["qpos36"], dtype=np.float64),
                pose_scale=reference.POSE_SCALE,
            )
            sequences.append(sequence)
            timestamps.append(timestamp)
            positions.append(joint_pos)
            quaternions.append(quat_xyzw)
            last_sequence = sequence

    if len(timestamps) < 2:
        raise ValueError(f"{path}: need at least two valid HMCP frames")

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    position_array = np.stack(positions)
    quaternion_array = np.stack(quaternions)
    velocity_array = np.zeros_like(position_array)
    derivative_resets = 0
    for index in range(1, len(timestamp_array)):
        dt_s = float(timestamp_array[index] - timestamp_array[index - 1])
        if (
            dt_s < reference.TIMESTAMP_MIN_DT_S
            or dt_s > reference.TIMESTAMP_RESET_GAP_S
        ):
            derivative_resets += 1
            continue
        velocity_array[index] = (
            position_array[index] - position_array[index - 1]
        ) / dt_s

    rpy = np.asarray(
        [_quat_xyzw_to_rpy_deg(quat) for quat in quaternion_array],
        dtype=np.float64,
    )
    rpy[:, 2] = np.degrees(np.unwrap(np.radians(rpy[:, 2])))
    return Capture(
        sequence=np.asarray(sequences, dtype=np.int64),
        timestamp_s=timestamp_array,
        joint_pos=position_array,
        joint_vel=velocity_array,
        root_quat_xyzw=quaternion_array,
        root_rpy_deg=rpy,
        derivative_resets=derivative_resets,
    )


def parse_deploy_log(path: Path | None) -> tuple[float | None, float | None]:
    if path is None:
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    entry_match = CONTROL_ENTRY_RE.search(text)
    yaw_match = ROBOT_YAW_RE.search(text)
    return (
        float(entry_match.group(1)) if entry_match else None,
        float(yaw_match.group(1)) if yaw_match else None,
    )


def _percentile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, quantile))


def _top_joints(values: np.ndarray, count: int = 6) -> list[dict[str, float | str]]:
    order = np.argsort(np.abs(values))[::-1][:count]
    return [
        {
            "joint": reference.MUJOCO_JOINT_NAMES[int(index)],
            "value": float(values[int(index)]),
        }
        for index in order
    ]


def _window_summary(capture: Capture, indices: np.ndarray) -> dict[str, object]:
    pos = capture.joint_pos[indices]
    vel = capture.joint_vel[indices]
    offset = pos - reference.DEFAULT_ANGLES[None, :]
    steps = np.diff(pos, axis=0) if len(indices) > 1 else np.zeros((0, pos.shape[1]))
    result: dict[str, object] = {
        "frames": int(len(indices)),
        "duration_s": float(
            capture.timestamp_s[indices[-1]] - capture.timestamp_s[indices[0]]
        ),
        "groups": {},
    }
    group_output: dict[str, object] = {}
    for name, group_indices in GROUPS.items():
        group = np.asarray(group_indices, dtype=np.int64)
        group_offset = np.abs(offset[:, group])
        group_velocity = np.abs(vel[:, group])
        group_steps = np.abs(steps[:, group]) if len(steps) else np.asarray([])
        group_range = np.ptp(pos[:, group], axis=0)
        group_output[name] = {
            "offset_abs_p95_rad": _percentile(group_offset, 95.0),
            "offset_abs_max_rad": float(np.max(group_offset)),
            "velocity_abs_p95_rad_s": _percentile(group_velocity, 95.0),
            "velocity_abs_max_rad_s": float(np.max(group_velocity)),
            "step_abs_max_rad": float(np.max(group_steps)) if group_steps.size else 0.0,
            "range_max_rad": float(np.max(group_range)),
            "range_max_joint": reference.MUJOCO_JOINT_NAMES[
                int(group[int(np.argmax(group_range))])
            ],
        }
    result["groups"] = group_output

    rpy = capture.root_rpy_deg[indices]
    result["root"] = {
        "roll_median_deg": float(np.median(rpy[:, 0])),
        "pitch_median_deg": float(np.median(rpy[:, 1])),
        "yaw_median_deg_unwrapped": float(np.median(rpy[:, 2])),
        "roll_range_deg": float(np.ptp(rpy[:, 0])),
        "pitch_range_deg": float(np.ptp(rpy[:, 1])),
        "yaw_range_deg": float(np.ptp(rpy[:, 2])),
    }
    return result


def _low_motion_score(summary: dict[str, object]) -> float:
    """Rank a window without declaring it safe for powered use.

    The denominators are only scale factors so leg, waist, arm, head, and root
    motion contribute comparably. The selected window is an observed
    low-motion candidate, not an acceptance threshold.
    """
    range_scales = {"leg": 0.08, "waist": 0.08, "arm": 0.12, "head": 0.05}
    velocity_scales = {"leg": 0.25, "waist": 0.25, "arm": 0.50, "head": 0.25}
    terms: list[float] = []
    for name in GROUPS:
        group = summary["groups"][name]
        terms.append(float(group["range_max_rad"]) / range_scales[name])
        terms.append(
            float(group["velocity_abs_p95_rad_s"]) / velocity_scales[name]
        )
    root = summary["root"]
    terms.extend(
        float(root[key]) / 3.0
        for key in ("roll_range_deg", "pitch_range_deg", "yaw_range_deg")
    )
    return max(terms) + 0.05 * sum(terms)


def find_low_motion_window(
    capture: Capture,
    window_s: float,
    *,
    require_neutral_envelope: bool = False,
) -> tuple[np.ndarray, dict[str, object], float]:
    best: tuple[np.ndarray, dict[str, object], float] | None = None
    timestamps = capture.timestamp_s
    minimum_duration = 0.9 * window_s
    for end in range(1, len(timestamps)):
        start = int(np.searchsorted(timestamps, timestamps[end] - window_s, side="left"))
        if timestamps[end] - timestamps[start] < minimum_duration:
            continue
        indices = np.arange(start, end + 1, dtype=np.int64)
        summary = _window_summary(capture, indices)
        if require_neutral_envelope:
            groups = summary["groups"]
            if any(
                float(groups[name]["offset_abs_max_rad"])
                > NEUTRAL_REFERENCE_OFFSET_LIMITS_RAD[name]
                for name in GROUPS
            ):
                continue
            rpy = capture.root_rpy_deg[indices]
            if np.max(np.abs(rpy[:, :2])) > 15.0:
                continue
        score = _low_motion_score(summary)
        if best is None or score < best[2]:
            best = (indices, summary, score)
    if best is None:
        qualifier = " neutral" if require_neutral_envelope else ""
        raise ValueError(
            f"capture has no complete {window_s:.2f}s{qualifier} analysis window"
        )
    return best


def _read_numeric_csv(path: Path | None, prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = [name for name in (reader.fieldnames or []) if name.startswith(prefix)]
        timestamps: list[float] = []
        rows: list[list[float]] = []
        for row in reader:
            timestamps.append(float(row["t"]))
            rows.append([float(row[name]) for name in fields])
    if not rows:
        return None
    return np.asarray(timestamps), np.asarray(rows)


def _dry_run_summary(
    action_path: Path | None,
    target_path: Path | None,
) -> dict[str, object] | None:
    actions = _read_numeric_csv(action_path, "a_il_")
    targets = _read_numeric_csv(target_path, "target_")
    if actions is None and targets is None:
        return None
    result: dict[str, object] = {}
    if actions is not None:
        action_t, action = actions
        abs_action = np.abs(action)
        result["action_frames"] = int(len(action))
        result["action_abs_max"] = float(np.max(abs_action))
        result["action_abs_p99"] = _percentile(abs_action, 99.0)
        result["action_ticks_at_clip_20"] = int(np.count_nonzero(np.max(abs_action, axis=1) >= 20.0))
        per_joint_peak = np.max(abs_action, axis=0)
        result["action_peak_joints"] = _top_joints(per_joint_peak)
        first_five = action[action_t - action_t[0] <= 5.0]
        result["first_5s_action_abs_max"] = float(np.max(np.abs(first_five)))
    if targets is not None:
        target_t, target = targets
        delta = target - reference.DEFAULT_ANGLES[None, :]
        first_five = delta[target_t - target_t[0] <= 5.0]
        result["first_5s_target_delta_abs_max_rad"] = float(np.max(np.abs(first_five)))
        result["first_5s_target_delta_peak_joints"] = _top_joints(
            np.max(np.abs(first_five), axis=0)
        )
    return result


def analyze(
    capture: Capture,
    control_entry_s: float | None,
    robot_yaw_deg: float | None,
    pre_entry_window_s: float,
    blend_seconds: float,
    action_path: Path | None,
    target_path: Path | None,
) -> dict[str, object]:
    timestamps = capture.timestamp_s
    gaps = np.diff(timestamps)
    source_hz = float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))

    if control_entry_s is None:
        entry_index = len(timestamps) - 1
        entry_source = "last capture frame (no deploy entry timestamp)"
    else:
        candidates = np.flatnonzero(timestamps <= control_entry_s)
        entry_index = int(candidates[-1]) if len(candidates) else 0
        entry_source = "last HMCP frame at or before deploy CONTROL entry"
    entry_timestamp = float(timestamps[entry_index])
    start_timestamp = entry_timestamp - pre_entry_window_s
    window_indices = np.flatnonzero(
        (timestamps >= start_timestamp) & (timestamps <= entry_timestamp)
    )
    if len(window_indices) < 2:
        raise ValueError("pre-entry window contains fewer than two frames")

    entry_offset = capture.joint_pos[entry_index] - reference.DEFAULT_ANGLES
    entry_velocity = capture.joint_vel[entry_index]
    entry_rpy = capture.root_rpy_deg[entry_index]
    group_entry: dict[str, object] = {}
    for name, group_indices in GROUPS.items():
        group = np.asarray(group_indices, dtype=np.int64)
        group_entry[name] = {
            "offset_abs_max_rad": float(np.max(np.abs(entry_offset[group]))),
            "velocity_abs_max_rad_s": float(np.max(np.abs(entry_velocity[group]))),
            "blend_induced_velocity_abs_max_rad_s": float(
                np.max(np.abs(entry_offset[group])) / blend_seconds
            ),
        }

    yaw_gap = None
    if robot_yaw_deg is not None:
        yaw_gap = _wrap_deg(entry_rpy[2] - robot_yaw_deg)

    low_motion_indices, low_motion_summary, low_motion_score = find_low_motion_window(
        capture, pre_entry_window_s
    )
    low_motion_entry_index = int(low_motion_indices[-1])
    low_motion_offset = (
        capture.joint_pos[low_motion_entry_index] - reference.DEFAULT_ANGLES
    )
    neutral_window: dict[str, object] | None = None
    try:
        neutral_indices, neutral_summary, neutral_score = find_low_motion_window(
            capture,
            pre_entry_window_s,
            require_neutral_envelope=True,
        )
        neutral_end = int(neutral_indices[-1])
        neutral_offset = capture.joint_pos[neutral_end] - reference.DEFAULT_ANGLES
        neutral_window = {
            **neutral_summary,
            "score_for_ranking_only": neutral_score,
            "start_sequence": int(capture.sequence[int(neutral_indices[0])]),
            "end_sequence": int(capture.sequence[neutral_end]),
            "start_timestamp_s": float(
                capture.timestamp_s[int(neutral_indices[0])]
            ),
            "end_timestamp_s": float(capture.timestamp_s[neutral_end]),
            "top_joint_offsets_at_end": _top_joints(neutral_offset),
            "root_rpy_at_end_deg": [
                float(v) for v in capture.root_rpy_deg[neutral_end]
            ],
            "offset_limits_rad": NEUTRAL_REFERENCE_OFFSET_LIMITS_RAD,
        }
    except ValueError:
        neutral_window = None

    return {
        "schema": "x2-garment-entry-analysis-v1",
        "capture": {
            "frames": int(len(timestamps)),
            "duration_s": float(timestamps[-1] - timestamps[0]),
            "source_hz": source_hz,
            "gap_p95_ms": _percentile(gaps * 1000.0, 95.0),
            "gap_max_ms": float(np.max(gaps) * 1000.0),
            "derivative_resets": capture.derivative_resets,
        },
        "entry": {
            "source": entry_source,
            "frame_index": entry_index,
            "sequence": int(capture.sequence[entry_index]),
            "timestamp_s": entry_timestamp,
            "deploy_control_entry_s": control_entry_s,
            "robot_yaw_deg": robot_yaw_deg,
            "reference_root_roll_pitch_yaw_deg": [float(v) for v in entry_rpy],
            "uncorrected_reference_minus_robot_yaw_deg": yaw_gap,
            "groups": group_entry,
            "top_joint_offsets": _top_joints(entry_offset),
            "top_joint_velocities": _top_joints(entry_velocity),
            "blend_seconds": blend_seconds,
        },
        "pre_entry_window": _window_summary(capture, window_indices),
        "lowest_motion_window": {
            **low_motion_summary,
            "score_for_ranking_only": low_motion_score,
            "start_sequence": int(capture.sequence[int(low_motion_indices[0])]),
            "end_sequence": int(capture.sequence[low_motion_entry_index]),
            "start_timestamp_s": float(
                capture.timestamp_s[int(low_motion_indices[0])]
            ),
            "end_timestamp_s": float(capture.timestamp_s[low_motion_entry_index]),
            "top_joint_offsets_at_end": _top_joints(low_motion_offset),
            "root_rpy_at_end_deg": [
                float(v) for v in capture.root_rpy_deg[low_motion_entry_index]
            ],
        },
        "lowest_motion_neutral_window": neutral_window,
        "dry_run": _dry_run_summary(action_path, target_path),
        "interpretation": {
            "yaw_alignment_required": (
                yaw_gap is not None and abs(yaw_gap) > 10.0
            ),
            "note": (
                "Dry-run robot joints remained under official MC; action/target "
                "statistics are interface evidence, not powered tracking evidence."
            ),
        },
    }


def _format_top_joints(items: Iterable[dict[str, float | str]]) -> str:
    return ", ".join(f"{item['joint']}={float(item['value']):+.3f}" for item in items)


def format_report(result: dict[str, object]) -> str:
    capture = result["capture"]
    entry = result["entry"]
    window = result["pre_entry_window"]
    lines = [
        "# X2 Garment Entry Analysis",
        "",
        "Offline only; no ROS, ZMQ, MC, or HAL process was started.",
        "",
        "## Source",
        "",
        f"- Frames: `{capture['frames']}` over `{capture['duration_s']:.3f} s`",
        f"- Rate: `{capture['source_hz']:.2f} Hz`",
        f"- Gap p95/max: `{capture['gap_p95_ms']:.1f}/{capture['gap_max_ms']:.1f} ms`",
        f"- Timestamp derivative resets: `{capture['derivative_resets']}`",
        "",
        "## Candidate Entry",
        "",
        f"- Selection: {entry['source']}",
        f"- HMCP sequence: `{entry['sequence']}`",
        "- Reference root roll/pitch/yaw: "
        f"`{entry['reference_root_roll_pitch_yaw_deg'][0]:+.2f}/"
        f"{entry['reference_root_roll_pitch_yaw_deg'][1]:+.2f}/"
        f"{entry['reference_root_roll_pitch_yaw_deg'][2]:+.2f} deg`",
    ]
    if entry["robot_yaw_deg"] is not None:
        lines += [
            f"- Robot yaw from deploy log: `{entry['robot_yaw_deg']:+.2f} deg`",
            "- Uncorrected reference-minus-robot yaw: "
            f"`{entry['uncorrected_reference_minus_robot_yaw_deg']:+.2f} deg`",
        ]
    lines += [
        "- Largest StandStill joint offsets: " + _format_top_joints(entry["top_joint_offsets"]),
        "- Largest instantaneous reference velocities: "
        + _format_top_joints(entry["top_joint_velocities"]),
        "",
        f"## Pre-entry Window ({window['duration_s']:.3f} s / {window['frames']} frames)",
        "",
        "| Group | Offset p95/max rad | Velocity p95/max rad/s | Range max rad | Largest-range joint |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name in ("leg", "waist", "arm", "head"):
        item = window["groups"][name]
        lines.append(
            f"| {name} | {item['offset_abs_p95_rad']:.3f}/{item['offset_abs_max_rad']:.3f} "
            f"| {item['velocity_abs_p95_rad_s']:.3f}/{item['velocity_abs_max_rad_s']:.3f} "
            f"| {item['range_max_rad']:.3f} | {item['range_max_joint']} |"
        )
    root = window["root"]
    lines += [
        "",
        "Root roll/pitch/yaw ranges in the window: "
        f"`{root['roll_range_deg']:.2f}/{root['pitch_range_deg']:.2f}/"
        f"{root['yaw_range_deg']:.2f} deg`.",
        "",
        "## Lowest-motion Observed Window",
        "",
    ]
    low = result["lowest_motion_window"]
    lines += [
        f"Sequences `{low['start_sequence']}..{low['end_sequence']}` over "
        f"`{low['duration_s']:.3f} s` were the lowest-motion complete window "
        "in this capture. This is a ranking result, not a powered-safe label.",
        "",
        "| Group | Offset p95/max rad | Velocity p95/max rad/s | Range max rad |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("leg", "waist", "arm", "head"):
        item = low["groups"][name]
        lines.append(
            f"| {name} | {item['offset_abs_p95_rad']:.3f}/{item['offset_abs_max_rad']:.3f} "
            f"| {item['velocity_abs_p95_rad_s']:.3f}/{item['velocity_abs_max_rad_s']:.3f} "
            f"| {item['range_max_rad']:.3f} |"
        )
    low_root = low["root"]
    lines += [
        "",
        "Lowest-motion root roll/pitch/yaw ranges: "
        f"`{low_root['roll_range_deg']:.2f}/{low_root['pitch_range_deg']:.2f}/"
        f"{low_root['yaw_range_deg']:.2f} deg`.",
        "",
        "Largest StandStill offsets at the end of that window: "
        + _format_top_joints(low["top_joint_offsets_at_end"]),
        "",
        "## Lowest-motion Neutral-envelope Window",
        "",
    ]
    neutral = result["lowest_motion_neutral_window"]
    if neutral is None:
        lines.append(
            "No complete window also satisfied the staging neutral-reference "
            "offset envelope; this capture cannot arm a stationary-wearer test."
        )
    else:
        lines += [
            f"Sequences `{neutral['start_sequence']}..{neutral['end_sequence']}` "
            f"over `{neutral['duration_s']:.3f} s` were the lowest-motion window "
            "that also rejected the calibration T-Pose.",
            "",
            "| Group | Offset p95/max rad | Velocity p95/max rad/s | Range max rad |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name in ("leg", "waist", "arm", "head"):
            item = neutral["groups"][name]
            lines.append(
                f"| {name} | {item['offset_abs_p95_rad']:.3f}/{item['offset_abs_max_rad']:.3f} "
                f"| {item['velocity_abs_p95_rad_s']:.3f}/{item['velocity_abs_max_rad_s']:.3f} "
                f"| {item['range_max_rad']:.3f} |"
            )
        neutral_root = neutral["root"]
        lines += [
            "",
            "Neutral-window root roll/pitch/yaw ranges: "
            f"`{neutral_root['roll_range_deg']:.2f}/"
            f"{neutral_root['pitch_range_deg']:.2f}/"
            f"{neutral_root['yaw_range_deg']:.2f} deg`.",
            "",
            "Largest StandStill offsets at the end of that window: "
            + _format_top_joints(neutral["top_joint_offsets_at_end"]),
        ]
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if result["interpretation"]["yaw_alignment_required"]:
        lines.append(
            "- The live root yaw is not in the robot entry frame. A single captured "
            "yaw rebase is required before powered use."
        )
    else:
        lines.append("- This sample does not show a large root-yaw entry mismatch.")
    lines.append(
        "- Entry must be gated on a continuous stationary-reference window; frame "
        "freshness alone is insufficient."
    )
    lines.append(
        "- StandStill-to-live must use a bounded blend. The report's induced blend "
        "velocity is joint offset divided by the configured blend duration."
    )
    lines.append(f"- {result['interpretation']['note']}")
    if result["dry_run"] is not None:
        dry = result["dry_run"]
        lines += [
            "",
            "## Dry-run Policy Evidence",
            "",
            f"- Logged action max/p99: `{dry.get('action_abs_max', float('nan')):.3f}/"
            f"{dry.get('action_abs_p99', float('nan')):.3f}`",
            f"- Ticks at the logged `20` action clip: `{dry.get('action_ticks_at_clip_20', 0)}`",
            f"- First-5-second target delta max: "
            f"`{dry.get('first_5s_target_delta_abs_max_rad', float('nan')):.3f} rad`",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--deploy-log", type=Path)
    parser.add_argument("--action-csv", type=Path)
    parser.add_argument("--target-csv", type=Path)
    parser.add_argument("--pre-entry-window-s", type=float, default=2.0)
    parser.add_argument("--blend-seconds", type=float, default=4.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    if args.pre_entry_window_s <= 0.0:
        parser.error("--pre-entry-window-s must be > 0")
    if args.blend_seconds <= 0.0:
        parser.error("--blend-seconds must be > 0")

    capture = load_capture(args.capture)
    control_entry_s, robot_yaw_deg = parse_deploy_log(args.deploy_log)
    result = analyze(
        capture,
        control_entry_s,
        robot_yaw_deg,
        args.pre_entry_window_s,
        args.blend_seconds,
        args.action_csv,
        args.target_csv,
    )
    report = format_report(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
