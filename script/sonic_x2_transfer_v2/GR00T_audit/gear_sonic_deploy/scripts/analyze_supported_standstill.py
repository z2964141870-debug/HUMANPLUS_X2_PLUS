#!/usr/bin/env python3
"""Align and replay supported StandStill X2 hardware logs.

This tool is deliberately offline. It reconstructs the deploy command-shaping
path for the two ankle-pitch joints:

    clipped policy action -> raw position target -> safety soft-start
    -> supported default-centered ramp/envelope -> target LPF -> target slew
    -> logged SafeCommand/HAL target -> measured q/dq

The logger writes the SafeCommand consumed by the 250 Hz writer. The writer
re-publishes that command unchanged, so target_pos.csv is the effective HAL
position command at the 50 Hz policy resolution. It is not an independent bus
capture. Joint effort, motor voltage, temperatures, and protection state were
not logged by these historical runs and must not be inferred here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np


CONTROL_DT = 0.02
SOFT_START_SECONDS = 3.0
SUPPORTED_RAMP_SECONDS = 4.0
TARGET_LPF_HZ = 8.0
MAX_TARGET_DEV = 2.0
LEG_ENVELOPE = 0.60
KP_SCALE_ANKLE = 1.5
KD_SCALE_ANKLE_PITCH = 3.31
ANKLE_MJ = (4, 10)
ANKLE_LABELS = ("left_ankle_pitch", "right_ankle_pitch")


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    slew_rate: float
    note: str
    terminal_action_is_applied: bool = False


@dataclass
class Table:
    header: list[str]
    rows: dict[int, list[float]]
    malformed_rows: int


@dataclass(frozen=True)
class TickRecord:
    ramp_alpha: float
    dry_run: bool
    tilt_trip: bool
    reason: str


def _microseconds(value: float) -> int:
    return int(round(value * 1_000_000.0))


def _read_clean_text(path: Path) -> str:
    # A hard stop left NUL bytes in one historical log. Removing only NUL is
    # sufficient to recover the surrounding complete CSV records.
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")


def read_numeric_csv(path: Path) -> Table:
    reader = csv.reader(io.StringIO(_read_clean_text(path)))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"empty CSV: {path}") from exc
    if not header or header[0] != "t":
        raise ValueError(f"unexpected CSV header in {path}: {header[:3]}")

    rows: dict[int, list[float]] = {}
    malformed = 0
    for fields in reader:
        if len(fields) != len(header):
            malformed += 1
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            malformed += 1
            continue
        rows[_microseconds(values[0])] = values
    return Table(header=header, rows=rows, malformed_rows=malformed)


def read_tick_csv(path: Path) -> tuple[dict[int, TickRecord], int]:
    reader = csv.reader(io.StringIO(_read_clean_text(path)))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"empty CSV: {path}") from exc
    if header != ["t", "ramp_alpha", "dry_run", "tilt_trip", "reason"]:
        raise ValueError(f"unexpected tick header in {path}: {header}")

    rows: dict[int, TickRecord] = {}
    malformed = 0
    for fields in reader:
        if len(fields) != len(header):
            malformed += 1
            continue
        try:
            key = _microseconds(float(fields[0]))
            ramp_alpha = float(fields[1])
            dry_run = bool(int(fields[2]))
            tilt_trip = bool(int(fields[3]))
        except ValueError:
            malformed += 1
            continue
        rows[key] = TickRecord(ramp_alpha, dry_run, tilt_trip, fields[4])
    return rows, malformed


def _strip_cpp_comments(body: str) -> str:
    return re.sub(r"//.*", "", body)


def read_cpp_array(header: Path, name: str, dtype: type = float) -> np.ndarray:
    text = header.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{name}\s*=\s*\{{(?P<body>.*?)\}}\s*;", text, re.DOTALL
    )
    if match is None:
        raise ValueError(f"could not find C++ array {name!r} in {header}")
    body = _strip_cpp_comments(match.group("body"))
    tokens = [token.strip() for token in body.split(",") if token.strip()]
    values = np.asarray([dtype(token) for token in tokens], dtype=dtype)
    if values.shape != (31,):
        raise ValueError(f"{name} has shape {values.shape}, expected (31,)")
    return values


def smoothstep(elapsed_s: np.ndarray, duration_s: float) -> np.ndarray:
    u = np.clip(elapsed_s / duration_s, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def gravity_from_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.where(norm > 0.0, norm, 1.0)
    w, x, y, z = quat.T
    # R(q)^T @ [0, 0, -1], matching body_frame_gravity_from_quat_wxyz.
    return np.column_stack(
        (
            2.0 * (w * y - x * z),
            -2.0 * (w * x + y * z),
            -1.0 + 2.0 * (x * x + y * y),
        )
    )


def pelvis_angles_deg(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gravity = gravity_from_quaternion_wxyz(quat)
    pitch = np.degrees(np.arcsin(np.clip(gravity[:, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(-gravity[:, 1], -gravity[:, 2]))
    tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
    return roll, pitch, tilt


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_row_before(table: Table, key: int) -> list[float]:
    candidates = [candidate for candidate in table.rows if candidate < key]
    if not candidates:
        raise ValueError("no static command row before supported policy")
    return table.rows[max(candidates)]


def _first_last_window_abs_max(values: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    elapsed = t - t[0]
    first = np.abs(values[elapsed <= min(1.0, elapsed[-1])])
    last = np.abs(values[elapsed >= max(0.0, elapsed[-1] - 1.0)])
    return float(np.max(first)), float(np.max(last))


def analyze_run(
    spec: RunSpec,
    constants: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, object]:
    tables = {
        name: read_numeric_csv(spec.path / f"{name}.csv")
        for name in ("action_il", "target_pos", "joint_pos", "joint_vel", "imu")
    }
    ticks, tick_malformed = read_tick_csv(spec.path / "tick.csv")
    policy_keys = {
        key for key, record in ticks.items()
        if record.reason == "supported_policy_probe"
    }
    aligned_keys = sorted(
        policy_keys.intersection(*(set(table.rows) for table in tables.values()))
    )
    if len(aligned_keys) < 2:
        raise ValueError(f"{spec.label}: fewer than two aligned policy records")

    def values(name: str) -> np.ndarray:
        return np.asarray([tables[name].rows[key] for key in aligned_keys], dtype=np.float64)

    action = values("action_il")[:, 1:]
    logged_target = values("target_pos")[:, 1:]
    measured_q = values("joint_pos")[:, 1:]
    measured_dq = values("joint_vel")[:, 1:]
    imu = values("imu")
    time_s = np.asarray(aligned_keys, dtype=np.float64) / 1_000_000.0
    elapsed_s = time_s - time_s[0]
    dt = np.diff(time_s, prepend=time_s[0])
    action_is_raw = np.ones(len(aligned_keys), dtype=bool)
    if spec.terminal_action_is_applied:
        # The early-return branch logs last_action_il_ together with the last
        # published SafeCommand. That terminal row preserves the trip state,
        # but its action is AppliedActionFromTarget, not a fresh ONNX output.
        action_is_raw[-1] = False

    default = constants["default_angles"]
    action_scale = constants["x2_action_scale"]
    mj_to_il = constants["mujoco_to_isaaclab"].astype(int)
    raw_target = default + action[:, mj_to_il] * action_scale

    # Replay the exact deploy ordering in x2_deploy_onnx_ref.cpp:
    # ApplySafetyStack's linear soft start and global clamp, then supported
    # default-centered smoothstep/envelope, then LPF, then target slew.
    soft_alpha = np.clip(elapsed_s / SOFT_START_SECONDS, 0.0, 1.0)
    safety_target = default + soft_alpha[:, None] * (raw_target - default)
    safety_target = np.clip(
        safety_target, default - MAX_TARGET_DEV, default + MAX_TARGET_DEV
    )
    supported_alpha = np.asarray(
        [ticks[key].ramp_alpha for key in aligned_keys], dtype=np.float64
    )
    bounded = default + np.clip(safety_target - default, -LEG_ENVELOPE, LEG_ENVELOPE)

    static_target = np.asarray(
        _latest_row_before(tables["target_pos"], aligned_keys[0])[1:],
        dtype=np.float64,
    )
    post_ramp = static_target + supported_alpha[:, None] * (bounded - static_target)

    # No shaping tick ran for an early-return terminal snapshot; every command
    # field is the prior published command. Carry the prior internal state so
    # the replay remains aligned while keeping raw-action statistics masked.
    if not action_is_raw[-1]:
        safety_target[-1] = safety_target[-2]
        bounded[-1] = bounded[-2]
        post_ramp[-1] = post_ramp[-2]

    lpf_alpha = 1.0 - math.exp(-2.0 * math.pi * TARGET_LPF_HZ * CONTROL_DT)
    post_lpf = np.empty_like(post_ramp)
    post_lpf[0] = post_ramp[0]
    for index in range(1, len(post_lpf)):
        if action_is_raw[index]:
            post_lpf[index] = (
                lpf_alpha * post_ramp[index]
                + (1.0 - lpf_alpha) * post_lpf[index - 1]
            )
        else:
            post_lpf[index] = post_lpf[index - 1]

    replay_target = np.empty_like(post_lpf)
    slew_limited = np.zeros_like(post_lpf, dtype=bool)
    replay_target[0] = post_lpf[0]
    previous = static_target.copy()
    for index in range(1, len(replay_target)):
        if not action_is_raw[index]:
            replay_target[index] = replay_target[index - 1]
            previous = replay_target[index]
            continue
        elapsed = float(np.clip(dt[index], 0.0, 0.1))
        max_step = spec.slew_rate * elapsed
        delta = post_lpf[index] - previous
        slew_limited[index] = np.abs(delta) > max_step + 1e-10
        previous = previous + np.clip(delta, -max_step, max_step)
        replay_target[index] = previous

    replay_error = replay_target - logged_target
    roll_deg, pitch_deg, tilt_deg = pelvis_angles_deg(imu[:, 1:5])
    first_tilt, last_tilt = _first_last_window_abs_max(tilt_deg, time_s)
    speed_peak_sample, speed_peak_joint = np.unravel_index(
        int(np.argmax(np.abs(measured_dq))), measured_dq.shape
    )
    velocity_names = tables["joint_vel"].header[1:]
    following_tick = next(
        (
            ticks[key]
            for key in sorted(ticks)
            if key > aligned_keys[-1]
        ),
        None,
    )

    kps = constants["kps"]
    kds = constants["kds"]
    effective_kp = kps * KP_SCALE_ANKLE
    effective_kd = kds * KD_SCALE_ANKLE_PITCH

    run_result: dict[str, object] = {
        "label": spec.label,
        "path": str(spec.path),
        "note": spec.note,
        "slew_rate_rad_s": spec.slew_rate,
        "aligned_policy_samples": len(aligned_keys),
        "policy_duration_s": float(time_s[-1] - time_s[0]),
        "policy_start_monotonic_s": float(time_s[0]),
        "policy_end_monotonic_s": float(time_s[-1]),
        "sample_period_median_s": percentile(np.diff(time_s), 50),
        "sample_period_p95_s": percentile(np.diff(time_s), 95),
        "malformed_rows": {
            **{name: table.malformed_rows for name, table in tables.items()},
            "tick": tick_malformed,
        },
        "terminal_action_semantics": (
            "applied_action_snapshot" if spec.terminal_action_is_applied
            else "raw_policy_action"
        ),
        "replay_residual_max_rad_all_joints": float(np.max(np.abs(replay_error))),
        "replay_residual_p95_rad_all_joints": percentile(np.abs(replay_error), 95),
        "pelvis": {
            "entry_roll_deg": float(roll_deg[0]),
            "entry_pitch_deg": float(pitch_deg[0]),
            "entry_tilt_deg": float(tilt_deg[0]),
            "peak_abs_roll_deg": float(np.max(np.abs(roll_deg))),
            "peak_abs_pitch_deg": float(np.max(np.abs(pitch_deg))),
            "peak_tilt_deg": float(np.max(tilt_deg)),
            "end_roll_deg": float(roll_deg[-1]),
            "end_pitch_deg": float(pitch_deg[-1]),
            "end_tilt_deg": float(tilt_deg[-1]),
            "first_1s_tilt_max_deg": first_tilt,
            "last_1s_tilt_max_deg": last_tilt,
        },
        "software_safety": {
            "policy_peak_abs_joint_speed_rad_s": float(
                np.abs(measured_dq[speed_peak_sample, speed_peak_joint])
            ),
            "policy_peak_speed_joint": velocity_names[speed_peak_joint],
            "policy_peak_speed_t_s": float(elapsed_s[speed_peak_sample]),
            "terminal_policy_tick_tilt_trip": bool(
                ticks[aligned_keys[-1]].tilt_trip
            ),
            "next_logged_state": (
                following_tick.reason if following_tick is not None else "missing"
            ),
        },
        "ankles": {},
    }

    trace_columns: list[tuple[str, np.ndarray]] = [
        ("t", time_s),
        ("policy_t", elapsed_s),
        ("pelvis_roll_deg", roll_deg),
        ("pelvis_pitch_deg", pitch_deg),
        ("pelvis_tilt_deg", tilt_deg),
        ("raw_policy_action_valid", action_is_raw.astype(float)),
    ]

    for label, mj in zip(ANKLE_LABELS, ANKLE_MJ):
        command = logged_target[:, mj]
        q = measured_q[:, mj]
        dq = measured_dq[:, mj]
        tracking_error = command - q
        implied_pd_torque = effective_kp[mj] * tracking_error - effective_kd[mj] * dq
        raw_backlog = raw_target[:, mj] - command
        shaped_backlog = post_lpf[:, mj] - command
        peak_index = int(np.argmax(np.abs(tracking_error)))
        first_track, last_track = _first_last_window_abs_max(
            tracking_error, time_s
        )

        valid = action_is_raw
        ankle_result = {
            "effective_kp_nm_rad": float(effective_kp[mj]),
            "effective_kd_nm_s_rad": float(effective_kd[mj]),
            "static_target_rad": float(static_target[mj]),
            "entry_measured_q_rad": float(q[0]),
            "entry_tracking_error_rad": float(tracking_error[0]),
            "raw_target_min_rad": float(np.min(raw_target[valid, mj])),
            "raw_target_max_rad": float(np.max(raw_target[valid, mj])),
            "post_ramp_min_rad": float(np.min(post_ramp[valid, mj])),
            "post_ramp_max_rad": float(np.max(post_ramp[valid, mj])),
            "hal_target_min_rad": float(np.min(command)),
            "hal_target_max_rad": float(np.max(command)),
            "raw_to_hal_backlog_abs_max_rad": float(
                np.max(np.abs(raw_backlog[valid]))
            ),
            "post_lpf_to_hal_backlog_abs_max_rad": float(
                np.max(np.abs(shaped_backlog[valid]))
            ),
            "post_lpf_to_hal_backlog_positive_max_rad": float(
                np.max(shaped_backlog[valid])
            ),
            "slew_limited_fraction": float(np.mean(slew_limited[valid, mj])),
            "measured_q_min_rad": float(np.min(q)),
            "measured_q_max_rad": float(np.max(q)),
            "measured_dq_abs_max_rad_s": float(np.max(np.abs(dq))),
            "tracking_error_abs_max_rad": float(np.max(np.abs(tracking_error))),
            "tracking_error_rms_rad": float(np.sqrt(np.mean(tracking_error**2))),
            "tracking_error_peak_policy_t_s": float(elapsed_s[peak_index]),
            "tracking_error_end_rad": float(tracking_error[-1]),
            "tracking_error_first_1s_abs_max_rad": first_track,
            "tracking_error_last_1s_abs_max_rad": last_track,
            "implied_pd_torque_abs_max_nm": float(np.max(np.abs(implied_pd_torque))),
            "replay_residual_abs_max_rad": float(np.max(np.abs(replay_error[:, mj]))),
        }
        run_result["ankles"][label] = ankle_result

        prefix = "left" if mj == ANKLE_MJ[0] else "right"
        trace_columns.extend(
            [
                (f"{prefix}_raw_target", raw_target[:, mj]),
                (f"{prefix}_post_ramp_target", post_ramp[:, mj]),
                (f"{prefix}_post_lpf_target", post_lpf[:, mj]),
                (f"{prefix}_replay_hal_target", replay_target[:, mj]),
                (f"{prefix}_logged_hal_target", command),
                (f"{prefix}_measured_q", q),
                (f"{prefix}_measured_dq", dq),
                (f"{prefix}_tracking_error", tracking_error),
                (f"{prefix}_implied_pd_torque_nm", implied_pd_torque),
                (f"{prefix}_slew_limited", slew_limited[:, mj].astype(float)),
            ]
        )

    trace_path = output_dir / f"{spec.label}_ankle_trace.csv"
    with trace_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([name for name, _ in trace_columns])
        for row in zip(*(column for _, column in trace_columns)):
            writer.writerow([f"{float(value):.9f}" for value in row])
    run_result["trace_csv"] = str(trace_path)
    return run_result


def render_markdown(results: list[dict[str, object]], metadata: dict[str, object]) -> str:
    lines = [
        "# Fixed StandStill Supported Log Analysis",
        "",
        "Generated by `scripts/analyze_supported_standstill.py`.",
        "",
        "## Scope and telemetry semantics",
        "",
        "- Fixed StandStill reference only; no garment, HMCP, offload, or ZMQ input.",
            "- `action_il.csv` is post-action-clip policy output in IsaacLab order on normal policy ticks.",
            "- On an early tilt-return snapshot, the terminal row stores the previous applied action rather than a fresh ONNX action; the analyzer marks and excludes that row from raw-action shaping statistics.",
        "- `target_pos.csv` is the final `SafeCommand.target_pos_mj`; the 250 Hz writer republishes it unchanged.",
        "- The reconstructed replay is: raw target -> 3 s linear safety ramp -> 4 s supported smoothstep/envelope -> 8 Hz LPF -> configured slew.",
        "- Historical logs do not contain measured effort/current, motor voltage, temperature, per-tick gains, foot force, gantry force, or hardware protection state.",
        "- Implied PD torque is diagnostic only: `kp * (target-q) - kd*dq`; it is not measured torque.",
        "",
        "## Run comparison",
        "",
        "| Run | Slew | Samples / duration | Entry roll/pitch | Entry ankle error L/R | Slew-limited L/R | Peak tracking error L/R | Peak ankle speed L/R | Full-body speed peak | Replay residual L/R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        pelvis = result["pelvis"]
        safety = result["software_safety"]
        left = result["ankles"]["left_ankle_pitch"]
        right = result["ankles"]["right_ankle_pitch"]
        lines.append(
            "| {label} | {slew:.2f} | {samples} / {duration:.3f}s | "
            "{roll:+.2f}/{pitch:+.2f}deg | {le:+.3f}/{re:+.3f} | "
            "{ls:.1%}/{rs:.1%} | {lt:.3f}/{rt:.3f} | "
            "{lv:.3f}/{rv:.3f} | {sv:.3f} ({sj}) | "
            "{lr:.6f}/{rr:.6f} |".format(
                label=result["label"],
                slew=result["slew_rate_rad_s"],
                samples=result["aligned_policy_samples"],
                duration=result["policy_duration_s"],
                roll=pelvis["entry_roll_deg"],
                pitch=pelvis["entry_pitch_deg"],
                le=left["entry_tracking_error_rad"],
                re=right["entry_tracking_error_rad"],
                ls=left["slew_limited_fraction"],
                rs=right["slew_limited_fraction"],
                lt=left["tracking_error_abs_max_rad"],
                rt=right["tracking_error_abs_max_rad"],
                lv=left["measured_dq_abs_max_rad_s"],
                rv=right["measured_dq_abs_max_rad_s"],
                sv=safety["policy_peak_abs_joint_speed_rad_s"],
                sj=safety["policy_peak_speed_joint"].removeprefix("dq_"),
                lr=left["replay_residual_abs_max_rad"],
                rr=right["replay_residual_abs_max_rad"],
            )
        )

    lines.extend(
        [
            "",
            "## Growth diagnostic",
            "",
            "The acceptance gate rejects growth, not merely a large isolated value.",
            "",
            "| Run | Tilt max first -> last 1 s | Left tracking max first -> last 1 s | Right tracking max first -> last 1 s |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        pelvis = result["pelvis"]
        left = result["ankles"]["left_ankle_pitch"]
        right = result["ankles"]["right_ankle_pitch"]
        lines.append(
            "| {label} | {tf:.2f} -> {tl:.2f} deg | "
            "{lf:.3f} -> {ll:.3f} rad | {rf:.3f} -> {rl:.3f} rad |".format(
                label=result["label"],
                tf=pelvis["first_1s_tilt_max_deg"],
                tl=pelvis["last_1s_tilt_max_deg"],
                lf=left["tracking_error_first_1s_abs_max_rad"],
                ll=left["tracking_error_last_1s_abs_max_rad"],
                rf=right["tracking_error_first_1s_abs_max_rad"],
                rl=right["tracking_error_last_1s_abs_max_rad"],
            )
        )

    lines.extend(
        [
            "",
            "## Software safety evidence",
            "",
            "All three logs transition from the policy state into `supported_policy_return`. The CSV does not encode the initiating return reason, and its terminal policy tick has `tilt_trip=false`; console observations identify the two short runs as tilt returns. Hardware overcurrent, torque limiting, and motor protection were not logged.",
            "",
        ]
    )
    for result in results:
        safety = result["software_safety"]
        lines.append(
            "- `{label}`: policy peak `{speed:.3f} rad/s` at `{joint}` "
            "(`t={time:.3f}s`); next logged state `{state}`.".format(
                label=result["label"],
                speed=safety["policy_peak_abs_joint_speed_rad_s"],
                joint=safety["policy_peak_speed_joint"],
                time=safety["policy_peak_speed_t_s"],
                state=safety["next_logged_state"],
            )
        )
    lines.extend(["", "## Per-run ankle chain", ""])
    for result in results:
        lines.extend(
            [
                f"### {result['label']}",
                "",
                str(result["note"]),
                "",
                "| Joint | Raw target range | Post-ramp range | HAL target range | LPF-to-HAL backlog max | Tracking RMS / peak | Implied PD torque peak |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in ANKLE_LABELS:
            ankle = result["ankles"][name]
            lines.append(
                "| {name} | [{raw_min:+.3f}, {raw_max:+.3f}] | "
                "[{ramp_min:+.3f}, {ramp_max:+.3f}] | "
                "[{hal_min:+.3f}, {hal_max:+.3f}] | {backlog:.3f} | "
                "{rms:.3f} / {peak:.3f} rad | {torque:.1f} Nm |".format(
                    name=name,
                    raw_min=ankle["raw_target_min_rad"],
                    raw_max=ankle["raw_target_max_rad"],
                    ramp_min=ankle["post_ramp_min_rad"],
                    ramp_max=ankle["post_ramp_max_rad"],
                    hal_min=ankle["hal_target_min_rad"],
                    hal_max=ankle["hal_target_max_rad"],
                    backlog=ankle["post_lpf_to_hal_backlog_abs_max_rad"],
                    rms=ankle["tracking_error_rms_rad"],
                    peak=ankle["tracking_error_abs_max_rad"],
                    torque=ankle["implied_pd_torque_abs_max_nm"],
                )
            )
        pelvis = result["pelvis"]
        lines.extend(
            [
                "",
                "Pelvis: entry roll/pitch/tilt "
                f"`{pelvis['entry_roll_deg']:+.2f}/{pelvis['entry_pitch_deg']:+.2f}/{pelvis['entry_tilt_deg']:.2f} deg`; "
                "peak absolute roll/pitch and tilt "
                f"`{pelvis['peak_abs_roll_deg']:.2f}/{pelvis['peak_abs_pitch_deg']:.2f}/{pelvis['peak_tilt_deg']:.2f} deg`.",
                "",
                f"Trace: `{Path(result['trace_csv']).name}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Offline attribution",
            "",
            "1. **Control-output limiting is proven in the 0.12 run.** The replay quantifies how often ankle commands are slew-limited and how much shaped target remains queued behind the published HAL command.",
            "2. **Increasing slew to 0.30 did not solve the physical response.** The 0.30 run reached a larger HAL command but developed large command-to-measured tracking error and pelvis tilt. Further blind slew increases are rejected.",
            "3. **A simple pelvis entry-angle gate is insufficient.** The 0.30 run entered inside the temporary pitch/roll window and still failed. The 300 s run also began with materially smaller ankle tracking error, so entry load/contact state matters beyond roll/pitch.",
            "4. **PD authority versus contact cannot be separated from these logs.** The diagnostic PD torque remains below the checkpoint effort limit, but actual effort/current and foot/gantry forces were not captured. A large target-to-q error can be compliance, contact geometry, load, inner-loop gain realization, or torque protection.",
            "5. **Gross joint ordering is guarded, but zero/reference alignment remains open.** Runtime joint names are validated against the MuJoCo order. Persistent unloaded command-to-q error would implicate zero/gain/actuation; error appearing only under load would implicate authority/contact. Historical CSVs do not label unloaded versus loaded support strongly enough to close this split.",
            "",
            "## Decision before the next powered run",
            "",
            "Do not change target slew. First add read-only telemetry capture for state effort, motor voltage, temperatures, effective gains, controller protection/reason, and operator-annotated support/contact state. Then replay the exact 0.12 parent in MuJoCo with deploy-order shaping and compare an unloaded versus loaded fixed-reference response. Only a frozen configuration that passes offline replay may proceed to three consecutive supported 30 s probes.",
            "",
            "## Reproducibility",
            "",
            f"Policy parameter header SHA-256: `{metadata['policy_header_sha256']}`",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result['label']}`: `{result['path']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root", type=Path, default=root / "analysis_logs",
        help="Directory containing the three copied hardware logs.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "analysis_reports" / "standstill_20260830",
    )
    parser.add_argument(
        "--policy-header", type=Path,
        default=root / "src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = [
        RunSpec(
            "success_300s",
            args.logs_root / "suspended_sonic_20260828_170450",
            0.12,
            "Historical 300 s supported pass; strong/long gantry support and old raw-action feedback semantics.",
        ),
        RunSpec(
            "failure_slew012",
            args.logs_root / "suspended_sonic_20260829_210922",
            0.12,
            "Corrected applied-action feedback; returned on tilt after about 2.96 s.",
            True,
        ),
        RunSpec(
            "failure_slew030",
            args.logs_root / "suspended_sonic_pdhot_20260829_222159",
            0.30,
            "Same fixed reference and controller family with 0.30 rad/s slew; returned on tilt after about 3.2 s.",
            True,
        ),
    ]
    for spec in specs:
        if not spec.path.is_dir():
            raise SystemExit(f"missing log directory: {spec.path}")

    constants = {
        name: read_cpp_array(args.policy_header, name, int if name == "mujoco_to_isaaclab" else float)
        for name in (
            "default_angles", "x2_action_scale", "mujoco_to_isaaclab", "kps", "kds"
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [analyze_run(spec, constants, args.output_dir) for spec in specs]
    metadata = {
        "policy_header": str(args.policy_header),
        "policy_header_sha256": _sha256(args.policy_header),
        "control_dt_s": CONTROL_DT,
        "soft_start_seconds": SOFT_START_SECONDS,
        "supported_ramp_seconds": SUPPORTED_RAMP_SECONDS,
        "target_lpf_hz": TARGET_LPF_HZ,
        "max_target_dev_rad": MAX_TARGET_DEV,
        "leg_envelope_rad": LEG_ENVELOPE,
        "kp_scale_ankle": KP_SCALE_ANKLE,
        "kd_scale_ankle_pitch": KD_SCALE_ANKLE_PITCH,
    }
    payload = {"metadata": metadata, "runs": results}
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        render_markdown(results, metadata), encoding="utf-8"
    )
    print(args.output_dir / "README.md")
    for result in results:
        ankle_residual = max(
            result["ankles"][name]["replay_residual_abs_max_rad"]
            for name in ANKLE_LABELS
        )
        print(
            f"{result['label']}: {result['aligned_policy_samples']} samples, "
            f"{result['policy_duration_s']:.3f}s, ankle replay max residual "
            f"{ankle_residual:.6f}rad"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
