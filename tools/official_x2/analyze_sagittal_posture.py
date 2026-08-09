#!/usr/bin/env python3
"""Pure, offline sagittal-posture audit for official-X2 traces.

The adapter logs pelvis/root projected gravity, joint position relative to its
selected default pose, and the final issued normalized lower-body action.  This
module deliberately keeps those quantities separate.  In particular, the
issued action is not relabelled as the actor's raw output because supervisors,
templates, clipping, and controller handoffs may already have modified it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
LOWER_SCALE_RAD = np.asarray(
    [0.4, 0.4, 0.4, 0.4, 0.12, 0.08, 0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
     0.4, 0.16, 0.16],
    dtype=np.float64,
)

ROOT_GRAVITY_OBS_SLICE = slice(6, 9)
JOINT_POS_OBS_OFFSET = 12
WAIST_PITCH_ISAAC_INDEX = ISAAC_JOINTS.index("waist_pitch_joint")
WAIST_PITCH_LOWER_INDEX = LOWER_JOINTS.index("waist_pitch_joint")

SAGITTAL_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "waist_pitch_joint",
)


def _sagittal_default_pose(profile: str) -> dict[str, float]:
    if profile == "stage208":
        leg = {"hip": -0.248, "knee": 0.5303, "ankle": -0.2823}
    elif profile == "official_v1":
        leg = {"hip": -0.312, "knee": 0.669, "ankle": -0.363}
    else:
        raise ValueError(f"unsupported default pose profile: {profile}")
    return {
        "left_hip_pitch_joint": leg["hip"],
        "right_hip_pitch_joint": leg["hip"],
        "left_knee_joint": leg["knee"],
        "right_knee_joint": leg["knee"],
        "left_ankle_pitch_joint": leg["ankle"],
        "right_ankle_pitch_joint": leg["ankle"],
        "waist_pitch_joint": 0.0,
    }


def signed_pitch_from_projected_gravity_rad(gravity: Iterable[float]) -> float:
    """Return signed ZYX pelvis/root pitch from R^T [0, 0, -1].

    This exactly matches the adapter observation convention.  With X2's +X
    forward convention, negative is backward lean and positive is forward lean.
    """
    gx, _gy, _gz = np.asarray(tuple(gravity), dtype=np.float64)
    return math.asin(float(np.clip(gx, -1.0, 1.0)))


def projected_gravity_from_xyzw(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Apply the adapter's quaternion-to-body-gravity formula."""
    x, y, z, w = np.asarray(tuple(quaternion_xyzw), dtype=np.float64)
    return np.asarray(
        [
            2.0 * (-z * x + w * y),
            -2.0 * (z * y + w * x),
            1.0 - 2.0 * (w * w + z * z),
        ],
        dtype=np.float64,
    )


def signed_pitch_from_xyzw_rad(quaternion_xyzw: Iterable[float]) -> float:
    return signed_pitch_from_projected_gravity_rad(
        projected_gravity_from_xyzw(quaternion_xyzw)
    )


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def split_trace_phases(
    rows: Sequence[Mapping[str, object]], *, startup_seconds: float
) -> dict[str, list[Mapping[str, object]]]:
    """Split labeled adapter rows into stand/start/move/stop.

    ``start`` is the first pre-registered startup window within ``move``;
    ``move`` is the remainder.  Prepare rows are intentionally excluded.
    """
    if startup_seconds <= 0.0:
        raise ValueError("startup_seconds must be positive")
    move_rows = [row for row in rows if row.get("stage") == "move"]
    return {
        "stand": [row for row in rows if row.get("stage") == "stand"],
        "start": [
            row for row in move_rows
            if float(row.get("elapsed_s", 0.0)) < startup_seconds
        ],
        "move": [
            row for row in move_rows
            if float(row.get("elapsed_s", 0.0)) >= startup_seconds
        ],
        "stop": [row for row in rows if row.get("stage") == "stop"],
    }


def _valid_policy_row(row: Mapping[str, object]) -> bool:
    return (
        len(row.get("obs", [])) >= JOINT_POS_OBS_OFFSET + len(ISAAC_JOINTS)
        and len(row.get("action", [])) >= len(LOWER_JOINTS)
    )


def summarize_trace_phase(
    rows: Sequence[Mapping[str, object]], *, default_pose_profile: str
) -> dict[str, object]:
    valid = [row for row in rows if _valid_policy_row(row)]
    default_pose = _sagittal_default_pose(default_pose_profile)
    default_waist = default_pose["waist_pitch_joint"]
    root_pitch_deg: list[float] = []
    waist_actual_deg: list[float] = []
    waist_issued_target_deg: list[float] = []
    waist_tracking_error_deg: list[float] = []
    sagittal_actual: dict[str, list[float]] = {name: [] for name in SAGITTAL_JOINTS}
    sagittal_target: dict[str, list[float]] = {name: [] for name in SAGITTAL_JOINTS}
    for row in valid:
        obs = np.asarray(row["obs"], dtype=np.float64)
        action = np.asarray(row["action"], dtype=np.float64)
        root_pitch = signed_pitch_from_projected_gravity_rad(
            obs[ROOT_GRAVITY_OBS_SLICE]
        )
        waist_actual = (
            default_waist
            + obs[JOINT_POS_OBS_OFFSET + WAIST_PITCH_ISAAC_INDEX]
        )
        waist_issued_target = (
            default_waist
            + action[WAIST_PITCH_LOWER_INDEX]
            * LOWER_SCALE_RAD[WAIST_PITCH_LOWER_INDEX]
        )
        root_pitch_deg.append(math.degrees(root_pitch))
        waist_actual_deg.append(math.degrees(waist_actual))
        waist_issued_target_deg.append(math.degrees(waist_issued_target))
        waist_tracking_error_deg.append(math.degrees(waist_actual - waist_issued_target))
        for name in SAGITTAL_JOINTS:
            actual = (
                default_pose[name]
                + obs[JOINT_POS_OBS_OFFSET + ISAAC_JOINTS.index(name)]
            )
            target = (
                default_pose[name]
                + action[LOWER_JOINTS.index(name)]
                * LOWER_SCALE_RAD[LOWER_JOINTS.index(name)]
            )
            sagittal_actual[name].append(math.degrees(actual))
            sagittal_target[name].append(math.degrees(target))
    return {
        "rows_total": len(rows),
        "rows_with_policy_contract": len(valid),
        "root_pelvis_pitch_deg": _summary(root_pitch_deg),
        "waist_pitch_actual_deg": _summary(waist_actual_deg),
        "waist_pitch_issued_target_deg": _summary(waist_issued_target_deg),
        "waist_pitch_actual_minus_issued_target_deg": _summary(
            waist_tracking_error_deg
        ),
        "default_waist_pitch_deg": math.degrees(default_waist),
        "sagittal_joint_contract_deg": {
            name: {
                "default": math.degrees(default_pose[name]),
                "actual": _summary(sagittal_actual[name]),
                "issued_target": _summary(sagittal_target[name]),
                "actual_minus_issued_target": _summary(
                    np.asarray(sagittal_actual[name]) - np.asarray(sagittal_target[name])
                ),
            }
            for name in SAGITTAL_JOINTS
        },
    }


def analyze_trace_payload(payload: Mapping[str, object]) -> dict[str, object]:
    summary = payload["summary"]
    rows = payload["trace"]
    default_profile = str(summary.get("default_pose_profile", "stage208"))
    # Matched-event traces explicitly record a 1 s acceleration interval.  For
    # older step-command traces the analysis rule is fixed at 1 s, rather than
    # selected after inspecting the pitch curve.
    startup_seconds = float(summary.get("move_accelerate_seconds") or 1.0)
    phases = split_trace_phases(rows, startup_seconds=startup_seconds)
    return {
        "contract": {
            "pitch_sign": "negative=backward_lean; positive=forward_lean",
            "root_quantity": "pelvis/root ZYX pitch recovered from obs[6:9] projected gravity",
            "waist_actual_quantity": "default waist pitch + obs joint-position residual",
            "waist_target_quantity": "final issued normalized action * 0.16 rad + default; not raw actor output",
            "default_pose_profile": default_profile,
            "startup_seconds": startup_seconds,
        },
        "source_summary": {
            key: summary.get(key)
            for key in (
                "domain", "model", "template", "command_vx_mps", "fixed_wz_radps",
                "pd_profile", "pd_kp_multiplier", "pd_kd_multiplier",
                "default_pose_profile", "stationary_controller", "stop_controller",
                "action_bias_mode", "waist_tilt_action_multiplier",
                "symmetry_projection", "symmetry_projection_groups",
            )
            if key in summary
        },
        "phases": {
            name: summarize_trace_phase(phase_rows, default_pose_profile=default_profile)
            for name, phase_rows in phases.items()
        },
    }


def analyze_trace(path: Path) -> dict[str, object]:
    return analyze_trace_payload(json.loads(path.read_text(encoding="utf-8")))


def analyze_official_native_npz(path: Path) -> dict[str, object]:
    archive = np.load(path, allow_pickle=False)
    names = [str(name) for name in archive["joint_names"].tolist()]
    waist_index = names.index("waist_pitch_joint")
    root_pitch_deg = np.degrees(
        [signed_pitch_from_xyzw_rad(q) for q in archive["root_quat_xyzw"]]
    )
    torso_pitch_deg = np.degrees(
        [signed_pitch_from_xyzw_rad(q) for q in archive["torso_imu_quat_xyzw"]]
    )
    waist_actual_deg = np.degrees(archive["joint_q_rad"][:, waist_index])
    waist_command_deg = np.degrees(archive["command_q_rad"][:, waist_index])
    horizontal_speed = np.linalg.norm(archive["root_lin_vel_w_mps"][:, :2], axis=1)
    # The shipped dance trace has no stand/start/move/stop labels.  Do not infer
    # locomotion phases from speed and thereby create a false matched baseline.
    sagittal_contract = {}
    for name in SAGITTAL_JOINTS:
        index = names.index(name)
        actual_deg = np.degrees(archive["joint_q_rad"][:, index])
        command_deg = np.degrees(archive["command_q_rad"][:, index])
        sagittal_contract[name] = {
            "actual": _summary(actual_deg),
            "command": _summary(command_deg),
            "actual_minus_command": _summary(actual_deg - command_deg),
        }
    return {
        "source": str(archive["source"].item()),
        "truth_label": str(archive["truth_label"].item()),
        "frame_count": int(root_pitch_deg.size),
        "duration_s": float(archive["time_s"][-1] - archive["time_s"][0]),
        "phase_limitation": "unlabeled dance; overall distribution only, not a locomotion gate",
        "pitch_sign": "negative=backward_lean; positive=forward_lean",
        "root_pelvis_pitch_deg": _summary(root_pitch_deg),
        "torso_imu_pitch_deg": _summary(torso_pitch_deg),
        "waist_pitch_actual_deg": _summary(waist_actual_deg),
        "waist_pitch_command_deg": _summary(waist_command_deg),
        "waist_pitch_actual_minus_command_deg": _summary(
            waist_actual_deg - waist_command_deg
        ),
        "horizontal_speed_mps": _summary(horizontal_speed),
        "sagittal_joint_contract_deg": sagittal_contract,
    }


def _labelled_path(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", type=_labelled_path, default=[])
    parser.add_argument("--official-native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, object] = {
        "schema": "x2_sagittal_posture_audit_v1",
        "traces": {label: analyze_trace(path) for label, path in args.trace},
    }
    if args.official_native is not None:
        result["official_native_reference"] = analyze_official_native_npz(
            args.official_native
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
