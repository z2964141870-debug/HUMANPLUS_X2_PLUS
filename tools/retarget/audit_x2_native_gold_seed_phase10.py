#!/usr/bin/env python3
"""Audit the shipped X2 dance trace as a native dynamic Gold sanity seed.

This is intentionally not a GMR replacement.  It converts actual official-
simulation q/root/dq through the frozen official-31 -> WBT-29-with-locked-head
contract, audits model geometry and temporal integrity, and exports only when
all pre-registered gates pass.  Command targets are response evidence only.

Contact labels are derived from official MuJoCo active-sole geometry.  They are
not hardware foot force, GRF, wrench, or COP truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_strict_20260807/official_native_dance_teacher_50hz_60s.npz"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE.with_suffix(".manifest.json")
DEFAULT_MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
DEFAULT_OUTPUT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "gold_dynamic_native_seed_v1"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_native_gold_seed_phase10.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_gold_seed_phase10.md"

FPS = 50
CLIP_FRAMES = 400          # 8.0 s, matching the fixed WBT panel horizon.
TRAIN_FRAMES = (0, 1600)  # four non-overlapping clips.
EMBARGO_FRAMES = (1600, 1800)  # 4.0 s temporal guard; never exported.
HELD_OUT_FRAMES = (1800, 3000)  # three non-overlapping clips.

# Pre-registered audit gates.  They describe recorder/model consistency, not
# human motion style or real-hardware dynamics.
LIMIT_MAX_OVERSHOOT_RAD = 0.040
LIMIT_VIOLATION_FRACTION = 0.005
TIMESTAMP_P99_ERROR_S = 0.005
STEP_BOUND_VIOLATION_FRACTION = 0.001
DQ_CONSISTENCY_P95_RADPS = 1.0
ROOT_LIN_CONSISTENCY_P95_MPS = 0.5
ROOT_ANG_CONSISTENCY_P95_RADPS = 2.0
ROOT_Z_MIN_M = 0.45
ROOT_TILT_MAX_RAD = np.deg2rad(45.0)
SOLE_DEEP_PENETRATION_MIN_M = -0.010
FULL_SINGLE_SUPPORT_MIN = 0.10
SPLIT_SINGLE_SUPPORT_MIN = 0.05
DOUBLE_SUPPORT_MIN = 0.20
FLIGHT_MAX = 0.02
FULL_DS_SS_DS_CYCLES_MIN = 10
COMMAND_TRACKING_P95_RAD = 0.60
COMMAND_DELTA_P99_RAD = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_shape(name: str, values: np.ndarray, shape: tuple[int | None, ...]) -> None:
    if values.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(values.shape, shape)
    ):
        raise ValueError(f"{name} shape {values.shape}, expected {shape}")


def reorder_columns(values: np.ndarray, source_names: list[str], target_names: list[str]) -> np.ndarray:
    if len(source_names) != len(set(source_names)):
        raise ValueError("source joint names are not unique")
    if set(source_names) != set(target_names):
        raise ValueError("source and official joint-name sets differ")
    indices = [source_names.index(name) for name in target_names]
    return np.asarray(values)[:, indices]


def lock_head(
    q31: np.ndarray,
    dq31: np.ndarray,
    official_names: list[str],
    joint_contract: dict[str, dict[str, Any]],
    head_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    q = np.asarray(q31, dtype=np.float64).copy()
    dq = np.asarray(dq31, dtype=np.float64).copy()
    source_head = {}
    for name in head_names:
        index = official_names.index(name)
        source_head[name] = {
            "actual_q_min_max_rad": [float(np.min(q[:, index])), float(np.max(q[:, index]))],
            "actual_dq_abs_max_radps": float(np.max(np.abs(dq[:, index]))),
            "locked_value_rad": float(joint_contract[name]["model_nominal_rad"]),
        }
        q[:, index] = joint_contract[name]["model_nominal_rad"]
        dq[:, index] = 0.0
    return q, dq, source_head


def pose_aa_from_dof(
    dof31: np.ndarray,
    root_quat_xyzw: np.ndarray,
    official_names: list[str],
    joint_contract: dict[str, dict[str, Any]],
) -> np.ndarray:
    pose = np.zeros((len(dof31), len(official_names) + 1, 3), dtype=np.float32)
    pose[:, 0] = Rotation.from_quat(np.asarray(root_quat_xyzw)).as_rotvec().astype(np.float32)
    for index, name in enumerate(official_names):
        pose[:, index + 1] = (
            np.asarray(dof31[:, index], dtype=np.float64)[:, None]
            * np.asarray(joint_contract[name]["axis_xyz"], dtype=np.float64)[None]
        ).astype(np.float32)
    return pose


def temporal_metrics(
    time_s: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    root: np.ndarray,
    quat_xyzw: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
) -> dict[str, Any]:
    dt = np.diff(time_s)
    q_step = np.abs(np.diff(q, axis=0))
    step_bound = 1.25 * dt[:, None] * np.maximum(np.abs(dq[:-1]), np.abs(dq[1:])) + 0.020
    dq_from_q = np.diff(q, axis=0) / dt[:, None]
    dq_error = dq_from_q - 0.5 * (dq[:-1] + dq[1:])
    root_lin_from_pos = np.diff(root, axis=0) / dt[:, None]
    root_lin_error = root_lin_from_pos - 0.5 * (root_lin_vel[:-1] + root_lin_vel[1:])
    rotations = Rotation.from_quat(quat_xyzw)
    root_ang_from_quat = np.asarray([
        (rotations[index].inv() * rotations[index + 1]).as_rotvec() / dt[index]
        for index in range(len(dt))
    ])
    root_ang_error = root_ang_from_quat - 0.5 * (root_ang_vel[:-1] + root_ang_vel[1:])
    return {
        "dt_min_median_max_s": [float(np.min(dt)), float(np.median(dt)), float(np.max(dt))],
        "dt_error_abs_p99_s": float(np.percentile(np.abs(dt - 1.0 / FPS), 99)),
        "joint_step_p95_max_rad": [float(np.percentile(q_step, 95)), float(np.max(q_step))],
        "joint_step_physics_bound_violation_fraction": float(np.mean(q_step > step_bound)),
        "dq_consistency_error_abs_p50_p95_max_radps": [
            float(np.percentile(np.abs(dq_error), quantile)) for quantile in (50, 95, 100)
        ],
        "root_linear_consistency_error_norm_p50_p95_max_mps": [
            float(np.percentile(np.linalg.norm(root_lin_error, axis=1), quantile))
            for quantile in (50, 95, 100)
        ],
        "root_angular_consistency_error_norm_p50_p95_max_radps": [
            float(np.percentile(np.linalg.norm(root_ang_error, axis=1), quantile))
            for quantile in (50, 95, 100)
        ],
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(quat_xyzw, axis=1) - 1.0))
        ),
    }


def limit_metrics(
    q31: np.ndarray, official_names: list[str], joint_contract: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    limits = np.asarray([joint_contract[name]["range_rad"] for name in official_names])
    lower_over = np.maximum(limits[:, 0][None] - q31, 0.0)
    upper_over = np.maximum(q31 - limits[:, 1][None], 0.0)
    overshoot = np.maximum(lower_over, upper_over)
    per_joint = {}
    for index, name in enumerate(official_names):
        value = float(np.max(overshoot[:, index]))
        if value > 0.0:
            per_joint[name] = {
                "max_overshoot_rad": value,
                "violation_frame_fraction": float(np.mean(overshoot[:, index] > 1.0e-5)),
            }
    return {
        "max_overshoot_rad": float(np.max(overshoot)),
        "violation_sample_fraction": float(np.mean(overshoot > 1.0e-5)),
        "violation_frame_fraction": float(np.mean(np.any(overshoot > 1.0e-5, axis=1))),
        "per_joint_nonzero": per_joint,
    }


def phase_statistics(contact: dict[str, np.ndarray]) -> dict[str, Any]:
    left = np.asarray(contact["left"], dtype=bool)
    right = np.asarray(contact["right"], dtype=bool)
    ds = left & right
    left_ss = left & ~right
    right_ss = right & ~left
    flight = ~left & ~right
    state = np.where(ds, 0, np.where(left_ss, 1, np.where(right_ss, 2, 3)))
    starts = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1]
    ends = np.r_[starts[1:], len(state)]
    segments = [(int(start), int(end), int(state[start])) for start, end in zip(starts, ends)]
    cycles = sum(
        segments[index - 1][2] == 0
        and segments[index][2] in (1, 2)
        and segments[index + 1][2] == 0
        for index in range(1, len(segments) - 1)
    )
    total = max(1, len(left))
    return {
        "frames": len(left),
        "contact_window_count": {
            side: len(phase3.boolean_intervals(np.asarray(contact[side], dtype=bool), 1))
            for side in ("left", "right")
        },
        "double_support_ratio": float(np.sum(ds) / total),
        "single_support_ratio": float(np.sum(left_ss | right_ss) / total),
        "left_single_support_ratio": float(np.sum(left_ss) / total),
        "right_single_support_ratio": float(np.sum(right_ss) / total),
        "flight_ratio": float(np.sum(flight) / total),
        "ds_ss_ds_cycle_count": int(cycles),
    }


def active_sole_geometry(
    model: mujoco.MjModel,
    q31: np.ndarray,
    root: np.ndarray,
    quat_xyzw: np.ndarray,
    official_names: list[str],
    phase2,
    physics,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    qpos_addresses, _ = phase2.joint_addresses(model, official_names)
    floor, feet = phase7.active_sole_spheres(model)
    reset = phase7.official_reset_geometry(model, physics)
    threshold = reset["reset_clearance_m"] + reset["sole_sphere_radius_m"]
    distances = {side: np.zeros(len(q31), dtype=np.float64) for side in feet}
    data = mujoco.MjData(model)
    fromto = np.zeros(6, dtype=np.float64)
    for frame in range(len(q31)):
        phase2.set_reference_state(
            model, data, root[frame], quat_xyzw[frame], q31[frame], qpos_addresses
        )
        for side, geoms in feet.items():
            distances[side][frame] = min(
                float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto))
                for geom in geoms
            )
    contact = {side: values <= threshold for side, values in distances.items()}
    return {
        "semantics": "minimum signed floor distance over 12 active official sole spheres per foot",
        "contact_threshold_m": float(threshold),
        "reset_clearance_m": float(reset["reset_clearance_m"]),
        "sphere_radius_m": float(reset["sole_sphere_radius_m"]),
        "per_side_distance_p00_p05_p50_p95_m": {
            side: [float(np.percentile(values, q)) for q in (0, 5, 50, 95)]
            for side, values in distances.items()
        },
        "minimum_signed_distance_m": float(min(np.min(values) for values in distances.values())),
        "full_trace_phase": phase_statistics(contact),
    }, contact


def action_response_metrics(
    q_source_order: np.ndarray,
    command_source_order: np.ndarray,
    valid_source_order: np.ndarray,
    kp_source_order: np.ndarray,
    source_names: list[str],
    wbt_names: list[str],
) -> dict[str, Any]:
    indices = [source_names.index(name) for name in wbt_names]
    q = q_source_order[:, indices]
    command = command_source_order[:, indices]
    valid = valid_source_order[:, indices]
    kp = kp_source_order[:, indices]
    error = np.abs(q - command)
    delta = np.abs(np.diff(command, axis=0))
    return {
        "semantics": "command is response evidence only and is not exported as reference",
        "command_valid_fraction": float(np.mean(valid)),
        "positive_kp_fraction": float(np.mean(kp > 0.0)),
        "actual_minus_command_abs_p50_p95_max_rad": [
            float(np.percentile(error[valid], q)) for q in (50, 95, 100)
        ],
        "command_delta_abs_p50_p95_p99_max_rad": [
            float(np.percentile(delta, q)) for q in (50, 95, 99, 100)
        ],
    }


def split_plan(frame_count: int) -> dict[str, list[tuple[int, int]]]:
    if frame_count != HELD_OUT_FRAMES[1]:
        raise ValueError(f"registered split expects {HELD_OUT_FRAMES[1]} frames, got {frame_count}")
    return {
        "train": [
            (start, start + CLIP_FRAMES)
            for start in range(TRAIN_FRAMES[0], TRAIN_FRAMES[1], CLIP_FRAMES)
        ],
        "held_out": [
            (start, start + CLIP_FRAMES)
            for start in range(HELD_OUT_FRAMES[0], HELD_OUT_FRAMES[1], CLIP_FRAMES)
        ],
    }


def split_phase_statistics(
    contact: dict[str, np.ndarray], plan: dict[str, list[tuple[int, int]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    clips: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    for split, ranges in plan.items():
        clips[split] = []
        concatenated = {
            side: np.concatenate([contact[side][start:end] for start, end in ranges])
            for side in ("left", "right")
        }
        aggregate[split] = phase_statistics(concatenated)
        for index, (start, end) in enumerate(ranges):
            clips[split].append({
                "index": index,
                "start_frame": start,
                "end_frame_exclusive": end,
                "start_time_s": start / FPS,
                "end_time_s": end / FPS,
                "phase": phase_statistics({side: contact[side][start:end] for side in contact}),
            })
    return clips, aggregate


def audit_gate(
    temporal: dict[str, Any],
    limits: dict[str, Any],
    root: dict[str, Any],
    geometry: dict[str, Any],
    split_aggregate: dict[str, Any],
    response: dict[str, Any],
    contract_checks: dict[str, bool],
) -> dict[str, Any]:
    phase = geometry["full_trace_phase"]
    checks = {
        **contract_checks,
        "timestamp_jitter_bounded": temporal["dt_error_abs_p99_s"] <= TIMESTAMP_P99_ERROR_S,
        "joint_step_consistent_with_logged_dq": temporal["joint_step_physics_bound_violation_fraction"] <= STEP_BOUND_VIOLATION_FRACTION,
        "joint_dq_consistency": temporal["dq_consistency_error_abs_p50_p95_max_radps"][1] <= DQ_CONSISTENCY_P95_RADPS,
        "root_linear_velocity_consistency": temporal["root_linear_consistency_error_norm_p50_p95_max_mps"][1] <= ROOT_LIN_CONSISTENCY_P95_MPS,
        "root_angular_velocity_consistency": temporal["root_angular_consistency_error_norm_p50_p95_max_radps"][1] <= ROOT_ANG_CONSISTENCY_P95_RADPS,
        "quaternion_normalized": temporal["quaternion_norm_max_error"] <= 1.0e-5,
        "joint_limit_overshoot_bounded": limits["max_overshoot_rad"] <= LIMIT_MAX_OVERSHOOT_RAD,
        "joint_limit_violation_sparse": limits["violation_sample_fraction"] <= LIMIT_VIOLATION_FRACTION,
        "recorded_root_survives": root["root_z_min_m"] >= ROOT_Z_MIN_M and root["root_tilt_max_rad"] <= ROOT_TILT_MAX_RAD,
        "active_sole_no_deep_penetration": geometry["minimum_signed_distance_m"] >= SOLE_DEEP_PENETRATION_MIN_M,
        "full_trace_has_dynamic_single_support": phase["single_support_ratio"] >= FULL_SINGLE_SUPPORT_MIN,
        "full_trace_has_double_support": phase["double_support_ratio"] >= DOUBLE_SUPPORT_MIN,
        "full_trace_flight_sparse": phase["flight_ratio"] <= FLIGHT_MAX,
        "full_trace_has_contact_cycles": phase["ds_ss_ds_cycle_count"] >= FULL_DS_SS_DS_CYCLES_MIN,
        "train_split_dynamic": split_aggregate["train"]["single_support_ratio"] >= SPLIT_SINGLE_SUPPORT_MIN,
        "held_out_split_dynamic": split_aggregate["held_out"]["single_support_ratio"] >= SPLIT_SINGLE_SUPPORT_MIN,
        "both_splits_have_double_support": all(
            value["double_support_ratio"] >= DOUBLE_SUPPORT_MIN for value in split_aggregate.values()
        ),
        "both_splits_have_sparse_flight": all(
            value["flight_ratio"] <= FLIGHT_MAX for value in split_aggregate.values()
        ),
        "command_valid_for_all_wbt29": response["command_valid_fraction"] == 1.0,
        "command_kp_positive_for_all_wbt29": response["positive_kp_fraction"] == 1.0,
        "command_response_error_bounded": response["actual_minus_command_abs_p50_p95_max_rad"][1] <= COMMAND_TRACKING_P95_RAD,
        "command_delta_bounded": response["command_delta_abs_p50_p95_p99_max_rad"][2] <= COMMAND_DELTA_P99_RAD,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {"checks": checks, "pass": bool(all(checks.values()))}


def build_motion_entries(
    q31: np.ndarray,
    dq31: np.ndarray,
    root: np.ndarray,
    quat: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    contact: dict[str, np.ndarray],
    official_names: list[str],
    joint_contract: dict[str, dict[str, Any]],
    plan: dict[str, list[tuple[int, int]]],
    source: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    pose = pose_aa_from_dof(q31, quat, official_names, joint_contract)
    output = {"train": {}, "held_out": {}}
    for split, ranges in plan.items():
        for index, (start, end) in enumerate(ranges):
            key = f"official_native_dance_{split}_{index:03d}"
            output[split][key] = {
                "root_trans_offset": root[start:end].astype(np.float32),
                "pose_aa": pose[start:end].copy(),
                "dof": q31[start:end].astype(np.float32),
                "root_rot": quat[start:end].astype(np.float32),
                "smpl_joints": np.zeros((end - start, 24, 3), dtype=np.float32),
                "fps": FPS,
                "joint_names_mujoco": list(official_names),
                "dof_vel": dq31[start:end].astype(np.float32),
                "root_lin_vel_w_mps": root_lin_vel[start:end].astype(np.float32),
                "root_ang_vel": root_ang_vel[start:end].astype(np.float32),
                "model_contact": {
                    side: contact[side][start:end].astype(bool) for side in ("left", "right")
                },
                "source_trace": str(source),
                "source_frame_range": [start, end],
                "split": split,
                "retarget_method": "none; official native X2 actual-state trace",
                "root_quaternion_convention": "xyzw",
                "head_contract": "official 31 storage with two head joints locked at model nominal; WBT target is 29",
                "contact_truth_boundary": "official-model active-sole geometry estimate, not hardware GRF/COP/wrench",
            }
    return output


def render(report: dict[str, Any]) -> str:
    audit = report["audit"]
    phase = audit["active_sole_geometry"]["full_trace_phase"]
    temporal = audit["temporal"]
    root = audit["root_survival"]
    response = audit["command_response_evidence"]
    lines = [
        "# X2 Official Native Dynamic Gold Seed Phase10",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 输入是 AimDK X2 v1.0 官方 MuJoCo + 随包 `kuailechongbai.onnx` 的 actual-state 舞蹈 trace；只读源数据、未训练、未重放、未修改物理。",
        "- 它不是 GMR/AMASS Silver，不是实机真值，也不能证明 Any2Any；用途只是隔离“训练管线能否跟踪目标本体原生可执行 reference”。",
        "- contact 是官方模型 active-sole 几何估计，**不是实机 GRF、COP、wrench 或足底力真值**。",
        "",
        "## 31DoF → WBT29 合同",
        "",
        "- 源 joint-name 集与官方 31DoF 完全一致，先按名字重排到 MuJoCo 31 顺序；WBT29 保持官方顺序。",
        "- 两个头关节不进入 WBT：导出时锁到 model nominal，`dof_vel=0`；其余29关节逐样本保持 actual q/dq。",
        "- MotionLib 存储仍为官方 MuJoCo 31 顺序；`pose_aa` root来自actual root quaternion，其余由轴×实际关节角构造，SMPL joints 明确置零。",
        "",
        "## Audit",
        "",
        f"- 时间：{audit['source_integrity']['samples']} 帧，{audit['source_integrity']['duration_s']:.3f}s；dt p99 误差 `{temporal['dt_error_abs_p99_s']:.6f}s`。",
        f"- 连续性：joint-step `{temporal['joint_step_p95_max_rad'][0]:.4f}/{temporal['joint_step_p95_max_rad'][1]:.4f}rad` (p95/max)；dq 一致性 p95 `{temporal['dq_consistency_error_abs_p50_p95_max_radps'][1]:.3f}rad/s`。",
        f"- limits：最大软越界 `{audit['joint_limits']['max_overshoot_rad']:.4f}rad`，样本比例 `{audit['joint_limits']['violation_sample_fraction']:.5f}`。",
        f"- recorded survival：root-z min `{root['root_z_min_m']:.3f}m`，tilt max `{np.rad2deg(root['root_tilt_max_rad']):.2f}°`。",
        f"- active sole：最低 signed distance `{audit['active_sole_geometry']['minimum_signed_distance_m']:.4f}m`；DS/SS/flight `{phase['double_support_ratio']:.3f}/{phase['single_support_ratio']:.3f}/{phase['flight_ratio']:.3f}`；DS→SS→DS `{phase['ds_ss_ds_cycle_count']}` 次。",
        f"- command response（仅证据）：valid `{response['command_valid_fraction']:.3f}`，q-command p95 `{response['actual_minus_command_abs_p50_p95_max_rad'][1]:.3f}rad`，command Δ p99 `{response['command_delta_abs_p50_p95_p99_max_rad'][2]:.3f}rad`。",
        "",
        "## 时间块与泄漏边界",
        "",
        "| split | clips | frame blocks | duration | SS | DS | flight | cycles |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "held_out"):
        values = report["split"]["aggregate_phase"][split]
        blocks = report["split"]["ranges"][split]
        lines.append(
            f"| {split} | {len(blocks)} | {blocks} | {len(blocks) * CLIP_FRAMES / FPS:.1f}s | "
            f"{values['single_support_ratio']:.3f} | {values['double_support_ratio']:.3f} | "
            f"{values['flight_ratio']:.3f} | {values['ds_ss_ds_cycle_count']} |"
        )
    lines += [
        "",
        f"- train `[0,1600)`，embargo `[1600,1800)`，held-out `[1800,3000)`；clip固定400帧、无重叠。随机相邻帧切分被禁止。",
        "",
        "## 裁决",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_MODEL_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    archive = np.load(args.source, allow_pickle=False)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    model_contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    source_names = [str(value) for value in archive["joint_names"].tolist()]
    official_names = model_contract["control_boundaries"]["official_mjcf_actuated_31"]
    wbt_names = model_contract["control_boundaries"]["wbt_target_29"]
    head_names = model_contract["control_boundaries"]["head_locked_2"]
    joint_contract = {value["name"]: value for value in model_contract["joints"]}

    samples = len(archive["time_s"])
    for name, width in (
        ("joint_q_rad", 31), ("joint_dq_radps", 31), ("command_q_rad", 31),
        ("command_valid", 31), ("command_kp", 31),
    ):
        require_shape(name, archive[name], (samples, width))
    for name, width in (
        ("root_pos_w_m", 3), ("root_quat_xyzw", 4),
        ("root_lin_vel_w_mps", 3), ("root_ang_vel", 3),
    ):
        require_shape(name, archive[name], (samples, width))

    q_source = np.asarray(archive["joint_q_rad"], dtype=np.float64)
    dq_source = np.asarray(archive["joint_dq_radps"], dtype=np.float64)
    q31_actual = reorder_columns(q_source, source_names, official_names)
    dq31_actual = reorder_columns(dq_source, source_names, official_names)
    q31, dq31, source_head = lock_head(
        q31_actual, dq31_actual, official_names, joint_contract, head_names
    )
    root = np.asarray(archive["root_pos_w_m"], dtype=np.float64)
    quat = np.asarray(archive["root_quat_xyzw"], dtype=np.float64)
    root_lin = np.asarray(archive["root_lin_vel_w_mps"], dtype=np.float64)
    root_ang = np.asarray(archive["root_ang_vel"], dtype=np.float64)
    time_s = np.asarray(archive["time_s"], dtype=np.float64)

    finite_fields = (q31_actual, dq31_actual, root, quat, root_lin, root_ang, time_s)
    source_hash = sha256(args.source)
    contract_checks = {
        "all_required_arrays_finite": bool(all(np.all(np.isfinite(value)) for value in finite_fields)),
        "source_manifest_hash_and_samples_match": source_manifest["sha256"] == source_hash
        and source_manifest["samples"] == samples,
        "timestamps_strictly_increasing": bool(np.all(np.diff(time_s) > 0.0)),
        "source_joint_name_set_exact_official31": set(source_names) == set(official_names) and len(set(source_names)) == 31,
        "wbt29_and_head2_partition_official31": set(wbt_names).isdisjoint(head_names)
        and set(wbt_names) | set(head_names) == set(official_names),
        "wbt29_actual_q_preserved_by_name": bool(np.array_equal(
            q31[:, [official_names.index(name) for name in wbt_names]],
            q_source[:, [source_names.index(name) for name in wbt_names]],
        )),
        "head_locked_exact_nominal": bool(all(
            np.all(q31[:, official_names.index(name)] == joint_contract[name]["model_nominal_rad"])
            and np.all(dq31[:, official_names.index(name)] == 0.0)
            for name in head_names
        )),
    }

    temporal = temporal_metrics(time_s, q31_actual, dq31_actual, root, quat, root_lin, root_ang)
    limits = limit_metrics(q31_actual, official_names, joint_contract)
    rotations = Rotation.from_quat(quat)
    world_up = rotations.apply(np.asarray([0.0, 0.0, 1.0]))
    tilt = np.arccos(np.clip(world_up[:, 2], -1.0, 1.0))
    root_survival = {
        "semantics": "recorded official-simulator state; no replay in Phase10",
        "complete_trace_available": samples == source_manifest["samples"],
        "root_z_min_p05_p50_m": [
            float(np.min(root[:, 2])), float(np.percentile(root[:, 2], 5)), float(np.median(root[:, 2]))
        ],
        "root_z_min_m": float(np.min(root[:, 2])),
        "root_tilt_p95_max_rad": [float(np.percentile(tilt, 95)), float(np.max(tilt))],
        "root_tilt_max_rad": float(np.max(tilt)),
    }

    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase10")
    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase10")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    geometry, contact = active_sole_geometry(
        model, q31, root, quat, official_names, phase2, physics
    )
    plan = split_plan(samples)
    clip_phase, split_phase = split_phase_statistics(contact, plan)
    response = action_response_metrics(
        q_source,
        np.asarray(archive["command_q_rad"], dtype=np.float64),
        np.asarray(archive["command_valid"], dtype=bool),
        np.asarray(archive["command_kp"], dtype=np.float64),
        source_names,
        wbt_names,
    )
    gate = audit_gate(temporal, limits, root_survival, geometry, split_phase, response, contract_checks)

    exported = False
    asset_files: dict[str, Any] = {}
    if gate["pass"]:
        entries = build_motion_entries(
            q31, dq31, root, quat, root_lin, root_ang, contact,
            official_names, joint_contract, plan, args.source,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        for split in ("train", "held_out"):
            split_dir = args.output / split
            split_dir.mkdir(parents=True, exist_ok=True)
            path = split_dir / f"official_native_dance_{split}.pkl"
            joblib.dump(entries[split], path)
            asset_files[split] = {
                "path": str(path), "sha256": sha256(path), "clips": len(entries[split])
            }
        exported = True

    report = {
        "schema_version": "x2_native_gold_seed_phase10_v1",
        "provenance": {
            "source_npz": {"path": str(args.source), "sha256": source_hash},
            "source_manifest": {"path": str(args.source_manifest), "sha256": sha256(args.source_manifest)},
            "model_contract": {"path": str(args.model_contract), "sha256": sha256(args.model_contract)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": sha256(physics.DEFAULT_CONTROL)},
            "exported_assets": asset_files,
        },
        "truth_boundary": {
            "source": str(archive["source"].item()),
            "source_truth_label": str(archive["truth_label"].item()),
            "official_simulation_not_real_hardware": True,
            "contact_is_model_geometry_not_grf_cop_wrench": True,
            "not_gmr_amass_silver_or_gmr_replacement": True,
            "cannot_prove_any2any_or_cross_embodiment": True,
            "command_is_response_evidence_not_reference": True,
            "training_replay_teacher_checkpoint_base_git_baidu_real_robot": False,
        },
        "audit": {
            "source_integrity": {
                "samples": samples,
                "duration_s": float(time_s[-1] - time_s[0]),
                "sample_rate_nominal_hz": FPS,
                "source_joint_order": source_names,
                "official_mujoco_order": official_names,
                "source_manifest_matches_npz": source_manifest["sha256"] == source_hash
                and source_manifest["samples"] == samples,
            },
            "contract_checks": contract_checks,
            "head_lock": source_head,
            "temporal": temporal,
            "joint_limits": limits,
            "root_survival": root_survival,
            "active_sole_geometry": geometry,
            "command_response_evidence": response,
        },
        "split": {
            "policy": {
                "clip_frames": CLIP_FRAMES,
                "clip_duration_s": CLIP_FRAMES / FPS,
                "train_frame_block": list(TRAIN_FRAMES),
                "embargo_frame_block": list(EMBARGO_FRAMES),
                "held_out_frame_block": list(HELD_OUT_FRAMES),
                "random_adjacent_frame_split": False,
                "overlap": False,
            },
            "ranges": {split: [list(value) for value in ranges] for split, ranges in plan.items()},
            "per_clip_phase": clip_phase,
            "aggregate_phase": split_phase,
        },
        "gate": gate,
        "decision": {
            "status": "PHASE10_NATIVE_DYNAMIC_GOLD_SEED_EXPORTED" if exported else "PHASE10_NATIVE_DYNAMIC_GOLD_SEED_REJECTED",
            "exported": exported,
            "failed_checks": [name for name, passed in gate["checks"].items() if not passed],
            "result": (
                "official native actual-state trace通过完整性、动力学记录一致性、active-sole接触周期和时间块泄漏门，已导出独立Gold sanity seed。"
                if exported
                else "源trace未通过预注册Gold seed门，仅保留audit，未导出MotionLib资产。"
            ),
            "conclusion": (
                "该资产可用于隔离X2原生可执行reference的训练/评估pipeline sanity；它不增加GMR Silver数量，也不证明跨本体迁移。"
                if exported
                else "不能把该官方舞蹈安全包装为Gold seed；失败不评价GMR或Any2Any。"
            ),
            "next_step": (
                "下一阶段只做pipeline tracking sanity对照，并严格保持held-out时间块；不得将train/held-out合并或随机切帧。"
                if exported
                else "停止导出；若需native Gold，应获取带可靠连续状态与接触周期的新官方X2 trace。"
            ),
        },
    }
    if exported:
        manifest_path = args.output / "manifest.json"
        manifest = copy.deepcopy(report)
        manifest["provenance"]["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["provenance"]["exported_assets"]["manifest"] = {
            "path": str(manifest_path), "sha256": sha256(manifest_path)
        }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
