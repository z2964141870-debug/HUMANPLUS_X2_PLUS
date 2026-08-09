#!/usr/bin/env python3
"""Low-dimensional, no-training joint reference-repair oracle for X2 WBT.

The oracle asks a deliberately narrow question: can a *coordinated* repair of
root support shift, contact timing, stance-foot locking and swing clearance
make the existing exact-30 Hz references less hostile to the official AimDK
v1.0 free-root dynamics?  It is not a deployable controller.  The search sees
the complete motion and all COM/contact quantities are estimates from the
official MuJoCo model, never hardware GRF/COP/COM measurements.

The intervention is kept isolated from training and checkpoints.  It first
passes a kinematic/continuity gate, then evaluates surviving candidates under
the exact same official position-PD contract in both prescribed-root and
free-root modes.  Current-v4 and the rejected toe-smooth9 variant remain fixed
controls.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d


REPO = Path(__file__).resolve().parents[2]
PHYSICS_SCRIPT = REPO / "tools/retarget/run_x2_forefoot_official_physics_screen.py"
CACHE_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_smoothing_ab"
)
DEFAULT_CURRENT = CACHE_ROOT / "current_v4_exact30/x2_current_v4_exact30.pkl"
DEFAULT_TOE = CACHE_ROOT / "smooth9/x2_toe_forefoot_smooth9.pkl"
DEFAULT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase2_joint_reference_repair/x2_joint_reference_repair_phase2.pkl"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_joint_reference_repair_phase2.json"
DEFAULT_MD = REPO / "reports/retarget/x2_joint_reference_repair_phase2.md"
DEFAULT_ROLES = ("walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand")

SOLE_POINT_LOCAL = {
    "left": np.array([0.025, 0.0, -0.068], dtype=np.float64),
    "right": np.array([0.025, 0.0, -0.068], dtype=np.float64),
}
FOOT_BODY = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
LEG_PREFIX = {"left": "left_", "right": "right_"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RepairParameters:
    root_support_gain: float
    root_height_gain: float
    contact_phase_shift_frames: int
    foot_terminal_blend: float
    swing_clearance_m: float
    stance_lock_gain: float = 0.01

    @property
    def is_noop(self) -> bool:
        return (
            self.root_support_gain == 0.0
            and self.root_height_gain == 0.0
            and self.contact_phase_shift_frames == 0
            and self.foot_terminal_blend == 0.0
            and self.swing_clearance_m == 0.0
            and self.stance_lock_gain == 0.0
        )


def joint_addresses(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos, dof = [], []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint missing from official model: {name}")
        qpos.append(int(model.jnt_qposadr[joint_id]))
        dof.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos, dtype=np.int64), np.asarray(dof, dtype=np.int64)


def point_world(data: mujoco.MjData, body_id: int, local: np.ndarray) -> np.ndarray:
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ local


def point_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point: np.ndarray
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return jacp


def set_reference_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_xyzw: np.ndarray,
    dof: np.ndarray,
    qpos_addresses: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_xyzw[[3, 0, 1, 2]]
    data.qpos[qpos_addresses] = dof
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def reference_kinematics(
    model: mujoco.MjModel, entry: dict[str, Any]
) -> dict[str, np.ndarray | float]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, _ = joint_addresses(model, names)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    dof = np.asarray(entry["dof"], dtype=np.float64)
    data = mujoco.MjData(model)
    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for side, body in FOOT_BODY.items()
    }
    foot = {side: np.zeros((len(dof), 3), dtype=np.float64) for side in body_ids}
    com = np.zeros((len(dof), 3), dtype=np.float64)
    for frame in range(len(dof)):
        set_reference_state(model, data, root[frame], quat[frame], dof[frame], qpos_addresses)
        for side, body_id in body_ids.items():
            foot[side][frame] = point_world(data, body_id, SOLE_POINT_LOCAL[side])
        com[frame] = data.subtree_com[0]
    fps = float(entry["fps"])
    velocity = {side: np.gradient(values, 1.0 / fps, axis=0) for side, values in foot.items()}
    com_velocity = np.gradient(com, 1.0 / fps, axis=0)
    ground = float(np.percentile(np.minimum(foot["left"][:, 2], foot["right"][:, 2]), 5))
    contact_probability = {}
    for side in ("left", "right"):
        height = np.maximum(foot[side][:, 2] - ground, 0.0)
        horizontal_speed = np.linalg.norm(velocity[side][:, :2], axis=1)
        contact_probability[side] = np.exp(-np.square(height / 0.028)) * np.exp(
            -np.square(horizontal_speed / 0.35)
        )
    return {
        "left_foot": foot["left"],
        "right_foot": foot["right"],
        "left_velocity": velocity["left"],
        "right_velocity": velocity["right"],
        "left_contact_probability": contact_probability["left"],
        "right_contact_probability": contact_probability["right"],
        "com": com,
        "com_velocity": com_velocity,
        "ground_z_m": ground,
    }


def shifted(array: np.ndarray, frames: int) -> np.ndarray:
    indices = np.clip(np.arange(len(array)) + int(frames), 0, len(array) - 1)
    return array[indices]


def phase_contract(kinematics: dict[str, Any], shift_frames: int = 0) -> dict[str, np.ndarray]:
    left = shifted(np.asarray(kinematics["left_contact_probability"]), shift_frames)
    right = shifted(np.asarray(kinematics["right_contact_probability"]), shift_frames)
    left_z = shifted(np.asarray(kinematics["left_foot"])[:, 2], shift_frames)
    right_z = shifted(np.asarray(kinematics["right_foot"])[:, 2], shift_frames)
    # A foot is swing only when both its model-estimated contact probability is
    # lower and its sole is geometrically higher.  Ambiguous frames stay double
    # support rather than inventing a hard contact label.
    left_swing = (right - left > 0.12) & (left_z - right_z > 0.012)
    right_swing = (left - right > 0.12) & (right_z - left_z > 0.012)
    stance_left = right_swing
    stance_right = left_swing
    double_support = ~(left_swing | right_swing)
    return {
        "left_swing": left_swing,
        "right_swing": right_swing,
        "stance_left": stance_left,
        "stance_right": stance_right,
        "double_support": double_support,
        "left_contact_probability": left,
        "right_contact_probability": right,
    }


def bounded_root_correction(
    root: np.ndarray,
    kinematics: dict[str, Any],
    phase: dict[str, np.ndarray],
    support_gain: float,
    height_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    correction = np.zeros_like(root)
    left_foot = np.asarray(kinematics["left_foot"])
    right_foot = np.asarray(kinematics["right_foot"])
    ground = float(kinematics["ground_z_m"])
    for frame in range(len(root)):
        support = []
        if phase["stance_left"][frame] or phase["double_support"][frame]:
            support.append(left_foot[frame])
        if phase["stance_right"][frame] or phase["double_support"][frame]:
            support.append(right_foot[frame])
        if not support:
            continue
        center = np.mean(support, axis=0)
        delta_xy = center[:2] - root[frame, :2]
        norm = float(np.linalg.norm(delta_xy))
        if norm > 0.0:
            delta_xy *= min(0.04, norm) / norm
        correction[frame, :2] = support_gain * delta_xy
        sole_height = float(np.mean([point[2] for point in support]))
        correction[frame, 2] = np.clip(-height_gain * (sole_height - ground), -0.025, 0.025)
    if len(root) > 2:
        correction = gaussian_filter1d(correction, sigma=1.5, axis=0, mode="nearest")
    correction[:, :2] = np.clip(correction[:, :2], -0.04, 0.04)
    correction[:, 2] = np.clip(correction[:, 2], -0.025, 0.025)
    return root + correction, correction


def ik_repair(
    model: mujoco.MjModel,
    entry: dict[str, Any],
    donor: dict[str, Any],
    params: RepairParameters,
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = list(entry["joint_names_mujoco"])
    dof_base = np.asarray(entry["dof"], dtype=np.float64)
    if params.is_noop:
        return copy.deepcopy(entry), {"exact_noop": True}
    dof = dof_base.copy()
    donor_dof = np.asarray(donor["dof"], dtype=np.float64)
    leg_indices = np.asarray(
        [i for i, name in enumerate(names) if any(token in name for token in ("hip", "knee", "ankle"))],
        dtype=np.int64,
    )
    dof[:, leg_indices] += params.foot_terminal_blend * (
        donor_dof[:, leg_indices] - dof[:, leg_indices]
    )
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    base_kin = reference_kinematics(model, entry)
    phase = phase_contract(base_kin, params.contact_phase_shift_frames)
    root_repaired, root_correction = bounded_root_correction(
        root, base_kin, phase, params.root_support_gain, params.root_height_gain
    )

    qpos_addresses, qvel_addresses = joint_addresses(model, names)
    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for side, body in FOOT_BODY.items()
    }
    side_indices = {
        side: np.asarray([i for i, name in enumerate(names) if name.startswith(LEG_PREFIX[side]) and any(
            token in name for token in ("hip", "knee", "ankle")
        )], dtype=np.int64)
        for side in ("left", "right")
    }
    side_dof_addresses = {side: qvel_addresses[indices] for side, indices in side_indices.items()}
    ground = float(base_kin["ground_z_m"])
    data = mujoco.MjData(model)
    anchors: dict[str, np.ndarray | None] = {"left": None, "right": None}
    stance_residuals: list[float] = []
    swing_residuals: list[float] = []
    previous_stance = {"left": False, "right": False}

    for frame in range(len(dof)):
        stance = {
            "left": bool(phase["stance_left"][frame] or phase["double_support"][frame]),
            "right": bool(phase["stance_right"][frame] or phase["double_support"][frame]),
        }
        swing = {
            "left": bool(phase["left_swing"][frame]),
            "right": bool(phase["right_swing"][frame]),
        }
        for _ in range(3):
            set_reference_state(model, data, root_repaired[frame], quat[frame], dof[frame], qpos_addresses)
            for side in ("left", "right"):
                body_id = body_ids[side]
                point = point_world(data, body_id, SOLE_POINT_LOCAL[side])
                if stance[side] and (anchors[side] is None or not previous_stance[side]):
                    anchors[side] = point.copy()
                    anchors[side][2] = ground
                if not stance[side]:
                    anchors[side] = None

                if stance[side] and anchors[side] is not None:
                    target = anchors[side].copy()
                    error = target - point
                    task_rows = np.arange(3)
                    gain = params.stance_lock_gain
                    stance_residuals.append(float(np.linalg.norm(error)))
                elif swing[side] and params.swing_clearance_m > 0.0:
                    probability = float(phase[f"{side}_contact_probability"][frame])
                    lift = params.swing_clearance_m * np.clip((0.55 - probability) / 0.55, 0.0, 1.0)
                    target_z = max(point[2], ground + lift)
                    error = np.array([target_z - point[2]])
                    task_rows = np.array([2])
                    gain = 0.65
                    swing_residuals.append(abs(float(error[0])))
                else:
                    continue

                jac = point_jacobian(model, data, body_id, point)[task_rows][:, side_dof_addresses[side]]
                normal = jac @ jac.T + 2.5e-3 * np.eye(len(task_rows))
                delta = gain * jac.T @ np.linalg.solve(normal, error)
                delta = np.clip(delta, -0.045, 0.045)
                dof[frame, side_indices[side]] += delta
                for local_index, joint_index in enumerate(side_indices[side]):
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[joint_index])
                    if bool(model.jnt_limited[joint_id]):
                        low, high = model.jnt_range[joint_id]
                        dof[frame, joint_index] = np.clip(dof[frame, joint_index], low, high)
        previous_stance = stance

    repaired = copy.deepcopy(entry)
    repaired["dof"] = dof.astype(np.float32)
    repaired["root_trans_offset"] = root_repaired.astype(np.float32)
    repaired["retarget_method"] = "phase2_joint_reference_repair_oracle"
    repaired["phase2_repair_parameters"] = asdict(params)
    return repaired, {
        "exact_noop": False,
        "root_correction_max_xy_m": float(np.max(np.linalg.norm(root_correction[:, :2], axis=1))),
        "root_correction_max_z_m": float(np.max(np.abs(root_correction[:, 2]))),
        "stance_ik_residual_p95_m": float(np.percentile(stance_residuals, 95)) if stance_residuals else 0.0,
        "swing_ik_residual_p95_m": float(np.percentile(swing_residuals, 95)) if swing_residuals else 0.0,
        "phase_counts": {name: int(np.count_nonzero(values)) for name, values in phase.items() if values.dtype == bool},
    }


def offline_metrics(model: mujoco.MjModel, entry: dict[str, Any]) -> dict[str, Any]:
    dof = np.asarray(entry["dof"], dtype=np.float64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    kin = reference_kinematics(model, entry)
    phase = phase_contract(kin)
    step = np.abs(np.diff(dof, axis=0))
    fps = float(entry["fps"])
    slips = []
    dcm_distances = []
    com = np.asarray(kin["com"])
    com_vel = np.asarray(kin["com_velocity"])
    ground = float(kin["ground_z_m"])
    omega_inv = np.sqrt(np.maximum(com[:, 2] - ground, 0.20) / 9.81)
    dcm = com[:, :2] + omega_inv[:, None] * com_vel[:, :2]
    for frame in range(len(dof)):
        support = []
        for side in ("left", "right"):
            stance_key = f"stance_{side}"
            if bool(phase[stance_key][frame] or phase["double_support"][frame]):
                support.append(np.asarray(kin[f"{side}_foot"])[frame, :2])
                if frame:
                    slips.append(float(np.linalg.norm(np.asarray(kin[f"{side}_velocity"])[frame, :2])))
        if support:
            support_array = np.asarray(support)
            if len(support_array) == 1:
                dcm_distances.append(float(np.linalg.norm(dcm[frame] - support_array[0])))
            else:
                start, end = support_array[0], support_array[1]
                vector = end - start
                alpha = np.clip(np.dot(dcm[frame] - start, vector) / (np.dot(vector, vector) + 1e-9), 0.0, 1.0)
                dcm_distances.append(float(np.linalg.norm(dcm[frame] - (start + alpha * vector))))
    left_swing = np.asarray(kin["left_foot"])[phase["left_swing"], 2] - ground
    right_swing = np.asarray(kin["right_foot"])[phase["right_swing"], 2] - ground
    return {
        "frames": len(dof),
        "duration_s": (len(dof) - 1) / fps,
        "joint_step_max_rad": float(np.max(step)) if step.size else 0.0,
        "joint_step_p95_rad": float(np.percentile(np.max(step, axis=1), 95)) if step.size else 0.0,
        "stance_slip_p95_mps_model_estimate": float(np.percentile(slips, 95)) if slips else None,
        "dcm_to_support_p95_m_model_estimate": float(np.percentile(dcm_distances, 95)) if dcm_distances else None,
        "swing_clearance_p50_m_model_estimate": {
            "left": float(np.percentile(left_swing, 50)) if left_swing.size else None,
            "right": float(np.percentile(right_swing, 50)) if right_swing.size else None,
        },
        "root_xy_displacement_m": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
    }


def offline_gate(candidate: dict[str, Any], baseline: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "joint_step_not_worse_than_110pct": candidate["joint_step_max_rad"] <= 1.10 * baseline["joint_step_max_rad"],
        "root_xy_correction_bounded": repair.get("root_correction_max_xy_m", 0.0) <= 0.0401,
        "root_z_correction_bounded": repair.get("root_correction_max_z_m", 0.0) <= 0.0251,
        "finite_metrics": all(
            np.isfinite(value)
            for value in (
                candidate["joint_step_max_rad"],
                candidate["joint_step_p95_rad"],
                candidate["root_xy_displacement_m"],
            )
        ),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def offline_cost(metrics: dict[str, Any], baseline: dict[str, Any], repair: dict[str, Any]) -> float:
    def ratio(key: str) -> float:
        value = metrics.get(key)
        reference = baseline.get(key)
        if value is None or reference is None or reference <= 1.0e-8:
            return 1.0
        return float(value / reference)

    clearance = [v for v in metrics["swing_clearance_p50_m_model_estimate"].values() if v is not None]
    clearance_reward = float(np.mean(clearance)) if clearance else 0.0
    return float(
        1.8 * ratio("stance_slip_p95_mps_model_estimate")
        + 1.2 * ratio("dcm_to_support_p95_m_model_estimate")
        + 1.0 * ratio("joint_step_p95_rad")
        - 3.0 * clearance_reward
        + 2.0 * repair.get("root_correction_max_xy_m", 0.0)
    )


def search_grid() -> list[RepairParameters]:
    # One coordinated intervention; this compact grid varies only its bounded
    # low-dimensional oracle parameters.  It intentionally excludes output
    # smoothing, reward changes and any policy/checkpoint update.
    values = itertools.product(
        (0.25, 0.50),       # root support shift
        (0.0, 0.50),        # bounded root-height correction
        (-2, 0, 2),         # contact-event timing
        (0.0, 0.25),        # current-v4 -> toe terminal blend
        (0.0, 0.020),       # swing sole clearance
    )
    full = [RepairParameters(*value) for value in values]
    # Put a small balanced covering set first so the default experiment does
    # not pretend that exhaustive 48-way tuning is necessary.  Callers may
    # still request the full grid explicitly.
    core = [
        RepairParameters(0.25, 0.00, 0, 0.00, 0.020),
        RepairParameters(0.25, 0.50, 0, 0.00, 0.020),
        RepairParameters(0.50, 0.00, 0, 0.00, 0.020),
        RepairParameters(0.50, 0.50, 0, 0.00, 0.020),
        RepairParameters(0.25, 0.00, -2, 0.00, 0.020),
        RepairParameters(0.25, 0.00, 2, 0.00, 0.020),
        RepairParameters(0.50, 0.50, -2, 0.00, 0.020),
        RepairParameters(0.50, 0.50, 2, 0.00, 0.020),
        RepairParameters(0.25, 0.00, 0, 0.25, 0.020),
        RepairParameters(0.50, 0.50, 0, 0.25, 0.020),
        RepairParameters(0.25, 0.50, -2, 0.25, 0.000),
        RepairParameters(0.50, 0.00, 2, 0.25, 0.000),
    ]
    return core + [value for value in full if value not in core]


def aggregate_physics(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    result = {}
    for variant in variants:
        result[variant] = {}
        for mode in ("prescribed_root_trackability", "free_root_balance"):
            subset = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
            result[variant][mode] = {
                "full_duration_count": int(sum(row["duration_fraction"] >= 0.999 for row in subset)),
                "motion_count": len(subset),
                "mean_duration_fraction": float(np.mean([row["duration_fraction"] for row in subset])),
                "mean_leg_rmse_rad": float(np.mean([row["joint_tracking"]["legs"]["rmse_rad"] for row in subset])),
                "mean_torque_saturation_fraction": float(np.mean([row["torque_saturation_fraction"] for row in subset])),
                "mean_contact_slip_p95_mps": float(np.mean([
                    value for row in subset for value in row["contact_slip_speed_p95_mps"].values() if value is not None
                ])),
                "mean_root_xy_error_rmse_m": float(np.mean([row["root_xy_error_rmse_m"] for row in subset])),
            }
    return result


def physics_cost(row: dict[str, Any], baseline: dict[str, Any]) -> float:
    slip = np.mean([value for value in row["contact_slip_speed_p95_mps"].values() if value is not None] or [5.0])
    baseline_slip = np.mean([value for value in baseline["contact_slip_speed_p95_mps"].values() if value is not None] or [5.0])
    return float(
        -4.0 * row["duration_fraction"]
        + 1.0 * slip / max(baseline_slip, 1.0e-3)
        + 0.5 * row["root_tilt_max_rad"]
        + 0.5 * row["torque_saturation_fraction"]
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# X2 Joint Reference Repair Phase2",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 本轮不训练、不载入 checkpoint、不接真机；唯一主干预是低维联合 reference repair。",
        "- COM、DCM、contact probability、支撑相位全部来自官方 AimDK v1.0 MuJoCo 模型估计，**不是真实足底力、COP 或实机 COM 真值**。",
        "- oracle 使用完整动作和官方物理结果选参，只用于回答‘是否存在可行修复’，不是可部署在线 adapter。",
        "",
        "## 假设 / 干预 / 对照",
        "",
        "- 假设：foot-only 修正失败，是因为 root/COM、接触时序、支撑脚锁定和摆脚 terminal 没有被联合处理。",
        "- 干预：在 current-v4 exact30 上联合搜索有界 root 支撑偏移/高度、±2 帧相位、stance-foot IK lock、0–2 cm 摆脚 clearance 与 0–25% foot-terminal blend。",
        "- 对照：current-v4 exact30 与已淘汰 toe-smooth9；官方 scene、PD、限矩、1 kHz physics、50 Hz control 完全相同。",
        "",
        "## 离线门",
        "",
        "| role | candidates | passed | selected offline cost | root XY/Z max(m) | joint step(rad) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role, value in report["offline_search"].items():
        selected = value["selected"]
        repair = selected["repair_diagnostics"]
        metrics = selected["metrics"]
        lines.append(
            f"| {role} | {value['candidate_count']} | {value['offline_pass_count']} | "
            f"{selected['offline_cost']:.4f} | {repair.get('root_correction_max_xy_m', 0):.4f}/"
            f"{repair.get('root_correction_max_z_m', 0):.4f} | {metrics['joint_step_max_rad']:.4f} |"
        )
    lines += [
        "",
        "## 官方物理结果",
        "",
        "| variant | mode | full | duration | leg RMSE | sat. | slip p95 | root XY RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, modes in report["physics_aggregate"].items():
        for mode, value in modes.items():
            lines.append(
                f"| {variant} | {mode} | {value['full_duration_count']}/{value['motion_count']} | "
                f"{value['mean_duration_fraction']:.3f} | {value['mean_leg_rmse_rad']:.4f} | "
                f"{value['mean_torque_saturation_fraction']:.4f} | {value['mean_contact_slip_p95_mps']:.4f} | "
                f"{value['mean_root_xy_error_rmse_m']:.4f} |"
            )
    lines += [
        "",
        "## 逐动作 free-root 对照",
        "",
        "| role | variant | duration | fall(s) | slip p95 mean(m/s) | root XY RMSE(m) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for role in DEFAULT_ROLES:
        for variant in ("current_v4_exact30", "toe_to_forefoot_exact30_smooth9", "joint_repair_oracle"):
            row = next(
                value for value in report["physics_results"]
                if value["panel_role"] == role
                and value["variant"] == variant
                and value["mode"] == "free_root_balance"
            )
            slip_values = [
                value for value in row["contact_slip_speed_p95_mps"].values() if value is not None
            ]
            slip = float(np.mean(slip_values)) if slip_values else float("nan")
            fall = "-" if row["fall_time_s"] is None else f"{row['fall_time_s']:.3f}"
            lines.append(
                f"| {role} | {variant} | {row['duration_fraction']:.3f} | {fall} | "
                f"{slip:.4f} | {row['root_xy_error_rmse_m']:.4f} |"
            )
    lines += [
        "",
        "## 结果 / 结论 / 下一步",
        "",
        f"- 结果：{report['decision']['result']} ",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
        "## 严格解释边界",
        "",
        "`prescribed_root_trackability` 每个物理步回写 root，只证明关节/力矩可跟踪；`free_root_balance` 不施加 root 外力，但仍只是开环 PD reference 可行性，不等价于闭环 RL 成功。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--toe", type=Path, default=DEFAULT_TOE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--max-offline-candidates", type=int, default=12)
    parser.add_argument("--physics-shortlist", type=int, default=2)
    args = parser.parse_args()

    physics = load_module(PHYSICS_SCRIPT, "x2_forefoot_official_physics_helpers_phase2")
    current_all = joblib.load(args.current)
    toe_all = joblib.load(args.toe)
    physics.validate_motion_pair(current_all, toe_all)
    role_to_key = {entry["panel_role"]: key for key, entry in current_all.items()}
    if any(role not in role_to_key for role in args.roles):
        raise ValueError("requested role missing from exact30 panel")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))

    grid = search_grid()[: args.max_offline_candidates]
    offline_search: dict[str, Any] = {}
    candidates_by_role: dict[str, list[tuple[float, RepairParameters, dict[str, Any], dict[str, Any], dict[str, Any]]]] = {}
    for role in args.roles:
        print(f"[offline] {role}", flush=True)
        key = role_to_key[role]
        current = current_all[key]
        toe = toe_all[key]
        baseline_metrics = offline_metrics(model, current)
        rows = []
        for params in grid:
            candidate, diagnostics = ik_repair(model, current, toe, params)
            metrics = offline_metrics(model, candidate)
            gate = offline_gate(metrics, baseline_metrics, diagnostics)
            cost = offline_cost(metrics, baseline_metrics, diagnostics)
            rows.append((cost, params, candidate, metrics, {"repair": diagnostics, "gate": gate}))
        passed = [row for row in rows if row[4]["gate"]["pass"]]
        if not passed:
            raise RuntimeError(
                f"{role}: no repair candidate passed the offline continuity/bounds gate; "
                "official physics is intentionally not started"
            )
        ranked = sorted(passed, key=lambda item: item[0])
        candidates_by_role[role] = ranked
        selected = ranked[0]
        offline_search[role] = {
            "motion_key": key,
            "baseline_metrics": baseline_metrics,
            "candidate_count": len(rows),
            "offline_pass_count": len(passed),
            "selected": {
                "parameters": asdict(selected[1]),
                "offline_cost": selected[0],
                "metrics": selected[3],
                "repair_diagnostics": selected[4]["repair"],
                "gate": selected[4]["gate"],
            },
        }

    # Evaluate fixed controls once, then the top offline candidates under
    # free-root physics.  Select the oracle winner per motion by one unified
    # physics cost; only that winner proceeds to prescribed-root reporting.
    physics_rows = []
    selected_cache = {}
    for role in args.roles:
        key = role_to_key[role]
        control_rows = {}
        for variant, entry in (("current_v4_exact30", current_all[key]), ("toe_to_forefoot_exact30_smooth9", toe_all[key])):
            for mode in physics.MODES:
                print(f"[physics] {role} / {variant} / {mode}", flush=True)
                row = physics.simulate_case(physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, entry, mode)
                row.update({"panel_role": role, "motion_key": key, "variant": variant})
                physics_rows.append(row)
                control_rows[(variant, mode)] = row
        baseline_free = control_rows[("current_v4_exact30", "free_root_balance")]
        scored = []
        for index, candidate_row in enumerate(candidates_by_role[role][: args.physics_shortlist]):
            _, params, candidate, _, _ = candidate_row
            variant = f"joint_repair_candidate_{index}"
            print(f"[physics] {role} / {variant} / free_root_balance", flush=True)
            free = physics.simulate_case(physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, candidate, "free_root_balance")
            free.update({"panel_role": role, "motion_key": key, "variant": variant, "parameters": asdict(params)})
            score = physics_cost(free, baseline_free)
            scored.append((score, candidate_row, free))
        score, winner, free = min(scored, key=lambda item: item[0])
        _, params, candidate, metrics, diagnostic = winner
        variant = "joint_repair_oracle"
        free["variant"] = variant
        free["physics_selection_cost"] = score
        physics_rows.append(free)
        print(f"[physics] {role} / {variant} / prescribed_root_trackability", flush=True)
        prescribed = physics.simulate_case(
            physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, candidate, "prescribed_root_trackability"
        )
        prescribed.update({
            "panel_role": role,
            "motion_key": key,
            "variant": variant,
            "parameters": asdict(params),
            "physics_selection_cost": score,
        })
        physics_rows.append(prescribed)
        candidate["phase2_oracle_physics_selection_cost"] = score
        selected_cache[key] = candidate
        offline_search[role]["selected_after_physics"] = {
            "parameters": asdict(params),
            "physics_cost": score,
            "offline_metrics": metrics,
            "repair_diagnostics": diagnostic["repair"],
        }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_cache, args.cache)
    variants = ["current_v4_exact30", "toe_to_forefoot_exact30_smooth9", "joint_repair_oracle"]
    aggregate = aggregate_physics(physics_rows, variants)
    baseline_free = aggregate["current_v4_exact30"]["free_root_balance"]
    oracle_free = aggregate["joint_repair_oracle"]["free_root_balance"]
    baseline_fixed = aggregate["current_v4_exact30"]["prescribed_root_trackability"]
    oracle_fixed = aggregate["joint_repair_oracle"]["prescribed_root_trackability"]
    survival_better = oracle_free["mean_duration_fraction"] > baseline_free["mean_duration_fraction"] + 0.02
    slip_not_worse = oracle_free["mean_contact_slip_p95_mps"] <= 1.10 * baseline_free["mean_contact_slip_p95_mps"]
    fixed_not_worse = (
        oracle_fixed["mean_leg_rmse_rad"] <= 1.10 * baseline_fixed["mean_leg_rmse_rad"]
        and oracle_fixed["mean_torque_saturation_fraction"] <= baseline_fixed["mean_torque_saturation_fraction"] + 0.02
    )
    joint_hypothesis_supported = bool(survival_better and slip_not_worse and fixed_not_worse)
    report = {
        "schema_version": "x2_joint_reference_repair_phase2_v1",
        "provenance": {
            "current": {"path": str(args.current), "sha256": sha256(args.current)},
            "toe": {"path": str(args.toe), "sha256": sha256(args.toe)},
            "scene": {"path": str(physics.DEFAULT_SCENE), "sha256": sha256(physics.DEFAULT_SCENE)},
            "control": {"path": str(physics.DEFAULT_CONTROL), "sha256": sha256(physics.DEFAULT_CONTROL)},
            "output_cache": {"path": str(args.cache), "sha256": sha256(args.cache)},
        },
        "truth_boundary": {
            "com_dcm_contact_source": "official AimDK v1.0 MuJoCo model estimate",
            "not_measured": ["hardware GRF", "hardware COP", "hardware COM", "real foot-contact truth"],
            "oracle_deployable": False,
            "training_or_checkpoint_load": False,
        },
        "search_contract": {
            "single_main_intervention": "coordinated low-dimensional joint reference repair",
            "candidate_count_per_motion": len(grid),
            "physics_shortlist_per_motion": args.physics_shortlist,
            "parameters": [asdict(value) for value in grid],
        },
        "offline_search": offline_search,
        "physics_results": physics_rows,
        "physics_aggregate": aggregate,
        "decision": {
            "checks": {
                "free_root_survival_improves_gt_0p02": survival_better,
                "free_root_slip_not_worse_10pct": slip_not_worse,
                "prescribed_root_trackability_not_worse": fixed_not_worse,
            },
            "joint_hypothesis_supported": joint_hypothesis_supported,
            "status": "JOINT_REPAIR_ORACLE_SUPPORTED" if joint_hypothesis_supported else "JOINT_REPAIR_ORACLE_NOT_SUPPORTED",
            "result": (
                "联合 oracle 同时改善 free-root 生存、保持滑移和 fixed-root 跟踪。"
                if joint_hypothesis_supported
                else "联合 oracle 未同时满足 free-root 生存、滑移与 fixed-root 跟踪三项门禁。"
            ),
            "conclusion": (
                "现有 reference 中存在低维可修复的 root-contact-foot 协调缺口，但该 per-motion oracle 尚不可部署。"
                if joint_hypothesis_supported
                else "现有低维参数化不足以证明 reference 可由短窗口联合修复变成动力学可行数据；不得解锁 WBT PPO。"
            ),
            "next_step": (
                "把 oracle 收敛到因果短窗口并在 held-out 动作复核，然后才考虑生成 Silver teacher。"
                if joint_hypothesis_supported
                else "停止脚部/平滑参数扫描，转向显式接触约束的轨迹优化或目标机器人原生动态 teacher。"
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
