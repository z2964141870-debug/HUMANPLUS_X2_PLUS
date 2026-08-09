#!/usr/bin/env python3
"""Audit and repair three WBT contracts before any further teacher search.

1. Paired official free-root replay with zero versus reference-frame0 qvel.
2. Contact events extracted from official prescribed-root MuJoCo contacts,
   compared against the previous FK height/speed labels.
3. A turn representation that contains root-yaw and foot-orientation tasks,
   validated first as an exact zero-update path only.

No policy, training, checkpoint, real robot, BASE code, or legacy data is
modified.  COM/DCM/contact remain official-simulator model estimates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import sys


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT = phase3.DEFAULT_CURRENT
DEFAULT_JSON = REPO / "reports/retarget/x2_wbt_contract_phase4.json"
DEFAULT_MD = REPO / "reports/retarget/x2_wbt_contract_phase4.md"
DEFAULT_CONTACT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase4_official_contact_contract/official_contact_schedule.pkl"
)
ROLES = ("walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand")


def reference_frame0_qvel(
    model: mujoco.MjModel, entry: dict[str, Any], physics_dt: float, physics
) -> np.ndarray:
    times = np.asarray([0.0, physics_dt], dtype=np.float64)
    qpos = physics.reference_qpos(entry, times)
    qvel = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel, physics_dt, qpos[0], qpos[1])
    return qvel


def contact_frame_labels(
    active_steps: dict[str, np.ndarray], sim_times: np.ndarray, frame_count: int, fps: float
) -> dict[str, np.ndarray]:
    labels = {}
    # Each exact-30 frame owns the half-open interval centred on that frame.
    # A contact is official-active only if it occupies at least half its 1 kHz
    # samples; no gap closing or hand smoothing is applied.
    frame_index = np.clip(np.floor(sim_times * fps + 0.5).astype(int), 0, frame_count - 1)
    for side in ("left", "right"):
        values = np.zeros(frame_count, dtype=bool)
        source = np.asarray(active_steps[side], dtype=bool)
        for frame in range(frame_count):
            selected = source[frame_index == frame]
            values[frame] = bool(selected.size and np.mean(selected) >= 0.50)
        labels[side] = values
    return labels


def event_frames(values: np.ndarray) -> dict[str, list[int]]:
    values = np.asarray(values, dtype=bool)
    return {
        "touchdown": np.flatnonzero(values & ~np.r_[False, values[:-1]]).astype(int).tolist(),
        "liftoff": np.flatnonzero(~values & np.r_[False, values[:-1]]).astype(int).tolist(),
    }


def nearest_event_errors(reference: list[int], candidate: list[int]) -> list[int]:
    if not reference or not candidate:
        return []
    return [int(min(abs(value - other) for other in candidate)) for value in reference]


def compare_contact_labels(fk: dict[str, np.ndarray], official: dict[str, np.ndarray]) -> dict[str, Any]:
    per_side = {}
    for side in ("left", "right"):
        truth = np.asarray(official[side], dtype=bool)
        prediction = np.asarray(fk[side], dtype=bool)
        tp = int(np.sum(truth & prediction))
        fp = int(np.sum(~truth & prediction))
        fn = int(np.sum(truth & ~prediction))
        tn = int(np.sum(~truth & ~prediction))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1.0e-12, precision + recall)
        official_events = event_frames(truth)
        fk_events = event_frames(prediction)
        errors = {
            name: nearest_event_errors(official_events[name], fk_events[name])
            for name in ("liftoff", "touchdown")
        }
        per_side[side] = {
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "agreement": float(np.mean(truth == prediction)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "official_events": official_events,
            "fk_events": fk_events,
            "nearest_event_error_frames": errors,
        }
    return {
        "per_side": per_side,
        "macro_agreement": float(np.mean([value["agreement"] for value in per_side.values()])),
        "macro_f1": float(np.mean([value["f1"] for value in per_side.values()])),
        "max_nearest_event_error_frames": max(
            [error for value in per_side.values() for errors in value["nearest_event_error_frames"].values() for error in errors]
            or [0]
        ),
    }


def official_geometry_clearance(model: mujoco.MjModel, entry: dict[str, Any], phase2, physics) -> dict[str, Any]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, _ = phase2.joint_addresses(model, names)
    floor, foot_geoms = physics.foot_geom_contract(model)
    data = mujoco.MjData(model)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    dof = np.asarray(entry["dof"], dtype=np.float64)
    distance = {side: [] for side in ("left", "right")}
    fromto = np.zeros(6, dtype=np.float64)
    for frame in range(len(dof)):
        phase2.set_reference_state(model, data, root[frame], quat[frame], dof[frame], qpos_addresses)
        for side in ("left", "right"):
            values = [
                float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto))
                for geom in foot_geoms[side]
            ]
            distance[side].append(min(values))
    per_side = {}
    for side, values in distance.items():
        array = np.asarray(values)
        signed_min = float(np.min(array))
        signed_p05 = float(np.percentile(array, 5))
        signed_p50 = float(np.percentile(array, 50))
        collision_fraction = float(np.mean(array <= 0.0))
        per_side[side] = {
            "signed_distance_min_m": signed_min,
            "signed_distance_p05_m": signed_p05,
            "signed_distance_p50_m": signed_p50,
            "collision_frame_fraction": collision_fraction,
            "within_5mm_frame_fraction": float(np.mean(array <= 0.005)),
            "hover_signature": bool(signed_p05 > 0.005 and collision_fraction < 0.10),
            "deep_penetration_signature": bool(signed_min < -0.020 or signed_p50 < -0.010),
        }
    return {
        "semantics": "official MuJoCo signed floor-to-foot-collision geometry distance; positive means separated",
        "per_side": per_side,
        "minimum_p05_m": float(min(value["signed_distance_p05_m"] for value in per_side.values())),
        "both_feet_hover_p05_gt_5mm": bool(all(value["signed_distance_p05_m"] > 0.005 for value in per_side.values())),
        "severe_root_ground_misalignment": bool(any(
            value["hover_signature"] or value["deep_penetration_signature"] for value in per_side.values()
        )),
    }


def simulate_contract_case(
    scene: Path,
    control_yaml: Path,
    entry: dict[str, Any],
    mode: str,
    initial_velocity: str,
    physics,
    phase2,
    expected_contact: dict[str, np.ndarray] | None = None,
    collect_contact_trace: bool = False,
    contact_activation: str = "collision",
) -> dict[str, Any]:
    if initial_velocity not in ("zero", "reference_frame0"):
        raise ValueError(initial_velocity)
    if contact_activation not in ("collision", "force_gt_1n"):
        raise ValueError(contact_activation)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    contract = physics.build_control_contract(model, control_yaml)
    if tuple(entry["joint_names_mujoco"]) != contract.joint_names:
        raise ValueError("joint order mismatch")
    duration = (len(entry["dof"]) - 1) / float(entry["fps"])
    n_steps = int(round(duration / model.opt.timestep))
    sim_times = np.minimum(np.arange(n_steps + 1) * model.opt.timestep, duration)
    qpos_ref = physics.reference_qpos(entry, sim_times)
    control_times = np.minimum(
        np.floor(sim_times / contract.control_dt + 1.0e-10) * contract.control_dt, duration
    )
    _, _, dof_control = physics.interpolate_motion(entry, control_times)
    qvel0_reference = reference_frame0_qvel(model, entry, model.opt.timestep, physics)
    qvel_root_reference = np.zeros((n_steps + 1, model.nv)) if mode == "prescribed_root_trackability" else None
    if qvel_root_reference is not None:
        for index in range(n_steps):
            mujoco.mj_differentiatePos(
                model, qvel_root_reference[index], model.opt.timestep, qpos_ref[index], qpos_ref[index + 1]
            )
        qvel_root_reference[-1] = qvel_root_reference[-2]

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = 0.0 if initial_velocity == "zero" else qvel0_reference
    mujoco.mj_forward(model, data)
    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    entry_index = {name: index for index, name in enumerate(contract.joint_names)}
    target_indices = np.asarray([entry_index[name] for name in contract.actuator_joint_names])
    previous_geom_pos = data.geom_xpos.copy()
    contact_steps = {side: np.zeros(n_steps + 1, dtype=bool) for side in ("left", "right")}
    slips = {"left": [], "right": []}
    contact_match = []
    tilts = []
    root_z = []
    saturation = 0
    effort = 0
    achieved = n_steps
    fall_time = None
    force_buffer = np.zeros(6)

    for step in range(n_steps):
        if mode == "prescribed_root_trackability":
            data.qpos[:7] = qpos_ref[step, :7]
            data.qvel[:6] = qvel_root_reference[step, :6]
        target = dof_control[step, target_indices]
        actual_q = data.qpos[contract.qpos_addresses]
        actual_dq = data.qvel[contract.qvel_addresses]
        raw = contract.kp * (target - actual_q) - contract.kd * actual_dq
        clipped = np.clip(raw, contract.torque_low, contract.torque_high)
        saturation += int(np.count_nonzero(raw != clipped))
        effort += model.nu
        data.ctrl[:] = clipped
        mujoco.mj_step(model, data)
        if mode == "prescribed_root_trackability":
            data.qpos[:7] = qpos_ref[step + 1, :7]
            data.qvel[:6] = qvel_root_reference[step + 1, :6]
            mujoco.mj_forward(model, data)

        active_geoms = {"left": set(), "right": set()}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            other = geom2 if geom1 == floor else geom1 if geom2 == floor else -1
            side = geom_side.get(other)
            if side is None:
                continue
            mujoco.mj_contactForce(model, data, contact_id, force_buffer)
            if contact_activation == "collision" or force_buffer[0] > 1.0:
                active_geoms[side].add(other)
        for side in ("left", "right"):
            active = bool(active_geoms[side])
            contact_steps[side][step + 1] = active
            if active:
                for geom in active_geoms[side]:
                    slips[side].append(
                        float(np.linalg.norm(data.geom_xpos[geom, :2] - previous_geom_pos[geom, :2]) / model.opt.timestep)
                    )
            if expected_contact is not None:
                frame = min(len(expected_contact[side]) - 1, int(round((step + 1) * model.opt.timestep * entry["fps"])))
                contact_match.append(float(active == bool(expected_contact[side][frame])))
        previous_geom_pos[:] = data.geom_xpos
        tilt = phase3.root_tilt(data.qpos[3:7])
        tilts.append(tilt)
        root_z.append(float(data.qpos[2]))
        if mode == "free_root_balance" and (data.qpos[2] < phase3.FALL_ROOT_Z_M or tilt > phase3.FALL_TILT_RAD):
            achieved = step + 1
            fall_time = achieved * model.opt.timestep
            break

    simulated = min(achieved * model.opt.timestep, duration)
    slip_values = [value for values in slips.values() for value in values]
    result = {
        "mode": mode,
        "initial_velocity": initial_velocity,
        "contact_activation": contact_activation,
        "reference_duration_s": duration,
        "simulated_duration_s": simulated,
        "duration_fraction": float(simulated / duration),
        "fell": fall_time is not None,
        "fall_time_s": fall_time,
        "root_tilt_max_rad": float(max(tilts, default=phase3.root_tilt(data.qpos[3:7]))),
        "root_z_min_m": float(min(root_z, default=data.qpos[2])),
        "stance_slip_p95_mps": float(np.percentile(slip_values, 95)) if slip_values else 0.0,
        "contact_match_fraction_to_official_schedule": float(np.mean(contact_match)) if contact_match else None,
        "torque_saturation_fraction": float(saturation / effort) if effort else 0.0,
        "initial_reference_qvel": {
            "root_linear_norm_mps": float(np.linalg.norm(qvel0_reference[:3])),
            "root_angular_norm_radps": float(np.linalg.norm(qvel0_reference[3:6])),
            "joint_p95_abs_radps": float(np.percentile(np.abs(qvel0_reference[6:]), 95)),
            "joint_max_abs_radps": float(np.max(np.abs(qvel0_reference[6:]))),
        },
    }
    if collect_contact_trace:
        # Preserve the full-duration trace for prescribed-root extraction.
        result["contact_trace"] = {
            "sim_times_s": sim_times.tolist(),
            "active": {side: values.tolist() for side, values in contact_steps.items()},
        }
    return result


@dataclass(frozen=True)
class TurnOrientationVector:
    root_yaw_knots_rad: tuple[float, float, float]
    landing_foot_yaw_rad: float
    stance_orientation_gain: float

    @property
    def is_zero(self) -> bool:
        return (
            all(value == 0.0 for value in self.root_yaw_knots_rad)
            and self.landing_foot_yaw_rad == 0.0
            and self.stance_orientation_gain == 0.0
        )


def build_turn_orientation_teacher(
    model: mujoco.MjModel,
    entry: dict[str, Any],
    official_contact: dict[str, np.ndarray],
    vector: TurnOrientationVector,
    phase2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if vector.is_zero:
        result = copy.deepcopy(entry)
        result["phase4_turn_orientation_vector"] = asdict(vector)
        return result, {"exact_zero_update": True, "orientation_residual_p95_rad": 0.0}

    result = copy.deepcopy(entry)
    names = list(entry["joint_names_mujoco"])
    dof_ref = np.asarray(entry["dof"], dtype=np.float64)
    dof = dof_ref.copy()
    root_rot = Rotation.from_quat(np.asarray(entry["root_rot"], dtype=np.float64))
    frames = len(dof)
    swing = {side: ~np.asarray(official_contact[side], dtype=bool) for side in ("left", "right")}
    events = []
    for side in ("left", "right"):
        events.extend((start, side, end) for start, end in phase3.boolean_intervals(swing[side], 5))
    if not events:
        raise ValueError("official schedule has no turn swing event")
    start, swing_side, end = min(events)
    active_end = min(frames - 1, end + 15)
    yaw_residual = phase3.spline_values(vector.root_yaw_knots_rad, frames, end)
    root_rot_new = root_rot * Rotation.from_euler("z", yaw_residual)

    qpos_addresses, qvel_addresses = phase2.joint_addresses(model, names)
    body_ids = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, phase2.FOOT_BODY[side]) for side in ("left", "right")}
    side_indices = {
        side: np.asarray([i for i, name in enumerate(names) if name.startswith(phase2.LEG_PREFIX[side]) and any(
            token in name for token in ("hip", "knee", "ankle")
        )], dtype=np.int64)
        for side in ("left", "right")
    }
    side_dofs = {side: qvel_addresses[index] for side, index in side_indices.items()}
    root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    data = mujoco.MjData(model)
    baseline_pos = {side: np.zeros((frames, 3)) for side in ("left", "right")}
    baseline_rot = {side: [] for side in ("left", "right")}
    for frame in range(frames):
        phase2.set_reference_state(model, data, root_pos[frame], np.asarray(entry["root_rot"])[frame], dof_ref[frame], qpos_addresses)
        for side in ("left", "right"):
            baseline_pos[side][frame] = phase2.point_world(data, body_ids[side], phase2.SOLE_POINT_LOCAL[side])
            baseline_rot[side].append(Rotation.from_matrix(data.xmat[body_ids[side]].reshape(3, 3)))
    anchors = {side: None for side in ("left", "right")}
    orientation_residuals = []
    for frame in range(active_end + 1):
        if frame:
            dof[frame] = dof[frame - 1] + (dof_ref[frame] - dof_ref[frame - 1])
        for _ in range(5):
            phase2.set_reference_state(
                model, data, root_pos[frame], root_rot_new[frame].as_quat(), dof[frame], qpos_addresses
            )
            for side in ("left", "right"):
                contact = bool(official_contact[side][frame])
                if contact and anchors[side] is None:
                    anchors[side] = (baseline_pos[side][frame].copy(), baseline_rot[side][frame])
                if not contact:
                    anchors[side] = None
                current_pos = phase2.point_world(data, body_ids[side], phase2.SOLE_POINT_LOCAL[side])
                current_rot = Rotation.from_matrix(data.xmat[body_ids[side]].reshape(3, 3))
                if contact and anchors[side] is not None:
                    target_pos, target_rot = anchors[side]
                    orientation_gain = vector.stance_orientation_gain
                else:
                    target_pos = baseline_pos[side][frame]
                    progress = np.clip((frame - start) / max(1, end - start), 0.0, 1.0)
                    extra_yaw = vector.landing_foot_yaw_rad * progress if side == swing_side else 0.0
                    target_rot = baseline_rot[side][frame] * Rotation.from_euler("z", extra_yaw)
                    orientation_gain = 1.0
                pos_error = target_pos - current_pos
                rot_error = (current_rot.inv() * target_rot).as_rotvec()
                orientation_residuals.append(float(np.linalg.norm(rot_error)))
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, jacr, current_pos, body_ids[side])
                jac = np.vstack([jacp[:, side_dofs[side]], orientation_gain * jacr[:, side_dofs[side]]])
                error = np.r_[pos_error, orientation_gain * rot_error]
                normal = jac @ jac.T + 7.5e-3 * np.eye(6)
                delta = 0.55 * jac.T @ np.linalg.solve(normal, error)
                dof[frame, side_indices[side]] += np.clip(delta, -0.04, 0.04)
        if frame:
            dof[frame] = np.clip(dof[frame], dof[frame - 1] - 0.18, dof[frame - 1] + 0.18)
    for frame in range(active_end + 1, frames):
        # Return to the original trajectory through the same collocation
        # velocity box; never introduce a hidden exit-boundary jump.
        dof[frame] = np.clip(dof_ref[frame], dof[frame - 1] - 0.18, dof[frame - 1] + 0.18)
    result["dof"] = dof.astype(np.float32)
    result["root_rot"] = root_rot_new.as_quat().astype(np.float32)
    result["phase4_turn_orientation_vector"] = asdict(vector)
    return result, {
        "exact_zero_update": False,
        "swing_side": swing_side,
        "event_frames": [start, end],
        "root_yaw_residual_max_rad": float(np.max(np.abs(yaw_residual))),
        "orientation_residual_p95_rad": float(np.percentile(orientation_residuals, 95)),
        "joint_step_max_rad": float(np.max(np.abs(np.diff(dof, axis=0)))),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT Contract Phase4",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 无训练、无 checkpoint、无真机；COM/DCM/contact 仍是官方 MuJoCo 模型估计。",
        "",
        "## 1. frame0 qvel paired A/B",
        "",
        "| role | zero(s) | ref-qvel(s) | Δ(s) | zero/ref slip | ref root lin/ang | ref joint p95/max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["initial_velocity_ab"][role]
        zero, ref = value["zero"], value["reference_frame0"]
        qvel = ref["initial_reference_qvel"]
        lines.append(
            f"| {role} | {zero['simulated_duration_s']:.3f} | {ref['simulated_duration_s']:.3f} | "
            f"{value['survival_delta_s']:.3f} | {zero['stance_slip_p95_mps']:.4f}/{ref['stance_slip_p95_mps']:.4f} | "
            f"{qvel['root_linear_norm_mps']:.3f}/{qvel['root_angular_norm_radps']:.3f} | "
            f"{qvel['joint_p95_abs_radps']:.3f}/{qvel['joint_max_abs_radps']:.3f} |"
        )
    lines += [
        "",
        "## 2. official contact vs FK contact",
        "",
        "| role | agreement | macro F1 | max event error(frames) | floor distance L/R p05(m) | collision L/R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["contact_contract"][role]["comparison"]
        geometry = report["contact_contract"][role].get("official_geometry_clearance")
        distance = (
            f"{geometry['per_side']['left']['signed_distance_p05_m']:.4f}/"
            f"{geometry['per_side']['right']['signed_distance_p05_m']:.4f}"
            if geometry else "n/a"
        )
        collision = (
            f"{geometry['per_side']['left']['collision_frame_fraction']:.3f}/"
            f"{geometry['per_side']['right']['collision_frame_fraction']:.3f}"
            if geometry else "n/a"
        )
        lines.append(
            f"| {role} | {value['macro_agreement']:.3f} | {value['macro_f1']:.3f} | "
            f"{value['max_nearest_event_error_frames']} | {distance} | {collision} |"
        )
    lines += [
        "",
        "## 3. turn orientation representation zero-update gate",
        "",
        "| role | arrays exact | metric exact | survival(s) |",
        "|---|---:|---:|---:|",
    ]
    for role, value in report["turn_zero_update_gate"].items():
        lines.append(
            f"| {role} | {value['arrays_exact']} | {value['physics_metrics_exact']} | "
            f"{value['adapter_result']['simulated_duration_s']:.3f} |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
        "turn 表示的零更新门只证明新代码不破坏旧参考；它不证明非零 yaw/orientation teacher 有效。contact/FK 不一致属于生成器与标签契约问题，不能写成 X2 动力学不可执行。",
        "",
    ]
    return "\n".join(lines)


def geometry_audit_existing(args, phase2, physics, motions, role_to_key, model) -> None:
    report = json.loads(args.json.read_text(encoding="utf-8"))
    report["decision"]["checks"].pop("official_geometry_hover_detected", None)
    report["decision"].pop("official_geometry_hover_roles", None)
    hover_roles = []
    for role in ROLES:
        geometry = official_geometry_clearance(model, motions[role_to_key[role]], phase2, physics)
        report["contact_contract"][role]["official_geometry_clearance"] = geometry
        if geometry["severe_root_ground_misalignment"]:
            hover_roles.append(role)
    report["truth_boundary"]["official_contact_interpretation"] = (
        "collision events are authoritative for the current official geometry, but sparse events can expose "
        "root-z/ground misalignment rather than intended gait phase"
    )
    report["decision"]["checks"]["official_geometry_root_ground_misalignment_detected"] = bool(hover_roles)
    report["decision"]["official_geometry_misaligned_roles"] = hover_roles
    if hover_roles:
        report["decision"].update({
            "status": "CONTRACT_AUDIT_FOUND_REFERENCE_START_AND_ROOT_GROUND_MISMATCH",
            "conclusion": (
                "raw frame0 qvel 不安全；官方几何确认walk存在悬空、turn/start-stop存在厘米级穿地。当前contact事件首先反映root-z/ground生成器错误，不是X2动力学否证。"
            ),
            "next_step": (
                "先修reference起始连续性与官方floor几何对齐，再重新提取official collision contact；turn非零panel继续锁定。"
            ),
        })
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--contact-cache", type=Path, default=DEFAULT_CONTACT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--geometry-audit-existing", action="store_true")
    args = parser.parse_args()

    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase4")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase4")
    motions = joblib.load(args.current)
    role_to_key = {entry["panel_role"]: key for key, entry in motions.items()}
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    if args.geometry_audit_existing:
        geometry_audit_existing(args, phase2, physics, motions, role_to_key, model)
        return

    contact_contract = {}
    contact_cache = {}
    initial_velocity_ab = {}
    for role in ROLES:
        print(f"[phase4] official contact extract: {role}", flush=True)
        key = role_to_key[role]
        entry = motions[key]
        trace_result = simulate_contract_case(
            physics.DEFAULT_SCENE,
            physics.DEFAULT_CONTROL,
            entry,
            "prescribed_root_trackability",
            "reference_frame0",
            physics,
            phase2,
            collect_contact_trace=True,
        )
        trace = trace_result.pop("contact_trace")
        official = contact_frame_labels(
            {side: np.asarray(trace["active"][side], dtype=bool) for side in ("left", "right")},
            np.asarray(trace["sim_times_s"]),
            len(entry["dof"]),
            float(entry["fps"]),
        )
        kin = phase2.reference_kinematics(model, entry)
        phase = phase2.phase_contract(kin)
        fk = {side: ~phase[f"{side}_swing"] for side in ("left", "right")}
        comparison = compare_contact_labels(fk, official)
        contact_cache[key] = {
            "panel_role": role,
            "fps": int(entry["fps"]),
            "official_contact": {side: values.astype(np.uint8) for side, values in official.items()},
            "fk_contact": {side: values.astype(np.uint8) for side, values in fk.items()},
        }
        contact_contract[role] = {
            "motion_key": key,
            "extraction": {
                "mode": "official prescribed-root replay",
                "initial_velocity": "reference_frame0",
                "contact_activation": "official MuJoCo floor-foot collision object (not GRF)",
                "frame_occupancy_threshold": 0.50,
                "smoothing": False,
            },
            "comparison": comparison,
        }

        paired = {}
        for initial_velocity in ("zero", "reference_frame0"):
            print(f"[phase4] qvel AB: {role} / {initial_velocity}", flush=True)
            paired[initial_velocity] = simulate_contract_case(
                physics.DEFAULT_SCENE,
                physics.DEFAULT_CONTROL,
                entry,
                "free_root_balance",
                initial_velocity,
                physics,
                phase2,
                expected_contact=official,
            )
        paired["survival_delta_s"] = (
            paired["reference_frame0"]["simulated_duration_s"] - paired["zero"]["simulated_duration_s"]
        )
        initial_velocity_ab[role] = paired

    args.contact_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(contact_cache, args.contact_cache)

    zero_vector = TurnOrientationVector((0.0, 0.0, 0.0), 0.0, 0.0)
    turn_zero_gate = {}
    for role in ("turn_left", "turn_right"):
        key = role_to_key[role]
        entry = motions[key]
        official = {
            side: np.asarray(contact_cache[key]["official_contact"][side], dtype=bool)
            for side in ("left", "right")
        }
        adapted, diagnostics = build_turn_orientation_teacher(model, entry, official, zero_vector, phase2)
        arrays_exact = all(
            np.array_equal(np.asarray(entry[name]), np.asarray(adapted[name]))
            for name in ("dof", "root_rot", "root_trans_offset")
        )
        baseline = initial_velocity_ab[role]["reference_frame0"]
        adapter_result = simulate_contract_case(
            physics.DEFAULT_SCENE,
            physics.DEFAULT_CONTROL,
            adapted,
            "free_root_balance",
            "reference_frame0",
            physics,
            phase2,
            expected_contact=official,
        )
        metric_keys = ("simulated_duration_s", "root_tilt_max_rad", "root_z_min_m", "stance_slip_p95_mps")
        metrics_exact = all(np.isclose(adapter_result[name], baseline[name], atol=1e-12) for name in metric_keys)
        turn_zero_gate[role] = {
            "vector": asdict(zero_vector),
            "diagnostics": diagnostics,
            "arrays_exact": arrays_exact,
            "physics_metrics_exact": metrics_exact,
            "adapter_result": adapter_result,
        }

    mean_delta = float(np.mean([value["survival_delta_s"] for value in initial_velocity_ab.values()]))
    qvel_material = bool(any(abs(value["survival_delta_s"]) >= 0.10 for value in initial_velocity_ab.values()))
    qvel_safe = bool(all(value["survival_delta_s"] >= -0.05 for value in initial_velocity_ab.values()))
    contact_mismatch = bool(any(value["comparison"]["macro_agreement"] < 0.90 for value in contact_contract.values()))
    zero_gate = bool(all(value["arrays_exact"] and value["physics_metrics_exact"] for value in turn_zero_gate.values()))
    report = {
        "schema_version": "x2_wbt_contract_phase4_v1",
        "provenance": {
            "current": {"path": str(args.current), "sha256": phase3.sha256(args.current)},
            "scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase3.sha256(physics.DEFAULT_SCENE)},
            "control": {"path": str(physics.DEFAULT_CONTROL), "sha256": phase3.sha256(physics.DEFAULT_CONTROL)},
            "contact_cache": {"path": str(args.contact_cache), "sha256": phase3.sha256(args.contact_cache)},
        },
        "truth_boundary": {
            "contact_source": "official prescribed-root MuJoCo floor-foot collision object, >=50% frame occupancy",
            "com_dcm_contact_are_model_estimates": True,
            "not_hardware_truth": True,
            "training_or_checkpoint_load": False,
            "turn_nonzero_search_executed": False,
        },
        "initial_velocity_ab": initial_velocity_ab,
        "contact_contract": contact_contract,
        "turn_representation": {
            "new_variables": ["root_yaw_knots_rad", "landing_foot_yaw_rad", "stance_orientation_gain"],
            "tasks": ["stance foot XYZ+orientation", "swing foot XYZ+landing orientation"],
            "nonzero_panel_preregistered_but_not_run": {
                "roles": ["turn_left", "turn_right"],
                "required_before_run": [
                    "official contact schedule", "reference_frame0 qvel", "zero-update exact gate", "fixed candidate vectors before physics"
                ],
                "promotion_gate": "survival +0.5s, official-contact slip not worse, prescribed action/torque smoothness",
            },
        },
        "turn_zero_update_gate": turn_zero_gate,
        "decision": {
            "checks": {
                "initial_qvel_has_material_effect_on_at_least_one_action": qvel_material,
                "raw_reference_qvel_safe_to_adopt": qvel_safe,
                "fk_contact_mismatch_detected": contact_mismatch,
                "turn_orientation_zero_update_exact": zero_gate,
            },
            "mean_reference_qvel_survival_delta_s": mean_delta,
            "status": (
                "CONTRACT_REPAIR_READY_FOR_PREREGISTERED_TURN_PANEL"
                if zero_gate and qvel_safe
                else "CONTRACT_AUDIT_FOUND_REFERENCE_START_DISCONTINUITY"
            ),
            "result": (
                f"reference-frame0 qvel paired mean survival delta={mean_delta:+.3f}s; "
                f"contact mismatch={contact_mismatch}; turn zero-update exact={zero_gate}."
            ),
            "conclusion": (
                "raw frame0 qvel 在当前 reference 起点不安全，必须先修复起始连续性；official collision contact 已替换FK标签。turn表示虽通过零更新门，仍禁止非零实验。"
                if zero_gate and not qvel_safe
                else "初始化/接触标签属于可证实的评估与生成器契约；turn表示代码已通过零更新门，但尚无非零效果证据。"
                if zero_gate and qvel_safe
                else "turn表示零更新门失败，禁止任何非零实验。"
            ),
            "next_step": (
                "冻结official collision contact；寻找reset-compatible静止起点或连续warm-start，禁止直接采用raw frame0 qvel，也不运行turn非零panel。"
                if zero_gate and not qvel_safe
                else "冻结官方contact与reference-qvel契约，先写死两个turn候选再做最小paired panel；不得解释为动力学已解决。"
                if zero_gate and qvel_safe
                else "修复零更新等价性，不运行turn物理候选。"
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
