#!/usr/bin/env python3
"""Contact-constrained dynamic-teacher existence test for X2 WBT.

This is intentionally not another foot-offset, output-smoothing, or small root
grid experiment.  A low-dimensional spline/event vector defines a complete
first-step intent: model-estimated COM lateral path, root height path, contact
switch times, stance-foot anchors, swing clearance, and landing terminal.  A
contact-aware IK layer converts that intent to 31-DOF targets.  CEM then selects
the vector *only by free-root rollout in the official AimDK v1.0 MuJoCo model*.

All COM/DCM/contact values are simulator-model estimates.  They are not real
robot GRF, COP, COM, or contact measurements.  The oracle uses the full motion
and simulator feedback and is not deployable.  No training/checkpoint/robot is
used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline


REPO = Path(__file__).resolve().parents[2]
PHASE2_SCRIPT = REPO / "tools/retarget/run_x2_joint_reference_repair_phase2.py"
PHYSICS_SCRIPT = REPO / "tools/retarget/run_x2_forefoot_official_physics_screen.py"
DEFAULT_CURRENT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_smoothing_ab/current_v4_exact30/"
    "x2_current_v4_exact30.pkl"
)
DEFAULT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase3_contact_dynamic_teacher/"
    "x2_contact_constrained_dynamic_teacher_phase3.pkl"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_contact_constrained_dynamic_teacher_phase3.json"
DEFAULT_MD = REPO / "reports/retarget/x2_contact_constrained_dynamic_teacher_phase3.md"

FALL_ROOT_Z_M = 0.42
FALL_TILT_RAD = 0.90
SMOKE_ROLE = "walk_straight"


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
class TeacherVector:
    com_lateral_knots_m: tuple[float, float, float, float]
    root_height_knots_m: tuple[float, float, float]
    liftoff_shift_frames: int
    touchdown_shift_frames: int
    swing_clearance_m: float
    landing_dx_m: float
    landing_dy_m: float


LOW = np.array([-0.08] * 4 + [-0.05] * 3 + [-12, -12, 0.025, -0.08, -0.08])
HIGH = np.array([0.08] * 4 + [0.05] * 3 + [12, 12, 0.12, 0.08, 0.08])


def decode_vector(raw: np.ndarray) -> TeacherVector:
    value = np.clip(np.asarray(raw, dtype=np.float64), LOW, HIGH)
    return TeacherVector(
        com_lateral_knots_m=tuple(float(x) for x in value[:4]),
        root_height_knots_m=tuple(float(x) for x in value[4:7]),
        liftoff_shift_frames=int(round(float(value[7]))),
        touchdown_shift_frames=int(round(float(value[8]))),
        swing_clearance_m=float(value[9]),
        landing_dx_m=float(value[10]),
        landing_dy_m=float(value[11]),
    )


def encode_vector(value: TeacherVector) -> np.ndarray:
    return np.asarray(
        list(value.com_lateral_knots_m)
        + list(value.root_height_knots_m)
        + [
            value.liftoff_shift_frames,
            value.touchdown_shift_frames,
            value.swing_clearance_m,
            value.landing_dx_m,
            value.landing_dy_m,
        ],
        dtype=np.float64,
    )


def boolean_intervals(values: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start + 1 >= min_length]


def first_swing_event(phase: dict[str, np.ndarray], min_length: int = 8) -> tuple[str, int, int]:
    events = []
    for side in ("left", "right"):
        events.extend((start, side, end) for start, end in boolean_intervals(phase[f"{side}_swing"], min_length))
    if not events:
        raise ValueError("no non-trivial swing event in smoke reference")
    start, side, end = min(events)
    return side, start, end


def event_schedule(
    phase: dict[str, np.ndarray], vector: TeacherVector
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    side, original_start, original_end = first_swing_event(phase)
    frames = len(phase["double_support"])
    start = int(np.clip(original_start + vector.liftoff_shift_frames, 8, frames - 12))
    end = int(np.clip(original_end + vector.touchdown_shift_frames, start + 8, frames - 2))
    result = {
        "left_swing": np.asarray(phase["left_swing"], dtype=bool).copy(),
        "right_swing": np.asarray(phase["right_swing"], dtype=bool).copy(),
    }
    result[f"{side}_swing"][original_start : original_end + 1] = False
    result[f"{side}_swing"][start : end + 1] = True
    # Never ask both feet to swing in this existence smoke.
    other = "right" if side == "left" else "left"
    result[f"{other}_swing"][start : end + 1] = False
    result["left_contact"] = ~result["left_swing"]
    result["right_contact"] = ~result["right_swing"]
    result["double_support"] = result["left_contact"] & result["right_contact"]
    return result, {
        "swing_side": side,
        "original_liftoff_frame": original_start,
        "original_touchdown_frame": original_end,
        "optimized_liftoff_frame": start,
        "optimized_touchdown_frame": end,
    }


def spline_values(knots: tuple[float, ...], frames: int, end_frame: int) -> np.ndarray:
    end_frame = int(np.clip(end_frame, len(knots) - 1, frames - 1))
    x = np.linspace(0.0, float(end_frame), len(knots))
    values = CubicSpline(x, np.asarray(knots), bc_type="natural")(
        np.minimum(np.arange(frames, dtype=np.float64), float(end_frame))
    )
    # The teacher intervention is a first-step window; decay back to zero over
    # 0.5 s instead of silently changing the whole clip.
    tail = min(frames - 1, end_frame + 15)
    if tail > end_frame:
        values[end_frame : tail + 1] = np.linspace(values[end_frame], 0.0, tail - end_frame + 1)
    if tail + 1 < frames:
        values[tail + 1 :] = 0.0
    return np.asarray(values, dtype=np.float64)


def target_contact_points(
    base_kin: dict[str, Any], schedule: dict[str, np.ndarray], event: dict[str, Any], vector: TeacherVector
) -> dict[str, np.ndarray]:
    frames = len(schedule["double_support"])
    foot = {
        side: np.asarray(base_kin[f"{side}_foot"], dtype=np.float64)
        for side in ("left", "right")
    }
    target = {side: foot[side].copy() for side in foot}
    ground = float(base_kin["ground_z_m"])
    for side in ("left", "right"):
        anchor = None
        previous_contact = False
        for frame in range(frames):
            contact = bool(schedule[f"{side}_contact"][frame])
            if contact and (anchor is None or not previous_contact):
                anchor = foot[side][frame].copy()
                anchor[2] = ground
            if contact and anchor is not None:
                target[side][frame] = anchor
            previous_contact = contact

    side = event["swing_side"]
    start = event["optimized_liftoff_frame"]
    end = event["optimized_touchdown_frame"]
    takeoff = target[side][max(0, start - 1)].copy()
    landing_frame = min(frames - 1, end + 1)
    landing = foot[side][landing_frame].copy()
    landing[:2] += np.array([vector.landing_dx_m, vector.landing_dy_m])
    landing[2] = ground
    for frame in range(start, end + 1):
        phase = (frame - start) / max(1, end - start)
        point = (1.0 - phase) * takeoff + phase * landing
        point[2] = ground + vector.swing_clearance_m * 4.0 * phase * (1.0 - phase)
        target[side][frame] = point
    return target


def build_teacher_entry(
    model: mujoco.MjModel,
    entry: dict[str, Any],
    vector: TeacherVector,
    phase2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = list(entry["joint_names_mujoco"])
    dof_reference = np.asarray(entry["dof"], dtype=np.float64)
    dof = dof_reference.copy()
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64).copy()
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    frames = len(dof)
    qpos_addresses, qvel_addresses = phase2.joint_addresses(model, names)
    base_kin = phase2.reference_kinematics(model, entry)
    base_phase = phase2.phase_contract(base_kin)
    schedule, event = event_schedule(base_phase, vector)
    target_foot = target_contact_points(base_kin, schedule, event, vector)

    window_end = event["optimized_touchdown_frame"]
    lateral = spline_values(vector.com_lateral_knots_m, frames, window_end)
    height = spline_values(vector.root_height_knots_m, frames, window_end)
    com = np.asarray(base_kin["com"])
    # Zero spline is an exact COM-path no-op.  The CEM variable is a lateral
    # correction to the existing model-estimated COM trajectory, not an
    # implicit 10 cm teleport to the support centre.
    desired_com_y = com[:, 1] + lateral
    root[:, 1] += np.clip(desired_com_y - com[:, 1], -0.10, 0.10)
    root[:, 2] += np.clip(height, -0.06, 0.06)

    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, phase2.FOOT_BODY[side])
        for side in ("left", "right")
    }
    side_indices = {
        side: np.asarray(
            [
                i for i, name in enumerate(names)
                if name.startswith(phase2.LEG_PREFIX[side])
                and any(token in name for token in ("hip", "knee", "ankle"))
            ],
            dtype=np.int64,
        )
        for side in ("left", "right")
    }
    side_dof_addresses = {side: qvel_addresses[index] for side, index in side_indices.items()}
    data = mujoco.MjData(model)
    active_end = min(frames - 1, event["optimized_touchdown_frame"] + 15)
    for frame in range(frames):
        if frame > active_end:
            dof[frame] = dof_reference[frame]
        else:
            ramp_in = min(1.0, frame / 15.0)
            ramp_out = (
                1.0
                if frame <= event["optimized_touchdown_frame"]
                else max(0.0, (active_end - frame) / max(1, active_end - event["optimized_touchdown_frame"]))
            )
            constraint_weight = ramp_in * ramp_out
            if frame:
                # Warm-start from the previous IK branch while preserving the
                # original reference increment.  This is a direct continuity
                # constraint, not output smoothing after the fact.
                predicted = dof[frame - 1] + (dof_reference[frame] - dof_reference[frame - 1])
                dof[frame] = dof_reference[frame] + constraint_weight * (predicted - dof_reference[frame])
            # Direct collocation-like projection: each knot-derived task target is
            # projected into the six leg joints, with joint limits and damping.
            for _ in range(6):
                phase2.set_reference_state(model, data, root[frame], quat[frame], dof[frame], qpos_addresses)
                for side in ("left", "right"):
                    body_id = body_ids[side]
                    point = phase2.point_world(data, body_id, phase2.SOLE_POINT_LOCAL[side])
                    error = target_foot[side][frame] - point
                    jac = phase2.point_jacobian(model, data, body_id, point)[:, side_dof_addresses[side]]
                    normal = jac @ jac.T + 5.0e-3 * np.eye(3)
                    delta = 0.70 * constraint_weight * jac.T @ np.linalg.solve(normal, error)
                    delta = np.clip(delta, -0.055, 0.055)
                    dof[frame, side_indices[side]] += delta
                    for joint_index in side_indices[side]:
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[joint_index])
                        if bool(model.jnt_limited[joint_id]):
                            low, high = model.jnt_range[joint_id]
                            dof[frame, joint_index] = np.clip(dof[frame, joint_index], low, high)
        if frame:
            # Explicit joint-velocity box in the collocation problem.  This is
            # applied during trajectory construction and is part of the hard
            # feasible set; it is not a post-hoc moving-average/smoothing pass.
            dof[frame] = np.clip(dof[frame], dof[frame - 1] - 0.18, dof[frame - 1] + 0.18)

    final_residuals = []
    for frame in range(active_end + 1):
        phase2.set_reference_state(model, data, root[frame], quat[frame], dof[frame], qpos_addresses)
        for side in ("left", "right"):
            point = phase2.point_world(data, body_ids[side], phase2.SOLE_POINT_LOCAL[side])
            final_residuals.append(float(np.linalg.norm(target_foot[side][frame] - point)))

    result = copy.deepcopy(entry)
    result["dof"] = dof.astype(np.float32)
    result["root_trans_offset"] = root.astype(np.float32)
    result["retarget_method"] = "phase3_contact_constrained_dynamic_teacher_oracle"
    result["phase3_teacher_vector"] = asdict(vector)
    result["phase3_expected_contact"] = {
        side: schedule[f"{side}_contact"].astype(np.uint8) for side in ("left", "right")
    }
    return result, {
        "event": event,
        "root_y_correction_max_m": float(np.max(np.abs(root[:, 1] - np.asarray(entry["root_trans_offset"])[:, 1]))),
        "root_z_correction_max_m": float(np.max(np.abs(root[:, 2] - np.asarray(entry["root_trans_offset"])[:, 2]))),
        "ik_terminal_residual_p95_m": float(np.percentile(final_residuals, 95)),
        "desired_com_lateral_p95_m_model_estimate": float(np.percentile(np.abs(lateral), 95)),
    }


def offline_gate(entry: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    dof = np.asarray(entry["dof"], dtype=np.float64)
    step = np.abs(np.diff(dof, axis=0))
    checks = {
        "finite": bool(np.all(np.isfinite(dof)) and np.all(np.isfinite(entry["root_trans_offset"]))),
        "joint_step_below_0p22_rad": bool(step.size == 0 or np.max(step) <= 0.22),
        "root_y_correction_below_0p101_m": diagnostics["root_y_correction_max_m"] <= 0.101,
        "root_z_correction_below_0p061_m": diagnostics["root_z_correction_max_m"] <= 0.061,
        "ik_terminal_residual_p95_below_0p06_m": diagnostics["ik_terminal_residual_p95_m"] <= 0.06,
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "joint_step_max_rad": float(np.max(step)) if step.size else 0.0,
        "joint_step_p95_rad": float(np.percentile(np.max(step, axis=1), 95)) if step.size else 0.0,
    }


def root_tilt(q_wxyz: np.ndarray) -> float:
    # z component of body-up is the third row/column equivalent from quaternion.
    w, x, y, z = q_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arccos(np.clip(up_z, -1.0, 1.0)))


def expected_contacts(entry: dict[str, Any], times: np.ndarray) -> dict[str, np.ndarray]:
    source_t = np.arange(len(entry["dof"])) / float(entry["fps"])
    result = {}
    contract = entry.get("phase3_expected_contact")
    if contract is None:
        raise ValueError("teacher entry has no explicit contact schedule")
    for side in ("left", "right"):
        values = np.asarray(contract[side], dtype=np.float64)
        result[side] = np.interp(times, source_t, values) >= 0.5
    return result


def simulate_dynamic_teacher(
    scene: Path,
    control_yaml: Path,
    entry: dict[str, Any],
    mode: str,
    physics,
    phase2,
) -> dict[str, Any]:
    if mode not in physics.MODES:
        raise ValueError(mode)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    contract = physics.build_control_contract(model, control_yaml)
    names = tuple(entry["joint_names_mujoco"])
    if names != contract.joint_names:
        raise ValueError("teacher joint order disagrees with official model")
    duration = (len(entry["dof"]) - 1) / float(entry["fps"])
    steps_per_control = int(round(contract.control_dt / model.opt.timestep))
    n_steps = int(round(duration / model.opt.timestep))
    sim_times = np.minimum(np.arange(n_steps + 1) * model.opt.timestep, duration)
    qpos_ref = physics.reference_qpos(entry, sim_times)
    control_times = np.minimum(
        np.floor(sim_times / contract.control_dt + 1.0e-10) * contract.control_dt, duration
    )
    _, _, dof_control = physics.interpolate_motion(entry, control_times)
    contacts_expected = expected_contacts(entry, sim_times)
    qvel_ref = None
    if mode == "prescribed_root_trackability":
        qvel_ref = np.zeros((n_steps + 1, model.nv))
        for index in range(n_steps):
            mujoco.mj_differentiatePos(model, qvel_ref[index], model.opt.timestep, qpos_ref[index], qpos_ref[index + 1])
        qvel_ref[-1] = qvel_ref[-2]

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, phase2.FOOT_BODY[side])
        for side in ("left", "right")
    }
    entry_index = {name: index for index, name in enumerate(names)}
    target_indices = np.asarray([entry_index[name] for name in contract.actuator_joint_names])
    previous_geom_pos = data.geom_xpos.copy()
    previous_com = data.subtree_com[0].copy()
    previous_target = dof_control[0, target_indices].copy()
    previous_torque = np.zeros(model.nu)
    slips = {"left": [], "right": []}
    contact_match = []
    dcm_support = []
    target_delta = []
    torque_delta = []
    tilts = []
    root_z = []
    saturation = 0
    effort = 0
    fall_time = None
    achieved = n_steps
    force_buffer = np.zeros(6)

    for step in range(n_steps):
        if mode == "prescribed_root_trackability":
            data.qpos[:7] = qpos_ref[step, :7]
            data.qvel[:6] = qvel_ref[step, :6]
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
            data.qvel[:6] = qvel_ref[step + 1, :6]
            mujoco.mj_forward(model, data)

        actual_contact = {"left": False, "right": False}
        contact_geoms = {"left": set(), "right": set()}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            other = geom2 if geom1 == floor else geom1 if geom2 == floor else -1
            side = geom_side.get(other)
            if side is None:
                continue
            mujoco.mj_contactForce(model, data, contact_id, force_buffer)
            if force_buffer[0] > 1.0:
                actual_contact[side] = True
                contact_geoms[side].add(other)
        for side in ("left", "right"):
            contact_match.append(float(actual_contact[side] == bool(contacts_expected[side][step + 1])))
            if actual_contact[side] and contacts_expected[side][step + 1]:
                for geom in contact_geoms[side]:
                    slips[side].append(float(np.linalg.norm(data.geom_xpos[geom, :2] - previous_geom_pos[geom, :2]) / model.opt.timestep))
        previous_geom_pos[:] = data.geom_xpos

        com = data.subtree_com[0].copy()
        com_velocity = (com - previous_com) / model.opt.timestep
        previous_com = com
        expected_support = [
            phase2.point_world(data, body_ids[side], phase2.SOLE_POINT_LOCAL[side])[:2]
            for side in ("left", "right") if contacts_expected[side][step + 1]
        ]
        if expected_support:
            dcm = com[:2] + math.sqrt(max(com[2], 0.20) / 9.81) * com_velocity[:2]
            support = np.asarray(expected_support)
            if len(support) == 1:
                distance = np.linalg.norm(dcm - support[0])
            else:
                vector = support[1] - support[0]
                alpha = np.clip(np.dot(dcm - support[0], vector) / (np.dot(vector, vector) + 1e-9), 0, 1)
                distance = np.linalg.norm(dcm - (support[0] + alpha * vector))
            dcm_support.append(float(distance))
        if step % steps_per_control == 0:
            target_delta.append(float(np.linalg.norm(target - previous_target)))
        torque_delta.append(float(np.linalg.norm(clipped - previous_torque)))
        previous_target = target.copy()
        previous_torque = clipped.copy()
        tilt = root_tilt(data.qpos[3:7])
        tilts.append(tilt)
        root_z.append(float(data.qpos[2]))
        if mode == "free_root_balance" and (data.qpos[2] < FALL_ROOT_Z_M or tilt > FALL_TILT_RAD):
            fall_time = (step + 1) * model.opt.timestep
            achieved = step + 1
            break

    simulated = min(achieved * model.opt.timestep, duration)
    slip_values = [value for values in slips.values() for value in values]
    return {
        "mode": mode,
        "reference_duration_s": duration,
        "simulated_duration_s": simulated,
        "duration_fraction": float(simulated / duration),
        "fell": fall_time is not None,
        "fall_time_s": fall_time,
        "root_tilt_max_rad": float(max(tilts, default=root_tilt(data.qpos[3:7]))),
        "root_z_min_m": float(min(root_z, default=data.qpos[2])),
        "dcm_to_expected_support_p95_m_model_estimate": float(np.percentile(dcm_support, 95)) if dcm_support else None,
        "expected_contact_match_fraction_model_estimate": float(np.mean(contact_match)) if contact_match else None,
        "stance_slip_p95_mps": float(np.percentile(slip_values, 95)) if slip_values else 0.0,
        "stance_slip_p95_mps_by_side": {
            side: float(np.percentile(values, 95)) if values else 0.0 for side, values in slips.items()
        },
        "action_target_delta_p95_rad": float(np.percentile(target_delta, 95)) if target_delta else 0.0,
        "torque_delta_p95_nm": float(np.percentile(torque_delta, 95)) if torque_delta else 0.0,
        "torque_saturation_fraction": float(saturation / effort) if effort else 0.0,
        "interpretation": (
            "root externally prescribed; trackability diagnostic only"
            if mode == "prescribed_root_trackability"
            else "official free-root dynamic-teacher existence rollout; not a policy evaluation"
        ),
    }


def dynamic_cost(result: dict[str, Any]) -> float:
    dcm = result["dcm_to_expected_support_p95_m_model_estimate"]
    contact = result["expected_contact_match_fraction_model_estimate"]
    return float(
        -12.0 * result["duration_fraction"]
        + 1.5 * result["root_tilt_max_rad"] / FALL_TILT_RAD
        + 1.5 * (dcm if dcm is not None else 1.0) / 0.20
        + 1.2 * result["stance_slip_p95_mps"] / 0.10
        + 1.0 * (1.0 - (contact if contact is not None else 0.0))
        + 0.4 * result["action_target_delta_p95_rad"] / 0.10
        + 0.2 * result["torque_delta_p95_nm"] / 50.0
        + 0.5 * result["torque_saturation_fraction"]
    )


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline_free_root"]
    best = report["best_teacher_free_root"]
    lines = [
        "# X2 Contact-Constrained Dynamic Teacher Phase3",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 本轮不训练、不加载 checkpoint、不接真机；只做 `walk_straight` dynamic-teacher existence smoke。",
        "- COM、DCM、contact 和支撑状态全部是官方 AimDK v1.0 MuJoCo 模型估计，不是真实 GRF/COP/COM/contact 真值。",
        "- teacher 使用整段未来 reference、显式接触约束和官方 free-root 物理选参，因此是不可部署 oracle。",
        "",
        "## 可证伪契约",
        "",
        "- 优化变量：4-knot COM 横向样条、3-knot root 高度样条、liftoff/touchdown 时刻、支撑脚固定 terminal、摆脚 clearance、落脚 XY terminal。",
        "- 离线硬门：有限值、joint step ≤0.22 rad/frame、root Y/Z 修正≤0.10/0.06 m、足端 IK p95≤0.06 m。",
        "- free-root 代价：生存、root tilt、模型估计 DCM-support、stance slip、接触一致、action delta、torque delta/饱和。",
        "- 晋升门：相对 current-v4 至少多生存 0.5 s，stance slip 不超过 `max(1.1×baseline, baseline+0.01 m/s)`，且 prescribed-root 平滑/饱和不劣化。",
        "",
        "## 搜索覆盖",
        "",
        f"- CEM seed/population/iterations/elites：`{report['search']['seed']}/{report['search']['population']}/{report['search']['iterations']}/{report['search']['elite_count']}`。",
        f"- 生成/离线通过/free-root rollout：`{report['search']['generated_count']}/{report['search']['offline_pass_count']}/{report['search']['free_root_rollout_count']}`。",
        f"- 首步原始→优化 liftoff/touchdown：`{report['best_teacher_diagnostics']['event']['original_liftoff_frame']}/{report['best_teacher_diagnostics']['event']['original_touchdown_frame']} → {report['best_teacher_diagnostics']['event']['optimized_liftoff_frame']}/{report['best_teacher_diagnostics']['event']['optimized_touchdown_frame']}`。",
        "",
        "## 物理对照",
        "",
        "| variant | mode | survival(s) | duration | tilt max | DCM p95(m) | slip p95(m/s) | contact match | action Δp95 | torque Δp95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (
        ("current-v4", baseline),
        ("teacher", best),
        ("teacher", report["best_teacher_prescribed_root"]),
    ):
        lines.append(
            f"| {name} | {value['mode']} | {value['simulated_duration_s']:.3f} | {value['duration_fraction']:.3f} | "
            f"{value['root_tilt_max_rad']:.3f} | {value['dcm_to_expected_support_p95_m_model_estimate']:.3f} | "
            f"{value['stance_slip_p95_mps']:.4f} | {value['expected_contact_match_fraction_model_estimate']:.3f} | "
            f"{value['action_target_delta_p95_rad']:.4f} | {value['torque_delta_p95_nm']:.2f} |"
        )
    lines += [
        "",
        "## 结果 / 结论 / 下一步",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=5)
    args = parser.parse_args()

    phase2 = load_module(PHASE2_SCRIPT, "x2_phase2_helpers_for_phase3")
    physics = load_module(PHYSICS_SCRIPT, "x2_official_physics_helpers_for_phase3")
    motions = joblib.load(args.current)
    key = next(key for key, entry in motions.items() if entry.get("panel_role") == SMOKE_ROLE)
    entry = motions[key]
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))

    # Baseline gets the unmodified reference but the same explicit model-
    # estimated contact schedule for a fair dynamic metric computation.
    base_kin = phase2.reference_kinematics(model, entry)
    base_phase = phase2.phase_contract(base_kin)
    baseline = copy.deepcopy(entry)
    baseline["phase3_expected_contact"] = {
        side: (~base_phase[f"{side}_swing"]).astype(np.uint8) for side in ("left", "right")
    }
    baseline_free = simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, baseline, "free_root_balance", physics, phase2
    )

    rng = np.random.default_rng(args.seed)
    mean = (LOW + HIGH) / 2.0
    mean[9] = 0.065
    std = (HIGH - LOW) / 3.0
    evaluations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    generated = offline_pass = rollouts = 0
    iteration_summaries = []
    for iteration in range(args.iterations):
        population = rng.normal(mean, std, size=(args.population, len(mean)))
        population[0] = mean
        scored = []
        for raw in population:
            generated += 1
            vector = decode_vector(raw)
            teacher, diagnostics = build_teacher_entry(model, entry, vector, phase2)
            gate = offline_gate(teacher, diagnostics)
            record = {
                "iteration": iteration,
                "vector": asdict(vector),
                "diagnostics": diagnostics,
                "offline_gate": gate,
            }
            if not gate["pass"]:
                record["cost"] = 1.0e6 + gate["joint_step_max_rad"]
                evaluations.append(record)
                scored.append((record["cost"], raw, record, teacher))
                continue
            offline_pass += 1
            result = simulate_dynamic_teacher(
                physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, teacher, "free_root_balance", physics, phase2
            )
            rollouts += 1
            cost = dynamic_cost(result)
            record.update({"free_root": result, "cost": cost})
            evaluations.append(record)
            scored.append((cost, raw, record, teacher))
            if best is None or cost < best["cost"]:
                best = {"cost": cost, "record": record, "teacher": teacher, "raw": raw.copy()}
        scored.sort(key=lambda item: item[0])
        finite = [item for item in scored if item[0] < 1.0e6]
        if not finite:
            break
        elites = finite[: min(args.elite_count, len(finite))]
        elite_values = np.asarray([item[1] for item in elites])
        mean = 0.25 * mean + 0.75 * np.mean(elite_values, axis=0)
        std = np.maximum(0.15 * (HIGH - LOW), 0.25 * std + 0.75 * np.std(elite_values, axis=0))
        iteration_summaries.append({
            "iteration": iteration,
            "best_cost": float(elites[0][0]),
            "best_survival_s": elites[0][2]["free_root"]["simulated_duration_s"],
            "best_slip_p95_mps": elites[0][2]["free_root"]["stance_slip_p95_mps"],
            "offline_pass_count": int(sum(item[0] < 1.0e6 for item in scored)),
        })

    if best is None:
        raise RuntimeError("no candidate passed the offline gate; official teacher does not exist in searched space")
    best_teacher = best["teacher"]
    best_free = best["record"]["free_root"]
    best_prescribed = simulate_dynamic_teacher(
        physics.DEFAULT_SCENE,
        physics.DEFAULT_CONTROL,
        best_teacher,
        "prescribed_root_trackability",
        physics,
        phase2,
    )
    survival_gain = best_free["simulated_duration_s"] - baseline_free["simulated_duration_s"]
    slip_limit = max(1.10 * baseline_free["stance_slip_p95_mps"], baseline_free["stance_slip_p95_mps"] + 0.01)
    survival_pass = survival_gain >= 0.50
    slip_pass = best_free["stance_slip_p95_mps"] <= slip_limit
    prescribed_pass = (
        best_prescribed["action_target_delta_p95_rad"] <= 0.22
        and best_prescribed["torque_saturation_fraction"] <= 0.03
    )
    existence = bool(survival_pass and slip_pass and prescribed_pass)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({key: best_teacher}, args.cache)
    report = {
        "schema_version": "x2_contact_constrained_dynamic_teacher_phase3_v1",
        "provenance": {
            "current": {"path": str(args.current), "sha256": sha256(args.current)},
            "scene": {"path": str(physics.DEFAULT_SCENE), "sha256": sha256(physics.DEFAULT_SCENE)},
            "control": {"path": str(physics.DEFAULT_CONTROL), "sha256": sha256(physics.DEFAULT_CONTROL)},
            "output_cache": {"path": str(args.cache), "sha256": sha256(args.cache)},
        },
        "truth_boundary": {
            "com_dcm_contact_source": "official AimDK v1.0 MuJoCo model estimate",
            "not_measured": ["hardware GRF", "hardware COP", "hardware COM", "real foot contact"],
            "oracle_deployable": False,
            "uses_full_future_reference": True,
            "training_or_checkpoint_load": False,
        },
        "smoke_motion": {"role": SMOKE_ROLE, "motion_key": key},
        "search": {
            "method": "bounded low-dimensional cubic splines + contact-event variables + CEM selected by official free-root rollout",
            "seed": args.seed,
            "population": args.population,
            "iterations": args.iterations,
            "elite_count": args.elite_count,
            "generated_count": generated,
            "offline_pass_count": offline_pass,
            "free_root_rollout_count": rollouts,
            "parameter_low": LOW.tolist(),
            "parameter_high": HIGH.tolist(),
            "iteration_summaries": iteration_summaries,
        },
        "baseline_free_root": baseline_free,
        "best_teacher_vector": best["record"]["vector"],
        "best_teacher_diagnostics": best["record"]["diagnostics"],
        "best_teacher_offline_gate": best["record"]["offline_gate"],
        "best_teacher_cost": best["cost"],
        "best_teacher_free_root": best_free,
        "best_teacher_prescribed_root": best_prescribed,
        "evaluations": evaluations,
        "decision": {
            "checks": {
                "survival_gain_at_least_0p5s": survival_pass,
                "stance_slip_not_worse": slip_pass,
                "prescribed_root_smooth_and_unsaturated": prescribed_pass,
            },
            "baseline_survival_s": baseline_free["simulated_duration_s"],
            "teacher_survival_s": best_free["simulated_duration_s"],
            "survival_gain_s": survival_gain,
            "baseline_slip_p95_mps": baseline_free["stance_slip_p95_mps"],
            "teacher_slip_p95_mps": best_free["stance_slip_p95_mps"],
            "slip_limit_mps": slip_limit,
            "dynamic_teacher_exists_in_searched_space": existence,
            "status": "DYNAMIC_TEACHER_EXISTENCE_SUPPORTED" if existence else "DYNAMIC_TEACHER_EXISTENCE_NOT_SUPPORTED",
            "result": (
                "walk smoke 同时获得显著生存增益、低滑移和 prescribed-root 平滑性。"
                if existence
                else "walk smoke 未同时达到显著生存增益、低滑移和 prescribed-root 平滑性。"
            ),
            "conclusion": (
                "搜索空间内存在可作为下一步 Silver teacher 的动力学轨迹，但仍需因果化和 held-out 复核。"
                if existence
                else "当前显式接触约束/CEM 搜索覆盖未证明存在可用 dynamic teacher；按预注册契约不扩大到五动作。"
            ),
            "next_step": (
                "扩展到其余四动作并复核，不训练。"
                if existence
                else "停止扩大本参数化；核查官方初始状态/控制语义，或采用带动力学约束的 direct collocation/MPC teacher。"
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
