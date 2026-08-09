#!/usr/bin/env python3
"""Phase12 frozen official-AimDK replay of Phase10 native Gold held-out clip0.

This is a single, pre-registered reference-trackability smoke, not training.
The reference is the recorded *actual q* loaded through Phase11 MotionLib plus
its recorded-state adapter.  Recorded commands are never used as targets.
Prescribed-root is run first; free-root is run once only if it passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO = Path(__file__).resolve().parents[2]
SONIC_ROOT = Path("/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox")
SRC_ROOT = REPO / "src"
for value in (REPO / "tools", SONIC_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from easydict import EasyDict
from gear_sonic.utils.motion_lib.motion_lib_base import FixHeightMode
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

from x2_native_gold_motionlib_adapter import apply_recorded_state_adapter
from retarget.run_x2_native_gold_motionlib_phase11 import native_config
import retarget.run_x2_forefoot_official_physics_screen as physics


DEFAULT_GOLD = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "gold_dynamic_native_seed_v1/held_out/official_native_dance_held_out.pkl"
)
DEFAULT_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_gold_trackability_phase12.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_gold_trackability_phase12.md"
PHASE11_REPORT = REPO / "reports/retarget/x2_native_gold_motionlib_phase11.json"

MODE_PRESCRIBED = "prescribed_root_trackability"
MODE_FREE = "free_root_balance"
FALL_ROOT_Z_M = 0.42
FALL_TILT_RAD = 0.90

# Frozen before physics.  They are deliberately broad sanity thresholds, not
# a claim of deployable imitation quality.
GATES = {
    "source_joint_step_p95_max_rad": [0.12, 0.45],
    "source_joint_velocity_p95_max_radps": [6.0, 20.0],
    "prescribed_q_rmse_max_rad": 0.20,
    "prescribed_body_relative_position_p95_m": 0.12,
    "prescribed_body_relative_orientation_p95_rad": 0.35,
    "prescribed_root_position_rmse_m": 0.005,
    "prescribed_root_orientation_p95_rad": 0.01,
    "prescribed_contact_agreement_min": 0.60,
    "prescribed_realized_single_support_min": 0.02,
    "prescribed_realized_single_support_fraction_of_source_min": 0.25,
    "prescribed_slip_p95_max_mps": 0.50,
    "prescribed_torque_saturation_max": 0.10,
    "free_full_duration_fraction_min": 0.999,
    "free_q_rmse_max_rad": 0.25,
    "free_root_xy_rmse_max_m": 0.25,
    "free_contact_agreement_min": 0.55,
    "free_slip_p95_max_mps": 0.50,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float] | np.ndarray, q: float) -> float | None:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    return None if array.size == 0 else float(np.percentile(array, q))


def quat_distance_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.0e-12)
    b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1.0e-12)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), -1.0, 1.0))


def f1_binary(reference: np.ndarray, realized: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    realized = np.asarray(realized, dtype=bool)
    tp = int(np.count_nonzero(reference & realized))
    fp = int(np.count_nonzero(~reference & realized))
    fn = int(np.count_nonzero(reference & ~realized))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def phase_stats(contact: dict[str, np.ndarray]) -> dict[str, float]:
    left = np.asarray(contact["left"], dtype=bool)
    right = np.asarray(contact["right"], dtype=bool)
    return {
        "left_ratio": float(np.mean(left)),
        "right_ratio": float(np.mean(right)),
        "double_support_ratio": float(np.mean(left & right)),
        "single_support_ratio": float(np.mean(left ^ right)),
        "flight_ratio": float(np.mean(~left & ~right)),
    }


def build_reference(gold: Path, contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = joblib.load(gold)
    first_key = list(entries)[0]
    source = entries[first_key]
    model_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    official_names = model_contract["control_boundaries"]["official_mjcf_actuated_31"]

    motion = MotionLibRobot(native_config(gold), len(entries), "cpu")
    motion.load_motions_for_evaluation()
    adapter_meta = apply_recorded_state_adapter(motion, gold)
    frame_count = int(motion._motion_num_frames[0].item())
    if frame_count != len(source["dof"]):
        raise ValueError("Phase11 MotionLib/source clip0 frame mismatch")
    if list(motion.curr_motion_keys)[0] != first_key:
        raise ValueError("Phase11 MotionLib/source clip0 key mismatch")

    body_names = list(motion.mesh_parsers.body_names)
    reference = {
        "key": first_key,
        "fps": float(motion._motion_fps[0].item()),
        "joint_names": list(official_names),
        "body_names": body_names,
        "q": motion.dof_pos[:frame_count].cpu().numpy().astype(np.float64),
        "dq": motion.dof_vel[:frame_count].cpu().numpy().astype(np.float64),
        "root_pos": motion.body_pos_w[:frame_count, 0].cpu().numpy().astype(np.float64),
        "root_quat_xyzw": motion.body_quat_w[:frame_count, 0].cpu().numpy().astype(np.float64),
        "root_lin_vel": motion.body_lin_vel_w[:frame_count, 0].cpu().numpy().astype(np.float64),
        "root_ang_vel": motion.body_ang_vel_w[:frame_count, 0].cpu().numpy().astype(np.float64),
        "body_pos": motion.body_pos_w[:frame_count].cpu().numpy().astype(np.float64),
        "body_quat_xyzw": motion.body_quat_w[:frame_count].cpu().numpy().astype(np.float64),
        "contact": {
            "left": motion.feet_l[:frame_count, 0].cpu().numpy().astype(bool),
            "right": motion.feet_r[:frame_count, 0].cpu().numpy().astype(bool),
        },
    }
    # Hard truth boundary: only recorded actual q is accepted as target.
    reference["target_source"] = "Phase11 MotionLib dof_pos after state adapter; recorded actual q"
    reference["recorded_command_used"] = False
    return reference, adapter_meta


def interpolation(reference: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fps = float(reference["fps"])
    source_t = np.arange(len(reference["q"]), dtype=np.float64) / fps
    times = np.minimum(np.asarray(times, dtype=np.float64), source_t[-1])
    root_pos = np.column_stack([
        np.interp(times, source_t, reference["root_pos"][:, axis]) for axis in range(3)
    ])
    root_quat = Slerp(source_t, Rotation.from_quat(reference["root_quat_xyzw"]))(times).as_quat()
    root_lin = np.column_stack([
        np.interp(times, source_t, reference["root_lin_vel"][:, axis]) for axis in range(3)
    ])
    root_ang = np.column_stack([
        np.interp(times, source_t, reference["root_ang_vel"][:, axis]) for axis in range(3)
    ])
    q = np.column_stack([
        np.interp(times, source_t, reference["q"][:, axis]) for axis in range(31)
    ])
    return root_pos, root_quat, root_lin, root_ang, q


def source_kinematics(reference: dict[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
    q = reference["q"]
    dq = reference["dq"]
    step = np.abs(np.diff(q, axis=0))
    model_joint_limits = physics.target_limit_report(model, {
        "dof": q, "joint_names_mujoco": reference["joint_names"]
    })
    return {
        "joint_step_abs_p95_max_rad": [float(np.percentile(step, 95)), float(np.max(step))],
        "recorded_joint_velocity_abs_p95_max_radps": [float(np.percentile(np.abs(dq), 95)), float(np.max(np.abs(dq)))],
        "target_joint_limit_saturation": model_joint_limits,
        "contact_phase": phase_stats(reference["contact"]),
    }


def contact_now(model: mujoco.MjModel, data: mujoco.MjData, floor: int, geom_side: dict[int, str]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {"left": set(), "right": set()}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        other = geom2 if geom1 == floor else geom1 if geom2 == floor else -1
        side = geom_side.get(other)
        if side is not None:
            result[side].add(other)
    return result


def simulate(reference: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in (MODE_PRESCRIBED, MODE_FREE):
        raise ValueError(mode)
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    data = mujoco.MjData(model)
    control = physics.build_control_contract(model, physics.DEFAULT_CONTROL)
    if tuple(reference["joint_names"]) != control.joint_names:
        raise ValueError("Phase11 joint order differs from official control contract")
    if not np.isclose(model.opt.timestep, 0.001) or not np.isclose(control.control_dt, 0.02):
        raise ValueError("frozen official 1kHz physics / 50Hz control contract changed")
    steps_per_control = int(round(control.control_dt / model.opt.timestep))
    frames = len(reference["q"])
    n_steps = (frames - 1) * steps_per_control
    times = np.arange(n_steps + 1, dtype=np.float64) * model.opt.timestep
    root_pos, root_quat, root_lin, root_ang, q_interp = interpolation(reference, times)
    root_wxyz = root_quat[:, [3, 0, 1, 2]]

    entry_index = {name: index for index, name in enumerate(reference["joint_names"])}
    actuator_target_indices = np.asarray([entry_index[name] for name in control.actuator_joint_names])
    body_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in reference["body_names"]
    ], dtype=np.int64)
    if np.any(body_ids < 0):
        raise ValueError("Phase11 body order is not available in official scene")

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = root_pos[0]
    data.qpos[3:7] = root_wxyz[0]
    data.qpos[control.qpos_addresses] = reference["q"][0, actuator_target_indices]
    data.qvel[:3] = reference["root_lin_vel"][0]
    data.qvel[3:6] = reference["root_ang_vel"][0]
    data.qvel[control.qvel_addresses] = reference["dq"][0, actuator_target_indices]
    mujoco.mj_forward(model, data)

    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    previous_geom_pos = data.geom_xpos.copy()
    saturation = np.zeros(model.nu, dtype=np.int64)
    effort_count = 0
    torque_fraction: list[float] = []
    slip: list[float] = []
    q_samples: list[np.ndarray] = [reference["q"][0].copy()]
    body_pos_samples: list[np.ndarray] = [data.xpos[body_ids].copy()]
    body_quat_samples: list[np.ndarray] = [data.xquat[body_ids][:, [1, 2, 3, 0]].copy()]
    root_pos_samples: list[np.ndarray] = [data.qpos[:3].copy()]
    root_quat_samples: list[np.ndarray] = [data.qpos[3:7][[1, 2, 3, 0]].copy()]
    realized = {"left": [], "right": []}
    interval_contact = {"left": False, "right": False}
    fell_at: float | None = None
    nonfinite_at: float | None = None
    achieved_steps = n_steps

    for step in range(n_steps):
        if mode == MODE_PRESCRIBED:
            data.qpos[:3] = root_pos[step]
            data.qpos[3:7] = root_wxyz[step]
            data.qvel[:3] = root_lin[step]
            data.qvel[3:6] = root_ang[step]

        control_frame = step // steps_per_control
        target = reference["q"][control_frame, actuator_target_indices]
        actual_q = data.qpos[control.qpos_addresses]
        actual_dq = data.qvel[control.qvel_addresses]
        raw_torque = control.kp * (target - actual_q) - control.kd * actual_dq
        saturated = (raw_torque < control.torque_low) | (raw_torque > control.torque_high)
        saturation += saturated
        effort_count += model.nu
        denom = np.maximum(np.maximum(np.abs(control.torque_low), np.abs(control.torque_high)), 1.0e-9)
        torque_fraction.extend((np.abs(raw_torque) / denom).tolist())
        data.ctrl[:] = np.clip(raw_torque, control.torque_low, control.torque_high)
        mujoco.mj_step(model, data)

        if mode == MODE_PRESCRIBED:
            data.qpos[:3] = root_pos[step + 1]
            data.qpos[3:7] = root_wxyz[step + 1]
            data.qvel[:3] = root_lin[step + 1]
            data.qvel[3:6] = root_ang[step + 1]
            mujoco.mj_forward(model, data)

        contacts = contact_now(model, data, floor, geom_side)
        for side in ("left", "right"):
            interval_contact[side] = interval_contact[side] or bool(contacts[side])
            for geom_id in contacts[side]:
                delta = data.geom_xpos[geom_id, :2] - previous_geom_pos[geom_id, :2]
                slip.append(float(np.linalg.norm(delta) / model.opt.timestep))
        previous_geom_pos[:] = data.geom_xpos

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite_at = (step + 1) * model.opt.timestep
            achieved_steps = step + 1
            break

        if mode == MODE_FREE:
            tilt = physics.root_tilt(data.qpos[3:7])
            if float(data.qpos[2]) < FALL_ROOT_Z_M or tilt > FALL_TILT_RAD:
                fell_at = (step + 1) * model.opt.timestep
                achieved_steps = step + 1
                break

        if (step + 1) % steps_per_control == 0:
            by_name = {
                name: float(data.qpos[address])
                for name, address in zip(control.actuator_joint_names, control.qpos_addresses)
            }
            q_samples.append(np.asarray([by_name[name] for name in control.joint_names]))
            body_pos_samples.append(data.xpos[body_ids].copy())
            body_quat_samples.append(data.xquat[body_ids][:, [1, 2, 3, 0]].copy())
            root_pos_samples.append(data.qpos[:3].copy())
            root_quat_samples.append(data.qpos[3:7][[1, 2, 3, 0]].copy())
            for side in ("left", "right"):
                realized[side].append(interval_contact[side])
                interval_contact[side] = False

    sample_count = len(q_samples)
    q_actual = np.asarray(q_samples)
    body_pos = np.asarray(body_pos_samples)
    body_quat = np.asarray(body_quat_samples)
    replay_root = np.asarray(root_pos_samples)
    replay_root_quat = np.asarray(root_quat_samples)
    ref_q = reference["q"][:sample_count]
    ref_body_pos = reference["body_pos"][:sample_count]
    ref_body_quat = reference["body_quat_xyzw"][:sample_count]
    ref_root = reference["root_pos"][:sample_count]
    ref_root_quat = reference["root_quat_xyzw"][:sample_count]
    q_error = q_actual - ref_q
    body_world_error = np.linalg.norm(body_pos - ref_body_pos, axis=-1)
    body_relative_error = np.linalg.norm(
        (body_pos - replay_root[:, None]) - (ref_body_pos - ref_root[:, None]), axis=-1
    )
    body_ori_error = quat_distance_xyzw(body_quat, ref_body_quat)
    root_pos_error = np.linalg.norm(replay_root - ref_root, axis=-1)
    root_ori_error = quat_distance_xyzw(replay_root_quat, ref_root_quat)
    actual_step = np.abs(np.diff(q_actual, axis=0))
    actual_velocity = actual_step * reference["fps"]

    # Interval i ends at source frame i+1, hence compare realized intervals to [1:N].
    realized_contact = {side: np.asarray(values, dtype=bool) for side, values in realized.items()}
    ref_contact = {
        side: reference["contact"][side][1:1 + len(realized_contact[side])]
        for side in ("left", "right")
    }
    agreement = {
        side: float(np.mean(realized_contact[side] == ref_contact[side])) if len(realized_contact[side]) else 0.0
        for side in ("left", "right")
    }
    agreement["mean"] = float(np.mean([agreement["left"], agreement["right"]]))
    contact_f1 = {
        side: f1_binary(ref_contact[side], realized_contact[side]) for side in ("left", "right")
    }
    duration = (frames - 1) / reference["fps"]
    achieved = min(achieved_steps * model.opt.timestep, duration)
    return {
        "mode": mode,
        "interpretation": (
            "root pose/velocity externally prescribed every physics step; trackability only"
            if mode == MODE_PRESCRIBED else
            "free-root bare position-PD actual-q replay; not a learned-policy evaluation"
        ),
        "reference_duration_s": duration,
        "simulated_duration_s": achieved,
        "duration_fraction": float(achieved / duration),
        "fell": fell_at is not None,
        "fall_time_s": fell_at,
        "nonfinite_time_s": nonfinite_at,
        "q_tracking": {
            "rmse_rad": float(np.sqrt(np.mean(q_error ** 2))),
            "abs_p95_rad": float(np.percentile(np.abs(q_error), 95)),
            "abs_max_rad": float(np.max(np.abs(q_error))),
        },
        "body_tracking": {
            "world_position_p95_max_m": [float(np.percentile(body_world_error, 95)), float(np.max(body_world_error))],
            "root_relative_position_p95_max_m": [float(np.percentile(body_relative_error, 95)), float(np.max(body_relative_error))],
            "orientation_p95_max_rad": [float(np.percentile(body_ori_error, 95)), float(np.max(body_ori_error))],
        },
        "root_tracking": {
            "position_rmse_final_m": [float(np.sqrt(np.mean(root_pos_error ** 2))), float(root_pos_error[-1])],
            "xy_rmse_final_m": [float(np.sqrt(np.mean(np.sum((replay_root[:, :2] - ref_root[:, :2]) ** 2, axis=-1)))), float(np.linalg.norm(replay_root[-1, :2] - ref_root[-1, :2]))],
            "orientation_p95_max_rad": [float(np.percentile(root_ori_error, 95)), float(np.max(root_ori_error))],
            "z_min_m": float(np.min(replay_root[:, 2])),
            "tilt_max_rad": float(max(physics.root_tilt(quat[[3, 0, 1, 2]]) for quat in replay_root_quat)),
        },
        "contact": {
            "reference_phase": phase_stats(ref_contact),
            "realized_phase": phase_stats(realized_contact),
            "agreement": agreement,
            "f1": contact_f1,
            "truth_boundary": "reference=model active-sole geometry; realized=official MuJoCo collision; neither is hardware GRF/COP/wrench",
        },
        "slip_p95_max_mps": [percentile(np.asarray(slip), 95), float(max(slip)) if slip else None],
        "action_target": source_kinematics(reference, model),
        "torque": {
            "saturation_fraction": float(np.sum(saturation) / max(1, effort_count)),
            "raw_abs_over_limit_p95_max": [float(np.percentile(torque_fraction, 95)), float(np.max(torque_fraction))],
            "saturation_by_joint": {
                name: float(count / max(1, achieved_steps))
                for name, count in zip(control.actuator_joint_names, saturation)
            },
        },
        "replay_joint_motion": {
            "step_abs_p95_max_rad": [float(np.percentile(actual_step, 95)), float(np.max(actual_step))],
            "velocity_abs_p95_max_radps": [float(np.percentile(actual_velocity, 95)), float(np.max(actual_velocity))],
        },
        "initialization": {
            "q": "recorded actual q frame0",
            "dq": "recorded actual dq frame0 from Phase11 state adapter",
            "root_pose": "recorded actual root position/quaternion frame0",
            "root_velocity": "recorded actual root lin/ang velocity frame0 from Phase11 state adapter",
            "solver_warmstart": "unavailable; reset default/zero",
            "contact_constraint_warmstart": "unavailable; reconstructed by mj_forward",
            "controller_hidden_state": "unavailable; replay uses stateless official position-PD, not original ONNX controller",
        },
    }


def prescribed_gate(result: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "full_duration": result["duration_fraction"] >= 0.999,
        "source_joint_step": source["joint_step_abs_p95_max_rad"][0] <= GATES["source_joint_step_p95_max_rad"][0] and source["joint_step_abs_p95_max_rad"][1] <= GATES["source_joint_step_p95_max_rad"][1],
        "source_joint_velocity": source["recorded_joint_velocity_abs_p95_max_radps"][0] <= GATES["source_joint_velocity_p95_max_radps"][0] and source["recorded_joint_velocity_abs_p95_max_radps"][1] <= GATES["source_joint_velocity_p95_max_radps"][1],
        "q_trackable": result["q_tracking"]["rmse_rad"] <= GATES["prescribed_q_rmse_max_rad"],
        "body_position_trackable": result["body_tracking"]["root_relative_position_p95_max_m"][0] <= GATES["prescribed_body_relative_position_p95_m"],
        "body_orientation_trackable": result["body_tracking"]["orientation_p95_max_rad"][0] <= GATES["prescribed_body_relative_orientation_p95_rad"],
        "root_externally_prescribed_exact": result["root_tracking"]["position_rmse_final_m"][0] <= GATES["prescribed_root_position_rmse_m"] and result["root_tracking"]["orientation_p95_max_rad"][0] <= GATES["prescribed_root_orientation_p95_rad"],
        "contact_agreement": result["contact"]["agreement"]["mean"] >= GATES["prescribed_contact_agreement_min"],
        "single_support_not_deleted": result["contact"]["realized_phase"]["single_support_ratio"] >= max(GATES["prescribed_realized_single_support_min"], GATES["prescribed_realized_single_support_fraction_of_source_min"] * result["contact"]["reference_phase"]["single_support_ratio"]),
        "slip_bounded": result["slip_p95_max_mps"][0] is not None and result["slip_p95_max_mps"][0] <= GATES["prescribed_slip_p95_max_mps"],
        "torque_saturation_bounded": result["torque"]["saturation_fraction"] <= GATES["prescribed_torque_saturation_max"],
        "target_limits_bounded": source["target_joint_limit_saturation"]["fraction"] <= 0.002 and source["target_joint_limit_saturation"]["max_excess_rad"] <= 0.05,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"checks": checks, "pass": all(checks.values()), "failed": [key for key, value in checks.items() if not value]}


def free_gate(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "full_duration": result["duration_fraction"] >= GATES["free_full_duration_fraction_min"],
        "q_trackable": result["q_tracking"]["rmse_rad"] <= GATES["free_q_rmse_max_rad"],
        "root_xy_trackable": result["root_tracking"]["xy_rmse_final_m"][0] <= GATES["free_root_xy_rmse_max_m"],
        "contact_agreement": result["contact"]["agreement"]["mean"] >= GATES["free_contact_agreement_min"],
        "slip_bounded": result["slip_p95_max_mps"][0] is not None and result["slip_p95_max_mps"][0] <= GATES["free_slip_p95_max_mps"],
        "torque_saturation_bounded": result["torque"]["saturation_fraction"] <= GATES["prescribed_torque_saturation_max"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"checks": checks, "pass": all(checks.values()), "failed": [key for key, value in checks.items() if not value]}


def render(report: dict[str, Any]) -> str:
    prescribed = report["replay"][MODE_PRESCRIBED]
    free = report["replay"].get(MODE_FREE)
    lines = [
        "# X2 Native Gold Official Trackability Phase12",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 唯一 reference 是 Phase11 MotionLib+state adapter 输出的 recorded actual q；recorded command 未作 reference。",
        "- source trace 自身稳定不等于本次 replay 稳定；prescribed-root 也不证明平衡。",
        "- contact 是模型几何/官方仿真碰撞，不是实机 GRF、COP、wrench 或足底力真值。",
        "",
        "## 假设",
        "",
        "目标本体原生 actual-q Gold 若能通过同一官方模型的固定-root trackability，至少可作为训练管线 sanity reference；free-root 裸PD只额外检验无闭环策略时的开环平衡。",
        "",
        "## 干预 / 对照",
        "",
        "- 对照：Phase10 recorded source trace（仅作 reference/provenance，不冒充 replay）。",
        "- 干预A：官方 AimDK v1.0 scene、1kHz physics/50Hz official PD，root pose/velocity逐物理步外部 prescribed。",
        "- 干预B：A过门后，同一 reference、同一初始化只跑一次 free-root；无参数扫描。",
        "- 初始化：q、dq、root pose、root lin/ang velocity均取 frame0 actual；solver/contact warmstart和原ONNX hidden state不可得。",
        "",
        "## 结果",
        "",
        "| mode | survival | q RMSE | body rel-pos p95 | root XY RMSE | contact agreement | SS(ref/real) | slip p95 | torque sat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in [prescribed] + ([free] if free is not None else []):
        lines.append(
            f"| {result['mode']} | {result['simulated_duration_s']:.3f}/{result['reference_duration_s']:.3f}s | "
            f"{result['q_tracking']['rmse_rad']:.4f} | {result['body_tracking']['root_relative_position_p95_max_m'][0]:.4f}m | "
            f"{result['root_tracking']['xy_rmse_final_m'][0]:.4f}m | {result['contact']['agreement']['mean']:.3f} | "
            f"{result['contact']['reference_phase']['single_support_ratio']:.3f}/{result['contact']['realized_phase']['single_support_ratio']:.3f} | "
            f"{result['slip_p95_max_mps'][0]:.3f}m/s | {result['torque']['saturation_fraction']:.4f} |"
        )
    lines += [
        "",
        f"- prescribed gate：`{report['gate']['prescribed']['pass']}`，失败项 `{report['gate']['prescribed']['failed']}`。",
        f"- free-root executed：`{free is not None}`" + (f"；gate `{report['gate']['free']['pass']}`，失败项 `{report['gate']['free']['failed']}`。" if free is not None else "；因 prescribed 未过门而停止。"),
        "",
        "## 结论",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    reference, adapter_meta = build_reference(args.gold, args.model_contract)
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    source = source_kinematics(reference, model)
    prescribed = simulate(reference, MODE_PRESCRIBED)
    prescribed_decision = prescribed_gate(prescribed, source)
    replays: dict[str, Any] = {MODE_PRESCRIBED: prescribed}
    free_decision = None
    if prescribed_decision["pass"]:
        free = simulate(reference, MODE_FREE)
        replays[MODE_FREE] = free
        free_decision = free_gate(free)

    if not prescribed_decision["pass"]:
        status = "PHASE12_PRESCRIBED_TRACKABILITY_REJECTED"
        result = "prescribed-root 未通过预注册 trackability 门；按规则停止，未运行 free-root。"
        conclusion = "Gold/PD replay contract 仍有不兼容；该结果只否定此裸PD sanity replay，不评价训练策略。"
        next_step = "停止本阶段；先核查失败指标，不扫参数、不训练。"
    elif free_decision is not None and free_decision["pass"]:
        status = "PHASE12_PRESCRIBED_AND_FREE_PASSED"
        result = "同一 actual-q reference 同时通过 prescribed-root trackability 与一次 free-root 裸PD门。"
        conclusion = "Gold可作为目标本体原生训练管线 sanity set；这仍不证明GMR、Any2Any或跨具身成功。"
        next_step = "可由主线决定是否进入独立的最小训练sanity；本阶段不训练。"
    else:
        status = "PHASE12_PRESCRIBED_PASSED_FREE_REJECTED"
        result = "prescribed-root 可跟踪，但同一 reference 的一次 free-root 裸PD replay 失败。"
        conclusion = "生成/ingestion管线可跟踪；裸PD没有复现原控制器闭环平衡。free失败不否定Any2Any。"
        next_step = "保留Gold作pipeline sanity；若未来训练，必须用闭环policy裁决，不能靠裸PD替代。"

    report = {
        "schema_version": "x2_native_gold_trackability_phase12_v1",
        "provenance": {
            "held_out_gold": {"path": str(args.gold), "sha256": sha256(args.gold)},
            "phase11_report": {"path": str(PHASE11_REPORT), "sha256": sha256(PHASE11_REPORT)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": sha256(physics.DEFAULT_CONTROL)},
            "clip": reference["key"],
        },
        "truth_boundary": {
            "target_is_recorded_actual_q_not_command": True,
            "phase11_motionlib_and_state_adapter_reused": True,
            "source_trace_stability_is_not_replay_stability": True,
            "prescribed_root_is_not_balance": True,
            "free_failure_only_rejects_bare_pd_replay_not_any2any": True,
            "contact_is_not_hardware_grf_cop_wrench": True,
            "training_ppo_lora_base_real_robot_git_baidu": False,
            "invalid_pre_report_attempt": "one identical prescribed execution completed before a post-simulation list-index reporting TypeError; no metric, threshold, control, seed, or physics parameter was changed before deterministic rerun",
        },
        "hypothesis": "X2-native actual-q Gold should be trackable in its official model under prescribed root; only then is one free-root bare-PD replay informative.",
        "intervention": "One prescribed-root replay followed, conditionally, by one free-root replay; actual-q target, official PD, no scan.",
        "control": "Frozen Phase10 source clip0 through Phase11 MotionLib+state adapter; source trace is provenance only.",
        "pre_registered_gates": GATES,
        "adapter": adapter_meta,
        "source_reference": source,
        "initialization": prescribed["initialization"],
        "replay": replays,
        "gate": {"prescribed": prescribed_decision, "free": free_decision},
        "decision": {"status": status, "result": result, "conclusion": conclusion, "next_step": next_step},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
