#!/usr/bin/env python3
"""Read-only earliest-divergence audit for Stage250 vs Phase28.

This tool deliberately does not replay physics.  It aligns the historical
closed AimDK/ROS trace with the already-recorded Phase28 direct-MuJoCo cache
at the first stand state, then reports what can and cannot be attributed from
the surviving evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
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


def _default_pose() -> dict[str, float]:
    pose = {name: 0.0 for name in ISAAC_JOINTS}
    for side in ("left", "right"):
        pose[f"{side}_hip_pitch_joint"] = -0.248
        pose[f"{side}_knee_joint"] = 0.5303
        pose[f"{side}_ankle_pitch_joint"] = -0.2823
        pose[f"{side}_shoulder_pitch_joint"] = 0.4
        pose[f"{side}_elbow_joint"] = -1.2
    return pose


def _quat_from_yaw_gravity(yaw: float, gravity: np.ndarray) -> np.ndarray:
    gx, gy, gz = gravity
    pitch = math.asin(float(np.clip(gx, -1.0, 1.0)))
    roll = math.atan2(-float(gy), -float(gz))
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.asarray([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def _rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def decode_row(
    model: mujoco.MjModel,
    row: dict[str, Any],
    qpos_adr: dict[str, int],
    dof_adr: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(row["obs"], dtype=np.float64)
    if obs.shape != (93,):
        raise ValueError(obs.shape)
    quat = _quat_from_yaw_gravity(float(row["root_yaw_rad"]), obs[6:9])
    qpos = np.zeros(model.nq, dtype=np.float64)
    qvel = np.zeros(model.nv, dtype=np.float64)
    qpos[:3] = [row["root_x_m"], row["root_y_m"], row["root_z_m"]]
    qpos[3:7] = quat
    qvel[:3] = _rotation(quat) @ obs[:3]
    qvel[3:6] = obs[3:6]
    pose = _default_pose()
    for index, name in enumerate(ISAAC_JOINTS):
        qpos[qpos_adr[name]] = pose[name] + obs[12 + index]
        qvel[dof_adr[name]] = obs[43 + index]
    return qpos, qvel


ROOT = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
OFFICIAL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
DEFAULT_HISTORICAL = OFFICIAL / "results/official_native_strict_20260807/stage250_video_straight.json"
DEFAULT_SUBSTEPS = OFFICIAL / "cache/phase28_stage250_extended_direct/stage250_straight_substeps.jsonl.gz"
DEFAULT_CONTROL = OFFICIAL / "cache/phase28_stage250_extended_direct/stage250_straight_control_ticks.json"
DEFAULT_SCENE = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"
DEFAULT_BINARY = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/bin/aima-sim-app"
DEFAULT_MODULE = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/lib/libaima-sim-module-mujoco.so.0.0.0"
DEFAULT_VENDOR_MJ = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/lib/libmujoco.so.3.3.7"
DEFAULT_HISTORICAL_SCENE = ROOT / "assets/official_x2/scene_report.xml"
DEFAULT_LAUNCHER = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/bin/start_sim.sh"
DEFAULT_ROBOT_CONFIG = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml"
DEFAULT_SIM_CONFIG = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/simulator/default.yaml"
DEFAULT_RESOLVED_CONFIG = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/bin/cfg/tmp/temp_cfg_file_for_MujocoSimModule.yaml"
DEFAULT_PHASE28_RUNNER = ROOT / "tools/official_x2/run_phase28_stage250_extended_direct.py"
DEFAULT_OUTPUT = ROOT / "reports/official_x2/phase29_closed_wrapper_divergence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quat_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def vector_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(values * values))),
        "abs_max": float(np.max(np.abs(values))),
        "l2": float(np.linalg.norm(values)),
    }


def top_joint_errors(error: np.ndarray, count: int = 5) -> list[dict[str, Any]]:
    indices = np.argsort(np.abs(error))[::-1][:count]
    return [
        {"joint": ISAAC_JOINTS[int(index)], "signed_error": float(error[index]), "abs_error": float(abs(error[index]))}
        for index in indices
    ]


def compare_state(
    direct_qpos: np.ndarray,
    direct_qvel: np.ndarray,
    historical_qpos: np.ndarray,
    historical_qvel: np.ndarray,
    qpos_adr: dict[str, int],
    dof_adr: dict[str, int],
) -> dict[str, Any]:
    q_error = np.asarray([
        direct_qpos[qpos_adr[name]] - historical_qpos[qpos_adr[name]] for name in ISAAC_JOINTS
    ])
    dq_error = np.asarray([
        direct_qvel[dof_adr[name]] - historical_qvel[dof_adr[name]] for name in ISAAC_JOINTS
    ])
    return {
        "root_position": vector_summary(direct_qpos[:3] - historical_qpos[:3]),
        "root_z_abs_error_m": float(abs(direct_qpos[2] - historical_qpos[2])),
        "root_quaternion_geodesic_rad": quat_geodesic(direct_qpos[3:7], historical_qpos[3:7]),
        "root_linear_velocity": vector_summary(direct_qvel[:3] - historical_qvel[:3]),
        "root_angular_velocity": vector_summary(direct_qvel[3:6] - historical_qvel[3:6]),
        "joint_position": vector_summary(q_error),
        "joint_velocity": vector_summary(dq_error),
        "top_joint_position": top_joint_errors(q_error),
        "top_joint_velocity": top_joint_errors(dq_error),
    }


def load_prefix(path: Path, rows: int = 200) -> list[dict[str, Any]]:
    output = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for _ in range(rows):
            line = handle.readline()
            if not line:
                break
            output.append(json.loads(line))
    return output


def first_crossing(aligned: list[dict[str, Any]], field: tuple[str, ...], threshold: float) -> dict[str, Any] | None:
    for row in aligned:
        value: Any = row["state_error"]
        for key in field:
            value = value[key]
        if float(value) > threshold:
            return {"elapsed_s": row["elapsed_s"], "value": float(value), "threshold": threshold}
    return None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.historical.read_text(encoding="utf-8"))
    historical = [row for row in payload["trace"] if len(row.get("obs", [])) == 93]
    controls = json.loads(args.control.read_text(encoding="utf-8"))
    substeps = load_prefix(args.substeps, 200)
    if len(historical) < 11 or len(controls) < 10 or len(substeps) < 200:
        raise RuntimeError("insufficient 0.2 s prefix")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dof_adr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    historical_states = [decode_row(model, row, qpos_adr, dof_adr) for row in historical[:11]]

    aligned = []
    for tick in range(1, 11):
        direct = substeps[tick * 20 - 1]
        direct_qpos = np.asarray(direct["qpos"], dtype=np.float64)
        direct_qvel = np.asarray(direct["qvel"], dtype=np.float64)
        hq, hv = historical_states[tick]
        action_error = np.asarray(controls[tick]["final_clip_action"], dtype=np.float64) - np.asarray(
            historical[tick]["action"], dtype=np.float64
        )
        aligned.append({
            "elapsed_s": tick * 0.02,
            "historical_trace_index": tick + 10,
            "direct_physics_step": tick * 20,
            "state_error": compare_state(direct_qpos, direct_qvel, hq, hv, qpos_adr, dof_adr),
            "action_error": vector_summary(action_error),
            "top_action_indices": np.argsort(np.abs(action_error))[::-1][:5].astype(int).tolist(),
        })

    initial_qpos, initial_qvel = historical_states[0]
    direct_qacc_first_ms = (np.asarray(substeps[0]["qvel"]) - initial_qvel) / 0.001
    historical_qacc_mean_20ms = (historical_states[1][1] - initial_qvel) / 0.02
    qacc_delta = direct_qacc_first_ms - historical_qacc_mean_20ms
    joint_qacc_delta = np.asarray([qacc_delta[dof_adr[name]] for name in ISAAC_JOINTS])

    initial_action_error = np.asarray(controls[0]["final_clip_action"]) - np.asarray(historical[0]["action"])
    initial_contacts = substeps[0]["contacts"]
    initial_sides = sorted({str(contact["side"]) for contact in initial_contacts if contact.get("side")})
    compiled = {
        "python_direct_runtime_version": mujoco.__version__,
        "python_direct_mj_version_string": mujoco.mj_versionString(),
        "closed_module_linked_library_filename": args.vendor_mujoco.name,
        "version_matched": mujoco.mj_versionString() == "3.3.7",
        "model_dimensions": {"nq": model.nq, "nv": model.nv, "na": model.na, "nu": model.nu, "nkey": model.nkey},
        "model_options": {
            "timestep_s": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "disableflags": int(model.opt.disableflags),
            "enableflags": int(model.opt.enableflags),
        },
    }
    timing = {}
    for key in ("control_wall_dt_s", "source_meas_skew_s", "source_callback_age_max_s"):
        values = np.asarray([float(row[key]) for row in historical[:11]], dtype=np.float64)
        timing[key] = {
            "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    observable_matrix = {
        "qpos_qvel_root_obs_action_at_50hz": {"historical": True, "direct": True, "comparable": True},
        "qacc": {
            "historical": "finite difference at 50 Hz only",
            "direct": "finite difference at 1 kHz only; Phase28 did not persist data.qacc",
            "comparable": "diagnostic only, bandwidths differ",
        },
        "contact_and_mj_contactForce": {"historical": False, "direct": True, "comparable": False},
        "ctrl_actuator_force_qfrc_constraint": {"historical": False, "direct": True, "comparable": False},
        "actuator_activation": {
            "historical": False,
            "direct": "structurally absent because official compiled model na=0",
            "comparable": "not a hidden activation candidate for this motor model",
        },
        "qacc_warmstart_solver_efc_state": {"historical": False, "direct": False, "comparable": False},
        "closed_wrapper_reset_order_and_command_arrival_order": {
            "historical": "binary/log evidence only, no per-step state",
            "direct": "source-visible: assign visible qpos/qvel then mj_forward",
            "comparable": False,
        },
        "control_sensor_scheduling": {
            "historical": "ROS best-effort, 1 kHz joint + 500 Hz IMU publishers, asynchronous subscriber workers",
            "direct": "synchronous 20 mj_step calls per policy tick",
            "comparable": False,
        },
        "compiled_model": {
            "historical": "same source scene path was mounted, but closed compiled mjModel was not dumped",
            "direct": "source scene hash plus Python MuJoCo compiled model options",
            "comparable": "source XML yes; compiled mjModel no",
        },
    }

    crossings = {
        "root_position_l2_gt_0p01_m": first_crossing(aligned, ("root_position", "l2"), 0.01),
        "root_orientation_gt_0p05_rad": first_crossing(aligned, ("root_quaternion_geodesic_rad",), 0.05),
        "joint_position_absmax_gt_0p05_rad": first_crossing(aligned, ("joint_position", "abs_max"), 0.05),
        "joint_velocity_absmax_gt_0p5_radps": first_crossing(aligned, ("joint_velocity", "abs_max"), 0.5),
    }

    assets = {}
    for key, path in {
        "historical_trace": args.historical,
        "phase28_substeps": args.substeps,
        "phase28_controls": args.control,
        "scene_xml": args.scene,
        "historical_reporting_scene_xml": args.historical_scene,
        "closed_binary": args.binary,
        "closed_mujoco_module": args.module,
        "closed_vendor_mujoco": args.vendor_mujoco,
        "closed_launcher": args.launcher,
        "closed_robot_config": args.robot_config,
        "closed_simulator_config": args.sim_config,
        "closed_resolved_runtime_config": args.resolved_config,
        "phase28_direct_runner": args.phase28_runner,
    }.items():
        assets[key] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    independent_domain_differences = [
        {
            "name": "MuJoCo runtime version",
            "closed": "libmujoco.so.3.3.7",
            "direct": mujoco.mj_versionString(),
            "status": "confirmed mismatch",
        },
        {
            "name": "pre-stand physical history",
            "closed": "wrapper reset/default followed by 0.2 s prepare; trace rows 0-9 survive",
            "direct": "visible stand-start qpos/qvel assigned, then mj_forward; no prepare integration history restored",
            "status": "confirmed pipeline mismatch; hidden warmstart/contact state unavailable",
        },
        {
            "name": "physics/control scheduling",
            "closed": "real-time ROS callbacks with nonzero sensor skew and callback age",
            "direct": "synchronous exact 20x1 ms stepping then inference",
            "status": "confirmed pipeline mismatch; exact command application substep is unobserved",
        },
        {
            "name": "control and observation at stand t=0",
            "closed": "historical row 10",
            "direct": "reconstructed from historical row 10",
            "status": "matched in visible state/action contract",
        },
    ]

    unique_single_variable = False
    return {
        "stage": "BASE Phase29",
        "scope": "read-only; no replay, training, controller change, Git, cloud, or hardware",
        "assets": assets,
        "compiled_provenance": compiled,
        "closed_runtime_readonly_audit": {
            "launcher": "start_sim.sh selects lx2501_3_t2d5 and launches stripped aima-sim-app with simulator/default.yaml",
            "model_initialization_binary_evidence": (
                "module strings state nominal_configuration initial-qpos initialization; dynamic symbols include "
                "mj_makeData/mj_forward/mj_step/mj_resetData/mj_resetDataKeyframe/mj_getState/mj_setState"
            ),
            "reset_symbol_boundary": (
                "ModelManager::Reset() disassembly deletes mjData/mjModel; closed simulation/UI code owns the "
                "subsequent reset/keyframe and command timing"
            ),
            "publishers": {"joint_state_hz": 1000, "imu_and_odom_hz": 500},
            "subscriber_executor_threads": 5,
            "ros_channel_qos": "best_effort",
            "full_state_injection_or_snapshot_api_exposed_to_project": False,
        },
        "alignment": {
            "historical_prepare_rows": 10,
            "historical_prepare_duration_s": 0.2,
            "historical_stand_zero_trace_index": 10,
            "direct_initialization": "decode historical stand-zero qpos/qvel + mj_forward",
            "initial_visible_state_exact_by_construction": True,
            "initial_action_abs_max_error": float(np.max(np.abs(initial_action_error))),
            "initial_direct_contacts": {"count": len(initial_contacts), "foot_sides": initial_sides},
            "historical_prefix_timing": timing,
            "scene_source_note": (
                "Historical Stage250 mounted scene_report.xml; its only semantic change from the vendor scene is "
                "the reporting camera statistic center/extent. Physics include, floor, and x2.xml are unchanged, "
                "but the closed compiled mjModel was not persisted."
            ),
        },
        "first_0p2s": {
            "aligned_control_boundaries": aligned,
            "diagnostic_crossings_not_gates": crossings,
            "first_ms_vs_historical_first_20ms_acceleration": {
                "bandwidth_warning": "not an equal-bandwidth comparison",
                "root_linear_delta": vector_summary(qacc_delta[:3]),
                "root_angular_delta": vector_summary(qacc_delta[3:6]),
                "joint_delta": vector_summary(joint_qacc_delta),
                "top_joint_delta": top_joint_errors(joint_qacc_delta),
            },
        },
        "observable_matrix": observable_matrix,
        "independent_domain_differences": independent_domain_differences,
        "root_cause_decision": {
            "unique_single_variable_identified": unique_single_variable,
            "status": "BLOCKED_BY_CLOSED_WRAPPER",
            "reason": (
                "Visible state/action match at t=0, but multiple independent variables remain: "
                "MuJoCo 3.3.7 vs 3.4.0, missing pre-stand solver/contact history, and asynchronous ROS "
                "command/sensor scheduling vs synchronous direct stepping.  The historical "
                "trace exposes neither compiled mjModel nor warmstart/contact/ctrl state, so existing "
                "evidence cannot assign the first acceleration divergence to one variable."
            ),
            "minimal_probe_preregistered_not_run": {
                "control": "Phase28 direct prefix under Python MuJoCo 3.4.0 from identical visible state",
                "candidate": "same direct prefix under Python MuJoCo 3.3.7, identical script/assets/state/action",
                "horizon_s": 0.2,
                "single_variable": "MuJoCo runtime version only",
                "metrics": "qpos/qvel/action/contact at 1 ms and 20 ms boundaries",
                "interpretation": (
                    "If 3.3.7 remains far from the historical trace, version is eliminated as sufficient; "
                    "the next required evidence is a closed-wrapper full integration-state snapshot at stand t=0."
                ),
                "execution_authorized": False,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--substeps", type=Path, default=DEFAULT_SUBSTEPS)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument("--vendor-mujoco", type=Path, default=DEFAULT_VENDOR_MJ)
    parser.add_argument("--historical-scene", type=Path, default=DEFAULT_HISTORICAL_SCENE)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--sim-config", type=Path, default=DEFAULT_SIM_CONFIG)
    parser.add_argument("--resolved-config", type=Path, default=DEFAULT_RESOLVED_CONFIG)
    parser.add_argument("--phase28-runner", type=Path, default=DEFAULT_PHASE28_RUNNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["root_cause_decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
