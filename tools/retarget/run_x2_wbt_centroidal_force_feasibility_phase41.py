#!/usr/bin/env python3
"""Phase41 official-model centroidal/contact-force feasibility oracle.

This evaluates the frozen Phase40 25-frame path.  It does not integrate the
model and does not train a policy.  COM, centroidal momentum and contact force
are MuJoCo-model estimates, not measured X2 GRF/COP truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.optimize import linprog


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29
import retarget.run_x2_wbt_time_dilation_phase30 as phase30
import retarget.run_x2_wbt_official_contact_repair_phase37 as phase37
import retarget.run_x2_wbt_hard_constraint_certificate_phase38 as phase38
import retarget.run_x2_wbt_rootxy_sequential_qp_phase40 as phase40


CONTACT_DISTANCE_LOWER_M = phase38.NONPENETRATION_LOWER_M
CONTACT_DISTANCE_UPPER_M = phase38.STANCE_DISTANCE_UPPER_M
BALANCE_EQUALITY_TOL = 1.0e-5
FRICTION_TOL = 1.0e-7
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_centroidal_force_feasibility_phase41.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_centroidal_force_feasibility_phase41.md"
OUTPUT_NPZ = REPO / "artifacts/official_x2/x2_wbt_phase41_centroidal_force_oracle.npz"


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def build_qpos(model: mujoco.MjModel, entry: dict[str, Any]) -> np.ndarray:
    names = list(entry["joint_names_mujoco"])
    addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in names
    ]
    qpos = np.tile(model.qpos0, (len(entry["dof"]), 1)).astype(np.float64)
    qpos[:, :3] = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat_xyzw = np.asarray(entry["root_rot"], dtype=np.float64)
    qpos[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
    for entry_index, address in enumerate(addresses):
        qpos[:, address] = np.asarray(entry["dof"], dtype=np.float64)[:, entry_index]
    return qpos


def differentiate_qpos(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    """Frame-centred generalized velocity using MuJoCo quaternion difference."""
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    dt = 1.0 / fps
    mujoco.mj_differentiatePos(model, qvel[0], dt, qpos[0], qpos[1])
    mujoco.mj_differentiatePos(model, qvel[-1], dt, qpos[-2], qpos[-1])
    for frame in range(1, len(qpos) - 1):
        mujoco.mj_differentiatePos(model, qvel[frame], 2.0 * dt, qpos[frame - 1], qpos[frame + 1])
    return qvel


def model_centroidal_series(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    com = np.zeros((len(qpos), 3), dtype=np.float64)
    angular_momentum = np.zeros((len(qpos), 3), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        # subtree_angmom is an optional velocity-derived field and is not
        # populated by mj_forward in the pinned MuJoCo build.
        mujoco.mj_subtreeVel(model, data)
        # MuJoCo subtree 0 is the complete floating-base robot; angmom is
        # expressed about subtree_com[0] in world coordinates.
        com[frame] = data.subtree_com[0]
        angular_momentum[frame] = data.subtree_angmom[0]
    return com, angular_momentum


def contact_candidates(
    model: mujoco.MjModel, qpos: np.ndarray, floor: int, feet: dict[str, list[int]],
) -> list[dict[str, Any]]:
    data = mujoco.MjData(model)
    fromto = np.zeros(6, dtype=np.float64)
    result: list[dict[str, Any]] = []
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        per_side: dict[str, Any] = {}
        for side, geoms in feet.items():
            distances, points = [], []
            for geom in geoms:
                distance = float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto))
                distances.append(distance)
                # Closest point on floor. With the official scene floor is the
                # first geom; keep an explicit guard rather than assume order.
                point = fromto[:3].copy()
                point[2] = 0.0
                points.append(point)
            distances = np.asarray(distances, dtype=np.float64)
            points = np.asarray(points, dtype=np.float64)
            active = np.flatnonzero(
                (distances >= CONTACT_DISTANCE_LOWER_M)
                & (distances <= CONTACT_DISTANCE_UPPER_M)
            )
            per_side[side] = {
                "distance": distances, "point": points, "active_indices": active,
                "min_distance": float(np.min(distances)),
            }
        result.append(per_side)
    return result


def solve_frame_force_lp(
    mass: float, gravity: np.ndarray, com: np.ndarray, com_accel: np.ndarray,
    hdot: np.ndarray, candidate_points: list[tuple[str, int, np.ndarray]], mu: float,
) -> dict[str, Any]:
    count = len(candidate_points)
    if count == 0:
        return {"feasible": False, "solver_status": -1, "solver_message": "no geometrically active stance contact sphere"}
    variables = 3 * count
    aeq = np.zeros((6, variables), dtype=np.float64)
    for index, (_, _, point) in enumerate(candidate_points):
        columns = slice(3 * index, 3 * index + 3)
        aeq[:3, columns] = np.eye(3)
        aeq[3:, columns] = skew(point - com)
    beq = np.r_[mass * (com_accel - gravity), hdot]
    aub, bub = [], []
    for index in range(count):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                row = np.zeros(variables, dtype=np.float64)
                row[3 * index] = sx
                row[3 * index + 1] = sy
                row[3 * index + 2] = -mu
                aub.append(row); bub.append(0.0)
    bounds = []
    for _ in range(count):
        bounds.extend([(None, None), (None, None), (0.0, None)])
    optimization = linprog(
        np.zeros(variables, dtype=np.float64), A_ub=np.asarray(aub), b_ub=np.asarray(bub),
        A_eq=aeq, b_eq=beq, bounds=bounds, method="highs",
        options={"presolve": True, "dual_feasibility_tolerance": 1.0e-8, "primal_feasibility_tolerance": 1.0e-8},
    )
    if not optimization.success:
        return {
            "feasible": False, "solver_status": int(optimization.status),
            "solver_message": str(optimization.message), "candidate_count": count,
            "target_force_n": beq[:3], "target_moment_nm": beq[3:],
        }
    force = np.asarray(optimization.x).reshape(count, 3)
    balance = aeq @ optimization.x - beq
    friction = np.abs(force[:, 0]) + np.abs(force[:, 1]) - mu * force[:, 2]
    per_side: dict[str, Any] = {}
    for side in phase29.SIDES:
        indices = [i for i, item in enumerate(candidate_points) if item[0] == side]
        normal = float(np.sum(force[indices, 2])) if indices else 0.0
        cop = None
        if normal > 1.0e-12:
            points = np.asarray([candidate_points[i][2] for i in indices])
            cop = np.sum(points[:, :2] * force[indices, 2, None], axis=0) / normal
        per_side[side] = {"normal_force_n": normal, "cop_xy_m": cop, "candidate_count": len(indices)}
    residual = float(np.max(np.abs(balance)))
    friction_violation = float(max(0.0, np.max(friction)))
    return {
        "feasible": bool(residual <= BALANCE_EQUALITY_TOL and friction_violation <= FRICTION_TOL),
        "solver_status": int(optimization.status), "solver_message": str(optimization.message),
        "candidate_count": count, "target_force_n": beq[:3], "target_moment_nm": beq[3:],
        "balance_residual_max": residual, "friction_violation_max": friction_violation,
        "force_n": force, "per_side": per_side,
    }


def render(report: dict[str, Any]) -> str:
    result = report["result"]
    return f"""# X2 WBT Phase41：centroidal / contact-force feasibility oracle

## 假设

Phase40 固定的25帧 DS→R-SS→DS 轨迹只有在 official X2 模型上存在满足六维平衡、接触激活和摩擦锥的 GRF，才值得进入动力学感知 alternating teacher。

## 干预

- 路径、30Hz、root orientation、contact intent、official `x2.xml` 全部冻结。
- MuJoCo 模型计算 COM、COM acceleration、centroidal angular-momentum rate。
- 每帧只允许 intended stance 且 official sole signed-distance 位于 [-0.01, 0.5]mm 的 sphere 承力；非接触脚 force=0。
- 接触力满足非负法向力、official μ={report['pre_registration']['friction_mu']:.3f} 的保守L1摩擦锥、总力/总矩等式。非负法向力在实际sphere点的凸组合同时形成足底内COP。

## 对照与结果

- 有效导数帧：{result['evaluated_frames']}；几何激活通过：{result['geometry_qualified_frames']}/{result['evaluated_frames']}。
- force LP feasible：{result['force_lp_feasible_frames']}/{result['evaluated_frames']}；geometry+force共同可行：{result['jointly_feasible_frames']}/{result['evaluated_frames']}。
- 不可行帧：{result['infeasible_frames']}。
- COM acceleration norm p95/max：{result['com_acceleration_norm_p95_max_mps2']}m/s²。
- centroidal Hdot norm p95/max：{result['centroidal_hdot_norm_p95_max_nms']}N·m。

## 结论

**{report['decision']['status']}**

{report['decision']['conclusion']}

这些 GRF/COP/contact 都是 official MuJoCo 模型估计量，**不是 X2 实机足底力、COP 或动力学真值**。

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    phase40_report = json.loads(phase40.OUTPUT_JSON.read_text())
    artifact = np.load(phase40.OUTPUT_NPZ)
    panel = {row["id"]: row for row in json.loads(phase28.PANEL.read_text())["motions"]}
    row = panel[phase38.MOTION_ID]
    source = joblib.load(phase30.OUTPUT_CACHE)[phase38.MOTION_ID]
    warm = joblib.load(phase37.OUTPUT_CACHE)[phase38.MOTION_ID]
    source29 = joblib.load(phase29.OUTPUT_CACHE)[phase38.MOTION_ID]
    physics = phase38.phase3.load_module(phase38.phase3.PHYSICS_SCRIPT, "x2_phase41_centroidal_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    source_contract = phase29.source_contract(row, source29, model)
    intent = {
        side: phase30.nearest_phase_resample(source_contract["contact"][side], len(source["dof"]))
        for side in phase29.SIDES
    }
    lower = phase29.lower_body_indices(list(source["joint_names_mujoco"]))
    problem = phase40.RootXYWindowProblem(model, source, warm, intent, lower)
    correction = np.asarray(artifact["correction"], dtype=np.float64)
    if correction.shape != (problem.nframes, problem.width):
        raise ValueError(f"Phase40 correction shape mismatch: {correction.shape}")
    entry = problem.entry_from_x(correction.reshape(-1))
    frames = problem.frames
    fps = float(entry["fps"])
    qpos_all = build_qpos(model, entry)
    # Derivatives use the complete Phase30 trajectory so window endpoints do
    # not receive invented zero velocities. Phase40 only changes frames 0:25.
    qvel_all = differentiate_qpos(model, qpos_all, fps)
    com_all, h_all = model_centroidal_series(model, qpos_all, qvel_all)
    dt = 1.0 / fps
    com_accel = (com_all[2:] - 2.0 * com_all[1:-1] + com_all[:-2]) / (dt * dt)
    hdot = (h_all[2:] - h_all[:-2]) / (2.0 * dt)
    floor, feet = phase38.phase7.active_sole_spheres(model)
    contacts = contact_candidates(model, qpos_all, floor, feet)
    mu = float(min(
        model.geom_friction[floor, 0],
        *[model.geom_friction[geom, 0] for geoms in feet.values() for geom in geoms],
    ))
    mass = float(np.sum(model.body_mass))
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    frame_results, force_rows, force_offsets = [], [], [0]
    evaluated = list(range(phase38.WINDOW_START + 1, phase38.WINDOW_END_EXCLUSIVE - 1))
    for frame in evaluated:
        points: list[tuple[str, int, np.ndarray]] = []
        geometry_ok = True
        geometry: dict[str, Any] = {}
        for side in phase29.SIDES:
            intended = bool(intent[side][frame])
            info = contacts[frame][side]
            indices = np.asarray(info["active_indices"], dtype=np.int64) if intended else np.asarray([], dtype=np.int64)
            if intended and len(indices) == 0:
                geometry_ok = False
            if float(np.min(info["distance"])) < CONTACT_DISTANCE_LOWER_M - 1.0e-12:
                geometry_ok = False
            for sphere in indices:
                points.append((side, int(sphere), np.asarray(info["point"])[int(sphere)]))
            geometry[side] = {
                "intended": intended, "active_sphere_indices": indices.tolist(),
                "min_signed_distance_m": float(info["min_distance"]),
            }
        force = solve_frame_force_lp(
            mass, gravity, com_all[frame], com_accel[frame - 1], hdot[frame - 1], points, mu,
        )
        force_feasible = bool(force["feasible"])
        jointly = bool(geometry_ok and force_feasible)
        forces = np.asarray(force.get("force_n", np.zeros((0, 3))), dtype=np.float64)
        force_rows.append(forces)
        force_offsets.append(force_offsets[-1] + len(forces))
        frame_results.append({
            "frame": frame, "time_s": frame / fps, "contact_geometry": geometry,
            "geometry_qualified": geometry_ok, "force_lp": force,
            "jointly_feasible": jointly, "com_m": com_all[frame],
            "com_acceleration_mps2": com_accel[frame - 1], "centroidal_hdot_nms": hdot[frame - 1],
        })
    geometry_count = sum(item["geometry_qualified"] for item in frame_results)
    force_count = sum(item["force_lp"]["feasible"] for item in frame_results)
    joint_count = sum(item["jointly_feasible"] for item in frame_results)
    all_feasible = joint_count == len(frame_results)
    com_norm = np.linalg.norm(np.asarray([item["com_acceleration_mps2"] for item in frame_results]), axis=1)
    hdot_norm = np.linalg.norm(np.asarray([item["centroidal_hdot_nms"] for item in frame_results]), axis=1)
    result = {
        "evaluated_frames": len(frame_results), "frame_indices": evaluated,
        "geometry_qualified_frames": int(geometry_count), "force_lp_feasible_frames": int(force_count),
        "jointly_feasible_frames": int(joint_count),
        "infeasible_frames": [item["frame"] for item in frame_results if not item["jointly_feasible"]],
        "com_acceleration_norm_p95_max_mps2": [float(np.percentile(com_norm, 95)), float(np.max(com_norm))],
        "centroidal_hdot_norm_p95_max_nms": [float(np.percentile(hdot_norm, 95)), float(np.max(hdot_norm))],
        "all_frames_centroidal_contact_force_feasible": all_feasible,
    }
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ, frames=np.asarray(evaluated), com=com_all[evaluated],
        com_acceleration=np.asarray([item["com_acceleration_mps2"] for item in frame_results]),
        centroidal_hdot=np.asarray([item["centroidal_hdot_nms"] for item in frame_results]),
        force=np.concatenate(force_rows, axis=0) if force_rows else np.zeros((0, 3)),
        force_offsets=np.asarray(force_offsets, dtype=np.int64),
        jointly_feasible=np.asarray([item["jointly_feasible"] for item in frame_results]),
    )
    if all_feasible:
        status = "PHASE41_CENTROIDAL_FORCE_ORACLE_FEASIBLE"
        conclusion = "冻结Phase40局部轨迹在全部可评帧存在模型GRF硬可行解，满足进入一次固定q/root+force alternating teacher的必要条件；这仍不是充分的可执行控制证明。"
        next_step = "允许同阶段唯一一次预注册alternating/SQP teacher；不得扫权重，完成后停止。"
    else:
        status = "PHASE41_CENTROIDAL_FORCE_ORACLE_INFEASIBLE"
        conclusion = "冻结Phase40局部轨迹未通过模型接触几何+六维GRF硬可行性层；当前不允许用soft reward或policy训练绕过，也不进入alternating teacher。"
        next_step = "按门停止；仅报告具体几何激活或力/矩平衡缺口，不调路径、contact schedule、摩擦或阈值。"
    report = {
        "schema_version": "x2_wbt_centroidal_force_feasibility_phase41_v1",
        "truth_boundary": {
            "official_mujoco_model_estimate_not_hardware_truth": True,
            "mujoco_forward_calls_no_integration": int(len(qpos_all) * 2),
            "mujoco_integration_steps": 0, "policy_physics_rollout": False,
            "ppo_or_policy_training": False, "held_out_read": False,
            "force_oracle_configuration_count": 1,
            "invalid_first_invocation": "mj_forward did not populate subtree_angmom in MuJoCo 3.3.7; all-zero Hdot made that report invalid. Added required mj_subtreeVel call and reran unchanged force-oracle contract.",
        },
        "pre_registration": {
            "window_frames": [phase38.WINDOW_START, phase38.WINDOW_END_EXCLUSIVE],
            "derivative_evaluation_frames": [phase38.WINDOW_START + 1, phase38.WINDOW_END_EXCLUSIVE - 1],
            "contact_candidate_signed_distance_band_m": [CONTACT_DISTANCE_LOWER_M, CONTACT_DISTANCE_UPPER_M],
            "friction_mu": mu, "friction_cone": "inscribed L1 pyramid: |fx|+|fy|<=mu*fz",
            "normal_force_nonnegative": True, "noncontact_foot_force_zero": True,
            "cop_contract": "normal-force convex combination of official sole-sphere ground points; hence inside active contact-point convex hull",
            "force_moment_balance": "sum(f)=m*(COMddot-gravity); sum((p-COM)xf)=dHcentroidal/dt",
            "balance_equality_tolerance": BALANCE_EQUALITY_TOL,
            "parameter_scan": False,
            "alternating_teacher_unlock": "all derivative-valid frames jointly pass geometry activation and force LP",
        },
        "provenance": {
            "phase40_report_sha256": phase28.sha256(phase40.OUTPUT_JSON),
            "phase40_artifact_sha256": phase28.sha256(phase40.OUTPUT_NPZ),
            "official_scene_sha256": phase28.sha256(physics.DEFAULT_SCENE),
            "phase30_source_sha256": phase28.sha256(phase30.OUTPUT_CACHE),
        },
        "model": {"total_mass_kg": mass, "gravity_mps2": gravity, "official_sole_spheres_per_foot": 12},
        "result": result, "frames": frame_results,
        "artifact": {"path": str(OUTPUT_NPZ), "sha256": phase28.sha256(OUTPUT_NPZ)},
        "decision": {
            "status": status, "centroidal_force_oracle_feasible": all_feasible,
            "alternating_teacher_run": False, "alternating_teacher_allowed": all_feasible,
            "full_trajectory_true_silver": False, "physics_or_ppo_allowed": False,
            "conclusion": conclusion, "next_step": next_step,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
