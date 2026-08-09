#!/usr/bin/env python3
"""Phase39 fixed sequential-convex QP certificate for the Phase38 window."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from qpsolvers import solve_qp


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29
import retarget.run_x2_wbt_time_dilation_phase30 as phase30
import retarget.run_x2_wbt_official_contact_repair_phase37 as phase37
import retarget.run_x2_wbt_hard_constraint_certificate_phase38 as phase38


OUTER_ITERATIONS = 20
ROOT_Z_TRUST_M = 0.005
JOINT_TRUST_RAD = 0.05
QP_REGULARIZATION = 1.0e-8
SOLVER = "daqp"
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_sequential_qp_certificate_phase39.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_sequential_qp_certificate_phase39.md"
OUTPUT_NPZ = REPO / "artifacts/official_x2/x2_wbt_phase39_sequential_qp_window.npz"


def add_quadratic_pair(P: np.ndarray, a: int, b: int, weight: float) -> None:
    P[a, a] += 2.0 * weight
    P[b, b] += 2.0 * weight
    P[a, b] -= 2.0 * weight
    P[b, a] -= 2.0 * weight


def add_quadratic_second(P: np.ndarray, a: int, b: int, c: int, weight: float) -> None:
    indices, values = (a, b, c), (1.0, -2.0, 1.0)
    for i, vi in zip(indices, values):
        for j, vj in zip(indices, values):
            P[i, j] += 2.0 * weight * vi * vj


def linearization(problem: phase38.WindowProblem, x: np.ndarray) -> list[dict[str, Any]]:
    entry = problem.entry_from_x(x)
    data = mujoco.MjData(problem.model)
    fromto = np.zeros(6, dtype=np.float64)
    states = []
    for local, frame in enumerate(problem.frames):
        phase28.set_entry_frame(problem.model, data, entry, int(frame), problem.qpos_addresses)
        points, point_jac = [], []
        for body_id in problem.body_ids:
            jacp = np.zeros((3, problem.model.nv), dtype=np.float64)
            jacr = np.zeros((3, problem.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(problem.model, data, jacp, jacr, body_id)
            points.append(data.xpos[body_id].copy())
            point_jac.append(jacp[:, problem.variable_dofs].copy())
        sphere_distance, sphere_jac, active, foot_xy, foot_xy_jac = {}, {}, {}, {}, {}
        for side, geoms in problem.feet.items():
            distances, jacobians = [], []
            for geom in geoms:
                distances.append(float(mujoco.mj_geomDistance(problem.model, data, problem.floor, geom, 1.0, fromto)))
                jacp = np.zeros((3, problem.model.nv), dtype=np.float64)
                jacr = np.zeros((3, problem.model.nv), dtype=np.float64)
                mujoco.mj_jacGeom(problem.model, data, jacp, jacr, geom)
                jacobians.append(jacp[2, problem.variable_dofs].copy())
            sphere_distance[side] = np.asarray(distances)
            sphere_jac[side] = np.asarray(jacobians)
            active[side] = int(np.argmin(distances))
            body_id = mujoco.mj_name2id(problem.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            jacp = np.zeros((3, problem.model.nv), dtype=np.float64)
            jacr = np.zeros((3, problem.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(problem.model, data, jacp, jacr, body_id)
            foot_xy[side] = data.xpos[body_id, :2].copy()
            foot_xy_jac[side] = jacp[:2, problem.variable_dofs].copy()
        states.append({
            "points": np.asarray(points), "point_jac": np.asarray(point_jac),
            "sphere_distance": sphere_distance, "sphere_jac": sphere_jac,
            "active": active, "foot_xy": foot_xy, "foot_xy_jac": foot_xy_jac,
        })
    return states


def objective_qp(problem: phase38.WindowProblem, xk: np.ndarray, states: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    n, width = problem.nvar, problem.width
    weights = phase38.OBJECTIVE_WEIGHTS
    P = np.eye(n, dtype=np.float64) * QP_REGULARIZATION
    q = np.zeros(n, dtype=np.float64)
    for local in range(problem.nframes):
        columns = problem.cols(local)
        P[columns[0], columns[0]] += 2.0 * weights["root_z_correction"]
        for column in columns[1:]:
            P[column, column] += 2.0 * weights["joint_correction"]
    for local in range(1, problem.nframes):
        for component in range(width):
            add_quadratic_pair(
                P, int(problem.cols(local)[component]), int(problem.cols(local - 1)[component]),
                weights["correction_velocity"],
            )
    for local in range(1, problem.nframes - 1):
        for component in range(width):
            add_quadratic_second(
                P, int(problem.cols(local - 1)[component]), int(problem.cols(local)[component]),
                int(problem.cols(local + 1)[component]), weights["correction_acceleration"],
            )
    for local, state in enumerate(states):
        columns = problem.cols(local)
        xlocal = xk[columns]
        for point, jacobian, target in zip(state["points"], state["point_jac"], problem.original_points[local]):
            # p(x) ~= J*x + (p_k-J*x_k), objective ||p-target||^2.
            offset = point - jacobian @ xlocal - target
            P[np.ix_(columns, columns)] += 2.0 * weights["source_fit_keypoint"] * (jacobian.T @ jacobian)
            q[columns] += 2.0 * weights["source_fit_keypoint"] * (jacobian.T @ offset)
    return 0.5 * (P + P.T), q


def linear_constraints(
    problem: phase38.WindowProblem, xk: np.ndarray, states: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows, bounds, counts = [], [], {
        "sphere_nonpenetration": 0, "stance_active_band": 0, "swing_clearance": 0,
        "stance_speed": 0, "stance_excursion": 0, "qstep": 0,
    }

    def upper(row: np.ndarray, value: float, category: str) -> None:
        rows.append(row); bounds.append(float(value)); counts[category] += 1

    def interval(row: np.ndarray, constant: float, lower: float, high: float, category: str) -> None:
        # linearized f(x)=row*x+constant
        upper(row, high - constant, category)
        upper(-row, -lower + constant, category)

    for local, frame in enumerate(problem.frames):
        columns = problem.cols(local)
        for side in phase29.SIDES:
            state = states[local]
            active = state["active"][side]
            for sphere in range(12):
                row = np.zeros(problem.nvar)
                jac = state["sphere_jac"][side][sphere]
                row[columns] = jac
                distance = state["sphere_distance"][side][sphere]
                constant = distance - row @ xk
                desired = phase38.NONPENETRATION_LOWER_M
                if not problem.intent[side][frame]:
                    desired = max(desired, max(0.0, float(problem.original_distance[side][local])))
                    category = "swing_clearance"
                else:
                    category = "sphere_nonpenetration"
                upper(-row, -desired + constant, category)
            if problem.intent[side][frame]:
                row = np.zeros(problem.nvar)
                jac = state["sphere_jac"][side][active]
                row[columns] = jac
                distance = state["sphere_distance"][side][active]
                constant = distance - row @ xk
                interval(row, constant, 0.0, phase38.STANCE_DISTANCE_UPPER_M, "stance_active_band")

    # Convex norms are linearized at the current iterate, then exact norms are
    # rechecked after every QP. Trust regions limit tangent error.
    for side in phase29.SIDES:
        for local in range(1, problem.nframes):
            f0, f1 = int(problem.frames[local - 1]), int(problem.frames[local])
            if problem.intent[side][f0] and problem.intent[side][f1]:
                delta = states[local]["foot_xy"][side] - states[local - 1]["foot_xy"][side]
                norm = float(np.linalg.norm(delta))
                if norm > 1.0e-12:
                    direction = delta / norm
                    row = np.zeros(problem.nvar)
                    row[problem.cols(local)] = direction @ states[local]["foot_xy_jac"][side]
                    row[problem.cols(local - 1)] -= direction @ states[local - 1]["foot_xy_jac"][side]
                    constant = norm - row @ xk
                    upper(row, phase38.STANCE_SPEED_MAX_MPS / float(problem.source["fps"]) - constant, "stance_speed")
        mask = problem.intent[side][problem.frames]
        for start, end in phase28.contiguous_true(mask):
            for local in range(start + 1, end):
                delta = states[local]["foot_xy"][side] - states[start]["foot_xy"][side]
                norm = float(np.linalg.norm(delta))
                if norm > 1.0e-12:
                    direction = delta / norm
                    row = np.zeros(problem.nvar)
                    row[problem.cols(local)] = direction @ states[local]["foot_xy_jac"][side]
                    row[problem.cols(start)] -= direction @ states[start]["foot_xy_jac"][side]
                    constant = norm - row @ xk
                    upper(row, phase38.STANCE_EXCURSION_MAX_M - constant, "stance_excursion")

    # Exact affine q-step constraints, including fixed frame25 boundary.
    for frame in range(phase38.WINDOW_START + 1, phase38.WINDOW_END_EXCLUSIVE + 1):
        for joint_local, entry_index in enumerate(problem.lower_indices):
            row = np.zeros(problem.nvar)
            if frame < phase38.WINDOW_END_EXCLUSIVE:
                row[problem.cols(frame - phase38.WINDOW_START)[1 + joint_local]] = 1.0
            row[problem.cols(frame - 1 - phase38.WINDOW_START)[1 + joint_local]] -= 1.0
            base = problem.original_q[frame, entry_index] - problem.original_q[frame - 1, entry_index]
            interval(row, base, -phase38.JOINT_STEP_MAX_RAD, phase38.JOINT_STEP_MAX_RAD, "qstep")
    return np.asarray(rows), np.asarray(bounds), counts


def exact_metrics(problem: phase38.WindowProblem, x: np.ndarray) -> dict[str, Any]:
    values = problem.constraint_values(x)
    lower, upper = problem.nonlinear_bounds()
    violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
    entry = problem.entry_from_x(x)
    kin = phase28.target_kinematics(entry, problem.model)
    distances = {side: np.asarray(kin["sole_distance"][side])[problem.frames] for side in phase29.SIDES}
    stance = np.concatenate([
        np.abs(distances[side][problem.intent[side][problem.frames]]) for side in phase29.SIDES
    ])
    speed, excursion = [], []
    for item, value in zip(problem.constraint_spec, values):
        if item[0] == "stance_speed_step": speed.append(float(value * float(problem.source["fps"])))
        elif item[0] == "stance_excursion": excursion.append(float(value))
    q = np.asarray(entry["dof"])
    qstep = float(np.max(np.abs(np.diff(q[phase38.WINDOW_START:phase38.WINDOW_END_EXCLUSIVE + 1, problem.lower_indices], axis=0))))
    return {
        "max_phase38_constraint_violation": float(np.max(violation)),
        "stance_abs_distance_p95_max_m": [float(np.percentile(stance, 95)), float(np.max(stance))],
        "minimum_all_sphere_distance_m": float(min(np.min(values) for values in distances.values())),
        "stance_speed_max_mps": max(speed, default=0.0),
        "stance_excursion_max_m": max(excursion, default=0.0),
        "qstep_max_rad": qstep,
        "feasible": bool(
            np.max(violation) <= 1.0e-7
            and np.max(stance) <= phase38.STANCE_DISTANCE_UPPER_M + 1.0e-7
            and min(np.min(values) for values in distances.values()) >= phase38.NONPENETRATION_LOWER_M - 1.0e-7
            and max(speed, default=0.0) <= phase38.STANCE_SPEED_MAX_MPS + 1.0e-7
            and max(excursion, default=0.0) <= phase38.STANCE_EXCURSION_MAX_M + 1.0e-7
            and qstep <= phase38.JOINT_STEP_MAX_RAD + 1.0e-7
        ),
    }


def render(report: dict[str, Any]) -> str:
    final = report["final_metrics"]
    return f"""# X2 WBT Phase39：sequential-convex QP hard certificate

## 裁决

- **{report['decision']['status']}**
- 同Phase38 frames `[0,25)`、同变量/门；DAQP固定trust-region外层最多20轮，未扫参数。
- active最低sphere允许随迭代切换；全部12 sphere每帧都有nonpenetration线性约束。

## 结果

- QP solved outer iterations：{report['summary']['qp_solved_iterations']}；failure：`{report['summary']['qp_failure']}`。
- active sphere switches L/R累计：`{report['summary']['active_switches_lr']}`。
- stance distance p95/max：`{final['stance_abs_distance_p95_max_m']}`m；minimum all-sphere distance：{final['minimum_all_sphere_distance_m']:.6f}m。
- stance speed max：{final['stance_speed_max_mps']:.4f}m/s；excursion：{final['stance_excursion_max_m']:.4f}m；qstep：{final['qstep_max_rad']:.4f}rad。
- max exact constraint violation：{final['max_phase38_constraint_violation']:.6g}；local feasible：`{final['feasible']}`。

## 结论

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    panel = {row["id"]: row for row in json.loads(phase28.PANEL.read_text())["motions"]}
    row = panel[phase38.MOTION_ID]
    source = joblib.load(phase30.OUTPUT_CACHE)[phase38.MOTION_ID]
    warm = joblib.load(phase37.OUTPUT_CACHE)[phase38.MOTION_ID]
    source29 = joblib.load(phase29.OUTPUT_CACHE)[phase38.MOTION_ID]
    physics = phase38.phase3.load_module(phase38.phase3.PHYSICS_SCRIPT, "x2_phase39_static_fk_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    source_contract = phase29.source_contract(row, source29, model)
    intent = {
        side: phase30.nearest_phase_resample(source_contract["contact"][side], len(source["dof"]))
        for side in phase29.SIDES
    }
    lower = phase29.lower_body_indices(list(source["joint_names_mujoco"]))
    problem = phase38.WindowProblem(model, source, warm, intent, lower)
    x = problem.x0.copy()
    global_bounds = problem.variable_bounds()
    iterations, previous_active = [], None
    switches = np.zeros(2, dtype=np.int64)
    qp_failure = None
    for outer in range(OUTER_ITERATIONS):
        states = linearization(problem, x)
        active = np.asarray([[state["active"][side] for side in phase29.SIDES] for state in states], dtype=np.int64)
        switch_lr = np.zeros(2, dtype=np.int64) if previous_active is None else np.count_nonzero(active != previous_active, axis=0)
        switches += switch_lr
        previous_active = active.copy()
        P, q = objective_qp(problem, x, states)
        G, h, counts = linear_constraints(problem, x, states)
        trust = np.tile(np.r_[ROOT_Z_TRUST_M, np.full(problem.width - 1, JOINT_TRUST_RAD)], problem.nframes)
        lb = np.maximum(global_bounds.lb, x - trust)
        ub = np.minimum(global_bounds.ub, x + trust)
        solution = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=SOLVER, initvals=x, verbose=False)
        if solution is None:
            qp_failure = f"{SOLVER} returned no solution at outer iteration {outer + 1}"
            break
        x = np.asarray(solution, dtype=np.float64)
        metrics = exact_metrics(problem, x)
        iterations.append({
            "outer_iteration": outer + 1, "linear_constraint_counts": counts,
            "active_sphere_indices_lr": active.tolist(),
            "active_switch_count_lr": switch_lr.tolist(),
            "exact_metrics": metrics,
            "root_z_correction_max_abs_m": float(np.max(np.abs(x.reshape(problem.nframes, problem.width)[:, 0]))),
            "joint_correction_max_abs_rad": float(np.max(np.abs(x.reshape(problem.nframes, problem.width)[:, 1:]))),
        })
        print(
            f"[phase39] outer={outer + 1}/{OUTER_ITERATIONS} feasible={metrics['feasible']} "
            f"violation={metrics['max_phase38_constraint_violation']:.6g} "
            f"stance={metrics['stance_abs_distance_p95_max_m'][1]:.6g}", flush=True,
        )
        if metrics["feasible"]:
            break
    final = exact_metrics(problem, x)
    feasible = final["feasible"]
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ, correction=x.reshape(problem.nframes, problem.width), frames=problem.frames,
        active_sphere_last=previous_active if previous_active is not None else np.zeros((0, 2), dtype=np.int64),
    )
    if feasible:
        status = "PHASE39_LOCAL_SEQUENTIAL_QP_CERTIFIED"
        conclusion = "同一25帧窗口获得真正硬约束几何可行证书；只证明局部表示可行，不是完整轨迹Silver。"
        next_step = "保持physics/PPO锁定；可由主线决定是否以完全同一合同扩展一次完整轨迹。"
    else:
        status = "PHASE39_LOCAL_SEQUENTIAL_QP_CERTIFICATE_FAILED"
        conclusion = "固定sequential-QP合同仍未取得局部证书；这是solver/表示证书失败，不证明X2或该动作物理上不可能。"
        next_step = "按门停止；不调trust-region、不换solver/窗口、不扩完整轨迹或physics。"
    report = {
        "schema_version": "x2_wbt_sequential_qp_certificate_phase39_v1",
        "truth_boundary": {
            "same_phase38_window_thresholds": True, "held_out_read": False,
            "mujoco_integration_steps": 0, "policy_optimizer_ppo_training": False,
            "sequential_qp_configuration_count": 1, "full_trajectory_silver": False,
        },
        "pre_registration": {
            "solver": SOLVER, "outer_iterations_max": OUTER_ITERATIONS,
            "trust_region_root_z_m": ROOT_Z_TRUST_M, "trust_region_joint_rad": JOINT_TRUST_RAD,
            "qp_regularization": QP_REGULARIZATION,
            "window_frames": [phase38.WINDOW_START, phase38.WINDOW_END_EXCLUSIVE],
            "active_sphere_rule": "current minimum signed-distance sphere per intended stance foot/frame; switches recorded",
            "all_sphere_rule": "all 12 active sole spheres/foot constrained nonpenetrating each frame",
            "thresholds_identical_to_phase38": {
                "stance_distance_upper_m": phase38.STANCE_DISTANCE_UPPER_M,
                "nonpenetration_lower_m": phase38.NONPENETRATION_LOWER_M,
                "stance_speed_mps": phase38.STANCE_SPEED_MAX_MPS,
                "stance_excursion_m": phase38.STANCE_EXCURSION_MAX_M,
                "qstep_max_rad": phase38.JOINT_STEP_MAX_RAD,
            },
            "parameter_scan": False,
        },
        "provenance": {
            "phase30_cache_sha256": phase28.sha256(phase30.OUTPUT_CACHE),
            "phase37_warmstart_sha256": phase28.sha256(phase37.OUTPUT_CACHE),
            "official_scene_sha256": phase28.sha256(physics.DEFAULT_SCENE),
        },
        "iterations": iterations,
        "summary": {
            "qp_solved_iterations": len(iterations), "qp_failure": qp_failure,
            "active_switches_lr": switches.tolist(),
        },
        "final_metrics": final,
        "artifact": {"path": str(OUTPUT_NPZ), "sha256": phase28.sha256(OUTPUT_NPZ)},
        "decision": {
            "status": status, "local_window_feasible": feasible,
            "full_trajectory_true_silver": False, "physics_proposal_allowed": False,
            "conclusion": conclusion, "next_step": next_step,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
