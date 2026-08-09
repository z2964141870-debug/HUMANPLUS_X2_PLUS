#!/usr/bin/env python3
"""Phase42 contact-first centroidal teacher, preregistered A layer.

Layer A jointly solves COM and model contact forces on the frozen 25-frame
DS->R-SS->DS schedule.  Only if every hard gate passes may layer B whole-body
IK be attempted.  All quantities are official-MuJoCo model estimates, not
hardware GRF/COP truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from qpsolvers import solve_qp
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
import retarget.run_x2_wbt_centroidal_force_feasibility_phase41 as phase41


OUTER_ITERATIONS = 10
SOLVER = "daqp"
QP_REGULARIZATION = 1.0e-8
COM_TRUST_M = 0.03
COM_XY_SOURCE_MAX_M = 0.20
COM_Z_SOURCE_MAX_M = 0.08
COM_HORIZONTAL_SPEED_MAX_MPS = 1.5
COM_VERTICAL_SPEED_MAX_MPS = 0.75
COM_HORIZONTAL_ACCEL_MAX_MPS2 = 4.0
COM_VERTICAL_ACCEL_MAX_MPS2 = 4.0
POLYGON_SIDES = 32
POLYGON_SCALE = float(np.cos(np.pi / POLYGON_SIDES))
MOMENT_TOL_NM = 1.0e-3
DYNAMICS_TOL_N = 1.0e-5
FRICTION_TOL_N = 1.0e-7
OBJECTIVE_WEIGHTS = {
    "source_com": 10.0,
    "com_velocity": 1.0,
    "com_acceleration": 2.0,
    "contact_force": 1.0e-4,
}
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_contact_first_centroidal_teacher_phase42.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_contact_first_centroidal_teacher_phase42.md"
OUTPUT_NPZ = REPO / "artifacts/official_x2/x2_wbt_phase42_contact_first_centroidal_A.npz"


def directions() -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(POLYGON_SIDES) / POLYGON_SIDES
    return np.column_stack([np.cos(theta), np.sin(theta)])


def add_pair(P: np.ndarray, a: int, b: int, weight: float) -> None:
    P[a, a] += 2.0 * weight
    P[b, b] += 2.0 * weight
    P[a, b] -= 2.0 * weight
    P[b, a] -= 2.0 * weight


def add_second(P: np.ndarray, a: int, b: int, c: int, weight: float) -> None:
    values = (1.0, -2.0, 1.0)
    for i, vi in zip((a, b, c), values):
        for j, vj in zip((a, b, c), values):
            P[i, j] += 2.0 * weight * vi * vj


def phase_anchored_ground_points(
    model: mujoco.MjModel, qpos: np.ndarray, intent: dict[str, np.ndarray],
    floor: int, feet: dict[str, list[int]], frames: np.ndarray,
) -> dict[int, list[tuple[str, int, np.ndarray]]]:
    """Freeze each stance phase's sole XY at its first frame and set z=0."""
    data = mujoco.MjData(model)
    result: dict[int, list[tuple[str, int, np.ndarray]]] = {int(frame): [] for frame in frames}
    for side in phase29.SIDES:
        mask = np.asarray(intent[side][frames], dtype=bool)
        for start, end in phase28.contiguous_true(mask):
            anchor_frame = int(frames[start])
            data.qpos[:] = qpos[anchor_frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            anchored = []
            for sphere, geom in enumerate(feet[side]):
                point = data.geom_xpos[geom].copy()
                point[2] = 0.0
                anchored.append((side, sphere, point))
            for local in range(start, end):
                result[int(frames[local])].extend(anchored)
    return result


class Layout:
    def __init__(self, frames: np.ndarray, contacts: dict[int, list[tuple[str, int, np.ndarray]]]) -> None:
        self.frames = np.asarray(frames, dtype=np.int64)
        self.nframes = len(frames)
        self.com_size = 3 * self.nframes
        self.force_slices: dict[int, slice] = {}
        cursor = self.com_size
        for frame in self.frames[1:-1]:
            count = len(contacts[int(frame)])
            self.force_slices[int(frame)] = slice(cursor, cursor + 3 * count)
            cursor += 3 * count
        self.nvar = cursor

    def com_cols(self, local: int) -> np.ndarray:
        return np.arange(3 * local, 3 * local + 3, dtype=np.int64)

    def force_cols(self, frame: int, point: int) -> np.ndarray:
        block = self.force_slices[frame]
        return np.arange(block.start + 3 * point, block.start + 3 * point + 3, dtype=np.int64)


def initial_x(
    layout: Layout, source_com: np.ndarray, contacts: dict[int, list[tuple[str, int, np.ndarray]]],
    mass: float, gravity: np.ndarray, fps: float,
) -> np.ndarray:
    x = np.zeros(layout.nvar, dtype=np.float64)
    x[:layout.com_size] = source_com.reshape(-1)
    accel = np.diff(source_com, n=2, axis=0) * fps * fps
    for local, frame in enumerate(layout.frames[1:-1], start=1):
        points = contacts[int(frame)]
        count = len(points)
        total = mass * (accel[local - 1] - gravity)
        force = np.tile(total / max(count, 1), (count, 1))
        force[:, 2] = max(total[2] / max(count, 1), 1.0e-6)
        x[layout.force_slices[int(frame)]] = force.reshape(-1)
    return x


def objective(layout: Layout, source_com: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.eye(layout.nvar, dtype=np.float64) * QP_REGULARIZATION
    q = np.zeros(layout.nvar, dtype=np.float64)
    w = OBJECTIVE_WEIGHTS
    for local in range(layout.nframes):
        cols = layout.com_cols(local)
        P[cols, cols] += 2.0 * w["source_com"]
        q[cols] -= 2.0 * w["source_com"] * source_com[local]
    for local in range(1, layout.nframes):
        for axis in range(3):
            add_pair(P, int(layout.com_cols(local)[axis]), int(layout.com_cols(local - 1)[axis]), w["com_velocity"])
    for local in range(1, layout.nframes - 1):
        for axis in range(3):
            add_second(
                P, int(layout.com_cols(local - 1)[axis]), int(layout.com_cols(local)[axis]),
                int(layout.com_cols(local + 1)[axis]), w["com_acceleration"],
            )
    for block in layout.force_slices.values():
        P[block, block] += np.eye(block.stop - block.start) * 2.0 * w["contact_force"]
    return 0.5 * (P + P.T), q


def constraints(
    layout: Layout, xk: np.ndarray, source_com: np.ndarray,
    contacts: dict[int, list[tuple[str, int, np.ndarray]]], mass: float,
    gravity: np.ndarray, fps: float, mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    eq_rows, eq_values, ineq_rows, ineq_values = [], [], [], []
    counts = {"linear_momentum": 0, "linearized_zero_moment": 0, "friction": 0,
              "com_xy": 0, "com_velocity": 0, "com_acceleration": 0}

    def equality(row: np.ndarray, value: np.ndarray | float, category: str) -> None:
        values = np.atleast_1d(value)
        rows = row if row.ndim == 2 else row[None, :]
        eq_rows.extend(rows); eq_values.extend(values); counts[category] += len(values)

    def upper(row: np.ndarray, value: float, category: str) -> None:
        ineq_rows.append(row); ineq_values.append(float(value)); counts[category] += 1

    dt2_inv = fps * fps
    # Fix only start/end COM position to source; interior path is contact-first.
    for local in (0, layout.nframes - 1):
        rows = np.zeros((3, layout.nvar))
        rows[:, layout.com_cols(local)] = np.eye(3)
        equality(rows, source_com[local], "linear_momentum")

    for local, frame in enumerate(layout.frames[1:-1], start=1):
        points = contacts[int(frame)]
        # m*cddot - sum(force) = m*gravity.
        rows = np.zeros((3, layout.nvar))
        rows[:, layout.com_cols(local - 1)] = np.eye(3) * mass * dt2_inv
        rows[:, layout.com_cols(local)] = np.eye(3) * -2.0 * mass * dt2_inv
        rows[:, layout.com_cols(local + 1)] = np.eye(3) * mass * dt2_inv
        for point_index in range(len(points)):
            rows[:, layout.force_cols(int(frame), point_index)] -= np.eye(3)
        equality(rows, mass * gravity, "linear_momentum")

        # Sequential linearization of sum((p-c) x f)=0.
        ccols = layout.com_cols(local)
        ck = xk[ccols]
        force_k = xk[layout.force_slices[int(frame)]].reshape(len(points), 3)
        moment_rows = np.zeros((3, layout.nvar))
        sum_force_k = np.sum(force_k, axis=0)
        moment_rows[:, ccols] = phase41.skew(sum_force_k)
        for point_index, (_, _, point) in enumerate(points):
            moment_rows[:, layout.force_cols(int(frame), point_index)] = phase41.skew(point - ck)
        equality(moment_rows, phase41.skew(sum_force_k) @ ck, "linearized_zero_moment")

        for point_index in range(len(points)):
            cols = layout.force_cols(int(frame), point_index)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    row = np.zeros(layout.nvar)
                    row[cols] = [sx, sy, -mu]
                    upper(row, 0.0, "friction")

    dirs = directions()
    for local in range(layout.nframes):
        cols = layout.com_cols(local)
        for direction in dirs:
            row = np.zeros(layout.nvar); row[cols[:2]] = direction
            upper(row, float(direction @ source_com[local, :2]) + COM_XY_SOURCE_MAX_M * POLYGON_SCALE, "com_xy")
            upper(-row, -float(direction @ source_com[local, :2]) + COM_XY_SOURCE_MAX_M * POLYGON_SCALE, "com_xy")
    # Component/polygon velocity and acceleration hard continuity bounds.
    for local in range(1, layout.nframes):
        for direction in dirs:
            row = np.zeros(layout.nvar)
            row[layout.com_cols(local)[:2]] = direction * fps
            row[layout.com_cols(local - 1)[:2]] -= direction * fps
            upper(row, COM_HORIZONTAL_SPEED_MAX_MPS * POLYGON_SCALE, "com_velocity")
        row = np.zeros(layout.nvar)
        row[layout.com_cols(local)[2]] = fps; row[layout.com_cols(local - 1)[2]] = -fps
        upper(row, COM_VERTICAL_SPEED_MAX_MPS, "com_velocity")
        upper(-row, COM_VERTICAL_SPEED_MAX_MPS, "com_velocity")
    for local in range(1, layout.nframes - 1):
        for direction in dirs:
            row = np.zeros(layout.nvar)
            row[layout.com_cols(local - 1)[:2]] = direction * dt2_inv
            row[layout.com_cols(local)[:2]] = -2.0 * direction * dt2_inv
            row[layout.com_cols(local + 1)[:2]] = direction * dt2_inv
            upper(row, COM_HORIZONTAL_ACCEL_MAX_MPS2 * POLYGON_SCALE, "com_acceleration")
        row = np.zeros(layout.nvar)
        row[layout.com_cols(local - 1)[2]] = dt2_inv
        row[layout.com_cols(local)[2]] = -2.0 * dt2_inv
        row[layout.com_cols(local + 1)[2]] = dt2_inv
        upper(row, COM_VERTICAL_ACCEL_MAX_MPS2, "com_acceleration")
        upper(-row, COM_VERTICAL_ACCEL_MAX_MPS2, "com_acceleration")
    return np.asarray(ineq_rows), np.asarray(ineq_values), np.asarray(eq_rows), np.asarray(eq_values), counts


def bounds(layout: Layout, source_com: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lb = np.full(layout.nvar, -np.inf); ub = np.full(layout.nvar, np.inf)
    for local in range(layout.nframes):
        cols = layout.com_cols(local)
        lb[cols[:2]] = source_com[local, :2] - COM_XY_SOURCE_MAX_M
        ub[cols[:2]] = source_com[local, :2] + COM_XY_SOURCE_MAX_M
        lb[cols[2]] = source_com[local, 2] - COM_Z_SOURCE_MAX_M
        ub[cols[2]] = source_com[local, 2] + COM_Z_SOURCE_MAX_M
    for block in layout.force_slices.values():
        for column in range(block.start, block.stop, 3):
            lb[column + 2] = 0.0
    return lb, ub


def exact_metrics(
    layout: Layout, x: np.ndarray, source_com: np.ndarray,
    contacts: dict[int, list[tuple[str, int, np.ndarray]]], mass: float,
    gravity: np.ndarray, fps: float, mu: float,
) -> dict[str, Any]:
    com = x[:layout.com_size].reshape(layout.nframes, 3)
    accel = np.diff(com, n=2, axis=0) * fps * fps
    velocity = np.diff(com, axis=0) * fps
    dyn_residual, moment_residual, friction_violation, min_normal = [], [], [], []
    cop_inside = True
    for local, frame in enumerate(layout.frames[1:-1], start=1):
        points = contacts[int(frame)]
        force = x[layout.force_slices[int(frame)]].reshape(len(points), 3)
        dyn_residual.append(mass * accel[local - 1] - np.sum(force, axis=0) - mass * gravity)
        moment_residual.append(np.sum([
            np.cross(point - com[local], force[index])
            for index, (_, _, point) in enumerate(points)
        ], axis=0))
        friction_violation.extend(np.abs(force[:, 0]) + np.abs(force[:, 1]) - mu * force[:, 2])
        min_normal.extend(force[:, 2])
        # COP is a nonnegative normal-force convex combination of the active
        # points by construction; no rectangle proxy is substituted.
        for side in phase29.SIDES:
            indices = [i for i, item in enumerate(points) if item[0] == side]
            if indices and np.sum(force[indices, 2]) < -FRICTION_TOL_N:
                cop_inside = False
    correction = com - source_com
    xy = np.linalg.norm(correction[:, :2], axis=1)
    speed_xy = np.linalg.norm(velocity[:, :2], axis=1)
    accel_xy = np.linalg.norm(accel[:, :2], axis=1)
    feasible = bool(
        np.max(np.abs(dyn_residual)) <= DYNAMICS_TOL_N
        and np.max(np.abs(moment_residual)) <= MOMENT_TOL_NM
        and max(0.0, float(np.max(friction_violation))) <= FRICTION_TOL_N
        and float(np.min(min_normal)) >= -FRICTION_TOL_N
        and float(np.max(xy)) <= COM_XY_SOURCE_MAX_M + 1e-7
        and float(np.max(np.abs(correction[:, 2]))) <= COM_Z_SOURCE_MAX_M + 1e-7
        and float(np.max(speed_xy)) <= COM_HORIZONTAL_SPEED_MAX_MPS + 1e-7
        and float(np.max(np.abs(velocity[:, 2]))) <= COM_VERTICAL_SPEED_MAX_MPS + 1e-7
        and float(np.max(accel_xy)) <= COM_HORIZONTAL_ACCEL_MAX_MPS2 + 1e-7
        and float(np.max(np.abs(accel[:, 2]))) <= COM_VERTICAL_ACCEL_MAX_MPS2 + 1e-7
        and cop_inside
    )
    return {
        "dynamics_residual_max_n": float(np.max(np.abs(dyn_residual))),
        "zero_moment_residual_max_nm": float(np.max(np.abs(moment_residual))),
        "friction_violation_max_n": max(0.0, float(np.max(friction_violation))),
        "minimum_normal_force_n": float(np.min(min_normal)),
        "cop_inside_active_sphere_convex_hull": cop_inside,
        "com_xy_source_correction_max_m": float(np.max(xy)),
        "com_z_source_correction_max_m": float(np.max(np.abs(correction[:, 2]))),
        "com_horizontal_speed_max_mps": float(np.max(speed_xy)),
        "com_vertical_speed_max_mps": float(np.max(np.abs(velocity[:, 2]))),
        "com_horizontal_accel_max_mps2": float(np.max(accel_xy)),
        "com_vertical_accel_max_mps2": float(np.max(np.abs(accel[:, 2]))),
        "layer_A_feasible": feasible,
    }


def render(report: dict[str, Any]) -> str:
    result = report["layer_A_result"]
    return f"""# X2 WBT Phase42：contact-first centroidal teacher

## 假设

不再修 Phase40 joint path；先从水平foot placement与DS→R-SS→DS时序构造地面接触，再求动力学一致COM/GRF，能够隔离原GMR路径的接触几何错误。

## 干预

- A层：25帧COM xyz + official sole sphere GRF，固定DAQP sequential-convex 10轮。
- stance phase的12个sole点使用该phase首帧水平位置并投影z=0；非接触脚没有force变量。
- 硬约束：线动量、线性化零净矩、fz>=0、μ=1摩擦锥、COP凸包、COM边界/速度/加速度。
- 所有量均为official MuJoCo模型估计，不是实机GRF/COP真值。

## 结果

- QP solved：{report['summary']['qp_solved_iterations']}/10；failure=`{report['summary']['qp_failure']}`。
- dynamics residual：{result['dynamics_residual_max_n']:.6g}N；zero-moment residual：{result['zero_moment_residual_max_nm']:.6g}N·m。
- friction violation：{result['friction_violation_max_n']:.6g}N；min normal：{result['minimum_normal_force_n']:.6g}N。
- COM correction XY/Z：{result['com_xy_source_correction_max_m']:.4f}/{result['com_z_source_correction_max_m']:.4f}m。
- COM speed horizontal/vertical：{result['com_horizontal_speed_max_mps']:.4f}/{result['com_vertical_speed_max_mps']:.4f}m/s。
- COM accel horizontal/vertical：{result['com_horizontal_accel_max_mps2']:.4f}/{result['com_vertical_accel_max_mps2']:.4f}m/s²。

## 裁决

**{report['decision']['status']}**

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
    artifact40 = np.load(phase40.OUTPUT_NPZ)
    physics = phase38.phase3.load_module(phase38.phase3.PHYSICS_SCRIPT, "x2_phase42_centroidal_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    source_contract = phase29.source_contract(row, source29, model)
    intent = {
        side: phase30.nearest_phase_resample(source_contract["contact"][side], len(source["dof"]))
        for side in phase29.SIDES
    }
    lower = phase29.lower_body_indices(list(source["joint_names_mujoco"]))
    problem40 = phase40.RootXYWindowProblem(model, source, warm, intent, lower)
    entry40 = problem40.entry_from_x(np.asarray(artifact40["correction"]).reshape(-1))
    qpos = phase41.build_qpos(model, entry40)
    qvel = phase41.differentiate_qpos(model, qpos, float(entry40["fps"]))
    source_com_all, _ = phase41.model_centroidal_series(model, qpos, qvel)
    frames = problem40.frames
    source_com = source_com_all[frames]
    floor, feet = phase38.phase7.active_sole_spheres(model)
    contacts = phase_anchored_ground_points(model, qpos, intent, floor, feet, frames)
    layout = Layout(frames, contacts)
    mass = float(np.sum(model.body_mass)); gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    mu = float(min(model.geom_friction[floor, 0], *[
        model.geom_friction[geom, 0] for geoms in feet.values() for geom in geoms
    ]))
    fps = float(entry40["fps"])
    x = initial_x(layout, source_com, contacts, mass, gravity, fps)
    P, q = objective(layout, source_com)
    global_lb, global_ub = bounds(layout, source_com)
    iterations, failure, linear_feasibility_audit = [], None, None
    for outer in range(OUTER_ITERATIONS):
        G, h, A, b, counts = constraints(layout, x, source_com, contacts, mass, gravity, fps, mu)
        lb, ub = global_lb.copy(), global_ub.copy()
        lb[:layout.com_size] = np.maximum(lb[:layout.com_size], x[:layout.com_size] - COM_TRUST_M)
        ub[:layout.com_size] = np.minimum(ub[:layout.com_size], x[:layout.com_size] + COM_TRUST_M)
        solution = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver=SOLVER, initvals=x, verbose=False)
        if solution is None:
            failure = f"{SOLVER} returned no solution at outer iteration {outer + 1}"
            # Same rows/bounds, zero LP objective: diagnosis only, not a second
            # teacher configuration and not a fallback solution.
            audit = linprog(
                np.zeros(layout.nvar), A_ub=G, b_ub=h, A_eq=A, b_eq=b,
                bounds=list(zip(lb, ub)), method="highs",
            )
            linear_feasibility_audit = {
                "same_constraints_success": bool(audit.success),
                "status": int(audit.status), "message": str(audit.message),
                "nvar": layout.nvar, "neq": len(b), "nineq": len(h),
                "constraint_counts": counts,
            }
            break
        x = np.asarray(solution, dtype=np.float64)
        metrics = exact_metrics(layout, x, source_com, contacts, mass, gravity, fps, mu)
        iterations.append({"outer_iteration": outer + 1, "constraint_counts": counts, "exact_metrics": metrics})
        print(
            f"[phase42:A] outer={outer + 1}/{OUTER_ITERATIONS} feasible={metrics['layer_A_feasible']} "
            f"moment={metrics['zero_moment_residual_max_nm']:.6g}", flush=True,
        )
        if metrics["layer_A_feasible"]:
            break
    metrics = exact_metrics(layout, x, source_com, contacts, mass, gravity, fps, mu)
    layer_a = bool(metrics["layer_A_feasible"])
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ, frames=frames, com=x[:layout.com_size].reshape(layout.nframes, 3),
        source_com=source_com, solution=x,
        force_frame=np.concatenate([[frame] * len(contacts[int(frame)]) for frame in frames[1:-1]]),
    )
    if layer_a:
        status = "PHASE42_LAYER_A_CENTROIDAL_FEASIBLE_B_PENDING"
        conclusion = "contact-first COM/GRF A层通过全部硬门；按任务合同下一步必须进入一次冻结whole-body IK B层，当前不得宣称teacher完成。"
        next_step = "同阶段实现B层whole-body IK并用Phase41 oracle复核；不改A层参数。"
    else:
        status = "PHASE42_LAYER_A_CENTROIDAL_INFEASIBLE"
        conclusion = "固定contact-first centroidal A层未取得硬可行解；按门停止，不用soft reward或调整边界/权重绕过，B/C均不运行。"
        next_step = "停止Phase42；报告solver/硬门失败，不进入IK、完整轨迹、physics或PPO。"
    report = {
        "schema_version": "x2_wbt_contact_first_centroidal_teacher_phase42_v1",
        "truth_boundary": {
            "official_model_estimates_not_hardware_truth": True,
            "held_out_read": False, "mujoco_integration_steps": 0,
            "policy_physics_or_ppo": False, "configuration_count": 1,
        },
        "pre_registration": {
            "window_frames": [phase38.WINDOW_START, phase38.WINDOW_END_EXCLUSIVE],
            "contact_schedule": "frozen Phase40 DS->R-SS->DS source intent",
            "contact_point_rule": "all 12 official sole sphere XY frozen at first frame of each contiguous stance phase; z projected to official ground",
            "solver": SOLVER, "outer_iterations_max": OUTER_ITERATIONS, "com_trust_m": COM_TRUST_M,
            "bounds": {
                "com_xy_source_max_m": COM_XY_SOURCE_MAX_M, "com_z_source_max_m": COM_Z_SOURCE_MAX_M,
                "horizontal_speed_mps": COM_HORIZONTAL_SPEED_MAX_MPS, "vertical_speed_mps": COM_VERTICAL_SPEED_MAX_MPS,
                "horizontal_accel_mps2": COM_HORIZONTAL_ACCEL_MAX_MPS2, "vertical_accel_mps2": COM_VERTICAL_ACCEL_MAX_MPS2,
            },
            "objective_weights": OBJECTIVE_WEIGHTS, "friction_mu": mu,
            "friction_cone": "inscribed L1 pyramid |fx|+|fy|<=mu*fz",
            "zero_net_moment_target": True, "parameter_scan": False,
            "layer_B_unlock": "layer A exact hard gates all pass",
        },
        "provenance": {
            "phase40_artifact_sha256": phase28.sha256(phase40.OUTPUT_NPZ),
            "official_scene_sha256": phase28.sha256(physics.DEFAULT_SCENE),
        },
        "layer_A_iterations": iterations,
        "layer_A_result": metrics,
        "summary": {
            "qp_solved_iterations": len(iterations), "qp_failure": failure,
            "same_constraint_highs_feasibility_audit": linear_feasibility_audit,
        },
        "artifact": {"path": str(OUTPUT_NPZ), "sha256": phase28.sha256(OUTPUT_NPZ)},
        "decision": {
            "status": status, "layer_A_feasible": layer_a,
            "layer_B_whole_body_ik_run": False, "layer_C_phase41_oracle_run": False,
            "full_trajectory_jointly_feasible": False, "physics_or_ppo_allowed": False,
            "conclusion": conclusion, "next_step": next_step,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
