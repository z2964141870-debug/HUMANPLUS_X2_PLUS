#!/usr/bin/env python3
"""Phase40 root-XY structural A/B on the frozen Phase39 25-frame QP.

The A arm is the already-recorded Phase39 fixed-root-XY result.  The B arm
adds only per-frame root XY correction.  Solver, outer iterations, trust,
contact constraints, source timing, intent and all other pose variables stay
frozen.  This is a local kinematic certificate, never a Silver/physics claim.
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
from scipy.optimize import Bounds


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29
import retarget.run_x2_wbt_time_dilation_phase30 as phase30
import retarget.run_x2_wbt_official_contact_repair_phase37 as phase37
import retarget.run_x2_wbt_hard_constraint_certificate_phase38 as phase38
import retarget.run_x2_wbt_sequential_qp_certificate_phase39 as phase39


ROOT_XY_MAX_M = 0.20
ROOT_HORIZONTAL_ACCEL_MAX_MPS2 = 4.0
POLYGON_SIDES = 32
POLYGON_SCALE = float(np.cos(np.pi / POLYGON_SIDES))
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_rootxy_sequential_qp_phase40.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_rootxy_sequential_qp_phase40.md"
OUTPUT_NPZ = REPO / "artifacts/official_x2/x2_wbt_phase40_rootxy_qp_window.npz"
PHASE39_JSON = phase39.OUTPUT_JSON


def polygon_directions() -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(POLYGON_SIDES) / POLYGON_SIDES
    return np.column_stack([np.cos(theta), np.sin(theta)])


class RootXYWindowProblem:
    """Phase38 window with root XYZ + the same lower/waist 15 DoF."""

    def __init__(
        self, model: mujoco.MjModel, source: dict[str, Any], warm: dict[str, Any],
        intent: dict[str, np.ndarray], lower_indices: np.ndarray,
    ) -> None:
        self.model = model
        self.source = source
        self.intent = {side: np.asarray(intent[side], dtype=bool) for side in phase29.SIDES}
        self.lower_indices = np.asarray(lower_indices, dtype=np.int64)
        self.frames = np.arange(phase38.WINDOW_START, phase38.WINDOW_END_EXCLUSIVE, dtype=np.int64)
        self.nframes = len(self.frames)
        self.width = 3 + len(self.lower_indices)
        self.nvar = self.nframes * self.width
        self.names = list(source["joint_names_mujoco"])
        self.qpos_addresses = [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.names
        ]
        lower_dofs = np.asarray([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.names[int(index)])]
            for index in self.lower_indices
        ], dtype=np.int64)
        self.variable_dofs = np.concatenate([np.asarray([0, 1, 2], dtype=np.int64), lower_dofs])
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in phase28.TARGET_BODIES
        ]
        self.floor, self.feet = phase38.phase7.active_sole_spheres(model)
        self.original_kin = phase28.target_kinematics(source, model)
        self.original_points = self.original_kin["target_points"][self.frames]
        self.original_distance = {
            side: np.asarray(self.original_kin["sole_distance"][side])[self.frames] for side in phase29.SIDES
        }
        self.original_q = np.asarray(source["dof"], dtype=np.float64)
        self.original_root = np.asarray(source["root_trans_offset"], dtype=np.float64)
        warm_root = np.asarray(warm["root_trans_offset"], dtype=np.float64)
        # Phase37 did not alter XY, so this is exactly the Phase39 warm start
        # with two zero columns inserted.
        root_seed = warm_root[self.frames] - self.original_root[self.frames]
        joint_seed = (
            np.asarray(warm["dof"], dtype=np.float64)[self.frames][:, self.lower_indices]
            - self.original_q[self.frames][:, self.lower_indices]
        )
        self.x0 = np.column_stack([root_seed, joint_seed]).reshape(-1)

    def cols(self, local_frame: int) -> np.ndarray:
        start = local_frame * self.width
        return np.arange(start, start + self.width, dtype=np.int64)

    def entry_from_x(self, x: np.ndarray) -> dict[str, Any]:
        root = self.original_root.copy()
        q = self.original_q.copy()
        correction = np.asarray(x, dtype=np.float64).reshape(self.nframes, self.width)
        root[self.frames] += correction[:, :3]
        q[np.ix_(self.frames, self.lower_indices)] += correction[:, 3:]
        return {**self.source, "root_trans_offset": root, "dof": q}

    def variable_bounds(self) -> Bounds:
        lower = np.full((self.nframes, self.width), -phase38.JOINT_CORRECTION_MAX_RAD)
        upper = np.full((self.nframes, self.width), phase38.JOINT_CORRECTION_MAX_RAD)
        lower[:, :2], upper[:, :2] = -ROOT_XY_MAX_M, ROOT_XY_MAX_M
        lower[:, 2], upper[:, 2] = -phase38.ROOT_Z_CORRECTION_MAX_M, phase38.ROOT_Z_CORRECTION_MAX_M
        for local, frame in enumerate(self.frames):
            for joint_local, entry_index in enumerate(self.lower_indices):
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, self.names[int(entry_index)]
                )
                if self.model.jnt_limited[joint_id]:
                    low, high = self.model.jnt_range[joint_id]
                    column = 3 + joint_local
                    lower[local, column] = max(lower[local, column], low - self.original_q[frame, entry_index])
                    upper[local, column] = min(upper[local, column], high - self.original_q[frame, entry_index])
        return Bounds(lower.reshape(-1), upper.reshape(-1))


def objective_qp(problem: RootXYWindowProblem, xk: np.ndarray, states: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    weights = phase38.OBJECTIVE_WEIGHTS
    P = np.eye(problem.nvar, dtype=np.float64) * phase39.QP_REGULARIZATION
    q = np.zeros(problem.nvar, dtype=np.float64)
    for local in range(problem.nframes):
        columns = problem.cols(local)
        for column in columns[:3]:
            P[column, column] += 2.0 * weights["root_z_correction"]
        for column in columns[3:]:
            P[column, column] += 2.0 * weights["joint_correction"]
    for local in range(1, problem.nframes):
        for component in range(problem.width):
            phase39.add_quadratic_pair(
                P, int(problem.cols(local)[component]), int(problem.cols(local - 1)[component]),
                weights["correction_velocity"],
            )
    for local in range(1, problem.nframes - 1):
        for component in range(problem.width):
            phase39.add_quadratic_second(
                P, int(problem.cols(local - 1)[component]), int(problem.cols(local)[component]),
                int(problem.cols(local + 1)[component]), weights["correction_acceleration"],
            )
    for local, state in enumerate(states):
        columns = problem.cols(local)
        xlocal = xk[columns]
        for point, jacobian, target in zip(state["points"], state["point_jac"], problem.original_points[local]):
            offset = point - jacobian @ xlocal - target
            P[np.ix_(columns, columns)] += 2.0 * weights["source_fit_keypoint"] * (jacobian.T @ jacobian)
            q[columns] += 2.0 * weights["source_fit_keypoint"] * (jacobian.T @ offset)
    return 0.5 * (P + P.T), q


def linear_constraints(
    problem: RootXYWindowProblem, xk: np.ndarray, states: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    counts = {
        "sphere_nonpenetration": 0, "stance_active_band": 0, "swing_clearance": 0,
        "stance_speed": 0, "stance_excursion": 0, "qstep": 0,
        "root_xy_polygon": 0, "root_horizontal_acceleration_polygon": 0,
    }

    def upper(row: np.ndarray, value: float, category: str) -> None:
        rows.append(row); bounds.append(float(value)); counts[category] += 1

    def interval(row: np.ndarray, constant: float, low: float, high: float, category: str) -> None:
        upper(row, high - constant, category)
        upper(-row, -low + constant, category)

    # Identical Phase39 collision/contact constraints.
    for local, frame in enumerate(problem.frames):
        columns = problem.cols(local)
        for side in phase29.SIDES:
            state = states[local]
            active = state["active"][side]
            for sphere in range(12):
                row = np.zeros(problem.nvar)
                row[columns] = state["sphere_jac"][side][sphere]
                distance = state["sphere_distance"][side][sphere]
                constant = distance - row @ xk
                desired = phase38.NONPENETRATION_LOWER_M
                category = "sphere_nonpenetration"
                if not problem.intent[side][frame]:
                    desired = max(desired, max(0.0, float(problem.original_distance[side][local])))
                    category = "swing_clearance"
                upper(-row, -desired + constant, category)
            if problem.intent[side][frame]:
                row = np.zeros(problem.nvar)
                row[columns] = state["sphere_jac"][side][active]
                distance = state["sphere_distance"][side][active]
                constant = distance - row @ xk
                interval(row, constant, 0.0, phase38.STANCE_DISTANCE_UPPER_M, "stance_active_band")

    # Same Phase39 tangent constraints; exact norms are rechecked each outer iteration.
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

    # Same exact affine lower-body q-step constraints, with the joint offset changed 1 -> 3.
    for frame in range(phase38.WINDOW_START + 1, phase38.WINDOW_END_EXCLUSIVE + 1):
        for joint_local, entry_index in enumerate(problem.lower_indices):
            row = np.zeros(problem.nvar)
            if frame < phase38.WINDOW_END_EXCLUSIVE:
                row[problem.cols(frame - phase38.WINDOW_START)[3 + joint_local]] = 1.0
            row[problem.cols(frame - 1 - phase38.WINDOW_START)[3 + joint_local]] -= 1.0
            base = problem.original_q[frame, entry_index] - problem.original_q[frame - 1, entry_index]
            interval(row, base, -phase38.JOINT_STEP_MAX_RAD, phase38.JOINT_STEP_MAX_RAD, "qstep")

    # New root-XY freedom remains inside the frozen Phase29 0.20m Euclidean bound.
    directions = polygon_directions()
    for local in range(problem.nframes):
        for direction in directions:
            row = np.zeros(problem.nvar)
            row[problem.cols(local)[:2]] = direction
            upper(row, ROOT_XY_MAX_M * POLYGON_SCALE, "root_xy_polygon")

    # Existing horizontal root-acceleration <=4m/s^2 gate, conservatively
    # enforced for every evaluable local center by an inscribed 32-gon.
    fps2 = float(problem.source["fps"]) ** 2
    root = problem.original_root
    for frame in range(phase38.WINDOW_START + 1, phase38.WINDOW_END_EXCLUSIVE):
        base = (root[frame - 1, :2] - 2.0 * root[frame, :2] + root[frame + 1, :2]) * fps2
        for direction in directions:
            row = np.zeros(problem.nvar)
            local = frame - phase38.WINDOW_START
            row[problem.cols(local - 1)[:2]] += direction * fps2
            row[problem.cols(local)[:2]] -= 2.0 * direction * fps2
            if frame + 1 < phase38.WINDOW_END_EXCLUSIVE:
                row[problem.cols(local + 1)[:2]] += direction * fps2
            upper(
                row, ROOT_HORIZONTAL_ACCEL_MAX_MPS2 * POLYGON_SCALE - float(direction @ base),
                "root_horizontal_acceleration_polygon",
            )
    return np.asarray(rows), np.asarray(bounds), counts


def exact_metrics(problem: RootXYWindowProblem, x: np.ndarray) -> dict[str, Any]:
    entry = problem.entry_from_x(x)
    kin = phase28.target_kinematics(entry, problem.model)
    frames = problem.frames
    distances = {side: np.asarray(kin["sole_distance"][side])[frames] for side in phase29.SIDES}
    stance = np.concatenate([
        np.abs(distances[side][problem.intent[side][frames]]) for side in phase29.SIDES
    ])
    # Phase28's compact FK contract deliberately exports body points and sole
    # distances only.  Reuse the exact same Phase39 static Jacobian pass for
    # ankle XY; this performs FK only (zero integration steps).
    states = phase39.linearization(problem, np.asarray(x, dtype=np.float64))
    speeds, excursions = [], []
    fps = float(problem.source["fps"])
    for side in phase29.SIDES:
        mask = problem.intent[side][frames]
        local_xy = np.asarray([state["foot_xy"][side] for state in states])
        for local in range(1, problem.nframes):
            if mask[local - 1] and mask[local]:
                speeds.append(float(np.linalg.norm(local_xy[local] - local_xy[local - 1]) * fps))
        for start, end in phase28.contiguous_true(mask):
            for local in range(start + 1, end):
                excursions.append(float(np.linalg.norm(local_xy[local] - local_xy[start])))
    q = np.asarray(entry["dof"])
    qstep = float(np.max(np.abs(np.diff(
        q[phase38.WINDOW_START:phase38.WINDOW_END_EXCLUSIVE + 1, problem.lower_indices], axis=0
    ))))
    correction = np.asarray(x).reshape(problem.nframes, problem.width)
    root_xy_norm = np.linalg.norm(correction[:, :2], axis=1)
    # Include fixed frame25 as Phase39 q-step does; frame-1 is unavailable, so centers 1..24.
    root_window = np.asarray(entry["root_trans_offset"])[phase38.WINDOW_START:phase38.WINDOW_END_EXCLUSIVE + 1, :2]
    root_acc = np.linalg.norm(np.diff(root_window, n=2, axis=0), axis=1) * fps * fps
    minimum = float(min(np.min(values) for values in distances.values()))
    max_distance = float(np.max(stance))
    speed_max = max(speeds, default=0.0)
    excursion_max = max(excursions, default=0.0)
    feasible = bool(
        max_distance <= phase38.STANCE_DISTANCE_UPPER_M + 1.0e-7
        and minimum >= phase38.NONPENETRATION_LOWER_M - 1.0e-7
        and speed_max <= phase38.STANCE_SPEED_MAX_MPS + 1.0e-7
        and excursion_max <= phase38.STANCE_EXCURSION_MAX_M + 1.0e-7
        and qstep <= phase38.JOINT_STEP_MAX_RAD + 1.0e-7
        and float(np.max(root_xy_norm)) <= ROOT_XY_MAX_M + 1.0e-7
        and float(np.max(root_acc)) <= ROOT_HORIZONTAL_ACCEL_MAX_MPS2 + 1.0e-7
    )
    violations = {
        "stance_distance": max(0.0, max_distance - phase38.STANCE_DISTANCE_UPPER_M),
        "nonpenetration": max(0.0, phase38.NONPENETRATION_LOWER_M - minimum),
        "stance_speed": max(0.0, speed_max - phase38.STANCE_SPEED_MAX_MPS),
        "stance_excursion": max(0.0, excursion_max - phase38.STANCE_EXCURSION_MAX_M),
        "qstep": max(0.0, qstep - phase38.JOINT_STEP_MAX_RAD),
        "root_xy": max(0.0, float(np.max(root_xy_norm)) - ROOT_XY_MAX_M),
        "root_horizontal_acceleration": max(0.0, float(np.max(root_acc)) - ROOT_HORIZONTAL_ACCEL_MAX_MPS2),
    }
    return {
        "max_exact_gate_violation": float(max(violations.values())),
        "exact_gate_violations": violations,
        "stance_abs_distance_p95_max_m": [float(np.percentile(stance, 95)), max_distance],
        "minimum_all_sphere_distance_m": minimum,
        "stance_speed_max_mps": speed_max,
        "stance_excursion_max_m": excursion_max,
        "qstep_max_rad": qstep,
        "root_xy_correction_max_m": float(np.max(root_xy_norm)),
        "root_horizontal_acceleration_p95_max_mps2": [float(np.percentile(root_acc, 95)), float(np.max(root_acc))],
        "feasible": feasible,
    }


def render(report: dict[str, Any]) -> str:
    a, b = report["ab_result"]["A_fixed_root_xy_phase39"], report["ab_result"]["B_free_root_xy_phase40"]
    return f"""# X2 WBT Phase40：root-XY structural A/B local certificate

## 假设

Phase39 的25帧局部证书失败，可能主要由冻结 root XY 造成；只释放 root XY 后，同一接触/连续性硬门可能变得可行。

## 干预

- A：直接读取已冻结 Phase39，不重跑。
- B：唯一新增每帧 root XY；相对source欧氏硬界0.20m，水平root acceleration硬门4.0m/s²。
- DAQP、20轮、trust 5mm/0.05rad、active-sphere、contact intent、时间、root orientation、upper/head全部冻结。

## 对照与结果

- A feasible：`{a['feasible']}`；stance speed={a['stance_speed_max_mps']:.4f}m/s；stance distance max={a['stance_distance_max_m']:.6f}m。
- B feasible：`{b['feasible']}`；stance speed={b['stance_speed_max_mps']:.4f}m/s；stance distance max={b['stance_distance_max_m']:.6f}m。
- B rootXY correction max={b['root_xy_correction_max_m']:.4f}m；root horizontal accel p95/max={b['root_horizontal_acceleration_p95_max_mps2']}m/s²。
- B QP solved={report['summary']['qp_solved_iterations']}；failure=`{report['summary']['qp_failure']}`；active switches L/R={report['summary']['active_switches_lr']}。
- 执行记录：首次调用在首个QP后因metric读取了不存在的`foot_xy`导出键而中止、未形成结果；仅修正为复用Phase39静态FK后，以完全相同配置完成本次20轮。未据中止结果调参。

## 结论

**{report['decision']['status']}**

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    if not PHASE39_JSON.exists():
        raise FileNotFoundError("Phase39 frozen A-arm report is required; Phase40 must not rerun A")
    phase39_report = json.loads(PHASE39_JSON.read_text())
    panel = {row["id"]: row for row in json.loads(phase28.PANEL.read_text())["motions"]}
    row = panel[phase38.MOTION_ID]
    source = joblib.load(phase30.OUTPUT_CACHE)[phase38.MOTION_ID]
    warm = joblib.load(phase37.OUTPUT_CACHE)[phase38.MOTION_ID]
    source29 = joblib.load(phase29.OUTPUT_CACHE)[phase38.MOTION_ID]
    physics = phase38.phase3.load_module(phase38.phase3.PHYSICS_SCRIPT, "x2_phase40_static_fk_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    source_contract = phase29.source_contract(row, source29, model)
    intent = {
        side: phase30.nearest_phase_resample(source_contract["contact"][side], len(source["dof"]))
        for side in phase29.SIDES
    }
    lower = phase29.lower_body_indices(list(source["joint_names_mujoco"]))
    problem = RootXYWindowProblem(model, source, warm, intent, lower)
    x = problem.x0.copy()
    global_bounds = problem.variable_bounds()
    iterations, previous_active = [], None
    switches = np.zeros(2, dtype=np.int64)
    qp_failure = None
    for outer in range(phase39.OUTER_ITERATIONS):
        states = phase39.linearization(problem, x)
        active = np.asarray([[state["active"][side] for side in phase29.SIDES] for state in states], dtype=np.int64)
        switch_lr = np.zeros(2, dtype=np.int64) if previous_active is None else np.count_nonzero(active != previous_active, axis=0)
        switches += switch_lr
        previous_active = active.copy()
        P, q = objective_qp(problem, x, states)
        G, h, counts = linear_constraints(problem, x, states)
        trust = np.tile(
            np.r_[np.full(3, phase39.ROOT_Z_TRUST_M), np.full(problem.width - 3, phase39.JOINT_TRUST_RAD)],
            problem.nframes,
        )
        lb = np.maximum(global_bounds.lb, x - trust)
        ub = np.minimum(global_bounds.ub, x + trust)
        solution = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=phase39.SOLVER, initvals=x, verbose=False)
        if solution is None:
            qp_failure = f"{phase39.SOLVER} returned no solution at outer iteration {outer + 1}"
            break
        x = np.asarray(solution, dtype=np.float64)
        metrics = exact_metrics(problem, x)
        correction = x.reshape(problem.nframes, problem.width)
        iterations.append({
            "outer_iteration": outer + 1,
            "linear_constraint_counts": counts,
            "active_sphere_indices_lr": active.tolist(),
            "active_switch_count_lr": switch_lr.tolist(),
            "exact_metrics": metrics,
            "root_xyz_correction_max_abs_m": np.max(np.abs(correction[:, :3]), axis=0).tolist(),
            "joint_correction_max_abs_rad": float(np.max(np.abs(correction[:, 3:]))),
        })
        print(
            f"[phase40] outer={outer + 1}/{phase39.OUTER_ITERATIONS} feasible={metrics['feasible']} "
            f"violation={metrics['max_exact_gate_violation']:.6g} "
            f"speed={metrics['stance_speed_max_mps']:.6g}", flush=True,
        )
        if metrics["feasible"]:
            break

    final = exact_metrics(problem, x)
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ, correction=x.reshape(problem.nframes, problem.width), frames=problem.frames,
        active_sphere_last=previous_active if previous_active is not None else np.zeros((0, 2), dtype=np.int64),
    )
    a_metrics = phase39_report["final_metrics"]
    a = {
        "source": str(PHASE39_JSON), "source_sha256": phase28.sha256(PHASE39_JSON),
        "rerun": False, "feasible": bool(a_metrics["feasible"]),
        "stance_speed_max_mps": float(a_metrics["stance_speed_max_mps"]),
        "stance_distance_max_m": float(a_metrics["stance_abs_distance_p95_max_m"][1]),
    }
    b = {
        "feasible": bool(final["feasible"]), "stance_speed_max_mps": final["stance_speed_max_mps"],
        "stance_distance_max_m": final["stance_abs_distance_p95_max_m"][1],
        "root_xy_correction_max_m": final["root_xy_correction_max_m"],
        "root_horizontal_acceleration_p95_max_mps2": final["root_horizontal_acceleration_p95_max_mps2"],
    }
    if final["feasible"]:
        status = "PHASE40_ROOT_XY_LOCAL_CERTIFIED"
        conclusion = "同一局部窗仅释放root XY后取得硬门证书；这支持fixed-rootXY是Phase39局部不可行的重要原因，但不证明完整轨迹Silver或物理可执行。"
        next_step = "按预注册门停止；不自动扩完整轨迹或physics，由主线另行裁决。"
    else:
        status = "PHASE40_ROOT_XY_LOCAL_CERTIFICATE_FAILED"
        conclusion = "只释放root XY仍未取得同一25帧局部硬门证书；fixed-rootXY不是单独足以解释Phase39失败的主因。这是否定当前生成器/solver结构，不是X2不可能。"
        next_step = "按预注册门停止；不缩放rootXY、不调trust/门、不扩完整轨迹或physics。"
    report = {
        "schema_version": "x2_wbt_rootxy_sequential_qp_phase40_v1",
        "truth_boundary": {
            "local_25_frame_certificate_only": True, "phase39_A_rerun": False,
            "mujoco_integration_steps": 0, "policy_optimizer_ppo_training": False,
            "held_out_read": False, "configuration_count_B": 1,
            "aborted_invocation_before_result": "first invocation stopped after QP1 because metric adapter requested absent Phase28 foot_xy export; no report/result produced and no parameter changed",
        },
        "pre_registration": {
            "single_structural_variable": "per-frame root XY correction",
            "root_xy_relative_source_max_m": ROOT_XY_MAX_M,
            "root_horizontal_acceleration_max_mps2": ROOT_HORIZONTAL_ACCEL_MAX_MPS2,
            "hard_norm_implementation": f"{POLYGON_SIDES}-direction inscribed polygon; scale=cos(pi/{POLYGON_SIDES})",
            "solver": phase39.SOLVER, "outer_iterations_max": phase39.OUTER_ITERATIONS,
            "trust_region_root_xyz_m": phase39.ROOT_Z_TRUST_M,
            "trust_region_joint_rad": phase39.JOINT_TRUST_RAD,
            "window_frames": [phase38.WINDOW_START, phase38.WINDOW_END_EXCLUSIVE],
            "frozen": ["root orientation", "upper/head", "source timing", "contact intent", "Phase39 active-sphere rule", "all Phase39 gates"],
            "parameter_scan": False,
        },
        "provenance": {
            "phase39_A_sha256": phase28.sha256(PHASE39_JSON),
            "phase30_cache_sha256": phase28.sha256(phase30.OUTPUT_CACHE),
            "phase37_warmstart_sha256": phase28.sha256(phase37.OUTPUT_CACHE),
            "official_scene_sha256": phase28.sha256(physics.DEFAULT_SCENE),
        },
        "iterations": iterations,
        "summary": {"qp_solved_iterations": len(iterations), "qp_failure": qp_failure, "active_switches_lr": switches.tolist()},
        "final_metrics": final,
        "ab_result": {"A_fixed_root_xy_phase39": a, "B_free_root_xy_phase40": b},
        "artifact": {"path": str(OUTPUT_NPZ), "sha256": phase28.sha256(OUTPUT_NPZ)},
        "decision": {
            "status": status, "local_window_feasible": bool(final["feasible"]),
            "fixed_root_xy_supported_as_major_local_blocker": bool(final["feasible"] and not a["feasible"]),
            "full_trajectory_true_silver": False, "physics_proposal_allowed": False,
            "conclusion": conclusion, "next_step": next_step,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
