#!/usr/bin/env python3
"""Phase38 one-shot nonlinear hard-constraint feasibility certificate.

The fixed window is selected from source intent only: the first complete
DS->SS->DS cycle of PHUMA-LUNGE-R (frames [0, 25), 30 Hz).  SLSQP receives
analytic Jacobians for official sole signed-distance and stance-foot XY
constraints.  No weights/thresholds/window are selected from the result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29
import retarget.run_x2_wbt_time_dilation_phase30 as phase30
import retarget.run_x2_wbt_official_contact_repair_phase37 as phase37


MOTION_ID = "PHUMA-LUNGE-R-001"
WINDOW_START = 0
WINDOW_END_EXCLUSIVE = 25
MAX_ITERATIONS = 100
FTOL = 1.0e-9
STANCE_DISTANCE_UPPER_M = 5.0e-4
NONPENETRATION_LOWER_M = -1.0e-5
STANCE_SPEED_MAX_MPS = 0.10
STANCE_EXCURSION_MAX_M = 0.03
JOINT_STEP_MAX_RAD = 0.15
ROOT_Z_CORRECTION_MAX_M = 0.08
JOINT_CORRECTION_MAX_RAD = 0.45
OBJECTIVE_WEIGHTS = {
    "source_fit_keypoint": 4.0,
    "root_z_correction": 1.5,
    "joint_correction": 1.0,
    "correction_velocity": 10.0,
    "correction_acceleration": 18.0,
}

OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_hard_constraint_certificate_phase38.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_hard_constraint_certificate_phase38.md"
OUTPUT_NPZ = REPO / "artifacts/official_x2/x2_wbt_phase38_hard_constraint_window.npz"


def solver_audit() -> dict[str, Any]:
    result = {
        "scipy_slsqp": {"available": True, "nonlinear_constraints": True, "analytic_jacobian": True, "sparse_native": False},
        "scipy_trust_constr": {"available": True, "nonlinear_constraints": True, "analytic_jacobian": True, "sparse_native": True},
    }
    try:
        import qpsolvers
        result["qpsolvers"] = {
            "available": True, "backends": list(qpsolvers.available_solvers),
            "limitation": "linear/quadratic subproblem only; official signed-distance is nonlinear",
        }
    except ImportError:
        result["qpsolvers"] = {"available": False}
    for name in ("cvxpy", "casadi", "cyipopt", "osqp", "clarabel"):
        import importlib.util
        result[name] = {"available": importlib.util.find_spec(name) is not None}
    result["selected"] = "scipy.optimize.SLSQP on one preregistered 25-frame nonlinear window"
    result["reason"] = "small dense window, true nonlinear equality/inequality support, analytic FK Jacobians; no outer soft-weight scan"
    return result


class WindowProblem:
    def __init__(
        self, model: mujoco.MjModel, source: dict[str, Any], warm: dict[str, Any],
        intent: dict[str, np.ndarray], lower_indices: np.ndarray,
    ) -> None:
        self.model = model
        self.source = source
        self.intent = {side: np.asarray(intent[side], dtype=bool) for side in phase29.SIDES}
        self.lower_indices = np.asarray(lower_indices, dtype=np.int64)
        self.frames = np.arange(WINDOW_START, WINDOW_END_EXCLUSIVE, dtype=np.int64)
        self.nframes = len(self.frames)
        self.width = 1 + len(self.lower_indices)
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
        self.variable_dofs = np.concatenate([np.asarray([2], dtype=np.int64), lower_dofs])
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in phase28.TARGET_BODIES]
        self.floor, self.feet = phase7.active_sole_spheres(model)
        self.original_kin = phase28.target_kinematics(source, model)
        self.original_points = self.original_kin["target_points"][self.frames]
        self.original_distance = {
            side: np.asarray(self.original_kin["sole_distance"][side])[self.frames] for side in phase29.SIDES
        }
        self.original_q = np.asarray(source["dof"], dtype=np.float64)
        self.original_root = np.asarray(source["root_trans_offset"], dtype=np.float64)
        root_seed = np.asarray(warm["root_trans_offset"], dtype=np.float64)[self.frames, 2] - self.original_root[self.frames, 2]
        joint_seed = np.asarray(warm["dof"], dtype=np.float64)[self.frames][:, self.lower_indices] - self.original_q[self.frames][:, self.lower_indices]
        self.x0 = np.column_stack([root_seed, joint_seed]).reshape(-1)
        self._cache_x: np.ndarray | None = None
        self._cache_states: list[dict[str, Any]] | None = None
        self.constraint_spec = self._build_constraint_spec()

    def cols(self, local_frame: int) -> np.ndarray:
        start = local_frame * self.width
        return np.arange(start, start + self.width, dtype=np.int64)

    def entry_from_x(self, x: np.ndarray) -> dict[str, Any]:
        entry = self.source
        root = np.asarray(entry["root_trans_offset"], dtype=np.float64).copy()
        q = np.asarray(entry["dof"], dtype=np.float64).copy()
        correction = np.asarray(x).reshape(self.nframes, self.width)
        root[self.frames, 2] += correction[:, 0]
        q[np.ix_(self.frames, self.lower_indices)] += correction[:, 1:]
        return {**entry, "root_trans_offset": root, "dof": q}

    def states(self, x: np.ndarray) -> list[dict[str, Any]]:
        x = np.asarray(x, dtype=np.float64)
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            assert self._cache_states is not None
            return self._cache_states
        entry = self.entry_from_x(x)
        data = mujoco.MjData(self.model)
        states = []
        fromto = np.zeros(6, dtype=np.float64)
        for local, frame in enumerate(self.frames):
            phase28.set_entry_frame(self.model, data, entry, int(frame), self.qpos_addresses)
            points, point_jac = [], []
            for body_id in self.body_ids:
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacBody(self.model, data, jacp, jacr, body_id)
                points.append(data.xpos[body_id].copy())
                point_jac.append(jacp[:, self.variable_dofs].copy())
            sole, foot_xy, foot_xy_jac = {}, {}, {}
            for side, geoms in self.feet.items():
                distances = np.asarray([
                    mujoco.mj_geomDistance(self.model, data, self.floor, geom, 1.0, fromto)
                    for geom in geoms
                ])
                active = int(geoms[int(np.argmin(distances))])
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacGeom(self.model, data, jacp, jacr, active)
                sole[side] = {"distance": float(np.min(distances)), "jacobian": jacp[2, self.variable_dofs].copy()}
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
                mujoco.mj_jacBody(self.model, data, jacp, jacr, body_id)
                foot_xy[side] = data.xpos[body_id, :2].copy()
                foot_xy_jac[side] = jacp[:2, self.variable_dofs].copy()
            states.append({
                "points": np.asarray(points), "point_jac": np.asarray(point_jac),
                "sole": sole, "foot_xy": foot_xy, "foot_xy_jac": foot_xy_jac,
            })
        self._cache_x = x.copy()
        self._cache_states = states
        return states

    def _build_constraint_spec(self) -> list[tuple[Any, ...]]:
        spec: list[tuple[Any, ...]] = []
        fps = float(self.source["fps"])
        for local, frame in enumerate(self.frames):
            for side in phase29.SIDES:
                if self.intent[side][frame]:
                    spec.append(("stance_distance", local, side, 0.0, STANCE_DISTANCE_UPPER_M))
                else:
                    desired = max(0.0, float(self.original_distance[side][local]))
                    spec.append(("swing_distance", local, side, desired, np.inf))
        for side in phase29.SIDES:
            for local in range(1, self.nframes):
                f0, f1 = int(self.frames[local - 1]), int(self.frames[local])
                if self.intent[side][f0] and self.intent[side][f1]:
                    spec.append(("stance_speed_step", local - 1, local, side, -np.inf, STANCE_SPEED_MAX_MPS / fps))
            # Excursion anchors restart at each contiguous intent phase inside
            # the fixed window, exactly as the source phase contract dictates.
            mask = self.intent[side][self.frames]
            for start, end in phase28.contiguous_true(mask):
                for local in range(start + 1, end):
                    spec.append(("stance_excursion", start, local, side, -np.inf, STANCE_EXCURSION_MAX_M))
        return spec

    def objective(self, x: np.ndarray) -> float:
        correction = np.asarray(x).reshape(self.nframes, self.width)
        value = OBJECTIVE_WEIGHTS["root_z_correction"] * np.sum(correction[:, 0] ** 2)
        value += OBJECTIVE_WEIGHTS["joint_correction"] * np.sum(correction[:, 1:] ** 2)
        value += OBJECTIVE_WEIGHTS["correction_velocity"] * np.sum(np.diff(correction, axis=0) ** 2)
        value += OBJECTIVE_WEIGHTS["correction_acceleration"] * np.sum(np.diff(correction, n=2, axis=0) ** 2)
        states = self.states(x)
        for local, state in enumerate(states):
            error = state["points"] - self.original_points[local]
            value += OBJECTIVE_WEIGHTS["source_fit_keypoint"] * np.sum(error ** 2)
        return float(value)

    def objective_jac(self, x: np.ndarray) -> np.ndarray:
        correction = np.asarray(x).reshape(self.nframes, self.width)
        grad = np.zeros_like(correction)
        grad[:, 0] += 2 * OBJECTIVE_WEIGHTS["root_z_correction"] * correction[:, 0]
        grad[:, 1:] += 2 * OBJECTIVE_WEIGHTS["joint_correction"] * correction[:, 1:]
        velocity = np.diff(correction, axis=0)
        grad[:-1] -= 2 * OBJECTIVE_WEIGHTS["correction_velocity"] * velocity
        grad[1:] += 2 * OBJECTIVE_WEIGHTS["correction_velocity"] * velocity
        acceleration = np.diff(correction, n=2, axis=0)
        grad[:-2] += 2 * OBJECTIVE_WEIGHTS["correction_acceleration"] * acceleration
        grad[1:-1] -= 4 * OBJECTIVE_WEIGHTS["correction_acceleration"] * acceleration
        grad[2:] += 2 * OBJECTIVE_WEIGHTS["correction_acceleration"] * acceleration
        for local, state in enumerate(self.states(x)):
            error = state["points"] - self.original_points[local]
            grad[local] += 2 * OBJECTIVE_WEIGHTS["source_fit_keypoint"] * np.einsum(
                "pa,paw->w", error, state["point_jac"]
            )
        return grad.reshape(-1)

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        states = self.states(x)
        values = []
        for item in self.constraint_spec:
            kind = item[0]
            if kind in ("stance_distance", "swing_distance"):
                _, local, side, _, _ = item
                values.append(states[local]["sole"][side]["distance"])
            else:
                _, anchor, local, side, _, _ = item
                delta = states[local]["foot_xy"][side] - states[anchor]["foot_xy"][side]
                values.append(float(np.linalg.norm(delta)))
        return np.asarray(values)

    def constraint_jac(self, x: np.ndarray) -> sparse.csr_matrix:
        states = self.states(x)
        rows, cols, values = [], [], []
        for row, item in enumerate(self.constraint_spec):
            kind = item[0]
            if kind in ("stance_distance", "swing_distance"):
                _, local, side, _, _ = item
                jac = states[local]["sole"][side]["jacobian"]
                for column, value in zip(self.cols(local), jac):
                    if value:
                        rows.append(row); cols.append(int(column)); values.append(float(value))
            else:
                _, anchor, local, side, _, _ = item
                delta = states[local]["foot_xy"][side] - states[anchor]["foot_xy"][side]
                norm = float(np.linalg.norm(delta))
                if norm > 1.0e-12:
                    direction = delta / norm
                    jac_local = direction @ states[local]["foot_xy_jac"][side]
                    jac_anchor = -direction @ states[anchor]["foot_xy_jac"][side]
                    for column, value in zip(self.cols(local), jac_local):
                        if value:
                            rows.append(row); cols.append(int(column)); values.append(float(value))
                    for column, value in zip(self.cols(anchor), jac_anchor):
                        if value:
                            rows.append(row); cols.append(int(column)); values.append(float(value))
        return sparse.coo_matrix((values, (rows, cols)), shape=(len(self.constraint_spec), self.nvar)).tocsr()

    def nonlinear_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([item[-2] for item in self.constraint_spec], dtype=np.float64),
            np.asarray([item[-1] for item in self.constraint_spec], dtype=np.float64),
        )

    def variable_bounds(self) -> Bounds:
        lower = np.full((self.nframes, self.width), -JOINT_CORRECTION_MAX_RAD)
        upper = np.full((self.nframes, self.width), JOINT_CORRECTION_MAX_RAD)
        lower[:, 0], upper[:, 0] = -ROOT_Z_CORRECTION_MAX_M, ROOT_Z_CORRECTION_MAX_M
        for local, frame in enumerate(self.frames):
            for joint_local, entry_index in enumerate(self.lower_indices):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.names[int(entry_index)])
                if self.model.jnt_limited[joint_id]:
                    low, high = self.model.jnt_range[joint_id]
                    lower[local, 1 + joint_local] = max(lower[local, 1 + joint_local], low - self.original_q[frame, entry_index])
                    upper[local, 1 + joint_local] = min(upper[local, 1 + joint_local], high - self.original_q[frame, entry_index])
        return Bounds(lower.reshape(-1), upper.reshape(-1))

    def qstep_constraint(self) -> LinearConstraint:
        rows, cols, values, residual = [], [], [], []
        row = 0
        # Include all internal steps and the fixed frame25 boundary.
        for frame in range(WINDOW_START + 1, WINDOW_END_EXCLUSIVE + 1):
            for joint_local, entry_index in enumerate(self.lower_indices):
                base_step = self.original_q[frame, entry_index] - self.original_q[frame - 1, entry_index]
                if frame < WINDOW_END_EXCLUSIVE:
                    local = frame - WINDOW_START
                    rows.append(row); cols.append(int(self.cols(local)[1 + joint_local])); values.append(1.0)
                prev_local = frame - 1 - WINDOW_START
                rows.append(row); cols.append(int(self.cols(prev_local)[1 + joint_local])); values.append(-1.0)
                residual.append(base_step)
                row += 1
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(row, self.nvar)).tocsr()
        residual = np.asarray(residual)
        return LinearConstraint(matrix, -JOINT_STEP_MAX_RAD - residual, JOINT_STEP_MAX_RAD - residual)


def render(report: dict[str, Any]) -> str:
    result = report["result"]
    return f"""# X2 WBT Phase38：hard-constraint feasibility sweet-point

## 裁决

- **{report['decision']['status']}**
- 只运行source-intent预注册的首个完整DS→SS→DS窗口：frames `[0,25)`，未读held、未运行physics/PPO。
- solver：`{report['solver_audit']['selected']}`；这不是Phase37 soft-weight扫描。

## 硬约束

- intended stance official sole signed-distance：`0 <= d <= 0.5mm`。
- 全帧nonpenetration；swing clearance不得低于原Phase30；stance speed<=0.10m/s、excursion<=0.03m。
- joint limits/root-z correction bounds、每步qstep<=0.15rad；rootXY/root horizontal acceleration因表示冻结。
- source-fit keypoint仅作objective，不取代接触硬门。

## 结果

- solver success/status：`{result['success']}` / `{result['status']}`；message：`{result['message']}`。
- iterations/evaluations：`{result['nit']}` / `{result['nfev']}`；objective：{result['objective']:.6f}。
- max constraint violation：{result['max_constraint_violation']:.6g}。
- stance distance p95/max：`{result['stance_abs_distance_p95_max_m']}`m；minimum distance：{result['minimum_distance_m']:.6f}m。
- stance speed max：{result['stance_speed_max_mps']:.4f}m/s；excursion max：{result['stance_excursion_max_m']:.4f}m；qstep max：{result['qstep_max_rad']:.4f}rad。

## 结论

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    panel = {row["id"]: row for row in json.loads(phase28.PANEL.read_text())["motions"]}
    row = panel[MOTION_ID]
    source = joblib.load(phase30.OUTPUT_CACHE)[MOTION_ID]
    warm = joblib.load(phase37.OUTPUT_CACHE)[MOTION_ID]
    source29 = joblib.load(phase29.OUTPUT_CACHE)[MOTION_ID]
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_phase38_static_fk_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    source_contract = phase29.source_contract(row, source29, model)
    intent = {
        side: phase30.nearest_phase_resample(source_contract["contact"][side], len(source["dof"]))
        for side in phase29.SIDES
    }
    lower = phase29.lower_body_indices(list(source["joint_names_mujoco"]))
    problem = WindowProblem(model, source, warm, intent, lower)
    # Window selection must exactly match the first source-intent DS->SS->DS.
    left, right = intent["left"], intent["right"]
    state = np.where(left & right, 2, np.where(left ^ right, 1, 0))
    window_state = state[WINDOW_START:WINDOW_END_EXCLUSIVE]
    window_contract_pass = (
        np.all(window_state[:10] == 2) and np.all(window_state[10:20] == 1)
        and np.all(window_state[20:] == 2)
    )
    audit = solver_audit()
    nl_lower, nl_upper = problem.nonlinear_bounds()
    nonlinear = NonlinearConstraint(
        problem.constraint_values, nl_lower, nl_upper, jac=problem.constraint_jac
    )
    print(
        f"[phase38] preflight window={problem.nframes} vars={problem.nvar} "
        f"constraints={len(problem.constraint_spec)} contract={window_contract_pass}", flush=True,
    )
    optimization = minimize(
        problem.objective, problem.x0, jac=problem.objective_jac,
        method="SLSQP", bounds=problem.variable_bounds(),
        constraints=[nonlinear, problem.qstep_constraint()],
        options={"maxiter": MAX_ITERATIONS, "ftol": FTOL, "disp": True},
    )
    x = np.asarray(optimization.x)
    values = problem.constraint_values(x)
    lower_violation = np.maximum(nl_lower - values, 0.0)
    upper_violation = np.maximum(values - nl_upper, 0.0)
    max_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
    entry = problem.entry_from_x(x)
    kin = phase28.target_kinematics(entry, model)
    distances = {side: np.asarray(kin["sole_distance"][side]) for side in phase29.SIDES}
    stance_values = np.concatenate([
        np.abs(distances[side][problem.frames][intent[side][problem.frames]]) for side in phase29.SIDES
    ])
    min_distance = min(float(np.min(distances[side][problem.frames])) for side in phase29.SIDES)
    states = problem.states(x)
    speed, excursion = [], []
    fps = float(source["fps"])
    for item, value in zip(problem.constraint_spec, values):
        if item[0] == "stance_speed_step": speed.append(float(value * fps))
        elif item[0] == "stance_excursion": excursion.append(float(value))
    corrected_q = np.asarray(entry["dof"])
    qstep = float(np.max(np.abs(np.diff(corrected_q[WINDOW_START:WINDOW_END_EXCLUSIVE + 1, lower], axis=0))))
    feasible = bool(
        optimization.success and max_violation <= 1.0e-7
        and np.max(stance_values) <= STANCE_DISTANCE_UPPER_M + 1.0e-7
        and min_distance >= NONPENETRATION_LOWER_M - 1.0e-7
        and max(speed, default=0.0) <= STANCE_SPEED_MAX_MPS + 1.0e-7
        and max(excursion, default=0.0) <= STANCE_EXCURSION_MAX_M + 1.0e-7
        and qstep <= JOINT_STEP_MAX_RAD + 1.0e-7
    )
    result = {
        "success": bool(optimization.success), "status": int(optimization.status),
        "message": str(optimization.message), "nit": int(optimization.nit),
        "nfev": int(optimization.nfev), "njev": int(optimization.njev),
        "objective": float(optimization.fun), "max_constraint_violation": max_violation,
        "stance_abs_distance_p95_max_m": [float(np.percentile(stance_values, 95)), float(np.max(stance_values))],
        "minimum_distance_m": min_distance,
        "stance_speed_max_mps": max(speed, default=0.0),
        "stance_excursion_max_m": max(excursion, default=0.0),
        "qstep_max_rad": qstep,
        "root_z_correction_max_abs_m": float(np.max(np.abs(x.reshape(problem.nframes, problem.width)[:, 0]))),
        "joint_correction_max_abs_rad": float(np.max(np.abs(x.reshape(problem.nframes, problem.width)[:, 1:]))),
        "window_feasible_certificate": feasible,
    }
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ, correction=x.reshape(problem.nframes, problem.width),
        frames=problem.frames, constraint_values=values,
        constraint_lower=nl_lower, constraint_upper=nl_upper,
    )
    if feasible:
        status = "PHASE38_LOCAL_HARD_CONSTRAINT_WINDOW_FEASIBLE"
        conclusion = "首个完整DS→SS→DS局部窗存在满足全部硬约束的root-z+腰腿解；这仅是局部可行证书，不是完整轨迹Silver。"
        next_step = "保持physics/PPO锁定；如获授权，可用同一硬约束合同扩展一次完整轨迹，不得改参数。"
    else:
        status = "PHASE38_LOCAL_HARD_CONSTRAINT_WINDOW_INFEASIBLE_OR_SOLVER_FAILED"
        conclusion = "当前表示/边界下，真正非线性硬约束局部窗未取得可行证书；不得把Phase37 soft改善晋升为Silver。"
        next_step = "按门停止；不换窗口、不放宽阈值、不扫参数、不运行完整轨迹或physics。"
    report = {
        "schema_version": "x2_wbt_hard_constraint_certificate_phase38_v1",
        "truth_boundary": {
            "window_only_not_full_trajectory_silver": True,
            "mujoco_integration_steps": 0, "policy_optimizer_ppo_training": False,
            "offline_slsqp_run_count": 1, "held_out_read": False,
            "contact_is_model_geometry_not_hardware_truth": True,
        },
        "solver_audit": audit,
        "pre_registration": {
            "window_frames": [WINDOW_START, WINDOW_END_EXCLUSIVE],
            "selection": "first complete source-intent DS(0:10)->right-SS(10:20)->DS(20:25); no Phase37 metric used",
            "window_contract_pass": bool(window_contract_pass),
            "variables": "root-z + lower/waist15 only; Phase37 candidate used only as feasible-set warm start",
            "thresholds": {
                "stance_distance_upper_m": STANCE_DISTANCE_UPPER_M,
                "nonpenetration_lower_m": NONPENETRATION_LOWER_M,
                "stance_speed_mps": STANCE_SPEED_MAX_MPS,
                "stance_excursion_m": STANCE_EXCURSION_MAX_M,
                "qstep_max_rad": JOINT_STEP_MAX_RAD,
            },
            "parameter_or_window_scan": False,
        },
        "provenance": {
            "phase30_cache_sha256": phase28.sha256(phase30.OUTPUT_CACHE),
            "phase37_warmstart_sha256": phase28.sha256(phase37.OUTPUT_CACHE),
            "official_scene_sha256": phase28.sha256(physics.DEFAULT_SCENE),
            "tier_gates_sha256": phase28.sha256(phase28.TIER_GATES),
        },
        "result": result,
        "artifact": {"path": str(OUTPUT_NPZ), "sha256": phase28.sha256(OUTPUT_NPZ)},
        "decision": {
            "status": status, "local_window_feasible": feasible,
            "full_trajectory_true_silver": False,
            "physics_proposal_allowed": False,
            "conclusion": conclusion, "next_step": next_step,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
