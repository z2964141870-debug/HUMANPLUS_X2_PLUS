#!/usr/bin/env python3
"""Phase29: contact-constrained full-trajectory repair existence test.

This experiment is deliberately offline.  It takes three frozen train motions
from Phase28 and jointly optimizes root translation, the 15 waist/lower-body
joints, and one horizontal anchor for every *source-intended* stance phase.
Every Gauss--Newton step is one sparse trajectory solve; this is not the
frame-local DLS used by Phase8 and is not a clipwise parameter search.

Contacts and FK are official-AimDK-v1 model estimates.  They are not measured
GRF, COP, force, or real-robot contact truth.  No physics integration, policy
forward, optimizer/PPO update, checkpoint, or robot operation occurs here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_panel_phase28 as phase28


PHASE28_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase28_panel24/official_v1_model/x2_phase28_panel24.pkl"
)
OUTPUT_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase29_full_trajectory_repair"
)
OUTPUT_CACHE = OUTPUT_ROOT / "x2_phase29_full_trajectory_repair.pkl"
REPORT_JSON = REPO / "reports/retarget/x2_wbt_full_trajectory_repair_phase29.json"
REPORT_MD = REPO / "reports/retarget/x2_wbt_full_trajectory_repair_phase29.md"
PREFLIGHT_JSON = REPO / "reports/retarget/x2_wbt_full_trajectory_repair_phase29_preflight.json"

MOTION_IDS = ("AMASS-WALK-001", "PHUMA-LUNGE-R-001", "AMASS-KICK-L-001")
SIDES = ("left", "right")
ROOT_DIMS = 3
ROOT_XY_CORRECTION_MAX_M = 0.20
ROOT_Z_CORRECTION_MAX_M = 0.08
JOINT_CORRECTION_MAX_RAD = 0.45
GN_ITERATIONS = 8
GN_STEP_SCALE = 0.65
LSQR_ATOL = 1.0e-6
LSQR_BTOL = 1.0e-6
LSQR_ITER_LIMIT = 240

# Frozen single configuration.  Values are residual square-root weights.
WEIGHTS = {
    "keypoint": 4.0,
    "stance_xy": 14.0,
    "stance_z": 12.0,
    "swing_clearance": 8.0,
    "anchor_source": 1.5,
    "root_correction": 0.8,
    "joint_correction": 0.7,
    "root_velocity": 2.0,
    "root_acceleration": 5.0,
    "joint_velocity": 1.2,
    "joint_acceleration": 2.5,
}


def lower_body_indices(names: list[str]) -> np.ndarray:
    def selected(name: str) -> bool:
        return (
            name.startswith("waist_")
            or any(name.startswith(f"{side}_{token}") for side in SIDES for token in ("hip_", "knee_", "ankle_"))
        )
    return np.asarray([index for index, name in enumerate(names) if selected(name)], dtype=np.int64)


def source_contract(row: dict[str, Any], entry: dict[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
    target = phase28.target_kinematics(entry, model)["target_points"]
    source = phase28.source_points(row, len(entry["dof"]))
    scale, rotation, translation = phase28.fit_similarity(source, target)
    fitted = scale * (source @ rotation.T) + translation
    ground = min(phase28.percentile(fitted[:, 2, 2], 5), phase28.percentile(fitted[:, 3, 2], 5))
    contact = {
        "left": fitted[:, 2, 2] <= ground + phase28.SOURCE_CONTACT_HEIGHT_M,
        "right": fitted[:, 3, 2] <= ground + phase28.SOURCE_CONTACT_HEIGHT_M,
    }
    phases = {side: phase28.contiguous_true(contact[side]) for side in SIDES}
    return {
        "fitted_keypoints": fitted,
        "ground_m": float(ground),
        "contact": contact,
        "phases": phases,
        "fit": {"scale": float(scale), "rotation": rotation, "translation": translation},
    }


@dataclass(frozen=True)
class Layout:
    frames: int
    lower_count: int
    phases: dict[str, list[tuple[int, int]]]

    @property
    def frame_width(self) -> int:
        return ROOT_DIMS + self.lower_count

    @property
    def trajectory_size(self) -> int:
        return self.frames * self.frame_width

    @property
    def anchor_count(self) -> int:
        return sum(len(self.phases[side]) for side in SIDES)

    @property
    def size(self) -> int:
        return self.trajectory_size + 2 * self.anchor_count

    def frame_cols(self, frame: int) -> np.ndarray:
        start = frame * self.frame_width
        return np.arange(start, start + self.frame_width, dtype=np.int64)

    def anchor_col(self, side: str, phase_index: int) -> int:
        prior = len(self.phases["left"]) if side == "right" else 0
        return self.trajectory_size + 2 * (prior + phase_index)


class SparseSystem:
    def __init__(self, columns: int):
        self.columns = columns
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.values: list[float] = []
        self.rhs: list[float] = []

    def add(self, coefficients: dict[int, float], residual: float, weight: float) -> None:
        row = len(self.rhs)
        for column, value in coefficients.items():
            weighted = weight * float(value)
            if weighted != 0.0:
                self.rows.append(row)
                self.cols.append(int(column))
                self.values.append(weighted)
        self.rhs.append(-weight * float(residual))

    def matrix(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        shape = (len(self.rhs), self.columns)
        matrix = sparse.coo_matrix((self.values, (self.rows, self.cols)), shape=shape).tocsr()
        return matrix, np.asarray(self.rhs, dtype=np.float64)


def selected_jacobian(jacp: np.ndarray, lower_dof_addresses: np.ndarray) -> np.ndarray:
    return np.concatenate((jacp[:, :ROOT_DIMS], jacp[:, lower_dof_addresses]), axis=1)


def dictionary_row(columns: np.ndarray, values: np.ndarray) -> dict[int, float]:
    return {int(column): float(value) for column, value in zip(columns, values, strict=True) if value != 0.0}


def add_difference(
    system: SparseSystem,
    columns_a: np.ndarray,
    columns_b: np.ndarray,
    current_a: np.ndarray,
    current_b: np.ndarray,
    weight: float,
) -> None:
    for index in range(len(columns_a)):
        system.add(
            {int(columns_a[index]): 1.0, int(columns_b[index]): -1.0},
            float(current_a[index] - current_b[index]), weight,
        )


def add_second_difference(
    system: SparseSystem,
    columns_prev: np.ndarray,
    columns_now: np.ndarray,
    columns_next: np.ndarray,
    prev: np.ndarray,
    now: np.ndarray,
    nxt: np.ndarray,
    weight: float,
) -> None:
    for index in range(len(columns_now)):
        system.add(
            {
                int(columns_prev[index]): 1.0,
                int(columns_now[index]): -2.0,
                int(columns_next[index]): 1.0,
            },
            float(prev[index] - 2.0 * now[index] + nxt[index]), weight,
        )


def candidate_from_state(
    original: dict[str, Any], root_delta: np.ndarray, joint_delta: np.ndarray,
    lower_indices: np.ndarray, joint_axes: np.ndarray,
) -> dict[str, Any]:
    result = copy.deepcopy(original)
    root = np.asarray(original["root_trans_offset"], dtype=np.float64) + root_delta
    dof = np.asarray(original["dof"], dtype=np.float64).copy()
    dof[:, lower_indices] += joint_delta
    result["root_trans_offset"] = root.astype(np.asarray(original["root_trans_offset"]).dtype)
    result["dof"] = dof.astype(np.asarray(original["dof"]).dtype)
    pose = np.zeros((len(dof), len(result["joint_names_mujoco"]) + 1, 3), dtype=np.float32)
    pose[:, 1:, :] = dof[:, :, None].astype(np.float32) * joint_axes[None]
    result["pose_aa"] = pose
    return result


def frame_state(
    model: mujoco.MjModel, data: mujoco.MjData, candidate: dict[str, Any], frame: int,
    qpos_addresses: list[int], body_ids: list[int], feet: dict[str, list[int]], floor: int,
    lower_dof_addresses: np.ndarray,
) -> dict[str, Any]:
    phase28.set_entry_frame(model, data, candidate, frame, qpos_addresses)
    point_positions, point_jacobians = [], []
    for body_id in body_ids:
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        point_positions.append(data.xpos[body_id].copy())
        point_jacobians.append(selected_jacobian(jacp, lower_dof_addresses))
    sole: dict[str, Any] = {}
    fromto = np.zeros(6, dtype=np.float64)
    for side, geom_ids in feet.items():
        distances = [float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto)) for geom in geom_ids]
        active = int(geom_ids[int(np.argmin(distances))])
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(model, data, jacp, jacr, active)
        sole[side] = {
            "distance": float(min(distances)),
            "jacobian_z": selected_jacobian(jacp, lower_dof_addresses)[2],
        }
    return {
        "points": np.asarray(point_positions),
        "point_jacobians": np.asarray(point_jacobians),
        "sole": sole,
    }


def build_system(
    model: mujoco.MjModel, original: dict[str, Any], candidate: dict[str, Any],
    contract: dict[str, Any], reset: dict[str, Any], layout: Layout,
    lower_indices: np.ndarray, lower_dof_addresses: np.ndarray,
    anchors: dict[str, np.ndarray], root_delta: np.ndarray, joint_delta: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, float]]:
    names = list(original["joint_names_mujoco"])
    qpos_addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in phase28.TARGET_BODIES]
    floor, feet = phase7.active_sole_spheres(model)
    system = SparseSystem(layout.size)
    data = mujoco.MjData(model)
    states: list[dict[str, Any]] = []
    source_fit = np.asarray(contract["fitted_keypoints"], dtype=np.float64)
    original_kin = phase28.target_kinematics(original, model)
    original_distance = original_kin["sole_distance"]
    contact_target_distance = float(reset["reset_clearance_m"])

    phase_lookup = {side: np.full(layout.frames, -1, dtype=np.int64) for side in SIDES}
    for side in SIDES:
        for phase_index, (start, end) in enumerate(layout.phases[side]):
            phase_lookup[side][start:end] = phase_index

    objective = {"keypoint_l2": 0.0, "stance_xy_l2": 0.0, "stance_z_l2": 0.0, "swing_hinge_l2": 0.0}
    for frame in range(layout.frames):
        state = frame_state(model, data, candidate, frame, qpos_addresses, body_ids, feet, floor, lower_dof_addresses)
        states.append(state)
        columns = layout.frame_cols(frame)

        # Keep the six source action semantics under the Phase28 whole-clip fit.
        for point_index in range(len(phase28.TARGET_BODIES)):
            error = state["points"][point_index] - source_fit[frame, point_index]
            objective["keypoint_l2"] += float(error @ error)
            jacobian = state["point_jacobians"][point_index]
            for axis in range(3):
                system.add(dictionary_row(columns, jacobian[axis]), error[axis], WEIGHTS["keypoint"])

        for side, point_index in (("left", 2), ("right", 3)):
            phase_index = int(phase_lookup[side][frame])
            if phase_index >= 0:
                anchor_col = layout.anchor_col(side, phase_index)
                xy_error = state["points"][point_index, :2] - anchors[side][phase_index]
                objective["stance_xy_l2"] += float(xy_error @ xy_error)
                jacobian = state["point_jacobians"][point_index]
                for axis in range(2):
                    coefficients = dictionary_row(columns, jacobian[axis])
                    coefficients[anchor_col + axis] = -1.0
                    system.add(coefficients, xy_error[axis], WEIGHTS["stance_xy"])
                z_error = state["sole"][side]["distance"] - contact_target_distance
                objective["stance_z_l2"] += z_error * z_error
                system.add(dictionary_row(columns, state["sole"][side]["jacobian_z"]), z_error, WEIGHTS["stance_z"])
            else:
                # Preserve source swing height and never collapse a swing into contact.
                source_clearance = max(0.012, float(source_fit[frame, point_index, 2] - contract["ground_m"]))
                original_clearance = max(0.0, float(original_distance[side][frame] - reset["reset_clearance_m"]))
                desired = float(reset["reset_clearance_m"] + max(source_clearance, min(original_clearance, 0.20)))
                violation = desired - state["sole"][side]["distance"]
                if violation > 0.0:
                    objective["swing_hinge_l2"] += violation * violation
                    jac = -np.asarray(state["sole"][side]["jacobian_z"])
                    system.add(dictionary_row(columns, jac), violation, WEIGHTS["swing_clearance"])

    # Optimize one anchor per complete source stance phase, with a weak source-semantic prior.
    for side, point_index in (("left", 2), ("right", 3)):
        for phase_index, (start, end) in enumerate(layout.phases[side]):
            source_anchor = np.mean(source_fit[start:end, point_index, :2], axis=0)
            anchor_col = layout.anchor_col(side, phase_index)
            for axis in range(2):
                system.add({anchor_col + axis: 1.0}, anchors[side][phase_index, axis] - source_anchor[axis], WEIGHTS["anchor_source"])

    # Absolute correction priors and global temporal coupling.  These operate on
    # the whole trajectory, including phase boundaries; no C2 window splice.
    root_now = np.asarray(candidate["root_trans_offset"], dtype=np.float64)
    joint_now = np.asarray(candidate["dof"], dtype=np.float64)[:, lower_indices]
    for frame in range(layout.frames):
        columns = layout.frame_cols(frame)
        for axis in range(ROOT_DIMS):
            system.add({int(columns[axis]): 1.0}, root_delta[frame, axis], WEIGHTS["root_correction"])
        for joint in range(layout.lower_count):
            system.add({int(columns[ROOT_DIMS + joint]): 1.0}, joint_delta[frame, joint], WEIGHTS["joint_correction"])
    for frame in range(1, layout.frames):
        prev, now = layout.frame_cols(frame - 1), layout.frame_cols(frame)
        add_difference(system, now[:ROOT_DIMS], prev[:ROOT_DIMS], root_now[frame], root_now[frame-1], WEIGHTS["root_velocity"])
        add_difference(system, now[ROOT_DIMS:], prev[ROOT_DIMS:], joint_now[frame], joint_now[frame-1], WEIGHTS["joint_velocity"])
    for frame in range(1, layout.frames - 1):
        prev, now, nxt = layout.frame_cols(frame-1), layout.frame_cols(frame), layout.frame_cols(frame+1)
        add_second_difference(system, prev[:ROOT_DIMS], now[:ROOT_DIMS], nxt[:ROOT_DIMS], root_now[frame-1], root_now[frame], root_now[frame+1], WEIGHTS["root_acceleration"])
        add_second_difference(system, prev[ROOT_DIMS:], now[ROOT_DIMS:], nxt[ROOT_DIMS:], joint_now[frame-1], joint_now[frame], joint_now[frame+1], WEIGHTS["joint_acceleration"])

    matrix, rhs = system.matrix()
    objective["rows"] = float(matrix.shape[0])
    objective["columns"] = float(matrix.shape[1])
    objective["nnz"] = float(matrix.nnz)
    objective["weighted_rhs_l2"] = float(np.linalg.norm(rhs))
    return matrix, rhs, objective


def project(
    model: mujoco.MjModel, original: dict[str, Any], root_delta: np.ndarray,
    joint_delta: np.ndarray, lower_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(root_delta[:, :2], axis=1)
    scale = np.minimum(1.0, ROOT_XY_CORRECTION_MAX_M / np.maximum(norm, 1.0e-12))
    root_delta[:, :2] *= scale[:, None]
    root_delta[:, 2] = np.clip(root_delta[:, 2], -ROOT_Z_CORRECTION_MAX_M, ROOT_Z_CORRECTION_MAX_M)
    joint_delta[:] = np.clip(joint_delta, -JOINT_CORRECTION_MAX_RAD, JOINT_CORRECTION_MAX_RAD)
    original_dof = np.asarray(original["dof"], dtype=np.float64)
    names = list(original["joint_names_mujoco"])
    for local, entry_index in enumerate(lower_indices):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[int(entry_index)])
        if model.jnt_limited[jid]:
            low, high = model.jnt_range[jid]
            joint_delta[:, local] = np.clip(joint_delta[:, local], low-original_dof[:, entry_index], high-original_dof[:, entry_index])
    return root_delta, joint_delta


def solve_motion(
    row: dict[str, Any], original: dict[str, Any], model: mujoco.MjModel,
    reset: dict[str, Any], gates: dict[str, Any], mirror: dict[str, Any], joint_axes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    names = list(original["joint_names_mujoco"])
    lower_indices = lower_body_indices(names)
    lower_dof_addresses = np.asarray([
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[int(index)])])
        for index in lower_indices
    ], dtype=np.int64)
    contract = source_contract(row, original, model)
    layout = Layout(len(original["dof"]), len(lower_indices), contract["phases"])
    root_delta = np.zeros((layout.frames, ROOT_DIMS), dtype=np.float64)
    joint_delta = np.zeros((layout.frames, layout.lower_count), dtype=np.float64)
    original_kin = phase28.target_kinematics(original, model)["target_points"]
    anchors = {
        side: np.asarray([
            np.mean(original_kin[start:end, 2 if side == "left" else 3, :2], axis=0)
            for start, end in layout.phases[side]
        ], dtype=np.float64).reshape(-1, 2)
        for side in SIDES
    }
    iterations: list[dict[str, Any]] = []
    candidate = candidate_from_state(original, root_delta, joint_delta, lower_indices, joint_axes)
    for iteration in range(GN_ITERATIONS):
        iter_started = time.perf_counter()
        matrix, rhs, objective = build_system(
            model, original, candidate, contract, reset, layout, lower_indices,
            lower_dof_addresses, anchors, root_delta, joint_delta,
        )
        solution = lsqr(matrix, rhs, atol=LSQR_ATOL, btol=LSQR_BTOL, iter_lim=LSQR_ITER_LIMIT, show=False)
        step = GN_STEP_SCALE * np.asarray(solution[0], dtype=np.float64)
        trajectory = step[:layout.trajectory_size].reshape(layout.frames, layout.frame_width)
        root_delta += trajectory[:, :ROOT_DIMS]
        joint_delta += trajectory[:, ROOT_DIMS:]
        root_delta, joint_delta = project(model, original, root_delta, joint_delta, lower_indices)
        for side in SIDES:
            for phase_index in range(len(layout.phases[side])):
                column = layout.anchor_col(side, phase_index)
                anchors[side][phase_index] += step[column:column+2]
        candidate = candidate_from_state(original, root_delta, joint_delta, lower_indices, joint_axes)
        iterations.append({
            "iteration": iteration + 1,
            "objective_before_linear_step": objective,
            "lsqr_istop": int(solution[1]), "lsqr_iterations": int(solution[2]),
            "lsqr_residual_norm": float(solution[3]), "step_l2": float(np.linalg.norm(step)),
            "root_xy_correction_max_m": float(np.max(np.linalg.norm(root_delta[:, :2], axis=1))),
            "root_z_correction_max_abs_m": float(np.max(np.abs(root_delta[:, 2]))),
            "joint_correction_max_abs_rad": float(np.max(np.abs(joint_delta))),
            "elapsed_s": float(time.perf_counter()-iter_started),
        })
        print(
            f"[phase29] {row['id']} iter={iteration+1}/{GN_ITERATIONS} "
            f"lsqr={solution[2]} residual={solution[3]:.3f} rootxy={iterations[-1]['root_xy_correction_max_m']:.3f}",
            flush=True,
        )

    candidate["phase29_full_trajectory_repair"] = {
        "contract": "single-config sparse full-trajectory Gauss-Newton with phase-level source-stance anchors",
        "source_contact_truth_boundary": "model/source-height intent; not measured GRF/COP/force",
        "lower_body_joint_names": [names[int(index)] for index in lower_indices],
        "source_stance_phases": {side: [[int(a), int(b)] for a, b in layout.phases[side]] for side in SIDES},
        "optimized_anchor_xy_m": {side: anchors[side].tolist() for side in SIDES},
        "root_xy_correction_max_m": float(np.max(np.linalg.norm(root_delta[:, :2], axis=1))),
        "root_z_correction_max_abs_m": float(np.max(np.abs(root_delta[:, 2]))),
        "joint_correction_max_abs_rad": float(np.max(np.abs(joint_delta))),
        "configuration": configuration_manifest(),
    }
    diagnostics = {
        "frames": layout.frames, "variables": layout.size,
        "trajectory_variables": layout.trajectory_size, "anchor_variables": 2*layout.anchor_count,
        "stance_phase_count": {side: len(layout.phases[side]) for side in SIDES},
        "stance_frame_count": {side: int(np.count_nonzero(contract["contact"][side])) for side in SIDES},
        "source_contact_transitions": {side: int(len(phase28.transitions(contract["contact"][side]))) for side in SIDES},
        "iterations": iterations,
        "wall_time_s": float(time.perf_counter()-started),
        "root_xy_bound_respected": bool(np.max(np.linalg.norm(root_delta[:, :2], axis=1)) <= ROOT_XY_CORRECTION_MAX_M + 1e-9),
        "no_frame_or_segment_deleted": len(candidate["dof"]) == len(original["dof"]),
        "root_not_fixed": bool(np.max(np.linalg.norm(root_delta, axis=1)) > 0.0),
        "both_feet_not_forced": bool(all(np.any(~contract["contact"][side]) for side in SIDES)),
    }
    return candidate, diagnostics


def configuration_manifest() -> dict[str, Any]:
    return {
        "motions": list(MOTION_IDS), "per_clip_tuning": False,
        "optimizer": "sparse trajectory Gauss-Newton with scipy.sparse.linalg.lsqr",
        "gn_iterations": GN_ITERATIONS, "gn_step_scale": GN_STEP_SCALE,
        "lsqr_atol": LSQR_ATOL, "lsqr_btol": LSQR_BTOL, "lsqr_iteration_limit": LSQR_ITER_LIMIT,
        "root_xy_correction_max_m": ROOT_XY_CORRECTION_MAX_M,
        "root_z_correction_max_abs_m": ROOT_Z_CORRECTION_MAX_M,
        "joint_correction_max_abs_rad": JOINT_CORRECTION_MAX_RAD,
        "weights": WEIGHTS,
        "stance_schedule": "all contiguous Phase28 source-height contact intervals, none dropped/merged",
        "phase_anchor": "one jointly optimized XY variable per side per complete source stance phase",
        "forbidden": ["clipwise_parameter_search", "fixed_root", "frame_deletion", "segment_deletion", "forced_double_contact", "physics", "PPO"],
    }


def preflight(
    panel_rows: dict[str, dict[str, Any]], cache: dict[str, Any], model: mujoco.MjModel,
) -> dict[str, Any]:
    rows, overall = [], True
    for motion_id in MOTION_IDS:
        row = panel_rows.get(motion_id)
        entry = cache.get(motion_id)
        checks = {
            "panel_row_present": row is not None,
            "phase28_entry_present": entry is not None,
            "train_split": row is not None and row["recommended_split"] == "train_candidate",
            "source_sha_exact": row is not None and entry is not None and entry.get("source_sha256") == row["source_sha256"],
            "official_joint_order_31": entry is not None and len(entry["joint_names_mujoco"]) == 31,
            "head_locked": entry is not None and float(np.max(np.abs(np.asarray(entry["dof"])[:, -2:]))) <= 1e-12,
        }
        if row is not None and entry is not None:
            contract = source_contract(row, entry, model)
            lower = lower_body_indices(list(entry["joint_names_mujoco"]))
            layout = Layout(len(entry["dof"]), len(lower), contract["phases"])
            checks.update({
                "lower_body_15": len(lower) == 15,
                "both_sides_have_stance_phase": all(len(contract["phases"][side]) > 0 for side in SIDES),
                "single_support_semantics_present": bool(np.any(contract["contact"]["left"] ^ contract["contact"]["right"])),
                "swing_semantics_present_each_side": all(np.any(~contract["contact"][side]) for side in SIDES),
            })
            details = {
                "frames": len(entry["dof"]), "variables": layout.size,
                "stance_phase_count": {side: len(contract["phases"][side]) for side in SIDES},
                "stance_phase_lengths": {side: [int(b-a) for a, b in contract["phases"][side]] for side in SIDES},
                "lower_body_joint_names": [entry["joint_names_mujoco"][int(index)] for index in lower],
            }
        else:
            details = {}
        passed = bool(all(checks.values()))
        overall &= passed
        rows.append({"id": motion_id, "checks": checks, "details": details, "pass": passed})
    return {
        "schema_version": "x2_wbt_full_trajectory_repair_phase29_preflight_v1",
        "configuration": configuration_manifest(),
        "dependency": {"mujoco_jacBody": hasattr(mujoco, "mj_jacBody"), "mujoco_jacGeom": hasattr(mujoco, "mj_jacGeom"), "scipy_sparse_lsqr": True},
        "motions": rows,
        "pass": bool(overall and hasattr(mujoco, "mj_jacBody") and hasattr(mujoco, "mj_jacGeom")),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT Phase29：统一接触约束全轨迹 repair existence test", "",
        "## 裁决", "",
        f"- 冻结条件下 train Silver：`{report['summary']['silver_count']}/3`；可冻结：`{report['decision']['freeze_repair_contract']}`。",
        "- 本阶段 physics/PPO/policy-optimizer/checkpoint/真机均为 `0`；另有每条8轮 SciPy LSQR 离线轨迹求解，已逐轮记录。",
        "- contact/FK 是模型估计，不是实机 GRF、COP、足底力或真实接触真值。", "",
        "## 假设 / 干预 / 对照", "",
        "- 假设：Phase28 的系统性 stance speed/excursion/timing 失败，可能来自逐帧运动学重定向缺少整段接触一致性，而非数据完全不可用。",
        "- 干预：一个统一稀疏 Gauss–Newton 同时优化每帧 root XYZ、腰腿15DOF和每个 source stance phase 的XY anchor；root XY硬限20 cm。",
        "- 对照：Phase28 official-v1 原产物；Phase3 的短窗CEM/局部接触事件和 Phase8 的碎片窗口首帧anchor＋逐帧DLS均不复用。",
        "- 禁止项：逐clip调参、固定root、删帧/删段、强制双脚接触、physics选参。", "",
        "## 结果", "",
        "| motion | original tier | repair tier | rootXY max | q correction max | wall(s) | repair rejects |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["motions"]:
        diag = row["diagnostics"]
        phase = row["repair_entry_contract"]
        lines.append(
            f"| `{row['id']}` | {row['original_audit']['tier']} | {row['repair_audit']['tier']} | "
            f"{phase['root_xy_correction_max_m']:.4f} | {phase['joint_correction_max_abs_rad']:.4f} | "
            f"{diag['wall_time_s']:.1f} | `{' ; '.join(row['repair_audit']['reject_reasons'])}` |"
        )
    lines += ["", "## 结论", "", report["decision"]["conclusion"], "", "## 下一步", "", report["decision"]["next_step"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-cache", type=Path, default=OUTPUT_CACHE)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()

    panel = json.loads(phase28.PANEL.read_text())
    panel_rows = {row["id"]: row for row in panel["motions"]}
    gates = json.loads(phase28.TIER_GATES.read_text())
    mirror = json.loads(phase28.MIRROR_CONTRACT.read_text())
    cache = joblib.load(PHASE28_CACHE)
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase29")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)
    pre = preflight(panel_rows, cache, model)
    PREFLIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(phase28.json_safe(pre), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[phase29] preflight pass={pre['pass']} motions={sum(x['pass'] for x in pre['motions'])}/3", flush=True)
    if args.preflight_only or not pre["pass"]:
        return

    joint_axes = phase28.amass_adapter.parse_joint_axes(phase28.OFFICIAL_MJCF)
    outputs, rows = {}, []
    for index, motion_id in enumerate(MOTION_IDS, start=1):
        print(f"[phase29] optimize {index}/3 {motion_id}", flush=True)
        row = panel_rows[motion_id]
        original = cache[motion_id]
        original_audit = phase28.audit_tier(row, original, model, reset, gates, mirror)
        repaired, diagnostics = solve_motion(row, original, model, reset, gates, mirror, joint_axes)
        repaired["source_sha256"] = row["source_sha256"]
        repaired["recommended_split"] = row["recommended_split"]
        repair_audit = phase28.audit_tier(row, repaired, model, reset, gates, mirror)
        outputs[motion_id] = repaired
        rows.append({
            "id": motion_id, "source_path": row["source_path"], "source_sha256": row["source_sha256"],
            "split": row["recommended_split"], "frames": len(original["dof"]),
            "original_entry_sha256": phase28.array_hash(original), "repair_entry_sha256": phase28.array_hash(repaired),
            "original_audit": original_audit, "repair_audit": repair_audit,
            "diagnostics": diagnostics,
            "repair_entry_contract": repaired["phase29_full_trajectory_repair"],
        })
        print(f"[phase29] result {motion_id} {original_audit['tier']}->{repair_audit['tier']}", flush=True)

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(outputs, args.output_cache, compress=True)
    silver = [row["id"] for row in rows if row["repair_audit"]["tier"] == "Silver"]
    freeze = bool(silver)
    conclusion = (
        f"统一全轨迹表示在 {len(silver)}/3 条 train 动作产生 Silver，存在性门通过；只冻结该离线 repair contract，仍不授权 physics/PPO。"
        if freeze else
        "统一全轨迹表示在三条固定 train 动作上仍未产生 Silver；按预注册门停止，不扩大权重、动作或参数搜索。这只否定当前生成器/配置，不否定 X2、Any2Any 或闭环策略。"
    )
    report = {
        "schema_version": "x2_wbt_full_trajectory_repair_phase29_v1",
        "mode": "offline_sparse_full_trajectory_FK_no_physics_no_training",
        "truth_boundary": {"contact_and_fk_are_model_estimates": True, "not_real_grf_cop_force": True, "physics_steps": 0, "ppo_updates": 0, "policy_optimizer_steps": 0, "policy_forwards": 0, "offline_trajectory_solver_iterations": 3*GN_ITERATIONS, "real_robot": False},
        "provenance": {
            "phase28_cache": {"path": str(PHASE28_CACHE), "sha256": phase28.sha256(PHASE28_CACHE)},
            "panel": {"path": str(phase28.PANEL), "sha256": phase28.sha256(phase28.PANEL)},
            "tier_gates": {"path": str(phase28.TIER_GATES), "sha256": phase28.sha256(phase28.TIER_GATES)},
            "official_mjcf": {"path": str(phase28.OFFICIAL_MJCF), "sha256": phase28.sha256(phase28.OFFICIAL_MJCF)},
            "phase3_report": {"path": str(REPO/'reports/retarget/x2_contact_constrained_dynamic_teacher_phase3.json'), "sha256": phase28.sha256(REPO/'reports/retarget/x2_contact_constrained_dynamic_teacher_phase3.json')},
            "phase8_report": {"path": str(REPO/'reports/retarget/x2_wbt_stance_xy_lock_phase8.json'), "sha256": phase28.sha256(REPO/'reports/retarget/x2_wbt_stance_xy_lock_phase8.json')},
            "output_cache": {"path": str(args.output_cache)},
        },
        "configuration": configuration_manifest(), "preflight": pre, "motions": rows,
        "summary": {"completed": len(rows), "silver_count": len(silver), "silver_ids": silver, "bronze_count": sum(row["repair_audit"]["tier"] == "Bronze" for row in rows), "reject_count": sum(row["repair_audit"]["tier"] == "Reject" for row in rows), "total_wall_time_s": float(sum(row["diagnostics"]["wall_time_s"] for row in rows))},
        "decision": {
            "freeze_repair_contract": freeze, "minimum_one_train_silver": freeze,
            "conclusion": conclusion,
            "next_step": "Stop for review; do not automatically run physics or PPO." if freeze else "Stop this generator. Do not tune weights per clip or expand the search.",
        },
    }
    report["provenance"]["output_cache"]["sha256"] = phase28.sha256(args.output_cache)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report_md.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
