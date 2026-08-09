#!/usr/bin/env python3
"""Phase32 train-only gate-constrained full-WBT29 trajectory feasibility test.

This is a sequential linearized feasibility projection, not a reward-weight
scan.  The immutable Phase29 repaired paths initialize full WBT29 joint + root
variables (head remains locked).  Bronze/Silver gate bounds are encoded as
projection limits or active linearized inequalities.  One configuration is
used for the same three train motions and held-out files/reports are never read.

All contact/FK values are model estimates, not real GRF/COP/foot-force truth.
There are zero physics integration steps, policy forwards, PPO updates, or
real-robot operations.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.sparse.linalg import lsqr


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29


MOTION_IDS = phase29.MOTION_IDS
OUTPUT_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "silver_contact/phase32_gate_constrained_fullbody"
)
OUTPUT_CACHE = OUTPUT_ROOT / "x2_phase32_gate_constrained_fullbody.pkl"
REPORT_JSON = REPO / "reports/retarget/x2_wbt_gate_constrained_fullbody_phase32.json"
REPORT_MD = REPO / "reports/retarget/x2_wbt_gate_constrained_fullbody_phase32.md"
PREFLIGHT_JSON = REPO / "reports/retarget/x2_wbt_gate_constrained_fullbody_phase32_preflight.json"
SHARD_ROOT = OUTPUT_ROOT / "worker_shards"

# Frozen feasibility configuration, identical for every motion.
OUTER_ITERATIONS = 12
LINEAR_STEP_SCALE = 0.65
LSQR_ITER_LIMIT = 320
JOINT_STEP_HARD_RAD = 0.095
ROOT_ACCEL_HARD_MPS2 = 3.80
ROOT_XY_CORRECTION_MAX_M = 0.20
ROOT_Z_CORRECTION_MAX_M = 0.08
STANCE_ANCHOR_TOL_M = 0.006
KEYPOINT_P95_TARGET_M = 0.095
SWING_CLEARANCE_TARGET_M = 0.012
JOINT_PROJECTION_SWEEPS = 80
ROOT_PROJECTION_SWEEPS = 100
ROOT_TRUST_STEP_M = 0.030
JOINT_TRUST_STEP_RAD = 0.080


def configuration_manifest() -> dict[str, Any]:
    return {
        "motions": list(MOTION_IDS), "train_only": True, "held_out_read": False,
        "initialization": "immutable Phase29 repaired spatial trajectories",
        "variables": "root XYZ + all WBT29 joints + phase-level stance anchor XY; head2 excluded and locked",
        "solver": "sequential sparse linearized feasibility projection with LSQR minimum-norm step",
        "soft_reward_weight_scan": False, "per_clip_tuning": False,
        "outer_iterations": OUTER_ITERATIONS, "linear_step_scale": LINEAR_STEP_SCALE,
        "lsqr_iteration_limit": LSQR_ITER_LIMIT,
        "hard_projection": {
            "joint_step_rad": JOINT_STEP_HARD_RAD,
            "root_acceleration_mps2": ROOT_ACCEL_HARD_MPS2,
            "root_xy_correction_m": ROOT_XY_CORRECTION_MAX_M,
            "root_z_correction_m": ROOT_Z_CORRECTION_MAX_M,
            "joint_limits": "official model hard ranges",
            "head_lock": True,
        },
        "linearized_constraints": {
            "stance_anchor_xy_tolerance_m": STANCE_ANCHOR_TOL_M,
            "stance_contact_distance": "official reset clearance",
            "swing_clearance_m": SWING_CLEARANCE_TARGET_M,
            "tracked_keypoint_norm_m": KEYPOINT_P95_TARGET_M,
            "contact_timing_and_flight": "source phase schedule with exactly one carried support in source-flight gaps",
        },
        "forbidden": ["held_metric_read", "per_clip_tuning", "frame_or_segment_deletion", "fixed_root", "forced_double_contact", "physics", "PPO"],
    }


def full_wbt29_indices(names: list[str]) -> np.ndarray:
    return np.asarray([index for index, name in enumerate(names) if not name.startswith("head_")], dtype=np.int64)


def desired_contact_schedule(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    """Preserve source labels and fill flight gaps with one, never two, support feet."""
    result = {side: np.asarray(contract["contact"][side], dtype=bool).copy() for side in phase29.SIDES}
    source_fit = np.asarray(contract["fitted_keypoints"], dtype=np.float64)
    previous: str | None = None
    for frame in range(len(result["left"])):
        active = [side for side in phase29.SIDES if result[side][frame]]
        if len(active) == 1:
            previous = active[0]
        elif len(active) == 0:
            if previous is None:
                previous = "left" if source_fit[frame, 2, 2] <= source_fit[frame, 3, 2] else "right"
            result[previous][frame] = True
    return result


def project_joint_path(
    q: np.ndarray, model: mujoco.MjModel, names: list[str], controlled: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    ranges = []
    for index in controlled:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[int(index)])
        ranges.append(model.jnt_range[jid].copy())
    ranges = np.asarray(ranges)
    values = q[:, controlled]
    for _ in range(JOINT_PROJECTION_SWEEPS):
        values[:] = np.clip(values, ranges[:, 0], ranges[:, 1])
        for direction in (range(1, len(values)), range(len(values)-1, 0, -1)):
            for frame in direction:
                previous = frame-1
                difference = values[frame]-values[previous]
                excess = np.maximum(np.abs(difference)-JOINT_STEP_HARD_RAD, 0.0) * np.sign(difference) * 0.5
                values[frame] -= excess
                values[previous] += excess
    q[:, controlled] = np.clip(values, ranges[:, 0], ranges[:, 1])
    return q


def project_root_path(root: np.ndarray, base_root: np.ndarray, fps: float) -> np.ndarray:
    root = np.asarray(root, dtype=np.float64).copy()
    acceleration_step = ROOT_ACCEL_HARD_MPS2 / (fps*fps)
    for _ in range(ROOT_PROJECTION_SWEEPS):
        correction = root-base_root
        norm = np.linalg.norm(correction[:, :2], axis=1)
        scale = np.minimum(1.0, ROOT_XY_CORRECTION_MAX_M/np.maximum(norm, 1e-12))
        correction[:, :2] *= scale[:, None]
        correction[:, 2] = np.clip(correction[:, 2], -ROOT_Z_CORRECTION_MAX_M, ROOT_Z_CORRECTION_MAX_M)
        root[:] = base_root+correction
        for frame in range(1, len(root)-1):
            second = root[frame-1, :2]-2.0*root[frame, :2]+root[frame+1, :2]
            magnitude = np.linalg.norm(second)
            if magnitude > acceleration_step:
                projected = second*(acceleration_step/magnitude)
                root[frame, :2] += 0.5*(second-projected)
    correction = root-base_root
    norm = np.linalg.norm(correction[:, :2], axis=1)
    scale = np.minimum(1.0, ROOT_XY_CORRECTION_MAX_M/np.maximum(norm, 1e-12))
    correction[:, :2] *= scale[:, None]
    correction[:, 2] = np.clip(correction[:, 2], -ROOT_Z_CORRECTION_MAX_M, ROOT_Z_CORRECTION_MAX_M)
    return base_root+correction


def make_candidate(initial: dict[str, Any], root: np.ndarray, q: np.ndarray, axes: np.ndarray) -> dict[str, Any]:
    result = copy.deepcopy(initial)
    result["root_trans_offset"] = root.astype(np.asarray(initial["root_trans_offset"]).dtype)
    result["dof"] = q.astype(np.asarray(initial["dof"]).dtype)
    pose = np.zeros((len(q), len(result["joint_names_mujoco"])+1, 3), dtype=np.float32)
    pose[:, 1:, :] = q[:, :, None].astype(np.float32)*axes[None]
    result["pose_aa"] = pose
    return result


def kinematic_projection_system(
    model: mujoco.MjModel, candidate: dict[str, Any], contract: dict[str, Any],
    desired_contact: dict[str, np.ndarray], reset: dict[str, Any], controlled: np.ndarray,
    anchors: dict[str, np.ndarray], phases: dict[str, list[tuple[int, int]]],
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    names = list(candidate["joint_names_mujoco"])
    qpos_addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
    dof_addresses = np.asarray([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[int(index)])]) for index in controlled], dtype=np.int64)
    layout = phase29.Layout(len(candidate["dof"]), len(controlled), phases)
    system = phase29.SparseSystem(layout.size)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in phase28.TARGET_BODIES]
    floor, feet = phase7.active_sole_spheres(model)
    data = mujoco.MjData(model)
    source_fit = np.asarray(contract["fitted_keypoints"], dtype=np.float64)
    phase_lookup = {side: np.full(layout.frames, -1, dtype=np.int64) for side in phase29.SIDES}
    for side in phase29.SIDES:
        for phase_index, (start, end) in enumerate(phases[side]):
            phase_lookup[side][start:end] = phase_index
    active_counts = {name: 0 for name in ("keypoint", "stance_xy", "stance_z", "swing_clearance")}
    contact_distance = float(reset["reset_clearance_m"])
    threshold = float(reset["reset_clearance_m"]+reset["sole_sphere_radius_m"])

    for frame in range(layout.frames):
        state = phase29.frame_state(model, data, candidate, frame, qpos_addresses, body_ids, feet, floor, dof_addresses)
        columns = layout.frame_cols(frame)
        for point_index in range(len(phase28.TARGET_BODIES)):
            error = state["points"][point_index]-source_fit[frame, point_index]
            magnitude = float(np.linalg.norm(error))
            if magnitude > KEYPOINT_P95_TARGET_M:
                direction = error/max(magnitude, 1e-12)
                jac = direction @ state["point_jacobians"][point_index]
                phase29.SparseSystem.add(system, phase29.dictionary_row(columns, jac/KEYPOINT_P95_TARGET_M), (magnitude-KEYPOINT_P95_TARGET_M)/KEYPOINT_P95_TARGET_M, 1.0)
                active_counts["keypoint"] += 1
        for side, point_index in (("left", 2), ("right", 3)):
            if desired_contact[side][frame]:
                phase_index = int(phase_lookup[side][frame])
                if phase_index < 0:
                    continue
                anchor_col = layout.anchor_col(side, phase_index)
                error = state["points"][point_index, :2]-anchors[side][phase_index]
                magnitude = float(np.linalg.norm(error))
                if magnitude > STANCE_ANCHOR_TOL_M:
                    direction = error/max(magnitude, 1e-12)
                    jac = direction @ state["point_jacobians"][point_index][:2]
                    coefficients = phase29.dictionary_row(columns, jac/STANCE_ANCHOR_TOL_M)
                    coefficients[anchor_col] = coefficients.get(anchor_col, 0.0)-direction[0]/STANCE_ANCHOR_TOL_M
                    coefficients[anchor_col+1] = coefficients.get(anchor_col+1, 0.0)-direction[1]/STANCE_ANCHOR_TOL_M
                    system.add(coefficients, (magnitude-STANCE_ANCHOR_TOL_M)/STANCE_ANCHOR_TOL_M, 1.0)
                    active_counts["stance_xy"] += 1
                z_error = state["sole"][side]["distance"]-contact_distance
                if abs(z_error) > 0.001:
                    system.add(phase29.dictionary_row(columns, state["sole"][side]["jacobian_z"]/0.005), z_error/0.005, 1.0)
                    active_counts["stance_z"] += 1
            else:
                desired = threshold+SWING_CLEARANCE_TARGET_M
                violation = desired-state["sole"][side]["distance"]
                if violation > 0.0:
                    system.add(phase29.dictionary_row(columns, -state["sole"][side]["jacobian_z"]/0.01), violation/0.01, 1.0)
                    active_counts["swing_clearance"] += 1
    matrix, rhs = system.matrix()
    return matrix, rhs, {"active_constraints": active_counts, "rows": matrix.shape[0], "columns": matrix.shape[1], "nnz": matrix.nnz}


def max_projection_violations(
    q: np.ndarray, root: np.ndarray, base_root: np.ndarray, fps: float,
) -> dict[str, float]:
    step = float(np.max(np.abs(np.diff(q[:, :29], axis=0)))) if len(q)>1 else 0.0
    acceleration = np.diff(root[:, :2], n=2, axis=0)*(fps**2) if len(root)>2 else np.zeros((1, 2))
    return {
        "joint_step_max_rad": step,
        "joint_step_hard_excess_rad": max(0.0, step-JOINT_STEP_HARD_RAD),
        "root_acceleration_max_mps2": float(np.max(np.linalg.norm(acceleration, axis=1))),
        "root_acceleration_hard_excess_mps2": max(0.0, float(np.max(np.linalg.norm(acceleration, axis=1)))-ROOT_ACCEL_HARD_MPS2),
        "root_xy_correction_max_m": float(np.max(np.linalg.norm(root[:, :2]-base_root[:, :2], axis=1))),
    }


def solve_motion(
    row: dict[str, Any], original: dict[str, Any], initial: dict[str, Any],
    model: mujoco.MjModel, reset: dict[str, Any], gates: dict[str, Any],
    mirror: dict[str, Any], axes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    names = list(initial["joint_names_mujoco"])
    controlled = full_wbt29_indices(names)
    contract = phase29.source_contract(row, initial, model)
    desired = desired_contact_schedule(contract)
    phases = {side: phase28.contiguous_true(desired[side]) for side in phase29.SIDES}
    q = np.asarray(initial["dof"], dtype=np.float64).copy()
    root = np.asarray(initial["root_trans_offset"], dtype=np.float64).copy()
    base_root = np.asarray(original["root_trans_offset"], dtype=np.float64)
    # Initialize phase anchors from the Phase29 path, then optimize them jointly.
    kin = phase28.target_kinematics(initial, model)["target_points"]
    anchors = {side: np.asarray([np.mean(kin[start:end, 2 if side=="left" else 3, :2], axis=0) for start, end in phases[side]], dtype=np.float64).reshape(-1, 2) for side in phase29.SIDES}
    iterations = []
    candidate = make_candidate(initial, root, q, axes)
    for iteration in range(OUTER_ITERATIONS):
        iter_started = time.perf_counter()
        q = project_joint_path(q, model, names, controlled)
        q[:, -2:] = 0.0
        root = project_root_path(root, base_root, float(initial["fps"]))
        candidate = make_candidate(initial, root, q, axes)
        matrix, rhs, active = kinematic_projection_system(model, candidate, contract, desired, reset, controlled, anchors, phases)
        if matrix.shape[0] == 0:
            solution = np.zeros(matrix.shape[1], dtype=np.float64)
            lsqr_iterations, residual = 0, 0.0
        else:
            solved = lsqr(matrix, rhs, atol=1e-6, btol=1e-6, iter_lim=LSQR_ITER_LIMIT, damp=1e-4)
            solution, lsqr_iterations, residual = np.asarray(solved[0]), int(solved[2]), float(solved[3])
        layout = phase29.Layout(len(q), len(controlled), phases)
        step = LINEAR_STEP_SCALE*solution[:layout.trajectory_size].reshape(len(q), layout.frame_width)
        root_step = np.clip(step[:, :3], -ROOT_TRUST_STEP_M, ROOT_TRUST_STEP_M)
        joint_step = np.clip(step[:, 3:], -JOINT_TRUST_STEP_RAD, JOINT_TRUST_STEP_RAD)
        root += root_step
        q[:, controlled] += joint_step
        for side in phase29.SIDES:
            for phase_index in range(len(phases[side])):
                column = layout.anchor_col(side, phase_index)
                anchors[side][phase_index] += LINEAR_STEP_SCALE*solution[column:column+2]
        q = project_joint_path(q, model, names, controlled)
        q[:, -2:] = 0.0
        root = project_root_path(root, base_root, float(initial["fps"]))
        candidate = make_candidate(initial, root, q, axes)
        iterations.append({
            "iteration": iteration+1, "active": active,
            "lsqr_iterations": lsqr_iterations, "lsqr_residual_norm": residual,
            "projection": max_projection_violations(q, root, base_root, float(initial["fps"])),
            "tier": None, "bronze_pass": None, "silver_pass": None,
            "audit_boundary": "full Bronze/Silver audit intentionally runs once after all 12 iterations to avoid repeated SMPL/AMASS loader memory growth",
            "elapsed_s": float(time.perf_counter()-iter_started),
        })
        print(f"[phase32] {row['id']} iter={iteration+1}/{OUTER_ITERATIONS} active={active['rows']}", flush=True)
        del matrix, rhs, solution
        gc.collect()
    candidate["phase32_gate_constrained_fullbody"] = {
        "configuration": configuration_manifest(),
        "controlled_joint_names": [names[int(index)] for index in controlled],
        "head_locked_joint_names": names[-2:],
        "desired_contact_phases": {side: [[int(a), int(b)] for a, b in phases[side]] for side in phase29.SIDES},
        "source_flight_filled_with_single_support_fraction": float(np.mean(~contract["contact"]["left"] & ~contract["contact"]["right"])),
        "forced_double_contact": False,
        "root_fixed": False,
        "frame_or_segment_deleted": False,
        "contact_truth_boundary": "model/source intent; not real GRF/COP/force",
    }
    diagnostics = {
        "frames": len(q), "controlled_joint_count": len(controlled),
        "iterations": iterations, "final_projection": max_projection_violations(q, root, base_root, float(initial["fps"])),
        "root_xy_bound_respected": bool(np.max(np.linalg.norm(root[:, :2]-base_root[:, :2], axis=1)) <= ROOT_XY_CORRECTION_MAX_M+1e-9),
        "head_lock_respected": bool(np.max(np.abs(q[:, -2:])) <= 1e-12),
        "no_frame_or_segment_deleted": len(q) == len(initial["dof"]),
        "wall_time_s": float(time.perf_counter()-started),
    }
    return candidate, diagnostics


def preflight(rows: dict[str, dict[str, Any]], original_cache: dict[str, Any], initial_cache: dict[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
    records = []
    for motion_id in MOTION_IDS:
        row, original, initial = rows.get(motion_id), original_cache.get(motion_id), initial_cache.get(motion_id)
        names = list(initial["joint_names_mujoco"]) if initial else []
        checks = {
            "row_present": row is not None, "train_split": row is not None and row["recommended_split"] == "train_candidate",
            "phase28_original_present": original is not None, "phase29_initial_present": initial is not None,
            "source_sha_exact": row is not None and original is not None and initial is not None and original.get("source_sha256") == row["source_sha256"] and initial.get("source_sha256") == row["source_sha256"],
            "wbt29_plus_head2": len(names) == 31 and len(full_wbt29_indices(names)) == 29 and names[-2:] == ["head_yaw_joint", "head_pitch_joint"],
            "head_locked": initial is not None and float(np.max(np.abs(np.asarray(initial["dof"])[:, -2:]))) <= 1e-12,
        }
        records.append({"id": motion_id, "checks": checks, "pass": bool(all(checks.values())), "frames": len(initial["dof"]) if initial else None})
    dependency = {"scipy_sparse_lsqr": True, "mujoco_jacBody": hasattr(mujoco, "mj_jacBody"), "mujoco_jacGeom": hasattr(mujoco, "mj_jacGeom")}
    expressible = {
        "joint_step_and_limits_projectable": True, "root_acceleration_and_xy_projectable": True,
        "contact_keypoint_clearance_linearizable": all(dependency.values()),
        "percentile_gates_finally_audited_not_approximated_as_reward": True,
    }
    return {"schema_version": "x2_wbt_gate_constrained_fullbody_phase32_preflight_v1", "held_out_paths_or_reports_read": False, "configuration": configuration_manifest(), "dependency": dependency, "constraint_expressibility": expressible, "motions": records, "pass": bool(all(dependency.values()) and all(expressible.values()) and all(record["pass"] for record in records))}


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# X2 WBT Phase32：train-only gate-constrained full-WBT29 feasibility", "", "## 裁决", "",
        f"- train Silver：`{report['summary']['silver_count']}/3`；未来重新评 held：`{report['decision']['heldout_reevaluation_allowed']}`（硬门要求2/3）。",
        "- full WBT29＋root参与；head2锁定。硬投影覆盖joint limits/step、root acceleration和rootXY，FK约束用逐轮激活线性化，最终仍由原Bronze/Silver门裁决。",
        "- 没有soft reward扫权、逐clip调参、删段、fixed root、forced double contact、physics或PPO。", "",
        "## 结果", "", "| motion | Phase29→Phase32 | qstep p95/max | root acc | speed L/R | timing L/R | rootXY max | rejects |", "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["motions"]:
        b, s = row["phase32_audit"]["bronze"]["metrics"], row["phase32_audit"]["silver"]["metrics"]
        lines.append(f"| `{row['id']}` | {row['phase29_audit']['tier']}→{row['phase32_audit']['tier']} | {b['joint_step_p95_rad']:.4f}/{b['joint_step_max_rad']:.4f} | {s['root_horizontal_acceleration_p95_mps2']:.3f} | {s['stance_speed_p95_mps']['left']:.3f}/{s['stance_speed_p95_mps']['right']:.3f} | {s['contact_timing_error_p95_s']['left']:.3f}/{s['contact_timing_error_p95_s']['right']:.3f} | {row['diagnostics']['final_projection']['root_xy_correction_max_m']:.3f} | `{' ; '.join(row['phase32_audit']['reject_reasons'])}` |")
    lines += ["", "## 结论", "", report["decision"]["conclusion"], "", "## 下一步", "", report["decision"]["next_step"]]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--motion-id", choices=MOTION_IDS)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--shard-root", type=Path, default=SHARD_ROOT)
    parser.add_argument("--output-cache", type=Path, default=OUTPUT_CACHE)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()
    panel = json.loads(phase28.PANEL.read_text())
    rows = {row["id"]: row for row in panel["motions"] if row["id"] in MOTION_IDS}
    # Deliberately only the train initialization caches; no Phase31/held report is opened.
    original_cache = joblib.load(phase29.PHASE28_CACHE)
    initial_cache = joblib.load(phase29.OUTPUT_CACHE)
    gates, mirror = json.loads(phase28.TIER_GATES.read_text()), json.loads(phase28.MIRROR_CONTRACT.read_text())
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase32_offline")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)
    axes = phase28.amass_adapter.parse_joint_axes(phase28.OFFICIAL_MJCF)
    pre = preflight(rows, original_cache, initial_cache, model)
    PREFLIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(phase28.json_safe(pre), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"[phase32] preflight pass={pre['pass']} train={sum(row['pass'] for row in pre['motions'])}/3", flush=True)
    if args.preflight_only or not pre["pass"]:
        return
    outputs, records = {}, []
    if args.assemble_only:
        for motion_id in MOTION_IDS:
            shard_path = args.shard_root / f"{motion_id}.joblib"
            if not shard_path.is_file():
                raise FileNotFoundError(f"missing frozen worker shard: {shard_path}")
            shard = joblib.load(shard_path)
            if shard["configuration"] != configuration_manifest() or shard["id"] != motion_id:
                raise RuntimeError(f"worker shard contract mismatch: {motion_id}")
            outputs[motion_id] = shard["candidate"]
            records.append(shard["record"])
    else:
        selected = (args.motion_id,) if args.motion_id else MOTION_IDS
        for motion_id in selected:
            row, original, initial = rows[motion_id], original_cache[motion_id], initial_cache[motion_id]
            phase29_audit = phase28.audit_tier(row, initial, model, reset, gates, mirror)
            candidate, diagnostics = solve_motion(row, original, initial, model, reset, gates, mirror, axes)
            candidate["source_sha256"], candidate["recommended_split"] = row["source_sha256"], row["recommended_split"]
            audit = phase28.audit_tier(row, candidate, model, reset, gates, mirror)
            record = {"id": motion_id, "split": row["recommended_split"], "source_sha256": row["source_sha256"], "phase29_entry_sha256": phase28.array_hash(initial), "phase32_entry_sha256": phase28.array_hash(candidate), "phase29_audit": phase29_audit, "phase32_audit": audit, "diagnostics": diagnostics}
            outputs[motion_id] = candidate
            records.append(record)
            print(f"[phase32] result {motion_id} {phase29_audit['tier']}->{audit['tier']}", flush=True)
            if args.motion_id:
                args.shard_root.mkdir(parents=True, exist_ok=True)
                shard_path = args.shard_root / f"{motion_id}.joblib"
                joblib.dump({"schema_version": "phase32_worker_shard_v1", "id": motion_id, "configuration": configuration_manifest(), "candidate": candidate, "record": record}, shard_path, compress=True)
                print(f"[phase32] worker shard={shard_path}", flush=True)
                return
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(outputs, args.output_cache, compress=True)
    silver = [row["id"] for row in records if row["phase32_audit"]["tier"] == "Silver"]
    allowed = len(silver) >= 2
    conclusion = f"统一硬门full-WBT29求解在 {len(silver)}/3 条train产生Silver；" + ("达到2/3，未来可单独授权重评held，但本阶段不读取held。" if allowed else "未达到2/3，按门停止且held仍锁定。")
    report = {"schema_version": "x2_wbt_gate_constrained_fullbody_phase32_v1", "mode": "train_only_sequential_feasibility_no_physics_no_PPO", "truth_boundary": {"held_out_read": False, "contact_and_fk_are_model_estimates": True, "not_real_grf_cop_force": True, "physics_steps": 0, "ppo_updates": 0, "policy_forwards": 0, "real_robot": False}, "configuration": configuration_manifest(), "preflight": pre, "provenance": {"phase28_original_cache": {"path": str(phase29.PHASE28_CACHE), "sha256": phase28.sha256(phase29.PHASE28_CACHE)}, "phase29_initial_cache": {"path": str(phase29.OUTPUT_CACHE), "sha256": phase28.sha256(phase29.OUTPUT_CACHE)}, "tier_gates": {"path": str(phase28.TIER_GATES), "sha256": phase28.sha256(phase28.TIER_GATES)}, "output_cache": {"path": str(args.output_cache)}}, "motions": records, "summary": {"completed": len(records), "silver_count": len(silver), "silver_ids": silver, "bronze_count": sum(row["phase32_audit"]["tier"]=="Bronze" for row in records), "reject_count": sum(row["phase32_audit"]["tier"]=="Reject" for row in records)}, "decision": {"heldout_reevaluation_allowed": allowed, "required_train_silver": 2, "conclusion": conclusion, "next_step": "Stop for review; do not automatically read/re-evaluate held-out or run physics/PPO." if allowed else "Stop this configuration. Do not retune constraints or read held-out."}}
    report["provenance"]["output_cache"]["sha256"] = phase28.sha256(args.output_cache)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_md(report, args.report_md)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
