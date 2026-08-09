#!/usr/bin/env python3
"""Phase37 single-config official-collision contact repair existence test.

The frozen Phase30 PHUMA-LUNGE-R path and 1.46x timing are the only input.
Root XY/orientation, upper body, source contact intent, frames and fps remain
immutable.  A single sparse full-trajectory Gauss--Newton configuration may
change only root-z and the 15 waist/lower-body joints.  Intended stance targets
official x2.xml sphere-floor signed distance zero; every frame is constrained
against penetration and source-intended swing clearance loss.

This is an offline trajectory solve, not a policy optimizer.  It performs no
MuJoCo integration, PPO, policy forward, checkpoint, or robot operation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
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
import retarget.run_x2_wbt_time_dilation_phase30 as phase30


MOTION_ID = "PHUMA-LUNGE-R-001"
PHASE36_REPORT = REPO / "reports/retarget/x2_wbt_tier_provenance_correction_phase36.json"
OUTPUT_DIR = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase37_official_contact_repair"
)
OUTPUT_CACHE = OUTPUT_DIR / "x2_phase37_official_contact_repair.pkl"
OUTPUT_PREFLIGHT = REPO / "reports/retarget/x2_wbt_official_contact_repair_phase37_preflight.json"
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_official_contact_repair_phase37.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_official_contact_repair_phase37.md"

# Frozen before the solve; never adapted to intermediate/final metrics.
ITERATIONS = 12
STEP_SCALE = 0.65
LSQR_ATOL = 1.0e-7
LSQR_BTOL = 1.0e-7
LSQR_ITER_LIMIT = 320
ROOT_Z_CORRECTION_MAX_M = 0.08
JOINT_CORRECTION_MAX_RAD = 0.45
CONTACT_EQUALITY_TARGET_M = 0.0
CONTACT_EQUALITY_TOLERANCE_M = 5.0e-4
NONPENETRATION_TOLERANCE_M = 1.0e-5
WEIGHTS = {
    "stance_contact_equality": 160.0,
    "nonpenetration_hinge": 220.0,
    "swing_clearance_hinge": 100.0,
    "keypoint_preservation": 4.0,
    "root_z_correction": 1.5,
    "joint_correction": 1.0,
    "correction_velocity": 10.0,
    "correction_acceleration": 18.0,
}


@dataclass(frozen=True)
class Layout:
    frames: int
    lower_count: int

    @property
    def width(self) -> int:
        return 1 + self.lower_count

    @property
    def size(self) -> int:
        return self.frames * self.width

    def columns(self, frame: int) -> np.ndarray:
        start = frame * self.width
        return np.arange(start, start + self.width, dtype=np.int64)


def configuration() -> dict[str, Any]:
    return {
        "motion_id": MOTION_ID,
        "input": "frozen Phase30 1.46x PHUMA-LUNGE-R Bronze after Phase36 correction",
        "iterations": ITERATIONS,
        "step_scale": STEP_SCALE,
        "lsqr": {"atol": LSQR_ATOL, "btol": LSQR_BTOL, "iter_limit": LSQR_ITER_LIMIT},
        "variables": ["root_z", "waist_and_lower_body_15DoF_continuous_trajectory"],
        "frozen": ["root_xy", "root_orientation", "upper_body", "head", "fps_30", "frames_175", "time_dilation_1.46", "source_contact_intent", "motion_semantics"],
        "stance_foot_orientation_variable": False,
        "root_z_correction_max_abs_m": ROOT_Z_CORRECTION_MAX_M,
        "joint_correction_max_abs_rad": JOINT_CORRECTION_MAX_RAD,
        "official_contact_equation": "minimum signed surface distance across official 12 active sole spheres per foot == 0 m during source-intended stance",
        "all_frame_nonpenetration": f"signed distance >= -{NONPENETRATION_TOLERANCE_M:g} m",
        "equality_acceptance": f"stance abs signed distance p95/max <= {CONTACT_EQUALITY_TOLERANCE_M:g} m",
        "weights": WEIGHTS,
        "per_clip_or_result_tuning": False,
    }


def candidate_from_correction(
    original: dict[str, Any], root_z_delta: np.ndarray, joint_delta: np.ndarray,
    lower_indices: np.ndarray, joint_axes: np.ndarray,
) -> dict[str, Any]:
    result = copy.deepcopy(original)
    root = np.asarray(original["root_trans_offset"], dtype=np.float64).copy()
    root[:, 2] += root_z_delta
    dof = np.asarray(original["dof"], dtype=np.float64).copy()
    dof[:, lower_indices] += joint_delta
    result["root_trans_offset"] = root.astype(np.asarray(original["root_trans_offset"]).dtype)
    result["dof"] = dof.astype(np.asarray(original["dof"]).dtype)
    pose = np.zeros((len(dof), len(result["joint_names_mujoco"]) + 1, 3), dtype=np.float32)
    pose[:, 1:] = dof[:, :, None].astype(np.float32) * joint_axes[None]
    result["pose_aa"] = pose
    return result


def project(
    model: mujoco.MjModel, original: dict[str, Any], root_z_delta: np.ndarray,
    joint_delta: np.ndarray, lower_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root_z_delta[:] = np.clip(root_z_delta, -ROOT_Z_CORRECTION_MAX_M, ROOT_Z_CORRECTION_MAX_M)
    joint_delta[:] = np.clip(joint_delta, -JOINT_CORRECTION_MAX_RAD, JOINT_CORRECTION_MAX_RAD)
    original_q = np.asarray(original["dof"], dtype=np.float64)
    names = list(original["joint_names_mujoco"])
    for local, entry_index in enumerate(lower_indices):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[int(entry_index)])
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
            joint_delta[:, local] = np.clip(
                joint_delta[:, local], low - original_q[:, entry_index], high - original_q[:, entry_index]
            )
    return root_z_delta, joint_delta


def frame_state(
    model: mujoco.MjModel, data: mujoco.MjData, entry: dict[str, Any], frame: int,
    qpos_addresses: list[int], body_ids: list[int], feet: dict[str, list[int]], floor: int,
    variable_dof_addresses: np.ndarray,
) -> dict[str, Any]:
    phase28.set_entry_frame(model, data, entry, frame, qpos_addresses)
    points, point_jacobians = [], []
    for body_id in body_ids:
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        points.append(data.xpos[body_id].copy())
        point_jacobians.append(jacp[:, variable_dof_addresses].copy())
    fromto = np.zeros(6, dtype=np.float64)
    sole = {}
    for side, geom_ids in feet.items():
        distances = np.asarray([
            mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto) for geom in geom_ids
        ], dtype=np.float64)
        active = int(geom_ids[int(np.argmin(distances))])
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(model, data, jacp, jacr, active)
        sole[side] = {
            "distance": float(np.min(distances)),
            "jacobian_z": jacp[2, variable_dof_addresses].copy(),
        }
    return {"points": np.asarray(points), "point_jacobians": np.asarray(point_jacobians), "sole": sole}


def build_system(
    model: mujoco.MjModel, original_points: np.ndarray, original_distance: dict[str, np.ndarray],
    candidate: dict[str, Any], intent: dict[str, np.ndarray], layout: Layout,
    qpos_addresses: list[int], body_ids: list[int], feet: dict[str, list[int]], floor: int,
    variable_dof_addresses: np.ndarray, root_z_delta: np.ndarray, joint_delta: np.ndarray,
) -> tuple[Any, np.ndarray, dict[str, float]]:
    system = phase29.SparseSystem(layout.size)
    data = mujoco.MjData(model)
    objective = {"stance_frames": 0, "penetration_frames": 0, "swing_hinge_frames": 0}
    correction = np.column_stack([root_z_delta, joint_delta])
    for frame in range(layout.frames):
        columns = layout.columns(frame)
        state = frame_state(
            model, data, candidate, frame, qpos_addresses, body_ids, feet, floor, variable_dof_addresses
        )
        for point_index in range(len(body_ids)):
            residual = state["points"][point_index] - original_points[frame, point_index]
            for axis in range(3):
                system.add(
                    phase29.dictionary_row(columns, state["point_jacobians"][point_index, axis]),
                    residual[axis], WEIGHTS["keypoint_preservation"],
                )
        for side in phase29.SIDES:
            distance = state["sole"][side]["distance"]
            jacobian = state["sole"][side]["jacobian_z"]
            if bool(intent[side][frame]):
                objective["stance_frames"] += 1
                system.add(
                    phase29.dictionary_row(columns, jacobian),
                    distance - CONTACT_EQUALITY_TARGET_M,
                    WEIGHTS["stance_contact_equality"],
                )
            else:
                desired = max(0.0, float(original_distance[side][frame]))
                if distance < desired:
                    objective["swing_hinge_frames"] += 1
                    system.add(
                        phase29.dictionary_row(columns, jacobian),
                        distance - desired,
                        WEIGHTS["swing_clearance_hinge"],
                    )
            if distance < 0.0:
                objective["penetration_frames"] += 1
                system.add(
                    phase29.dictionary_row(columns, jacobian),
                    distance,
                    WEIGHTS["nonpenetration_hinge"],
                )
        system.add({int(columns[0]): 1.0}, root_z_delta[frame], WEIGHTS["root_z_correction"])
        for local in range(layout.lower_count):
            system.add({int(columns[1 + local]): 1.0}, joint_delta[frame, local], WEIGHTS["joint_correction"])
    for frame in range(1, layout.frames):
        phase29.add_difference(
            system, layout.columns(frame), layout.columns(frame - 1),
            correction[frame], correction[frame - 1], WEIGHTS["correction_velocity"],
        )
    for frame in range(1, layout.frames - 1):
        phase29.add_second_difference(
            system, layout.columns(frame - 1), layout.columns(frame), layout.columns(frame + 1),
            correction[frame - 1], correction[frame], correction[frame + 1],
            WEIGHTS["correction_acceleration"],
        )
    matrix, rhs = system.matrix()
    objective.update({"rows": matrix.shape[0], "columns": matrix.shape[1], "nnz": matrix.nnz})
    return matrix, rhs, objective


def contact_constraint_metrics(
    entry: dict[str, Any], intent: dict[str, np.ndarray], model: mujoco.MjModel,
) -> dict[str, Any]:
    kin = phase28.target_kinematics(entry, model)
    distance = {side: np.asarray(kin["sole_distance"][side]) for side in phase29.SIDES}
    stance_abs = np.concatenate([np.abs(distance[side][intent[side]]) for side in phase29.SIDES])
    all_distances = np.concatenate([distance[side] for side in phase29.SIDES])
    swing_loss = []
    return {
        "stance_abs_signed_distance_p95_max_m": [
            float(np.percentile(stance_abs, 95)), float(np.max(stance_abs))
        ],
        "minimum_signed_distance_m": float(np.min(all_distances)),
        "penetrating_fraction_below_minus_tolerance": float(np.mean(all_distances < -NONPENETRATION_TOLERANCE_M)),
        "official_contact_ratio": {
            side: float(np.mean(kin["official_collision_contact"][side])) for side in phase29.SIDES
        },
        "collision_signed_distance_exact": bool(kin["collision_signed_distance_exact"]),
    }


def solve(
    original: dict[str, Any], row: dict[str, Any], model: mujoco.MjModel,
    intent: dict[str, np.ndarray], joint_axes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    names = list(original["joint_names_mujoco"])
    lower_indices = phase29.lower_body_indices(names)
    lower_dof_addresses = np.asarray([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[int(index)])]
        for index in lower_indices
    ], dtype=np.int64)
    variable_dof_addresses = np.concatenate([np.asarray([2], dtype=np.int64), lower_dof_addresses])
    layout = Layout(len(original["dof"]), len(lower_indices))
    qpos_addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in names
    ]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in phase28.TARGET_BODIES]
    floor, feet = phase7.active_sole_spheres(model)
    original_kin = phase28.target_kinematics(original, model)
    original_points = original_kin["target_points"]
    original_distance = original_kin["sole_distance"]
    root_z_delta = np.zeros(layout.frames, dtype=np.float64)
    joint_delta = np.zeros((layout.frames, layout.lower_count), dtype=np.float64)
    candidate = candidate_from_correction(original, root_z_delta, joint_delta, lower_indices, joint_axes)
    iterations = []
    for iteration in range(ITERATIONS):
        iter_started = time.perf_counter()
        matrix, rhs, objective = build_system(
            model, original_points, original_distance, candidate, intent, layout,
            qpos_addresses, body_ids, feet, floor, variable_dof_addresses,
            root_z_delta, joint_delta,
        )
        result = lsqr(matrix, rhs, atol=LSQR_ATOL, btol=LSQR_BTOL, iter_lim=LSQR_ITER_LIMIT, show=False)
        step = STEP_SCALE * np.asarray(result[0]).reshape(layout.frames, layout.width)
        root_z_delta += step[:, 0]
        joint_delta += step[:, 1:]
        root_z_delta, joint_delta = project(model, original, root_z_delta, joint_delta, lower_indices)
        candidate = candidate_from_correction(original, root_z_delta, joint_delta, lower_indices, joint_axes)
        constraints = contact_constraint_metrics(candidate, intent, model)
        iterations.append({
            "iteration": iteration + 1,
            "objective": objective,
            "lsqr_istop_iterations_residual": [int(result[1]), int(result[2]), float(result[3])],
            "step_l2": float(np.linalg.norm(step)),
            "root_z_correction_max_abs_m": float(np.max(np.abs(root_z_delta))),
            "joint_correction_max_abs_rad": float(np.max(np.abs(joint_delta))),
            "constraints": constraints,
            "elapsed_s": float(time.perf_counter() - iter_started),
        })
        print(
            f"[phase37] iter={iteration + 1}/{ITERATIONS} stance="
            f"{constraints['stance_abs_signed_distance_p95_max_m'][0]:.5f}/"
            f"{constraints['stance_abs_signed_distance_p95_max_m'][1]:.5f} "
            f"min={constraints['minimum_signed_distance_m']:.5f}", flush=True,
        )
    candidate["phase37_official_contact_repair"] = {
        "configuration": configuration(),
        "source_intent_phases": {
            side: [[int(start), int(end)] for start, end in phase28.contiguous_true(intent[side])]
            for side in phase29.SIDES
        },
        "lower_body_joint_names": [names[int(index)] for index in lower_indices],
        "root_z_correction_m": root_z_delta.astype(np.float32),
        "root_z_correction_max_abs_m": float(np.max(np.abs(root_z_delta))),
        "joint_correction_max_abs_rad": float(np.max(np.abs(joint_delta))),
        "root_xy_unchanged_exact": bool(np.array_equal(candidate["root_trans_offset"][:, :2], original["root_trans_offset"][:, :2])),
    }
    return candidate, {
        "iterations": iterations,
        "wall_time_s": float(time.perf_counter() - started),
        "final_constraints": contact_constraint_metrics(candidate, intent, model),
    }


def render(report: dict[str, Any]) -> str:
    before, after = report["before"], report["after"]
    constraints = report["solver"]["final_constraints"]
    lines = [
        "# X2 WBT Phase37：official-collision phase-aware contact repair", "",
        "## 裁决", "",
        f"- **{report['decision']['status']}**；corrected tier `{before['tier']} → {after['tier']}`。",
        "- 0 physics integration/PPO/policy optimizer；offline sparse trajectory LSQR固定12轮。",
        "- official contact严格为x2.xml 12 active sole spheres/foot signed distance<=0；未使用10.05mm Bronze容差。", "",
        "## 假设 / 干预 / 对照", "",
        "- 假设：只修改root-z与腰腿15DoF的连续全轨迹，可能使冻结source-intended stance满足真实sphere-floor contact，同时保留动作语义与Silver其余门。",
        "- 干预：stance distance=0等式、全帧不穿透、swing原clearance不下降；root XY/orientation、上肢、1.46x、175帧/30Hz均冻结。",
        "- 对照：Phase30经Phase36更正后的Bronze，同一auditor复评。", "",
        "## 结果", "",
        f"- stance |distance| p95/max：`{constraints['stance_abs_signed_distance_p95_max_m']}`m；全帧minimum={constraints['minimum_signed_distance_m']:.6f}m；penetration fraction={constraints['penetrating_fraction_below_minus_tolerance']:.6f}。",
        f"- official contact L/R：`{constraints['official_contact_ratio']}`；collision↔signed-distance exact：`{constraints['collision_signed_distance_exact']}`。",
        f"- correction root-z max={report['candidate_contract']['root_z_correction_max_abs_m']:.4f}m、joint max={report['candidate_contract']['joint_correction_max_abs_rad']:.4f}rad；root XY exact={report['candidate_contract']['root_xy_unchanged_exact']}。",
        f"- Silver failed：`{[name for name, value in after['silver_checks'].items() if not value]}`。", "",
        "## 结论", "", report["decision"]["conclusion"], "", "## 下一步", "", report["decision"]["next_step"], "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    phase36 = json.loads(PHASE36_REPORT.read_text())
    row = next(row for row in json.loads(phase28.PANEL.read_text())["motions"] if row["id"] == MOTION_ID)
    gates = json.loads(phase28.TIER_GATES.read_text())
    mirror = json.loads(phase28.MIRROR_CONTRACT.read_text())
    source30 = joblib.load(phase30.OUTPUT_CACHE)[MOTION_ID]
    source29 = joblib.load(phase29.OUTPUT_CACHE)[MOTION_ID]
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_phase37_static_fk_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)
    joint_axes = phase28.amass_adapter.parse_joint_axes(phase28.OFFICIAL_MJCF)
    source_contract = phase29.source_contract(row, source29, model)
    intent = {
        side: phase30.nearest_phase_resample(source_contract["contact"][side], len(source30["dof"]))
        for side in phase29.SIDES
    }
    corrected36 = next(row for row in phase36["phase30_corrected"] if row["id"] == MOTION_ID)
    lower = phase29.lower_body_indices(list(source30["joint_names_mujoco"]))
    preflight = {
        "schema_version": "x2_wbt_official_contact_repair_phase37_preflight_v1",
        "configuration": configuration(),
        "provenance": {
            "phase30_cache": {"path": str(phase30.OUTPUT_CACHE), "sha256": phase28.sha256(phase30.OUTPUT_CACHE)},
            "phase36_report": {"path": str(PHASE36_REPORT), "sha256": phase28.sha256(PHASE36_REPORT)},
            "tier_gates": {"path": str(phase28.TIER_GATES), "sha256": phase28.sha256(phase28.TIER_GATES)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase28.sha256(physics.DEFAULT_SCENE)},
        },
        "checks": {
            "phase36_corrected_bronze": corrected36["corrected"]["tier"] == "Bronze",
            "exact_motion_and_split": row["id"] == MOTION_ID and row["recommended_split"] == "train_candidate",
            "phase30_time_1p46_frozen": float(source30["phase30_time_dilation"]["requested_factor"]) == 1.46,
            "frames_fps_frozen": len(source30["dof"]) == 175 and float(source30["fps"]) == 30.0,
            "lower_body_15": len(lower) == 15,
            "both_intent_feet_have_phases": all(len(phase28.contiguous_true(intent[side])) > 0 for side in phase29.SIDES),
            "official_contact_threshold_zero": CONTACT_EQUALITY_TARGET_M == 0.0,
            "no_bronze_contact_tolerance": CONTACT_EQUALITY_TARGET_M != reset["reset_clearance_m"] + reset["sole_sphere_radius_m"],
            "no_held_out_or_parameter_scan": True,
        },
    }
    preflight["pass"] = bool(all(preflight["checks"].values()))
    OUTPUT_PREFLIGHT.write_text(json.dumps(phase28.json_safe(preflight), indent=2, ensure_ascii=False) + "\n")
    print(f"[phase37] preflight pass={preflight['pass']}", flush=True)
    if args.preflight_only or not preflight["pass"]:
        return
    before_audit = phase30.audit_with_frozen_contact(row, source30, model, reset, gates, mirror, intent)
    candidate, solver = solve(source30, row, model, intent, joint_axes)
    after_audit = phase30.audit_with_frozen_contact(row, candidate, model, reset, gates, mirror, intent)
    constraints = solver["final_constraints"]
    equality_pass = (
        constraints["stance_abs_signed_distance_p95_max_m"][0] <= CONTACT_EQUALITY_TOLERANCE_M
        and constraints["stance_abs_signed_distance_p95_max_m"][1] <= CONTACT_EQUALITY_TOLERANCE_M
    )
    nonpenetration_pass = constraints["penetrating_fraction_below_minus_tolerance"] == 0.0
    true_silver = bool(after_audit["tier"] == "Silver" and equality_pass and nonpenetration_pass)
    if true_silver:
        status = "PHASE37_OFFICIAL_COLLISION_SILVER_EXISTS"
        conclusion = "统一最小变量全轨迹repair产生了真正official-collision Silver；这仍只是离线Silver，不是物理Gold或policy证明。"
        next_step = "停止等待评审；只可提出一次独立official physics Gold门，不自动运行。"
    else:
        status = "PHASE37_OFFICIAL_COLLISION_SILVER_NOT_FOUND"
        failed = [name for name, value in after_audit["silver"]["checks"].items() if not value]
        conclusion = f"固定表示/配置未得到真正Silver；失败门={failed}，equality={equality_pass}，nonpenetration={nonpenetration_pass}。这否定本生成器，不否定X2或Any2Any。"
        next_step = "按门停止；不扫权重、不改root-z阈值、不跑physics/PPO。"
    candidate["phase37_official_contact_repair"].update({
        "equality_pass": equality_pass,
        "nonpenetration_pass": nonpenetration_pass,
        "phase36_auditor_tier": after_audit["tier"],
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({MOTION_ID: candidate}, OUTPUT_CACHE, compress=True)
    report = {
        "schema_version": "x2_wbt_official_contact_repair_phase37_v1",
        "mode": "offline_sparse_full_trajectory_solve_static_FK_no_physics_no_policy_optimizer",
        "truth_boundary": {
            "mujoco_integration_steps": 0, "policy_optimizer_ppo_training": False,
            "offline_lsqr_iterations": ITERATIONS, "real_robot_base_git_baidu": False,
            "contact_is_official_model_collision_not_hardware_grf_cop_wrench": True,
            "no_bronze_tolerance_as_contact": True,
        },
        "preflight": preflight,
        "before": {
            "tier": before_audit["tier"], "silver_checks": before_audit["silver"]["checks"],
            "silver_metrics": before_audit["silver"]["metrics"],
        },
        "after": {
            "tier": after_audit["tier"], "silver_checks": after_audit["silver"]["checks"],
            "silver_metrics": after_audit["silver"]["metrics"],
        },
        "solver": solver,
        "candidate_contract": candidate["phase37_official_contact_repair"],
        "candidate_cache": {"path": str(OUTPUT_CACHE), "sha256": phase28.sha256(OUTPUT_CACHE)},
        "decision": {
            "status": status, "true_official_collision_silver": true_silver,
            "equality_pass": equality_pass, "nonpenetration_pass": nonpenetration_pass,
            "conclusion": conclusion, "next_step": next_step,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
