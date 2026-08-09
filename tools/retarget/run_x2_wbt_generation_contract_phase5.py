#!/usr/bin/env python3
"""Minimal reset-start and root-ground contract repair for the five X2 motions.

Intervention A time-warps the first second behind a stationary prefix; it never
injects the raw, discontinuous frame-0 qvel.  Intervention B solves a bounded,
smooth root-z trajectory from intended support phases and the official MuJoCo
floor/foot signed geometry distance.  B is a phase-constrained geometry solve,
not a constant post-hoc height offset.

Only original/A/A+B zero-update prescribed/free replays are run.  There is no
teacher search, policy, checkpoint, training, or real robot.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_contract_phase4 as phase4


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT = phase3.DEFAULT_CURRENT
DEFAULT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase5_generation_contract/x2_phase5_reset_ground.pkl"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_wbt_generation_contract_phase5.json"
DEFAULT_MD = REPO / "reports/retarget/x2_wbt_generation_contract_phase5.md"
ROLES = ("walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand")

STATIC_PREFIX_FRAMES = 15
WARM_SOURCE_HORIZON_FRAMES = 30
ROOT_Z_BOUND_M = 0.10
ROOT_Z_STEP_BOUND_M = 0.004
GROUND_TARGET_M = 0.001


def warm_start_source_map(frame_count: int) -> np.ndarray:
    if frame_count <= WARM_SOURCE_HORIZON_FRAMES + 1:
        raise ValueError("motion too short for registered warm-start horizon")
    values = [0.0] * STATIC_PREFIX_FRAMES
    # Cubic has derivative 0 at the reset and 1 at the hand-off, so velocity
    # joins the original trajectory continuously instead of stopping twice.
    for step in range(1, WARM_SOURCE_HORIZON_FRAMES + 1):
        u = step / WARM_SOURCE_HORIZON_FRAMES
        values.append(WARM_SOURCE_HORIZON_FRAMES * (u * u * (2.0 - u)))
    values.extend(range(WARM_SOURCE_HORIZON_FRAMES + 1, frame_count))
    return np.asarray(values, dtype=np.float64)


def interpolate_array(values: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    source = np.arange(len(values), dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.column_stack([
        np.interp(source_indices, source, flat[:, index]) for index in range(flat.shape[1])
    ])
    return result.reshape((len(source_indices),) + values.shape[1:])


def make_reset_compatible(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_indices = warm_start_source_map(len(entry["dof"]))
    result = copy.deepcopy(entry)
    for key in ("dof", "root_trans_offset", "smpl_joints", "pose_aa"):
        if key in entry and isinstance(entry[key], np.ndarray) and len(entry[key]) == len(entry["dof"]):
            result[key] = interpolate_array(np.asarray(entry[key]), source_indices).astype(entry[key].dtype)
    source_times = np.arange(len(entry["root_rot"]), dtype=np.float64)
    result["root_rot"] = Slerp(source_times, Rotation.from_quat(np.asarray(entry["root_rot"])))(
        source_indices
    ).as_quat().astype(np.asarray(entry["root_rot"]).dtype)
    result["phase5_reset_compatible"] = {
        "static_prefix_frames": STATIC_PREFIX_FRAMES,
        "warm_source_horizon_frames": WARM_SOURCE_HORIZON_FRAMES,
        "time_map_source_indices": source_indices,
        "raw_frame0_qvel_injected": False,
    }
    result["source_segment_duration_s"] = (len(result["dof"]) - 1) / float(result["fps"])
    metrics = continuity_metrics(result)
    metrics.update({
        "source_frames": len(entry["dof"]),
        "output_frames": len(result["dof"]),
        "extra_frames": len(result["dof"]) - len(entry["dof"]),
    })
    return result, metrics


def continuity_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    fps = float(entry["fps"])
    dof = np.asarray(entry["dof"], dtype=np.float64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = Rotation.from_quat(np.asarray(entry["root_rot"], dtype=np.float64))
    joint_velocity = np.diff(dof, axis=0) * fps
    root_linear = np.diff(root, axis=0) * fps
    root_angular = np.asarray([
        (quat[index].inv() * quat[index + 1]).as_rotvec() * fps
        for index in range(len(quat) - 1)
    ])
    join = STATIC_PREFIX_FRAMES + WARM_SOURCE_HORIZON_FRAMES - 1
    return {
        "frame0_joint_velocity_p95_radps": float(np.percentile(np.abs(joint_velocity[0]), 95)),
        "frame0_joint_velocity_max_radps": float(np.max(np.abs(joint_velocity[0]))),
        "frame0_root_linear_norm_mps": float(np.linalg.norm(root_linear[0])),
        "frame0_root_angular_norm_radps": float(np.linalg.norm(root_angular[0])),
        "global_joint_velocity_max_radps": float(np.max(np.abs(joint_velocity))),
        "global_root_linear_max_mps": float(np.max(np.linalg.norm(root_linear, axis=1))),
        "global_root_angular_max_radps": float(np.max(np.linalg.norm(root_angular, axis=1))),
        "join_joint_velocity_jump_p95_radps": float(
            np.percentile(np.abs(joint_velocity[join] - joint_velocity[join - 1]), 95)
        ),
        "join_root_linear_velocity_jump_mps": float(np.linalg.norm(root_linear[join] - root_linear[join - 1])),
        "join_root_angular_velocity_jump_radps": float(np.linalg.norm(root_angular[join] - root_angular[join - 1])),
        "quaternion_norm_max_error": float(np.max(np.abs(np.linalg.norm(np.asarray(entry["root_rot"]), axis=1) - 1.0))),
    }


def geometry_distance_series(model, entry, phase2, physics) -> dict[str, np.ndarray]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, _ = phase2.joint_addresses(model, names)
    floor, foot_geoms = physics.foot_geom_contract(model)
    data = mujoco.MjData(model)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    dof = np.asarray(entry["dof"], dtype=np.float64)
    result = {side: np.zeros(len(dof), dtype=np.float64) for side in ("left", "right")}
    fromto = np.zeros(6)
    for frame in range(len(dof)):
        phase2.set_reference_state(model, data, root[frame], quat[frame], dof[frame], qpos_addresses)
        for side in ("left", "right"):
            result[side][frame] = min(
                float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto))
                for geom in foot_geoms[side]
            )
    return result


def solve_root_ground_contract(
    model: mujoco.MjModel, entry: dict[str, Any], phase2, physics
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_distance = geometry_distance_series(model, entry, phase2, physics)
    kin = phase2.reference_kinematics(model, entry)
    phase = phase2.phase_contract(kin)
    contact = {
        side: ~np.asarray(phase[f"{side}_swing"], dtype=bool) for side in ("left", "right")
    }
    frames = len(entry["dof"])
    desired = np.zeros(frames)
    weight = np.full(frames, 0.05)
    for frame in range(frames):
        distances = [
            before_distance[side][frame] for side in ("left", "right") if contact[side][frame]
        ]
        if distances:
            desired[frame] = GROUND_TARGET_M - float(np.median(distances))
            weight[frame] = float(len(distances))
        elif frame:
            desired[frame] = desired[frame - 1]

    # Tikhonov phase trajectory solve: contact geometry fit plus second-order
    # smoothness.  Bounds and slope are projected as hard feasibility limits.
    d2 = np.zeros((max(0, frames - 2), frames))
    for index in range(frames - 2):
        d2[index, index : index + 3] = (1.0, -2.0, 1.0)
    system = np.diag(weight) + 40.0 * (d2.T @ d2) + 1.0e-8 * np.eye(frames)
    correction = np.linalg.solve(system, weight * desired)
    correction = np.clip(correction, -ROOT_Z_BOUND_M, ROOT_Z_BOUND_M)
    for _ in range(3):
        for frame in range(1, frames):
            correction[frame] = np.clip(
                correction[frame], correction[frame - 1] - ROOT_Z_STEP_BOUND_M, correction[frame - 1] + ROOT_Z_STEP_BOUND_M
            )
        for frame in range(frames - 2, -1, -1):
            correction[frame] = np.clip(
                correction[frame], correction[frame + 1] - ROOT_Z_STEP_BOUND_M, correction[frame + 1] + ROOT_Z_STEP_BOUND_M
            )
    # The registered static prefix must remain physically static.
    correction[:STATIC_PREFIX_FRAMES] = correction[STATIC_PREFIX_FRAMES - 1]

    result = copy.deepcopy(entry)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64).copy()
    root[:, 2] += correction
    result["root_trans_offset"] = root.astype(np.asarray(entry["root_trans_offset"]).dtype)
    result["phase5_ground_contract"] = {
        "source": "official floor-foot signed collision geometry + FK intended support phase",
        "target_distance_m": GROUND_TARGET_M,
        "second_difference_weight": 40.0,
        "root_z_bound_m": ROOT_Z_BOUND_M,
        "root_z_step_bound_m": ROOT_Z_STEP_BOUND_M,
        "constant_segment_shift": False,
        "correction_m": correction,
    }
    after_distance = geometry_distance_series(model, result, phase2, physics)
    diagnostics = ground_diagnostics(before_distance, after_distance, contact, correction)
    return result, diagnostics


def ground_diagnostics(before, after, contact, correction) -> dict[str, Any]:
    def selected(values, labels):
        data = [values[side][labels[side]] for side in ("left", "right")]
        return np.concatenate([value for value in data if value.size])

    before_contact = selected(before, contact)
    after_contact = selected(after, contact)
    return {
        "root_z_correction_min_m": float(np.min(correction)),
        "root_z_correction_max_m": float(np.max(correction)),
        "root_z_correction_abs_max_m": float(np.max(np.abs(correction))),
        "root_z_correction_step_max_m": float(np.max(np.abs(np.diff(correction)))),
        "contact_signed_distance_before_p05_p50_p95_m": [
            float(np.percentile(before_contact, q)) for q in (5, 50, 95)
        ],
        "contact_signed_distance_after_p05_p50_p95_m": [
            float(np.percentile(after_contact, q)) for q in (5, 50, 95)
        ],
        "contact_abs_distance_before_p95_m": float(np.percentile(np.abs(before_contact), 95)),
        "contact_abs_distance_after_p95_m": float(np.percentile(np.abs(after_contact), 95)),
        "deep_penetration_fraction_before": float(np.mean(before_contact < -0.020)),
        "deep_penetration_fraction_after": float(np.mean(after_contact < -0.020)),
        "hover_gt_5mm_fraction_before": float(np.mean(before_contact > 0.005)),
        "hover_gt_5mm_fraction_after": float(np.mean(after_contact > 0.005)),
    }


def extract_official_contact(entry, physics, phase2) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    trace_result = phase4.simulate_contract_case(
        physics.DEFAULT_SCENE,
        physics.DEFAULT_CONTROL,
        entry,
        "prescribed_root_trackability",
        "zero",
        physics,
        phase2,
        collect_contact_trace=True,
        contact_activation="collision",
    )
    trace = trace_result.pop("contact_trace")
    labels = phase4.contact_frame_labels(
        {side: np.asarray(trace["active"][side], dtype=bool) for side in ("left", "right")},
        np.asarray(trace["sim_times_s"]),
        len(entry["dof"]),
        float(entry["fps"]),
    )
    return labels, trace_result


def continuity_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "frame0_joint_velocity_near_zero": metrics["frame0_joint_velocity_max_radps"] <= 0.05,
        "frame0_root_linear_near_zero": metrics["frame0_root_linear_norm_mps"] <= 0.02,
        "frame0_root_angular_near_zero": metrics["frame0_root_angular_norm_radps"] <= 0.02,
        "join_joint_velocity_jump_bounded": metrics["join_joint_velocity_jump_p95_radps"] <= 0.50,
        "join_root_linear_jump_bounded": metrics["join_root_linear_velocity_jump_mps"] <= 0.10,
        "join_root_angular_jump_bounded": metrics["join_root_angular_velocity_jump_radps"] <= 0.50,
        "quaternion_normalized": metrics["quaternion_norm_max_error"] <= 1.0e-5,
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT Generation Contract Phase5",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 无训练/teacher搜索/checkpoint/真机；只比较 original、reset-compatible(A)、A+ground(B)。",
        "",
        "## 预注册门",
        "",
        "A：0.5s静止前缀、1s速度连续time-warp；frame0 joint/root速度近零，连接处joint/root速度跳变受限。B：official signed geometry+intended support的平滑root-z轨迹，|z|≤0.10m、每帧≤4mm。A+B相对A任一动作生存下降>0.1s或slip超过`max(1.1×A, A+0.02)`即停止。",
        "",
        "## 离线连续性与ground",
        "",
        "| role | A continuity | join q/root-ang jump | z range/step(m) | contact |dist| p95 before→after(m) | deep penetration before→after |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["per_role"][role]
        continuity = value["reset_compatible"]["continuity"]
        ground = value["ground_contract"]["diagnostics"]
        lines.append(
            f"| {role} | {value['reset_compatible']['gate']['pass']} | "
            f"{continuity['join_joint_velocity_jump_p95_radps']:.3f}/{continuity['join_root_angular_velocity_jump_radps']:.3f} | "
            f"{ground['root_z_correction_min_m']:.3f}..{ground['root_z_correction_max_m']:.3f}/{ground['root_z_correction_step_max_m']:.4f} | "
            f"{ground['contact_abs_distance_before_p95_m']:.3f}→{ground['contact_abs_distance_after_p95_m']:.3f} | "
            f"{ground['deep_penetration_fraction_before']:.3f}→{ground['deep_penetration_fraction_after']:.3f} |"
        )
    lines += [
        "",
        "## prescribed/free zero-update paired gate",
        "",
        "| role | variant | prescribed full | free survival(s) | slip p95 | contact agreement vs intended | sat. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        for variant in ("original", "reset_compatible", "reset_plus_ground"):
            value = report["per_role"][role]["physics"][variant]
            lines.append(
                f"| {role} | {variant} | {value['prescribed']['duration_fraction']:.3f} | "
                f"{value['free']['simulated_duration_s']:.3f} | {value['free']['stance_slip_p95_mps']:.4f} | "
                f"{value['contact_comparison']['macro_agreement']:.3f} | "
                f"{value['prescribed']['torque_saturation_fraction']:.4f} |"
            )
    lines += [
        "",
        "## 结论",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
        "若失败，结论只否定当前reset/ground生成器，不构成X2动力学不可能证明。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase5")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase5")
    motions = joblib.load(args.current)
    role_to_key = {entry["panel_role"]: key for key, entry in motions.items()}
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    output_cache = {}
    per_role = {}

    for role in ROLES:
        print(f"[phase5] build {role}", flush=True)
        key = role_to_key[role]
        original = motions[key]
        reset, continuity = make_reset_compatible(original)
        reset_gate = continuity_gate(continuity)
        grounded, ground = solve_root_ground_contract(model, reset, phase2, physics)
        output_cache[key] = {
            "reset_compatible": reset,
            "reset_plus_ground": grounded,
        }
        variants = {
            "original": original,
            "reset_compatible": reset,
            "reset_plus_ground": grounded,
        }
        physics_results = {}
        for variant, entry in variants.items():
            print(f"[phase5] physics {role}/{variant}", flush=True)
            official, prescribed = extract_official_contact(entry, physics, phase2)
            kin = phase2.reference_kinematics(model, entry)
            phase = phase2.phase_contract(kin)
            intended = {side: ~phase[f"{side}_swing"] for side in ("left", "right")}
            comparison = phase4.compare_contact_labels(intended, official)
            free = phase4.simulate_contract_case(
                physics.DEFAULT_SCENE,
                physics.DEFAULT_CONTROL,
                entry,
                "free_root_balance",
                "zero",
                physics,
                phase2,
                expected_contact=official,
                contact_activation="collision",
            )
            physics_results[variant] = {
                "prescribed": prescribed,
                "free": free,
                "official_contact": {side: values.astype(np.uint8).tolist() for side, values in official.items()},
                "contact_comparison": comparison,
            }
        per_role[role] = {
            "motion_key": key,
            "reset_compatible": {"continuity": continuity, "gate": reset_gate},
            "ground_contract": {"diagnostics": ground},
            "physics": physics_results,
        }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_cache, args.cache)
    continuity_all = all(value["reset_compatible"]["gate"]["pass"] for value in per_role.values())
    ground_offline_all = all(
        value["ground_contract"]["diagnostics"]["contact_abs_distance_after_p95_m"] <= 0.020
        and value["ground_contract"]["diagnostics"]["root_z_correction_abs_max_m"] <= ROOT_Z_BOUND_M + 1e-8
        and value["ground_contract"]["diagnostics"]["root_z_correction_step_max_m"] <= ROOT_Z_STEP_BOUND_M + 1e-8
        for value in per_role.values()
    )
    no_survival_regression = all(
        value["physics"]["reset_plus_ground"]["free"]["simulated_duration_s"]
        >= value["physics"]["reset_compatible"]["free"]["simulated_duration_s"] - 0.10
        for value in per_role.values()
    )
    no_slip_regression = all(
        value["physics"]["reset_plus_ground"]["free"]["stance_slip_p95_mps"]
        <= max(
            1.10 * value["physics"]["reset_compatible"]["free"]["stance_slip_p95_mps"],
            value["physics"]["reset_compatible"]["free"]["stance_slip_p95_mps"] + 0.020,
        )
        for value in per_role.values()
    )
    prescribed_all = all(
        value["physics"]["reset_plus_ground"]["prescribed"]["duration_fraction"] >= 0.999
        and value["physics"]["reset_plus_ground"]["prescribed"]["torque_saturation_fraction"]
        <= value["physics"]["reset_compatible"]["prescribed"]["torque_saturation_fraction"] + 0.020
        for value in per_role.values()
    )
    promote = bool(continuity_all and ground_offline_all and no_survival_regression and no_slip_regression and prescribed_all)
    report = {
        "schema_version": "x2_wbt_generation_contract_phase5_v1",
        "provenance": {
            "current": {"path": str(args.current), "sha256": phase3.sha256(args.current)},
            "scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase3.sha256(physics.DEFAULT_SCENE)},
            "control": {"path": str(physics.DEFAULT_CONTROL), "sha256": phase3.sha256(physics.DEFAULT_CONTROL)},
            "output_cache": {"path": str(args.cache), "sha256": phase3.sha256(args.cache)},
        },
        "preregistered_contract": {
            "static_prefix_frames": STATIC_PREFIX_FRAMES,
            "warm_source_horizon_frames": WARM_SOURCE_HORIZON_FRAMES,
            "raw_frame0_qvel_injected": False,
            "root_z_bound_m": ROOT_Z_BOUND_M,
            "root_z_step_bound_m": ROOT_Z_STEP_BOUND_M,
            "ground_target_m": GROUND_TARGET_M,
            "stop_if_any_survival_regression_gt_s": 0.10,
            "slip_limit": "max(1.10 * reset-compatible, reset-compatible + 0.020 m/s)",
        },
        "truth_boundary": {
            "official_contact": "MuJoCo floor-foot collision event, not GRF",
            "intended_contact": "FK relative height/speed phase used only to constrain geometry solve",
            "training_or_teacher_search": False,
            "raw_frame0_qvel_used": False,
        },
        "per_role": per_role,
        "decision": {
            "checks": {
                "reset_continuity_all_actions": continuity_all,
                "ground_offline_all_actions": ground_offline_all,
                "ground_no_free_survival_regression": no_survival_regression,
                "ground_no_free_slip_regression": no_slip_regression,
                "ground_prescribed_trackability_all_actions": prescribed_all,
            },
            "promote_generation_contract": promote,
            "status": "PHASE5_GENERATION_CONTRACT_PROMOTABLE" if promote else "PHASE5_GENERATION_CONTRACT_NOT_PROMOTABLE",
            "result": (
                "reset-compatible与phase-constrained ground contract通过五动作离线/官方物理门。"
                if promote
                else "至少一项预注册连续性、ground或官方物理门失败。"
            ),
            "conclusion": (
                "可冻结为下一轮reference生成基线，但仍不代表WBT策略可执行。"
                if promote
                else "只否定当前最小生成器修复；禁止turn teacher/PPO，并保留original作为对照。"
            ),
            "next_step": (
                "在独立held-out动作复核生成契约，不训练。"
                if promote
                else "定位失败门，不扩大参数或搜索。"
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
