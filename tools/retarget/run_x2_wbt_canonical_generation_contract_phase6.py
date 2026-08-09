#!/usr/bin/env python3
"""Phase6: stop-gated canonical generation-contract experiment for X2 WBT.

This experiment changes exactly one principal variable relative to the Phase5
reset-compatible reference: source time.  After the registered stationary
prefix, the source trajectory is entered with a C2 ramp and remains at 0.4x
source-time speed.  The fixed five-motion panel is checked for continuity,
official root/foot-orientation semantics, and paired prescribed/free-root
zero-update behaviour.

The pre-registered stop rule is intentionally hard: if any motion loses more
than 0.1 s free-root survival, or exceeds its Phase5 slip allowance, no ground,
contact, turn-teacher, policy, or training intervention may be added.  MuJoCo
collision/contact, COM and DCM are model estimates, not hardware GRF truth.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
import retarget.run_x2_wbt_generation_contract_phase5 as phase5


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT = phase3.DEFAULT_CURRENT
DEFAULT_PHASE5 = REPO / "reports/retarget/x2_wbt_generation_contract_phase5.json"
DEFAULT_MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
DEFAULT_MIRROR_CONTRACT = REPO / "reports/retarget/x2_official_mirror_contract.json"
DEFAULT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase6_canonical_generation_contract/x2_phase6_timescale.pkl"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_wbt_canonical_generation_contract_phase6.json"
DEFAULT_MD = REPO / "reports/retarget/x2_wbt_canonical_generation_contract_phase6.md"

ROLES = phase5.ROLES
STATIC_PREFIX_FRAMES = 15
SOURCE_RAMP_FRAMES = 30
SOURCE_TIME_SCALE = 0.40
OUTPUT_RAMP_FRAMES = int(round(SOURCE_RAMP_FRAMES / SOURCE_TIME_SCALE))
SURVIVAL_REGRESSION_LIMIT_S = 0.10
SLIP_RELATIVE_ALLOWANCE = 1.10
SLIP_ABSOLUTE_ALLOWANCE_MPS = 0.020


def c2_ramp_coefficients(source_horizon: float, output_frames: int, source_scale: float) -> np.ndarray:
    """Return cubic/quartic/quintic terms with C2 endpoint matching."""
    matrix = np.asarray(((1, 1, 1), (3, 4, 5), (6, 12, 20)), dtype=np.float64)
    rhs = np.asarray((source_horizon, source_scale * output_frames, 0.0), dtype=np.float64)
    return np.linalg.solve(matrix, rhs)


def canonical_source_map(frame_count: int) -> np.ndarray:
    if frame_count <= SOURCE_RAMP_FRAMES + 1:
        raise ValueError("motion too short for Phase6 source-time contract")
    coefficient = c2_ramp_coefficients(
        SOURCE_RAMP_FRAMES, OUTPUT_RAMP_FRAMES, SOURCE_TIME_SCALE
    )
    u = np.arange(1, OUTPUT_RAMP_FRAMES + 1, dtype=np.float64) / OUTPUT_RAMP_FRAMES
    ramp = sum(coefficient[index] * u ** (index + 3) for index in range(3))
    tail_count = int(math.ceil((frame_count - 1 - SOURCE_RAMP_FRAMES) / SOURCE_TIME_SCALE))
    tail = SOURCE_RAMP_FRAMES + SOURCE_TIME_SCALE * np.arange(1, tail_count + 1)
    tail = np.minimum(tail, frame_count - 1)
    result = np.concatenate((np.zeros(STATIC_PREFIX_FRAMES), ramp, tail))
    if np.any(np.diff(result) < -1.0e-12) or result[-1] != frame_count - 1:
        raise AssertionError("source-time map must be monotone and reach the final source frame")
    return result


def apply_source_time_contract(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_indices = canonical_source_map(len(entry["dof"]))
    result = copy.deepcopy(entry)
    for key in ("dof", "root_trans_offset", "smpl_joints", "pose_aa"):
        value = entry.get(key)
        if isinstance(value, np.ndarray) and len(value) == len(entry["dof"]):
            result[key] = phase5.interpolate_array(value, source_indices).astype(value.dtype)
    source_times = np.arange(len(entry["root_rot"]), dtype=np.float64)
    root_rotation = np.asarray(entry["root_rot"])
    result["root_rot"] = Slerp(source_times, Rotation.from_quat(root_rotation))(
        source_indices
    ).as_quat().astype(root_rotation.dtype)
    result["source_segment_duration_s"] = (len(result["dof"]) - 1) / float(result["fps"])
    result["phase6_source_time_contract"] = {
        "static_prefix_frames": STATIC_PREFIX_FRAMES,
        "source_ramp_frames": SOURCE_RAMP_FRAMES,
        "output_ramp_frames": OUTPUT_RAMP_FRAMES,
        "source_time_scale": SOURCE_TIME_SCALE,
        "c2_endpoint_matching": True,
        "time_map_source_indices": source_indices,
        "only_principal_variable": "source time",
    }
    return result, continuity_metrics(result)


def continuity_metrics(entry: dict[str, Any]) -> dict[str, float]:
    fps = float(entry["fps"])
    dof = np.asarray(entry["dof"], dtype=np.float64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    rotation = Rotation.from_quat(np.asarray(entry["root_rot"], dtype=np.float64))
    joint_velocity = np.diff(dof, axis=0) * fps
    root_linear = np.diff(root, axis=0) * fps
    root_angular = np.asarray([
        (rotation[index].inv() * rotation[index + 1]).as_rotvec() * fps
        for index in range(len(rotation) - 1)
    ])
    join = STATIC_PREFIX_FRAMES + OUTPUT_RAMP_FRAMES - 1
    return {
        "frame0_joint_velocity_max_radps": float(np.max(np.abs(joint_velocity[0]))),
        "frame0_root_linear_norm_mps": float(np.linalg.norm(root_linear[0])),
        "frame0_root_angular_norm_radps": float(np.linalg.norm(root_angular[0])),
        "join_joint_velocity_jump_p95_radps": float(
            np.percentile(np.abs(joint_velocity[join] - joint_velocity[join - 1]), 95)
        ),
        "join_root_linear_velocity_jump_mps": float(
            np.linalg.norm(root_linear[join] - root_linear[join - 1])
        ),
        "join_root_angular_velocity_jump_radps": float(
            np.linalg.norm(root_angular[join] - root_angular[join - 1])
        ),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(np.asarray(entry["root_rot"]), axis=1) - 1.0))
        ),
    }


def continuity_gate(metrics: dict[str, float]) -> dict[str, Any]:
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


def yaw_delta(rotation: np.ndarray) -> float:
    yaw = np.unwrap(Rotation.from_quat(np.asarray(rotation)).as_euler("xyz")[:, 2])
    return float(yaw[-1] - yaw[0])


def turn_orientation_metrics(model, entry, phase2) -> dict[str, Any]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, _ = phase2.joint_addresses(model, names)
    data = mujoco.MjData(model)
    foot_yaw = {side: [] for side in ("left", "right")}
    for frame in range(len(entry["dof"])):
        phase2.set_reference_state(
            model,
            data,
            entry["root_trans_offset"][frame],
            entry["root_rot"][frame],
            entry["dof"][frame],
            qpos_addresses,
        )
        for side in foot_yaw:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, phase2.FOOT_BODY[side]
            )
            matrix = data.xmat[body_id].reshape(3, 3)
            foot_yaw[side].append(np.arctan2(matrix[1, 0], matrix[0, 0]))
    root = yaw_delta(np.asarray(entry["root_rot"]))
    foot = {
        side: float(np.unwrap(values)[-1] - np.unwrap(values)[0])
        for side, values in foot_yaw.items()
    }
    return {
        "root_yaw_delta_rad": root,
        "foot_yaw_delta_rad": foot,
        "root_and_both_feet_same_turn_sign": bool(
            abs(root) > 0.20 and all(np.sign(value) == np.sign(root) and abs(value) > 0.20 for value in foot.values())
        ),
        "tracking_frames": {
            "root": "pelvis floating base",
            "left_foot": phase2.FOOT_BODY["left"],
            "right_foot": phase2.FOOT_BODY["right"],
        },
    }


def slip_limit(baseline_slip: float) -> float:
    return max(
        SLIP_RELATIVE_ALLOWANCE * baseline_slip,
        baseline_slip + SLIP_ABSOLUTE_ALLOWANCE_MPS,
    )


def paired_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    survival_delta = candidate["simulated_duration_s"] - baseline["simulated_duration_s"]
    allowed_slip = slip_limit(baseline["stance_slip_p95_mps"])
    checks = {
        "survival_not_worse_gt_0p1s": survival_delta >= -SURVIVAL_REGRESSION_LIMIT_S,
        "slip_within_allowance": candidate["stance_slip_p95_mps"] <= allowed_slip,
    }
    return {
        "survival_delta_s": float(survival_delta),
        "slip_limit_mps": float(allowed_slip),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT Canonical Generation Contract Phase6",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 无训练、无 teacher 搜索、无 checkpoint、无真机；只做固定五动作零更新 reference/PD 对照。",
        "- official MuJoCo collision/contact、COM、DCM 均是模型估计量，**不是实机 GRF/COP/足底力真值**。",
        "",
        "## 假设与单变量",
        "",
        "Phase5 turn/stop 源段没有静止拼接点。A 只把 source-time 改为：0.5 s 静止前缀 → C2 ramp → 全程 0.4× source speed；关节、root、ground、contact、PD、控制频率均不改。若任一动作 survival 下降 >0.1 s 或 slip 越门，立即停止，不运行 B（root-ground/contact）或 turn teacher。",
        "",
        "## 连续性、turn 与官方物理门",
        "",
        "| role | continuity | join q/root-lin/root-ang jump | turn root/L-foot/R-foot yaw | prescribed full | Phase5→A free(s) | Δ(s) | slip/limit | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["per_role"][role]
        continuity = value["continuity"]
        orientation = value.get("turn_orientation")
        turn = "-" if orientation is None else (
            f"{orientation['root_yaw_delta_rad']:.3f}/"
            f"{orientation['foot_yaw_delta_rad']['left']:.3f}/"
            f"{orientation['foot_yaw_delta_rad']['right']:.3f}"
        )
        baseline = value["phase5_baseline_free"]
        candidate = value["phase6_candidate_free"]
        gate = value["paired_gate"]
        lines.append(
            f"| {role} | {value['continuity_gate']['pass']} | "
            f"{continuity['join_joint_velocity_jump_p95_radps']:.3f}/"
            f"{continuity['join_root_linear_velocity_jump_mps']:.3f}/"
            f"{continuity['join_root_angular_velocity_jump_radps']:.3f} | {turn} | "
            f"{value['phase6_candidate_prescribed']['duration_fraction']:.3f} | "
            f"{baseline['simulated_duration_s']:.3f}→{candidate['simulated_duration_s']:.3f} | "
            f"{gate['survival_delta_s']:+.3f} | {candidate['stance_slip_p95_mps']:.4f}/"
            f"{gate['slip_limit_mps']:.4f} | {gate['pass']} |"
        )
    lines += [
        "",
        "## B1 canonical / Phase5 ground 前置门",
        "",
        f"- official 31DoF 顺序一致且 head 2DoF 锁零：`{report['canonical_model_gate']['pass']}`。WBT 控制边界仍是 29DoF；缓存保留 31 列仅用于匹配 MJCF，最后两列不参与动作。",
        f"- Phase5 已有 root-ground 离线门：`{report['phase5_ground_prerequisite']['ground_offline_all_actions']}`；失败动作：`{report['phase5_ground_prerequisite']['failed_roles']}`。Phase6 没有把旧失败 B 冒充新 canonical contract。",
        "",
        "## 停止裁决",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "- 本结果只否定当前 0.4× source-time 生成器，不能据此否定 X2 动力学、Any2Any 或 WBT 路线。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--phase5", type=Path, default=DEFAULT_PHASE5)
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_MODEL_CONTRACT)
    parser.add_argument("--mirror-contract", type=Path, default=DEFAULT_MIRROR_CONTRACT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase6")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase6")
    motions = joblib.load(args.current)
    phase5_report = json.loads(args.phase5.read_text(encoding="utf-8"))
    model_contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    role_to_key = {entry["panel_role"]: key for key, entry in motions.items()}
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    official_names_31 = model_contract["control_boundaries"]["official_mjcf_actuated_31"]
    official_names_29 = model_contract["control_boundaries"]["wbt_target_29"]
    head_names = model_contract["control_boundaries"]["head_locked_2"]

    output_cache: dict[str, Any] = {}
    per_role: dict[str, Any] = {}
    stop_roles = []
    for role in ROLES:
        print(f"[phase6] {role}", flush=True)
        key = role_to_key[role]
        candidate, continuity = apply_source_time_contract(motions[key])
        candidate_names = list(candidate["joint_names_mujoco"])
        head_indices = [candidate_names.index(name) for name in head_names]
        model_checks = {
            "joint_order_matches_official_31": candidate_names == official_names_31,
            "wbt_29_joint_set_plus_locked_head_2": (
                set(candidate_names[:-2]) == set(official_names_29)
                and candidate_names[-2:] == head_names
            ),
            "head_trajectory_locked_zero": bool(np.max(np.abs(np.asarray(candidate["dof"])[:, head_indices])) <= 1.0e-8),
        }
        continuity_result = continuity_gate(continuity)
        orientation = turn_orientation_metrics(model, candidate, phase2) if role.startswith("turn_") else None
        official_contact, prescribed = phase5.extract_official_contact(candidate, physics, phase2)
        free = phase4.simulate_contract_case(
            physics.DEFAULT_SCENE,
            physics.DEFAULT_CONTROL,
            candidate,
            "free_root_balance",
            "zero",
            physics,
            phase2,
            expected_contact=official_contact,
            contact_activation="collision",
        )
        baseline = phase5_report["per_role"][role]["physics"]["reset_compatible"]["free"]
        gate = paired_gate(baseline, free)
        if not gate["pass"]:
            stop_roles.append(role)
        output_cache[key] = candidate
        per_role[role] = {
            "motion_key": key,
            "continuity": continuity,
            "continuity_gate": continuity_result,
            "canonical_model_checks": model_checks,
            "turn_orientation": orientation,
            "phase5_baseline_free": baseline,
            "phase6_candidate_prescribed": prescribed,
            "phase6_candidate_free": free,
            "paired_gate": gate,
        }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_cache, args.cache)
    continuity_all = all(value["continuity_gate"]["pass"] for value in per_role.values())
    turn_all = all(
        value["turn_orientation"] is None
        or value["turn_orientation"]["root_and_both_feet_same_turn_sign"]
        for value in per_role.values()
    )
    prescribed_all = all(
        value["phase6_candidate_prescribed"]["duration_fraction"] >= 0.999
        for value in per_role.values()
    )
    paired_all = not stop_roles
    canonical_model_all = all(all(value["canonical_model_checks"].values()) for value in per_role.values())
    phase5_ground_failed_roles = [
        role
        for role, value in phase5_report["per_role"].items()
        if value["ground_contract"]["diagnostics"]["contact_abs_distance_after_p95_m"] > 0.020
    ]
    phase5_ground_offline_all = bool(
        phase5_report["decision"]["checks"]["ground_offline_all_actions"]
    )
    promotable = bool(canonical_model_all and continuity_all and turn_all and prescribed_all and paired_all)
    report = {
        "schema_version": "x2_wbt_canonical_generation_contract_phase6_v1",
        "provenance": {
            "current": {"path": str(args.current), "sha256": phase3.sha256(args.current)},
            "phase5_report": {"path": str(args.phase5), "sha256": phase3.sha256(args.phase5)},
            "official_model_contract": {"path": str(args.model_contract), "sha256": phase3.sha256(args.model_contract)},
            "official_mirror_contract": {"path": str(args.mirror_contract), "sha256": phase3.sha256(args.mirror_contract)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase3.sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": phase3.sha256(physics.DEFAULT_CONTROL)},
            "output_cache": {"path": str(args.cache), "sha256": phase3.sha256(args.cache)},
        },
        "truth_boundary": {
            "official_mujoco_version": "AimDK v1.0 project scene",
            "contact_source": "official MuJoCo floor-foot collision geometry, >=50% frame occupancy",
            "contact_is_model_estimate_not_hardware_grf": True,
            "com_dcm_are_model_estimates": True,
            "training_teacher_checkpoint_real_robot": False,
        },
        "single_variable_intervention": {
            "name": "source_time_scale",
            "phase5": "1.0x after 30-source-frame warm-start",
            "phase6": SOURCE_TIME_SCALE,
            "static_prefix_frames": STATIC_PREFIX_FRAMES,
            "c2_ramp_source_frames": SOURCE_RAMP_FRAMES,
            "c2_ramp_output_frames": OUTPUT_RAMP_FRAMES,
            "unchanged": ["joint path", "root path", "ground", "contact", "PD", "control rate"],
        },
        "canonical_model_gate": {
            "checks": {
                "all_five_joint_order_head_boundary": canonical_model_all,
                "tracking_root": model_contract["coordinate_system"]["tracking_frames"]["pelvis"],
                "tracking_feet": [
                    model_contract["coordinate_system"]["tracking_frames"]["left_foot"],
                    model_contract["coordinate_system"]["tracking_frames"]["right_foot"],
                ],
                "official_foot_collision_points_each": [
                    len(model_contract["foot_contact_geometry"]["left"]),
                    len(model_contract["foot_contact_geometry"]["right"]),
                ],
            },
            "official_rl_29_to_mjcf_31_indices": [
                official_names_31.index(name) for name in official_names_29
            ],
            "joint_order_note": "official RL 29 is interleaved; reference cache is MJCF 31 order, so an explicit permutation is mandatory",
            "pass": canonical_model_all,
        },
        "phase5_ground_prerequisite": {
            "source": "read-only inherited Phase5 offline signed-geometry gate",
            "ground_offline_all_actions": phase5_ground_offline_all,
            "failed_roles": phase5_ground_failed_roles,
            "phase6_new_ground_intervention_executed": False,
        },
        "per_role": per_role,
        "decision": {
            "checks": {
                "canonical_model_boundary_all": canonical_model_all,
                "five_action_continuity_all": continuity_all,
                "turn_root_yaw_and_foot_orientation_all": turn_all,
                "prescribed_zero_update_full_all": prescribed_all,
                "free_root_survival_and_slip_all": paired_all,
                "root_ground_contact_stage_executed": False,
                "training_or_teacher_executed": False,
            },
            "stop_roles": stop_roles,
            "status": (
                "PHASE6_CANONICAL_SOURCE_TIME_PROMOTABLE_TO_GROUND_STAGE"
                if promotable
                else "PHASE6_CANONICAL_SOURCE_TIME_STOPPED_NOT_PROMOTABLE"
            ),
            "promote_generation_contract": promotable,
            "result": (
                "五动作全部通过 source-time A 门，可进入单变量 root-ground/contact B。"
                if promotable
                else f"source-time A 在 {stop_roles} 触发 survival/slip 停止门；未执行 B。"
            ),
            "conclusion": (
                "固定慢速 source-time 可作为 canonical reset contract 的候选。"
                if promotable
                else "0.4× source-time 虽修复五动作连续性，但不能跨五动作保持 free-root 生存；只否定该生成器。"
            ),
            "next_step": (
                "冻结 A，另开单变量 B 修复 root-ground/contact。"
                if promotable
                else "停止扩展该分支；回到原始 source 生成器，优先修数据级 reset/ground/contact 联合边界，而非继续扫时间尺度。"
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
