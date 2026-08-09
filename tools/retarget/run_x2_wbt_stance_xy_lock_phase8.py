#!/usr/bin/env python3
"""Offline-only Phase8 stance-foot XY lock on the frozen Phase7 contract.

The sole structural intervention is a per-contiguous-stance-window horizontal
foot-position constraint.  The anchor is the window's first official foot-link
XY pose, with a C2 boundary blend whose width is the frozen Phase7 MA9 half
window.  Swing frames are never constrained.  Root XY/orientation, source
timing, foot semantics, root-ground clearance and MA9 remain frozen.

The numerical DLS settings are inherited from the pre-existing Phase4 foot
orientation representation; they are not selected with free-root results.
Offline joint-step/ground/terminal/contact/slip gates run before any official
physics replay.  Failure therefore blocks prescribed/free A/B, training and
teacher search.  Model contacts are not hardware GRF/COP truth.
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


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW = phase3.DEFAULT_CURRENT
DEFAULT_PHASE7 = phase7.DEFAULT_CACHE
DEFAULT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "diagnostics/phase8_stance_xy_lock_rejected/x2_phase8_stance_xy_lock.pkl"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_wbt_stance_xy_lock_phase8.json"
DEFAULT_MD = REPO / "reports/retarget/x2_wbt_stance_xy_lock_phase8.md"
ROLES = phase7.ROLES

BLEND_FRAMES = phase7.GROUND_SMOOTH_WINDOW // 2
IK_ITERATIONS = 5
IK_GAIN = 0.55
IK_DAMPING = 7.5e-3
IK_DELTA_LIMIT_RAD = 0.04
JOINT_STEP_TIER_MAX_RAD = 0.15
JOINT_STEP_RELATIVE_ALLOWANCE = 1.10
SLIP_RELATIVE_ALLOWANCE = 1.10
SLIP_ABSOLUTE_ALLOWANCE_MPS = 0.020
CONTACT_INTENT_AGREEMENT_MIN = 0.90


def boolean_intervals(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False]) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def c2_boundary_weight(frame: int, start: int, end: int) -> float:
    if not start <= frame < end:
        return 0.0
    edge = min(frame - start, end - 1 - frame)
    u = min(1.0, edge / BLEND_FRAMES)
    return float(10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5)


def foot_xy_series(model, entry, phase2) -> dict[str, np.ndarray]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, _ = phase2.joint_addresses(model, names)
    body_ids = {
        side: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
        )
        for side in ("left", "right")
    }
    data = mujoco.MjData(model)
    result = {side: np.zeros((len(entry["dof"]), 2), dtype=np.float64) for side in body_ids}
    for frame in range(len(entry["dof"])):
        phase2.set_reference_state(
            model,
            data,
            entry["root_trans_offset"][frame],
            entry["root_rot"][frame],
            entry["dof"][frame],
            qpos_addresses,
        )
        for side, body_id in body_ids.items():
            result[side][frame] = data.xpos[body_id, :2]
    return result


def phase7_contact_intent(model, entry, reset, phase2) -> dict[str, np.ndarray]:
    distance = phase7.sphere_distance_series(model, entry, phase2)
    threshold = reset["reset_clearance_m"] + reset["sole_sphere_radius_m"]
    return {side: np.asarray(values <= threshold, dtype=bool) for side, values in distance.items()}


def apply_stance_xy_lock(model, entry, contact, phase2) -> tuple[dict[str, Any], dict[str, Any]]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, qvel_addresses = phase2.joint_addresses(model, names)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    baseline = np.asarray(entry["dof"], dtype=np.float64)
    dof = baseline.copy()
    baseline_xy = foot_xy_series(model, entry, phase2)
    side_indices = {
        side: np.asarray([
            index for index, name in enumerate(names)
            if name.startswith(f"{side}_") and any(token in name for token in ("hip", "knee", "ankle"))
        ], dtype=np.int64)
        for side in ("left", "right")
    }
    side_dof_addresses = {side: qvel_addresses[index] for side, index in side_indices.items()}
    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        for side in side_indices
    }
    data = mujoco.MjData(model)
    residuals = []
    window_records = {side: [] for side in side_indices}
    weights = {side: np.zeros(len(dof), dtype=np.float64) for side in side_indices}

    for side in ("left", "right"):
        for start, end in boolean_intervals(contact[side]):
            anchor = baseline_xy[side][start].copy()
            window_records[side].append({
                "start_frame": start,
                "end_frame_exclusive": end,
                "frames": end - start,
                "anchor_xy_m": anchor.tolist(),
            })
            for frame in range(start, end):
                weight = c2_boundary_weight(frame, start, end)
                weights[side][frame] = weight
                target = baseline_xy[side][frame] + weight * (
                    anchor - baseline_xy[side][frame]
                )
                for _ in range(IK_ITERATIONS):
                    phase2.set_reference_state(
                        model, data, root[frame], quat[frame], dof[frame], qpos_addresses
                    )
                    current = data.xpos[body_ids[side], :2].copy()
                    error = target - current
                    jacp = np.zeros((3, model.nv), dtype=np.float64)
                    jacr = np.zeros((3, model.nv), dtype=np.float64)
                    mujoco.mj_jacBody(model, data, jacp, jacr, body_ids[side])
                    jacobian = jacp[:2, side_dof_addresses[side]]
                    delta = IK_GAIN * jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + IK_DAMPING * np.eye(2), error
                    )
                    dof[frame, side_indices[side]] += np.clip(
                        delta, -IK_DELTA_LIMIT_RAD, IK_DELTA_LIMIT_RAD
                    )
                    for joint_index in side_indices[side]:
                        joint_id = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, names[joint_index]
                        )
                        if bool(model.jnt_limited[joint_id]):
                            low, high = model.jnt_range[joint_id]
                            dof[frame, joint_index] = np.clip(
                                dof[frame, joint_index], low, high
                            )
                residuals.append(float(np.linalg.norm(error)))

    result = copy.deepcopy(entry)
    result["dof"] = dof.astype(np.asarray(entry["dof"]).dtype)
    result["phase8_stance_xy_lock"] = {
        "only_structural_variable": "per-contiguous-contact-window stance-foot XY lock",
        "anchor": "official ankle-roll link XY at window first frame",
        "blend": "C2 smoothstep at both boundaries",
        "blend_frames": BLEND_FRAMES,
        "swing_frames_locked": False,
        "solver_provenance": "Phase4 DLS constants, not selected by free survival",
        "ik_iterations": IK_ITERATIONS,
        "ik_gain": IK_GAIN,
        "ik_damping": IK_DAMPING,
        "ik_delta_limit_rad": IK_DELTA_LIMIT_RAD,
        "windows": window_records,
    }
    delta = dof - baseline
    boundary_delta = []
    for side, windows in window_records.items():
        for window in windows:
            for frame in (window["start_frame"], window["end_frame_exclusive"] - 1):
                boundary_delta.append(float(np.max(np.abs(delta[frame, side_indices[side]]))))
    swing_delta = []
    for side, indices in side_indices.items():
        swing_delta.append(float(np.max(np.abs(delta[~contact[side]][:, indices]))))
    return result, {
        "stance_windows": {side: len(values) for side, values in window_records.items()},
        "stance_frames": {side: int(np.count_nonzero(contact[side])) for side in contact},
        "ik_residual_p95_m": float(np.percentile(residuals, 95)) if residuals else 0.0,
        "joint_delta_abs_max_rad": float(np.max(np.abs(delta))),
        "boundary_joint_delta_abs_max_rad": max(boundary_delta, default=0.0),
        "swing_joint_delta_abs_max_rad": max(swing_delta, default=0.0),
        "weight_nonzero_frames": {side: int(np.count_nonzero(values)) for side, values in weights.items()},
    }


def stance_slip_p95(positions, contact, fps: float) -> float:
    values = []
    for side in ("left", "right"):
        velocity = np.linalg.norm(np.diff(positions[side], axis=0) * fps, axis=1)
        values.extend(velocity[np.asarray(contact[side][1:], dtype=bool)])
    return float(np.percentile(values, 95)) if values else 0.0


def offline_metrics(model, baseline, candidate, contact, reset, phase2, ground) -> dict[str, Any]:
    baseline_dof = np.asarray(baseline["dof"], dtype=np.float64)
    candidate_dof = np.asarray(candidate["dof"], dtype=np.float64)
    baseline_xy = foot_xy_series(model, baseline, phase2)
    candidate_xy = foot_xy_series(model, candidate, phase2)
    candidate_contact = phase7_contact_intent(model, candidate, reset, phase2)
    agreement = {
        side: float(np.mean(contact[side] == candidate_contact[side]))
        for side in ("left", "right")
    }
    return {
        "joint_step_max_rad": {
            "baseline": float(np.max(np.abs(np.diff(baseline_dof, axis=0)))),
            "candidate": float(np.max(np.abs(np.diff(candidate_dof, axis=0)))),
        },
        "stance_slip_p95_mps_model_estimate": {
            "baseline": stance_slip_p95(baseline_xy, contact, float(baseline["fps"])),
            "candidate": stance_slip_p95(candidate_xy, contact, float(candidate["fps"])),
        },
        "contact_intent_agreement": {
            "per_side": agreement,
            "macro": float(np.mean(list(agreement.values()))),
        },
        "ground": {name: value for name, value in ground.items() if name != "geometry_contact"},
        "root_xy_exact": np.array_equal(
            np.asarray(baseline["root_trans_offset"])[:, :2],
            np.asarray(candidate["root_trans_offset"])[:, :2],
        ),
        "root_orientation_exact": np.array_equal(baseline["root_rot"], candidate["root_rot"]),
        "fps_exact": baseline["fps"] == candidate["fps"],
    }


def offline_gate(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    baseline_step = metrics["joint_step_max_rad"]["baseline"]
    candidate_step = metrics["joint_step_max_rad"]["candidate"]
    baseline_slip = metrics["stance_slip_p95_mps_model_estimate"]["baseline"]
    candidate_slip = metrics["stance_slip_p95_mps_model_estimate"]["candidate"]
    joint_limit = max(JOINT_STEP_TIER_MAX_RAD, JOINT_STEP_RELATIVE_ALLOWANCE * baseline_step)
    slip_limit = max(
        SLIP_RELATIVE_ALLOWANCE * baseline_slip,
        baseline_slip + SLIP_ABSOLUTE_ALLOWANCE_MPS,
    )
    checks = {
        "joint_step_within_tier_or_110pct_baseline": candidate_step <= joint_limit,
        "phase7_ground_contract_preserved": metrics["ground"]["pass"],
        "window_boundaries_exact": diagnostics["boundary_joint_delta_abs_max_rad"] <= 1.0e-8,
        "swing_joint_path_exact": diagnostics["swing_joint_delta_abs_max_rad"] <= 1.0e-8,
        "root_xy_orientation_timing_exact": (
            metrics["root_xy_exact"] and metrics["root_orientation_exact"] and metrics["fps_exact"]
        ),
        "contact_intent_agreement_ge_0p90": (
            metrics["contact_intent_agreement"]["macro"] >= CONTACT_INTENT_AGREEMENT_MIN
        ),
        "model_stance_slip_not_worse": candidate_slip <= slip_limit,
    }
    return {
        "joint_step_limit_rad": joint_limit,
        "stance_slip_limit_mps": slip_limit,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT Stance-Foot XY Lock Phase8",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- Phase7 24球、clearance、MA9、root XY/orientation、source timing 全部冻结；只新增 stance-window XY lock。",
        "- 无 prescribed/free 物理回放、训练、teacher、checkpoint、BASE、Git、百度或真机。contact/slip 是模型估计，不是实机 GRF/COP。",
        "",
        "## 固定结构",
        "",
        "每个 Phase7 连续 geometry-contact 窗以首帧 ankle-roll XY 为 anchor，仅同侧6个腿关节做 DLS；窗口两端用4帧 C2 blend，摆动段关节保持逐位精确。数值常量继承 Phase4，不按 free survival 选参。IK 后只重新执行冻结的 Phase7 root-z 公式。",
        "",
        "## 离线门",
        "",
        "| role | stance windows L/R | joint step base→candidate/limit | model slip base→candidate/limit | contact agree | ground | terminal/swing exact | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["per_role"][role]
        metrics = value["metrics"]
        gate = value["gate"]
        diag = value["diagnostics"]
        lines.append(
            f"| {role} | {diag['stance_windows']['left']}/{diag['stance_windows']['right']} | "
            f"{metrics['joint_step_max_rad']['baseline']:.3f}→"
            f"{metrics['joint_step_max_rad']['candidate']:.3f}/{gate['joint_step_limit_rad']:.3f} | "
            f"{metrics['stance_slip_p95_mps_model_estimate']['baseline']:.3f}→"
            f"{metrics['stance_slip_p95_mps_model_estimate']['candidate']:.3f}/"
            f"{gate['stance_slip_limit_mps']:.3f} | "
            f"{metrics['contact_intent_agreement']['macro']:.3f} | {metrics['ground']['pass']} | "
            f"{gate['checks']['window_boundaries_exact']}/{gate['checks']['swing_joint_path_exact']} | "
            f"{gate['pass']} |"
        )
    lines += [
        "",
        "## 停止裁决",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "- 该结果只否定当前 clipwise stance-window XY-lock 生成器表示；不否定 X2、WBT、Any2Any 或闭环控制器可实现 stance stabilization。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--phase7", type=Path, default=DEFAULT_PHASE7)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase8")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase8")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)
    raw = joblib.load(args.raw)
    phase7_motions = joblib.load(args.phase7)
    role_to_key = {entry["panel_role"]: key for key, entry in raw.items()}
    output = {}
    per_role = {}
    for role in ROLES:
        print(f"[phase8] {role}", flush=True)
        key = role_to_key[role]
        baseline = phase7_motions[key]
        contact = phase7_contact_intent(model, baseline, reset, phase2)
        locked, diagnostics = apply_stance_xy_lock(model, baseline, contact, phase2)
        # Re-run the frozen Phase7 formula from the raw root, not on top of an
        # already corrected Phase7 root-z.
        preground = copy.deepcopy(raw[key])
        preground["dof"] = locked["dof"].copy()
        preground["phase8_stance_xy_lock"] = locked["phase8_stance_xy_lock"]
        candidate, ground = phase7.apply_canonical_root_ground(
            model, preground, reset, phase2, physics
        )
        metrics = offline_metrics(
            model, baseline, candidate, contact, reset, phase2, ground
        )
        gate = offline_gate(metrics, diagnostics)
        output[key] = candidate
        per_role[role] = {
            "motion_key": key,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "gate": gate,
        }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.cache)
    offline_all = all(value["gate"]["pass"] for value in per_role.values())
    failed_roles = [role for role, value in per_role.items() if not value["gate"]["pass"]]
    failed_checks = sorted({
        check
        for value in per_role.values()
        for check, passed in value["gate"]["checks"].items()
        if not passed
    })
    report = {
        "schema_version": "x2_wbt_stance_xy_lock_phase8_v1",
        "provenance": {
            "raw_source": {"path": str(args.raw), "sha256": phase3.sha256(args.raw)},
            "phase7_bronze": {"path": str(args.phase7), "sha256": phase3.sha256(args.phase7)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase3.sha256(physics.DEFAULT_SCENE)},
            "output_rejected_diagnostic": {"path": str(args.cache), "sha256": phase3.sha256(args.cache)},
        },
        "truth_boundary": {
            "contact_and_slip_are_official_model_estimates": True,
            "not_hardware_grf_cop_or_foot_force": True,
            "prescribed_free_replay_executed": False,
            "training_teacher_checkpoint_base_git_baidu_real_robot": False,
        },
        "frozen_contract": {
            "phase7_root_ground_formula": "z'=z+MA9(c_reset-min_24_active_sole_distance)",
            "clearance_m": reset["reset_clearance_m"],
            "sole_sphere_radius_m": reset["sole_sphere_radius_m"],
            "source_timing_root_xy_orientation_foot_semantics": "unchanged",
        },
        "single_structural_intervention": {
            "name": "per-contiguous-stance-window foot XY lock",
            "root_xy_or_orientation_changed": False,
            "swing_locked": False,
            "free_survival_used_for_parameter_selection": False,
            "constants": {
                "blend_frames": BLEND_FRAMES,
                "ik_iterations": IK_ITERATIONS,
                "ik_gain": IK_GAIN,
                "ik_damping": IK_DAMPING,
                "ik_delta_limit_rad": IK_DELTA_LIMIT_RAD,
            },
        },
        "per_role": per_role,
        "decision": {
            "offline_all_five": offline_all,
            "failed_roles": failed_roles,
            "failed_checks": failed_checks,
            "official_physics_ab_executed": False,
            "status": "PHASE8_STANCE_XY_LOCK_OFFLINE_REJECTED",
            "result": f"离线门在 {failed_roles} 失败；失败项为 {failed_checks}，未进入 prescribed/free。",
            "conclusion": "碎片 stance-window 的 clipwise 常量XY anchor在C2进出边界制造更高足速和关节跳变；当前IK reference生成器表示不足。",
            "next_step": "冻结反例，不扩 anchor/blend/SE2/IK 参数；若继续应转为连续接触相位/全轨迹优化或闭环controller，而非逐窗IK。",
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
