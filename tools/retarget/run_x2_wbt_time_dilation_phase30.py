#!/usr/bin/env python3
"""Phase30: one-variable 1.46x time-dilation existence test.

The Phase29 spatial repairs are immutable inputs.  The only intervention is a
single global duration factor applied to all three train motions.  Translation,
joint paths, auxiliary point paths, and quaternion orientation are resampled at
30 Hz over normalized phase; contacts are nearest-neighbour resampled over the
same normalized phase.  No spatial optimization, physics, policy, PPO, or
real-robot operation is performed.
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


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29


TIME_DILATION = 1.46
TARGET_FPS = 30
OUTPUT_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "silver_contact/phase30_time_dilation"
)
OUTPUT_CACHE = OUTPUT_ROOT / "x2_phase30_time_dilation_1p46.pkl"
REPORT_JSON = REPO / "reports/retarget/x2_wbt_time_dilation_phase30.json"
REPORT_MD = REPO / "reports/retarget/x2_wbt_time_dilation_phase30.md"


def linear_resample(values: np.ndarray, new_frames: int) -> np.ndarray:
    values = np.asarray(values)
    old_phase = np.linspace(0.0, 1.0, len(values))
    new_phase = np.linspace(0.0, 1.0, new_frames)
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(new_phase, old_phase, flat[:, index]) for index in range(flat.shape[1])], axis=1)
    return result.reshape((new_frames,) + values.shape[1:]).astype(values.dtype)


def quaternion_resample_xyzw(values: np.ndarray, new_frames: int) -> np.ndarray:
    values = np.asarray(values)
    old_phase = np.linspace(0.0, 1.0, len(values))
    new_phase = np.linspace(0.0, 1.0, new_frames)
    result = Slerp(old_phase, Rotation.from_quat(values.astype(np.float64)))(new_phase).as_quat()
    # Slerp may choose an equivalent sign; normalize and preserve exact endpoints.
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    result[0] = values[0]
    result[-1] = values[-1]
    return result.astype(values.dtype)


def nearest_phase_resample(values: np.ndarray, new_frames: int) -> np.ndarray:
    values = np.asarray(values)
    indices = np.rint(np.linspace(0, len(values)-1, new_frames)).astype(np.int64)
    return values[indices]


def dilation_rationale(phase29_report: dict[str, Any]) -> dict[str, Any]:
    lunge = next(row for row in phase29_report["motions"] if row["id"] == "PHUMA-LUNGE-R-001")
    bronze = lunge["repair_audit"]["bronze"]["metrics"]
    silver = lunge["repair_audit"]["silver"]["metrics"]
    qstep_lower = float(bronze["joint_step_p95_rad"] / 0.10)
    root_acc_lower = float(np.sqrt(silver["root_horizontal_acceleration_p95_mps2"] / 4.0))
    timing_worst = max(silver["contact_timing_error_p95_s"].values())
    timing_upper = float(0.10 / timing_worst)
    feasible_lower = max(qstep_lower, root_acc_lower)
    return {
        "selection_motion": "PHUMA-LUNGE-R-001",
        "selection_split": "train_candidate",
        "held_out_used_for_selection": False,
        "qstep_scaling_assumption": "per-frame step approximately scales as 1/dilation",
        "root_acceleration_scaling_assumption": "finite-difference acceleration approximately scales as 1/dilation^2",
        "contact_timing_scaling_assumption": "seconds between normalized-phase events approximately scales as dilation",
        "qstep_lower_bound": qstep_lower,
        "root_acceleration_lower_bound": root_acc_lower,
        "feasible_lower_bound": feasible_lower,
        "contact_timing_upper_bound": timing_upper,
        "midpoint": float(0.5 * (feasible_lower + timing_upper)),
        "selected_global_factor": TIME_DILATION,
        "selected_inside_preregistered_interval": bool(feasible_lower <= TIME_DILATION <= timing_upper),
    }


def dilate_entry(
    entry: dict[str, Any], row: dict[str, Any], model: mujoco.MjModel,
    joint_axes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    old_frames = len(entry["dof"])
    new_frames = int(round((old_frames - 1) * TIME_DILATION)) + 1
    result = copy.deepcopy(entry)
    for key, value in list(entry.items()):
        if not isinstance(value, np.ndarray) or value.ndim == 0 or len(value) != old_frames:
            continue
        if key == "root_rot":
            result[key] = quaternion_resample_xyzw(value, new_frames)
        elif key == "pose_aa":
            continue
        else:
            result[key] = linear_resample(value, new_frames)
    dof = np.asarray(result["dof"])
    pose = np.zeros((new_frames, len(result["joint_names_mujoco"]) + 1, 3), dtype=np.float32)
    pose[:, 1:, :] = dof[:, :, None].astype(np.float32) * joint_axes[None]
    result["pose_aa"] = pose
    result["fps"] = TARGET_FPS

    # Freeze the labels that Phase29's actual Bronze/Silver evaluator saw.
    # These differ from the source-stance schedule used inside the solver; the
    # 1.46 interval came from evaluator timing, so changing contracts here
    # would invalidate the preregistration.
    source_contract = phase29.source_contract(row, entry, model)
    source_contact = source_contract["contact"]
    resampled_contact = {side: nearest_phase_resample(source_contact[side], new_frames) for side in phase29.SIDES}
    result["phase30_time_dilation"] = {
        "only_variable": "global time dilation",
        "requested_factor": TIME_DILATION,
        "effective_factor": float((new_frames-1)/(old_frames-1)),
        "old_frames": old_frames, "new_frames": new_frames, "fps": TARGET_FPS,
        "spatial_optimization_iterations": 0,
        "translation_and_joint_interpolation": "linear over normalized phase",
        "root_orientation_interpolation": "scipy Rotation/Slerp in xyzw",
        "contact_label_interpolation": "nearest neighbour over normalized phase",
        "contact_label_basis": "Phase29 Bronze/Silver evaluation labels computed once before dilation",
        "resampled_source_contact_phases": {
            side: [[int(start), int(end)] for start, end in phase28.contiguous_true(resampled_contact[side])]
            for side in phase29.SIDES
        },
        "resampled_source_contact_transitions": {side: int(len(phase28.transitions(resampled_contact[side]))) for side in phase29.SIDES},
        "frame_or_segment_deleted": False,
        "root_fixed": False,
        "forced_double_contact": False,
        "source_phase_order_preserved": True,
        "root_xy_path_modified_beyond_phase_resampling": False,
    }
    return result, {
        "old_frames": old_frames, "new_frames": new_frames,
        "old_duration_s": float((old_frames-1)/TARGET_FPS),
        "new_duration_s": float((new_frames-1)/TARGET_FPS),
        "effective_factor": float((new_frames-1)/(old_frames-1)),
        "endpoint_dof_max_abs_error": float(max(np.max(np.abs(result["dof"][0]-entry["dof"][0])), np.max(np.abs(result["dof"][-1]-entry["dof"][-1])))),
        "endpoint_root_max_abs_error_m": float(max(np.max(np.abs(result["root_trans_offset"][0]-entry["root_trans_offset"][0])), np.max(np.abs(result["root_trans_offset"][-1]-entry["root_trans_offset"][-1])))),
        "endpoint_quaternion_equivalent_error": float(max(1.0-abs(np.dot(result["root_rot"][0], entry["root_rot"][0])), 1.0-abs(np.dot(result["root_rot"][-1], entry["root_rot"][-1])))),
        "source_contact_phase_count_before": {side: len(phase28.contiguous_true(source_contact[side])) for side in phase29.SIDES},
        "source_contact_phase_count_after": {side: len(phase28.contiguous_true(resampled_contact[side])) for side in phase29.SIDES},
        "source_contact_transition_count_before": {side: int(len(phase28.transitions(source_contact[side]))) for side in phase29.SIDES},
        "source_contact_transition_count_after": {side: int(len(phase28.transitions(resampled_contact[side]))) for side in phase29.SIDES},
    }, resampled_contact


def audit_with_frozen_contact(
    row: dict[str, Any], entry: dict[str, Any], model: mujoco.MjModel,
    reset: dict[str, Any], gates: dict[str, Any], mirror: dict[str, Any],
    source_contact: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Reuse Phase28 gates but replace re-inferred contact with frozen Phase29 labels."""
    audit = phase28.audit_tier(row, entry, model, reset, gates, mirror)
    kin = phase28.target_kinematics(entry, model)
    target_contact = {
        side: np.asarray(kin["official_collision_contact"][side], dtype=bool)
        for side in phase29.SIDES
    }
    fps = float(entry["fps"])
    intended = [side for side in phase29.SIDES if len(phase28.transitions(source_contact[side])) >= 2]
    metrics = audit["silver"]["metrics"]
    metrics["source_intended_feet"] = intended
    metrics["contact_timing_error_p95_s"] = {
        side: phase28.event_timing_error(source_contact[side], target_contact[side], fps)
        for side in phase29.SIDES
    }
    metrics["source_contact_override_provenance"] = (
        "nearest normalized-phase resample of frozen Phase29 Bronze/Silver evaluation labels; "
        "not re-inferred from repaired geometry and not real GRF/COP/force"
    )
    metrics["source_intent_contact_ratio"] = {
        side: float(np.mean(source_contact[side])) for side in phase29.SIDES
    }
    metrics["official_geometry_contact_ratio"] = {
        side: float(np.mean(target_contact[side])) for side in phase29.SIDES
    }
    metrics["intent_official_frame_agreement"] = {
        side: float(np.mean(source_contact[side] == target_contact[side])) for side in phase29.SIDES
    }
    metrics["official_signed_distance_threshold_m"] = 0.0
    metrics["collision_signed_distance_exact"] = bool(kin["collision_signed_distance_exact"])
    s = gates["silver_contact"]["metrics"]
    is_static_exception = row["category"] in ("standing", "upper_only")
    event_consistent = bool(intended) and all(
        metrics["target_contact_transitions"][side] >= s["contact_transitions_per_intended_foot"]["min"]
        and metrics["contact_timing_error_p95_s"][side] <= s["contact_timing_error_s"]["max"]
        for side in intended
    )
    checks = {
        "bronze_prerequisite": audit["bronze"]["pass"],
        "dynamic_not_static_exception": not is_static_exception,
        "complete_cycle": metrics["complete_ds_ss_ds_cycles"] >= s["complete_ds_ss_ds_cycles"]["min"],
        "intended_feet_exist": bool(intended),
        "transitions_each_intended": bool(intended) and all(metrics["target_contact_transitions"][side] >= s["contact_transitions_per_intended_foot"]["min"] for side in intended),
        "stance_speed_each": all(value <= s["stance_speed_p95_mps_each_foot"]["max"] for value in metrics["stance_speed_p95_mps"].values()),
        "stance_excursion_each": all(value <= s["stance_excursion_m_each_stance"]["max"] for value in metrics["stance_excursion_max_m"].values()),
        "clearance_p50_each_intended": bool(intended) and all(metrics["swing_clearance_m"][side]["p50"] >= s["swing_clearance_p50_m_each_intended_foot"]["min"] for side in intended),
        "clearance_p95_each_intended": bool(intended) and all(metrics["swing_clearance_m"][side]["p95"] >= s["swing_clearance_p95_m_each_intended_foot"]["min"] for side in intended),
        "timing_each_intended": bool(intended) and all(metrics["contact_timing_error_p95_s"][side] <= s["contact_timing_error_s"]["max"] for side in intended),
        "intent_official_consistent_under_existing_event_gates": event_consistent,
        "official_collision_matches_signed_distance": bool(kin["collision_signed_distance_exact"]),
        "flight": metrics["unintended_flight_fraction"] <= s["unintended_flight_fraction"]["max"],
        "root_acceleration": metrics["root_horizontal_acceleration_p95_mps2"] <= s["root_horizontal_acceleration_p95_mps2"]["max"],
    }
    silver_pass = bool(all(checks.values()))
    audit["silver"]["checks"] = checks
    audit["silver"]["pass"] = silver_pass
    failed = [f"global:{key}" for key, value in audit["global_reject_checks"].items() if not value]
    failed += [f"bronze:{key}" for key, value in audit["bronze"]["checks"].items() if not value]
    if audit["bronze"]["pass"] and not is_static_exception:
        failed += [f"silver:{key}" for key, value in checks.items() if not value]
    elif is_static_exception:
        failed.append("silver:static_exception_not_dynamic_silver")
    audit["tier"] = "Silver" if silver_pass else "Bronze" if audit["bronze"]["pass"] else "Reject"
    audit["reject_reasons"] = failed
    return audit


def write_md(report: dict[str, Any], path: Path) -> None:
    rationale = report["factor_selection"]
    lines = [
        "# X2 WBT Phase30：统一 1.46× time dilation", "",
        "## 裁决", "",
        f"- Phase30 tier：Silver `{report['summary']['silver_count']}/3`、Bronze `{report['summary']['bronze_count']}/3`、Reject `{report['summary']['reject_count']}/3`。",
        f"- candidate冻结：`{report['decision']['freeze_candidate']}`；本阶段 physics/PPO/policy optimizer/真机均为 `0`。",
        "- 即使产生 Silver，也只是静态 FK/contact Silver，不是官方 free-root Gold 或可部署策略。", "",
        "## 1.46× 选择依据", "",
        "只使用 train `PHUMA-LUNGE-R-001` 的 Phase29 门值，不使用 held-out。按时间缩放下界：",
        f"- qstep：`0.14117/0.10 = {rationale['qstep_lower_bound']:.4f}`；root acceleration：`sqrt(5.969/4) = {rationale['root_acceleration_lower_bound']:.4f}`。",
        f"- contact timing 上界：`0.10/0.0667 = {rationale['contact_timing_upper_bound']:.4f}`。",
        f"- 可行区间约 `[{rationale['feasible_lower_bound']:.4f}, {rationale['contact_timing_upper_bound']:.4f}]`，中点 `{rationale['midpoint']:.4f}`，预注册统一取 `1.46`。", "",
        "## 结果", "",
        "| motion | frames old→new | Phase29→30 tier | qstep p95 | root acc | timing L/R | stance speed L/R | rejects |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report["motions"]:
        b = row["phase30_audit"]["bronze"]["metrics"]
        s = row["phase30_audit"]["silver"]["metrics"]
        lines.append(
            f"| `{row['id']}` | {row['resample']['old_frames']}→{row['resample']['new_frames']} | {row['phase29_audit']['tier']}→{row['phase30_audit']['tier']} | "
            f"{b['joint_step_p95_rad']:.4f} | {s['root_horizontal_acceleration_p95_mps2']:.3f} | "
            f"{s['contact_timing_error_p95_s']['left']:.3f}/{s['contact_timing_error_p95_s']['right']:.3f} | "
            f"{s['stance_speed_p95_mps']['left']:.3f}/{s['stance_speed_p95_mps']['right']:.3f} | `{' ; '.join(row['phase30_audit']['reject_reasons'])}` |"
        )
    lines += ["", "## 结论", "", report["decision"]["conclusion"], "", "## 下一步", "", report["decision"]["next_step"]]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-cache", type=Path, default=OUTPUT_CACHE)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()

    phase29_report = json.loads(phase29.REPORT_JSON.read_text())
    rationale = dilation_rationale(phase29_report)
    if not rationale["selected_inside_preregistered_interval"]:
        raise RuntimeError("1.46 is outside the train-only preregistered feasibility interval")
    cache = joblib.load(phase29.OUTPUT_CACHE)
    panel = json.loads(phase28.PANEL.read_text())
    rows_by_id = {row["id"]: row for row in panel["motions"]}
    gates = json.loads(phase28.TIER_GATES.read_text())
    mirror = json.loads(phase28.MIRROR_CONTRACT.read_text())
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase30_offline")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)
    joint_axes = phase28.amass_adapter.parse_joint_axes(phase28.OFFICIAL_MJCF)
    phase29_by_id = {row["id"]: row for row in phase29_report["motions"]}

    outputs, records = {}, []
    for motion_id in phase29.MOTION_IDS:
        row = rows_by_id[motion_id]
        source = cache[motion_id]
        candidate, resample, source_contact = dilate_entry(source, row, model, joint_axes)
        audit = audit_with_frozen_contact(row, candidate, model, reset, gates, mirror, source_contact)
        outputs[motion_id] = candidate
        records.append({
            "id": motion_id, "split": row["recommended_split"], "source_sha256": row["source_sha256"],
            "phase29_entry_sha256": phase28.array_hash(source), "phase30_entry_sha256": phase28.array_hash(candidate),
            "resample": resample, "phase29_audit": phase29_by_id[motion_id]["repair_audit"], "phase30_audit": audit,
            "phase30_contract": candidate["phase30_time_dilation"],
        })
        print(f"[phase30] {motion_id} frames={resample['old_frames']}->{resample['new_frames']} tier={audit['tier']}", flush=True)

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(outputs, args.output_cache, compress=True)
    silver = [row["id"] for row in records if row["phase30_audit"]["tier"] == "Silver"]
    freeze = bool(silver)
    conclusion = (
        f"统一1.46×时间扩展在 {len(silver)}/3 条 train 动作产生静态 Silver，单变量 existence test 通过；冻结该候选供独立评审，但它尚不是Gold、物理稳定或可部署证明。"
        if freeze else
        "统一1.46×时间扩展未产生任何 train Silver；按门停止，不扫描其他因子。这否定该单一time-dilation候选，不否定空间repair或闭环策略。"
    )
    report = {
        "schema_version": "x2_wbt_time_dilation_phase30_v1",
        "mode": "offline_resample_FK_collision_no_physics_no_PPO",
        "truth_boundary": {"contact_and_fk_are_model_estimates": True, "not_real_grf_cop_force": True, "physics_steps": 0, "ppo_updates": 0, "policy_optimizer_steps": 0, "spatial_optimization_iterations": 0, "real_robot": False, "silver_is_not_gold": True},
        "factor_selection": rationale,
        "provenance": {
            "phase29_cache": {"path": str(phase29.OUTPUT_CACHE), "sha256": phase28.sha256(phase29.OUTPUT_CACHE)},
            "phase29_report": {"path": str(phase29.REPORT_JSON), "sha256": phase28.sha256(phase29.REPORT_JSON)},
            "tier_gates": {"path": str(phase28.TIER_GATES), "sha256": phase28.sha256(phase28.TIER_GATES)},
            "official_mjcf": {"path": str(phase28.OFFICIAL_MJCF), "sha256": phase28.sha256(phase28.OFFICIAL_MJCF)},
            "output_cache": {"path": str(args.output_cache)},
        },
        "motions": records,
        "summary": {"completed": len(records), "silver_count": len(silver), "silver_ids": silver, "bronze_count": sum(row["phase30_audit"]["tier"] == "Bronze" for row in records), "reject_count": sum(row["phase30_audit"]["tier"] == "Reject" for row in records)},
        "decision": {"freeze_candidate": freeze, "minimum_one_train_silver": freeze, "static_silver_only_not_gold": True, "conclusion": conclusion, "next_step": "Stop for review; do not automatically run physics or PPO." if freeze else "Stop this factor; do not scan time dilation."},
    }
    report["provenance"]["output_cache"]["sha256"] = phase28.sha256(args.output_cache)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_md(report, args.report_md)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
