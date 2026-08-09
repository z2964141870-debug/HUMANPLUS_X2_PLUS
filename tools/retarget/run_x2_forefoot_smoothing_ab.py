#!/usr/bin/env python3
"""Test a bounded output-continuity repair for the forefoot near-winner.

The IK contract and 30 Hz solve are frozen.  A raw trajectory is generated
once, then the existing zero-phase moving average is evaluated at windows
9/11/13/15.  Window 9 must numerically reproduce Task3 before any larger
window is judged.  This is an offline kinematic A/B, not training permission.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO = Path(__file__).resolve().parents[2]
LEGACY = Path("/home/humanplus/x2_teleop_final")
TASK3_ROOT = LEGACY / "x2_sonic/x2_gmr_data_rebuild/external_agent/task3_foot_target_semantics"
TASK3_SCRIPT = TASK3_ROOT / "run_task3.py"
OUTPUT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_smoothing_ab"
)
REPORT = REPO / "reports/retarget/x2_forefoot_smoothing_ab.json"
WINDOWS = (9, 11, 13, 15)
SIDES = ("left", "right")

if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

import general_motion_retargeting.motion_retarget as gmr_motion_retarget
from x2_sonic.tools import retarget_smplx_subset_to_x2_gmr_cache as retarget_tool


def load_task3():
    spec = importlib.util.spec_from_file_location("x2_task3_foot_smoothing", TASK3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(TASK3_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_to_qpos(entry: dict[str, Any]) -> np.ndarray:
    frames = len(entry["dof"])
    qpos = np.zeros((frames, 38), dtype=np.float64)
    qpos[:, :3] = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    xyzw = np.asarray(entry["root_rot"], dtype=np.float64)
    qpos[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:38] = np.asarray(entry["dof"], dtype=np.float64)
    return qpos


def build_variants(task3, output_root: Path) -> dict[int, Path]:
    paths = {window: output_root / f"smooth{window}/x2_toe_forefoot_smooth{window}.pkl" for window in WINDOWS}
    if all(path.exists() for path in paths.values()):
        return paths
    panel = json.loads(task3.PANEL.read_text(encoding="utf-8"))
    config = TASK3_ROOT / "smplx_to_x2_toe_to_forefoot_proxy.json"
    model = TASK3_ROOT / "derived_x2_forefoot_proxy.xml"
    original_xml = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(model)
    axes = retarget_tool.parse_joint_axes(model)
    variants: dict[int, dict[str, Any]] = {window: {} for window in WINDOWS}
    try:
        for row in panel["selected"]:
            source = task3.INPUT_ROOT / row["relative_path"]
            key, raw, _ = retarget_tool.retarget_one(
                smplx_file=source,
                input_root=task3.INPUT_ROOT,
                smplx_models=task3.MODEL,
                joint_axes=axes,
                target_fps=30,
                max_frames=0,
                smooth_window=1,
                solver=task3.SOLVER,
                damping=task3.DAMPING,
                root_rotation_mode="full",
                root_z_mode="gmr",
                root_z=0.65,
                ik_config=config,
                segment_start_s=float(row.get("segment_start_s", 0.0)),
                segment_duration_s=float(row.get("segment_duration_s", 0.0)),
            )
            qpos = entry_to_qpos(raw)
            for window in WINDOWS:
                entry = retarget_tool.qpos_to_cache_entry(
                    qpos=qpos,
                    joint_axes=axes,
                    fps=30,
                    source_file=source,
                    smooth_window=window,
                    root_xy_origin=True,
                    root_rotation_mode="full",
                    root_z_mode="gmr",
                    root_z=0.65,
                )
                entry.update({
                    "source_segment_start_s": float(row.get("segment_start_s", 0.0)),
                    "source_segment_duration_s": float(row.get("segment_duration_s", 0.0)),
                    "panel_role": row["panel_role"],
                    "task3_variant": f"toe_to_forefoot_smooth{window}",
                    "continuity_intervention": {"moving_average_window_frames": window},
                })
                variants[window][key] = entry
    finally:
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original_xml
    for window, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(variants[window], path, compress=True)
    return paths


def build_current_exact30(task3, output_root: Path) -> Path:
    cache = output_root / "current_v4_exact30/x2_current_v4_exact30.pkl"
    if cache.exists():
        return cache
    panel = json.loads(task3.PANEL.read_text(encoding="utf-8"))
    config = TASK3_ROOT / "smplx_to_x2_current.json"
    model = task3.FORMAL_X2_MJCF
    original_xml = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(model)
    axes = retarget_tool.parse_joint_axes(model)
    entries: dict[str, Any] = {}
    try:
        for row in panel["selected"]:
            source = task3.INPUT_ROOT / row["relative_path"]
            key, entry, _ = retarget_tool.retarget_one(
                smplx_file=source,
                input_root=task3.INPUT_ROOT,
                smplx_models=task3.MODEL,
                joint_axes=axes,
                target_fps=30,
                max_frames=0,
                smooth_window=9,
                solver=task3.SOLVER,
                damping=task3.DAMPING,
                root_rotation_mode="full",
                root_z_mode="gmr",
                root_z=0.65,
                ik_config=config,
                segment_start_s=float(row.get("segment_start_s", 0.0)),
                segment_duration_s=float(row.get("segment_duration_s", 0.0)),
            )
            entry.update({
                "source_segment_start_s": float(row.get("segment_start_s", 0.0)),
                "source_segment_duration_s": float(row.get("segment_duration_s", 0.0)),
                "panel_role": row["panel_role"],
                "task3_variant": "current_v4_exact30",
            })
            entries[key] = entry
    finally:
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original_xml
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(entries, cache, compress=True)
    return cache


def exact_source_frames(task3, entry: dict[str, Any], frame_count: int) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    source = Path(entry["source_smplx_file"])
    source_data, body_model, output, _ = task3.load_smplx_file_compat(
        source,
        task3.MODEL,
        segment_start_s=float(entry.get("source_segment_start_s", 0.0)),
        segment_duration_s=float(entry.get("source_segment_duration_s", 0.0)),
    )
    native_fps = float(np.asarray(source_data["mocap_frame_rate"]).reshape(()))
    frames, aligned_fps = task3.get_smplx_data_offline_fast(
        source_data, body_model, output, tgt_fps=int(round(native_fps))
    )
    source_time = np.arange(len(frames), dtype=np.float64) / float(aligned_fps)
    target_time = np.arange(frame_count, dtype=np.float64) / 30.0
    if target_time[-1] > source_time[-1] + 1.0e-6:
        target_time = np.minimum(target_time, source_time[-1])
    result = [dict() for _ in range(frame_count)]
    for name in frames[0]:
        positions = np.asarray([frame[name][0] for frame in frames], dtype=np.float64)
        quaternions = np.asarray([frame[name][1] for frame in frames], dtype=np.float64)
        interpolated_pos = np.column_stack([
            np.interp(target_time, source_time, positions[:, dim]) for dim in range(3)
        ])
        interpolated_quat = Slerp(
            source_time, Rotation.from_quat(quaternions, scalar_first=True)
        )(target_time).as_quat(scalar_first=True)
        for index in range(frame_count):
            result[index][name] = (interpolated_pos[index], interpolated_quat[index])
    return result


def set_frame(model: mujoco.MjModel, data: mujoco.MjData, entry: dict[str, Any], index: int) -> None:
    data.qpos[:3] = np.asarray(entry["root_trans_offset"][index], dtype=np.float64)
    xyzw = np.asarray(entry["root_rot"][index], dtype=np.float64)
    data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
    data.qpos[7:38] = np.asarray(entry["dof"][index], dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def exact_semantic_report(task3, cache: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(TASK3_ROOT / "derived_x2_forefoot_proxy.xml"))
    data = mujoco.MjData(model)
    ids = {
        "pelvis": task3.body_id(model, "pelvis"),
        "torso": task3.body_id(model, "torso_link"),
        "left_foot": task3.body_id(model, "left_forefoot_proxy"),
        "right_foot": task3.body_id(model, "right_forefoot_proxy"),
    }
    motions = joblib.load(cache)
    rows = []
    for key, entry in motions.items():
        frame_count = len(entry["dof"])
        frames = exact_source_frames(task3, entry, frame_count)
        source = np.asarray([
            [frames[i]["pelvis"][0], frames[i]["spine3"][0], frames[i]["left_foot"][0], frames[i]["right_foot"][0]]
            for i in range(frame_count)
        ])
        target = []
        for index in range(frame_count):
            set_frame(model, data, entry, index)
            target.append([data.xpos[ids[name]].copy() for name in ("pelvis", "torso", "left_foot", "right_foot")])
        target = np.asarray(target)
        scale, rotation, translation = task3.fit_similarity(source, target)
        error = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=2)
        rows.append({
            "key": key,
            "panel_role": entry["panel_role"],
            "overall_p95_m": float(np.percentile(error, 95)),
            "overall_max_m": float(np.max(error)),
        })
    return {
        "timebase": "source landmarks interpolated from native FPS to exact 30Hz",
        "motions": rows,
        "foot_semantic_p95_max_m": float(max(row["overall_p95_m"] for row in rows)),
    }


def legacy_timebase_audit(exact_cache: Path, old_cache: Path) -> dict[str, Any]:
    exact, old = joblib.load(exact_cache), joblib.load(old_cache)
    rows = []
    for key in sorted(exact):
        exact_frames = len(exact[key]["dof"])
        old_frames = len(old[key]["dof"])
        rows.append({
            "key": key,
            "declared_fps_both": 30,
            "old_frames": old_frames,
            "exact30_frames": exact_frames,
            "frame_ratio_old_over_exact": float(old_frames / exact_frames),
        })
    return {
        "rows": rows,
        "mismatch_count": int(sum(row["old_frames"] != row["exact30_frames"] for row in rows)),
        "max_frame_ratio": float(max(row["frame_ratio_old_over_exact"] for row in rows)),
        "interpretation": "legacy Task3 cache predates exact-FPS resampling and cannot be used as the matched temporal baseline",
    }


def joint_steps(cache: Path) -> dict[str, float]:
    motions = joblib.load(cache)
    rows = []
    for entry in motions.values():
        dof = np.asarray(entry["dof"], dtype=np.float64)
        step = np.max(np.abs(np.diff(dof, axis=0)), axis=1)
        rows.append((float(np.max(step)), float(np.percentile(step, 95))))
    return {"max_rad": max(x[0] for x in rows), "p95_max_rad": max(x[1] for x in rows)}


def contact_aggregate(report: dict[str, Any], side: str, field: str, stat: str) -> float | None:
    values = [
        motion[side][field][stat]
        for motion in report["motions"].values()
        if motion["contact_valid_for_comparison"] and motion[side][field]["count"]
    ]
    return float(np.percentile(values, 95)) if values else None


def evaluate(task3, window: int, cache: Path, output_root: Path) -> dict[str, Any]:
    task3.OUT = output_root
    name = f"smooth{window}"
    semantic = exact_semantic_report(task3, cache)
    contact = task3.contact_report(name, cache)
    step = joint_steps(cache)
    return {
        "foot_semantic_p95_max_m": semantic["foot_semantic_p95_max_m"],
        "joint_step_max_rad": step["max_rad"],
        "joint_step_p95_max_rad": step["p95_max_rad"],
        "valid_contact_actions": int(sum(m["contact_valid_for_comparison"] for m in contact["motions"].values())),
        "stance_slip_p95_mps": {side: contact_aggregate(contact, side, "stance_slip_mps", "p95") for side in SIDES},
        "swing_clearance_p50_m": {side: contact_aggregate(contact, side, "swing_clearance_m", "p50") for side in SIDES},
    }


def gates(candidate: dict[str, Any], current: dict[str, Any], toe9: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "semantic_nonregression_vs_toe_smooth9": candidate["foot_semantic_p95_max_m"] <= 1.10 * toe9["foot_semantic_p95_max_m"],
        "joint_step_max_within_current_v4_110pct": candidate["joint_step_max_rad"] <= 1.10 * current["joint_step_max_rad"],
        "joint_step_p95_within_current_v4_110pct": candidate["joint_step_p95_max_rad"] <= 1.10 * current["joint_step_p95_max_rad"],
        "contact_coverage_7_of_9": candidate["valid_contact_actions"] >= 7,
        "stance_slip_nonregression_vs_current": all(
            candidate["stance_slip_p95_mps"][side] is not None
            and candidate["stance_slip_p95_mps"][side] <= current["stance_slip_p95_mps"][side]
            for side in SIDES
        ),
        "swing_clearance_not_compressed_vs_current": all(
            candidate["swing_clearance_p50_m"][side] is not None
            and candidate["swing_clearance_p50_m"][side] >= 0.90 * current["swing_clearance_p50_m"][side]
            for side in SIDES
        ),
    }
    checks["winner"] = bool(all(checks.values()))
    return checks


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Toe→Forefoot Output Smoothing A/B",
        "",
        "## 裁决",
        "",
        f"- 旧 Task3 timebase 不匹配动作数：`{report['legacy_task3_timebase_audit']['mismatch_count']}/9`；最大帧数比例 `{report['legacy_task3_timebase_audit']['max_frame_ratio']:.4f}`。",
        f"- first passing window: `{report['first_passing_window']}`。",
        "- IK、root 与 contact contract 固定；唯一变量是已有对称 moving-average 的窗口。",
        "",
        "| window | semantic p95 | step max/p95-max | valid | L/R slip p95 | L/R clearance p50 | winner |",
        "| ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for window in WINDOWS:
        row = report["variants"][str(window)]
        gate = report["gates"][str(window)]
        lines.append(
            f"| {window} | {row['foot_semantic_p95_max_m']:.5f} | "
            f"{row['joint_step_max_rad']:.5f}/{row['joint_step_p95_max_rad']:.5f} | "
            f"{row['valid_contact_actions']}/9 | "
            f"{row['stance_slip_p95_mps']['left']:.4f}/{row['stance_slip_p95_mps']['right']:.4f} | "
            f"{row['swing_clearance_p50_m']['left']:.4f}/{row['swing_clearance_p50_m']['right']:.4f} | "
            f"{'PASS' if gate['winner'] else 'FAIL'} |"
        )
    lines += [
        "",
        "这仍是 Bronze 运动学筛选；接触量是模型估计值，必须再过物理四域才可称 Silver。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    task3 = load_task3()
    paths = build_variants(task3, args.output_root)
    old_cache = TASK3_ROOT / "toe_to_forefoot_proxy/x2_toe_to_forefoot_proxy.pkl"
    current_cache = build_current_exact30(task3, args.output_root)
    timebase = legacy_timebase_audit(paths[9], old_cache)
    variants = {window: evaluate(task3, window, path, args.output_root) for window, path in paths.items()}
    current = evaluate(task3, 9, current_cache, args.output_root / "current_exact30_eval")
    gate_rows = {window: gates(variants[window], current, variants[9]) for window in WINDOWS}
    first = next((window for window in WINDOWS if gate_rows[window]["winner"]), None)
    report = {
        "schema_version": 1,
        "status": "completed_no_training",
        "fixed_contract": {"ik_solve_fps": 30, "output_fps": 30, "root_rotation": "full", "root_z": "gmr", "foot_semantics": "toe_to_forefoot_proxy"},
        "sweep": {"moving_average_windows_frames": list(WINDOWS)},
        "legacy_task3_timebase_audit": timebase,
        "variants": {str(key): value for key, value in variants.items()},
        "gates": {str(key): value for key, value in gate_rows.items()},
        "current_v4_reference": current,
        "first_passing_window": first,
        "training": False,
        "contact_truth": "kinematic_model_estimate",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.report.with_suffix(".md"))
    print(json.dumps({"report": str(args.report), "first_passing_window": first}, indent=2))


if __name__ == "__main__":
    main()
