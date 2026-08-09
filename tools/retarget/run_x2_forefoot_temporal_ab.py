#!/usr/bin/env python3
"""Test whether higher-rate IK removes the toe-to-forefoot joint jump.

Task3 established toe-to-forefoot as a semantic/contact near-winner whose only
pre-registered failure was joint continuity.  This script keeps that exact
contract, solves it at 60 Hz, resamples the free-joint trajectory to exact
30 Hz, then applies the same 9-frame smoothing as the 30 Hz baseline.  No
trajectory repair, clipping, filtering change, or training is introduced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


REPO = Path(__file__).resolve().parents[2]
LEGACY = Path("/home/humanplus/x2_teleop_final")
TASK3_ROOT = LEGACY / "x2_sonic/x2_gmr_data_rebuild/external_agent/task3_foot_target_semantics"
TASK3_SCRIPT = TASK3_ROOT / "run_task3.py"
OUTPUT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_temporal_ab"
)
REPORT = REPO / "reports/retarget/x2_forefoot_temporal_resolution_ab.json"

if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

import general_motion_retargeting.motion_retarget as gmr_motion_retarget
from x2_sonic.tools import retarget_smplx_subset_to_x2_gmr_cache as retarget_tool


def load_task3():
    spec = importlib.util.spec_from_file_location("x2_task3_foot_semantics", TASK3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(TASK3_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def entry_to_qpos(entry: dict[str, Any]) -> np.ndarray:
    frames = len(entry["dof"])
    qpos = np.zeros((frames, 38), dtype=np.float64)
    qpos[:, :3] = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    xyzw = np.asarray(entry["root_rot"], dtype=np.float64)
    qpos[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:38] = np.asarray(entry["dof"], dtype=np.float64)
    return qpos


def retarget_60_to_30(task3, output_root: Path) -> Path:
    cache = output_root / "toe_to_forefoot_60_to_30/x2_toe_to_forefoot_60_to_30.pkl"
    if cache.exists():
        return cache
    panel = json.loads(task3.PANEL.read_text(encoding="utf-8"))
    config = TASK3_ROOT / "smplx_to_x2_toe_to_forefoot_proxy.json"
    model = TASK3_ROOT / "derived_x2_forefoot_proxy.xml"
    if not config.exists() or not model.exists():
        raise FileNotFoundError("Task3 derived forefoot assets are missing")
    original_xml = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(model)
    joint_axes = retarget_tool.parse_joint_axes(model)
    entries: dict[str, Any] = {}
    try:
        for row in panel["selected"]:
            source = task3.INPUT_ROOT / row["relative_path"]
            key, high_rate, _ = retarget_tool.retarget_one(
                smplx_file=source,
                input_root=task3.INPUT_ROOT,
                smplx_models=task3.MODEL,
                joint_axes=joint_axes,
                target_fps=60,
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
            qpos_60 = entry_to_qpos(high_rate)
            qpos_30, resample = retarget_tool.resample_qpos_exact_fps(qpos_60, 60.0, 30)
            entry = retarget_tool.qpos_to_cache_entry(
                qpos=qpos_30,
                joint_axes=joint_axes,
                fps=30,
                source_file=source,
                smooth_window=9,
                root_xy_origin=True,
                root_rotation_mode="full",
                root_z_mode="gmr",
                root_z=0.65,
            )
            entry.update({
                "source_segment_start_s": float(row.get("segment_start_s", 0.0)),
                "source_segment_duration_s": float(row.get("segment_duration_s", 0.0)),
                "panel_role": row["panel_role"],
                "task3_variant": "toe_to_forefoot_60_to_30",
                "temporal_contract": {
                    "ik_solve_fps": 60,
                    "output_fps": 30,
                    "smooth_window_output_frames": 9,
                    "resample": resample,
                },
            })
            entries[key] = entry
    finally:
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original_xml
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(entries, cache, compress=True)
    return cache


def joint_step_report(cache: Path) -> dict[str, Any]:
    motions = joblib.load(cache)
    rows = []
    for key, entry in motions.items():
        dof = np.asarray(entry["dof"], dtype=np.float64)
        step = np.max(np.abs(np.diff(dof, axis=0)), axis=1)
        rows.append({"key": key, "panel_role": entry["panel_role"], **stats(step)})
    return {
        "motions": rows,
        "aggregate": {
            "max_rad": float(max(row["max"] for row in rows)),
            "p95_max_rad": float(max(row["p95"] for row in rows)),
        },
    }


def contact_aggregate(report: dict[str, Any], side: str, field: str, stat: str) -> float | None:
    values = [
        motion[side][field][stat]
        for motion in report["motions"].values()
        if motion["contact_valid_for_comparison"] and motion[side][field]["count"]
    ]
    return float(np.percentile(values, 95)) if values else None


def summarize_candidate(semantic: dict[str, Any], contact: dict[str, Any], steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "foot_semantic_p95_max_m": semantic["aggregate"]["foot"]["matched_semantic_p95_max_m"],
        "joint_step_max_rad": steps["aggregate"]["max_rad"],
        "joint_step_p95_max_rad": steps["aggregate"]["p95_max_rad"],
        "valid_contact_actions": int(sum(m["contact_valid_for_comparison"] for m in contact["motions"].values())),
        "stance_slip_p95_mps": {
            side: contact_aggregate(contact, side, "stance_slip_mps", "p95") for side in task_sides()
        },
        "swing_clearance_p50_m": {
            side: contact_aggregate(contact, side, "swing_clearance_m", "p50") for side in task_sides()
        },
    }


def task_sides() -> tuple[str, str]:
    return ("left", "right")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    old = report["variants"]["toe_to_forefoot_30hz"]
    new = report["variants"]["toe_to_forefoot_60_to_30"]
    current = report["reference_gates"]["current_v4"]
    lines = [
        "# Toe→Forefoot Temporal Resolution A/B",
        "",
        "## 裁决",
        "",
        f"- candidate gate: **{'PASS' if report['selection_checks']['winner'] else 'FAIL'}**。",
        f"- joint step max：`{old['joint_step_max_rad']:.5f} → {new['joint_step_max_rad']:.5f} rad/frame`；current-v4 110% 上限为 `{1.10 * current['joint_step_max_rad']:.5f}`。",
        f"- joint step p95-max：`{old['joint_step_p95_max_rad']:.5f} → {new['joint_step_p95_max_rad']:.5f}`；上限为 `{1.10 * current['joint_step_p95_max_rad']:.5f}`。",
        "- 该实验只改变 IK 内部采样密度；输出仍为 30 Hz、smooth9、full-root、GMR root-z。",
        "",
        "| variant | semantic p95 | step max/p95-max | valid contact | L/R slip p95 | L/R swing p50 |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for name, row in report["variants"].items():
        lines.append(
            f"| `{name}` | {row['foot_semantic_p95_max_m']:.5f} | "
            f"{row['joint_step_max_rad']:.5f}/{row['joint_step_p95_max_rad']:.5f} | "
            f"{row['valid_contact_actions']}/9 | "
            f"{row['stance_slip_p95_mps']['left']}/{row['stance_slip_p95_mps']['right']} | "
            f"{row['swing_clearance_p50_m']['left']}/{row['swing_clearance_p50_m']['right']} |"
        )
    lines += ["", "## 门禁", ""]
    for key, value in report["selection_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "接触、滑移和 clearance 仍是模型几何估计，不是真实足底力/COP；没有训练。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    task3 = load_task3()
    cache = retarget_60_to_30(task3, args.output_root)
    task3.OUT = args.output_root
    candidate_name = "toe_to_forefoot_60_to_30"
    semantic = task3.semantic_report(candidate_name, cache, TASK3_ROOT / "derived_x2_forefoot_proxy.xml")
    contact = task3.contact_report(candidate_name, cache)
    steps = joint_step_report(cache)
    candidate = summarize_candidate(semantic, contact, steps)
    old_summary = json.loads((TASK3_ROOT / "matched_semantic_ab_summary.json").read_text(encoding="utf-8"))
    current = old_summary["variants"]["current"]
    old = old_summary["variants"]["toe_to_forefoot_proxy"]
    checks = {
        "semantic_nonregression_vs_30hz": candidate["foot_semantic_p95_max_m"] <= 1.10 * old["foot_semantic_p95_max_m"],
        "joint_step_max_within_current_v4_110pct": candidate["joint_step_max_rad"] <= 1.10 * current["joint_step_max_rad"],
        "joint_step_p95_within_current_v4_110pct": candidate["joint_step_p95_max_rad"] <= 1.10 * current["joint_step_p95_max_rad"],
        "contact_coverage_7_of_9": candidate["valid_contact_actions"] >= 7,
        "stance_slip_nonregression_vs_current": all(
            candidate["stance_slip_p95_mps"][side] is not None
            and candidate["stance_slip_p95_mps"][side] <= current["stance_slip_p95_mps"][side]
            for side in task_sides()
        ),
        "swing_clearance_not_compressed_vs_current": all(
            candidate["swing_clearance_p50_m"][side] is not None
            and candidate["swing_clearance_p50_m"][side] >= 0.90 * current["swing_clearance_p50_m"][side]
            for side in task_sides()
        ),
    }
    checks["winner"] = bool(all(checks.values()))
    report = {
        "schema_version": 1,
        "status": "completed_no_training",
        "fixed_contract": {
            "foot_semantics": "SMPL-X foot to X2 forefoot proxy",
            "output_fps": 30,
            "smooth_window": 9,
            "root_rotation": "full",
            "root_z": "gmr",
            "solver": "quadprog",
            "damping": 0.5,
        },
        "intervention": {"ik_solve_fps": "30 -> 60", "resample": "exact 60 -> 30 before smooth9"},
        "variants": {
            "toe_to_forefoot_30hz": old,
            "toe_to_forefoot_60_to_30": candidate,
        },
        "reference_gates": {"current_v4": current},
        "selection_checks": checks,
        "candidate_cache": str(cache),
        "contact_truth": "kinematic_model_estimate",
        "training": False,
        "postprocess": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.report.with_suffix(".md"))
    print(json.dumps({"report": str(args.report), "selection_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
