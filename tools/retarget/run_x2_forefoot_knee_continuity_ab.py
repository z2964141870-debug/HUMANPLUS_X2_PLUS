#!/usr/bin/env python3
"""Apply the forefoot continuity intervention only to bilateral knees.

The exact-30 smooth sweep showed that every global step offender is a knee,
while smoothing all 31 DOFs worsens contact estimates.  This script keeps the
accepted smooth9 trajectory for 29 DOFs and substitutes only the two knee
channels from smooth11/13/15.  It then reruns the same exact-time semantic and
contact gates.  No IK rerun or training occurs.
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
SMOOTH_SCRIPT = REPO / "tools/retarget/run_x2_forefoot_smoothing_ab.py"
SOURCE_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_smoothing_ab"
)
OUTPUT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_knee_continuity_ab"
)
REPORT = REPO / "reports/retarget/x2_forefoot_knee_continuity_ab.json"
WINDOWS = (11, 13, 15)
KNEES = ("left_knee_joint", "right_knee_joint")

if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_candidate(window: int, output_root: Path) -> Path:
    output = output_root / f"knee_smooth{window}/x2_toe_forefoot_knee_smooth{window}.pkl"
    if output.exists():
        return output
    base = joblib.load(SOURCE_ROOT / "smooth9/x2_toe_forefoot_smooth9.pkl")
    donor = joblib.load(SOURCE_ROOT / f"smooth{window}/x2_toe_forefoot_smooth{window}.pkl")
    if set(base) != set(donor):
        raise AssertionError("smoothing caches have different motion keys")
    result: dict[str, Any] = {}
    for key in base:
        entry = {name: value for name, value in base[key].items()}
        entry["dof"] = np.asarray(base[key]["dof"]).copy()
        entry["pose_aa"] = np.asarray(base[key]["pose_aa"]).copy()
        names = list(entry["joint_names_mujoco"])
        for joint in KNEES:
            index = names.index(joint)
            entry["dof"][:, index] = np.asarray(donor[key]["dof"])[:, index]
            entry["pose_aa"][:, index + 1] = np.asarray(donor[key]["pose_aa"])[:, index + 1]
        entry["continuity_intervention"] = {
            "base_window": 9,
            "bilateral_knee_window": window,
            "untouched_dof_count": 29,
        }
        result[key] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, output, compress=True)
    return output


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Toe→Forefoot Bilateral-Knee Continuity A/B",
        "",
        "## 裁决",
        "",
        f"- first passing knee window: `{report['first_passing_window']}`。",
        "- 29/31 DoF 完全保留 smooth9；仅左右 knee 使用更宽窗口。",
        "",
        "| knee window | semantic p95 | step max/p95-max | valid | L/R slip p95 | L/R clearance p50 | winner |",
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
        "该门仅决定 Bronze 候选；模型接触估计不等于真实 GRF/COP，也不替代物理四域回放。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    smooth = load_module(SMOOTH_SCRIPT, "x2_forefoot_smoothing_helpers")
    task3 = smooth.load_task3()
    paths = {window: make_candidate(window, args.output_root) for window in WINDOWS}
    variants = {window: smooth.evaluate(task3, window, path, args.output_root) for window, path in paths.items()}
    parent_report = json.loads((REPO / "reports/retarget/x2_forefoot_smoothing_ab.json").read_text(encoding="utf-8"))
    current = parent_report["current_v4_reference"]
    toe9 = parent_report["variants"]["9"]
    gate_rows = {window: smooth.gates(variants[window], current, toe9) for window in WINDOWS}
    first = next((window for window in WINDOWS if gate_rows[window]["winner"]), None)
    report = {
        "schema_version": 1,
        "status": "completed_no_training",
        "intervention": {
            "base": "toe_to_forefoot_exact30_smooth9",
            "modified_joints": list(KNEES),
            "knee_windows": list(WINDOWS),
            "untouched_dof_count": 29,
        },
        "variants": {str(key): value for key, value in variants.items()},
        "gates": {str(key): value for key, value in gate_rows.items()},
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
