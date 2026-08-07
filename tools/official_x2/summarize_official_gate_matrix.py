#!/usr/bin/env python3
"""Aggregate repeat-level official X2 MuJoCo gates into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = (
    "startup_root_z_min_m",
    "startup_tilt_max_rad",
    "startup_forward_displacement_m",
    "startup_backward_excursion_m",
    "move_forward_displacement_m",
    "move_lateral_displacement_m",
    "move_heading_max_deviation_rad",
    "move_yaw_progress_rad",
    "turn_yaw_progress_ratio",
    "move_body_vx_mean_mps",
    "stop_settle_time_s",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    run_pattern = re.compile(
        rf"^{re.escape(args.prefix)}_(straight|turn_left|turn_right)_r([0-9]+)$"
    )
    paths = sorted(
        path
        for path in args.result_root.glob(f"{args.prefix}_*.json")
        if run_pattern.match(path.stem)
    )
    rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        match = run_pattern.match(path.stem)
        assert match is not None
        skill = match.group(1)
        row = {
            "file": str(path),
            "skill": skill,
            "startup_gate_pass": summary.get("startup_gate_pass"),
            "move_gate_pass": summary.get("move_gate_pass"),
            "stop_gate_pass": summary.get("stop_gate_pass"),
            "full_gate_pass": summary.get("full_gate_pass"),
        }
        row.update({field: summary.get(field) for field in FIELDS})
        rows.append(row)

    skills = {}
    for skill in sorted({row["skill"] for row in rows}):
        group = [row for row in rows if row["skill"] == skill]
        skills[skill] = {
            "runs": len(group),
            "passes": sum(bool(row["full_gate_pass"]) for row in group),
            "pass_rate": (
                sum(bool(row["full_gate_pass"]) for row in group) / len(group) if group else 0.0
            ),
        }
    expected = {"straight", "turn_left", "turn_right"}
    complete = set(skills) == expected and all(v["runs"] >= 2 for v in skills.values())
    matrix_pass = complete and all(v["passes"] == v["runs"] for v in skills.values())
    result = {
        "domain": "aimdk_x2_v1_official_mujoco",
        "prefix": args.prefix,
        "expected_skills": sorted(expected),
        "matrix_complete": complete,
        "matrix_pass": matrix_pass,
        "skills": skills,
        "runs": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# X2 官方 MuJoCo 起步—行走—转向—停车门禁",
        "",
        f"- 域：`{result['domain']}`",
        f"- 矩阵完整：`{complete}`",
        f"- 矩阵通过：`{matrix_pass}`",
        "",
        "| 技能 | 通过/运行 | 通过率 |",
        "|---|---:|---:|",
    ]
    for skill, item in skills.items():
        lines.append(f"| {skill} | {item['passes']}/{item['runs']} | {item['pass_rate']:.0%} |")
    lines += [
        "",
        "每次运行同时要求：站立、起步前 1 秒、移动语义和停车全部通过。起步门禁防止用末段结果掩盖开局下沉、反退或大倾斜。",
        "",
        "| 运行 | 起步 | 移动 | 停车 | 全门 | 前进(m) | 横漂(m) | 航向进展(rad) | 停稳(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value: object, digits: int) -> str:
        return "—" if value is None else f"{float(value):.{digits}f}"

    for row in rows:
        lines.append(
            f"| {Path(row['file']).stem} | {row['startup_gate_pass']} | {row['move_gate_pass']} | "
            f"{row['stop_gate_pass']} | {row['full_gate_pass']} | "
            f"{fmt(row['move_forward_displacement_m'], 3)} | {fmt(row['move_lateral_displacement_m'], 3)} | "
            f"{fmt(row['move_yaw_progress_rad'], 3)} | {fmt(row['stop_settle_time_s'], 2)} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"matrix_complete": complete, "matrix_pass": matrix_pass, "skills": skills}, indent=2))
    raise SystemExit(0 if matrix_pass else 2)


if __name__ == "__main__":
    main()
