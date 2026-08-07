#!/usr/bin/env python3
"""Verify the stronger left-turn feedback across fixed/slow/fast arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for profile in ("fixed", "slow", "fast"):
        paths = [
            args.result_root / f"stage253_left_yawg2p5_{profile}_r{i}.json"
            for i in range(1, args.repeats + 1)
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        runs = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths]
        ratios = [float(run["turn_yaw_progress_ratio"]) for run in runs]
        tilts = [float(run["root_tilt_max_rad"]) for run in runs]
        settles = [float(run["stop_settle_time_s"]) for run in runs]
        groups[profile] = {
            "runs": len(runs),
            "full_gate_passes": sum(bool(run.get("full_gate_pass")) for run in runs),
            "fallback_steps_total": sum(int(run.get("upper_fallback_steps", 0)) for run in runs),
            "turn_yaw_progress_ratio_median": float(np.median(ratios)),
            "turn_yaw_progress_ratio_min": float(min(ratios)),
            "turn_yaw_progress_ratio_max": float(max(ratios)),
            "root_tilt_max_rad_median": float(np.median(tilts)),
            "stop_settle_time_s_median": float(np.median(settles)),
        }
    panel_pass = all(
        group["full_gate_passes"] == args.repeats and group["fallback_steps_total"] == 0
        for group in groups.values()
    )
    result = {
        "stage": "stage253_left_yaw_gain_panel",
        "intervention": "left turn-progress feedback gain 2.0 -> 2.5 only",
        "frozen": "actor, upper motion, PD, gait template, thresholds",
        "summary": groups,
        "panel_pass": panel_pass,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage253：快摆臂左转反馈修复",
        "",
        "Stage252 快摆臂左转为 2/3；本实验只把左转进度反馈增益从 2.0 提高到 2.5。",
        "",
        "| upper profile | full gate | fallback | yaw ratio median [min,max] | tilt median | stop settle median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in groups.items():
        lines.append(
            f"| {name} | {row['full_gate_passes']}/{row['runs']} | {row['fallback_steps_total']} | "
            f"{row['turn_yaw_progress_ratio_median']:.3f} [{row['turn_yaw_progress_ratio_min']:.3f},{row['turn_yaw_progress_ratio_max']:.3f}] | "
            f"{row['root_tilt_max_rad_median']:.3f} | {row['stop_settle_time_s_median']:.2f} |"
        )
    lines += [
        "",
        f"结论：{'通过，增益 2.5 可作为当前左转鲁棒配置。' if panel_pass else '未通过，不能采用该增益。'}",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if panel_pass else 2)


if __name__ == "__main__":
    main()
