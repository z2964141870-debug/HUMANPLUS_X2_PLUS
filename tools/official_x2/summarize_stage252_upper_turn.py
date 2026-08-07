#!/usr/bin/env python3
"""Summarize upper-body disturbance during official-X2 left/right turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_group(root: Path, pattern: str, repeats: int) -> list[dict]:
    paths = [root / pattern.format(repeat=i) for i in range(1, repeats + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths]


def aggregate(runs: list[dict]) -> dict:
    result = {
        "runs": len(runs),
        "full_gate_passes": sum(bool(run.get("full_gate_pass")) for run in runs),
        "fallback_steps_total": sum(int(run.get("upper_fallback_steps", 0)) for run in runs),
    }
    for metric in (
        "turn_yaw_progress_ratio",
        "move_body_vx_mean_mps",
        "root_tilt_max_rad",
        "stop_settle_time_s",
        "upper_tracking_rmse_rad",
    ):
        values = [float(run[metric]) for run in runs if run.get(metric) is not None]
        if values:
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_min"] = float(min(values))
            result[f"{metric}_max"] = float(max(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    patterns = {
        "fixed_right": "stage246_rsl_actor_clip_turn_progress_right_r{repeat}.json",
        "fixed_left": "stage247_rsl_actor_clip_turn_progress_left_mirrorfix_r{repeat}.json",
        "slow_right": "stage252_upper_slow_right_r{repeat}.json",
        "slow_left": "stage252_upper_slow_left_r{repeat}.json",
        "fast_right": "stage252_upper_fast_right_r{repeat}.json",
        "fast_left": "stage252_upper_fast_left_r{repeat}.json",
    }
    groups = {
        name: aggregate(load_group(args.result_root, pattern, args.repeats))
        for name, pattern in patterns.items()
    }
    all_pass = all(group["full_gate_passes"] == args.repeats for group in groups.values())
    no_fallback = all(group["fallback_steps_total"] == 0 for group in groups.values())
    result = {
        "stage": "stage252_upper_turn_panel",
        "domain": "AimDK X2 v1.0 official MuJoCo",
        "upper_contract": "real X2-retargeted 14DOF swing-arms; relative scale 0.25; bounded excursion and slew",
        "summary": groups,
        "all_profiles_full_gate_pass": all_pass,
        "no_upper_health_fallback": no_fallback,
        "panel_pass": bool(all_pass and no_fallback),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage252：官方 X2 MuJoCo 上肢扰动转向门禁",
        "",
        "固定后端与转向反馈，只改变上肢轨迹速度。",
        "",
        "| condition | full gate | fallback | yaw ratio median [min,max] | body vx median | tilt median | stop settle median | upper RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in groups.items():
        lines.append(
            f"| {name} | {row['full_gate_passes']}/{row['runs']} | {row['fallback_steps_total']} | "
            f"{row.get('turn_yaw_progress_ratio_median', float('nan')):.3f} "
            f"[{row.get('turn_yaw_progress_ratio_min', float('nan')):.3f},{row.get('turn_yaw_progress_ratio_max', float('nan')):.3f}] | "
            f"{row.get('move_body_vx_mean_mps_median', float('nan')):.3f} | "
            f"{row.get('root_tilt_max_rad_median', float('nan')):.3f} | "
            f"{row.get('stop_settle_time_s_median', float('nan')):.2f} | "
            f"{row.get('upper_tracking_rmse_rad_median', float('nan')):.3f} |"
        )
    lines += [
        "",
        f"结论：{'通过' if result['panel_pass'] else '未通过'}。",
        "",
        "通过仅表示当前幅值与速度界限内，左右转和停车可承受该上肢扰动；不外推到任意大幅高速手臂动作。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["panel_pass"] else 2)


if __name__ == "__main__":
    main()
