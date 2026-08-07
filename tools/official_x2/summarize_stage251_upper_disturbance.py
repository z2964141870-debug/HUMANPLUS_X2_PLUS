#!/usr/bin/env python3
"""Summarize the fixed/slow/fast upper-body official-MuJoCo gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "move_forward_displacement_m",
    "move_lateral_displacement_m",
    "root_tilt_max_rad",
    "stop_settle_time_s",
    "upper_tracking_rmse_rad",
    "upper_target_excursion_abs_max_rad",
    "upper_target_speed_abs_max_radps",
)


def load_group(root: Path, pattern: str, repeats: int) -> list[dict]:
    paths = [root / pattern.format(repeat=repeat) for repeat in range(1, repeats + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing gate outputs: {missing}")
    return [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths]


def summarize(runs: list[dict]) -> dict:
    result = {
        "runs": len(runs),
        "full_gate_passes": sum(bool(run.get("full_gate_pass")) for run in runs),
        "fallback_steps_total": sum(int(run.get("upper_fallback_steps", 0)) for run in runs),
    }
    for metric in METRICS:
        values = [float(run[metric]) for run in runs if run.get(metric) is not None]
        if values:
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_worst_abs"] = float(max(values, key=abs))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    groups = {
        "fixed": load_group(args.result_root, "stage244_rsl_actor_clip_medium_straight_r{repeat}.json", args.repeats),
        "slow": load_group(args.result_root, "stage251_upper_slow_straight_r{repeat}.json", args.repeats),
        "fast": load_group(args.result_root, "stage251_upper_fast_straight_r{repeat}.json", args.repeats),
    }
    summaries = {name: summarize(runs) for name, runs in groups.items()}
    all_pass = all(item["full_gate_passes"] == args.repeats for item in summaries.values())
    no_fallback = all(item["fallback_steps_total"] == 0 for item in summaries.values())
    result = {
        "stage": "stage251_upper_straight_panel",
        "domain": "AimDK X2 v1.0 official MuJoCo",
        "profiles": {
            "fixed": "Stage244 exact fixed-arm baseline",
            "slow": "real X2-retargeted swing-arms, scale=0.25, time_scale=0.5, slew<=0.20 rad/s",
            "fast": "same motion, scale=0.25, time_scale=1.0, slew<=0.40 rad/s",
        },
        "summary": summaries,
        "all_profiles_full_gate_pass": all_pass,
        "no_upper_health_fallback": no_fallback,
        "panel_pass": bool(all_pass and no_fallback),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage251：官方 X2 MuJoCo 上肢扰动直行门禁",
        "",
        "同一冻结后端、同一行走/停车配置；只改变 14DOF 上肢目标。上肢轨迹来自已有 X2 重定向真实动作，不是临时正弦动作。",
        "",
        "| profile | full gate | fallback steps | forward median (m) | lateral median (m) | tilt max median (rad) | stop settle median (s) | upper RMSE (rad) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("fixed", "slow", "fast"):
        row = summaries[name]
        lines.append(
            f"| {name} | {row['full_gate_passes']}/{row['runs']} | {row['fallback_steps_total']} | "
            f"{row.get('move_forward_displacement_m_median', float('nan')):.3f} | "
            f"{row.get('move_lateral_displacement_m_median', float('nan')):.3f} | "
            f"{row.get('root_tilt_max_rad_median', float('nan')):.3f} | "
            f"{row.get('stop_settle_time_s_median', float('nan')):.2f} | "
            f"{row.get('upper_tracking_rmse_rad_median', float('nan')):.3f} |"
        )
    lines += [
        "",
        f"结论：{'通过' if result['panel_pass'] else '未通过'}。",
        "",
        "此门禁只证明直行条件下的扰动鲁棒性；转向与执行器参数扰动仍需单独验证。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["panel_pass"] else 2)


if __name__ == "__main__":
    main()
