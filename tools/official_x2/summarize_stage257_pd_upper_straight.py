#!/usr/bin/env python3
"""Summarize the PD-domain by upper-disturbance official straight gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefix", default="stage257")
    parser.add_argument("--action-bias", type=float, default=0.60)
    parser.add_argument("--recovery-enter", type=float, default=0.08)
    parser.add_argument("--recovery-exit", type=float, default=0.03)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for pd_name in ("soft0p9", "nominal1p0", "stiff1p2"):
        for upper_name in ("fixed", "fast"):
            name = f"{pd_name}_{upper_name}"
            paths = [
                args.result_root / f"{args.prefix}_{name}_straight_r{i}.json"
                for i in range(1, args.repeats + 1)
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing)
            runs = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths]
            def values(metric: str) -> list[float]:
                return [float(run[metric]) for run in runs if run.get(metric) is not None]
            groups[name] = {
                "runs": len(runs),
                "full_gate_passes": sum(bool(run.get("full_gate_pass")) for run in runs),
                "fallback_steps_total": sum(int(run.get("upper_fallback_steps", 0)) for run in runs),
                "forward_median_m": float(np.median(values("move_forward_displacement_m"))),
                "lateral_abs_max_m": float(max(abs(v) for v in values("move_lateral_displacement_m"))),
                "tilt_max_rad": float(max(values("root_tilt_max_rad"))),
                "stop_settle_median_s": float(np.median(values("stop_settle_time_s"))),
                "stop_root_z_min_m": float(min(values("stop_root_z_min_m"))),
                "upper_rmse_median_rad": (
                    float(np.median(values("upper_tracking_rmse_rad")))
                    if values("upper_tracking_rmse_rad") else None
                ),
            }
    panel_pass = all(
        group["full_gate_passes"] == args.repeats and group["fallback_steps_total"] == 0
        for group in groups.values()
    )
    result = {
        "stage": f"{args.prefix}_pd_upper_straight_panel",
        "domain": "AimDK X2 v1.0 official MuJoCo",
        "matrix": "PD {0.9,1.0,1.2} x upper {fixed,fast} x 3 repeats",
        "supervisor": (
            f"lateral recovery bias={args.action_bias:.2f}, "
            f"enter={args.recovery_enter:.2f}m, exit={args.recovery_exit:.2f}m"
        ),
        "summary": groups,
        "panel_pass": panel_pass,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {args.prefix.capitalize()}：官方 X2 MuJoCo 执行器参数 × 上肢扰动直行门禁",
        "",
        "冻结 actor、步态模板和门槛；PD Kp/Kd 同比扰动，并交叉固定/快摆臂。",
        "",
        "| condition | full gate | fallback | forward median | lateral abs max | tilt max | stop z min | stop settle median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in groups.items():
        lines.append(
            f"| {name} | {row['full_gate_passes']}/{row['runs']} | {row['fallback_steps_total']} | "
            f"{row['forward_median_m']:.3f} | {row['lateral_abs_max_m']:.3f} | "
            f"{row['tilt_max_rad']:.3f} | {row['stop_root_z_min_m']:.3f} | {row['stop_settle_median_s']:.2f} |"
        )
    lines += [
        "",
        f"结论：{'通过' if panel_pass else '未通过'}。",
        "",
        "该矩阵验证 PD 参数响应幅值，不等价于显式通信延迟或传感噪声；后二者仍需独立门禁。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if panel_pass else 2)


if __name__ == "__main__":
    main()
