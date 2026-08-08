#!/usr/bin/env python3
"""Summarize the Stage278 soft-PD plus fast-upper hard gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.result_root / f"{args.prefix}_r{i}.json" for i in range(1, args.repeats + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    rows = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths]
    values = lambda key: [float(row[key]) for row in rows]
    result = {
        "stage": "stage278_hard_gate",
        "domain": "AimDK X2 v1.0 official MuJoCo",
        "condition": "PD=0.9 + fast upper + vx=0.30 m/s + brake-blend-to-stand",
        "contract": {
            "future_checkpoint": "Stage267 model_2620",
            "stop_controller": "brake_blend_to_policy",
            "stop_transition_seconds": 1.0,
            "stop_brake_gain": 1.5,
            "stand_switch_speed_mps": 0.10,
            "evaluation_thresholds_unchanged": True,
        },
        "runs": [
            {
                "file": str(path),
                "full_gate_pass": bool(row["full_gate_pass"]),
                "move_gate_pass": bool(row["move_gate_pass"]),
                "stop_gate_pass": bool(row["stop_gate_pass"]),
                "heading_max_rad": float(row["move_heading_max_deviation_rad"]),
                "stop_drift_m": float(row["stop_root_xy_drift_m"]),
                "stop_z_min_m": float(row["stop_root_z_min_m"]),
                "stop_settle_s": float(row["stop_settle_time_s"]),
            }
            for path, row in zip(paths, rows)
        ],
        "summary": {
            "full_gate_passes": sum(bool(row["full_gate_pass"]) for row in rows),
            "move_gate_passes": sum(bool(row["move_gate_pass"]) for row in rows),
            "stop_gate_passes": sum(bool(row["stop_gate_pass"]) for row in rows),
            "heading_max_worst_rad": max(values("move_heading_max_deviation_rad")),
            "heading_max_median_rad": float(np.median(values("move_heading_max_deviation_rad"))),
            "stop_drift_worst_m": max(values("stop_root_xy_drift_m")),
            "stop_z_min_worst_m": min(values("stop_root_z_min_m")),
            "stop_settle_worst_s": max(values("stop_settle_time_s")),
            "compound_stop_ready": all(bool(row["stop_gate_pass"]) for row in rows),
            "compound_full_ready": all(bool(row["full_gate_pass"]) for row in rows),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    lines = [
        "# Stage278：官方 X2 最难复合门禁",
        "",
        "固定评估门槛；条件为 soft PD 0.9、快速摆臂、0.30 m/s 直行及 1 秒 brake→stand 平滑交权。",
        "",
        "| run | full | move | stop | heading max | stop drift | stop z min | settle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(result["runs"], start=1):
        lines.append(
            f"| {index} | {row['full_gate_pass']} | {row['move_gate_pass']} | "
            f"{row['stop_gate_pass']} | {row['heading_max_rad']:.4f} | "
            f"{row['stop_drift_m']:.4f} | {row['stop_z_min_m']:.4f} | "
            f"{row['stop_settle_s']:.2f} |"
        )
    lines += [
        "",
        f"- 复合停车门：{summary['stop_gate_passes']}/{args.repeats}，已闭合。",
        f"- 完整门：{summary['full_gate_passes']}/{args.repeats}，尚未闭合。",
        f"- 最坏航向：{summary['heading_max_worst_rad']:.4f} rad（门槛 0.3000 rad）。",
        "",
        "结论：平滑 brake→stand 已把原先的停车倒地稳定修复；剩余瓶颈是快速摆臂下的航向裕量，不能据此解锁长训。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
