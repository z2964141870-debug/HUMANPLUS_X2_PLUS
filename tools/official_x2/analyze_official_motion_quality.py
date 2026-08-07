#!/usr/bin/env python3
"""Compare functional and motion-quality evidence from official X2 rollouts."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


DT = 0.02


def run_metrics(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    rows = [row for row in payload["trace"] if row["stage"] == "move"]
    actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
    tilt = np.asarray([row["root_tilt_rad"] for row in rows], dtype=np.float64)
    lateral_velocity = np.asarray(
        [row["root_vy_b_mps"] for row in rows], dtype=np.float64
    )
    root_z = np.asarray([row["root_z_m"] for row in rows], dtype=np.float64)
    action_delta = np.diff(actions, axis=0)
    delta_norm = np.linalg.norm(action_delta, axis=1)
    return {
        "file": str(path),
        "full_gate_pass": bool(summary.get("full_gate_pass")),
        "forward_m": summary.get("move_forward_displacement_m"),
        "lateral_abs_m": abs(float(summary.get("move_lateral_displacement_m", 0.0))),
        "turn_ratio": summary.get("turn_yaw_progress_ratio"),
        "stop_settle_s": summary.get("stop_settle_time_s"),
        "tilt_p95_rad": float(np.quantile(tilt, 0.95)),
        "tilt_max_rad": float(tilt.max()),
        "root_z_peak_to_peak_m": float(np.ptp(root_z)),
        "lateral_velocity_rms_mps": float(np.sqrt(np.mean(lateral_velocity**2))),
        "lateral_velocity_p95_mps": float(np.quantile(np.abs(lateral_velocity), 0.95)),
        "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.999)),
        "waist_saturation_fraction": float(np.mean(np.abs(actions[:, 12:15]) >= 0.999)),
        "action_delta_element_rms": float(np.sqrt(np.mean(action_delta**2))),
        "action_delta_l2_p50": float(np.quantile(delta_norm, 0.50)),
        "action_delta_l2_p75": float(np.quantile(delta_norm, 0.75)),
        "action_delta_l2_p90": float(np.quantile(delta_norm, 0.90)),
        "action_delta_l2_p95": float(np.quantile(delta_norm, 0.95)),
        "action_delta_l2_max": float(delta_norm.max()),
        "action_ema_alpha": summary.get("action_ema_alpha", 1.0),
        "waist_tilt_action_multiplier": summary.get("waist_tilt_action_multiplier", 1.0),
        "pd_profile": summary.get("pd_profile"),
    }


def aggregate(paths: list[Path]) -> dict[str, object]:
    runs = [run_metrics(path) for path in paths]
    numeric = [
        "forward_m",
        "lateral_abs_m",
        "stop_settle_s",
        "tilt_p95_rad",
        "tilt_max_rad",
        "root_z_peak_to_peak_m",
        "lateral_velocity_rms_mps",
        "lateral_velocity_p95_mps",
        "action_saturation_fraction",
        "waist_saturation_fraction",
        "action_delta_element_rms",
        "action_delta_l2_p50",
        "action_delta_l2_p75",
        "action_delta_l2_p90",
        "action_delta_l2_p95",
        "action_delta_l2_max",
    ]
    result: dict[str, object] = {
        "runs": runs,
        "run_count": len(runs),
        "functional_passes": sum(bool(run["full_gate_pass"]) for run in runs),
    }
    for key in numeric:
        values = np.asarray(
            [float(run[key]) for run in runs if run[key] is not None], dtype=np.float64
        )
        result[f"{key}_mean"] = float(values.mean()) if values.size else None
        result[f"{key}_worst"] = float(values.max()) if values.size else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="Repeatable group name and rollout glob.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[str, object] = {}
    for spec in args.group:
        name, separator, pattern = spec.partition("=")
        if not separator or not name or not pattern:
            parser.error(f"invalid --group {spec!r}; expected NAME=GLOB")
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
        if not paths:
            parser.error(f"group {name!r} matched no files: {pattern}")
        groups[name] = aggregate(paths)

    result = {
        "domain": "aimdk_x2_v1_official_mujoco",
        "control_dt_s": DT,
        "interpretation": {
            "functional_gate": "survive + startup + skill semantics + stop",
            "quality_metrics": "diagnostic only; no post-hoc pass threshold is claimed",
            "action_saturation_reference": {
                "isaaclab_stage219_nominal": 0.14760,
                "source": "/home/humanplus/x2_teleop_final/x2_sonic/docs/reports/x2_stage219_s2600_clean_nominal.json",
            },
        },
        "groups": groups,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 官方 X2 MuJoCo 运动质量审计",
        "",
        "功能门只回答是否完成起步、技能和停车；下表额外量化肉眼可见的歪扭与突变。质量阈值尚未事后锁定，因此不把本表包装成新的通过率。",
        "",
        "| 组别 | 功能通过 | 横漂均值(m) | 倾角 p95/max(rad) | 动作饱和 | 腰部饱和 | Δaction RMS / L2-p95 | 停稳均值(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in groups.items():
        lines.append(
            f"| {name} | {item['functional_passes']}/{item['run_count']} | "
            f"{item['lateral_abs_m_mean']:.3f} | "
            f"{item['tilt_p95_rad_mean']:.3f}/{item['tilt_max_rad_worst']:.3f} | "
            f"{item['action_saturation_fraction_mean']:.3f} | "
            f"{item['waist_saturation_fraction_mean']:.3f} | "
            f"{item['action_delta_element_rms_mean']:.3f}/{item['action_delta_l2_p95_mean']:.3f} | "
            f"{item['stop_settle_s_mean']:.2f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- IsaacLab Stage219 nominal 的动作饱和率为 `0.14760`；官方域基线明显更高。",
        "- 腰 pitch/roll 缩放属于部署侧结构化诊断，不是新训练模型。",
        "- 降低腰部饱和不自动等价于降低腿踝突变，必须同时看 `Δaction`。",
        "- 官方 simulator 未发布可用足底接触/力 topic，本报告不能诚实补写 foot slip 或 COP。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"passes": item["functional_passes"], "runs": item["run_count"]} for name, item in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
