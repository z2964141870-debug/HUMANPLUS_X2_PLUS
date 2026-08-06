#!/usr/bin/env python3
"""Apply the pre-registered Stage6 Gate5 decision to four physical panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--current", type=Path, required=True)
parser.add_argument("--future", type=Path, required=True)
parser.add_argument("--future-nophase", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--output-md", type=Path, required=True)
args = parser.parse_args()


def load(path: Path) -> dict:
    result = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if len(result.get("cases", ())) != 12:
        raise ValueError(f"{path} does not contain the pre-registered 12 cases")
    return result


def indexed(panel: dict) -> dict[str, dict]:
    result = {case["case_id"]: case for case in panel["cases"]}
    if len(result) != len(panel["cases"]):
        raise ValueError("duplicate case ids in Stage6 panel")
    return result


def mean(panel: dict, field: str) -> float:
    return fmean(float(case[field]) for case in panel["cases"])


def count_survived(panel: dict) -> int:
    return sum(bool(case["survived_full_horizon"]) for case in panel["cases"])


panels = {
    "BASE": load(args.base),
    "CURRENT": load(args.current),
    "FUTURE": load(args.future),
    "FUTURE-NOPHASE": load(args.future_nophase),
}
case_ids = set(indexed(panels["BASE"]))
if any(set(indexed(panel)) != case_ids for panel in panels.values()):
    raise ValueError("Stage6 panels do not contain the same case ids")

base_by_case = indexed(panels["BASE"])
future_by_case = indexed(panels["FUTURE"])
summaries = {}
for name, panel in panels.items():
    summaries[name] = {
        "survived_count": count_survived(panel),
        "steps_completed_total": sum(
            int(case["steps_completed"]) for case in panel["cases"]
        ),
        "heading_abs_max_mean_rad": mean(
            panel, "heading_deviation_abs_max_rad"
        ),
        "lateral_drift_max_mean_m": mean(panel, "lateral_drift_max_m"),
        "tilt_max_mean_rad": mean(panel, "base_tilt_max_rad"),
        "residual_abs_max": max(
            float(case["coordination_residual_abs_max"])
            for case in panel["cases"]
        ),
    }

# A low-speed improvement is counted only when the candidate covers effectively
# the same physical window (within two 20-ms control frames) and improves both
# direction metrics by at least 0.01.  This prevents a shorter, earlier-failing
# trace from looking artificially better merely because maxima had less time to
# accumulate.
low_speed_improvements = []
for case_id in sorted(case_ids):
    base = base_by_case[case_id]
    future = future_by_case[case_id]
    if float(base["speed_mps"]) > 0.200001:
        continue
    heading_gain = float(base["heading_deviation_abs_max_rad"]) - float(
        future["heading_deviation_abs_max_rad"]
    )
    lateral_gain = float(base["lateral_drift_max_m"]) - float(
        future["lateral_drift_max_m"]
    )
    comparable_window = int(future["steps_completed"]) >= (
        int(base["steps_completed"]) - 2
    )
    if comparable_window and heading_gain >= 0.01 and lateral_gain >= 0.01:
        low_speed_improvements.append(
            {
                "case_id": case_id,
                "base_steps": int(base["steps_completed"]),
                "future_steps": int(future["steps_completed"]),
                "heading_gain_rad": heading_gain,
                "lateral_gain_m": lateral_gain,
            }
        )

normal_tags = ("swing_arms", "knocking", "box_lift")
normal_speed_checks = {}
for tag in normal_tags:
    case_id = f"{tag}__vx0.30"
    case = future_by_case[case_id]
    checks = {
        "survived": bool(case["survived_full_horizon"]),
        "tilt_le_0p45": float(case["base_tilt_max_rad"]) <= 0.45,
        "both_feet_lift_ge_0p02": (
            float(case["left_foot_lift_max_m"]) >= 0.02
            and float(case["right_foot_lift_max_m"]) >= 0.02
        ),
        "upper_active_ge_0p01": (
            float(case["upper_target_excursion_abs_max_rad"]) >= 0.01
        ),
        "upper_tracking_p95_le_0p35": (
            float(case["upper_tracking_abs_p95_rad"]) <= 0.35
        ),
        "heading_not_worse_than_base_plus_0p15": (
            float(case["heading_deviation_abs_max_rad"])
            <= float(base_by_case[case_id]["heading_deviation_abs_max_rad"])
            + 0.15
        ),
        "lateral_not_worse_than_base_plus_0p15": (
            float(case["lateral_drift_max_m"])
            <= float(base_by_case[case_id]["lateral_drift_max_m"]) + 0.15
        ),
    }
    normal_speed_checks[case_id] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

gate_checks = {
    "future_survival_not_below_base": (
        summaries["FUTURE"]["survived_count"]
        >= summaries["BASE"]["survived_count"]
    ),
    "failed_or_low_speed_case_improved": bool(low_speed_improvements),
    "future_heading_better_than_current": (
        summaries["FUTURE"]["heading_abs_max_mean_rad"]
        < summaries["CURRENT"]["heading_abs_max_mean_rad"]
    ),
    "future_heading_better_than_future_nophase": (
        summaries["FUTURE"]["heading_abs_max_mean_rad"]
        < summaries["FUTURE-NOPHASE"]["heading_abs_max_mean_rad"]
    ),
    "future_lateral_better_than_current": (
        summaries["FUTURE"]["lateral_drift_max_mean_m"]
        < summaries["CURRENT"]["lateral_drift_max_mean_m"]
    ),
    "future_lateral_better_than_future_nophase": (
        summaries["FUTURE"]["lateral_drift_max_mean_m"]
        < summaries["FUTURE-NOPHASE"]["lateral_drift_max_mean_m"]
    ),
    "normal_speed_regression_panel_passed": all(
        result["passed"] for result in normal_speed_checks.values()
    ),
    "future_residual_within_0p10": (
        summaries["FUTURE"]["residual_abs_max"] <= 0.100001
    ),
}
gate_passed = all(gate_checks.values())

result = {
    "schema_version": 1,
    "decision": "UNLOCK_GATE25" if gate_passed else "STOP_AT_GATE5",
    "gate5_passed": gate_passed,
    "interpretation": (
        "Gate5 only unlocks the pre-registered 25-update test; it is not "
        "evidence that Stage6 is a successful adapter."
    ),
    "panel_paths": {
        name: str(path.expanduser().resolve())
        for name, path in {
            "BASE": args.base,
            "CURRENT": args.current,
            "FUTURE": args.future,
            "FUTURE-NOPHASE": args.future_nophase,
        }.items()
    },
    "summaries": summaries,
    "low_speed_improvement_definition": {
        "speed_mps_max": 0.20,
        "future_steps_at_least_base_minus": 2,
        "heading_gain_rad_min": 0.01,
        "lateral_gain_m_min": 0.01,
    },
    "low_speed_improvements": low_speed_improvements,
    "normal_speed_checks": normal_speed_checks,
    "gate_checks": gate_checks,
}

args.output_json.parent.mkdir(parents=True, exist_ok=True)
args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

rows = []
for name in ("BASE", "CURRENT", "FUTURE", "FUTURE-NOPHASE"):
    summary = summaries[name]
    rows.append(
        "| {name} | {survived}/12 | {steps} | {heading:.4f} | "
        "{lateral:.4f} | {tilt:.4f} | {residual:.4f} |".format(
            name=name,
            survived=summary["survived_count"],
            steps=summary["steps_completed_total"],
            heading=summary["heading_abs_max_mean_rad"],
            lateral=summary["lateral_drift_max_mean_m"],
            tilt=summary["tilt_max_mean_rad"],
            residual=summary["residual_abs_max"],
        )
    )
check_rows = [
    f"- {'✓' if passed else '✗'} `{name}`"
    for name, passed in gate_checks.items()
]
improvement_rows = (
    [
        "- `{case_id}`：共同窗口 {future_steps}/{base_steps} 步，航向改善 "
        "{heading_gain_rad:.4f} rad，横漂改善 {lateral_gain_m:.4f} m。".format(
            **entry
        )
        for entry in low_speed_improvements
    ]
    or ["- 没有低速条件满足共同窗口与双方向指标的最小改善量。"]
)
args.output_md.write_text(
    "\n".join(
        [
            "# Stage6 Gate5 物理裁决",
            "",
            f"结论：**{result['decision']}**。",
            "",
            "> 该门只决定是否允许做 25-update 验证，不代表 Stage6 已成功。",
            "",
            "| 分支 | 生存 | 总步数 | 平均航向峰值 rad | 平均横漂 m | "
            "平均倾角峰值 rad | residual 最大值 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 预注册门",
            "",
            *check_rows,
            "",
            "## 低速共同窗口改善",
            "",
            *improvement_rows,
            "",
            "低速改善只在候选轨迹不少于 BASE 两个控制帧、且航向和横漂都至少"
            "改善 0.01 时计数，避免更早终止造成的假低峰值。",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps({"decision": result["decision"], "gate_checks": gate_checks}, indent=2))
