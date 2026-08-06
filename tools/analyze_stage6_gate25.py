#!/usr/bin/env python3
"""Conservatively decide whether Stage6 earns promotion after 25 updates."""

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
    panel = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if len(panel.get("cases", ())) != 12:
        raise ValueError(f"{path} must contain exactly 12 Stage6 cases")
    return panel


def by_case(panel: dict) -> dict[str, dict]:
    result = {case["case_id"]: case for case in panel["cases"]}
    if len(result) != len(panel["cases"]):
        raise ValueError("duplicate Stage6 case ids")
    return result


def summarize(panel: dict) -> dict:
    cases = panel["cases"]
    low_speed = [
        case for case in cases if float(case["speed_mps"]) <= 0.200001
    ]
    return {
        "survived_count": sum(
            bool(case["survived_full_horizon"]) for case in cases
        ),
        "steps_completed_total": sum(
            int(case["steps_completed"]) for case in cases
        ),
        "low_speed_steps_total": sum(
            int(case["steps_completed"]) for case in low_speed
        ),
        "heading_abs_max_mean_rad": fmean(
            float(case["heading_deviation_abs_max_rad"]) for case in cases
        ),
        "lateral_drift_max_mean_m": fmean(
            float(case["lateral_drift_max_m"]) for case in cases
        ),
        "tilt_max_mean_rad": fmean(
            float(case["base_tilt_max_rad"]) for case in cases
        ),
        "residual_abs_max": max(
            float(case["coordination_residual_abs_max"]) for case in cases
        ),
    }


panels = {
    "BASE": load(args.base),
    "CURRENT": load(args.current),
    "FUTURE": load(args.future),
    "FUTURE-NOPHASE": load(args.future_nophase),
}
indexed = {name: by_case(panel) for name, panel in panels.items()}
case_ids = set(indexed["BASE"])
if any(set(cases) != case_ids for cases in indexed.values()):
    raise ValueError("Stage6 Gate25 panels use different cases")
summaries = {name: summarize(panel) for name, panel in panels.items()}

controls = ("BASE", "CURRENT", "FUTURE-NOPHASE")
base = indexed["BASE"]
future = indexed["FUTURE"]

base_survivors_preserved = all(
    bool(future[case_id]["survived_full_horizon"])
    for case_id, case in base.items()
    if bool(case["survived_full_horizon"])
)
normal_regression_cases = {}
for case_id, case in future.items():
    if float(case["speed_mps"]) < 0.299999:
        continue
    baseline = base[case_id]
    failures = []
    if (
        float(case["heading_deviation_abs_max_rad"])
        > float(baseline["heading_deviation_abs_max_rad"]) + 0.15
    ):
        failures.append("heading_worse_than_base_plus_0p15")
    if (
        float(case["lateral_drift_max_m"])
        > float(baseline["lateral_drift_max_m"]) + 0.15
    ):
        failures.append("lateral_worse_than_base_plus_0p15")
    if bool(case["survived_full_horizon"]) and float(
        case["base_tilt_max_rad"]
    ) > 0.45:
        failures.append("surviving_case_tilt_gt_0p45")
    if bool(baseline["survived_full_horizon"]) and not bool(
        case["survived_full_horizon"]
    ):
        failures.append("lost_base_survival")
    if failures:
        normal_regression_cases[case_id] = failures

casewise = {}
for case_id in sorted(case_ids):
    baseline = base[case_id]
    candidate = future[case_id]
    casewise[case_id] = {
        "base_steps": int(baseline["steps_completed"]),
        "future_steps": int(candidate["steps_completed"]),
        "step_delta": (
            int(candidate["steps_completed"])
            - int(baseline["steps_completed"])
        ),
        "heading_delta_rad": (
            float(candidate["heading_deviation_abs_max_rad"])
            - float(baseline["heading_deviation_abs_max_rad"])
        ),
        "lateral_delta_m": (
            float(candidate["lateral_drift_max_m"])
            - float(baseline["lateral_drift_max_m"])
        ),
        "tilt_delta_rad": (
            float(candidate["base_tilt_max_rad"])
            - float(baseline["base_tilt_max_rad"])
        ),
    }

gate_checks = {
    "future_survival_count_not_below_any_control": all(
        summaries["FUTURE"]["survived_count"]
        >= summaries[name]["survived_count"]
        for name in controls
    ),
    "future_total_steps_not_below_any_control": all(
        summaries["FUTURE"]["steps_completed_total"]
        >= summaries[name]["steps_completed_total"]
        for name in controls
    ),
    "future_mean_heading_better_than_all_controls": all(
        summaries["FUTURE"]["heading_abs_max_mean_rad"]
        < summaries[name]["heading_abs_max_mean_rad"]
        for name in controls
    ),
    "future_mean_lateral_better_than_all_controls": all(
        summaries["FUTURE"]["lateral_drift_max_mean_m"]
        < summaries[name]["lateral_drift_max_mean_m"]
        for name in controls
    ),
    "all_base_surviving_cases_preserved": base_survivors_preserved,
    "low_speed_total_steps_not_below_base": (
        summaries["FUTURE"]["low_speed_steps_total"]
        >= summaries["BASE"]["low_speed_steps_total"]
    ),
    "no_normal_speed_casewise_safety_or_direction_regression": (
        not normal_regression_cases
    ),
    "future_residual_within_0p10": (
        summaries["FUTURE"]["residual_abs_max"] <= 0.100001
    ),
}
promoted = all(gate_checks.values())
decision = "PROMOTE_STAGE6" if promoted else "KEEP_STAGE208"

result = {
    "schema_version": 1,
    "decision": decision,
    "stage6_promoted": promoted,
    "result_class": "CONSISTENT_IMPROVEMENT" if promoted else "PARTIAL_SIGNAL",
    "honest_interpretation": (
        "FUTURE has a strong aggregate direction signal, but a general "
        "adapter must also preserve low-speed safety and held-out normal-speed "
        "cases. Failed consistency gates prohibit deployment promotion."
    ),
    "summaries": summaries,
    "gate_checks": gate_checks,
    "normal_speed_regression_cases": normal_regression_cases,
    "casewise_future_minus_base": casewise,
    "panel_paths": {
        name: str(path.expanduser().resolve())
        for name, path in {
            "BASE": args.base,
            "CURRENT": args.current,
            "FUTURE": args.future,
            "FUTURE-NOPHASE": args.future_nophase,
        }.items()
    },
}
args.output_json.parent.mkdir(parents=True, exist_ok=True)
args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

summary_rows = []
for name in ("BASE", "CURRENT", "FUTURE", "FUTURE-NOPHASE"):
    summary = summaries[name]
    summary_rows.append(
        "| {name} | {survived}/12 | {steps} | {low} | {heading:.4f} | "
        "{lateral:.4f} | {tilt:.4f} |".format(
            name=name,
            survived=summary["survived_count"],
            steps=summary["steps_completed_total"],
            low=summary["low_speed_steps_total"],
            heading=summary["heading_abs_max_mean_rad"],
            lateral=summary["lateral_drift_max_mean_m"],
            tilt=summary["tilt_max_mean_rad"],
        )
    )
gate_rows = [
    f"- {'✓' if passed else '✗'} `{name}`"
    for name, passed in gate_checks.items()
]
regression_rows = (
    [
        f"- `{case_id}`：{', '.join(failures)}"
        for case_id, failures in normal_regression_cases.items()
    ]
    or ["- 无。"]
)
case_rows = []
for case_id, delta in casewise.items():
    case_rows.append(
        "| {case_id} | {step_delta:+d} | {heading_delta_rad:+.4f} | "
        "{lateral_delta_m:+.4f} | {tilt_delta_rad:+.4f} |".format(
            case_id=case_id,
            **delta,
        )
    )
args.output_md.write_text(
    "\n".join(
        [
            "# Stage6 Gate25 最终裁决",
            "",
            f"结论：**{decision}**（{result['result_class']}）。",
            "",
            "FUTURE 的总体方向指标确有信号，但未满足跨动作的一致安全改善；"
            "因此 Stage208 继续作为可保留基线，Stage6 仅作为研究候选。",
            "",
            "| 分支 | 生存 | 总步数 | 低速总步数 | 平均航向峰值 rad | "
            "平均横漂 m | 平均倾角峰值 rad |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *summary_rows,
            "",
            "## Gate25 一致性门",
            "",
            *gate_rows,
            "",
            "## 正常速度回归",
            "",
            *regression_rows,
            "",
            "## FUTURE 相对 BASE 的逐案例变化",
            "",
            "| 案例 | 生存步数 Δ | 航向峰值 Δ rad | 横漂 Δ m | 倾角 Δ rad |",
            "|---|---:|---:|---:|---:|",
            *case_rows,
            "",
            "Δ 小于 0 对航向、横漂、倾角更好；生存步数 Δ 大于 0 更好。",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps({"decision": decision, "gate_checks": gate_checks}, indent=2))
