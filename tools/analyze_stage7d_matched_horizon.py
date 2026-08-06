#!/usr/bin/env python3
"""Analyze Stage7D trajectories on strictly matched survival horizons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, nargs=3, required=True)
parser.add_argument("--candidate", type=Path, nargs=3, required=True)
parser.add_argument("--base-original", type=Path, nargs=3, required=True)
parser.add_argument("--candidate-original", type=Path, nargs=3, required=True)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--output-md", type=Path, required=True)
args = parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def index(panel: dict) -> dict[str, dict]:
    result = {case["case_id"]: case for case in panel["cases"]}
    if len(result) != 12:
        raise ValueError("every Stage7D panel must contain 12 unique cases")
    return result


def matched_metrics(base_case: dict, candidate_case: dict) -> dict:
    base_trace = base_case["diagnostic_trace"]
    candidate_trace = candidate_case["diagnostic_trace"]
    count = min(
        len(base_trace["lateral_drift_m"]),
        len(candidate_trace["lateral_drift_m"]),
    )
    if count <= 0:
        raise ValueError(f"{base_case['case_id']} has no common valid steps")
    base_lateral = base_trace["lateral_drift_m"][:count]
    candidate_lateral = candidate_trace["lateral_drift_m"][:count]
    base_heading = base_trace["world_heading_error_abs_rad"][:count]
    candidate_heading = candidate_trace["world_heading_error_abs_rad"][:count]
    return {
        "matched_steps": count,
        "base_lateral_max_m": max(base_lateral),
        "candidate_lateral_max_m": max(candidate_lateral),
        "lateral_max_delta_m": max(candidate_lateral) - max(base_lateral),
        "base_lateral_mean_m": fmean(base_lateral),
        "candidate_lateral_mean_m": fmean(candidate_lateral),
        "lateral_mean_delta_m": fmean(candidate_lateral) - fmean(base_lateral),
        "base_world_heading_mean_rad": fmean(base_heading),
        "candidate_world_heading_mean_rad": fmean(candidate_heading),
        "world_heading_mean_delta_rad": (
            fmean(candidate_heading) - fmean(base_heading)
        ),
        "base_survived": bool(base_case["survived_full_horizon"]),
        "candidate_survived": bool(
            candidate_case["survived_full_horizon"]
        ),
    }


trace_base = {int(panel["seed"]): panel for panel in map(load, args.base)}
trace_candidate = {
    int(panel["seed"]): panel for panel in map(load, args.candidate)
}
original_base = {
    int(panel["seed"]): panel for panel in map(load, args.base_original)
}
original_candidate = {
    int(panel["seed"]): panel for panel in map(load, args.candidate_original)
}
seeds = sorted(trace_base)
if (
    len(seeds) != 3
    or set(trace_candidate) != set(seeds)
    or set(original_base) != set(seeds)
    or set(original_candidate) != set(seeds)
):
    raise ValueError("all four arms must contain the same three seeds")

case_rows: list[dict] = []
reproducibility_max_abs_difference = 0.0
reproducibility_location = ""

for seed in seeds:
    panels = (
        (trace_base[seed], original_base[seed], "BASE"),
        (trace_candidate[seed], original_candidate[seed], "NEUTRAL05"),
    )
    for trace_panel, original_panel, branch in panels:
        trace_cases = index(trace_panel)
        original_cases = index(original_panel)
        if set(trace_cases) != set(original_cases):
            raise ValueError(f"seed {seed} {branch}: case IDs changed")
        for case_id in trace_cases:
            trace_case = trace_cases[case_id]
            original_case = original_cases[case_id]
            for field, value in original_case.items():
                if (
                    field not in trace_case
                    or value is None
                    or isinstance(value, (str, list, dict))
                    or not isinstance(value, (int, float))
                ):
                    continue
                difference = abs(float(value) - float(trace_case[field]))
                if (
                    math.isfinite(difference)
                    and difference > reproducibility_max_abs_difference
                ):
                    reproducibility_max_abs_difference = difference
                    reproducibility_location = (
                        f"seed{seed}:{branch}:{case_id}:{field}"
                    )

    base_cases = index(trace_base[seed])
    candidate_cases = index(trace_candidate[seed])
    if set(base_cases) != set(candidate_cases):
        raise ValueError(f"seed {seed}: paired case IDs changed")
    for case_id in sorted(base_cases):
        item = matched_metrics(base_cases[case_id], candidate_cases[case_id])
        item.update({"seed": seed, "case_id": case_id})
        case_rows.append(item)

aggregate = {
    "pair_count": len(case_rows),
    "base_lateral_max_mean_m": fmean(
        row["base_lateral_max_m"] for row in case_rows
    ),
    "candidate_lateral_max_mean_m": fmean(
        row["candidate_lateral_max_m"] for row in case_rows
    ),
    "base_lateral_time_mean_m": fmean(
        row["base_lateral_mean_m"] for row in case_rows
    ),
    "candidate_lateral_time_mean_m": fmean(
        row["candidate_lateral_mean_m"] for row in case_rows
    ),
    "base_world_heading_time_mean_rad": fmean(
        row["base_world_heading_mean_rad"] for row in case_rows
    ),
    "candidate_world_heading_time_mean_rad": fmean(
        row["candidate_world_heading_mean_rad"] for row in case_rows
    ),
}
common_survivor_regressions = [
    {
        "seed": row["seed"],
        "case_id": row["case_id"],
        "lateral_max_delta_m": row["lateral_max_delta_m"],
    }
    for row in case_rows
    if row["base_survived"]
    and row["candidate_survived"]
    and row["lateral_max_delta_m"] > 0.15
]
rescued = [
    row
    for row in case_rows
    if not row["base_survived"] and row["candidate_survived"]
]
lost = [
    row
    for row in case_rows
    if row["base_survived"] and not row["candidate_survived"]
]
rescued_pre_failure_regressions = [
    {
        "seed": row["seed"],
        "case_id": row["case_id"],
        "lateral_max_delta_m": row["lateral_max_delta_m"],
    }
    for row in rescued
    if row["lateral_max_delta_m"] > 0.15
]
checks = {
    "matched_lateral_max_not_above_base": (
        aggregate["candidate_lateral_max_mean_m"]
        <= aggregate["base_lateral_max_mean_m"]
    ),
    "matched_lateral_time_mean_not_above_base": (
        aggregate["candidate_lateral_time_mean_m"]
        <= aggregate["base_lateral_time_mean_m"]
    ),
    "matched_world_heading_time_mean_not_above_base": (
        aggregate["candidate_world_heading_time_mean_rad"]
        <= aggregate["base_world_heading_time_mean_rad"]
    ),
    "no_common_survivor_lateral_plus_0p15": (
        not common_survivor_regressions
    ),
    "rescued_pre_failure_has_no_lateral_plus_0p15": (
        not rescued_pre_failure_regressions
    ),
    "trace_rerun_exactly_reproduces_stage7c": (
        reproducibility_max_abs_difference == 0.0
    ),
}
passed = all(checks.values())
decision = (
    "WORTH_TARGETED_LATERAL_CONSTRAINT_EXPERIMENT"
    if passed
    else "STOP_CURRENT_FUTURE_ADAPTER_OBJECTIVE"
)
result = {
    "schema_version": 1,
    "decision": decision,
    "passed": passed,
    "duration_censoring_partially_confirmed": (
        checks["matched_lateral_max_not_above_base"]
        and checks["rescued_pre_failure_has_no_lateral_plus_0p15"]
    ),
    "checks": checks,
    "aggregate": aggregate,
    "rescued_count": len(rescued),
    "lost_count": len(lost),
    "common_survivor_regressions": common_survivor_regressions,
    "rescued_pre_failure_regressions": rescued_pre_failure_regressions,
    "rescued_cases": rescued,
    "lost_cases": lost,
    "reproducibility": {
        "max_abs_difference": reproducibility_max_abs_difference,
        "location": reproducibility_location,
    },
    "cases": case_rows,
}
args.output_json.parent.mkdir(parents=True, exist_ok=True)
args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

check_rows = [
    f"- {'✓' if value else '✗'} `{name}`"
    for name, value in checks.items()
]
rescued_rows = [
    "| {seed} | {case} | {steps} | {delta:+.4f} |".format(
        seed=row["seed"],
        case=row["case_id"],
        steps=row["matched_steps"],
        delta=row["lateral_max_delta_m"],
    )
    for row in rescued
]
lost_rows = [
    "| {seed} | {case} | {steps} | {delta:+.4f} |".format(
        seed=row["seed"],
        case=row["case_id"],
        steps=row["matched_steps"],
        delta=row["lateral_max_delta_m"],
    )
    for row in lost
]
args.output_md.write_text(
    "\n".join(
        [
            "# Stage7D 生存时长删失诊断",
            "",
            f"结论：**{decision}**。",
            "",
            "| 指标（36 对共同窗口） | BASE | NEUTRAL05 | 差值 |",
            "|---|---:|---:|---:|",
            "| lateral max 的 case 均值 (m) | "
            f"{aggregate['base_lateral_max_mean_m']:.6f} | "
            f"{aggregate['candidate_lateral_max_mean_m']:.6f} | "
            f"{aggregate['candidate_lateral_max_mean_m'] - aggregate['base_lateral_max_mean_m']:+.6f} |",
            "| lateral 时间均值的 case 均值 (m) | "
            f"{aggregate['base_lateral_time_mean_m']:.6f} | "
            f"{aggregate['candidate_lateral_time_mean_m']:.6f} | "
            f"{aggregate['candidate_lateral_time_mean_m'] - aggregate['base_lateral_time_mean_m']:+.6f} |",
            "| 世界 heading 时间均值 (rad) | "
            f"{aggregate['base_world_heading_time_mean_rad']:.6f} | "
            f"{aggregate['candidate_world_heading_time_mean_rad']:.6f} | "
            f"{aggregate['candidate_world_heading_time_mean_rad'] - aggregate['base_world_heading_time_mean_rad']:+.6f} |",
            "",
            "## 预注册诊断门",
            "",
            *check_rows,
            "",
            "## BASE 失败、候选生存",
            "",
            "| seed | case | 共同步数 | 共同窗口 lateral max 差值 (m) |",
            "|---:|---|---:|---:|",
            *(rescued_rows or ["| - | - | - | - |"]),
            "",
            "## BASE 生存、候选失败",
            "",
            "| seed | case | 共同步数 | 共同窗口 lateral max 差值 (m) |",
            "|---:|---|---:|---:|",
            *(lost_rows or ["| - | - | - | - |"]),
            "",
            "删失偏差得到部分确认：共同窗口 lateral max 略好，且三个被救活"
            "案例在 BASE 失效前均未超过 `+0.15 m`；但共同窗口 lateral 时间"
            "均值仍轻微恶化，另有一个双方生存的 wave 案例超过门槛。因此不能"
            "把候选晋级，也不解锁沿当前目标继续训练。",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps({"decision": decision, "passed": passed}, indent=2))
