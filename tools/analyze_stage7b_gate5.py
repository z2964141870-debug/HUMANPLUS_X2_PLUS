#!/usr/bin/env python3
"""Decide Stage7B Gate5 and flag an unplanned conservative Pareto candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--neutral", type=Path, required=True)
parser.add_argument("--risk", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--output-md", type=Path, required=True)
args = parser.parse_args()


def load(path: Path) -> dict:
    panel = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if len(panel.get("cases", ())) != 12:
        raise ValueError(f"{path} must contain 12 cases")
    return panel


def index(panel: dict) -> dict[str, dict]:
    return {case["case_id"]: case for case in panel["cases"]}


def summary(panel: dict) -> dict:
    cases = panel["cases"]
    return {
        "survived": sum(bool(case["survived_full_horizon"]) for case in cases),
        "steps": sum(int(case["steps_completed"]) for case in cases),
        "low_speed_steps": sum(
            int(case["steps_completed"])
            for case in cases
            if float(case["speed_mps"]) <= 0.200001
        ),
        "heading_mean_rad": fmean(
            float(case["heading_deviation_abs_max_rad"]) for case in cases
        ),
        "lateral_mean_m": fmean(
            float(case["lateral_drift_max_m"]) for case in cases
        ),
        "tilt_mean_rad": fmean(
            float(case["base_tilt_max_rad"]) for case in cases
        ),
        "residual_abs_max": max(
            float(case["coordination_residual_abs_max"]) for case in cases
        ),
    }


panels = {
    "BASE": load(args.base),
    "NEUTRAL05": load(args.neutral),
    "RISK05": load(args.risk),
}
indexed = {name: index(panel) for name, panel in panels.items()}
case_ids = set(indexed["BASE"])
if any(set(cases) != case_ids for cases in indexed.values()):
    raise ValueError("Stage7B panels use different cases")
summaries = {name: summary(panel) for name, panel in panels.items()}
base = indexed["BASE"]
neutral = indexed["NEUTRAL05"]
risk = indexed["RISK05"]


def base_normal_survivors_preserved(candidate: dict[str, dict]) -> bool:
    return all(
        bool(candidate[case_id]["survived_full_horizon"])
        for case_id, case in base.items()
        if float(case["speed_mps"]) >= 0.299999
        and bool(case["survived_full_horizon"])
    )


def normal_heldout_regressions(candidate: dict[str, dict]) -> dict:
    result = {}
    for case_id, case in base.items():
        if (
            case["motion_tag"]
            not in {"wave_real", "wave_left_heldout", "female_lift_heldout"}
            or float(case["speed_mps"]) < 0.299999
        ):
            continue
        current = candidate[case_id]
        failures = []
        if (
            float(current["heading_deviation_abs_max_rad"])
            > float(case["heading_deviation_abs_max_rad"]) + 0.15
        ):
            failures.append("heading_worse_than_base_plus_0p15")
        if (
            float(current["lateral_drift_max_m"])
            > float(case["lateral_drift_max_m"]) + 0.15
        ):
            failures.append("lateral_worse_than_base_plus_0p15")
        if bool(current["survived_full_horizon"]) and float(
            current["base_tilt_max_rad"]
        ) > 0.45:
            failures.append("surviving_case_tilt_gt_0p45")
        if failures:
            result[case_id] = failures
    return result


risk_gate_checks = {
    "survival_not_below_neutral": (
        summaries["RISK05"]["survived"] >= summaries["NEUTRAL05"]["survived"]
    ),
    "steps_not_below_neutral": (
        summaries["RISK05"]["steps"] >= summaries["NEUTRAL05"]["steps"]
    ),
    "heading_not_above_neutral": (
        summaries["RISK05"]["heading_mean_rad"]
        <= summaries["NEUTRAL05"]["heading_mean_rad"]
    ),
    "lateral_not_above_neutral": (
        summaries["RISK05"]["lateral_mean_m"]
        <= summaries["NEUTRAL05"]["lateral_mean_m"]
    ),
    "base_normal_survivors_preserved": base_normal_survivors_preserved(risk),
    "wave_left_heading_and_tilt_safe": (
        float(risk["wave_left_heldout__vx0.30"][
            "heading_deviation_abs_max_rad"
        ])
        <= float(base["wave_left_heldout__vx0.30"][
            "heading_deviation_abs_max_rad"
        ])
        + 0.15
        and float(risk["wave_left_heldout__vx0.30"]["base_tilt_max_rad"])
        <= 0.45
    ),
    "heldout_heading_or_survival_improved_vs_neutral": any(
        (
            float(risk[case_id]["heading_deviation_abs_max_rad"])
            < float(neutral[case_id]["heading_deviation_abs_max_rad"])
            - 0.01
        )
        or (
            int(risk[case_id]["steps_completed"])
            > int(neutral[case_id]["steps_completed"]) + 2
        )
        for case_id in (
            "wave_real__vx0.30",
            "wave_left_heldout__vx0.30",
            "female_lift_heldout__vx0.30",
        )
    ),
}
risk_gate5_passed = all(risk_gate_checks.values())

neutral_regressions = normal_heldout_regressions(neutral)
neutral_candidate_checks = {
    "survival_not_below_base": (
        summaries["NEUTRAL05"]["survived"] >= summaries["BASE"]["survived"]
    ),
    "steps_not_below_base": (
        summaries["NEUTRAL05"]["steps"] >= summaries["BASE"]["steps"]
    ),
    "low_speed_steps_not_below_base": (
        summaries["NEUTRAL05"]["low_speed_steps"]
        >= summaries["BASE"]["low_speed_steps"]
    ),
    "heading_better_than_base": (
        summaries["NEUTRAL05"]["heading_mean_rad"]
        < summaries["BASE"]["heading_mean_rad"]
    ),
    "lateral_better_than_base": (
        summaries["NEUTRAL05"]["lateral_mean_m"]
        < summaries["BASE"]["lateral_mean_m"]
    ),
    "base_normal_survivors_preserved": base_normal_survivors_preserved(neutral),
    "no_normal_heldout_regression": not neutral_regressions,
    "residual_within_0p05": summaries["NEUTRAL05"]["residual_abs_max"] <= 0.050001,
}
neutral_pareto_candidate = all(neutral_candidate_checks.values())
decision = (
    "UNLOCK_RISK_GATE25"
    if risk_gate5_passed
    else (
        "STOP_RISK_KEEP_NEUTRAL_FOR_ROBUSTNESS"
        if neutral_pareto_candidate
        else "STOP_STAGE7B"
    )
)

result = {
    "schema_version": 1,
    "decision": decision,
    "risk_gate5_passed": risk_gate5_passed,
    "risk_gate_checks": risk_gate_checks,
    "neutral_pareto_candidate": neutral_pareto_candidate,
    "neutral_candidate_checks": neutral_candidate_checks,
    "neutral_normal_heldout_regressions": neutral_regressions,
    "summaries": summaries,
}
args.output_json.parent.mkdir(parents=True, exist_ok=True)
args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

rows = []
for name in ("BASE", "NEUTRAL05", "RISK05"):
    item = summaries[name]
    rows.append(
        "| {name} | {survived}/12 | {steps} | {low} | {heading:.4f} | "
        "{lateral:.4f} | {tilt:.4f} |".format(
            name=name,
            survived=item["survived"],
            steps=item["steps"],
            low=item["low_speed_steps"],
            heading=item["heading_mean_rad"],
            lateral=item["lateral_mean_m"],
            tilt=item["tilt_mean_rad"],
        )
    )
risk_rows = [
    f"- {'✓' if passed else '✗'} `{name}`"
    for name, passed in risk_gate_checks.items()
]
neutral_rows = [
    f"- {'✓' if passed else '✗'} `{name}`"
    for name, passed in neutral_candidate_checks.items()
]
args.output_md.write_text(
    "\n".join(
        [
            "# Stage7B Gate5 裁决",
            "",
            f"结论：**{decision}**。",
            "",
            "| 分支 | 生存 | 总步数 | 低速步数 | heading | lateral | tilt |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## RISK05 预注册门",
            "",
            *risk_rows,
            "",
            "## NEUTRAL05 意外 Pareto 候选审计",
            "",
            *neutral_rows,
            "",
            f"NEUTRAL05 robustness 复测候选：`{neutral_pareto_candidate}`。",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps({"decision": decision}, indent=2))
