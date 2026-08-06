#!/usr/bin/env python3
"""Apply the pre-registered Stage7 selection/held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument(
    "--candidate",
    action="append",
    required=True,
    help="BLEND=PATH; repeat for every pre-registered blend",
)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--output-md", type=Path, required=True)
args = parser.parse_args()


def load(path: Path) -> dict:
    panel = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if len(panel.get("cases", ())) != 12:
        raise ValueError(f"{path} must contain 12 cases")
    return panel


def index(panel: dict) -> dict[str, dict]:
    result = {case["case_id"]: case for case in panel["cases"]}
    if len(result) != len(panel["cases"]):
        raise ValueError("duplicate case ids")
    return result


base_panel = load(args.base)
base = index(base_panel)
candidates: dict[float, dict] = {}
candidate_paths = {}
for spec in args.candidate:
    blend_text, separator, path_text = spec.partition("=")
    if not separator:
        raise ValueError(f"invalid --candidate {spec!r}; expected BLEND=PATH")
    blend = float(blend_text)
    if blend in candidates:
        raise ValueError(f"duplicate blend {blend}")
    path = Path(path_text)
    candidates[blend] = load(path)
    candidate_paths[blend] = str(path.expanduser().resolve())
if sorted(candidates) != [0.25, 0.5, 0.75]:
    raise ValueError("Stage7 requires exactly blends 0.25, 0.50, 0.75")
if any(set(index(panel)) != set(base) for panel in candidates.values()):
    raise ValueError("candidate case ids differ from BASE")

selection_tags = {"swing_arms", "knocking", "box_lift"}


def selection_cases(panel: dict) -> list[dict]:
    return [
        case
        for case in panel["cases"]
        if case["motion_tag"] in selection_tags
        and float(case["speed_mps"]) >= 0.299999
    ]


base_selection = selection_cases(base_panel)
base_heading = fmean(
    float(case["heading_deviation_abs_max_rad"]) for case in base_selection
)
base_lateral = fmean(
    float(case["lateral_drift_max_m"]) for case in base_selection
)

selection_results = {}
eligible = []
for blend, panel in sorted(candidates.items()):
    cases = selection_cases(panel)
    heading = fmean(
        float(case["heading_deviation_abs_max_rad"]) for case in cases
    )
    lateral = fmean(float(case["lateral_drift_max_m"]) for case in cases)
    checks = {
        "survived_3_of_3": sum(
            bool(case["survived_full_horizon"]) for case in cases
        )
        == 3,
        "tilt_each_le_0p45": all(
            float(case["base_tilt_max_rad"]) <= 0.45 for case in cases
        ),
        "residual_within_scaled_budget": all(
            float(case["coordination_residual_abs_max"])
            <= 0.100001 * blend
            for case in cases
        ),
        "heading_improved_at_least_10pct": heading <= 0.90 * base_heading,
        "lateral_improved_at_least_10pct": lateral <= 0.90 * base_lateral,
    }
    score = heading / base_heading + lateral / base_lateral
    result = {
        "blend": blend,
        "selection_heading_mean_rad": heading,
        "selection_lateral_mean_m": lateral,
        "heading_improvement_fraction": 1.0 - heading / base_heading,
        "lateral_improvement_fraction": 1.0 - lateral / base_lateral,
        "normalized_score_lower_is_better": score,
        "checks": checks,
        "passed": all(checks.values()),
    }
    selection_results[str(blend)] = result
    if result["passed"]:
        eligible.append((score, blend))

if not eligible:
    locked_blend = None
    decision = "STOP_NO_SELECTION_BLEND"
    heldout_checks = {}
    regressions = {}
    phase_b_unlocked = False
else:
    _, locked_blend = min(eligible, key=lambda item: (item[0], item[1]))
    locked_panel = candidates[locked_blend]
    locked = index(locked_panel)
    normal_heldout_ids = [
        case_id
        for case_id, case in base.items()
        if case["motion_tag"]
        in {"wave_real", "wave_left_heldout", "female_lift_heldout"}
        and float(case["speed_mps"]) >= 0.299999
    ]
    regressions = {}
    for case_id in normal_heldout_ids:
        baseline = base[case_id]
        candidate = locked[case_id]
        failures = []
        if (
            float(candidate["heading_deviation_abs_max_rad"])
            > float(baseline["heading_deviation_abs_max_rad"]) + 0.15
        ):
            failures.append("heading_worse_than_base_plus_0p15")
        if (
            float(candidate["lateral_drift_max_m"])
            > float(baseline["lateral_drift_max_m"]) + 0.15
        ):
            failures.append("lateral_worse_than_base_plus_0p15")
        if bool(candidate["survived_full_horizon"]) and float(
            candidate["base_tilt_max_rad"]
        ) > 0.45:
            failures.append("surviving_case_tilt_gt_0p45")
        if failures:
            regressions[case_id] = failures

    base_survived = sum(
        bool(case["survived_full_horizon"]) for case in base_panel["cases"]
    )
    locked_survived = sum(
        bool(case["survived_full_horizon"]) for case in locked_panel["cases"]
    )
    base_low_steps = sum(
        int(case["steps_completed"])
        for case in base_panel["cases"]
        if float(case["speed_mps"]) <= 0.200001
    )
    locked_low_steps = sum(
        int(case["steps_completed"])
        for case in locked_panel["cases"]
        if float(case["speed_mps"]) <= 0.200001
    )
    base_mean_heading = fmean(
        float(case["heading_deviation_abs_max_rad"])
        for case in base_panel["cases"]
    )
    locked_mean_heading = fmean(
        float(case["heading_deviation_abs_max_rad"])
        for case in locked_panel["cases"]
    )
    base_mean_lateral = fmean(
        float(case["lateral_drift_max_m"]) for case in base_panel["cases"]
    )
    locked_mean_lateral = fmean(
        float(case["lateral_drift_max_m"]) for case in locked_panel["cases"]
    )
    female_id = "female_lift_heldout__vx0.30"
    heldout_checks = {
        "survival_count_not_below_base": locked_survived >= base_survived,
        "all_base_normal_speed_survivors_preserved": all(
            bool(locked[case_id]["survived_full_horizon"])
            for case_id, case in base.items()
            if float(case["speed_mps"]) >= 0.299999
            and bool(case["survived_full_horizon"])
        ),
        "no_normal_speed_heldout_regression": not regressions,
        "female_lift_steps_not_below_base": (
            int(locked[female_id]["steps_completed"])
            >= int(base[female_id]["steps_completed"])
        ),
        "low_speed_steps_not_below_base": locked_low_steps >= base_low_steps,
        "full_panel_mean_heading_better_than_base": (
            locked_mean_heading < base_mean_heading
        ),
        "full_panel_mean_lateral_better_than_base": (
            locked_mean_lateral < base_mean_lateral
        ),
    }
    pareto = all(heldout_checks.values())
    only_low_speed_failed = (
        not heldout_checks["low_speed_steps_not_below_base"]
        and all(
            passed
            for name, passed in heldout_checks.items()
            if name != "low_speed_steps_not_below_base"
        )
    )
    if pareto:
        decision = "PROMOTE_STATIC_BLEND"
    elif only_low_speed_failed:
        decision = "PROMOTE_SPEED_GATED_WORKING_ENVELOPE"
    else:
        decision = "STATIC_BLEND_FAILED_HELDOUT"

    selected = selection_results[str(locked_blend)]
    milder_relieves_full_regression = any(
        float(index(panel)["wave_left_heldout__vx0.30"][
            "heading_deviation_abs_max_rad"
        ])
        < float(index(candidates[0.75])["wave_left_heldout__vx0.30"][
            "heading_deviation_abs_max_rad"
        ])
        for blend, panel in candidates.items()
        if blend < 0.75
    )
    phase_b_unlocked = (
        selected["heading_improvement_fraction"] >= 0.15
        and selected["lateral_improvement_fraction"] >= 0.15
        and bool(regressions)
        and all(
            "heading" in failure or "tilt" in failure
            for failures in regressions.values()
            for failure in failures
        )
        and milder_relieves_full_regression
    )

result = {
    "schema_version": 1,
    "decision": decision,
    "base": str(args.base.expanduser().resolve()),
    "candidate_paths": {str(k): v for k, v in candidate_paths.items()},
    "selection_tags": sorted(selection_tags),
    "base_selection": {
        "heading_mean_rad": base_heading,
        "lateral_mean_m": base_lateral,
    },
    "selection_results": selection_results,
    "locked_blend": locked_blend,
    "heldout_checks": heldout_checks,
    "heldout_regressions": regressions,
    "phase_b_risk_training_unlocked": phase_b_unlocked,
}
args.output_json.parent.mkdir(parents=True, exist_ok=True)
args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

rows = []
for blend in sorted(candidates):
    entry = selection_results[str(blend)]
    rows.append(
        "| {blend:.2f} | {passed} | {heading:.4f} | {lateral:.4f} | "
        "{hg:.1%} | {lg:.1%} | {score:.4f} |".format(
            blend=blend,
            passed="✓" if entry["passed"] else "✗",
            heading=entry["selection_heading_mean_rad"],
            lateral=entry["selection_lateral_mean_m"],
            hg=entry["heading_improvement_fraction"],
            lg=entry["lateral_improvement_fraction"],
            score=entry["normalized_score_lower_is_better"],
        )
    )
heldout_rows = [
    f"- {'✓' if passed else '✗'} `{name}`"
    for name, passed in heldout_checks.items()
]
regression_rows = (
    [
        f"- `{case_id}`：{', '.join(failures)}"
        for case_id, failures in regressions.items()
    ]
    or ["- 无。"]
)
args.output_md.write_text(
    "\n".join(
        [
            "# Stage7 residual blend 预注册裁决",
            "",
            f"结论：**{decision}**。",
            "",
            f"仅按 selection 锁定的 alpha：`{locked_blend}`。"
            if locked_blend is not None
            else "没有 alpha 通过 selection 门。",
            "",
            "| alpha | selection通过 | heading | lateral | heading改善 | "
            "lateral改善 | 归一化分数↓ |",
            "|---:|:---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 留出门",
            "",
            *heldout_rows,
            "",
            "## 留出回归",
            "",
            *regression_rows,
            "",
            f"Phase B 风险约束训练解锁：`{phase_b_unlocked}`。",
            "",
        ]
    ),
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "decision": decision,
            "locked_blend": locked_blend,
            "phase_b_risk_training_unlocked": phase_b_unlocked,
        },
        indent=2,
    )
)
