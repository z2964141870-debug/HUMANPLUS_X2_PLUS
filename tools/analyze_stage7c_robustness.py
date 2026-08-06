#!/usr/bin/env python3
"""Apply the preregistered Stage7C paired-perturbation robustness gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, nargs=3, required=True)
parser.add_argument("--candidate", type=Path, nargs=3, required=True)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--output-md", type=Path, required=True)
args = parser.parse_args()

INITIAL_FIELDS = (
    "initial_root_quat_wxyz",
    "initial_heading_rad",
    "initial_base_tilt_rad",
    "initial_root_lin_vel_w_xyz_mps",
    "initial_root_ang_vel_w_xyz_radps",
)
HELDOUT_TAGS = {
    "wave_real",
    "wave_left_heldout",
    "female_lift_heldout",
}
PAIR_TOLERANCE = 1.0e-10


def load(path: Path) -> dict:
    panel = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if len(panel.get("cases", ())) != 12:
        raise ValueError(f"{path} must contain exactly 12 cases")
    return panel


def index(panel: dict) -> dict[str, dict]:
    result = {case["case_id"]: case for case in panel["cases"]}
    if len(result) != 12:
        raise ValueError("case identifiers must be unique")
    return result


def flattened(value: object) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


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
        "world_heading_mean_rad": fmean(
            float(case["world_heading_error_abs_max_rad"]) for case in cases
        ),
        "relative_heading_mean_rad": fmean(
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


base_panels = [load(path) for path in args.base]
candidate_panels = [load(path) for path in args.candidate]
base_by_seed = {int(panel["seed"]): panel for panel in base_panels}
candidate_by_seed = {int(panel["seed"]): panel for panel in candidate_panels}
if len(base_by_seed) != 3 or set(base_by_seed) != set(candidate_by_seed):
    raise ValueError("base and candidate must contain the same three unique seeds")

seeds = sorted(base_by_seed)
pairing_details: dict[str, dict] = {}
pairing_ok = True
base_normal_survivors_preserved = True
heldout_regressions: dict[str, list[str]] = {}
per_seed: dict[str, dict] = {}
all_base_cases: list[dict] = []
all_candidate_cases: list[dict] = []

for seed in seeds:
    base_panel = base_by_seed[seed]
    candidate_panel = candidate_by_seed[seed]
    if (
        base_panel.get("initial_perturbation_half_ranges")
        != candidate_panel.get("initial_perturbation_half_ranges")
    ):
        raise ValueError(f"seed {seed}: perturbation contracts differ")
    base_cases = index(base_panel)
    candidate_cases = index(candidate_panel)
    if set(base_cases) != set(candidate_cases):
        raise ValueError(f"seed {seed}: case identifiers differ")

    max_difference = 0.0
    mismatches: list[str] = []
    for case_id in sorted(base_cases):
        base_case = base_cases[case_id]
        candidate_case = candidate_cases[case_id]
        for field in INITIAL_FIELDS:
            left = flattened(base_case[field])
            right = flattened(candidate_case[field])
            if len(left) != len(right):
                mismatches.append(f"{case_id}:{field}:shape")
                continue
            difference = max(abs(a - b) for a, b in zip(left, right))
            max_difference = max(max_difference, difference)
            if difference > PAIR_TOLERANCE:
                mismatches.append(f"{case_id}:{field}:{difference:.3e}")

        if (
            float(base_case["speed_mps"]) >= 0.299999
            and bool(base_case["survived_full_horizon"])
            and not bool(candidate_case["survived_full_horizon"])
        ):
            base_normal_survivors_preserved = False

        if (
            float(base_case["speed_mps"]) >= 0.299999
            and base_case["motion_tag"] in HELDOUT_TAGS
        ):
            failures: list[str] = []
            if (
                float(candidate_case["world_heading_error_abs_max_rad"])
                > float(base_case["world_heading_error_abs_max_rad"]) + 0.15
            ):
                failures.append("world_heading_worse_than_base_plus_0p15")
            if (
                float(candidate_case["lateral_drift_max_m"])
                > float(base_case["lateral_drift_max_m"]) + 0.15
            ):
                failures.append("lateral_worse_than_base_plus_0p15")
            if (
                bool(candidate_case["survived_full_horizon"])
                and float(candidate_case["base_tilt_max_rad"]) > 0.45
            ):
                failures.append("surviving_case_tilt_gt_0p45")
            if failures:
                heldout_regressions[f"seed{seed}:{case_id}"] = failures

    seed_pairing_ok = not mismatches
    pairing_ok = pairing_ok and seed_pairing_ok
    pairing_details[str(seed)] = {
        "paired": seed_pairing_ok,
        "max_abs_difference": max_difference,
        "mismatches": mismatches,
    }
    base_summary = summary(base_panel)
    candidate_summary = summary(candidate_panel)
    seed_non_worse = (
        candidate_summary["steps"] >= base_summary["steps"]
        and candidate_summary["world_heading_mean_rad"]
        <= base_summary["world_heading_mean_rad"]
        and candidate_summary["lateral_mean_m"]
        <= base_summary["lateral_mean_m"]
    )
    per_seed[str(seed)] = {
        "base": base_summary,
        "candidate": candidate_summary,
        "candidate_non_worse_steps_heading_lateral": seed_non_worse,
    }
    all_base_cases.extend(base_panel["cases"])
    all_candidate_cases.extend(candidate_panel["cases"])


def aggregate(cases: list[dict]) -> dict:
    return {
        "case_count": len(cases),
        "survived": sum(bool(case["survived_full_horizon"]) for case in cases),
        "steps": sum(int(case["steps_completed"]) for case in cases),
        "low_speed_steps": sum(
            int(case["steps_completed"])
            for case in cases
            if float(case["speed_mps"]) <= 0.200001
        ),
        "world_heading_mean_rad": fmean(
            float(case["world_heading_error_abs_max_rad"]) for case in cases
        ),
        "relative_heading_mean_rad": fmean(
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


aggregates = {
    "BASE": aggregate(all_base_cases),
    "NEUTRAL05": aggregate(all_candidate_cases),
}
base = aggregates["BASE"]
candidate = aggregates["NEUTRAL05"]
seed_non_worse_count = sum(
    bool(item["candidate_non_worse_steps_heading_lateral"])
    for item in per_seed.values()
)
checks = {
    "initial_conditions_exactly_paired": pairing_ok,
    "survival_not_below_base": candidate["survived"] >= base["survived"],
    "steps_not_below_base": candidate["steps"] >= base["steps"],
    "low_speed_steps_not_below_base": (
        candidate["low_speed_steps"] >= base["low_speed_steps"]
    ),
    "world_heading_not_above_base": (
        candidate["world_heading_mean_rad"] <= base["world_heading_mean_rad"]
    ),
    "lateral_not_above_base": (
        candidate["lateral_mean_m"] <= base["lateral_mean_m"]
    ),
    "base_normal_survivors_preserved": base_normal_survivors_preserved,
    "no_normal_heldout_regression": not heldout_regressions,
    "at_least_two_of_three_seeds_non_worse": seed_non_worse_count >= 2,
    "residual_within_0p05": candidate["residual_abs_max"] <= 0.050001,
}
passed = all(checks.values())
decision = (
    "PROMOTE_NEUTRAL05_TO_PERTURBATION_ROBUST_CANDIDATE"
    if passed
    else "KEEP_STAGE208_BASE"
)
result = {
    "schema_version": 1,
    "decision": decision,
    "passed": passed,
    "checks": checks,
    "seeds": seeds,
    "seed_non_worse_count": seed_non_worse_count,
    "pairing": pairing_details,
    "heldout_regressions": heldout_regressions,
    "per_seed": per_seed,
    "aggregates": aggregates,
}

args.output_json.parent.mkdir(parents=True, exist_ok=True)
args.output_json.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

rows = []
for name in ("BASE", "NEUTRAL05"):
    item = aggregates[name]
    rows.append(
        "| {name} | {survived}/36 | {steps} | {low} | {heading:.4f} | "
        "{relative:.4f} | {lateral:.4f} | {tilt:.4f} |".format(
            name=name,
            survived=item["survived"],
            steps=item["steps"],
            low=item["low_speed_steps"],
            heading=item["world_heading_mean_rad"],
            relative=item["relative_heading_mean_rad"],
            lateral=item["lateral_mean_m"],
            tilt=item["tilt_mean_rad"],
        )
    )
seed_rows = []
for seed in seeds:
    item = per_seed[str(seed)]
    base_item = item["base"]
    candidate_item = item["candidate"]
    seed_rows.append(
        "| {seed} | {bs} | {cs} | {bh:.4f} | {ch:.4f} | "
        "{bl:.4f} | {cl:.4f} | {ok} |".format(
            seed=seed,
            bs=base_item["steps"],
            cs=candidate_item["steps"],
            bh=base_item["world_heading_mean_rad"],
            ch=candidate_item["world_heading_mean_rad"],
            bl=base_item["lateral_mean_m"],
            cl=candidate_item["lateral_mean_m"],
            ok="✓" if item["candidate_non_worse_steps_heading_lateral"] else "✗",
        )
    )
check_rows = [
    f"- {'✓' if value else '✗'} `{name}`"
    for name, value in checks.items()
]
args.output_md.write_text(
    "\n".join(
        [
            "# Stage7C 成对扰动鲁棒性裁决",
            "",
            f"结论：**{decision}**。",
            "",
            "| 分支 | 生存 | 总步数 | 低速步数 | 世界 heading | "
            "相对 heading | lateral | tilt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 分 seed",
            "",
            "| seed | BASE steps | NEUTRAL steps | BASE heading | "
            "NEUTRAL heading | BASE lateral | NEUTRAL lateral | 三项不劣 |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|",
            *seed_rows,
            "",
            "## 预注册门",
            "",
            *check_rows,
            "",
            f"完全配对：`{pairing_ok}`；满足三项不劣的 seed："
            f"`{seed_non_worse_count}/3`。",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps({"decision": decision, "passed": passed}, indent=2))
