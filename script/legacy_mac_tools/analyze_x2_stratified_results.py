#!/usr/bin/env python3
"""Join PHUMA source statistics with X2-Sonic closed-loop results.

This is deliberately an offline, dependency-free analysis step.  The closed-loop
evaluator reports canonical JSON paths, while the inventory reports original PHUMA
paths; the canonical manifest is the provenance bridge between the two.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def group_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "unknown")].append(row)
    out = {}
    for name, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        survival = [float(x["simulated_seconds"]) for x in items]
        drift = [float(x["max_root_xy_drift_m"]) for x in items]
        tilt = [float(x["max_tilt_rad"]) for x in items]
        violation = [float(x["max_joint_limit_violation_rad"]) for x in items]
        passed = sum(x["passed"] for x in items)
        reasons = Counter(x["fall_reason"] for x in items if x["fall_reason"])
        out[str(name)] = {
            "records": len(items),
            "passed": passed,
            "failed": len(items) - passed,
            "pass_rate": passed / len(items),
            "survival_seconds_mean": mean(survival),
            "survival_seconds_median": median(survival),
            "survival_seconds_min": min(survival),
            "root_xy_drift_mean_m": mean(drift),
            "root_xy_drift_max_m": max(drift),
            "tilt_mean_rad": mean(tilt),
            "tilt_max_rad": max(tilt),
            "joint_limit_violation_mean_rad": mean(violation),
            "fall_reasons": dict(reasons),
        }
    return out


def root_tilt_bin(value):
    if value < 0.15:
        return "low"
    if value < 0.35:
        return "mid"
    return "high"


def contact_gap_bin(value):
    if value < 0.25:
        return "good"
    if value < 0.50:
        return "mixed"
    return "poor"


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return None
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx and deny else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--canonical-manifest", type=Path, required=True)
    ap.add_argument("--closed-loop", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    inventory = load(args.inventory)
    canonical = load(args.canonical_manifest)
    closed = load(args.closed_loop)

    inv_by_source = {r["source"]: r for r in inventory["records"]}
    canon_by_output = {r["output"]: r for r in canonical["records"]}
    rows = []
    unmatched = []
    for result in closed["results"]:
        bridge = canon_by_output.get(result["source_file"])
        if bridge is None or bridge["source"] not in inv_by_source:
            unmatched.append(result.get("source_file"))
            continue
        stats = inv_by_source[bridge["source"]]
        fall = result.get("fall") or {}
        row = {
            "motion": result["motion"],
            "source": bridge["source"],
            "source_sha256": bridge["source_sha256"],
            "subset": stats["subset"],
            "category": stats["category"],
            "root_speed_bin": stats["root_speed_bin"],
            "joint_speed_bin": stats["joint_speed_bin"],
            "pose_amplitude_bin": stats["pose_amplitude_bin"],
            "stratum": stats["stratum"],
            "root_speed_p95_mps": stats["root_speed_p95_mps"],
            "root_xy_path_m": stats["root_xy_path_m"],
            "root_tilt_p95_rad": stats.get("root_tilt_p95_rad", 0.0),
            "root_tilt_max_rad": stats.get("root_tilt_max_rad", 0.0),
            "root_angular_speed_p95_radps": stats.get("root_angular_speed_p95_radps", 0.0),
            "root_tilt_bin": root_tilt_bin(float(stats.get("root_tilt_p95_rad", 0.0))),
            "foot_center_z_min_m": stats.get("foot_center_z_min_m", 0.0),
            "foot_center_z_p95_m": stats.get("foot_center_z_p95_m", 0.0),
            "foot_near_floor_fraction": stats.get("foot_near_floor_fraction", 0.0),
            "foot_contact_gap_fraction": stats.get("foot_contact_gap_fraction", 1.0),
            "contact_gap_bin": contact_gap_bin(float(stats.get("foot_contact_gap_fraction", 1.0))),
            "joint_speed_p95_radps": stats["joint_speed_p95_radps"],
            "joint_abs_angle_p99_rad": stats["joint_abs_angle_p99_rad"],
            "root_z_range_m": stats["root_z_range_m"],
            "passed": result.get("fall") is None,
            "simulated_seconds": result["simulated_seconds"],
            "max_root_xy_drift_m": result["max_root_xy_drift_m"],
            "max_tilt_rad": result["max_tilt_rad"],
            "max_joint_limit_violation_rad": result["max_joint_limit_violation_rad"],
            "fall_reason": fall.get("reason"),
            "fall_time_s": fall.get("time"),
        }
        rows.append(row)

    summary = closed.get("summary", {})
    # The four PHUMA subsets intentionally share some source motions.  Keep the
    # selection-level view (96 rows) for subset accounting, but also provide a
    # content-level view so duplicated files cannot inflate scientific claims.
    unique_by_sha = {}
    duplicate_outcome_mismatches = []
    for row in rows:
        prior = unique_by_sha.get(row["source_sha256"])
        if prior is None:
            unique_by_sha[row["source_sha256"]] = row
        else:
            for field in ("passed", "simulated_seconds", "max_tilt_rad", "max_root_xy_drift_m"):
                if prior[field] != row[field]:
                    duplicate_outcome_mismatches.append(
                        {"source_sha256": row["source_sha256"], "field": field, "first": prior[field], "duplicate": row[field]}
                    )
    unique_rows = list(unique_by_sha.values())
    correlations = {
        "root_speed_p95_vs_survival": pearson(
            [r["root_speed_p95_mps"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "joint_speed_p95_vs_survival": pearson(
            [r["joint_speed_p95_radps"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "pose_angle_p99_vs_survival": pearson(
            [r["joint_abs_angle_p99_rad"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "root_z_range_vs_survival": pearson(
            [r["root_z_range_m"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "root_xy_path_vs_survival": pearson(
            [r["root_xy_path_m"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "root_tilt_p95_vs_survival": pearson(
            [r["root_tilt_p95_rad"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "root_angular_speed_p95_vs_survival": pearson(
            [r["root_angular_speed_p95_radps"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
        "foot_contact_gap_vs_survival": pearson(
            [r["foot_contact_gap_fraction"] for r in rows],
            [r["simulated_seconds"] for r in rows],
        ),
    }
    report = {
        "schema": "x2_sonic_stratified_closed_loop_analysis_v1",
        "inventory": str(args.inventory),
        "canonical_manifest": str(args.canonical_manifest),
        "closed_loop": str(args.closed_loop),
        "closed_loop_summary": summary,
        "matched_records": len(rows),
        "unique_content_records": len(unique_rows),
        "duplicate_selection_records": len(rows) - len(unique_rows),
        "duplicate_outcome_mismatches": duplicate_outcome_mismatches,
        "unmatched_records": len(unmatched),
        "unmatched_sources": unmatched,
        "correlations": correlations,
        "by_subset": group_summary(rows, "subset"),
        "by_category": group_summary(rows, "category"),
        "by_root_speed_bin": group_summary(rows, "root_speed_bin"),
        "by_joint_speed_bin": group_summary(rows, "joint_speed_bin"),
        "by_pose_amplitude_bin": group_summary(rows, "pose_amplitude_bin"),
        "by_root_tilt_bin": group_summary(rows, "root_tilt_bin"),
        "by_contact_gap_bin": group_summary(rows, "contact_gap_bin"),
        "by_stratum": group_summary(rows, "stratum"),
        "unique_by_category": group_summary(unique_rows, "category"),
        "unique_by_root_speed_bin": group_summary(unique_rows, "root_speed_bin"),
        "unique_by_joint_speed_bin": group_summary(unique_rows, "joint_speed_bin"),
        "unique_by_pose_amplitude_bin": group_summary(unique_rows, "pose_amplitude_bin"),
        "unique_by_root_tilt_bin": group_summary(unique_rows, "root_tilt_bin"),
        "unique_by_contact_gap_bin": group_summary(unique_rows, "contact_gap_bin"),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    def md_table(title, data):
        lines = [f"### {title}", "", "| group | n | pass | rate | mean survival (s) | max tilt (rad) | max drift (m) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, x in data.items():
            lines.append(
                f"| `{name}` | {x['records']} | {x['passed']} | {x['pass_rate']:.3f} | "
                f"{x['survival_seconds_mean']:.2f} | {x['tilt_max_rad']:.3f} | {x['root_xy_drift_max_m']:.2f} |"
            )
        return lines

    lines = [
        "# X2-Sonic PHUMA stratified closed-loop analysis",
        "",
        f"- Matched selection records: **{len(rows)} / {len(closed['results'])}**",
        f"- Unique source contents: **{len(unique_rows)}** (duplicate selection rows: {len(rows) - len(unique_rows)}).",
        f"- Selection-level overall: **{summary.get('passed', 0)} / {summary.get('clips', 0)} passed**; minimum survival `{summary.get('min_survival_seconds')}` s.",
        f"- Content-level overall: **{sum(x['passed'] for x in unique_rows)} / {len(unique_rows)} passed**.",
        "- This is an offline MuJoCo diagnosis of the published X2-Sonic policy, not a real-robot test.",
        "",
        "## Interpretation guardrails",
        "",
        "Pass/fail is defined by the evaluator's root-height/tilt safety gate. Correlations are diagnostic, not causal; the next experiment must change one input-distribution variable at a time.",
        "",
        "## Correlations with survival time",
        "",
    ]
    for k, v in correlations.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    for title, key in [
        ("By subset", "by_subset"),
        ("By source category", "by_category"),
        ("By root-speed bin", "by_root_speed_bin"),
        ("By joint-speed bin", "by_joint_speed_bin"),
        ("By pose-amplitude bin", "by_pose_amplitude_bin"),
        ("By root-tilt bin", "by_root_tilt_bin"),
        ("By contact-gap bin", "by_contact_gap_bin"),
    ]:
        lines.extend(md_table(title, report[key]))
        lines.append("")
    lines.extend(["## Content-level tables (deduplicated by source SHA256)", ""])
    for title, key in [
        ("Unique by source category", "unique_by_category"),
        ("Unique by root-speed bin", "unique_by_root_speed_bin"),
        ("Unique by joint-speed bin", "unique_by_joint_speed_bin"),
        ("Unique by pose-amplitude bin", "unique_by_pose_amplitude_bin"),
        ("Unique by root-tilt bin", "unique_by_root_tilt_bin"),
        ("Unique by contact-gap bin", "unique_by_contact_gap_bin"),
    ]:
        lines.extend(md_table(title, report[key]))
        lines.append("")
    failures = sorted((r for r in rows if not r["passed"]), key=lambda r: r["simulated_seconds"])
    lines.extend(["### Earliest failures", "", "| survival (s) | category | subset | root p95 | joint p95 | pose p99 | reason | source |", "|---:|---|---|---:|---:|---:|---|---|"])
    for r in failures[:20]:
        lines.append(f"| {r['simulated_seconds']:.2f} | {r['category']} | {r['subset']} | {r['root_speed_p95_mps']:.2f} | {r['joint_speed_p95_radps']:.2f} | {r['joint_abs_angle_p99_rad']:.2f} | {r['fall_reason']} | `{Path(r['source']).name}` |")
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"matched": len(rows), "unique_content": len(unique_rows), "unmatched": len(unmatched), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
