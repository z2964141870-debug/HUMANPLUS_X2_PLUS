#!/usr/bin/env python3
"""Summarize the preregistered Stage5 upper-motion safety panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--expected", type=int, default=12)
    parser.add_argument("--variant-prefix", default="stage5")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        args.report_dir.glob(
            f"{args.variant_prefix}_*_delay_s{args.steps}_stage5.json"
        )
    )
    rows: list[dict] = []
    for path in paths:
        result = json.loads(path.read_text())
        safety = result["stage5_safety"]
        metrics = result["metrics"]
        row = {
            "case": path.name.removesuffix("_stage5.json").removeprefix(
                f"{args.variant_prefix}_"
            ),
            "path": str(path.resolve()),
            "passed": bool(result["passed"]),
            "survived": bool(result["gate"]["survived"]),
            "active_upper_target": bool(result["gate"]["active_upper_target"]),
            "target_excursion_rms_rad": float(
                result["upper"]["target_excursion_rms_rad"]
            ),
            "target_excursion_max_rad": float(
                safety["metrics"]["target_excursion_max_rad"]
            ),
            "target_velocity_max_radps": float(
                safety["metrics"]["target_velocity_abs_max_radps"]
            ),
            "tracking_p95_rad": float(result["upper"]["tracking_abs_p95_rad"]),
            "heading_degradation_rad": float(
                metrics["heading_deviation_abs_max_rad"]["delta"]
            ),
            "lateral_degradation_m": float(
                metrics["lateral_drift_max_m"]["delta"]
            ),
            "base_tilt_max_rad": float(
                metrics["base_tilt_max_rad"]["candidate"]
            ),
            "left_foot_lift_max_m": float(
                metrics["left_foot_lift_max_m"]["candidate"]
            ),
            "right_foot_lift_max_m": float(
                metrics["right_foot_lift_max_m"]["candidate"]
            ),
            "hazard_fraction": float(safety["metrics"]["hazard_fraction"]),
            "failed_gates": sorted(
                [
                    name
                    for name, passed in {
                        **result["gate"],
                        **safety["gate"],
                    }.items()
                    if not passed
                ]
            ),
        }
        rows.append(row)

    completed = len(rows)
    passed = sum(row["passed"] for row in rows)
    falls = sum(not row["survived"] for row in rows)
    if completed < args.expected:
        verdict = "INCOMPLETE"
        recommendation = "补齐缺失 case 后再裁决。"
    elif passed == args.expected:
        verdict = "UNLOCK_FROZEN_SONIC_UPPER"
        recommendation = "允许测试冻结 SONIC 上肢经过同一 Adapter 接入。"
    elif passed >= args.expected - 2 and falls == 0:
        verdict = "BOUNDED_CANDIDATE"
        recommendation = "保留候选，先分析失败动作类别，不解锁 SONIC。"
    else:
        verdict = "DO_NOT_UNLOCK"
        recommendation = "不接 SONIC，转向下层 yaw/lateral 扰动鲁棒性。"

    summary = {
        "schema_version": 1,
        "expected_cases": args.expected,
        "completed_cases": completed,
        "passed_cases": passed,
        "survival_failures": falls,
        "verdict": verdict,
        "recommendation": recommendation,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# Stage5 有界上肢安全 Adapter 面板",
        "",
        f"- 完成：`{completed}/{args.expected}`",
        f"- 通过：`{passed}/{args.expected}`",
        f"- 生存失败：`{falls}`",
        f"- 裁决：`{verdict}`",
        f"- 建议：{recommendation}",
        "",
        "| case | pass | survive | active | exc RMS/max rad | vel max rad/s | track p95 rad | Δheading rad | Δlateral m | tilt rad | feet lift mm L/R | hazard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {passed} | {survived} | {active} | {rms}/{exc} | "
            "{vel} | {track} | {heading} | {lateral} | {tilt} | "
            "{left}/{right} | {hazard} |".format(
                case=row["case"],
                passed="✓" if row["passed"] else "✗",
                survived="✓" if row["survived"] else "✗",
                active="✓" if row["active_upper_target"] else "✗",
                rms=_fmt(row["target_excursion_rms_rad"]),
                exc=_fmt(row["target_excursion_max_rad"]),
                vel=_fmt(row["target_velocity_max_radps"]),
                track=_fmt(row["tracking_p95_rad"]),
                heading=_fmt(row["heading_degradation_rad"]),
                lateral=_fmt(row["lateral_degradation_m"]),
                tilt=_fmt(row["base_tilt_max_rad"]),
                left=_fmt(row["left_foot_lift_max_m"] * 1000.0, 1),
                right=_fmt(row["right_foot_lift_max_m"] * 1000.0, 1),
                hazard=_fmt(row["hazard_fraction"], 3),
            )
        )
    failed = [row for row in rows if not row["passed"]]
    if failed:
        lines.extend(["", "## 未通过项", ""])
        for row in failed:
            gates = ", ".join(row["failed_gates"]) or "unknown"
            lines.append(f"- `{row['case']}`：{gates}")
    args.output_md.write_text("\n".join(lines) + "\n")
    print(
        f"Stage5: {passed}/{args.expected} passed, "
        f"{falls} survival failures, verdict={verdict}"
    )
    return 0 if completed == args.expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
