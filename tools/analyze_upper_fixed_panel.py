#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcpeft_upper_panel import compare_to_baseline, summarize_trace  # noqa: E402


def parse_trace(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trace must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid trace: {value}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True, type=parse_trace)
    parser.add_argument("--baseline")
    parser.add_argument("--panel-name", default="upper_body_capability")
    parser.add_argument("--title", default="Upper-Body Capability Panel")
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()

    labels = [label for label, _ in args.trace]
    baseline_label = args.baseline or labels[0]
    if baseline_label not in labels:
        raise SystemExit(f"baseline label not found: {baseline_label}")
    traces = {
        label: summarize_trace(path, control_dt=args.control_dt) for label, path in args.trace
    }
    comparisons = {
        label: compare_to_baseline(summary, traces[baseline_label])
        for label, summary in traces.items()
        if label != baseline_label
    }
    payload = {
        "schema_version": 1,
        "panel": args.panel_name,
        "baseline": baseline_label,
        "gate_thresholds": {
            "root_z_floor_m": 0.45,
            "anchor_p95_limit_m": 0.10,
            "foot_p95_limit_m": 0.10,
            "wrist_p95_limit_m": 0.15,
            "capability_relative_degradation_limit": 0.05,
        },
        "traces": traces,
        "comparisons": comparisons,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"# {args.title}",
        "",
        f"Baseline: `{baseline_label}`. Only the first frame-zero episode per motion is counted.",
        "",
        "| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in traces.items():
        agg = summary["aggregate"]
        lines.append(
            f"| `{label}` | {agg['completed']}/{agg['motions']} | {agg['stable']}/{agg['motions']} | "
            f"{agg['strict']}/{agg['motions']} | {1000*agg['wrist_error_m']['mean']:.2f} | "
            f"{1000*agg['wrist_error_m']['p95']:.2f} | {1000*agg['upper_error_m']['mean']:.2f} | "
            f"{1000*agg['anchor_pos_m']['p95']:.2f} | {1000*agg['foot_error_m']['p95']:.2f} | "
            f"{agg['root_z_min_m']:.3f} |"
        )
    lines += ["", "## Per-motion", ""]
    for label, summary in traces.items():
        lines += [f"### {label}", "", "| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |", "| --- | ---: | --- | --- | --- | ---: | ---: | ---: |"]
        for episode in summary["episodes"]:
            metrics = episode["metrics"]
            gates = episode["gates"]
            lines.append(
                f"| `{episode['motion_key']}` | {episode['samples']} | {gates['completed']} | "
                f"{gates['stable']} | {gates['strict']} | {1000*metrics['wrist_error_m']['mean']:.2f}/"
                f"{1000*metrics['wrist_error_m']['p95']:.2f} | {1000*metrics['anchor_pos_m']['p95']:.2f} | "
                f"{1000*metrics['foot_error_m']['p95']:.2f} |"
            )
        lines.append("")
    if comparisons:
        lines += ["## Baseline-relative preservation", ""]
        for label, comparison in comparisons.items():
            deltas = comparison["relative_deltas"]
            lines.append(
                f"- `{label}`: preserve@5%=`{comparison['upper_capability_preserved_5pct']}`, "
                f"wrist mean `{100*deltas['wrist_mean']:+.2f}%`, wrist p95 `{100*deltas['wrist_p95']:+.2f}%`, "
                f"upper mean `{100*deltas['upper_mean']:+.2f}%`, anchor p95 `{100*deltas['anchor_p95']:+.2f}%`."
            )
    args.report_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
