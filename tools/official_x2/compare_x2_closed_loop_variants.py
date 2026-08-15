#!/usr/bin/env python3
"""Compare closed-loop outcomes for motion-distribution variants by source SHA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def keyed(manifest: Path, report: Path):
    m = load(manifest)
    by_output = {r["output"]: r for r in m["records"]}
    out = {}
    for r in load(report)["results"]:
        bridge = by_output[r["source_file"]]
        sha = bridge["source_sha256"]
        out[sha] = {
            "source": bridge["source"],
            "passed": r.get("fall") is None,
            "survival_seconds": float(r["simulated_seconds"]),
            "drift_m": float(r["max_root_xy_drift_m"]),
            "tilt_rad": float(r["max_tilt_rad"]),
            "joint_violation_rad": float(r["max_joint_limit_violation_rad"]),
            "fall_reason": (r.get("fall") or {}).get("reason"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-manifest", type=Path, required=True)
    ap.add_argument("--baseline-report", type=Path, required=True)
    ap.add_argument("--variant", action="append", nargs=3, metavar=("NAME", "MANIFEST", "REPORT"), required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()
    base = keyed(args.baseline_manifest, args.baseline_report)
    variants = {name: keyed(Path(manifest), Path(report)) for name, manifest, report in args.variant}
    result = {"schema": "x2_sonic_closed_loop_variant_comparison_v1", "baseline_records": len(base), "variants": {}}
    lines = ["# X2-Sonic closed-loop motion-distribution variants", "", f"Baseline unique source contents: **{len(base)}**.", "", "| variant | pass | pass-rate | improved | regressed | mean Δsurvival (s) |", "|---|---:|---:|---:|---:|---:|"]
    for name, current in variants.items():
        rows = []
        for sha, b in base.items():
            v = current[sha]
            rows.append({
                "source_sha256": sha,
                "source": b["source"],
                "baseline_passed": b["passed"],
                "variant_passed": v["passed"],
                "pass_transition": f"{int(b['passed'])}->{int(v['passed'])}",
                "baseline_survival_seconds": b["survival_seconds"],
                "variant_survival_seconds": v["survival_seconds"],
                "delta_survival_seconds": v["survival_seconds"] - b["survival_seconds"],
                "baseline_drift_m": b["drift_m"],
                "variant_drift_m": v["drift_m"],
                "delta_drift_m": v["drift_m"] - b["drift_m"],
                "baseline_tilt_rad": b["tilt_rad"],
                "variant_tilt_rad": v["tilt_rad"],
                "delta_tilt_rad": v["tilt_rad"] - b["tilt_rad"],
                "variant_fall_reason": v["fall_reason"],
            })
        improved = sum((not r["baseline_passed"]) and r["variant_passed"] for r in rows)
        regressed = sum(r["baseline_passed"] and (not r["variant_passed"]) for r in rows)
        passed = sum(r["variant_passed"] for r in rows)
        result["variants"][name] = {
            "records": len(rows), "passed": passed, "pass_rate": passed / len(rows),
            "improved": improved, "regressed": regressed,
            "mean_delta_survival_seconds": mean(r["delta_survival_seconds"] for r in rows),
            "mean_delta_drift_m": mean(r["delta_drift_m"] for r in rows),
            "mean_delta_tilt_rad": mean(r["delta_tilt_rad"] for r in rows),
            "rows": rows,
        }
        lines.append(f"| `{name}` | {passed}/{len(rows)} | {passed/len(rows):.3f} | {improved} | {regressed} | {mean(r['delta_survival_seconds'] for r in rows):+.3f} |")
        lines.extend(["", f"## {name}: pass transitions", "", "| transition | count |", "|---|---:|"])
        transitions = {}
        for r in rows:
            transitions[r["pass_transition"]] = transitions.get(r["pass_transition"], 0) + 1
        for transition, count in sorted(transitions.items()):
            lines.append(f"| `{transition}` | {count} |")
        lines.extend(["", "Largest survival changes:", "", "| Δsurvival (s) | baseline→variant | source |", "|---:|---|---|"])
        for r in sorted(rows, key=lambda x: x["delta_survival_seconds"], reverse=True)[:5] + sorted(rows, key=lambda x: x["delta_survival_seconds"])[:5]:
            lines.append(f"| {r['delta_survival_seconds']:+.2f} | {int(r['baseline_passed'])}→{int(r['variant_passed'])} | `{Path(r['source']).name}` |")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "rows"} for k, v in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
