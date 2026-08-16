#!/usr/bin/env python3
"""Summarize X2-Sonic global and phase-aware root-tilt probes."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def root_tilt(q: list[float]) -> float:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    _, x, y, _ = q
    return math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def motion_tilt_stats(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = []
    for record in manifest["records"]:
        payload = json.loads(Path(record["output"]).read_text(encoding="utf-8"))
        values.extend(root_tilt(q) for q in payload["root_quat"])
    values = np.asarray(values, dtype=np.float64)
    return {
        "frames": int(len(values)),
        "mean_deg": float(np.mean(values) * 180.0 / math.pi),
        "p50_deg": float(np.quantile(values, 0.50) * 180.0 / math.pi),
        "p95_deg": float(np.quantile(values, 0.95) * 180.0 / math.pi),
        "max_deg": float(np.max(values) * 180.0 / math.pi),
    }


def unique_outcomes(report_path: Path, manifest_path: Path) -> dict[str, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_output = {record["output"]: record for record in manifest["records"]}
    outcomes: dict[str, dict] = {}
    for result in report["results"]:
        record = by_output[result["source_file"]]
        source_sha = record["source_sha256"]
        row = {
            "passed": result.get("fall") is None,
            "survival_s": float(result["simulated_seconds"]),
            "fall": result.get("fall"),
            "motion": result["motion"],
        }
        old = outcomes.get(source_sha)
        if old is None or (old["passed"] and not row["passed"]) or row["survival_s"] < old["survival_s"]:
            outcomes[source_sha] = row
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--variant", action="append", nargs=4, metavar=("NAME", "LABEL", "REPORT", "MANIFEST"), required=True)
    args = parser.parse_args()

    variants = {}
    for name, label, report_text, manifest_text in args.variant:
        report, manifest = Path(report_text), Path(manifest_text)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
        outcomes = unique_outcomes(report, manifest)
        entry = {
            "label": label,
            "report": fingerprint(report),
            "motion_manifest": fingerprint(manifest),
            "summary": summary,
            "root_tilt_output": motion_tilt_stats(manifest),
            "unique_source_count": len(outcomes),
            "unique_outcome": {"passed": sum(row["passed"] for row in outcomes.values()), "total": len(outcomes)},
        }
        if name.startswith("phase_aware"):
            scale_key = "scale_stats" if "scale_stats" in manifest_payload["records"][0] else "phase_scale_stats"
            all_stats = [record[scale_key] for record in manifest_payload["records"]]
            entry["allowance_stats"] = {
                key: {
                    "mean": float(np.mean([stats[key] for stats in all_stats])),
                    "median": float(np.median([stats[key] for stats in all_stats])),
                    "min": float(np.min([stats[key] for stats in all_stats])),
                    "max": float(np.max([stats[key] for stats in all_stats])),
                }
                for key in ("scale_mean", "scale_p50", "scale_p95", "fraction_zero", "fraction_below_max")
            }
            report_30s = report.with_name(report.stem.replace("_5s", "_30s") + report.suffix)
            if report_30s.exists():
                entry["report_30s"] = fingerprint(report_30s)
                entry["summary_30s"] = json.loads(report_30s.read_text(encoding="utf-8"))["summary"]
            if "contact_proxy_stats" in manifest_payload["records"][0]:
                contact_stats = [record["contact_proxy_stats"] for record in manifest_payload["records"]]
                entry["contact_proxy_stats"] = {
                    key: {
                        "mean": float(np.mean([stats[key] for stats in contact_stats])),
                        "median": float(np.median([stats[key] for stats in contact_stats])),
                        "p95": float(np.quantile([stats[key] for stats in contact_stats], 0.95)),
                        "max": float(np.max([stats[key] for stats in contact_stats])),
                    }
                    for key in ("upright_gap_fraction", "candidate_gap_fraction", "suppressed_fraction")
                }
        variants[name] = {"entry": entry, "outcomes": outcomes}

    base = variants["root_tilt_0"]["outcomes"]
    result = {
        "schema": "x2_sonic_phase_aware_root_tilt_probe_v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_sha256": "e7ccd6522010ea660facfb7265fb129dac2b580dd1483cc1dbada73c205309d7",
        "scope": "official X2 MuJoCo only; same 96 selected records and pose_scale=0.5; no Orin/BLE/robot",
        "variants": {},
    }
    for name, value in variants.items():
        transitions = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        for source_sha, old in base.items():
            new = value["outcomes"][source_sha]
            transitions[f"{int(old['passed'])}->{int(new['passed'])}"] += 1
        entry = dict(value["entry"])
        entry["unique_transitions_vs_root_tilt_0"] = transitions
        result["variants"][name] = entry

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# X2-Sonic root-tilt allowance and phase-aware probe (2026-08-16)",
        "",
        "Scope: official ONNX + official MuJoCo scene, same 96 selected PHUMA records, `pose_scale=0.5`, 5-second strict safety gate. No Orin/BLE/robot hardware.",
        "",
        "| variant | selected pass | unique pass | min survival (s) | mean output tilt (deg) | p95 output tilt (deg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for value in result["variants"].values():
        summary, unique, tilt = value["summary"], value["unique_outcome"], value["root_tilt_output"]
        lines.append(f"| {value['label']} | {summary['passed']}/{summary['clips']} | {unique['passed']}/{unique['total']} | {summary['min_survival_seconds']:.2f} | {tilt['mean_deg']:.2f} | {tilt['p95_deg']:.2f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Global root-tilt scale 0.05 is stable on 96/96 selected records; 0.10 drops to 94/96 (one unique downhill content duplicated twice). Larger global scales degrade further.",
        "- The phase-aware max=0.10 gate uses only reference root height and vertical descent, with smooth attenuation below 0.58 m / during descent below -0.05 m/s. The phase/contact variants are reported separately and must not be conflated with force feedback.",
        "- These variants retain more root tilt than global 0.05 while remaining an offline input-domain probe, not proof of garment or hardware readiness. The contact variant uses only a kinematic foot-height/speed proxy and has no measured contact force.",
        "- The gate must be compared against raw input and the root-tilt=0 safety baseline with the same motion IDs; it is not a new policy or a claim that the ONNX model accepts arbitrary human tilt.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{args.output_json}`",
        "- New phase-aware adapter: `/home/yu/projects/BFM-Zero/tools/official_x2/phase_aware_root_tilt_adapter.py`",
    ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: value["summary"] for name, value in result["variants"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
