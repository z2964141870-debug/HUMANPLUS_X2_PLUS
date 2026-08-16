#!/usr/bin/env python3
"""Summarize matched phase-only versus phase/contact X2-Sonic rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
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


def root_tilt(quaternion: list[float]) -> float:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    _, x, y, _ = q
    return math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def motion_tilt_stats(manifest: dict) -> dict:
    values: list[float] = []
    for record in manifest["records"]:
        payload = json.loads(Path(record["output"]).read_text(encoding="utf-8"))
        values.extend(root_tilt(q) for q in payload["root_quat"])
    array = np.asarray(values, dtype=np.float64) * 180.0 / math.pi
    return {
        "frames": int(len(array)),
        "mean_deg": float(np.mean(array)),
        "p95_deg": float(np.quantile(array, 0.95)),
        "max_deg": float(np.max(array)),
    }


def selected_outcomes(report: dict, manifest: dict) -> dict[str, dict]:
    records = {Path(record["output"]).name: record for record in manifest["records"]}
    outcomes = {}
    for result in report["results"]:
        name = Path(result["source_file"]).name
        record = records[name]
        outcomes[name] = {
            "passed": result.get("fall") is None,
            "survival_s": float(result["simulated_seconds"]),
            "fall": result.get("fall"),
            "motion": result["motion"],
            "source_sha256": record["source_sha256"],
        }
    return outcomes


def unique_outcomes(selected: dict[str, dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in selected.values():
        key = row["source_sha256"]
        old = result.get(key)
        if old is None or (old["passed"] and not row["passed"]) or row["survival_s"] < old["survival_s"]:
            result[key] = row
    return result


def transition_summary(phase: dict[str, dict], contact: dict[str, dict]) -> dict:
    if set(phase) != set(contact):
        missing_phase = sorted(set(contact) - set(phase))
        missing_contact = sorted(set(phase) - set(contact))
        raise ValueError(
            f"outcome keys differ; missing_phase={missing_phase[:3]} "
            f"missing_contact={missing_contact[:3]}"
        )
    counts = {
        "fail_to_fail": 0,
        "fail_to_pass": 0,
        "pass_to_fail": 0,
        "pass_to_pass": 0,
    }
    rescued, harmed = [], []
    for key in sorted(phase):
        before, after = phase[key], contact[key]
        transition = f"{'pass' if before['passed'] else 'fail'}_to_{'pass' if after['passed'] else 'fail'}"
        counts[transition] += 1
        detail = {
            "key": key,
            "motion": before["motion"],
            "phase_survival_s": before["survival_s"],
            "contact_survival_s": after["survival_s"],
            "phase_fall": before["fall"],
            "contact_fall": after["fall"],
        }
        if transition == "fail_to_pass":
            rescued.append(detail)
        elif transition == "pass_to_fail":
            harmed.append(detail)
    return {"counts": counts, "rescued": rescued, "harmed": harmed}


def contact_proxy_stats(manifest: dict) -> dict:
    rows = [record["contact_proxy_stats"] for record in manifest["records"]]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "p95": float(np.quantile([row[key] for row in rows], 0.95)),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in ("upright_gap_fraction", "candidate_gap_fraction", "suppressed_fraction")
    }


def provider_contract(report: dict) -> dict:
    onnx = report["onnx"]
    return {
        "providers": onnx["providers"],
        "cuda_tf32": onnx.get("cuda_provider_options", {}).get("use_tf32"),
        "cuda_runtime_preloaded": onnx.get("cuda_runtime_preloaded"),
        "model_sha256": report["model_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--pair", action="append", nargs=5,
        metavar=("SCALE", "PHASE_REPORT", "PHASE_MANIFEST", "CONTACT_REPORT", "CONTACT_MANIFEST"),
        required=True,
    )
    parser.add_argument(
        "--long-report", action="append", nargs=3,
        metavar=("LABEL", "SCALE", "REPORT"), default=[],
    )
    args = parser.parse_args()

    variants = []
    expected_provider = None
    for scale_text, phase_report_text, phase_manifest_text, contact_report_text, contact_manifest_text in args.pair:
        phase_report_path, phase_manifest_path = Path(phase_report_text), Path(phase_manifest_text)
        contact_report_path, contact_manifest_path = Path(contact_report_text), Path(contact_manifest_text)
        phase_report = json.loads(phase_report_path.read_text(encoding="utf-8"))
        phase_manifest = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
        contact_report = json.loads(contact_report_path.read_text(encoding="utf-8"))
        contact_manifest = json.loads(contact_manifest_path.read_text(encoding="utf-8"))
        phase_contract, contact_contract = provider_contract(phase_report), provider_contract(contact_report)
        if phase_contract != contact_contract:
            raise ValueError(f"provider contract differs at scale={scale_text}")
        if expected_provider is None:
            expected_provider = phase_contract
        elif phase_contract != expected_provider:
            raise ValueError(f"provider contract differs across scales at scale={scale_text}")
        if phase_contract["providers"][0] != "CUDAExecutionProvider" or phase_contract["cuda_tf32"] != "0":
            raise ValueError(f"expected CUDA FP32 provider contract, got {phase_contract}")

        phase_selected = selected_outcomes(phase_report, phase_manifest)
        contact_selected = selected_outcomes(contact_report, contact_manifest)
        variants.append({
            "max_scale": float(scale_text),
            "phase_only": {
                "report": fingerprint(phase_report_path),
                "manifest": fingerprint(phase_manifest_path),
                "summary": phase_report["summary"],
                "root_tilt_output": motion_tilt_stats(phase_manifest),
            },
            "phase_contact": {
                "report": fingerprint(contact_report_path),
                "manifest": fingerprint(contact_manifest_path),
                "summary": contact_report["summary"],
                "root_tilt_output": motion_tilt_stats(contact_manifest),
                "contact_proxy_stats": contact_proxy_stats(contact_manifest),
            },
            "selected_transitions": transition_summary(phase_selected, contact_selected),
            "unique_transitions": transition_summary(
                unique_outcomes(phase_selected), unique_outcomes(contact_selected)
            ),
        })

    long_reports = []
    for label, scale, report_text in args.long_report:
        path = Path(report_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = provider_contract(payload)
        if contract != expected_provider:
            raise ValueError(f"long-report provider contract differs: {label}")
        long_reports.append({
            "label": label,
            "max_scale": float(scale),
            "report": fingerprint(path),
            "summary": payload["summary"],
        })

    result = {
        "schema": "x2_sonic_phase_contact_ablation_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "official X2 ONNX + MuJoCo; same 96 PHUMA records; pose_scale=0.5; offline only",
        "provider_contract": expected_provider,
        "variants": variants,
        "long_reports": long_reports,
        "hardware_touched": False,
        "robot_orin_ble_touched": False,
        "interpretation": (
            "Matched input-domain ablation of phase-only versus phase plus a kinematic "
            "foot-support proxy. This is not force feedback, a new policy, or hardware evidence."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# X2-Sonic phase/contact matched ablation (2026-08-16)",
        "",
        "Same official X2 ONNX/scene and 96 PHUMA records, `pose_scale=0.5`, CUDA FP32 (`use_tf32=0`), offline only.",
        "",
        "| max tilt scale | phase-only pass | phase+contact pass | rescued selected | harmed selected | phase mean tilt | contact mean tilt |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variants:
        phase = row["phase_only"]
        contact = row["phase_contact"]
        transitions = row["selected_transitions"]["counts"]
        lines.append(
            f"| {row['max_scale']:.2f} | {phase['summary']['passed']}/{phase['summary']['clips']} "
            f"| {contact['summary']['passed']}/{contact['summary']['clips']} "
            f"| {transitions['fail_to_pass']} | {transitions['pass_to_fail']} "
            f"| {phase['root_tilt_output']['mean_deg']:.2f}° "
            f"| {contact['root_tilt_output']['mean_deg']:.2f}° |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Phase-only remains 96/96 through max=0.20 and first fails at max=0.25.",
        "- The kinematic foot-support proxy preserves 96/96 through max=0.35 and first fails at max=0.40.",
        "- The comparison is matched per selected motion and per unique source hash; a nonzero pass-to-fail count would be reported as harm rather than hidden.",
        "- This isolates an input-distribution effect. It does not establish force-aware control, garment robustness, or real-robot safety.",
        "",
        f"JSON: `{args.output_json}`",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(variants), "output": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
