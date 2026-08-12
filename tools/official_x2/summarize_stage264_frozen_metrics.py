#!/usr/bin/env python3
"""Build the immutable Stage264 baseline metric card from hash-bound evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "stage250": ROOT / "reports/official_x2/stage250_rsl_action_contract_fix_20260808.json",
    "task50": ROOT / "reports/official_x2/stage264_longtrain_readiness_task50.json",
    "phase34": ROOT / "reports/official_x2/phase34_closed_full_trace_manifest.json",
    "visual": ROOT / "reports/baseline/x2_stage250_visual_audit.json",
    "side_video": ROOT / "videos/official_x2/report/smooth_trace_replay/x2_straight_posture_official_trace_smooth.mp4",
    "turn_video": ROOT / "videos/official_x2/report/smooth_trace_replay/x2_turn_direction_official_trace_smooth.mp4",
}
EXPECTED = {
    "stage250": "7aedae2656b0071c5585d7bcd3d291dab41d71623a2f32345a7ffe46db7a915d",
    "task50": "e0ce0fa50ca0cce60bc64cd97958abe118ddfd9902c63a4db431bdda7753d81b",
    "phase34": "3371b9166480244c2f42f2f33e6e7b49c68b06851758e3ab76663433287c6551",
    "visual": "7567537456269eeaeb2e25e5f1ea10b0c3f0fa5aed4d4f952f6d8710deaafe70",
    "side_video": "793ba6d5d86a0ac0d596b734599dab563e6c2b2fb0a36ae26baf61c5e833c458",
    "turn_video": "cf454cbef7620a383e98be0e8e71d17464096f48b1860a9d5f5d6999036226e9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str) -> dict:
    path = INPUTS[name]
    actual = sha256(path)
    if actual != EXPECTED[name]:
        raise RuntimeError(f"{name} SHA mismatch: {actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    stage250, task50, phase34, visual = (
        load("stage250"), load("task50"), load("phase34"), load("visual")
    )
    for name in ("side_video", "turn_video"):
        actual = sha256(INPUTS[name])
        if actual != EXPECTED[name]:
            raise RuntimeError(f"{name} SHA mismatch: {actual}")

    pitch = visual["motion_metrics"]
    contact = phase34["realized_contact"]
    gate = phase34["rollout_gate"]
    report = {
        "stage": "stage264_frozen_baseline_metrics_v1",
        "date": "2026-08-12",
        "evidence_scope": "historical hash-bound official traces restored on the new machine; no new physics or optimizer steps",
        "identity": {
            "checkpoint": "Stage219 model_2600.pt",
            "deployment_contract": "Stage250",
            "stage264_is_separate_checkpoint": False,
        },
        "evidence_sha256": {
            name: {"path": str(path), "sha256": EXPECTED[name]}
            for name, path in INPUTS.items()
        },
        "functional_gates": {
            "nominal_start_walk_turn_stop": stage250["functional_gate"]["combined"],
            "nominal_upper_straight": task50["evidence"]["fixed_slow_fast_upper_straight"],
            "pd_x_fixed_fast_upper_straight": task50["evidence"]["pd_0p9_1p0_1p2_x_fixed_fast"],
        },
        "signed_root_pitch": {
            motion: {
                "mean_deg": row["pitch_mean_deg"],
                "mean_rad": math.radians(row["pitch_mean_deg"]),
                "p05_deg": row["pitch_p05_deg"],
                "p95_deg": row["pitch_p95_deg"],
            }
            for motion, row in pitch.items()
        },
        "representative_closed_straight": {
            "forward_m": gate["move_forward_m"],
            "mean_body_vx_mps": gate["move_forward_m"] / 4.0,
            "lateral_m": gate["move_lateral_m"],
            "stop_drift_m": gate["stop_xy_drift_m"],
            "stop_settle_s": gate["stop_settle_s"],
            "signed_pitch_mean_rad": phase34["posture"]["move_signed_root_pitch_mean_rad"],
        },
        "heading": {
            motion: {
                "command_wz_radps": row.get("command_wz_radps", 0.0),
                "yaw_progress_deg": row["yaw_progress_deg"],
                "forward_m": row["forward_m"],
            }
            for motion, row in pitch.items()
        },
        "realized_contact_model_truth": {
            "realized_ds_ss_ds_cycles_move_50hz": contact["realized_ds_ss_ds_cycles_move_50hz"],
            "generator_cycles_move_50hz": contact["generator_cycles_move_50hz"],
            "left": contact["left"],
            "right": contact["right"],
            "strict_slip_gate_pass": contact["strict_slip_gate_p95_le_0p20"],
            "hardware_truth": False,
        },
        "decision": {
            "historical_nominal_gate_frozen": True,
            "new_machine_gate_replayed": False,
            "natural_posture_pass": False,
            "contact_clean_silver": False,
            "task2_required": True,
            "reason": "approximately 10-11 degree backward lean and contact/slip issues remain despite historical 24/24 displacement gates",
        },
    }
    output = ROOT / "stage264_frozen_baseline/metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = report["signed_root_pitch"]
    md = [
        "# Stage264 frozen baseline metrics",
        "",
        "Historical evidence is SHA-bound and restored on the new machine. No new physics or optimizer steps were run.",
        "",
        "| motion | signed pitch mean | p05–p95 | yaw progress |",
        "|---|---:|---:|---:|",
    ]
    for motion in ("straight", "right", "left"):
        row = rows[motion]
        yaw = report["heading"][motion]["yaw_progress_deg"]
        md.append(
            f"| {motion} | {row['mean_deg']:.2f}° | {row['p05_deg']:.2f}° … {row['p95_deg']:.2f}° | {yaw:.2f}° |"
        )
    md.extend([
        "",
        "- Historical nominal start/walk/turn/stop: **24/24**.",
        "- PD×fixed/fast-upper straight robustness: **14/18**.",
        f"- Representative straight speed: **{report['representative_closed_straight']['mean_body_vx_mps']:.3f} m/s**.",
        f"- Realized stance-slip p95: left **{contact['left']['stance_slip_p95_mps']:.3f} m/s**, right **{contact['right']['stance_slip_p95_mps']:.3f} m/s**; strict 0.20 m/s gate fails.",
        "- The 50 fps side/turn review videos are part of the hash-bound evidence inventory.",
        "- New-machine 24/24 replay remains false until the exact gait-template asset is restored.",
        "",
    ])
    output.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
