#!/usr/bin/env python3
"""Offline audit for the single BASE Phase16 f005 recovery suffix capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from official_x2 import audit_phase14_recovery_ood as phase14
from official_x2.recovery_suffix_aggregation import (
    deduplicate_suffix_snapshots,
    load_suffix_sidecar,
    sha256_file,
)


EXPECTED_SLOT_COUNTS = {"main": 360, "stationary": 100, "recovery": 300}
EXPECTED_STAGE335_SHA = "4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
EXPECTED_ADAPTER_SHA = "46634845d68d1de80bb001b4e102855c80ae21192e5f111460ed9115254a05bd"
EXPECTED_F005_SHA = "9bc672fc3c535dbe6cd2709cdec4531eb9b61990457172e9c813a8ac713fc0ca"


def _root_state(row: dict[str, Any]) -> np.ndarray:
    physical = row["physical_state"]
    x, y, z, w = (float(value) for value in physical["root_quaternion_xyzw"])
    gravity = np.asarray(
        [2.0 * (-z * x + w * y), -2.0 * (z * y + w * x), 1.0 - 2.0 * (w * w + z * z)],
        dtype=np.float64,
    )
    gravity /= np.linalg.norm(gravity)
    tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
    pitch = math.asin(float(np.clip(gravity[0], -1.0, 1.0)))
    vx, vy, _ = (float(value) for value in physical["root_linear_velocity_world_mps"])
    return np.asarray(
        [float(physical["root_position_m"][2]), tilt, pitch, math.hypot(vx, vy)],
        dtype=np.float64,
    )


def _query(rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    observation = np.stack([np.asarray(row["observation_93d"], dtype=np.float64) for row in rows])
    groups = {
        name: observation[:, source_slice]
        for name, source_slice in phase14.OBS_GROUPS.items()
    }
    groups["current_action"] = np.stack(
        [np.asarray(row["actual_issued_action"], dtype=np.float64) for row in rows]
    )
    groups["root_state"] = np.stack([_root_state(row) for row in rows])
    meta = [
        {
            "episode": "phase16",
            "time_after_handoff_s": float(row["time_after_handoff_s"]),
            "root_z_m": float(groups["root_state"][index, 0]),
            "root_tilt_rad": float(groups["root_state"][index, 1]),
        }
        for index, row in enumerate(rows)
    ]
    return groups, meta


def _relative_group(current: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold_identical": abs(
            float(current["reference_loo_p95_threshold"])
            - float(historical["reference_loo_p95_threshold"])
        ) < 1.0e-12,
        "phase16_ood_fraction": current["query_ood_fraction"],
        "phase14_ood_fraction": historical["query_ood_fraction"],
        "phase16_first_ood_s": current["first_ood_s_by_episode"].get("phase16"),
        "phase14_first_ood_s_median": historical["first_ood_s_median"],
        "phase16_distance_median": current["query_distance_median"],
        "phase14_distance_median": historical["query_distance_median"],
        "phase16_distance_p95": current["query_distance_p95"],
        "phase14_distance_p95": historical["query_distance_p95"],
    }


def build_report(
    *,
    sidecar_path: Path,
    trace_path: Path,
    adapter_path: Path,
    recovery_model_path: Path,
    stage335_path: Path,
    stage335_report_path: Path,
    phase14_path: Path,
) -> dict[str, Any]:
    sidecar_file_sha = sha256_file(sidecar_path)
    sidecar = load_suffix_sidecar(sidecar_path, sidecar_file_sha)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = trace["summary"]
    hashes = {
        "sidecar_file_sha256": sidecar_file_sha,
        "trace_file_sha256": sha256_file(trace_path),
        "adapter_sha256": sha256_file(adapter_path),
        "recovery_model_sha256": sha256_file(recovery_model_path),
        "stage335_sha256": sha256_file(stage335_path),
    }
    hash_checks = {
        "trace": hashes["trace_file_sha256"] == sidecar["source_trace"]["sha256"],
        "adapter": hashes["adapter_sha256"] == sidecar["adapter"]["sha256"] == EXPECTED_ADAPTER_SHA,
        "f005": hashes["recovery_model_sha256"] == sidecar["recovery_model"]["sha256"] == EXPECTED_F005_SHA,
        "stage335": hashes["stage335_sha256"] == EXPECTED_STAGE335_SHA,
    }
    if not all(hash_checks.values()):
        raise ValueError(f"Phase16 frozen hash mismatch: {hash_checks}")
    if summary.get("policy_slot_inference_counts") != EXPECTED_SLOT_COUNTS:
        raise ValueError("Phase16 policy-slot counts differ from the frozen contract")

    rows = sidecar["rows"]
    times = np.asarray([row["time_after_handoff_s"] for row in rows], dtype=np.float64)
    ticks = np.asarray([row["source_tick"] for row in rows], dtype=np.int64)
    representatives, assignments = deduplicate_suffix_snapshots(rows)
    query, meta = _query(rows)
    training, labels, training_provenance = phase14.load_stage335(
        stage335_path, stage335_report_path
    )
    training_all, _ = phase14.analyze_reference(query, training, meta)
    historical = json.loads(phase14_path.read_text(encoding="utf-8"))[
        "references"
    ]["training_all"]
    group_comparison = {
        name: _relative_group(training_all["groups"][name], historical["groups"][name])
        for name in ("previous_action", "projected_gravity", "root_state")
    }
    group_comparison["equal_group_composite"] = _relative_group(
        training_all["equal_group_composite"], historical["equal_group_composite"]
    )
    root = query["root_state"]
    first_tilt = next(
        (float(meta[index]["time_after_handoff_s"]) for index in range(len(meta)) if root[index, 1] > 0.30),
        None,
    )
    first_height = next(
        (float(meta[index]["time_after_handoff_s"]) for index in range(len(meta)) if root[index, 0] < 0.45),
        None,
    )
    return {
        "stage": "BASE Phase16 single f005 on-policy recovery suffix capture",
        "analysis_only_after_single_official_capture": True,
        "training_executed": False,
        "causal_claim": False,
        "hypothesis": "A real f005 recovery-authority suffix may expose rollout-consistent previous-action coverage absent from isolated Stage335 reset states.",
        "intervention": "Exactly one frozen official matched-event episode with only the default-off Phase15 sidecar recorder enabled for handoff +0.0..+1.5 s.",
        "control": "Frozen Phase13 action-history-synchronized f005 contract and immutable Stage335; no source/f000 episode was added and no parameter was swept.",
        "provenance": {
            "sidecar_path": str(sidecar_path.resolve()),
            "trace_path": str(trace_path.resolve()),
            "adapter_path": str(adapter_path.resolve()),
            "recovery_model_path": str(recovery_model_path.resolve()),
            "stage335": training_provenance,
            "phase14_report": str(phase14_path),
            "hashes": hashes,
            "hash_checks": hash_checks,
            "sidecar_content_sha256": sidecar["content_sha256"],
        },
        "schema_hash_authority_audit": {
            "sidecar_schema": sidecar["schema"],
            "row_schema": rows[0]["schema"],
            "row_count": len(rows),
            "all_recovery_authority": all(row["authority_slot"] == "recovery" for row in rows),
            "policy_slot_inference_counts": summary["policy_slot_inference_counts"],
            "time_start_s": float(times[0]),
            "time_end_s": float(times[-1]),
            "time_step_max_error_s": float(np.max(np.abs(np.diff(times) - 0.02))),
            "ticks_strictly_consecutive": bool(np.all(np.diff(ticks) == 1)),
            "expected_closed_interval_76_rows": len(rows) == 76,
        },
        "episode_outcome": sidecar["episode_outcome"],
        "physical_summary": {
            "root_z_start_m": float(root[0, 0]),
            "root_z_end_m": float(root[-1, 0]),
            "root_z_min_m": float(root[:, 0].min()),
            "root_tilt_max_rad": float(root[:, 1].max()),
            "signed_root_pitch_min_rad": float(root[:, 2].min()),
            "signed_root_pitch_max_rad": float(root[:, 2].max()),
            "first_tilt_gt_0p30_s": first_tilt,
            "first_root_z_lt_0p45_s": first_height,
            "full_episode_stop_root_z_min_m": summary.get("stop_root_z_min_m"),
            "full_episode_stop_root_tilt_max_rad": summary.get("stop_root_tilt_max_rad"),
        },
        "dedup_inventory": {
            "raw_rows": len(rows),
            "representative_rows": len(representatives),
            "duplicates": len(rows) - len(representatives),
            "representative_snapshot_sha256": [row["snapshot_sha256"] for row in representatives],
            "assignments": assignments,
            "training_fraction": None,
            "training_unlocked": False,
        },
        "stage335_coverage": training_all,
        "relative_to_phase14": group_comparison,
        "conclusion": "The single suffix is schema/hash/authority valid and provides real closed-loop evidence. Its OOD statistics are descriptive for one stochastic episode and do not establish causality or unlock recovery training.",
        "next": "Keep training locked. Review this one inventory against the frozen Phase15 contract before preregistering any paired 5-update curriculum smoke.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
    result_root = root / "results/official_native_strict_20260807"
    parser.add_argument("--sidecar", type=Path, default=result_root / "phase16_f005_recovery_suffix_sidecar.json")
    parser.add_argument("--trace", type=Path, default=result_root / "phase16_f005_recovery_suffix_stiff1p2_fixed_r1.json")
    parser.add_argument("--adapter", type=Path, default=Path(__file__).with_name("stage208_official_mujoco_adapter.py"))
    parser.add_argument("--recovery-model", type=Path, default=root / "models/phase9_stateful_recovery_f005_u5_actor.onnx")
    parser.add_argument("--stage335", type=Path, default=root / "models/stage335_stage306_stiff_fixed_stop_recovery_states.npz")
    parser.add_argument("--stage335-report", type=Path, default=Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"))
    parser.add_argument("--phase14", type=Path, default=Path("reports/baseline/x2_recovery_phase14_ood.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        sidecar_path=args.sidecar,
        trace_path=args.trace,
        adapter_path=args.adapter,
        recovery_model_path=args.recovery_model,
        stage335_path=args.stage335,
        stage335_report_path=args.stage335_report,
        phase14_path=args.phase14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": report["schema_hash_authority_audit"]["row_count"], "representatives": report["dedup_inventory"]["representative_rows"]}, indent=2))


if __name__ == "__main__":
    main()
