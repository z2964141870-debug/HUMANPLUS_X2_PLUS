#!/usr/bin/env python3
"""Read-only inventory of existing BASE stop/recovery suffix evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from official_x2.recovery_suffix_aggregation import load_suffix_sidecar, sha256_file


FAMILIES = {
    "Stage250 nominal videos": "stage250_video_*.json",
    "Stage306 matched source (Stage350 gate)": "stage350_source_stage306_matched_event_stiff1p2_fixed_r*.json",
    "Stage326 stiff-fixed stop": "stage326_s2652_preview0p5_split_intent_brake_stiff1p2_fixed_r*.json",
    "Phase11 recovery role": "phase11_*recovery_matched_stiff1p2_fixed_r*.json",
    "Phase13 handoff continuity": "phase13_source_recovery_blend*_stiff1p2_fixed_r*.json",
    "Phase16 f005 capture trace": "phase16_f005_recovery_suffix_stiff1p2_fixed_r1.json",
}


def episode_class(summary: dict[str, Any]) -> str:
    """Classify whole-stop outcome without inventing a near-miss threshold."""
    if bool(summary.get("stop_gate_pass")):
        return "success"
    if bool(summary.get("survived_stop_height_gate")):
        return "critical_nonfall_gate_failure"
    return "failure_with_height_collapse"


def _all_shape(rows: list[dict[str, Any]], key: str, size: int) -> bool:
    return bool(rows) and all(len(row.get(key, ())) == size for row in rows)


def trace_record(path: Path, family: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    trace = payload["trace"]
    stop = [row for row in trace if row.get("stage") == "stop"]
    has_controller = any(
        "controller_state" in row or "controller_state_sha256" in row for row in stop
    )
    phase13_or_later = family in {
        "Phase13 handoff continuity", "Phase16 f005 capture trace"
    }
    return {
        "family": family,
        "path": str(path),
        "sha256": sha256_file(path),
        "trace_rows": len(trace),
        "stop_rows": len(stop),
        "episode_class": episode_class(summary),
        "gates": {
            key: bool(summary.get(key))
            for key in (
                "stand_gate_pass", "startup_gate_pass", "move_gate_pass",
                "stop_gate_pass", "full_gate_pass",
            )
        },
        "stop_metrics": {
            key: summary.get(key)
            for key in (
                "stop_root_z_min_m", "stop_root_tilt_max_rad",
                "stop_root_xy_drift_m", "stop_settle_time_s",
            )
        },
        "contract": {
            key: summary.get(key)
            for key in (
                "model", "stationary_model", "recovery_model", "clock_mode",
                "pd_profile", "pd_kp_multiplier", "pd_kd_multiplier",
                "stop_controller", "stop_transition_seconds",
                "curriculum_recovery_handoff_blend_seconds",
                "policy_slot_inference_counts",
            )
        },
        "evidence_fields": {
            "row_obs_93d": _all_shape(stop, "obs", 93),
            "obs_previous_action_slice": _all_shape(stop, "obs", 93),
            "row_action_15d": _all_shape(stop, "action", 15),
            "physical_lower_target_15d": _all_shape(stop, "physical_lower_target_rad", 15),
            "row_action_to_physical_target_history_sync_verified_by_phase13": phase13_or_later,
            "full_physical_q_dq_root_snapshot": False,
            "full_controller_snapshot": has_controller,
            "hash_bound_actual_previous_and_issued_action": False,
        },
        "use_boundary": (
            "observational closed-loop trace only; row action is not a separately "
            "hash-bound actual-issued/controller snapshot contract"
        ),
    }


def stage335_record(dataset: Path, source_report: Path) -> dict[str, Any]:
    archive = np.load(dataset, allow_pickle=False)
    report = json.loads(source_report.read_text(encoding="utf-8"))
    digest = sha256_file(dataset)
    if digest != report["dataset_sha256"]:
        raise ValueError("Stage335 dataset/source-report SHA mismatch")
    labels = archive["eventual_pass"].astype(bool)
    return {
        "path": str(dataset),
        "sha256": digest,
        "source_report": str(source_report),
        "source_episode_count": len(report["source_files"]),
        "state_count": int(len(labels)),
        "eventual_pass_states": int(labels.sum()),
        "eventual_fail_states": int((~labels).sum()),
        "npz_keys": list(archive.files),
        "contract": {
            "kind": "discrete reset states sampled at 0.2 s; not closed-loop suffixes",
            "93d_rejoin_requires_immutable_source_traces": True,
            "previous_action_slice_present": "previous_action" in archive.files,
            "current_issued_action_present": False,
            "controller_snapshot_present": False,
            "eventual_label_is_episode_continuation_not_state_oracle": True,
        },
        "overlap_warning": "All 90 states come from the same five Stage326 episodes and are not additional independent episodes.",
    }


def _tilt_from_snapshot(row: dict[str, Any]) -> float:
    x, y, z, w = (float(v) for v in row["physical_state"]["root_quaternion_xyzw"])
    gravity_z = 1.0 - 2.0 * (w * w + z * z)
    return math.acos(float(np.clip(-gravity_z, -1.0, 1.0)))


def phase16_sidecar_record(path: Path) -> dict[str, Any]:
    file_sha = sha256_file(path)
    sidecar = load_suffix_sidecar(path, file_sha)
    rows = sidecar["rows"]
    z = np.asarray([row["physical_state"]["root_position_m"][2] for row in rows])
    tilt = np.asarray([_tilt_from_snapshot(row) for row in rows])
    height_collapse = bool(np.any(z < 0.45))
    tilt_violation = bool(np.any(tilt > 0.30))
    if height_collapse:
        window_class = "failure_window_contains_height_collapse"
    elif tilt_violation:
        window_class = "precollapse_critical_window"
    else:
        window_class = "healthy_window_with_eventual_outcome_label"
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "content_sha256": sidecar["content_sha256"],
        "row_count": len(rows),
        "episode_outcome": sidecar["episode_outcome"],
        "window_class": window_class,
        "time_range_s": [rows[0]["time_after_handoff_s"], rows[-1]["time_after_handoff_s"]],
        "root_z_min_m": float(z.min()),
        "root_tilt_max_rad": float(tilt.max()),
        "all_recovery_authority": all(row["authority_slot"] == "recovery" for row in rows),
        "evidence_fields": {
            "full_physical_q_dq_root_snapshot": True,
            "observation_93d": True,
            "actual_previous_action": True,
            "actual_issued_action": True,
            "full_controller_snapshot": True,
            "row_physical_controller_snapshot_hashes": True,
        },
        "truth_boundary": sidecar["truth_boundary"],
    }


def build_inventory(
    result_root: Path,
    dataset: Path,
    stage335_report: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    family_counts: dict[str, Any] = {}
    for family, pattern in FAMILIES.items():
        family_records = [trace_record(path, family) for path in sorted(result_root.glob(pattern))]
        records.extend(family_records)
        family_counts[family] = {
            "episodes": len(family_records),
            "episode_classes": dict(Counter(row["episode_class"] for row in family_records)),
            "full_controller_snapshot_episodes": sum(
                row["evidence_fields"]["full_controller_snapshot"] for row in family_records
            ),
        }
    stage335 = stage335_record(dataset, stage335_report)
    phase16 = phase16_sidecar_record(sidecar_path)
    episode_counts = Counter(row["episode_class"] for row in records)
    return {
        "stage": "BASE stop/recovery suffix inventory after Phase16",
        "read_only_audit": True,
        "official_run_executed": False,
        "training_executed": False,
        "classification_contract": {
            "episode_success": "stop_gate_pass=true",
            "episode_critical": "stop_gate_pass=false but survived_stop_height_gate=true; no new distance threshold invented",
            "episode_failure": "survived_stop_height_gate=false",
            "window_class_is_separate": "a 0-1.5 s prefix can be pre-collapse critical even when its complete episode eventually fails",
        },
        "trace_inventory": records,
        "family_counts": family_counts,
        "episode_class_counts": dict(episode_counts),
        "stage335_discrete_reset_inventory": stage335,
        "phase16_stateful_sidecar": phase16,
        "three_class_assessment": {
            "observational_episode_success_exists": episode_counts["success"] > 0,
            "observational_episode_critical_exists": episode_counts["critical_nonfall_gate_failure"] > 0,
            "observational_episode_failure_exists": episode_counts["failure_with_height_collapse"] > 0,
            "stateful_success_suffix_exists": False,
            "stateful_critical_suffix_exists": phase16["window_class"] == "precollapse_critical_window",
            "stateful_collapse_suffix_exists": phase16["window_class"] == "failure_window_contains_height_collapse",
            "all_three_stateful_classes_under_one_frozen_contract": False,
        },
        "minimum_gap": {
            "missing": [
                "at least one successful stop suffix with full physical+controller snapshot and hash-bound actual previous/issued action",
                "at least one collapse-containing failure suffix under the same frozen contract",
                "a uniform event anchor spanning brake, handoff, and recovery; Phase16 begins only at recovery authority",
                "an independent episode count sufficient to avoid treating 76 correlated rows as 76 trials",
            ],
            "not_missing": [
                "observational successful and failed Stage326 trajectories",
                "90 discrete Stage335 reset states with eventual pass/fail labels",
                "one valid Phase16 stateful pre-collapse critical suffix with eventual failure",
            ],
        },
        "five_update_preregistration": {
            "ready": False,
            "reason": "Only one state-complete suffix exists; it is one eventual-failure episode and contains no successful or collapse-window peer under the same contract. A fixed nonzero fraction would confound suffix semantics with one correlated trajectory.",
            "training_remains_locked": True,
        },
        "single_next_collection_experiment": {
            "name": "outcome-stratified stop-event stateful suffix capture",
            "target_classes": ["successful continuation", "pre-collapse critical prefix", "collapse-containing failure"],
            "episode_budget": "maximum 5 episodes; stop early only after >=2 success and >=1 failure complete episodes are captured; do not retry beyond 5",
            "frozen_contract": (
                "the Stage326 historically mixed-outcome contract: moving Stage306, stationary source stand, "
                "stop_controller=brake_blend_to_policy with its existing post-latch stationary authority, stiff1.2 fixed upper, "
                "unchanged PD/command/50 Hz clock for every episode; recorder is the only control-path-neutral change"
            ),
            "capture_window": "event-aligned from stop command start through min(4.0 s, height collapse), not only post-handoff; every row uses Phase16 physical/controller/action hashes",
            "stopping_gates": [
                "any model/adapter/contract hash mismatch",
                "any row missing actual previous/issued action or full controller snapshot",
                "authority/event timing mismatch",
                "five-episode budget exhausted",
            ],
            "training_after_collection": "still locked pending offline class balance, dedup, and reset-integration audit",
        },
        "conclusion": "Successful and failed stop episodes exist observationally, and Phase16 adds one stateful pre-collapse critical suffix with an eventual-failure label. The three outcome classes do not exist as replay-complete suffixes under one frozen contract, so fraction=0 versus fixed-small-fraction 5-update training cannot yet be preregistered honestly.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    official = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
    result_root = official / "results/official_native_strict_20260807"
    parser.add_argument("--result-root", type=Path, default=result_root)
    parser.add_argument("--dataset", type=Path, default=official / "models/stage335_stage306_stiff_fixed_stop_recovery_states.npz")
    parser.add_argument("--stage335-report", type=Path, default=Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"))
    parser.add_argument("--sidecar", type=Path, default=result_root / "phase16_f005_recovery_suffix_sidecar.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_inventory(args.result_root, args.dataset, args.stage335_report, args.sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "episodes": len(report["trace_inventory"]), "classes": report["episode_class_counts"], "five_update_ready": report["five_update_preregistration"]["ready"]}, indent=2))


if __name__ == "__main__":
    main()
