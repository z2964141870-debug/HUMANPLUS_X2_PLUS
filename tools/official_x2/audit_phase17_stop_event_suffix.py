#!/usr/bin/env python3
"""Pure-offline audit of BASE Phase17 outcome-stratified stop-event suffixes.

This tool validates immutable sidecars, reports a conservative quantized
inventory, and reuses Phase14's Stage335 robust/LOO distance contract.  It does
not construct a training set and deliberately does not unlock training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from official_x2 import audit_phase14_recovery_ood as phase14
from official_x2.recovery_suffix_aggregation import sha256_file
from official_x2.validate_phase17_stop_event_episode import validate_pair


PREFIX = "phase17_stage326_stop_event_stateful"
CLASS_ORDER = ("success", "critical", "failure")


def classify_outcome(summary: dict[str, Any]) -> str:
    """Outcome-stratified inventory label; success means the complete gate."""
    if bool(summary.get("full_gate_pass")):
        return "success"
    if bool(summary.get("survived_stop_height_gate")):
        return "critical"
    return "failure"


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


def query_groups(rows: list[dict[str, Any]], episode_ids: list[str]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    observation = np.stack([np.asarray(row["observation_93d"], dtype=np.float64) for row in rows])
    groups = {name: observation[:, source_slice] for name, source_slice in phase14.OBS_GROUPS.items()}
    groups["current_action"] = np.stack(
        [np.asarray(row["actual_issued_action"], dtype=np.float64) for row in rows]
    )
    groups["root_state"] = np.stack([_root_state(row) for row in rows])
    meta = [
        {
            "episode": episode_ids[index],
            "time_after_handoff_s": float(row["stop_elapsed_s"]),
            "root_z_m": float(groups["root_state"][index, 0]),
            "root_tilt_rad": float(groups["root_state"][index, 1]),
        }
        for index, row in enumerate(rows)
    ]
    return groups, meta


_DEDUP_STEPS = np.concatenate(
    [
        np.full(3, 0.02),   # body linear velocity
        np.full(3, 0.02),   # body angular velocity
        np.full(3, 0.005),  # projected gravity
        np.full(3, 0.01),   # command
        np.full(31, 0.01),  # q relative
        np.full(31, 0.05),  # dq
        np.full(15, 0.02),  # previous action
        np.full(4, 0.02),   # gait/contact phase
        np.full(15, 0.02),  # actual issued action
        np.asarray([0.005, 0.02]),  # root z and stop-event time
    ]
)


def stop_event_dedup_key(row: dict[str, Any]) -> str:
    vector = np.concatenate(
        [
            np.asarray(row["observation_93d"], dtype=np.float64),
            np.asarray(row["actual_issued_action"], dtype=np.float64),
            np.asarray(
                [row["physical_state"]["root_position_m"][2], row["stop_elapsed_s"]],
                dtype=np.float64,
            ),
        ]
    )
    if vector.shape != _DEDUP_STEPS.shape or not np.isfinite(vector).all():
        raise ValueError("invalid Phase17 dedup vector")
    quantized = np.rint(vector / _DEDUP_STEPS).astype("<i8", copy=False)
    payload = row["authority_slot"].encode("utf-8") + b"\0" + quantized.tobytes()
    return hashlib.sha256(payload).hexdigest()


def dedup_inventory(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    representatives: dict[str, dict[str, Any]] = {}
    class_sources: dict[str, set[str]] = {}
    for row, label in zip(rows, labels):
        key = stop_event_dedup_key(row)
        representatives.setdefault(key, row)
        class_sources.setdefault(key, set()).add(label)
    return {
        "method": "fixed physical bins over 93D obs + actual issued action + root-z + event-time; authority included",
        "raw_rows": len(rows),
        "representative_rows": len(representatives),
        "duplicates": len(rows) - len(representatives),
        "cross_outcome_bin_collisions": sum(len(values) > 1 for values in class_sources.values()),
        "representative_keys_sha256": hashlib.sha256(
            "\n".join(sorted(representatives)).encode("ascii")
        ).hexdigest(),
        "training_fraction": None,
        "training_unlocked": False,
    }


def _selected_coverage(analysis: dict[str, Any]) -> dict[str, Any]:
    groups = analysis["groups"]
    names = ("previous_action", "projected_gravity", "root_state")
    return {
        "groups": {name: groups[name] for name in names},
        "equal_group_composite": analysis["equal_group_composite"],
    }


def _physical_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = np.stack([_root_state(row) for row in rows])
    times = [float(row["stop_elapsed_s"]) for row in rows]
    first_tilt = next((times[i] for i in range(len(rows)) if root[i, 1] > 0.30), None)
    first_height = next((times[i] for i in range(len(rows)) if root[i, 0] < 0.45), None)
    return {
        "root_z_start_m": float(root[0, 0]),
        "root_z_end_m": float(root[-1, 0]),
        "root_z_min_m": float(root[:, 0].min()),
        "root_tilt_max_rad": float(root[:, 1].max()),
        "signed_root_pitch_min_rad": float(root[:, 2].min()),
        "signed_root_pitch_max_rad": float(root[:, 2].max()),
        "first_root_tilt_gt_0p30_s": first_tilt,
        "first_root_z_lt_0p45_s": first_height,
    }


def build_report(
    *, result_root: Path, stage335_path: Path, stage335_report_path: Path,
    phase14_path: Path, episode_count: int = 4,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_ids: list[str] = []
    all_labels: list[str] = []
    contract = None
    adapter_sha = None
    for index in range(1, episode_count + 1):
        trace_path = result_root / f"{PREFIX}_r{index}.json"
        sidecar_path = result_root / f"{PREFIX}_r{index}_sidecar.json"
        validator_label = validate_pair(trace_path, sidecar_path)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        summary = trace["summary"]
        label = classify_outcome(summary)
        if label != validator_label:
            raise ValueError(f"validator/auditor class mismatch for episode {index}")
        if contract is None:
            contract = sidecar["frozen_control_contract"]
            adapter_sha = sidecar["adapter"]["sha256"]
        if sidecar["frozen_control_contract"] != contract or sidecar["adapter"]["sha256"] != adapter_sha:
            raise ValueError("Phase17 frozen contract differs across episodes")
        rows = sidecar["rows"]
        episode_id = f"r{index}:{label}"
        episodes.append(
            {
                "episode": index,
                "class": label,
                "trace_path": str(trace_path),
                "trace_sha256": sha256_file(trace_path),
                "sidecar_path": str(sidecar_path),
                "sidecar_file_sha256": sha256_file(sidecar_path),
                "sidecar_content_sha256": sidecar["content_sha256"],
                "row_count": len(rows),
                "authority_counts": dict(Counter(row["authority_slot"] for row in rows)),
                "observation_slot_counts": dict(Counter(row["observation_policy_slot"] for row in rows)),
                "outcome": sidecar["episode_outcome"],
                "physical": _physical_summary(rows),
            }
        )
        all_rows.extend(rows)
        all_ids.extend([episode_id] * len(rows))
        all_labels.extend([label] * len(rows))

    training, _, stage335_provenance = phase14.load_stage335(stage335_path, stage335_report_path)
    query, meta = query_groups(all_rows, all_ids)
    coverage_all, _ = phase14.analyze_reference(query, training, meta)
    coverage_by_class: dict[str, Any] = {}
    for label in CLASS_ORDER:
        indices = [i for i, item in enumerate(all_labels) if item == label]
        subset = {name: values[indices] for name, values in query.items()}
        subset_meta = [meta[i] for i in indices]
        analysis, _ = phase14.analyze_reference(subset, training, subset_meta)
        coverage_by_class[label] = _selected_coverage(analysis)

    historical = json.loads(phase14_path.read_text(encoding="utf-8"))["references"]["training_all"]
    relative = {}
    current = _selected_coverage(coverage_all)
    for name in ("previous_action", "projected_gravity", "root_state"):
        relative[name] = {
            "threshold_identical": abs(
                current["groups"][name]["reference_loo_p95_threshold"]
                - historical["groups"][name]["reference_loo_p95_threshold"]
            ) < 1.0e-12,
            "phase17_ood_fraction": current["groups"][name]["query_ood_fraction"],
            "phase14_ood_fraction": historical["groups"][name]["query_ood_fraction"],
            "phase17_first_ood_s_median": current["groups"][name]["first_ood_s_median"],
            "phase14_first_ood_s_median": historical["groups"][name]["first_ood_s_median"],
        }

    counts = Counter(item["class"] for item in episodes)
    return {
        "stage": "BASE Phase17 Stage326 outcome-stratified stop-event stateful suffix inventory",
        "analysis_only_after_frozen_official_capture": True,
        "training_executed": False,
        "training_unlocked": False,
        "causal_claim": False,
        "hypothesis": "A small frozen Stage326 matched-event sample can provide complete success, critical, and height-collapse closed-loop stop suffixes with actual action history.",
        "intervention": "At most five frozen official episodes; stopped at four immediately after obtaining two full successes and one height-collapse failure.",
        "control": "One immutable Stage326 contract and models across episodes; only the default-off sidecar recorder was enabled. No policy, PD, blend, horizon, or seed-selection sweep.",
        "provenance": {
            "result_root": str(result_root),
            "stage335": stage335_provenance,
            "phase14_report": str(phase14_path),
            "adapter_sha256": adapter_sha,
            "frozen_control_contract": contract,
            "classifier_bug": "Initial runner incorrectly labeled stop_gate_pass as complete success and stopped after r3. It was fail-closed corrected to full_gate_pass; r1-r3 were not rerun, and r4 alone was added.",
        },
        "outcome_inventory": {
            "completed_episodes": len(episodes),
            "counts": {name: int(counts[name]) for name in CLASS_ORDER},
            "target_met": counts["success"] >= 2 and counts["failure"] >= 1,
            "r5_not_run_due_early_stop": len(episodes) == 4,
            "episodes": episodes,
        },
        "schema_hash_audit": {
            "sidecar_schema": "aimdk_x2_stop_event_suffix_sidecar_v1",
            "row_schema": "aimdk_x2_stop_event_suffix_snapshot_v1",
            "all_rows_validated": True,
            "total_rows": len(all_rows),
            "rows_per_episode": sorted({item["row_count"] for item in episodes}),
            "window_s": [0.0, 4.0],
            "actual_previous_action_present": True,
            "actual_issued_action_present": True,
            "physical_state_present": True,
            "controller_state_and_hash_present": True,
        },
        "dedup_inventory": dedup_inventory(all_rows, all_labels),
        "stage335_coverage": {
            "semantic_boundary": "Descriptive only: rows use main/stationary actor slots during Stage326 stop, while Stage335 is a recovery-reset support set sharing the 93D schema.",
            "all": current,
            "by_outcome": coverage_by_class,
            "relative_to_phase14": relative,
        },
        "conclusion": "The requested outcome-stratified stateful inventory now exists and is hash/schema complete. This is a data-contract gain, not evidence that suffix training improves recovery.",
        "next": "Keep training locked. Pre-register any fraction=0 versus one fixed small suffix fraction only after defining how success/critical/failure continuations enter the curriculum without treating failed actions as expert labels.",
        "boundaries": [
            "Official AimDK MuJoCo, not real X2 hardware.",
            "No measured GRF/COP/foot wrench truth.",
            "Post-inference/pre-physics snapshots are stateful curriculum evidence, not a proof of closed ROS mid-event restore.",
            "OOD is descriptive correlation and not a failure-cause classifier.",
        ],
    }


def main() -> None:
    root = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=root / "results/official_native_strict_20260807")
    parser.add_argument("--stage335", type=Path, default=root / "models/stage335_stage306_stiff_fixed_stop_recovery_states.npz")
    parser.add_argument("--stage335-report", type=Path, default=Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"))
    parser.add_argument("--phase14", type=Path, default=Path("reports/baseline/x2_recovery_phase14_ood.json"))
    parser.add_argument("--episode-count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        result_root=args.result_root, stage335_path=args.stage335,
        stage335_report_path=args.stage335_report, phase14_path=args.phase14,
        episode_count=args.episode_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "counts": report["outcome_inventory"]["counts"],
        "rows": report["schema_hash_audit"]["total_rows"],
        "representatives": report["dedup_inventory"]["representative_rows"],
        "training_unlocked": False,
    }, indent=2))


if __name__ == "__main__":
    main()
