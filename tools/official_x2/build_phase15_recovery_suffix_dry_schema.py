#!/usr/bin/env python3
"""Emit the Phase15 dry schema and strict zero-fraction aggregation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_x2.recovery_suffix_aggregation import (
    AGGREGATION_SCHEMA,
    CAPTURE_BOUNDARY,
    SIDECAR_SCHEMA,
    SNAPSHOT_SCHEMA,
    build_aggregation_manifest,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[2]
STAGE335 = (
    REPO.parent
    / "x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
)
STAGE335_SHA = "4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
F005 = (
    REPO.parent
    / "x2_official_rl_deploy_v1/models/phase9_stateful_recovery_f005_u5_actor.onnx"
)


def build() -> dict:
    zero_plan = build_aggregation_manifest(
        base_dataset_path=STAGE335,
        base_dataset_sha256=STAGE335_SHA,
        base_state_count=90,
        # A nonexistent placeholder proves fraction=0 never opens a sidecar.
        suffix_sidecar_paths=["/not-collected/phase15_f005_failed_suffix.json"],
        suffix_fraction=0.0,
    )
    adapter = REPO / "tools/official_x2/stage208_official_mujoco_adapter.py"
    controller_contract = REPO / "tools/official_x2/controller_snapshot_contract.py"
    suffix_contract = REPO / "tools/official_x2/recovery_suffix_aggregation.py"
    return {
        "stage": "BASE Phase15 on-policy recovery suffix aggregation dry schema",
        "dry_schema_only": True,
        "official_collection_executed": False,
        "training_executed": False,
        "collected_suffix_rows": 0,
        "hypothesis": (
            "Phase14 previous-action manifold exit is better targeted by a temporally "
            "consistent on-policy recovery suffix than by more isolated reset states."
        ),
        "capture_contract": {
            "enabled_by_default": False,
            "cli": "--post-handoff-snapshot-output",
            "authority": "recovery only",
            "window_s": [0.0, 1.5],
            "control_clock": "50 Hz step",
            "capture_boundary": CAPTURE_BOUNDARY,
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "required_fields": {
                "physical": "31 q + 31 dq + root xyz/xyzw + world linear/angular velocity",
                "actor": "93D obs + exact observation previous_action + actual issued 15D action",
                "event": "command + gait phase/contact + controller sequence clock + handoff-relative time",
                "controller": "full stateful controller snapshot + physical/controller/snapshot SHA-256",
            },
            "truth_boundary": (
                "post-inference/pre-physics rows require applying the stored issued action for "
                "one physics interval before the next actor tick; they are not mid-event ROS replay proof"
            ),
        },
        "sidecar_contract": {
            "schema": SIDECAR_SCHEMA,
            "one_episode_per_sidecar": True,
            "source_trace_hash_required": True,
            "adapter_and_recovery_model_hash_required": True,
            "episode_outcome_required": True,
            "full_rows_remain_immutable": True,
        },
        "aggregation_contract": {
            "schema": AGGREGATION_SCHEMA,
            "base_stage335_is_immutable": True,
            "materialization": "hash-bound virtual union; no Stage335 NPZ rewrite",
            "dedup": (
                "fixed pre-registered bins over 93D obs, issued action, root z and event time; "
                "root XY/yaw invariant; full source rows retained"
            ),
            "suffix_fraction_zero": (
                "returns before opening suffix sidecars and consumes no suffix-source RNG"
            ),
        },
        "frozen_inputs": {
            "stage335": {
                "path": str(STAGE335),
                "sha256": sha256_file(STAGE335),
                "expected_sha256": STAGE335_SHA,
                "state_count": 90,
            },
            "f005_recovery_model": {
                "path": str(F005),
                "sha256": sha256_file(F005),
                "role": "future single official failed-suffix collection only; not run in Phase15",
            },
            "adapter": {"path": str(adapter), "sha256": sha256_file(adapter)},
            "controller_snapshot_contract": {
                "path": str(controller_contract),
                "sha256": sha256_file(controller_contract),
            },
            "suffix_aggregation_contract": {
                "path": str(suffix_contract),
                "sha256": sha256_file(suffix_contract),
            },
        },
        "fraction_zero_dry_plan": zero_plan,
        "decision": {
            "schema_ready": True,
            "collection_ready_but_locked": True,
            "training_unlocked": False,
            "next_authorized_action": (
                "at most one hash-frozen f005 official failed-suffix capture after backup approval"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema_ready": True}, indent=2))


if __name__ == "__main__":
    main()
