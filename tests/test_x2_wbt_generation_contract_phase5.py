from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retarget.run_x2_wbt_generation_contract_phase5 import (
    STATIC_PREFIX_FRAMES,
    WARM_SOURCE_HORIZON_FRAMES,
    continuity_gate,
    make_reset_compatible,
    warm_start_source_map,
)


def synthetic_entry(frames: int = 80):
    t = np.arange(frames, dtype=np.float64) / 30.0
    return {
        "fps": 30,
        "dof": np.column_stack([t, 2 * t]).astype(np.float32),
        "root_trans_offset": np.column_stack([t, np.zeros(frames), np.ones(frames)]).astype(np.float32),
        "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)).astype(np.float32),
        "source_segment_duration_s": (frames - 1) / 30.0,
    }


def test_warm_start_map_is_static_then_monotone_and_velocity_matched():
    values = warm_start_source_map(80)
    assert np.all(values[:STATIC_PREFIX_FRAMES] == 0.0)
    assert np.all(np.diff(values) >= 0.0)
    join = STATIC_PREFIX_FRAMES + WARM_SOURCE_HORIZON_FRAMES - 1
    assert abs((values[join + 1] - values[join]) - 1.0) < 1e-12
    assert values[-1] == 79


def test_reset_compatible_has_zero_initial_velocity_and_continuous_join():
    repaired, metrics = make_reset_compatible(synthetic_entry())
    assert metrics["extra_frames"] == STATIC_PREFIX_FRAMES - 1
    assert metrics["frame0_joint_velocity_max_radps"] == 0.0
    assert metrics["frame0_root_linear_norm_mps"] == 0.0
    assert metrics["join_joint_velocity_jump_p95_radps"] < 0.2
    assert continuity_gate(metrics)["pass"] is True
    assert repaired["phase5_reset_compatible"]["raw_frame0_qvel_injected"] is False


def test_continuity_gate_rejects_raw_velocity_spike():
    _, metrics = make_reset_compatible(synthetic_entry())
    metrics["frame0_root_angular_norm_radps"] = 10.0
    assert continuity_gate(metrics)["pass"] is False


def test_generated_report_preserves_truth_boundary_and_preregistered_failure():
    report_path = Path(__file__).resolve().parents[1] / (
        "reports/retarget/x2_wbt_generation_contract_phase5.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == "x2_wbt_generation_contract_phase5_v1"
    assert report["preregistered_contract"]["raw_frame0_qvel_injected"] is False
    assert report["truth_boundary"]["training_or_teacher_search"] is False
    assert report["truth_boundary"]["raw_frame0_qvel_used"] is False
    assert "not GRF" in report["truth_boundary"]["official_contact"]
    assert set(report["per_role"]) == {
        "walk_straight",
        "turn_left",
        "turn_right",
        "stand_to_walk",
        "walk_to_stand",
    }

    # This is a preregistered negative result.  Protect it from being silently
    # re-labelled as a successful generator or as a dynamics impossibility.
    decision = report["decision"]
    assert decision["promote_generation_contract"] is False
    assert decision["status"] == "PHASE5_GENERATION_CONTRACT_NOT_PROMOTABLE"
    assert decision["checks"]["reset_continuity_all_actions"] is False
    assert decision["checks"]["ground_offline_all_actions"] is False
    assert decision["checks"]["ground_no_free_survival_regression"] is False
    assert "只否定当前最小生成器修复" in decision["conclusion"]
