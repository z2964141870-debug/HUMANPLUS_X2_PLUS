from __future__ import annotations

import json

import numpy as np

from retarget.run_x2_wbt_contact_schedule_phase9 import (
    REPO,
    canonical_schedule,
    enforce_min_dwell,
    hysteresis_filter,
    labels_from_events,
    phase_statistics,
)


def test_event_reconstruction_is_lossless():
    values = labels_from_events(9, {"touchdown": [0, 6], "liftoff": [3, 8]})
    assert values.tolist() == [True, True, True, False, False, False, True, True, False]


def test_hysteresis_rejects_single_frame_pulse():
    raw = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    filtered = hysteresis_filter(raw)
    assert filtered.tolist() == [False, False, False, False, False, True, True, True]


def test_min_dwell_removes_short_binary_islands():
    raw = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert enforce_min_dwell(raw, 3).all()


def test_schedule_preserves_real_single_support_semantics():
    raw = {
        "left": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
        "right": np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool),
    }
    schedule = canonical_schedule(raw)
    stats = phase_statistics(schedule)
    assert stats["single_support_ratio"] == 3 / 9
    assert stats["ds_ss_ds_cycle_count"] == 1
    assert stats["contact_window_count"] == {"left": 1, "right": 2}


def test_report_keeps_truth_boundary_and_one_pass_contract():
    report = json.loads(
        (REPO / "reports/retarget/x2_wbt_contact_schedule_phase9.json").read_text(encoding="utf-8")
    )
    assert report["truth_boundary"]["not_hardware_grf_cop_wrench_or_foot_force"] is True
    assert report["truth_boundary"]["one_trace_only_free_replay_per_motion"] is True
    assert report["truth_boundary"]["second_physics_pass"] is False
    assert report["truth_boundary"]["stochastic_seed"].startswith("none")
    assert report["decision"]["phase7_aggregate_reproduction_all_five"] is True
