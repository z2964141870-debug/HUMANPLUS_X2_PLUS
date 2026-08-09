from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import retarget.run_x2_native_gold_trackability_phase12 as phase12


REPO = Path(__file__).resolve().parents[1]


def test_contact_helpers_preserve_single_support_semantics():
    contact = {
        "left": np.array([1, 1, 0, 0], dtype=bool),
        "right": np.array([1, 0, 1, 0], dtype=bool),
    }
    phase = phase12.phase_stats(contact)
    assert phase == {
        "left_ratio": 0.5,
        "right_ratio": 0.5,
        "double_support_ratio": 0.25,
        "single_support_ratio": 0.5,
        "flight_ratio": 0.25,
    }
    assert phase12.f1_binary(contact["left"], contact["left"]) == 1.0


def test_phase12_report_stops_before_free_when_prescribed_fails():
    report = json.loads(
        (REPO / "reports/retarget/x2_native_gold_trackability_phase12.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["truth_boundary"]["target_is_recorded_actual_q_not_command"] is True
    assert report["truth_boundary"]["phase11_motionlib_and_state_adapter_reused"] is True
    assert report["truth_boundary"]["source_trace_stability_is_not_replay_stability"] is True
    assert report["truth_boundary"]["free_failure_only_rejects_bare_pd_replay_not_any2any"] is True
    assert report["gate"]["prescribed"]["pass"] is False
    assert report["gate"]["free"] is None
    assert set(report["replay"]) == {phase12.MODE_PRESCRIBED}
    assert report["decision"]["status"] == "PHASE12_PRESCRIBED_TRACKABILITY_REJECTED"
    assert report["replay"][phase12.MODE_PRESCRIBED]["initialization"] == report["initialization"]


def test_phase12_prescribed_failure_is_physical_not_source_continuity():
    report = json.loads(
        (REPO / "reports/retarget/x2_native_gold_trackability_phase12.json").read_text(
            encoding="utf-8"
        )
    )
    checks = report["gate"]["prescribed"]["checks"]
    assert checks["full_duration"] is True
    assert checks["source_joint_step"] is True
    assert checks["source_joint_velocity"] is True
    assert checks["q_trackable"] is True
    assert checks["body_position_trackable"] is False
    assert checks["body_orientation_trackable"] is False
    assert checks["contact_agreement"] is False
    assert checks["slip_bounded"] is False
    replay = report["replay"][phase12.MODE_PRESCRIBED]
    assert replay["contact"]["truth_boundary"].endswith(
        "neither is hardware GRF/COP/wrench"
    )
