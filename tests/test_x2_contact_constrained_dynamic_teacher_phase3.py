from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from retarget.run_x2_contact_constrained_dynamic_teacher_phase3 import (
    HIGH,
    LOW,
    TeacherVector,
    boolean_intervals,
    decode_vector,
    event_schedule,
    first_swing_event,
    spline_values,
)
from retarget.run_x2_contact_constrained_dynamic_teacher_phase3_panel import decision_for


def synthetic_phase(frames: int = 100):
    left = np.zeros(frames, dtype=bool)
    right = np.zeros(frames, dtype=bool)
    right[30:50] = True
    return {"left_swing": left, "right_swing": right, "double_support": ~(left | right)}


def test_decode_vector_clips_and_rounds_event_frames():
    value = decode_vector(np.r_[np.ones(7) * 10, 4.6, -3.6, 2.0, -2.0, 2.0])
    assert value.com_lateral_knots_m == (0.08,) * 4
    assert value.root_height_knots_m == (0.05,) * 3
    assert value.liftoff_shift_frames == 5
    assert value.touchdown_shift_frames == -4
    assert value.swing_clearance_m == 0.12
    assert value.landing_dx_m == -0.08
    assert value.landing_dy_m == 0.08


def test_first_swing_event_ignores_one_frame_noise():
    phase = synthetic_phase()
    phase["left_swing"][1] = True
    assert first_swing_event(phase) == ("right", 30, 49)


def test_event_schedule_preserves_single_support_and_minimum_duration():
    vector = TeacherVector((0, 0, 0, 0), (0, 0, 0), -12, -12, 0.05, 0, 0)
    schedule, event = event_schedule(synthetic_phase(), vector)
    assert event["optimized_touchdown_frame"] - event["optimized_liftoff_frame"] >= 8
    assert not np.any(schedule["left_swing"] & schedule["right_swing"])
    assert np.array_equal(schedule["left_contact"], ~schedule["left_swing"])


def test_spline_values_is_windowed_and_decays_to_zero():
    values = spline_values((0.0, 0.02, -0.01, 0.03), 100, 50)
    assert len(values) == 100
    assert abs(values[-1]) < 1e-12
    assert np.max(np.abs(values)) <= 0.04


def test_parameter_contract_has_twelve_bounded_dimensions():
    assert LOW.shape == HIGH.shape == (12,)
    assert np.all(HIGH > LOW)
    assert boolean_intervals(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 2), (4, 4)]


def test_panel_decision_requires_all_three_gates():
    baseline = {"simulated_duration_s": 1.0, "stance_slip_p95_mps": 0.01}
    free = {"simulated_duration_s": 1.6, "stance_slip_p95_mps": 0.015}
    prescribed = {"action_target_delta_p95_rad": 0.1, "torque_saturation_fraction": 0.01}
    assert decision_for(baseline, free, prescribed)["pass"] is True
    free["stance_slip_p95_mps"] = 0.03
    assert decision_for(baseline, free, prescribed)["pass"] is False


def test_generated_panel_report_is_honest_about_evidence_boundary():
    path = Path(__file__).resolve().parents[1] / (
        "reports/retarget/x2_contact_constrained_dynamic_teacher_phase3.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["truth_boundary"]["oracle_deployable"] is False
    assert report["truth_boundary"]["training_or_checkpoint_load"] is False
    assert report["decision"]["pass_count"] == 1
    assert report["decision"]["robust_dynamic_teacher_exists"] is False
    assert set(report["per_role"]) == {
        "walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand"
    }
    audit = report["methodology_audit"]
    assert "no independent held-out" in audit["walk_evidence_boundary"]
    assert "qvel is forced to zero" in audit["initial_state_contract"]
