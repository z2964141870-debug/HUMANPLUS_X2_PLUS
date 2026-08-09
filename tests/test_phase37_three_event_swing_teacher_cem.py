import numpy as np

from official_x2.run_phase37_three_event_swing_teacher_cem import (
    BOUNDS,
    PARAMETER_NAMES,
    TEACHER_SECONDS,
    ctrl_with_position_residual,
    three_event_residual,
)


def test_zero_three_event_contract_is_exact_noop():
    zero = np.zeros(len(PARAMETER_NAMES))
    for time_s in (0.0, 0.1, 0.2, TEACHER_SECONDS):
        assert all(value == 0.0 for value in three_event_residual(zero, time_s, "right").values())


def test_event_contract_decays_exactly_to_zero_at_horizon():
    raw = BOUNDS.copy()
    residual = three_event_residual(raw, TEACHER_SECONDS, "right")
    assert max(abs(value) for value in residual.values()) <= 1e-12


def test_right_swing_uses_left_support_and_only_five_joint_outputs():
    residual = three_event_residual(0.5 * BOUNDS, 0.15, "right")
    assert set(residual) == {
        "left_hip_roll_joint", "waist_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint",
    }


def test_zero_residual_preserves_out_of_range_recorded_ctrl_exactly():
    baseline = np.asarray([123.0, -456.0])
    result = ctrl_with_position_residual(
        baseline, {"joint": 0.0}, {"joint": 1}, {"joint": (300.0, 20.0)}
    )
    assert np.array_equal(result, baseline)
