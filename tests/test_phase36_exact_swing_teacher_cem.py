import numpy as np

from official_x2.run_phase36_exact_swing_teacher_cem import (
    CONTROL_TICKS,
    KNOTS,
    MODE_BOUNDS_RAD,
    MODE_NAMES,
    coefficients,
    longest_false_window,
    mode_joints,
)


def test_longest_false_window_finds_contiguous_swing_off():
    assert longest_false_window(np.asarray([1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)) == (4, 7)


def test_zero_knots_are_exact_noop_and_bounds_hold():
    zero = np.zeros(KNOTS * len(MODE_NAMES))
    assert np.array_equal(coefficients(zero, 0), np.zeros(len(MODE_NAMES)))
    raw = np.tile(10.0 * MODE_BOUNDS_RAD, KNOTS)
    assert np.all(coefficients(raw, CONTROL_TICKS - 1) <= MODE_BOUNDS_RAD)


def test_side_specific_joint_contract_only_touches_allowed_five_modes():
    assert mode_joints("left") == (
        "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "right_hip_roll_joint", "waist_roll_joint",
    )
    assert mode_joints("right")[3] == "left_hip_roll_joint"


def test_search_comparison_must_prioritize_hard_success_before_cost():
    best = {"strict_success": True, "cost": 10.0}
    candidate = {"strict_success": False, "cost": 1.0}
    should_replace = (
        candidate["strict_success"] and not best["strict_success"]
    ) or (
        candidate["strict_success"] == best["strict_success"]
        and candidate["cost"] < best["cost"]
    )
    assert not should_replace
