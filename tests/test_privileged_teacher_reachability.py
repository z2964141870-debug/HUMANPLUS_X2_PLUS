from __future__ import annotations

import numpy as np

from cwi_x2.privileged_teacher_reachability import (
    PARAMETER_SHAPE,
    reachability_gates,
    support_outside_distance,
    teacher_residual,
)


def test_zero_parameters_are_exact_zero_and_nonzero_is_bounded() -> None:
    gait = np.asarray([0.3, -0.7, 1.0, 0.0])
    zero = teacher_residual(np.zeros(PARAMETER_SHAPE), gait, bound=0.2)
    assert np.array_equal(zero, np.zeros(15, dtype=np.float32))
    nonzero = teacher_residual(np.full(PARAMETER_SHAPE, 2.0), gait, bound=0.2)
    assert np.isfinite(nonzero).all()
    assert np.max(np.abs(nonzero)) <= 0.2


def test_support_distance_handles_single_double_and_flight() -> None:
    feet = np.asarray([[0.0, 0.1], [0.0, -0.1]])
    yaw = np.zeros(2)
    assert support_outside_distance(np.zeros(2), feet, yaw, np.asarray([False, False]), root_yaw=0.0) == 0.0
    assert support_outside_distance(np.asarray([0.0, 0.1]), feet, yaw, np.asarray([True, False]), root_yaw=0.0) == 0.0
    assert support_outside_distance(np.zeros(2), feet, yaw, np.asarray([True, True]), root_yaw=0.0) == 0.0
    assert support_outside_distance(np.asarray([0.4, 0.1]), feet, yaw, np.asarray([True, False]), root_yaw=0.0) > 0.2


def test_reachability_requires_material_pitch_and_all_safety_gates() -> None:
    source = {
        "signed_pitch_rad": {"mean": -0.20, "p05": -0.25},
        "velocity_rmse_mps": 0.10,
        "lateral_rms_mps": 0.04,
        "heading_abs_max_rad": 0.05,
        "support_outside_mean_m": 0.002,
        "stance_slip_p95_mps": 0.10,
        "root_z_min_m": 0.62,
        "root_tilt_max_rad": 0.22,
        "teacher_residual_abs_max": 0.0,
        "survived_full_horizon": True,
    }
    candidate = dict(source)
    candidate["signed_pitch_rad"] = {"mean": -0.18, "p05": -0.235}
    candidate["teacher_residual_abs_max"] = 0.15
    gates = {
        "pitch_mean_delta_rad_min": 0.015,
        "pitch_p05_delta_rad_min": 0.010,
        "velocity_rmse_regression_mps_max": 0.01,
        "lateral_rms_regression_mps_max": 0.015,
        "heading_regression_rad_max": 0.015,
        "support_regression_m_max": 0.001,
        "slip_regression_mps_max": 0.03,
        "root_z_absolute_min_m": 0.55,
        "root_z_regression_m_max": 0.005,
        "tilt_absolute_max_rad": 0.35,
        "tilt_regression_rad_max": 0.01,
        "teacher_residual_abs_max": 0.20,
    }
    assert all(reachability_gates(source, candidate, gates).values())
    candidate["support_outside_mean_m"] = 0.004
    assert not reachability_gates(source, candidate, gates)["support"]
