from __future__ import annotations

import numpy as np
import pytest

from cwi_x2.privileged_teacher_isaac_reachability import (
    reachability_cost,
    strict_gates,
    summarize_arrays,
)


def records() -> dict[str, np.ndarray]:
    shape = (4, 2)
    return {
        "pitch": np.full(shape, -0.20),
        "velocity_sq": np.full(shape, 0.01),
        "lateral_sq": np.full(shape, 0.0016),
        "yaw_sq": np.full(shape, 0.0025),
        "support": np.full(shape, 0.002),
        "slip": np.full(shape, 0.10),
        "root_height": np.full(shape, 0.65),
        "tilt": np.full(shape, 0.20),
        "residual": np.zeros(shape + (15,)),
        "done": np.zeros(shape, dtype=bool),
        "contact_count": np.full(shape, 1),
    }


def test_summary_and_cost_penalize_backward_pitch() -> None:
    source = summarize_arrays(records(), np.asarray([True, True]))
    better_records = records()
    better_records["pitch"] += 0.03
    better = summarize_arrays(better_records, np.asarray([True, True]))
    assert reachability_cost(better) < reachability_cost(source)


def test_strict_gates_require_safety_and_material_pitch() -> None:
    source = summarize_arrays(records(), np.asarray([True, True]))
    candidate_records = records()
    candidate_records["pitch"] += 0.02
    candidate = summarize_arrays(candidate_records, np.asarray([True, True]))
    gates = {
        "pitch_mean_delta_rad_min": 0.015,
        "pitch_p05_delta_rad_min": 0.010,
        "velocity_rmse_regression_mps_max": 0.010,
        "lateral_rms_regression_mps_max": 0.015,
        "yaw_rmse_regression_radps_max": 0.015,
        "support_regression_m_max": 0.001,
        "slip_regression_mps_max": 0.030,
        "flight_regression_max": 0.010,
        "root_z_absolute_min_m": 0.60,
        "root_z_regression_m_max": 0.005,
        "tilt_absolute_max_rad": 0.35,
        "tilt_regression_rad_max": 0.010,
        "teacher_residual_abs_max": 0.20,
        "teacher_residual_step_abs_max": 0.10,
    }
    assert all(strict_gates(source, candidate, gates).values())
    candidate["termination_count"] = 1
    candidate["survived_full_horizon"] = False
    assert not strict_gates(source, candidate, gates)["candidate_survival"]


def test_summary_rejects_nonfinite_decision_inputs() -> None:
    invalid = records()
    invalid["support"][1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        summarize_arrays(invalid, np.asarray([True, True]))
