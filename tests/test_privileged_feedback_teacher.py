from __future__ import annotations

import torch

from cwi_x2.privileged_feedback_teacher import (
    PARAMETER_COUNT,
    feedback_search_score,
    feedback_scale,
    feedback_validation_gates,
    slew_limited_residual,
)


def _state(count: int) -> dict[str, torch.Tensor]:
    return {
        "pitch_rad": torch.full((count,), -0.18),
        "pitch_rate_radps": torch.zeros(count),
        "support_outside_m": torch.zeros(count),
        "velocity_error_mps": torch.zeros(count),
        "tilt_rad": torch.full((count,), 0.18),
        "contact_count": torch.ones(count, dtype=torch.long),
        "moving": torch.ones(count, dtype=torch.bool),
        "terminal": torch.zeros(count, dtype=torch.bool),
    }


def test_feedback_is_bounded_and_role_conditioned() -> None:
    parameters = torch.zeros((2, PARAMETER_COUNT))
    parameters[0, 0] = -2.0
    parameters[1, 1] = 2.0
    scale = feedback_scale(parameters, torch.tensor([0, 1]), **_state(2))
    assert 0.0 < float(scale[0]) < float(scale[1]) < 0.5


def test_feedback_increases_for_backward_pitch_and_suppresses_risk() -> None:
    parameters = torch.zeros((2, PARAMETER_COUNT))
    parameters[:, 8] = 2.0
    parameters[:, 10:13] = 2.0
    state = _state(2)
    state["pitch_rad"] = torch.tensor([-0.24, -0.16])
    state["support_outside_m"] = torch.tensor([0.0, 0.08])
    state["velocity_error_mps"] = torch.tensor([0.0, 0.30])
    state["tilt_rad"] = torch.tensor([0.18, 0.40])
    scale = feedback_scale(parameters, torch.zeros(2, dtype=torch.long), **state)
    assert float(scale[0]) > float(scale[1])


def test_feedback_fails_closed_for_terminal_flight_and_nonfinite() -> None:
    parameters = torch.zeros((4, PARAMETER_COUNT))
    state = _state(4)
    state["terminal"][0] = True
    state["moving"][1] = False
    state["contact_count"][2] = 0
    state["pitch_rad"][3] = torch.nan
    scale = feedback_scale(parameters, torch.zeros(4, dtype=torch.long), **state)
    assert torch.equal(scale, torch.zeros_like(scale))


def test_slew_limiter_respects_amplitude_and_step() -> None:
    desired = torch.tensor([[0.20, -0.20, 0.005]])
    previous = torch.tensor([[0.09, -0.09, 0.0]])
    actual = slew_limited_residual(desired, previous, maximum_abs=0.10, maximum_step=0.01)
    torch.testing.assert_close(actual, torch.tensor([[0.10, -0.10, 0.005]]))


def test_feedback_shape_contract_is_fail_closed() -> None:
    try:
        feedback_scale(
            torch.zeros((1, PARAMETER_COUNT - 1)),
            torch.zeros(1, dtype=torch.long),
            **_state(1),
        )
    except ValueError as error:
        assert "raw_parameters" in str(error)
    else:
        raise AssertionError("invalid parameter width must fail")


def _search_row(role_id: int, **updates: float | bool | None) -> dict[str, float | bool | None]:
    row: dict[str, float | bool | None] = {
        "role_id": role_id,
        "signed_pitch_mean_rad": -0.18,
        "signed_pitch_p05_rad": -0.22,
        "velocity_mse": 0.04,
        "lateral_mse": 0.01,
        "yaw_mse": 0.01,
        "support_mean_m": 0.01,
        "stance_slip_p95_mps": 0.10,
        "flight_fraction": 0.0,
        "root_height_min_m": 0.65,
        "root_tilt_max_rad": 0.20,
        "terminated": False,
        "time_out": False,
        "teacher_residual_abs_max": 0.05,
        "teacher_residual_step_abs_max": 0.01,
        "terminal_speed_mean_mps": 0.04,
        "terminal_double_support_mean": 1.0,
    }
    row.update(updates)
    return row


def _limits() -> dict[str, float]:
    return {
        "pitch_mean_delta_rad_min": 0.005,
        "pitch_p05_delta_rad_min": 0.0,
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
        "teacher_residual_abs_max": 0.10,
        "teacher_residual_step_abs_max": 0.10,
        "terminal_speed_p95_absolute_max_mps": 0.12,
        "terminal_double_support_mean_min": 0.98,
    }


def test_search_score_prefers_safe_pitch_improvement() -> None:
    source = [_search_row(role) for role in range(8) for _ in range(2)]
    safe = [
        _search_row(role, signed_pitch_mean_rad=-0.17, signed_pitch_p05_rad=-0.215)
        for role in range(8) for _ in range(2)
    ]
    unsafe = [
        _search_row(role, signed_pitch_mean_rad=-0.17, signed_pitch_p05_rad=-0.215, terminated=True)
        for role in range(8) for _ in range(2)
    ]
    safe_cost, safe_roles = feedback_search_score(source, safe, _limits())
    unsafe_cost, _ = feedback_search_score(source, unsafe, _limits())
    assert safe_cost < unsafe_cost
    assert all(all(item["gates"].values()) for item in safe_roles.values())


def test_search_score_penalizes_missing_transition_terminal() -> None:
    source = [_search_row(role) for role in range(8) for _ in range(2)]
    candidate = [_search_row(role, signed_pitch_mean_rad=-0.17) for role in range(8) for _ in range(2)]
    for row in candidate:
        if row["role_id"] == 7:
            row["terminal_speed_mean_mps"] = None
    cost, roles = feedback_search_score(source, candidate, _limits())
    assert cost >= 1000.0
    assert "missing" in roles["7"]["error"]


def test_validation_requires_every_role_and_candidate_terminal_data() -> None:
    source = [_search_row(role) for role in range(8) for _ in range(4)]
    candidate = [
        _search_row(role, signed_pitch_mean_rad=-0.17, signed_pitch_p05_rad=-0.215)
        for role in range(8) for _ in range(4)
    ]
    passed, _ = feedback_validation_gates(source, candidate, _limits(), expected_per_role=4)
    assert passed
    candidate[-1]["terminal_speed_mean_mps"] = None
    passed, roles = feedback_validation_gates(source, candidate, _limits(), expected_per_role=4)
    assert not passed
    assert "missing" in roles["7"]["error"]
