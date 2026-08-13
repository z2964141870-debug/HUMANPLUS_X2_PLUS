from __future__ import annotations

import numpy as np

from cwi_x2.privileged_teacher_role_dose import cell_gates, summarize_cell


def test_role_scale_assignment_is_complete() -> None:
    pair_ids = np.arange(64)
    roles = pair_ids % 8
    blocks = pair_ids // 8
    observed = {(role, scale): [] for role in range(8) for scale in range(8)}
    for pass_index in range(8):
        scales = (blocks + pass_index) % 8
        for pair_id, (role, scale) in enumerate(zip(roles, scales, strict=True)):
            observed[(int(role), int(scale))].extend((2 * pair_id, 2 * pair_id + 1))
    assert all(len(env_ids) == 16 for env_ids in observed.values())
    for role in range(8):
        for env_id in np.flatnonzero(np.repeat(roles, 2) == role):
            assert sum(env_id in observed[(role, scale)] for scale in range(8)) == 8


def test_summary_rejects_missing_terminal_samples() -> None:
    rows = [{"terminal_speed_mean_mps": None} for _ in range(16)]
    try:
        summarize_cell(rows, transition_role=True)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing terminal data must fail")


def _row(**updates: float | bool) -> dict[str, float | bool]:
    row: dict[str, float | bool] = {
        "signed_pitch_mean_rad": -0.18,
        "signed_pitch_p05_rad": -0.22,
        "support_mean_m": 0.01,
        "stance_slip_p95_mps": 0.10,
        "velocity_mse": 0.04,
        "lateral_mse": 0.01,
        "yaw_mse": 0.01,
        "flight_fraction": 0.01,
        "terminal_speed_mean_mps": 0.04,
        "terminal_double_support_mean": 1.0,
        "survival_s": 10.24,
        "terminated": False,
        "time_out": False,
        "root_height_min_m": 0.64,
        "root_tilt_max_rad": 0.20,
        "teacher_residual_abs_max": 0.0,
        "teacher_residual_step_abs_max": 0.0,
    }
    row.update(updates)
    return row


def test_cell_gates_accept_safe_material_candidate() -> None:
    source = summarize_cell([_row() for _ in range(16)], transition_role=True)
    candidate = summarize_cell([
        _row(
            signed_pitch_mean_rad=-0.17,
            signed_pitch_p05_rad=-0.215,
            teacher_residual_abs_max=0.08,
            teacher_residual_step_abs_max=0.08,
        )
        for _ in range(16)
    ], transition_role=True)
    limits = {
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
    assert all(cell_gates(source, candidate, limits, transition_role=True).values())


def test_runner_transform_accepts_both_scientific_decisions() -> None:
    text = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/run_x2_privileged_teacher_role_dose_v2e.py"
    ).read_text(encoding="utf-8")
    assert "PASS_ROLE_DOSE_WINDOWS_LOCAL_ONLY" in text
    assert "FAIL_NO_COMPLETE_ROLE_DOSE_WINDOWS_STOP" in text
    assert "PANEL_LAUNCH_FINITE\\\" else 1" in text
