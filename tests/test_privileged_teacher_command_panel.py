from __future__ import annotations

from cwi_x2.privileged_teacher_command_panel import pair_rows, panel_gates


def row(treatment: str, env_id: int = 0) -> dict:
    return {"treatment": treatment, "env_id": env_id, "role_id": 2}


def test_pair_rows_requires_both_crossover_roles() -> None:
    reports = [
        {"seed": 7, "per_env": [row("source")]},
        {"seed": 7, "per_env": [row("candidate")]},
    ]
    pairs = pair_rows(reports)
    assert len(pairs) == 1
    assert pairs[0]["source"]["treatment"] == "source"


def test_panel_gates_allow_material_safe_candidate() -> None:
    source = {
        "termination_count": 2, "timeout_count": 0,
        "signed_pitch_mean_rad": -0.20, "signed_pitch_p05_rad": -0.24,
        "velocity_rmse_mps": 0.10, "lateral_rms_mps": 0.04,
        "yaw_rmse_radps": 0.10, "support_outside_mean_m": 0.04,
        "stance_slip_p95_mps": 0.20, "flight_fraction": 0.01,
        "root_z_min_m": 0.61, "root_tilt_max_rad": 0.40,
        "teacher_residual_abs_max": 0.0, "teacher_residual_step_abs_max": 0.0,
        "terminal_speed_p95_mps": 0.10, "terminal_double_support_mean": 0.99,
    }
    candidate = dict(source)
    candidate.update({
        "termination_count": 0, "signed_pitch_mean_rad": -0.18,
        "signed_pitch_p05_rad": -0.225, "root_tilt_max_rad": 0.30,
        "teacher_residual_abs_max": 0.09, "teacher_residual_step_abs_max": 0.09,
    })
    limits = {
        "pitch_mean_delta_rad_min": 0.015, "pitch_p05_delta_rad_min": 0.010,
        "velocity_rmse_regression_mps_max": 0.010, "lateral_rms_regression_mps_max": 0.015,
        "yaw_rmse_regression_radps_max": 0.015, "support_regression_m_max": 0.001,
        "slip_regression_mps_max": 0.030, "flight_regression_max": 0.010,
        "root_z_absolute_min_m": 0.60, "root_z_regression_m_max": 0.005,
        "tilt_absolute_max_rad": 0.35, "tilt_regression_rad_max": 0.010,
        "teacher_residual_abs_max": 0.10, "teacher_residual_step_abs_max": 0.10,
        "terminal_speed_p95_absolute_max_mps": 0.12,
        "terminal_double_support_mean_min": 0.98,
    }
    assert all(panel_gates(source, candidate, limits).values())
