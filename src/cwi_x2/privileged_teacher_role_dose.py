"""Pure cell summaries and gates for the role-by-dose teacher screen."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [row.get(field) for row in rows]
    if not values or any(value is None for value in values):
        raise ValueError(f"missing field: {field}")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite field: {field}")
    return array


def summarize_cell(rows: list[dict[str, Any]], *, transition_role: bool) -> dict[str, Any]:
    if len(rows) != 16:
        raise ValueError(f"cell needs 16 trajectories, received {len(rows)}")
    terminal_speed = _values(rows, "terminal_speed_mean_mps") if transition_role else np.asarray([0.0])
    terminal_ds = _values(rows, "terminal_double_support_mean") if transition_role else np.asarray([1.0])
    pitch_mean = _values(rows, "signed_pitch_mean_rad")
    pitch_p05 = _values(rows, "signed_pitch_p05_rad")
    return {
        "trajectory_count": len(rows),
        "signed_pitch_mean_rad": float(pitch_mean.mean()),
        "signed_pitch_p05_rad": float(np.quantile(pitch_p05, 0.05)),
        "velocity_rmse_mps": float(math.sqrt(float(_values(rows, "velocity_mse").mean()))),
        "lateral_rms_mps": float(math.sqrt(float(_values(rows, "lateral_mse").mean()))),
        "yaw_rmse_radps": float(math.sqrt(float(_values(rows, "yaw_mse").mean()))),
        "support_outside_mean_m": float(_values(rows, "support_mean_m").mean()),
        "stance_slip_p95_mps": float(np.quantile(_values(rows, "stance_slip_p95_mps"), 0.95)),
        "flight_fraction": float(_values(rows, "flight_fraction").mean()),
        "root_z_min_m": float(_values(rows, "root_height_min_m").min()),
        "root_tilt_max_rad": float(_values(rows, "root_tilt_max_rad").max()),
        "termination_count": int(sum(bool(row["terminated"]) for row in rows)),
        "timeout_count": int(sum(bool(row["time_out"]) for row in rows)),
        "teacher_residual_abs_max": float(_values(rows, "teacher_residual_abs_max").max()),
        "teacher_residual_step_abs_max": float(_values(rows, "teacher_residual_step_abs_max").max()),
        "terminal_speed_p95_mps": float(np.quantile(terminal_speed, 0.95)),
        "terminal_double_support_mean": float(terminal_ds.mean()),
    }


def cell_gates(source: dict[str, Any], candidate: dict[str, Any], limits: dict[str, float], *, transition_role: bool) -> dict[str, bool]:
    gates = {
        "candidate_zero_termination": candidate["termination_count"] == 0,
        "candidate_zero_timeout": candidate["timeout_count"] == 0,
        "pitch_mean": candidate["signed_pitch_mean_rad"] - source["signed_pitch_mean_rad"] >= limits["pitch_mean_delta_rad_min"],
        "pitch_p05": candidate["signed_pitch_p05_rad"] - source["signed_pitch_p05_rad"] >= limits["pitch_p05_delta_rad_min"],
        "velocity": candidate["velocity_rmse_mps"] <= source["velocity_rmse_mps"] + limits["velocity_rmse_regression_mps_max"],
        "lateral": candidate["lateral_rms_mps"] <= source["lateral_rms_mps"] + limits["lateral_rms_regression_mps_max"],
        "yaw": candidate["yaw_rmse_radps"] <= source["yaw_rmse_radps"] + limits["yaw_rmse_regression_radps_max"],
        "support": candidate["support_outside_mean_m"] <= source["support_outside_mean_m"] + limits["support_regression_m_max"],
        "slip": candidate["stance_slip_p95_mps"] <= source["stance_slip_p95_mps"] + limits["slip_regression_mps_max"],
        "flight": candidate["flight_fraction"] <= source["flight_fraction"] + limits["flight_regression_max"],
        "root_height": candidate["root_z_min_m"] >= max(limits["root_z_absolute_min_m"], source["root_z_min_m"] - limits["root_z_regression_m_max"]),
        "tilt": candidate["root_tilt_max_rad"] <= min(limits["tilt_absolute_max_rad"], source["root_tilt_max_rad"] + limits["tilt_regression_rad_max"]),
        "residual_bound": candidate["teacher_residual_abs_max"] <= limits["teacher_residual_abs_max"],
        "residual_slew": candidate["teacher_residual_step_abs_max"] <= limits["teacher_residual_step_abs_max"],
    }
    if transition_role:
        gates["terminal_speed"] = candidate["terminal_speed_p95_mps"] <= limits["terminal_speed_p95_absolute_max_mps"]
        gates["terminal_double_support"] = candidate["terminal_double_support_mean"] >= limits["terminal_double_support_mean_min"]
    return gates


__all__ = ["cell_gates", "summarize_cell"]
