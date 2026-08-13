"""Pure aggregation for the multi-command privileged-teacher panel."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


ROLE_NAMES = {
    0: "vx_0p20_a",
    1: "vx_0p20_b",
    2: "vx_0p35_a",
    3: "vx_0p35_b",
    4: "vx_0p50_a",
    5: "vx_0p50_b",
    6: "turn_vx_0p35_yaw_pm_0p15",
    7: "transition_vx_0p35",
}


def _finite(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"missing or non-finite field: {field}")
    return values


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty group")
    pitch_mean = _finite(rows, "signed_pitch_mean_rad")
    pitch_p05 = _finite(rows, "signed_pitch_p05_rad")
    velocity_mse = _finite(rows, "velocity_mse")
    lateral_mse = _finite(rows, "lateral_mse")
    yaw_mse = _finite(rows, "yaw_mse")
    transition = [row for row in rows if int(row["role_id"]) == 7]
    terminal_speed = (
        _finite(transition, "terminal_speed_mean_mps")
        if transition else np.asarray([0.0], dtype=np.float64)
    )
    terminal_ds = (
        _finite(transition, "terminal_double_support_mean")
        if transition else np.asarray([1.0], dtype=np.float64)
    )
    return {
        "env_count": len(rows),
        "role_counts": {
            ROLE_NAMES[role]: sum(int(row["role_id"]) == role for row in rows)
            for role in ROLE_NAMES
        },
        "signed_pitch_mean_rad": float(pitch_mean.mean()),
        "signed_pitch_p05_rad": float(np.quantile(pitch_p05, 0.05)),
        "velocity_rmse_mps": float(math.sqrt(float(velocity_mse.mean()))),
        "lateral_rms_mps": float(math.sqrt(float(lateral_mse.mean()))),
        "yaw_rmse_radps": float(math.sqrt(float(yaw_mse.mean()))),
        "support_outside_mean_m": float(_finite(rows, "support_mean_m").mean()),
        "stance_slip_p95_mps": float(np.quantile(_finite(rows, "stance_slip_p95_mps"), 0.95)),
        "flight_fraction": float(_finite(rows, "flight_fraction").mean()),
        "root_z_min_m": float(_finite(rows, "root_height_min_m").min()),
        "root_tilt_max_rad": float(_finite(rows, "root_tilt_max_rad").max()),
        "termination_count": int(sum(bool(row["terminated"]) for row in rows)),
        "timeout_count": int(sum(bool(row["time_out"]) for row in rows)),
        "survival_s_min": float(_finite(rows, "survival_s").min()),
        "teacher_residual_abs_max": float(_finite(rows, "teacher_residual_abs_max").max()),
        "teacher_residual_step_abs_max": float(_finite(rows, "teacher_residual_step_abs_max").max()),
        "terminal_speed_mean_mps": float(terminal_speed.mean()),
        "terminal_speed_p95_mps": float(np.quantile(terminal_speed, 0.95)),
        "terminal_double_support_mean": float(terminal_ds.mean()),
    }


def deltas(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    fields = (
        "signed_pitch_mean_rad", "signed_pitch_p05_rad", "velocity_rmse_mps",
        "lateral_rms_mps", "yaw_rmse_radps", "support_outside_mean_m",
        "stance_slip_p95_mps", "flight_fraction", "terminal_speed_mean_mps",
        "terminal_speed_p95_mps", "terminal_double_support_mean",
    )
    result = {field: float(candidate[field] - source[field]) for field in fields}
    result["termination_count"] = float(candidate["termination_count"] - source["termination_count"])
    return result


def panel_gates(source: dict[str, Any], candidate: dict[str, Any], limits: dict[str, float]) -> dict[str, bool]:
    return {
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
        "terminal_speed": candidate["terminal_speed_p95_mps"] <= limits["terminal_speed_p95_absolute_max_mps"],
        "terminal_double_support": candidate["terminal_double_support_mean"] >= limits["terminal_double_support_mean_min"],
    }


def pair_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for report in reports:
        seed = int(report["seed"])
        for row in report["per_env"]:
            key = (seed, int(row["env_id"]))
            indexed.setdefault(key, {})[row["treatment"]] = row
    pairs = []
    for (seed, env_id), treatments in sorted(indexed.items()):
        if set(treatments) != {"source", "candidate"}:
            raise ValueError(f"incomplete crossover pair: {(seed, env_id)}")
        source, candidate = treatments["source"], treatments["candidate"]
        if source["role_id"] != candidate["role_id"]:
            raise ValueError(f"role mismatch: {(seed, env_id)}")
        pairs.append({"seed": seed, "env_id": env_id, "source": source, "candidate": candidate})
    return pairs


__all__ = ["ROLE_NAMES", "deltas", "pair_rows", "panel_gates", "summarize_rows"]
