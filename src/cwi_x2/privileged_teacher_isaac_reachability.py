"""Pure aggregation and gates for the batched Isaac privileged teacher."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def summarize_arrays(records: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1 or not selected.any():
        raise ValueError("mask must select at least one environment")

    def values(name: str) -> np.ndarray:
        array = np.asarray(records[name], dtype=np.float64)[:, selected]
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"non-finite or empty values for {name}")
        return array.reshape(-1)

    pitch = values("pitch")
    velocity_sq = values("velocity_sq")
    lateral_sq = values("lateral_sq")
    yaw_sq = values("yaw_sq")
    support = values("support")
    slip = values("slip")
    root_height = values("root_height")
    tilt = values("tilt")
    residual = np.asarray(records["residual"], dtype=np.float64)[:, selected]
    if not np.isfinite(residual).all():
        raise ValueError("non-finite residual")
    residual_delta = np.diff(residual, axis=0)
    done = np.asarray(records["done"], dtype=bool)[:, selected]
    contact_count = np.asarray(records["contact_count"], dtype=np.int64)[:, selected]
    return {
        "sample_count": int(pitch.size),
        "environment_count": int(selected.sum()),
        "signed_pitch_rad": {
            "mean": float(pitch.mean()),
            "p05": float(np.quantile(pitch, 0.05)),
            "p50": float(np.quantile(pitch, 0.50)),
        },
        "velocity_rmse_mps": float(math.sqrt(float(velocity_sq.mean()))),
        "lateral_rms_mps": float(math.sqrt(float(lateral_sq.mean()))),
        "yaw_rmse_radps": float(math.sqrt(float(yaw_sq.mean()))),
        "support_outside_mean_m": float(support.mean()),
        "support_outside_p95_m": float(np.quantile(support, 0.95)),
        "stance_slip_p95_mps": float(np.quantile(slip, 0.95)),
        "flight_fraction": float(np.mean(contact_count == 0)),
        "root_z_min_m": float(root_height.min()),
        "root_tilt_max_rad": float(tilt.max()),
        "teacher_residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "teacher_residual_abs_max": float(np.max(np.abs(residual))),
        "teacher_residual_step_rms": float(
            np.sqrt(np.mean(np.square(residual_delta)))
        ),
        "teacher_residual_step_abs_max": float(np.max(np.abs(residual_delta))),
        "termination_count": int(done.sum()),
        "survived_full_horizon": not bool(done.any()),
    }


def reachability_cost(summary: dict[str, Any], *, target_pitch_rad: float = -0.05) -> float:
    pitch = summary["signed_pitch_rad"]
    backward_mean = max(0.0, target_pitch_rad - float(pitch["mean"])) / 0.10
    backward_tail = max(0.0, target_pitch_rad - float(pitch["p05"])) / 0.12
    cost = (
        4.0 * backward_mean**2
        + 2.0 * backward_tail**2
        + 2.0 * (float(summary["velocity_rmse_mps"]) / 0.15) ** 2
        + (float(summary["lateral_rms_mps"]) / 0.15) ** 2
        + (float(summary["yaw_rmse_radps"]) / 0.20) ** 2
        + 2.0 * (float(summary["support_outside_mean_m"]) / 0.04) ** 2
        + 0.5 * (float(summary["stance_slip_p95_mps"]) / 0.30) ** 2
        + 0.5 * (float(summary["flight_fraction"]) / 0.10) ** 2
        + 0.10 * (float(summary["teacher_residual_rms"]) / 0.10) ** 2
        + 0.25 * (float(summary["teacher_residual_step_rms"]) / 0.05) ** 2
    )
    cost += 200.0 * max(0.0, 0.60 - float(summary["root_z_min_m"])) ** 2
    cost += 200.0 * max(0.0, float(summary["root_tilt_max_rad"]) - 0.35) ** 2
    if not summary["survived_full_horizon"]:
        cost += 1000.0 + 10.0 * int(summary["termination_count"])
    return float(cost)


def strict_gates(source: dict[str, Any], candidate: dict[str, Any], gates: dict[str, float]) -> dict[str, bool]:
    return {
        "source_survival": bool(source["survived_full_horizon"]),
        "candidate_survival": bool(candidate["survived_full_horizon"]),
        "pitch_mean": candidate["signed_pitch_rad"]["mean"] - source["signed_pitch_rad"]["mean"] >= gates["pitch_mean_delta_rad_min"],
        "pitch_p05": candidate["signed_pitch_rad"]["p05"] - source["signed_pitch_rad"]["p05"] >= gates["pitch_p05_delta_rad_min"],
        "velocity": candidate["velocity_rmse_mps"] <= source["velocity_rmse_mps"] + gates["velocity_rmse_regression_mps_max"],
        "lateral": candidate["lateral_rms_mps"] <= source["lateral_rms_mps"] + gates["lateral_rms_regression_mps_max"],
        "yaw": candidate["yaw_rmse_radps"] <= source["yaw_rmse_radps"] + gates["yaw_rmse_regression_radps_max"],
        "support": candidate["support_outside_mean_m"] <= source["support_outside_mean_m"] + gates["support_regression_m_max"],
        "slip": candidate["stance_slip_p95_mps"] <= source["stance_slip_p95_mps"] + gates["slip_regression_mps_max"],
        "flight": candidate["flight_fraction"] <= source["flight_fraction"] + gates["flight_regression_max"],
        "root_height": candidate["root_z_min_m"] >= max(gates["root_z_absolute_min_m"], source["root_z_min_m"] - gates["root_z_regression_m_max"]),
        "tilt": candidate["root_tilt_max_rad"] <= min(gates["tilt_absolute_max_rad"], source["root_tilt_max_rad"] + gates["tilt_regression_rad_max"]),
        "residual_bound": candidate["teacher_residual_abs_max"] <= gates["teacher_residual_abs_max"],
        "residual_slew": candidate["teacher_residual_step_abs_max"] <= gates["teacher_residual_step_abs_max"],
    }
