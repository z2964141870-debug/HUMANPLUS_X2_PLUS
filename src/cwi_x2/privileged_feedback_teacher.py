"""Bounded state-feedback governor for a privileged coordinated teacher.

The governor does not learn or replace the Stage219 actor.  It modulates an
already-frozen coordinated 15-D direction from privileged physical state and
hard-disables intervention in terminal, standing, flight, or invalid states.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROLE_COUNT = 8
PARAMETER_COUNT = 13


def feedback_scale(
    raw_parameters: torch.Tensor,
    role_ids: torch.Tensor,
    *,
    pitch_rad: torch.Tensor,
    pitch_rate_radps: torch.Tensor,
    support_outside_m: torch.Tensor,
    velocity_error_mps: torch.Tensor,
    tilt_rad: torch.Tensor,
    contact_count: torch.Tensor,
    moving: torch.Tensor,
    terminal: torch.Tensor,
    maximum_scale: float = 0.5,
) -> torch.Tensor:
    """Return one bounded intervention scale per environment.

    Parameters 0..7 are role logits.  Parameter 8 increases intervention when
    root pitch is farther backward than -0.15 rad; parameter 9 is a signed
    pitch-rate term.  Parameters 10..12 can only suppress intervention as
    support, velocity error, or tilt worsen.
    """

    if raw_parameters.ndim != 2 or raw_parameters.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"raw_parameters must be [N,{PARAMETER_COUNT}]")
    count = raw_parameters.shape[0]
    tensors = (
        role_ids,
        pitch_rad,
        pitch_rate_radps,
        support_outside_m,
        velocity_error_mps,
        tilt_rad,
        contact_count,
        moving,
        terminal,
    )
    if any(value.shape != (count,) for value in tensors):
        raise ValueError("all feedback state tensors must be [N]")
    if not (0.0 < float(maximum_scale) <= 1.0):
        raise ValueError("maximum_scale must be in (0,1]")
    if bool(((role_ids < 0) | (role_ids >= ROLE_COUNT)).any()):
        raise ValueError("role id outside deployable panel")

    finite = torch.isfinite(raw_parameters).all(dim=-1)
    for value in (
        pitch_rad,
        pitch_rate_radps,
        support_outside_m,
        velocity_error_mps,
        tilt_rad,
    ):
        finite &= torch.isfinite(value)

    role_logit = raw_parameters.gather(1, role_ids.long().unsqueeze(-1)).squeeze(-1)
    pitch_gain = F.softplus(raw_parameters[:, 8])
    pitch_rate_gain = raw_parameters[:, 9].clamp(-4.0, 4.0)
    support_gain = F.softplus(raw_parameters[:, 10])
    velocity_gain = F.softplus(raw_parameters[:, 11])
    tilt_gain = F.softplus(raw_parameters[:, 12])

    backward = ((-0.15 - pitch_rad) / 0.05).clamp(-2.0, 2.0)
    backward_rate = (-pitch_rate_radps / 0.50).clamp(-2.0, 2.0)
    support_risk = (support_outside_m / 0.04).clamp(0.0, 2.0)
    velocity_risk = (velocity_error_mps / 0.15).clamp(0.0, 2.0)
    tilt_risk = ((tilt_rad - 0.20) / 0.10).clamp(0.0, 2.0)
    logit = (
        role_logit
        + pitch_gain * backward
        + pitch_rate_gain * backward_rate
        - support_gain * support_risk
        - velocity_gain * velocity_risk
        - tilt_gain * tilt_risk
    )
    scale = float(maximum_scale) * torch.sigmoid(logit)
    enabled = finite & moving.bool() & ~terminal.bool() & (contact_count > 0)
    return torch.where(enabled, scale, torch.zeros_like(scale))


def slew_limited_residual(
    desired_residual: torch.Tensor,
    previous_residual: torch.Tensor,
    *,
    maximum_abs: float = 0.10,
    maximum_step: float = 0.01,
) -> torch.Tensor:
    """Bound and rate-limit a normalized 15-D teacher residual."""

    if desired_residual.shape != previous_residual.shape or desired_residual.ndim != 2:
        raise ValueError("desired and previous residual must share [N,A] shape")
    if maximum_abs <= 0.0 or maximum_step <= 0.0:
        raise ValueError("residual limits must be positive")
    desired = desired_residual.clamp(-maximum_abs, maximum_abs)
    delta = (desired - previous_residual).clamp(-maximum_step, maximum_step)
    return (previous_residual + delta).clamp(-maximum_abs, maximum_abs)


def _aggregate_rows(rows: list[dict[str, Any]], *, transition: bool) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty feedback group")

    def values(field: str) -> np.ndarray:
        raw = [row.get(field) for row in rows]
        if any(value is None for value in raw):
            raise ValueError(f"missing search field: {field}")
        array = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite search field: {field}")
        return array

    terminal_speed = values("terminal_speed_mean_mps") if transition else np.zeros(2)
    terminal_ds = values("terminal_double_support_mean") if transition else np.ones(2)
    return {
        "pitch_mean": float(values("signed_pitch_mean_rad").mean()),
        "pitch_p05": float(values("signed_pitch_p05_rad").min()),
        "velocity_rmse": float(math.sqrt(float(values("velocity_mse").mean()))),
        "lateral_rms": float(math.sqrt(float(values("lateral_mse").mean()))),
        "yaw_rmse": float(math.sqrt(float(values("yaw_mse").mean()))),
        "support": float(values("support_mean_m").mean()),
        "slip": float(values("stance_slip_p95_mps").max()),
        "flight": float(values("flight_fraction").mean()),
        "root_z": float(values("root_height_min_m").min()),
        "tilt": float(values("root_tilt_max_rad").max()),
        "terminations": float(sum(bool(row["terminated"]) for row in rows)),
        "timeouts": float(sum(bool(row["time_out"]) for row in rows)),
        "residual_max": float(values("teacher_residual_abs_max").max()),
        "residual_step": float(values("teacher_residual_step_abs_max").max()),
        "terminal_speed": float(np.quantile(terminal_speed, 0.95)),
        "terminal_ds": float(terminal_ds.mean()),
    }


def _role_gates(
    source: dict[str, float],
    candidate: dict[str, float],
    limits: dict[str, float],
    *,
    transition: bool,
) -> dict[str, bool]:
    gates = {
        "candidate_zero_termination": candidate["terminations"] == 0,
        "candidate_zero_timeout": candidate["timeouts"] == 0,
        "pitch_mean": candidate["pitch_mean"] - source["pitch_mean"] >= limits["pitch_mean_delta_rad_min"],
        "pitch_p05": candidate["pitch_p05"] - source["pitch_p05"] >= limits["pitch_p05_delta_rad_min"],
        "velocity": candidate["velocity_rmse"] <= source["velocity_rmse"] + limits["velocity_rmse_regression_mps_max"],
        "lateral": candidate["lateral_rms"] <= source["lateral_rms"] + limits["lateral_rms_regression_mps_max"],
        "yaw": candidate["yaw_rmse"] <= source["yaw_rmse"] + limits["yaw_rmse_regression_radps_max"],
        "support": candidate["support"] <= source["support"] + limits["support_regression_m_max"],
        "slip": candidate["slip"] <= source["slip"] + limits["slip_regression_mps_max"],
        "flight": candidate["flight"] <= source["flight"] + limits["flight_regression_max"],
        "root_height": candidate["root_z"] >= max(limits["root_z_absolute_min_m"], source["root_z"] - limits["root_z_regression_m_max"]),
        "tilt": candidate["tilt"] <= min(limits["tilt_absolute_max_rad"], source["tilt"] + limits["tilt_regression_rad_max"]),
        "residual_bound": candidate["residual_max"] <= limits["teacher_residual_abs_max"],
        "residual_slew": candidate["residual_step"] <= limits["teacher_residual_step_abs_max"],
    }
    if transition:
        gates["terminal_speed"] = candidate["terminal_speed"] <= limits["terminal_speed_p95_absolute_max_mps"]
        gates["terminal_double_support"] = candidate["terminal_ds"] >= limits["terminal_double_support_mean_min"]
    return gates


def feedback_search_score(
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    limits: dict[str, float],
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Score a two-trajectory-per-role feedback candidate.

    This is a search objective only.  A selected controller still requires a
    fresh, higher-powered crossover validation before teacher data collection.
    """

    role_details: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    for role_id in range(ROLE_COUNT):
        source_group = [row for row in source_rows if int(row["role_id"]) == role_id]
        candidate_group = [row for row in candidate_rows if int(row["role_id"]) == role_id]
        try:
            if len(source_group) != 2 or len(candidate_group) != 2:
                raise ValueError("search requires exactly two trajectories per treatment and role")
            source = _aggregate_rows(source_group, transition=False)
            candidate = _aggregate_rows(candidate_group, transition=role_id == 7)
        except ValueError as error:
            role_details[str(role_id)] = {"error": str(error), "gates": {}, "cost": 10000.0}
            total_cost += 10000.0
            continue

        pitch_delta = candidate["pitch_mean"] - source["pitch_mean"]
        p05_delta = candidate["pitch_p05"] - source["pitch_p05"]
        gates = _role_gates(source, candidate, limits, transition=role_id == 7)

        def excess(value: float, threshold: float, scale: float) -> float:
            return max(0.0, value - threshold) / scale

        cost = 100.0 * candidate["terminations"] + 100.0 * candidate["timeouts"]
        cost += 8.0 * excess(limits["pitch_mean_delta_rad_min"], pitch_delta, 0.005) ** 2
        cost += 4.0 * excess(limits["pitch_p05_delta_rad_min"], p05_delta, 0.005) ** 2
        cost += 2.0 * excess(candidate["velocity_rmse"] - source["velocity_rmse"], limits["velocity_rmse_regression_mps_max"], 0.01) ** 2
        cost += excess(candidate["lateral_rms"] - source["lateral_rms"], limits["lateral_rms_regression_mps_max"], 0.015) ** 2
        cost += excess(candidate["yaw_rmse"] - source["yaw_rmse"], limits["yaw_rmse_regression_radps_max"], 0.015) ** 2
        cost += 2.0 * excess(candidate["support"] - source["support"], limits["support_regression_m_max"], 0.002) ** 2
        cost += excess(candidate["slip"] - source["slip"], limits["slip_regression_mps_max"], 0.03) ** 2
        cost += excess(candidate["flight"] - source["flight"], limits["flight_regression_max"], 0.01) ** 2
        cost += 10.0 * excess(limits["root_z_absolute_min_m"], candidate["root_z"], 0.02) ** 2
        cost += 10.0 * excess(candidate["tilt"], limits["tilt_absolute_max_rad"], 0.05) ** 2
        if role_id == 7:
            cost += 4.0 * excess(candidate["terminal_speed"], limits["terminal_speed_p95_absolute_max_mps"], 0.05) ** 2
            cost += 4.0 * excess(limits["terminal_double_support_mean_min"], candidate["terminal_ds"], 0.10) ** 2
        cost += 0.02 * (candidate["residual_max"] / max(limits["teacher_residual_abs_max"], 1.0e-9)) ** 2
        role_details[str(role_id)] = {
            "source": source,
            "candidate": candidate,
            "pitch_mean_delta_rad": pitch_delta,
            "pitch_p05_delta_rad": p05_delta,
            "gates": gates,
            "cost": float(cost),
        }
        total_cost += cost
    return float(total_cost / ROLE_COUNT), role_details


def feedback_validation_gates(
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    limits: dict[str, float],
    *,
    expected_per_role: int,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Evaluate a frozen feedback controller on a larger crossover panel."""

    details: dict[str, dict[str, Any]] = {}
    for role_id in range(ROLE_COUNT):
        source_group = [row for row in source_rows if int(row["role_id"]) == role_id]
        candidate_group = [row for row in candidate_rows if int(row["role_id"]) == role_id]
        try:
            if len(source_group) != expected_per_role or len(candidate_group) != expected_per_role:
                raise ValueError("validation role trajectory count changed")
            source = _aggregate_rows(source_group, transition=False)
            candidate = _aggregate_rows(candidate_group, transition=role_id == 7)
            gates = _role_gates(source, candidate, limits, transition=role_id == 7)
            details[str(role_id)] = {
                "source": source,
                "candidate": candidate,
                "pitch_mean_delta_rad": candidate["pitch_mean"] - source["pitch_mean"],
                "pitch_p05_delta_rad": candidate["pitch_p05"] - source["pitch_p05"],
                "gates": gates,
                "passed": bool(all(gates.values())),
                "error": None,
            }
        except ValueError as error:
            details[str(role_id)] = {
                "source": None, "candidate": None, "gates": {},
                "passed": False, "error": str(error),
            }
    return bool(all(item["passed"] for item in details.values())), details


__all__ = [
    "PARAMETER_COUNT",
    "ROLE_COUNT",
    "feedback_search_score",
    "feedback_scale",
    "feedback_validation_gates",
    "slew_limited_residual",
]
