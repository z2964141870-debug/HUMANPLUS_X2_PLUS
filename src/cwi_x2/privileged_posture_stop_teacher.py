"""Pure contracts for the X2 state-feedback posture/stop teacher.

This module intentionally contains no IsaacLab dependency.  The teacher is a
bounded feedback matrix, not a learned policy and not a scaled fixed action
direction.  Search candidates are ranked by a lexicographic safety contract so
that a posture gain can never numerically compensate for a fall or a support
regression.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch


ACTION_DIM = 15
FEATURE_DIM = 17
SEGMENTS = ("cruise", "decelerate", "hold")
TREATMENTS = ("source_direct", "fixed_half_direct", "feedback_gated")


def teacher_features(
    *,
    pitch_rad: torch.Tensor,
    pitch_rate_radps: torch.Tensor,
    roll_rad: torch.Tensor,
    roll_rate_radps: torch.Tensor,
    body_velocity_xy_mps: torch.Tensor,
    command_velocity_xy_mps: torch.Tensor,
    yaw_rate_error_radps: torch.Tensor,
    support_outside_m: torch.Tensor,
    contact: torch.Tensor,
    gait_suffix: torch.Tensor,
    decelerating: torch.Tensor,
    holding: torch.Tensor,
    target_pitch_rad: float = -0.05,
) -> torch.Tensor:
    """Build the normalized privileged feedback vector.

    The final two features are explicit mode intent.  The teacher therefore
    may produce different corrections at the same posture in cruise, braking
    and hold; it is not a fixed direction with a state-dependent dose.
    """

    count = pitch_rad.shape[0]
    scalar = (
        pitch_rad,
        pitch_rate_radps,
        roll_rad,
        roll_rate_radps,
        yaw_rate_error_radps,
        support_outside_m,
        decelerating,
        holding,
    )
    if any(value.shape != (count,) for value in scalar):
        raise ValueError("teacher scalar inputs must all have shape [N]")
    if body_velocity_xy_mps.shape != (count, 2):
        raise ValueError("body velocity must have shape [N,2]")
    if command_velocity_xy_mps.shape != (count, 2):
        raise ValueError("command velocity must have shape [N,2]")
    if contact.shape != (count, 2) or gait_suffix.shape != (count, 4):
        raise ValueError("contact/gait inputs must have shapes [N,2]/[N,4]")
    if not math.isfinite(target_pitch_rad):
        raise ValueError("target pitch must be finite")

    velocity_error = command_velocity_xy_mps - body_velocity_xy_mps
    speed = torch.linalg.vector_norm(body_velocity_xy_mps, dim=-1)
    contact_float = contact.to(dtype=pitch_rad.dtype)
    features = torch.stack(
        (
            torch.ones_like(pitch_rad),
            torch.clamp((target_pitch_rad - pitch_rad) / 0.15, -2.0, 2.0),
            torch.clamp(-pitch_rate_radps / 0.50, -2.0, 2.0),
            torch.clamp(roll_rad / 0.15, -2.0, 2.0),
            torch.clamp(-roll_rate_radps / 0.50, -2.0, 2.0),
            torch.clamp(velocity_error[:, 0] / 0.35, -2.0, 2.0),
            torch.clamp(velocity_error[:, 1] / 0.20, -2.0, 2.0),
            torch.clamp(-yaw_rate_error_radps / 0.30, -2.0, 2.0),
            torch.clamp(support_outside_m / 0.04, 0.0, 2.0),
            torch.clamp(speed / 0.35, 0.0, 2.0),
            gait_suffix[:, 0],
            gait_suffix[:, 1],
            gait_suffix[:, 2] - 0.5,
            gait_suffix[:, 3] - 0.5,
            contact_float[:, 0] - contact_float[:, 1],
            decelerating.to(dtype=pitch_rad.dtype),
            holding.to(dtype=pitch_rad.dtype),
        ),
        dim=-1,
    )
    if features.shape != (count, FEATURE_DIM):
        raise RuntimeError("internal teacher feature dimension changed")
    return features


def state_feedback_residual(
    matrix: torch.Tensor,
    features: torch.Tensor,
    previous_residual: torch.Tensor,
    *,
    contact_count: torch.Tensor,
    tilt_rad: torch.Tensor,
    support_outside_m: torch.Tensor,
    holding: torch.Tensor,
    maximum_abs: float = 0.12,
    hold_maximum_abs: float = 0.06,
    maximum_step: float = 0.015,
) -> torch.Tensor:
    """Map state directly to a bounded, slew-limited 15-D correction."""

    count = features.shape[0]
    if features.shape != (count, FEATURE_DIM):
        raise ValueError(f"features must have shape [N,{FEATURE_DIM}]")
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(count, -1, -1)
    if matrix.shape != (count, ACTION_DIM, FEATURE_DIM):
        raise ValueError(f"matrix must have shape [N,{ACTION_DIM},{FEATURE_DIM}]")
    if previous_residual.shape != (count, ACTION_DIM):
        raise ValueError(f"previous residual must have shape [N,{ACTION_DIM}]")
    if any(value.shape != (count,) for value in (contact_count, tilt_rad, support_outside_m, holding)):
        raise ValueError("state-feedback gate inputs must have shape [N]")
    if not (0.0 < hold_maximum_abs <= maximum_abs <= 1.0):
        raise ValueError("residual limits must satisfy 0 < hold <= maximum <= 1")
    if not 0.0 < maximum_step <= maximum_abs:
        raise ValueError("maximum step must lie in (0, maximum_abs]")

    finite = torch.isfinite(features).all(dim=-1) & torch.isfinite(matrix).all(dim=(1, 2))
    finite &= torch.isfinite(tilt_rad) & torch.isfinite(support_outside_m)
    raw = torch.einsum("naf,nf->na", matrix, features) / math.sqrt(FEATURE_DIM)
    limit = torch.where(
        holding.bool(),
        torch.full_like(tilt_rad, hold_maximum_abs),
        torch.full_like(tilt_rad, maximum_abs),
    )
    desired = limit.unsqueeze(-1) * torch.tanh(raw)

    # The risk gates can only reduce authority.  Flight, invalid state, large
    # tilt or gross support loss is fail-closed to zero rather than asking an
    # unvalidated residual to perform recovery.
    tilt_scale = torch.clamp((0.35 - tilt_rad) / 0.15, 0.0, 1.0)
    support_scale = torch.clamp((0.06 - support_outside_m) / 0.04, 0.0, 1.0)
    enabled = finite & (contact_count > 0) & (tilt_rad < 0.35) & (support_outside_m < 0.06)
    desired = desired * torch.minimum(tilt_scale, support_scale).unsqueeze(-1)
    desired = torch.where(enabled.unsqueeze(-1), desired, torch.zeros_like(desired))
    delta = torch.clamp(desired - previous_residual, -maximum_step, maximum_step)
    residual = previous_residual + delta
    residual = torch.maximum(torch.minimum(residual, limit.unsqueeze(-1)), -limit.unsqueeze(-1))
    return torch.where(enabled.unsqueeze(-1), residual, torch.zeros_like(residual))


@dataclass(frozen=True)
class HandoffState:
    dwell_steps: torch.Tensor
    latched: torch.Tensor
    blend: torch.Tensor


def advance_handoff(
    state: HandoffState,
    *,
    speed_mps: torch.Tensor,
    contact_count: torch.Tensor,
    tilt_rad: torch.Tensor,
    support_outside_m: torch.Tensor,
    terminal: torch.Tensor,
    dwell_required: int = 5,
    blend_steps: int = 50,
    speed_max_mps: float = 0.10,
    tilt_max_rad: float = 0.20,
    support_max_m: float = 0.015,
) -> HandoffState:
    """Advance the physical brake-to-hold handoff gate.

    A one-frame low-speed event cannot trigger the stationary actor.  After a
    consecutive safe dwell, blend grows monotonically from zero to one.
    """

    count = speed_mps.shape[0]
    if any(value.shape != (count,) for value in (contact_count, tilt_rad, support_outside_m, terminal)):
        raise ValueError("handoff inputs must have shape [N]")
    if state.dwell_steps.shape != (count,) or state.latched.shape != (count,) or state.blend.shape != (count,):
        raise ValueError("handoff state must have shape [N]")
    if dwell_required < 1 or blend_steps < 1:
        raise ValueError("handoff dwell and blend steps must be positive")

    finite = torch.isfinite(speed_mps) & torch.isfinite(tilt_rad) & torch.isfinite(support_outside_m)
    ready = (
        finite
        & terminal.bool()
        & (speed_mps <= speed_max_mps)
        & (contact_count == 2)
        & (tilt_rad <= tilt_max_rad)
        & (support_outside_m <= support_max_m)
    )
    dwell = torch.where(ready, state.dwell_steps + 1, torch.zeros_like(state.dwell_steps))
    latched = state.latched | (dwell >= dwell_required)
    blend_increment = torch.full_like(state.blend, 1.0 / float(blend_steps))
    blend = torch.where(latched, torch.clamp(state.blend + blend_increment, 0.0, 1.0), state.blend)
    return HandoffState(dwell_steps=dwell, latched=latched, blend=blend)


def summarize_segment(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate per-environment rows for one treatment/segment."""

    if not rows:
        raise ValueError("cannot summarize an empty segment")

    def values(name: str) -> np.ndarray:
        array = np.asarray([row[name] for row in rows], dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"non-finite segment field: {name}")
        return array

    return {
        "environment_count": float(len(rows)),
        "pitch_mean": float(values("pitch_mean_rad").mean()),
        "pitch_p05": float(np.quantile(values("pitch_p05_rad"), 0.05)),
        "velocity_rmse": float(math.sqrt(float(values("velocity_mse").mean()))),
        "lateral_rms": float(math.sqrt(float(values("lateral_mse").mean()))),
        "yaw_rmse": float(math.sqrt(float(values("yaw_mse").mean()))),
        "support": float(values("support_mean_m").mean()),
        "slip": float(np.quantile(values("slip_p95_mps"), 0.95)),
        "flight": float(values("flight_fraction").mean()),
        "root_z": float(values("root_height_min_m").min()),
        "tilt": float(values("tilt_max_rad").max()),
        "speed_p95": float(np.quantile(values("speed_p95_mps"), 0.95)),
        "double_support": float(values("double_support_fraction").mean()),
        "terminations": float(sum(bool(row["terminated"]) for row in rows)),
        "timeouts": float(sum(bool(row["time_out"]) for row in rows)),
        "residual_max": float(values("residual_abs_max").max()),
        "residual_step": float(values("residual_step_abs_max").max()),
        "action_clip": float(values("action_clip_fraction").max()),
        "effectiveness": float(values("effectiveness_ratio_min").min()),
        "wrong_sign": float(values("wrong_sign_fraction").max()),
    }


def feasibility_gates(
    source: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    limits: Mapping[str, float],
) -> dict[str, bool]:
    """Return the explicit three-segment feasibility contract."""

    cruise_s, cruise_c = source["cruise"], candidate["cruise"]
    decel_s, decel_c = source["decelerate"], candidate["decelerate"]
    hold_c = candidate["hold"]
    gates = {
        "candidate_zero_termination": sum(candidate[s]["terminations"] for s in SEGMENTS) == 0,
        "candidate_zero_timeout": sum(candidate[s]["timeouts"] for s in SEGMENTS) == 0,
        "cruise_pitch_mean": cruise_c["pitch_mean"] - cruise_s["pitch_mean"] >= limits["cruise_pitch_mean_delta_min"],
        "cruise_pitch_p05": cruise_c["pitch_p05"] - cruise_s["pitch_p05"] >= limits["cruise_pitch_p05_delta_min"],
        "decel_pitch_mean": decel_c["pitch_mean"] - decel_s["pitch_mean"] >= limits["decel_pitch_mean_delta_min"],
        "decel_pitch_p05": decel_c["pitch_p05"] - decel_s["pitch_p05"] >= limits["decel_pitch_p05_delta_min"],
        "hold_root_height": hold_c["root_z"] >= limits["root_z_absolute_min"],
        "hold_tilt": hold_c["tilt"] <= limits["tilt_absolute_max"],
        "hold_pitch_mean": hold_c["pitch_mean"] >= limits["hold_pitch_mean_min"],
        "hold_pitch_p05": hold_c["pitch_p05"] >= limits["hold_pitch_p05_min"],
        "hold_speed": hold_c["speed_p95"] <= limits["hold_speed_p95_max"],
        "hold_double_support": hold_c["double_support"] >= limits["hold_double_support_min"],
        "residual_bound": max(candidate[s]["residual_max"] for s in SEGMENTS) <= limits["residual_abs_max"],
        "residual_slew": max(candidate[s]["residual_step"] for s in SEGMENTS) <= limits["residual_step_abs_max"],
        "no_action_clip": max(candidate[s]["action_clip"] for s in SEGMENTS) <= limits["action_clip_fraction_max"],
        "physical_effectiveness": min(candidate[s]["effectiveness"] for s in SEGMENTS) >= limits["physical_effectiveness_min"],
        "physical_sign": max(candidate[s]["wrong_sign"] for s in SEGMENTS) <= limits["wrong_sign_fraction_max"],
    }
    prefixes = {"cruise": "cruise", "decelerate": "decel", "hold": "hold"}
    for segment in SEGMENTS:
        source_segment, candidate_segment = source[segment], candidate[segment]
        prefix = prefixes[segment]
        gates[f"{prefix}_zero_termination"] = candidate_segment["terminations"] == 0
        gates[f"{prefix}_zero_timeout"] = candidate_segment["timeouts"] == 0
        gates[f"{prefix}_velocity"] = candidate_segment["velocity_rmse"] <= source_segment["velocity_rmse"] + limits["velocity_regression_max"]
        gates[f"{prefix}_lateral"] = candidate_segment["lateral_rms"] <= source_segment["lateral_rms"] + limits["lateral_regression_max"]
        gates[f"{prefix}_yaw"] = candidate_segment["yaw_rmse"] <= source_segment["yaw_rmse"] + limits["yaw_regression_max"]
        gates[f"{prefix}_support"] = candidate_segment["support"] <= source_segment["support"] + limits["support_regression_max"]
        gates[f"{prefix}_slip"] = candidate_segment["slip"] <= source_segment["slip"] + limits["slip_regression_max"]
        gates[f"{prefix}_flight"] = candidate_segment["flight"] <= source_segment["flight"] + limits["flight_regression_max"]
        gates[f"{prefix}_root_height"] = candidate_segment["root_z"] >= max(
            limits["root_z_absolute_min"],
            source_segment["root_z"] - limits["root_z_regression_max"],
        )
        gates[f"{prefix}_tilt"] = candidate_segment["tilt"] <= min(
            limits["tilt_absolute_max"],
            source_segment["tilt"] + limits["tilt_regression_max"],
        )
    return gates


def hierarchical_candidate_key(
    source: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    limits: Mapping[str, float],
) -> tuple[float, ...]:
    """Return a lexicographic search key; lower is better.

    This is deliberately not a weighted scalar reward.  Earlier tuple entries
    dominate every later posture term regardless of numerical magnitude.
    """

    gates = feasibility_gates(source, candidate, limits)
    hard_names = [
        "candidate_zero_termination",
        "candidate_zero_timeout",
        "residual_bound",
        "residual_slew",
        "no_action_clip",
        "physical_effectiveness",
        "physical_sign",
    ]
    for prefix in ("cruise", "decel", "hold"):
        hard_names.extend((f"{prefix}_root_height", f"{prefix}_tilt"))
    hard_failures = float(sum(not gates[name] for name in hard_names))

    def positive_excess(value: float, threshold: float, scale: float) -> float:
        return max(0.0, value - threshold) / scale

    safety_excess = 0.0
    for segment in SEGMENTS:
        s, c = source[segment], candidate[segment]
        safety_excess = max(
            safety_excess,
            positive_excess(c["velocity_rmse"] - s["velocity_rmse"], limits["velocity_regression_max"], 0.01),
            positive_excess(c["lateral_rms"] - s["lateral_rms"], limits["lateral_regression_max"], 0.015),
            positive_excess(c["yaw_rmse"] - s["yaw_rmse"], limits["yaw_regression_max"], 0.015),
            positive_excess(c["support"] - s["support"], limits["support_regression_max"], 0.001),
            positive_excess(c["slip"] - s["slip"], limits["slip_regression_max"], 0.03),
            positive_excess(c["flight"] - s["flight"], limits["flight_regression_max"], 0.01),
        )
    terminal_excess = max(
        positive_excess(candidate["hold"]["speed_p95"], limits["hold_speed_p95_max"], 0.05),
        positive_excess(limits["hold_double_support_min"], candidate["hold"]["double_support"], 0.05),
    )
    pitch_shortfall = max(
        0.0,
        limits["cruise_pitch_mean_delta_min"] - (candidate["cruise"]["pitch_mean"] - source["cruise"]["pitch_mean"]),
        limits["decel_pitch_mean_delta_min"] - (candidate["decelerate"]["pitch_mean"] - source["decelerate"]["pitch_mean"]),
    )
    pitch_tail_shortfall = max(
        0.0,
        limits["cruise_pitch_p05_delta_min"] - (candidate["cruise"]["pitch_p05"] - source["cruise"]["pitch_p05"]),
        limits["decel_pitch_p05_delta_min"] - (candidate["decelerate"]["pitch_p05"] - source["decelerate"]["pitch_p05"]),
    )
    effort = max(candidate[s]["residual_max"] for s in SEGMENTS)
    slew = max(candidate[s]["residual_step"] for s in SEGMENTS)
    return (
        hard_failures,
        float(max(safety_excess, terminal_excess)),
        float(pitch_shortfall),
        float(pitch_tail_shortfall),
        float(effort),
        float(slew),
    )


__all__ = [
    "ACTION_DIM",
    "FEATURE_DIM",
    "HandoffState",
    "SEGMENTS",
    "TREATMENTS",
    "advance_handoff",
    "feasibility_gates",
    "hierarchical_candidate_key",
    "state_feedback_residual",
    "summarize_segment",
    "teacher_features",
]
