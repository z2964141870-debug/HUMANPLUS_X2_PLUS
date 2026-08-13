"""Pure state-machine helpers for the X2 brake/hold handoff panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from cwi_x2.privileged_posture_stop_teacher import HandoffState, advance_handoff


@dataclass(frozen=True)
class HandoffStrategy:
    name: str
    blend_mode: str
    stand_raw_weight: float = 0.5
    brake_gain: float = 0.0
    blend_start_s: float = 8.2
    blend_duration_s: float = 1.0
    speed_blend_upper_mps: float = 0.40
    speed_blend_lower_mps: float = 0.08
    direct_delay_s: float = 0.0


STRATEGIES: tuple[HandoffStrategy, ...] = (
    HandoffStrategy("direct_mix", "direct", 0.5),
    HandoffStrategy("locomotion_zero", "never", 0.5),
    HandoffStrategy("direct_main_stand", "direct", 0.0),
    HandoffStrategy("direct_stationary", "direct", 1.0),
    HandoffStrategy("gated_zero_mix", "gated", 0.5, 0.0),
    HandoffStrategy("gated_brake_025_mix", "gated", 0.5, 0.25),
    HandoffStrategy("gated_brake_050_mix", "gated", 0.5, 0.50),
    HandoffStrategy("gated_brake_100_mix", "gated", 0.5, 1.00),
    HandoffStrategy("decel_blend_2s_mix", "time", 0.5, 0.0, 6.2, 2.0),
    HandoffStrategy("decel_blend_4s_mix", "time", 0.5, 0.0, 6.2, 4.0),
    HandoffStrategy("decel_blend_2s_main", "time", 0.0, 0.0, 6.2, 2.0),
    HandoffStrategy("decel_blend_2s_stationary", "time", 1.0, 0.0, 6.2, 2.0),
    HandoffStrategy("speed_blend_mix", "speed", 0.5),
    HandoffStrategy("speed_blend_main", "speed", 0.0),
    HandoffStrategy("speed_blend_stationary", "speed", 1.0),
    HandoffStrategy("delayed_direct_mix", "direct", 0.5, 0.0, 8.2, 1.0, 0.40, 0.08, 1.0),
)


def strategy_by_id(strategy_id: int) -> HandoffStrategy:
    if not 0 <= strategy_id < len(STRATEGIES):
        raise IndexError(strategy_id)
    return STRATEGIES[strategy_id]


def mixed_strategy_parameters(
    strategy_ids: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Materialize immutable strategy constants for batched lanes."""
    ids = strategy_ids.detach().cpu().tolist()
    selected = [strategy_by_id(int(index)) for index in ids]
    return {
        "stand_raw_weight": torch.tensor([s.stand_raw_weight for s in selected], device=device, dtype=dtype),
        "brake_gain": torch.tensor([s.brake_gain for s in selected], device=device, dtype=dtype),
        "blend_start_s": torch.tensor([s.blend_start_s for s in selected], device=device, dtype=dtype),
        "blend_duration_s": torch.tensor([s.blend_duration_s for s in selected], device=device, dtype=dtype),
        "speed_upper": torch.tensor([s.speed_blend_upper_mps for s in selected], device=device, dtype=dtype),
        "speed_lower": torch.tensor([s.speed_blend_lower_mps for s in selected], device=device, dtype=dtype),
        "direct_delay_s": torch.tensor([s.direct_delay_s for s in selected], device=device, dtype=dtype),
        "mode_direct": torch.tensor([s.blend_mode == "direct" for s in selected], device=device),
        "mode_never": torch.tensor([s.blend_mode == "never" for s in selected], device=device),
        "mode_gated": torch.tensor([s.blend_mode == "gated" for s in selected], device=device),
        "mode_time": torch.tensor([s.blend_mode == "time" for s in selected], device=device),
        "mode_speed": torch.tensor([s.blend_mode == "speed" for s in selected], device=device),
    }


def handoff_control(
    *,
    state: HandoffState,
    strategy_ids: torch.Tensor,
    elapsed_s: torch.Tensor,
    scheduled_speed_mps: torch.Tensor,
    body_velocity_xy_mps: torch.Tensor,
    speed_mps: torch.Tensor,
    contact_count: torch.Tensor,
    tilt_rad: torch.Tensor,
    support_outside_m: torch.Tensor,
    hold_start_s: float,
    dwell_required: int,
    gate_blend_steps: int,
    gate_speed_max_mps: float,
    gate_tilt_max_rad: float,
    gate_support_max_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HandoffState, torch.Tensor]:
    """Return command, blend, stationary raw weight, next state, and braking mask."""
    params = mixed_strategy_parameters(
        strategy_ids, device=elapsed_s.device, dtype=elapsed_s.dtype
    )
    terminal = elapsed_s >= hold_start_s
    gated_terminal = terminal & params["mode_gated"]
    gated_state = advance_handoff(
        state,
        speed_mps=speed_mps,
        contact_count=contact_count,
        tilt_rad=tilt_rad,
        support_outside_m=support_outside_m,
        terminal=gated_terminal,
        dwell_required=dwell_required,
        blend_steps=gate_blend_steps,
        speed_max_mps=gate_speed_max_mps,
        tilt_max_rad=gate_tilt_max_rad,
        support_max_m=gate_support_max_m,
    )

    direct = terminal & params["mode_direct"] & (
        elapsed_s >= hold_start_s + params["direct_delay_s"]
    )
    time_blend = torch.clamp(
        (elapsed_s - params["blend_start_s"]) / params["blend_duration_s"].clamp_min(1.0e-6),
        0.0,
        1.0,
    )
    time_blend = torch.where(params["mode_time"], time_blend, torch.zeros_like(time_blend))
    speed_blend = torch.clamp(
        (params["speed_upper"] - speed_mps)
        / (params["speed_upper"] - params["speed_lower"]).clamp_min(1.0e-6),
        0.0,
        1.0,
    )
    speed_blend = torch.where(terminal & params["mode_speed"], speed_blend, torch.zeros_like(speed_blend))
    # A handoff is irreversible within an event.  This avoids action chattering
    # if the speed estimate crosses a blend threshold again.
    speed_blend = torch.maximum(speed_blend, torch.where(params["mode_speed"], state.blend, torch.zeros_like(state.blend)))
    transition_blend = torch.maximum(time_blend, speed_blend)
    transition_blend = torch.maximum(transition_blend, direct.to(transition_blend.dtype))
    transition_blend = torch.maximum(
        transition_blend,
        torch.where(params["mode_gated"], gated_state.blend, torch.zeros_like(transition_blend)),
    )
    transition_blend = torch.where(
        params["mode_never"], torch.zeros_like(transition_blend), transition_blend
    )
    # The pre-move standing actor is an event bootstrap, not a completed
    # terminal handoff.  Do not persist this temporary blend in state.
    blend = torch.where(elapsed_s < 1.0, torch.ones_like(transition_blend), transition_blend)

    next_state = HandoffState(
        dwell_steps=torch.where(params["mode_gated"], gated_state.dwell_steps, state.dwell_steps),
        latched=(state.latched | direct | (time_blend >= 1.0) | (speed_blend >= 1.0) | gated_state.latched),
        blend=transition_blend,
    )

    command = torch.zeros(elapsed_s.shape[0], 3, device=elapsed_s.device, dtype=elapsed_s.dtype)
    command[:, 0] = scheduled_speed_mps
    braking = terminal & params["mode_gated"] & (blend < 1.0) & (params["brake_gain"] > 0.0)
    brake = -params["brake_gain"].unsqueeze(-1) * body_velocity_xy_mps
    brake = torch.clamp(brake, -0.20, 0.20)
    command[:, :2] = torch.where(braking.unsqueeze(-1), brake, command[:, :2])
    command = torch.where((blend >= 1.0).unsqueeze(-1), torch.zeros_like(command), command)
    return command, blend, params["stand_raw_weight"], next_state, braking


def summarize_rows(rows: Iterable[dict], strategy_id: int) -> dict[str, dict[str, float]]:
    relevant = [row for row in rows if int(row["strategy_id"]) == strategy_id]
    result: dict[str, dict[str, float]] = {}
    for segment in ("cruise", "decelerate", "hold"):
        subset = [row for row in relevant if row["segment"] == segment]
        if not subset:
            raise ValueError(f"missing {segment} rows for strategy {strategy_id}")
        count = len(subset)
        result[segment] = {
            "environment_count": count,
            "terminations": sum(bool(row["terminated"]) for row in subset),
            "timeouts": sum(bool(row["time_out"]) for row in subset),
            "sample_fraction": sum(float(row["sample_fraction"]) for row in subset) / count,
            "pitch_mean": sum(float(row["pitch_mean_rad"]) for row in subset) / count,
            "pitch_p05": sum(float(row["pitch_p05_rad"]) for row in subset) / count,
            "velocity_rmse": (sum(float(row["velocity_mse"]) for row in subset) / count) ** 0.5,
            "lateral_rms": (sum(float(row["lateral_mse"]) for row in subset) / count) ** 0.5,
            "yaw_rmse": (sum(float(row["yaw_mse"]) for row in subset) / count) ** 0.5,
            "support": sum(float(row["support_mean_m"]) for row in subset) / count,
            "slip": max(float(row["slip_p95_mps"]) for row in subset),
            "flight": sum(float(row["flight_fraction"]) for row in subset) / count,
            "root_z": min(float(row["root_height_min_m"]) for row in subset),
            "tilt": max(float(row["tilt_max_rad"]) for row in subset),
            "speed_p95": max(float(row["speed_p95_mps"]) for row in subset),
            "double_support": sum(float(row["double_support_fraction"]) for row in subset) / count,
            "action_slew": max(float(row["action_slew_max"]) for row in subset),
            "normalized_clip": max(float(row["normalized_clip_fraction"]) for row in subset),
            "stationary_fraction": sum(float(row["stationary_fraction"]) for row in subset) / count,
        }
    return result


def strategy_key(summary: dict[str, dict[str, float]]) -> tuple[float, ...]:
    cruise, decel, hold = (summary[name] for name in ("cruise", "decelerate", "hold"))
    moving_terminations = cruise["terminations"] + decel["terminations"]
    safety_violations = sum((
        hold["root_z"] < 0.60,
        hold["tilt"] > 0.35,
        hold["speed_p95"] > 0.12,
        hold["double_support"] < 0.98,
        hold["flight"] > 0.01,
        hold["action_slew"] > 0.10,
    ))
    return (
        float(moving_terminations),
        float(hold["terminations"]),
        float(safety_violations),
        float(hold["speed_p95"]),
        float(hold["tilt"]),
        float(hold["support"]),
        float(hold["action_slew"]),
    )


def handoff_gates(
    source: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, bool]:
    c, d, h = (candidate[name] for name in ("cruise", "decelerate", "hold"))
    sc, sd = source["cruise"], source["decelerate"]
    return {
        "zero_termination": sum(candidate[s]["terminations"] for s in candidate) == 0,
        "zero_timeout": sum(candidate[s]["timeouts"] for s in candidate) == 0,
        "cruise_velocity": c["velocity_rmse"] <= sc["velocity_rmse"] + 0.010,
        "cruise_lateral": c["lateral_rms"] <= sc["lateral_rms"] + 0.015,
        "cruise_yaw": c["yaw_rmse"] <= sc["yaw_rmse"] + 0.015,
        "cruise_support": c["support"] <= sc["support"] + 0.002,
        "cruise_slip": c["slip"] <= sc["slip"] + 0.030,
        "decel_velocity": d["velocity_rmse"] <= sd["velocity_rmse"] + 0.015,
        "decel_lateral": d["lateral_rms"] <= sd["lateral_rms"] + 0.015,
        "decel_yaw": d["yaw_rmse"] <= sd["yaw_rmse"] + 0.020,
        "decel_support": d["support"] <= sd["support"] + 0.003,
        "decel_slip": d["slip"] <= sd["slip"] + 0.030,
        "hold_speed": h["speed_p95"] <= 0.12,
        "hold_double_support": h["double_support"] >= 0.98,
        "hold_root_height": h["root_z"] >= 0.60,
        "hold_tilt": h["tilt"] <= 0.35,
        "hold_support": h["support"] <= 0.015,
        "hold_slip": h["slip"] <= 0.15,
        "hold_flight": h["flight"] <= 0.01,
        "action_slew": max(candidate[s]["action_slew"] for s in candidate) <= 0.10,
        "normalized_clip": max(candidate[s]["normalized_clip"] for s in candidate)
        <= max(source[s]["normalized_clip"] for s in source) + 0.01,
    }
