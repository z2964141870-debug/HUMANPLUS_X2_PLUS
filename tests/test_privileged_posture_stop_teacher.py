from __future__ import annotations

import torch

from cwi_x2.privileged_posture_stop_teacher import (
    ACTION_DIM,
    FEATURE_DIM,
    HandoffState,
    advance_handoff,
    feasibility_gates,
    hierarchical_candidate_key,
    state_feedback_residual,
    teacher_features,
)


def _features(count: int = 4) -> torch.Tensor:
    z = torch.zeros(count)
    return teacher_features(
        pitch_rad=torch.full((count,), -0.20),
        pitch_rate_radps=z,
        roll_rad=z,
        roll_rate_radps=z,
        body_velocity_xy_mps=torch.tensor([[0.30, 0.0]]).repeat(count, 1),
        command_velocity_xy_mps=torch.tensor([[0.35, 0.0]]).repeat(count, 1),
        yaw_rate_error_radps=z,
        support_outside_m=z,
        contact=torch.ones(count, 2, dtype=torch.bool),
        gait_suffix=torch.tensor([[0.0, 1.0, 1.0, 1.0]]).repeat(count, 1),
        decelerating=torch.zeros(count, dtype=torch.bool),
        holding=torch.zeros(count, dtype=torch.bool),
    )


def test_features_are_state_and_mode_conditioned() -> None:
    features = _features()
    assert features.shape == (4, FEATURE_DIM)
    assert torch.all(features[:, 1] > 0.0)
    holding = _features()
    holding[:, -1] = 1.0
    assert not torch.equal(features, holding)


def test_feedback_is_bounded_slew_limited_and_flight_zero() -> None:
    features = _features()
    matrix = torch.ones(4, ACTION_DIM, FEATURE_DIM) * 10.0
    previous = torch.zeros(4, ACTION_DIM)
    contact_count = torch.tensor([2, 2, 0, 2])
    residual = state_feedback_residual(
        matrix,
        features,
        previous,
        contact_count=contact_count,
        tilt_rad=torch.tensor([0.0, 0.0, 0.0, 0.4]),
        support_outside_m=torch.zeros(4),
        holding=torch.tensor([False, True, False, False]),
    )
    assert float(residual[0].abs().max()) <= 0.0150001
    assert float(residual[1].abs().max()) <= 0.0150001
    assert torch.equal(residual[2], torch.zeros(ACTION_DIM))
    assert torch.equal(residual[3], torch.zeros(ACTION_DIM))


def test_handoff_requires_consecutive_safe_double_support() -> None:
    count = 2
    state = HandoffState(
        dwell_steps=torch.zeros(count, dtype=torch.int64),
        latched=torch.zeros(count, dtype=torch.bool),
        blend=torch.zeros(count),
    )
    for _ in range(4):
        state = advance_handoff(
            state,
            speed_mps=torch.tensor([0.08, 0.12]),
            contact_count=torch.tensor([2, 2]),
            tilt_rad=torch.tensor([0.10, 0.10]),
            support_outside_m=torch.tensor([0.0, 0.0]),
            terminal=torch.ones(count, dtype=torch.bool),
        )
    assert not bool(state.latched.any())
    state = advance_handoff(
        state,
        speed_mps=torch.tensor([0.08, 0.08]),
        contact_count=torch.tensor([2, 1]),
        tilt_rad=torch.tensor([0.10, 0.10]),
        support_outside_m=torch.tensor([0.0, 0.0]),
        terminal=torch.ones(count, dtype=torch.bool),
    )
    assert bool(state.latched[0])
    assert not bool(state.latched[1])
    assert 0.0 < float(state.blend[0]) < 1.0


def _summary(pitch: float = -0.20) -> dict[str, dict[str, float]]:
    row = {
        "pitch_mean": pitch,
        "pitch_p05": pitch - 0.03,
        "velocity_rmse": 0.05,
        "lateral_rms": 0.02,
        "yaw_rmse": 0.02,
        "support": 0.001,
        "slip": 0.10,
        "flight": 0.0,
        "root_z": 0.66,
        "tilt": 0.20,
        "speed_p95": 0.05,
        "double_support": 0.99,
        "terminations": 0.0,
        "timeouts": 0.0,
        "residual_max": 0.05,
        "residual_step": 0.01,
        "action_clip": 0.0,
        "effectiveness": 1.0,
        "wrong_sign": 0.0,
    }
    return {name: dict(row) for name in ("cruise", "decelerate", "hold")}


LIMITS = {
    "cruise_pitch_mean_delta_min": 0.020,
    "cruise_pitch_p05_delta_min": 0.010,
    "decel_pitch_mean_delta_min": 0.015,
    "decel_pitch_p05_delta_min": 0.0,
    "root_z_absolute_min": 0.60,
    "tilt_absolute_max": 0.35,
    "hold_speed_p95_max": 0.12,
    "hold_double_support_min": 0.95,
    "hold_pitch_mean_min": -0.15,
    "hold_pitch_p05_min": -0.25,
    "residual_abs_max": 0.12,
    "residual_step_abs_max": 0.015,
    "action_clip_fraction_max": 0.0,
    "physical_effectiveness_min": 0.90,
    "wrong_sign_fraction_max": 0.0,
    "velocity_regression_max": 0.010,
    "lateral_regression_max": 0.015,
    "yaw_regression_max": 0.015,
    "support_regression_max": 0.001,
    "slip_regression_max": 0.030,
    "flight_regression_max": 0.010,
    "root_z_regression_max": 0.005,
    "tilt_regression_max": 0.010,
}


def test_feasibility_and_hierarchical_safety_priority() -> None:
    source = _summary()
    safe = _summary(-0.175)
    safe["cruise"]["pitch_p05"] = source["cruise"]["pitch_p05"] + 0.011
    safe["decelerate"]["pitch_p05"] = source["decelerate"]["pitch_p05"]
    safe["hold"]["pitch_mean"] = -0.10
    safe["hold"]["pitch_p05"] = -0.15
    gates = feasibility_gates(source, safe, LIMITS)
    assert all(gates.values())

    unsafe = _summary(-0.05)
    unsafe["hold"]["terminations"] = 1.0
    assert hierarchical_candidate_key(source, safe, LIMITS) < hierarchical_candidate_key(source, unsafe, LIMITS)


def test_action_clipping_cannot_be_hidden_by_pitch() -> None:
    source = _summary()
    candidate = _summary(-0.05)
    candidate["hold"]["action_clip"] = 0.01
    gates = feasibility_gates(source, candidate, LIMITS)
    assert not gates["no_action_clip"]
    assert hierarchical_candidate_key(source, candidate, LIMITS)[0] >= 1.0
