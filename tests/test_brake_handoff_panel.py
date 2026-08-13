import torch

from cwi_x2.brake_handoff_panel import (
    STRATEGIES,
    handoff_control,
    handoff_gates,
    strategy_key,
)
from cwi_x2.privileged_posture_stop_teacher import HandoffState


def _state(n=4):
    return HandoffState(
        dwell_steps=torch.zeros(n, dtype=torch.int64),
        latched=torch.zeros(n, dtype=torch.bool),
        blend=torch.zeros(n),
    )


def test_strategy_panel_is_fixed_and_unique():
    assert len(STRATEGIES) == 16
    assert len({strategy.name for strategy in STRATEGIES}) == 16
    assert STRATEGIES[0].name == "direct_mix"
    assert STRATEGIES[1].name == "locomotion_zero"


def test_direct_never_time_and_speed_modes():
    ids = torch.tensor([0, 1, 8, 12])
    command, blend, weight, state, braking = handoff_control(
        state=_state(), strategy_ids=ids,
        elapsed_s=torch.tensor([8.3, 8.3, 7.2, 8.3]),
        scheduled_speed_mps=torch.zeros(4),
        body_velocity_xy_mps=torch.tensor([[0.2, 0.0]] * 4),
        speed_mps=torch.tensor([0.2, 0.2, 0.2, 0.2]),
        contact_count=torch.full((4,), 2), tilt_rad=torch.zeros(4),
        support_outside_m=torch.zeros(4), hold_start_s=8.2,
        dwell_required=5, gate_blend_steps=50, gate_speed_max_mps=0.1,
        gate_tilt_max_rad=0.2, gate_support_max_m=0.015,
    )
    assert torch.allclose(blend, torch.tensor([1.0, 0.0, 0.5, 0.625]))
    assert not braking.any()
    assert command[0].abs().sum() == 0
    assert state.latched[0]
    assert weight.tolist() == [0.5, 0.5, 0.5, 0.5]


def test_gated_brake_only_before_latch():
    ids = torch.tensor([6])
    command, blend, _, state, braking = handoff_control(
        state=_state(1), strategy_ids=ids, elapsed_s=torch.tensor([8.4]),
        scheduled_speed_mps=torch.zeros(1), body_velocity_xy_mps=torch.tensor([[0.3, 0.0]]),
        speed_mps=torch.tensor([0.3]), contact_count=torch.tensor([2]),
        tilt_rad=torch.zeros(1), support_outside_m=torch.zeros(1), hold_start_s=8.2,
        dwell_required=2, gate_blend_steps=4, gate_speed_max_mps=0.1,
        gate_tilt_max_rad=0.2, gate_support_max_m=0.015,
    )
    assert braking.item()
    assert blend.item() == 0.0
    assert torch.allclose(command[0, :2], torch.tensor([-0.15, 0.0]))
    assert not state.latched.item()


def test_premove_stationary_bootstrap_does_not_latch_speed_blend():
    ids = torch.tensor([12])
    _, blend0, _, state, _ = handoff_control(
        state=_state(1), strategy_ids=ids, elapsed_s=torch.tensor([0.5]),
        scheduled_speed_mps=torch.zeros(1), body_velocity_xy_mps=torch.zeros(1, 2),
        speed_mps=torch.zeros(1), contact_count=torch.tensor([2]), tilt_rad=torch.zeros(1),
        support_outside_m=torch.zeros(1), hold_start_s=8.2, dwell_required=2,
        gate_blend_steps=4, gate_speed_max_mps=0.1, gate_tilt_max_rad=0.2,
        gate_support_max_m=0.015,
    )
    assert blend0.item() == 1.0
    assert state.blend.item() == 0.0
    _, blend1, _, _, _ = handoff_control(
        state=state, strategy_ids=ids, elapsed_s=torch.tensor([2.0]),
        scheduled_speed_mps=torch.tensor([0.35]), body_velocity_xy_mps=torch.tensor([[0.35, 0.0]]),
        speed_mps=torch.tensor([0.35]), contact_count=torch.tensor([1]), tilt_rad=torch.zeros(1),
        support_outside_m=torch.zeros(1), hold_start_s=8.2, dwell_required=2,
        gate_blend_steps=4, gate_speed_max_mps=0.1, gate_tilt_max_rad=0.2,
        gate_support_max_m=0.015,
    )
    assert blend1.item() == 0.0


def test_key_prioritizes_moving_then_hold_terminations():
    def summary(moving, hold):
        base = {"terminations": 0, "root_z": 0.65, "tilt": 0.1, "speed_p95": 0.05,
                "double_support": 1.0, "flight": 0.0, "action_slew": 0.01, "support": 0.0}
        return {"cruise": dict(base, terminations=moving), "decelerate": dict(base),
                "hold": dict(base, terminations=hold)}
    assert strategy_key(summary(0, 10)) < strategy_key(summary(1, 0))
    assert strategy_key(summary(0, 1)) < strategy_key(summary(0, 2))


def test_handoff_gate_requires_candidate_absolute_safety():
    segment = {
        "terminations": 0, "timeouts": 0, "velocity_rmse": 0.1, "lateral_rms": 0.02,
        "yaw_rmse": 0.2, "support": 0.01, "slip": 0.1, "speed_p95": 0.05,
        "double_support": 1.0, "root_z": 0.65, "tilt": 0.2, "flight": 0.0,
        "action_slew": 0.05, "normalized_clip": 0.0,
    }
    source = {name: dict(segment) for name in ("cruise", "decelerate", "hold")}
    candidate = {name: dict(segment) for name in source}
    assert all(handoff_gates(source, candidate).values())
    candidate["hold"]["terminations"] = 1
    assert not handoff_gates(source, candidate)["zero_termination"]
