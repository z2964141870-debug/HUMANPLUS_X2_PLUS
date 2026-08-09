import pytest
import torch

from cwi_x2.transition_schedule import (
    audit_phase_consistent_event,
    smooth_transition_speed,
    stratified_phase_offsets,
)


def test_transition_schedule_hits_each_contract_boundary():
    elapsed = torch.tensor([0.0, 1.0, 1.5, 2.0, 6.0, 7.0, 8.0, 12.0])
    cruise = torch.full_like(elapsed, 0.30)
    speed = smooth_transition_speed(
        elapsed,
        cruise,
        stand_s=1.0,
        accelerate_s=1.0,
        cruise_s=4.0,
        decelerate_s=2.0,
    )
    torch.testing.assert_close(
        speed,
        torch.tensor([0.0, 0.0, 0.15, 0.30, 0.30, 0.15, 0.0, 0.0]),
    )


def test_transition_schedule_scales_per_environment_and_is_bounded():
    elapsed = torch.linspace(0.0, 10.0, 101)
    cruise = torch.linspace(0.2, 0.4, 101)
    speed = smooth_transition_speed(
        elapsed,
        cruise,
        stand_s=1.0,
        accelerate_s=1.0,
        cruise_s=4.0,
        decelerate_s=2.0,
    )
    assert torch.all(speed >= 0.0)
    assert torch.all(speed <= cruise)


@pytest.mark.parametrize(
    "durations",
    [(-1.0, 1.0, 4.0, 2.0), (1.0, 0.0, 4.0, 2.0), (1.0, 1.0, 4.0, 0.0)],
)
def test_transition_schedule_rejects_invalid_durations(durations):
    with pytest.raises(ValueError):
        smooth_transition_speed(
            torch.zeros(1),
            torch.ones(1),
            stand_s=durations[0],
            accelerate_s=durations[1],
            cruise_s=durations[2],
            decelerate_s=durations[3],
        )


def test_stratified_offsets_cover_requested_interval():
    torch.testing.assert_close(
        stratified_phase_offsets(5, 8.0),
        torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0]),
    )
    torch.testing.assert_close(stratified_phase_offsets(1, 8.0), torch.zeros(1))


@pytest.mark.parametrize("num_envs,maximum", [(0, 1.0), (4, -1.0)])
def test_stratified_offsets_reject_invalid_contract(num_envs, maximum):
    with pytest.raises(ValueError):
        stratified_phase_offsets(num_envs, maximum)


def test_phase_consistent_event_contains_terminal_double_support_hold():
    audit = audit_phase_consistent_event(
        stand_s=1.0,
        accelerate_s=1.0,
        cruise_s=4.0,
        decelerate_s=2.0,
        terminal_hold_s=2.0,
        gait_cycle_s=0.8,
        double_support_fraction=0.30,
        rollout_s=10.24,
        random_episode_phase=False,
        maximum_phase_offset_s=0.0,
    )
    assert audit.event_end_s == 8.0
    assert audit.required_rollout_s == 10.0
    assert audit.terminal_gait_phase == pytest.approx(0.0)
    assert audit.terminal_in_double_support


def test_handoff_owned_event_starts_at_acceleration_and_stops_in_double_support():
    audit = audit_phase_consistent_event(
        stand_s=0.0,
        accelerate_s=1.0,
        cruise_s=4.2,
        decelerate_s=2.0,
        terminal_hold_s=2.0,
        gait_cycle_s=0.8,
        double_support_fraction=0.30,
        rollout_s=10.24,
        random_episode_phase=False,
        maximum_phase_offset_s=0.0,
    )
    assert audit.event_end_s == pytest.approx(7.2)
    assert audit.required_rollout_s == pytest.approx(9.2)
    assert audit.terminal_gait_phase == pytest.approx(0.0)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"random_episode_phase": True}, "random episode phase"),
        ({"maximum_phase_offset_s": 0.1}, "zero command phase offset"),
        ({"rollout_s": 9.9}, "does not contain required"),
        ({"cruise_s": 3.8}, "outside double support"),
    ],
)
def test_phase_consistent_event_rejects_clock_or_contact_mismatch(overrides, match):
    kwargs = {
        "stand_s": 1.0,
        "accelerate_s": 1.0,
        "cruise_s": 4.0,
        "decelerate_s": 2.0,
        "terminal_hold_s": 2.0,
        "gait_cycle_s": 0.8,
        "double_support_fraction": 0.30,
        "rollout_s": 10.24,
        "random_episode_phase": False,
        "maximum_phase_offset_s": 0.0,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        audit_phase_consistent_event(**kwargs)
