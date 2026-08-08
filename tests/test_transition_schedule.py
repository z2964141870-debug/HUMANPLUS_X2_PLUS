import pytest
import torch

from cwi_x2.transition_schedule import smooth_transition_speed, stratified_phase_offsets


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
