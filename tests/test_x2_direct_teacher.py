import torch

from humanoidverse.x2_direct_teacher import (
    INITIAL_MEAN,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    direct_teacher_action,
    interpolate_cycle,
    update_cem,
)


def test_periodic_cycle_interpolation_wraps() -> None:
    cycle = torch.arange(60, dtype=torch.float32).reshape(4, 15)
    values = interpolate_cycle(cycle, torch.tensor((0.0, 1.0)))
    torch.testing.assert_close(values[0], values[1])


def test_zero_cycle_and_state_produce_bounded_action() -> None:
    envs = 3
    zeros3 = torch.zeros(envs, 3)
    action, preclip = direct_teacher_action(
        INITIAL_MEAN.repeat(envs, 1),
        torch.zeros(envs, 15),
        torch.ones(15),
        zeros3,
        zeros3,
        zeros3,
        zeros3,
        torch.zeros(envs, 15),
        torch.ones(envs),
    )
    assert action.shape == (envs, 15)
    torch.testing.assert_close(action, preclip)
    assert float(action.abs().max()) <= 1.0


def test_cem_moves_toward_best_candidate_and_stays_bounded() -> None:
    candidates = INITIAL_MEAN.repeat(4, 1)
    candidates[3] = UPPER_BOUNDS
    mean, std, elite_ids = update_cem(
        INITIAL_MEAN,
        0.2 * (UPPER_BOUNDS - LOWER_BOUNDS),
        candidates,
        torch.tensor((0.0, 1.0, 2.0, 3.0)),
        elite_count=1,
    )
    assert elite_ids.tolist() == [3]
    assert torch.all(mean >= LOWER_BOUNDS)
    assert torch.all(mean <= UPPER_BOUNDS)
    assert torch.all(std > 0)
    assert torch.linalg.vector_norm(mean - UPPER_BOUNDS) < torch.linalg.vector_norm(
        INITIAL_MEAN - UPPER_BOUNDS
    )
