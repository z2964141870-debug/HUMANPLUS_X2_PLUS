import torch
import pytest

from cwi_x2.phase72_antithetic import (
    ANCHOR_STEPS,
    HEAD_DIM,
    HORIZON_STEPS,
    NUM_ENVS,
    analytical_head_ascent64,
    anchor_returns64,
    cosine,
    generate_epsilon,
    paired_direction,
    paired_standardized_credit,
    projection,
    stratified_bootstrap_projection,
    tensor_hash,
)


def test_schedule_is_seeded_and_antithetic() -> None:
    first = generate_epsilon(721041)
    replay = generate_epsilon(721041)
    other = generate_epsilon(721042)
    assert first.shape == (HORIZON_STEPS, NUM_ENVS, 2)
    assert torch.equal(first, replay)
    assert not torch.equal(first, other)
    assert tensor_hash(first) == tensor_hash(replay)
    assert float((first + -first).abs().max()) == 0.0


def test_anchor_return_uses_full_horizon() -> None:
    reward = torch.zeros(400, 2)
    reward[-1] = torch.tensor([1.0, 2.0])
    value = anchor_returns64(reward, gamma=0.5)
    assert value.shape == (ANCHOR_STEPS, 2)
    assert torch.allclose(value[-1], torch.tensor([0.5**200, 2.0 * 0.5**200], dtype=torch.float64))


def test_pair_credit_uses_one_shared_scale() -> None:
    plus = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    minus = -plus
    plus_credit, minus_credit, scale = paired_standardized_credit(plus, minus)
    joined = torch.cat((plus_credit.flatten(), minus_credit.flatten()))
    assert abs(float(joined.mean())) < 1.0e-12
    assert abs(float(joined.std()) - 1.0) < 1.0e-12
    assert scale["paired_std"] > 0.0


def test_pair_direction_is_exact_average() -> None:
    generator = torch.Generator().manual_seed(72)
    features_plus = torch.randn(5, 3, 32, generator=generator)
    features_minus = torch.randn(5, 3, 32, generator=generator)
    latent = torch.randn(5, 3, 2, generator=generator) * 0.35
    raw_plus = torch.randn(5, 3, generator=generator)
    raw_minus = torch.randn(5, 3, generator=generator)
    cp, cm, _ = paired_standardized_credit(raw_plus, raw_minus)
    expected = 0.5 * (
        analytical_head_ascent64(features_plus, latent, cp)
        + analytical_head_ascent64(features_minus, -latent, cm)
    )
    actual, _ = paired_direction(
        features_plus,
        latent,
        raw_plus,
        features_minus,
        -latent,
        raw_minus,
    )
    assert actual.shape == (3, HEAD_DIM)
    assert torch.equal(actual, expected)


def test_projection_and_cosine() -> None:
    metric = torch.tensor([1.0, 0.0])
    component = torch.tensor([-0.4, 0.0])
    assert projection(component, metric) == pytest.approx(-0.4)
    assert cosine(component, metric) == pytest.approx(-1.0)


def test_two_level_bootstrap_resamples_seed_and_environment() -> None:
    metric = torch.ones(5, 64, 66, dtype=torch.float64)
    component = torch.stack(
        [torch.full((64, 66), float(index + 1), dtype=torch.float64) for index in range(5)]
    )
    report = stratified_bootstrap_projection(component, metric, seed=7200, draws=128)
    assert report["point"] == pytest.approx(3.0)
    assert report["p025"] < report["p975"]
