from __future__ import annotations

import torch

from tools.retarget.calibrate_x2_phase69_posthoc import (
    metric_summary,
    normalize,
    rsl_lambda_one_advantage,
)
from cwi_x2.phase69_reward_attribution import discounted_returns


def test_rsl_lambda_one_recurrence_matches_manual_zero_bootstrap():
    reward = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    value = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    result = rsl_lambda_one_advantage(reward, value, gamma=0.5)
    expected_return = torch.tensor([[3.75, 5.50], [5.50, 7.00], [5.00, 6.00]])
    torch.testing.assert_close(result, expected_return - value)


def test_rsl_lambda_one_recurrence_adds_terminal_value():
    reward = torch.zeros(3, 2)
    value = torch.zeros(3, 2)
    terminal = torch.tensor([8.0, 16.0])
    result = rsl_lambda_one_advantage(
        reward, value, gamma=0.5, last_value=terminal
    )
    expected = torch.tensor([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
    torch.testing.assert_close(result, expected)


def test_normalize_is_zero_mean_unit_sample_std():
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    result = normalize(value)
    assert abs(float(result.mean())) < 1.0e-7
    assert abs(float(result.std()) - 1.0) < 1.0e-7


def test_metric_summary_scaling_changes_projection_not_direction():
    reference = torch.tensor([1.0, -2.0, 3.0])
    scaled = 0.2 * reference
    summary = metric_summary(scaled, reference)
    assert summary["cosine_with_metric_gradient"] > 0.999999
    assert abs(summary["projection_on_metric_gradient"] - 0.2) < 1.0e-7


def test_rsl_lambda_one_matches_discounted_return_minus_value_float32():
    generator = torch.Generator().manual_seed(699042)
    reward = torch.randn(11, 5, generator=generator)
    value = torch.randn(11, 5, generator=generator)
    terminal = torch.randn(5, generator=generator)
    zero = rsl_lambda_one_advantage(reward, value)
    with_terminal = rsl_lambda_one_advantage(reward, value, last_value=terminal)
    torch.testing.assert_close(
        zero,
        discounted_returns(reward) - value,
        rtol=2.0e-6,
        atol=5.0e-6,
    )
    torch.testing.assert_close(
        with_terminal,
        discounted_returns(reward, terminal_bootstrap=terminal) - value,
        rtol=2.0e-6,
        atol=5.0e-6,
    )


def test_centered_component_and_baseline_credit_closure():
    generator = torch.Generator().manual_seed(699043)
    first = torch.randn(13, 4, generator=generator)
    second = torch.randn(13, 4, generator=generator)
    value = torch.randn(13, 4, generator=generator)
    total_advantage = rsl_lambda_one_advantage(first + second, value)
    denominator = total_advantage.std() + 1.0e-8
    component_first = rsl_lambda_one_advantage(first, torch.zeros_like(value))
    component_second = rsl_lambda_one_advantage(second, torch.zeros_like(value))
    baseline = rsl_lambda_one_advantage(torch.zeros_like(first), value)
    reconstructed = sum(
        (item - item.mean()) / denominator
        for item in (component_first, component_second, baseline)
    )
    torch.testing.assert_close(
        reconstructed,
        normalize(total_advantage),
        rtol=3.0e-6,
        atol=6.0e-6,
    )
