from __future__ import annotations

import torch

from cwi_x2.phase70_long_lookahead import (
    anchor_returns,
    environment_influence,
    normalized_advantage,
    standalone_credit,
    vector_comparison,
)


def test_anchor_returns_uses_later_rewards_but_returns_only_anchor_scores():
    reward = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    result = anchor_returns(reward, anchor_steps=2, gamma=0.5)
    torch.testing.assert_close(result, torch.tensor([[4.0], [6.0]]))


def test_terminal_bootstrap_exponent_is_horizon_minus_action_index():
    reward = torch.zeros(4, 1)
    result = anchor_returns(
        reward,
        anchor_steps=2,
        gamma=0.5,
        terminal_bootstrap=torch.tensor([16.0]),
    )
    torch.testing.assert_close(result, torch.tensor([[1.0], [2.0]]))


def test_normalized_advantage_uses_anchor_values_only():
    reward = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    values = torch.ones(6, 4)
    raw, normalized = normalized_advantage(reward, values, anchor_steps=3)
    torch.testing.assert_close(raw, anchor_returns(reward, anchor_steps=3) - 1.0)
    assert abs(float(normalized.mean())) < 1.0e-6
    assert abs(float(normalized.std()) - 1.0) < 1.0e-6


def test_standalone_credit_is_scale_invariant_for_positive_scale():
    reward = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    torch.testing.assert_close(
        standalone_credit(reward, anchor_steps=4),
        standalone_credit(reward * 7.0, anchor_steps=4),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_vector_comparison_and_environment_influence():
    reference = torch.tensor([1.0, -2.0, 3.0])
    summary = vector_comparison(0.5 * reference, reference)
    assert summary["cosine"] > 0.999999
    assert abs(summary["projection_on_reference"] - 0.5) < 1.0e-7
    influence = environment_influence(torch.stack((reference, reference)))
    assert abs(influence["kish_effective_sample_size"] - 2.0) < 1.0e-7
    assert abs(influence["maximum_absolute_env_contribution_fraction"] - 0.5) < 1.0e-7
