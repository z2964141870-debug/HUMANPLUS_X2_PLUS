from __future__ import annotations

import torch

from cwi_x2.phase69_reward_attribution import (
    additive_normalized_credits,
    analytical_head_ascent_per_env,
    bootstrap_projection,
    discounted_returns,
    semantic_phase_ids,
)
from tools.retarget.finalize_x2_phase69_reward_attribution import (
    classify_alignment,
    stable_sign,
)


def test_discounted_returns_uses_zero_bootstrap():
    reward = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = discounted_returns(reward, gamma=0.5)
    expected = torch.tensor([[3.75, 5.50], [5.50, 7.00], [5.00, 6.00]])
    torch.testing.assert_close(result, expected)


def test_additive_credits_reconstruct_normalized_advantage():
    generator = torch.Generator().manual_seed(69)
    rewards = {
        "locomotion": torch.randn(8, 5, generator=generator),
        "pitch": torch.randn(8, 5, generator=generator),
        "support": torch.randn(8, 5, generator=generator),
    }
    values = torch.randn(8, 5, generator=generator)
    credits, raw, normalized = additive_normalized_credits(rewards, values)
    torch.testing.assert_close(sum(credits.values()), normalized, rtol=1.0e-6, atol=1.0e-6)
    expected_raw = sum(discounted_returns(value) for value in rewards.values()) - values
    torch.testing.assert_close(raw, expected_raw)

    bootstrap = torch.randn(5, generator=generator)
    boot_credits, boot_raw, boot_normalized = additive_normalized_credits(
        rewards, values, terminal_bootstrap=bootstrap
    )
    expected_boot_raw = expected_raw + discounted_returns(
        torch.zeros_like(values), terminal_bootstrap=bootstrap
    )
    torch.testing.assert_close(boot_raw, expected_boot_raw)
    torch.testing.assert_close(
        sum(boot_credits.values()), boot_normalized, rtol=1.0e-6, atol=1.0e-6
    )


def test_analytical_head_ascent_matches_autograd_and_phase_adds():
    generator = torch.Generator().manual_seed(690)
    time, envs, hidden = 7, 4, 3
    features = torch.randn(time, envs, hidden, generator=generator)
    latent = torch.randn(time, envs, 2, generator=generator) * 0.35
    credit = torch.randn(time, envs, generator=generator)
    weight = torch.zeros(2, hidden, requires_grad=True)
    bias = torch.zeros(2, requires_grad=True)
    mean = torch.einsum("tnh,ah->tna", features, weight) + bias
    distribution = torch.distributions.Normal(mean, 0.35)
    old_log_prob = torch.distributions.Normal(torch.zeros_like(mean), 0.35).log_prob(latent).sum(-1)
    ratio = torch.exp(distribution.log_prob(latent).sum(-1) - old_log_prob)
    loss = -(credit * ratio).mean()
    gradient = torch.autograd.grad(loss, (weight, bias))
    autograd_ascent = -torch.cat((gradient[0].flatten(), gradient[1]))
    per_env = analytical_head_ascent_per_env(features, latent, credit)
    torch.testing.assert_close(per_env.mean(dim=0), autograd_ascent)

    phase = torch.arange(time).unsqueeze(1).expand(time, envs) % 4
    phase_sum = sum(
        analytical_head_ascent_per_env(features, latent, credit, sample_mask=phase == index)
        for index in range(4)
    )
    torch.testing.assert_close(phase_sum, per_env)


def test_semantic_phase_ids_are_fail_closed():
    observations = torch.zeros(6, 93)
    observations[:, -4:] = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.6, 0.6],
            [float("nan"), 1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(semantic_phase_ids(observations), torch.tensor([0, 1, 2, 3, -1, -1]))


def test_bootstrap_projection_is_deterministic_and_signed():
    generator = torch.Generator().manual_seed(691)
    total = torch.randn(16, 9, generator=generator)
    component = 0.75 * total
    first = bootstrap_projection(component, total, draws=128)
    second = bootstrap_projection(component, total, draws=128)
    assert first == second
    assert abs(first["point"] - 0.75) < 1.0e-8
    assert first["bootstrap_same_sign_fraction"] == 1.0
    assert first["leave_one_env_out_same_sign_fraction"] == 1.0


def alignment(point: float, low: float, high: float) -> dict[str, float]:
    return {
        "point": point,
        "bootstrap_p025": low,
        "bootstrap_p975": high,
        "bootstrap_same_sign_fraction": 0.99,
        "leave_one_env_out_same_sign_fraction": 0.95,
    }


def test_phase69_finalizer_decision_tree_is_fail_closed():
    positive = alignment(0.5, 0.1, 0.9)
    negative = alignment(-0.5, -0.9, -0.1)
    uncertain = alignment(0.1, -0.2, 0.3)
    assert stable_sign(positive) == 1
    assert stable_sign(negative) == -1
    assert stable_sign(uncertain) == 0
    assert (
        classify_alignment(negative, positive)
        == "FAIL_REWARD_CREDIT_DIRECTION_STOP"
    )
    assert (
        classify_alignment(positive, negative)
        == "PASS_REWARD_CONFLICT_DIAG_ONLY"
    )
    assert (
        classify_alignment(positive, positive)
        == "FAIL_STOCHASTIC_TO_MEAN_OR_ADAM_MISMATCH_STOP"
    )
    assert classify_alignment(positive, uncertain) == "INCONCLUSIVE_VARIANCE_STOP"
