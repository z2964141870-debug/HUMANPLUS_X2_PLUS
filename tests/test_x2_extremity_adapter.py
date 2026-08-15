from __future__ import annotations

import pytest
import torch
from torch import nn

from humanoidverse.x2_extremity_adapter import (
    ZeroPreservingResidualAdapter,
    build_x2_single_frame_adapter,
)


def test_adapter_reproduces_legacy_output_for_nonzero_goal() -> None:
    torch.manual_seed(7)
    legacy = nn.Sequential(nn.Linear(86, 32), nn.Tanh(), nn.Linear(32, 15))
    adapter = build_x2_single_frame_adapter(legacy, base_obs_dim=86, output_dim=15)
    base_obs = torch.randn(4, 86)
    goal = torch.randn(4, 36)
    with torch.no_grad():
        expected = legacy(base_obs)
        actual = adapter(base_obs, goal)
    assert torch.equal(actual, expected)


def test_adapter_preserves_legacy_parameters_and_learns_goal_branch() -> None:
    torch.manual_seed(11)
    legacy = nn.Linear(89, 1)
    before = {name: value.detach().clone() for name, value in legacy.named_parameters()}
    adapter = ZeroPreservingResidualAdapter(
        legacy,
        base_obs_dim=89,
        goal_dim=72,
        output_dim=1,
        hidden_dim=16,
    )
    base_obs = torch.randn(8, 89)
    goal = torch.randn(8, 72)
    initial = adapter(base_obs, goal).detach()
    loss = adapter(base_obs, goal).square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in legacy.parameters())
    assert adapter.goal_encoder[-1].weight.grad is not None
    optimizer = torch.optim.SGD(adapter.goal_encoder.parameters(), lr=0.1)
    optimizer.step()
    updated = adapter(base_obs, goal).detach()
    assert not torch.equal(updated, initial)
    for name, value in legacy.named_parameters():
        assert torch.equal(value, before[name])


def test_adapter_rejects_wrong_width_and_batch() -> None:
    adapter = build_x2_single_frame_adapter(nn.Linear(86, 15), base_obs_dim=86, output_dim=15)
    with pytest.raises(ValueError, match="base_obs"):
        adapter(torch.zeros(2, 85), torch.zeros(2, 36))
    with pytest.raises(ValueError, match="goal"):
        adapter(torch.zeros(2, 86), torch.zeros(2, 35))
    with pytest.raises(ValueError, match="batch"):
        adapter(torch.zeros(2, 86), torch.zeros(3, 36))

