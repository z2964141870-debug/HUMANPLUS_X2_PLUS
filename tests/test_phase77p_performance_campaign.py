from __future__ import annotations

import copy

import pytest
import torch

from cwi_x2.phase77p_performance_campaign import (
    AnchoredUpdateLimits,
    anchored_ppo_update,
    clone_frozen_source,
    diagonal_fixed_std_kl,
    inject_actor_lora_train_critic,
    posture_reward_weight,
    source_retention,
    state_hash,
    trainable_parameter_groups,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 2),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 1),
        )
        self.std = torch.nn.Parameter(torch.full((2,), 0.4))
        self.is_recurrent = False
        self.distribution = None

    @staticmethod
    def _policy_obs(obs):
        return obs["policy"] if isinstance(obs, dict) else obs

    @staticmethod
    def _critic_obs(obs):
        return obs["critic"] if isinstance(obs, dict) else obs

    def update_distribution(self, obs) -> None:
        mean = self.actor(self._policy_obs(obs))
        self.distribution = torch.distributions.Normal(mean, mean * 0.0 + self.std)

    def act(self, obs, **_) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        return self.actor(self._policy_obs(obs))

    def evaluate(self, obs, **_) -> torch.Tensor:
        return self.critic(self._critic_obs(obs))

    def get_actions_log_prob(self, actions) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(-1)


class FakeStorage:
    def __init__(self, batch) -> None:
        self.batch = batch
        self.cleared = False

    def mini_batch_generator(self, _mini_batches, _epochs):
        yield self.batch

    def clear(self) -> None:
        self.cleared = True


class FakeAlgorithm:
    def __init__(self, policy, optimizer, storage) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.storage = storage
        self.num_mini_batches = 1
        self.num_learning_epochs = 1
        self.normalize_advantage_per_mini_batch = False
        self.clip_param = 0.2
        self.use_clipped_value_loss = True
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.0
        self.max_grad_norm = 1.0
        self.rnd = None
        self.symmetry = None


def test_actor_lora_is_zero_output_and_critic_is_dense_trainable() -> None:
    torch.manual_seed(7)
    policy = TinyPolicy()
    source = copy.deepcopy(policy).eval()
    obs = {"policy": torch.randn(11, 5), "critic": torch.randn(11, 5)}
    actor_before = source.act_inference(obs)
    manifest = inject_actor_lora_train_critic(policy, rank=4, alpha=4.0)

    assert torch.equal(policy.act_inference(obs), actor_before)
    assert manifest["actor_trainable_parameters"] > 0
    assert manifest["critic_trainable_parameters"] > 0
    assert not policy.std.requires_grad
    assert all(
        parameter.requires_grad
        for name, parameter in policy.named_parameters()
        if name.startswith("critic.")
    )
    assert all(
        ("lora_" in name) == parameter.requires_grad
        for name, parameter in policy.named_parameters()
        if name.startswith("actor.")
    )


def test_parameter_groups_are_disjoint_and_use_independent_rates() -> None:
    policy = TinyPolicy()
    inject_actor_lora_train_critic(policy)
    groups = trainable_parameter_groups(policy, actor_lr=1e-5, critic_lr=1e-4)
    assert [group["name"] for group in groups] == ["actor_lora", "critic_dense"]
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))


def test_kl_retention_and_reward_schedule() -> None:
    mean = torch.tensor([[0.0, 0.4], [0.4, 0.0]])
    source = torch.zeros_like(mean)
    kl = diagonal_fixed_std_kl(mean, source, torch.tensor([0.4, 0.4]))
    assert torch.allclose(kl, torch.tensor([0.5, 0.5]))
    assert [posture_reward_weight(index) for index in (1, 5, 6, 10)] == [-0.5] * 4
    with pytest.raises(ValueError):
        posture_reward_weight(0)
    with pytest.raises(ValueError):
        posture_reward_weight(11)


def test_anchored_update_changes_only_trainable_candidate_state() -> None:
    torch.manual_seed(17)
    dense = TinyPolicy()
    source = clone_frozen_source(dense)
    source_before = state_hash(source)
    candidate = copy.deepcopy(dense)
    inject_actor_lora_train_critic(candidate)
    groups = trainable_parameter_groups(candidate, actor_lr=1e-5, critic_lr=1e-4)
    optimizer = torch.optim.Adam(groups)
    obs = {"policy": torch.randn(32, 5), "critic": torch.randn(32, 5)}
    with torch.no_grad():
        actions = candidate.act(obs).detach()
        old_log_prob = candidate.get_actions_log_prob(actions).detach().unsqueeze(-1)
        old_mu = candidate.action_mean.detach()
        old_sigma = candidate.action_std.detach()
        target_values = candidate.evaluate(obs).detach()
    batch = (
        obs,
        actions,
        target_values,
        torch.randn(32, 1),
        target_values + 0.25 * torch.randn(32, 1),
        old_log_prob,
        old_mu,
        old_sigma,
        (None, None),
        None,
    )
    storage = FakeStorage(batch)
    algorithm = FakeAlgorithm(candidate, optimizer, storage)
    before = {
        name: parameter.detach().clone()
        for name, parameter in candidate.named_parameters()
    }
    result = anchored_ppo_update(
        algorithm,
        source,
        source_kl_coefficient=0.10,
        limits=AnchoredUpdateLimits(
            source_kl_mean_max=1.0,
            source_kl_max=2.0,
            incremental_kl_mean_max=1.0,
            incremental_kl_max=2.0,
            action_drift_max=2.0,
        ),
    )

    assert result["optimizer_steps"] == 1
    assert result["finite"] is True
    assert storage.cleared
    assert state_hash(source) == source_before
    changed = {
        name
        for name, parameter in candidate.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert changed
    assert all(dict(candidate.named_parameters())[name].requires_grad for name in changed)
    retention = source_retention(candidate, source, obs)
    assert retention["finite"] is True
    assert retention["source_kl_mean"] >= 0.0


def test_post_step_trust_violation_restores_parameters_and_optimizer() -> None:
    torch.manual_seed(29)
    dense = TinyPolicy()
    source = clone_frozen_source(dense)
    candidate = copy.deepcopy(dense)
    inject_actor_lora_train_critic(candidate)
    optimizer = torch.optim.Adam(
        trainable_parameter_groups(candidate, actor_lr=0.5, critic_lr=0.5)
    )
    obs = {"policy": torch.randn(32, 5), "critic": torch.randn(32, 5)}
    with torch.no_grad():
        actions = candidate.act(obs).detach()
        old_log_prob = candidate.get_actions_log_prob(actions).detach().unsqueeze(-1)
        old_mu = candidate.action_mean.detach()
        old_sigma = candidate.action_std.detach()
        target_values = candidate.evaluate(obs).detach()
    batch = (
        obs,
        actions,
        target_values,
        torch.randn(32, 1),
        target_values + torch.randn(32, 1),
        old_log_prob,
        old_mu,
        old_sigma,
        (None, None),
        None,
    )
    algorithm = FakeAlgorithm(candidate, optimizer, FakeStorage(batch))
    before = {
        name: parameter.detach().clone()
        for name, parameter in candidate.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(RuntimeError, match="rolled back"):
        anchored_ppo_update(
            algorithm,
            source,
            source_kl_coefficient=0.10,
            limits=AnchoredUpdateLimits(
                source_kl_mean_max=1e-12,
                source_kl_max=1e-12,
                incremental_kl_mean_max=1e-12,
                incremental_kl_max=1e-12,
                action_drift_max=1e-12,
            ),
        )
    assert all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in candidate.named_parameters()
    )
    assert optimizer.state_dict()["state"] == optimizer_before["state"]
