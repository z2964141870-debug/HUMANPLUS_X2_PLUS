from __future__ import annotations

import copy

import pytest
import torch

from cwi_x2.native_full_actor_pilot import (
    FullActorUpdateLimits,
    PilotSpec,
    ROLE_NAMES,
    configure_full_actor_fresh_critic,
    full_actor_ppo_update,
    module_state_hash,
    objective_weights,
    optimizer_parameter_groups,
    role_count_dict,
    role_ids,
)


class TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 3),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 1),
        )
        self.std = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4]))
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
        self.schedule = "fixed"
        self.rnd = None
        self.symmetry = None


def test_default_pilot_budget_is_real_training_budget() -> None:
    spec = PilotSpec()
    assert spec.transitions == 614_400
    assert spec.optimizer_steps == 200
    assert spec.template_scale == 0.15
    assert spec.to_dict()["num_envs"] == 512


def test_command_roles_are_balanced_and_eval_roles_are_paired() -> None:
    training = role_ids(512)
    evaluation = role_ids(512, paired=True)
    assert set(training.tolist()) == set(range(len(ROLE_NAMES)))
    assert set(role_count_dict(512).values()) == {64}
    assert set(role_count_dict(512, paired=True).values()) == {64}
    assert torch.equal(evaluation[0::2], evaluation[1::2])
    with pytest.raises(ValueError):
        role_ids(510)
    with pytest.raises(ValueError):
        role_ids(504, paired=True)


def test_posture_curriculum_has_only_three_predeclared_levels() -> None:
    assert objective_weights(1).signed_backward_pitch == -0.5
    assert objective_weights(5).actual_support_com == -0.5
    assert objective_weights(6).signed_backward_pitch == -1.0
    assert objective_weights(15).actual_support_com == -1.0
    assert objective_weights(16).signed_backward_pitch == -1.5
    assert objective_weights(25).actual_support_com == -1.5
    with pytest.raises(ValueError):
        objective_weights(0)
    with pytest.raises(ValueError):
        objective_weights(26)


def test_full_actor_is_preserved_critic_is_fresh_and_std_is_frozen() -> None:
    torch.manual_seed(5)
    policy = TinyActorCritic()
    actor_before = module_state_hash(policy.actor)
    critic_before = module_state_hash(policy.critic)
    std_before = policy.std.detach().clone()
    manifest = configure_full_actor_fresh_critic(policy, critic_seed=780101)

    assert module_state_hash(policy.actor) == actor_before
    assert module_state_hash(policy.critic) != critic_before
    assert torch.equal(policy.std.detach(), std_before)
    assert all(parameter.requires_grad for parameter in policy.actor.parameters())
    assert all(parameter.requires_grad for parameter in policy.critic.parameters())
    assert not policy.std.requires_grad
    assert manifest["initialization"] == "stage219_actor_warm_start_fresh_critic"


def test_critic_reset_is_reproducible_and_seed_specific() -> None:
    torch.manual_seed(9)
    source = TinyActorCritic()
    first = copy.deepcopy(source)
    second = copy.deepcopy(source)
    third = copy.deepcopy(source)
    configure_full_actor_fresh_critic(first, critic_seed=780101)
    configure_full_actor_fresh_critic(second, critic_seed=780101)
    configure_full_actor_fresh_critic(third, critic_seed=780102)
    assert module_state_hash(first.critic) == module_state_hash(second.critic)
    assert module_state_hash(first.critic) != module_state_hash(third.critic)


def test_optimizer_groups_cover_only_trainable_actor_and_critic() -> None:
    policy = TinyActorCritic()
    configure_full_actor_fresh_critic(policy, critic_seed=4)
    groups = optimizer_parameter_groups(
        policy,
        actor_learning_rate=3.0e-5,
        critic_learning_rate=2.0e-4,
    )
    assert [group["name"] for group in groups] == ["actor_full", "critic_fresh"]
    assert [group["lr"] for group in groups] == [3.0e-5, 2.0e-4]
    identifiers = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identifiers) == len(set(identifiers))
    assert id(policy.std) not in identifiers


def test_full_actor_update_changes_actor_and_fresh_critic_but_not_source_or_std() -> None:
    torch.manual_seed(17)
    dense = TinyActorCritic()
    source = copy.deepcopy(dense).eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    source_before = module_state_hash(source)
    candidate = copy.deepcopy(dense)
    configure_full_actor_fresh_critic(candidate, critic_seed=780101)
    actor_before = module_state_hash(candidate.actor)
    critic_before = module_state_hash(candidate.critic)
    std_before = candidate.std.detach().clone()
    groups = optimizer_parameter_groups(
        candidate,
        actor_learning_rate=1.0e-5,
        critic_learning_rate=5.0e-5,
    )
    optimizer = torch.optim.Adam(groups)
    obs = {"policy": torch.randn(64, 5), "critic": torch.randn(64, 5)}
    with torch.no_grad():
        actions = candidate.act(obs).detach()
        old_log_prob = candidate.get_actions_log_prob(actions).detach().unsqueeze(-1)
        old_mean = candidate.action_mean.detach()
        old_std = candidate.action_std.detach()
        target_value = candidate.evaluate(obs).detach()
    batch = (
        obs,
        actions,
        target_value,
        torch.randn(64, 1),
        target_value + 0.2 * torch.randn(64, 1),
        old_log_prob,
        old_mean,
        old_std,
        (None, None),
        None,
    )
    storage = FakeStorage(batch)
    algorithm = FakeAlgorithm(candidate, optimizer, storage)
    result = full_actor_ppo_update(
        algorithm,
        source,
        limits=FullActorUpdateLimits(
            source_kl_mean_max=10.0,
            source_kl_max=10.0,
            incremental_kl_mean_max=10.0,
            incremental_kl_max=10.0,
            action_drift_max=10.0,
        ),
    )
    assert result["optimizer_steps"] == 1
    assert result["finite"]
    assert storage.cleared
    assert module_state_hash(candidate.actor) != actor_before
    assert module_state_hash(candidate.critic) != critic_before
    assert module_state_hash(source) == source_before
    assert torch.equal(candidate.std.detach(), std_before)


def test_full_actor_update_rolls_back_rejected_adam_step() -> None:
    torch.manual_seed(23)
    dense = TinyActorCritic()
    source = copy.deepcopy(dense).eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    candidate = copy.deepcopy(dense)
    configure_full_actor_fresh_critic(candidate, critic_seed=780102)
    candidate_before = module_state_hash(candidate)
    groups = optimizer_parameter_groups(
        candidate,
        actor_learning_rate=0.1,
        critic_learning_rate=0.1,
    )
    optimizer = torch.optim.Adam(groups)
    obs = {"policy": torch.randn(64, 5), "critic": torch.randn(64, 5)}
    with torch.no_grad():
        actions = candidate.act(obs).detach()
        old_log_prob = candidate.get_actions_log_prob(actions).detach().unsqueeze(-1)
        old_mean = candidate.action_mean.detach()
        old_std = candidate.action_std.detach()
        target_value = candidate.evaluate(obs).detach()
    batch = (
        obs,
        actions,
        target_value,
        torch.randn(64, 1),
        target_value + torch.randn(64, 1),
        old_log_prob,
        old_mean,
        old_std,
        (None, None),
        None,
    )
    algorithm = FakeAlgorithm(candidate, optimizer, FakeStorage(batch))
    with pytest.raises(RuntimeError, match="rolled back"):
        full_actor_ppo_update(
            algorithm,
            source,
            limits=FullActorUpdateLimits(
                source_kl_mean_max=1.0e-12,
                source_kl_max=1.0e-12,
                incremental_kl_mean_max=1.0e-12,
                incremental_kl_max=1.0e-12,
                action_drift_max=1.0e-12,
            ),
        )
    assert module_state_hash(candidate) == candidate_before
