from __future__ import annotations

import copy

import pytest
import torch

from cwi_x2.native_full_actor_pilot import (
    PilotSpec,
    ROLE_NAMES,
    configure_full_actor_fresh_critic,
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
