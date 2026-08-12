from __future__ import annotations

import hashlib

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch import nn

from cwi_x2.phase68_residual_ppo import (
    KNEE_ACTION_INDICES,
    PhysicalKneeResidualVecEnv,
    ResidualActorCritic,
    build_seeded_residual,
    deployable_moving_mask,
)
from cwi_x2.phase_conditioned_knee_residual import (
    PhaseConditionedKneeTargetResidual,
)


class FakeSourcePolicy(nn.Module):
    def __init__(self, action_bias: float = 0.0) -> None:
        super().__init__()
        self.actor = nn.Linear(93, 15)
        self.critic = nn.Linear(93, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.constant_(self.actor.bias, action_bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)
        self.std = nn.Parameter(torch.full((15,), 0.4))

    def act_inference(self, obs) -> torch.Tensor:
        return self.actor(obs["policy"])

    def evaluate(self, obs) -> torch.Tensor:
        return self.critic(obs["critic"])


def observations(batch: int, *, contacts=(1.0, 1.0)) -> TensorDict:
    policy = torch.randn(batch, 93)
    policy[:, -4:] = torch.tensor([0.0, 1.0, *contacts])
    return TensorDict(
        {"policy": policy, "critic": policy.clone()},
        batch_size=[batch],
    )


def module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_phase68_mask_accepts_only_three_deployable_contact_codes():
    obs = observations(7)["policy"]
    obs[:, -2:] = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.6, 0.6],
            [float("nan"), 1.0],
            [float("inf"), 0.0],
        ]
    )
    assert torch.equal(
        deployable_moving_mask(obs),
        torch.tensor([True, True, True, False, False, False, False]),
    )


def test_phase68_policy_is_two_dimensional_and_matches_deployable_mean():
    residual = build_seeded_residual()
    policy = ResidualActorCritic(FakeSourcePolicy(), residual)
    obs = observations(8)
    policy.update_distribution(obs)
    latent = policy.act(obs)
    assert latent.shape == (8, 2)
    assert policy.action_mean.shape == (8, 2)
    assert policy.action_std.shape == (8, 2)
    assert policy.get_actions_log_prob(latent).shape == (8,)
    assert torch.equal(policy.act_inference(obs), torch.zeros(8, 2))
    assert torch.equal(
        policy.physical_target_offset_from_latent(policy.act_inference(obs), obs),
        residual(obs["policy"]),
    )
    assert policy.trainable_manifest()["parameters"] == 4130
    assert all(not parameter.requires_grad for parameter in policy.source_policy.parameters())


def test_phase68_stock_ppo_performs_exactly_one_real_residual_step():
    residual = build_seeded_residual()
    policy = ResidualActorCritic(FakeSourcePolicy(), residual)
    algorithm = PPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=1.0,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=None,
        device="cpu",
    )
    algorithm.optimizer = torch.optim.Adam(residual.parameters(), lr=1.0e-3)
    obs = observations(16)
    algorithm.init_storage("rl", 16, 8, obs, [2])
    source_before = module_hash(policy.source_policy)
    encoder_before = module_hash(residual.encoder)
    head_before = module_hash(residual.head)
    for _ in range(8):
        action = algorithm.act(obs)
        reward = -((action[:, 0] - 0.5).square() + action[:, 1].square())
        done = torch.zeros(16, dtype=torch.long)
        algorithm.process_env_step(obs, reward, done, {})
    algorithm.storage.compute_returns(torch.zeros(16, 1), gamma=0.99, lam=1.0)
    steps = {"count": 0}
    original_step = algorithm.optimizer.step

    def counted_step(*args, **kwargs):
        steps["count"] += 1
        return original_step(*args, **kwargs)

    algorithm.optimizer.step = counted_step
    loss = algorithm.update()
    assert steps["count"] == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in loss.values())
    assert module_hash(policy.source_policy) == source_before
    assert module_hash(residual.encoder) == encoder_before
    assert module_hash(residual.head) != head_before


def test_phase68_seeded_residual_is_reproducible_without_consuming_rng():
    torch.manual_seed(17)
    state_before = torch.random.get_rng_state().clone()
    first = build_seeded_residual()
    assert torch.equal(torch.random.get_rng_state(), state_before)
    second = build_seeded_residual()
    assert module_hash(first) == module_hash(second)


class FakeTerm:
    def __init__(self, num_envs: int) -> None:
        self._processed_actions = torch.zeros(num_envs, 15)
        self.cfg = type("Cfg", (), {"clip": None})()

    def process_actions(self, actions: torch.Tensor) -> None:
        self._processed_actions = actions.clone()


class FakeInner:
    def __init__(self, obs: TensorDict, term: FakeTerm) -> None:
        self.num_envs = obs.batch_size[0]
        self.device = torch.device("cpu")
        self.max_episode_length = 1000
        self.cfg = type("Cfg", (), {})()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self._obs = obs
        self.unwrapped = type("Unwrapped", (), {})()
        self.unwrapped.action_manager = type("ActionManager", (), {})()
        self.unwrapped.action_manager._terms = {"joint_pos": term}

    def get_observations(self):
        return self._obs

    def step(self, actions: torch.Tensor):
        self.unwrapped.action_manager._terms["joint_pos"].process_actions(actions)
        target = self.unwrapped.action_manager._terms["joint_pos"]._processed_actions
        reward = target[:, list(KNEE_ACTION_INDICES)].sum(dim=-1)
        done = torch.zeros(self.num_envs, dtype=torch.long)
        return self._obs, reward, done, {}

    def close(self):
        return None


def test_phase68_vecenv_maps_received_latent_to_only_physical_knees():
    obs = observations(4)
    source = FakeSourcePolicy(action_bias=2.0)
    policy = ResidualActorCritic(source, PhaseConditionedKneeTargetResidual())
    term = FakeTerm(4)
    records = []
    env = PhysicalKneeResidualVecEnv(
        FakeInner(obs, term), policy, step_callback=records.append
    )
    latent = torch.tensor([[0.5, -0.5]]).repeat(4, 1)
    env.step(latent)
    expected = 0.003 * torch.tanh(latent)
    assert env.num_actions == 2
    assert torch.allclose(records[0]["requested_knee_offset"], expected)
    torch.testing.assert_close(
        records[0]["effective_knee_offset"], expected, rtol=1.0e-5, atol=1.0e-7
    )
    assert torch.equal(records[0]["non_knee_effective"], torch.zeros(4, 15))
    source_target = records[0]["source_target"]
    assert torch.equal(source_target, torch.ones_like(source_target))
    env.close()
