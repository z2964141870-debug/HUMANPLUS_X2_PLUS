from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv
from humanoidverse.x2_scratch import (
    ACTION_DIM,
    STATE_DIM,
    X2_ALL_AUX_REWARD_NAMES,
    X2_AUX_REWARD_SCALES,
    X2_JOINTS_31,
)


class FakeRewardManager:
    def __init__(self, num_envs: int):
        self._term_names = list(X2_AUX_REWARD_SCALES)
        self._weights = {
            name: float(index + 1) for index, name in enumerate(self._term_names)
        }
        raw = torch.arange(1, num_envs + 1, dtype=torch.float32).reshape(-1, 1)
        weights = torch.tensor(list(self._weights.values()), dtype=torch.float32)
        self._step_reward = raw * weights

    def get_term_cfg(self, name: str):
        return SimpleNamespace(weight=self._weights[name])


class FakeWrapped:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.episode_length_buf.zero_()
        return {"policy": torch.zeros(self.env.num_envs, 1)}

    def step(self, action):
        self.env.episode_length_buf += 1
        self.env.scene["robot"].data.joint_pos[:, :ACTION_DIM] += action * 0.01
        reward = torch.ones(self.env.num_envs)
        done = torch.zeros(self.env.num_envs, dtype=torch.bool)
        extras = {"time_outs": torch.zeros_like(done)}
        return {"policy": torch.zeros(self.env.num_envs, 1)}, reward, done, extras


class FakeDoneWrapped(FakeWrapped):
    def step(self, action):
        result = list(super().step(action))
        result[2][0] = True
        self.env.scene["robot"].data.joint_pos[0].zero_()
        self.env.episode_length_buf[0] = 0
        return tuple(result)


def fake_env(num_envs: int = 3):
    data = SimpleNamespace(
        joint_pos=torch.zeros(num_envs, 31),
        default_joint_pos=torch.zeros(num_envs, 31),
        joint_vel=torch.zeros(num_envs, 31),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
        root_ang_vel_b=torch.zeros(num_envs, 3),
        root_pos_w=torch.zeros(num_envs, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
        root_lin_vel_w=torch.zeros(num_envs, 3),
        root_lin_vel_b=torch.zeros(num_envs, 3),
        root_ang_vel_w=torch.zeros(num_envs, 3),
    )
    robot = SimpleNamespace(data=data, joint_names=list(X2_JOINTS_31))
    command = torch.tensor([[0.35, 0.0, 0.0]]).repeat(num_envs, 1)
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": robot},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        step_dt=0.02,
        command_manager=SimpleNamespace(
            get_command=lambda name: command
            if name == "base_velocity"
            else (_ for _ in ()).throw(KeyError(name))
        ),
        reward_manager=FakeRewardManager(num_envs),
    )


def test_reset_and_step_contract() -> None:
    env = fake_env()
    adapter = X2IsaacLabVectorEnv(env, FakeWrapped(env), history_length=4)
    observation, info = adapter.reset(seed=12)
    assert observation["state"].shape == (3, STATE_DIM)
    assert observation["history_actor"].shape == (3, 4 * (STATE_DIM + ACTION_DIM))
    assert observation["time"].shape == (3, 1)
    assert info["qpos"].shape == (3, 38)
    action = np.full((3, ACTION_DIM), 0.25, dtype=np.float32)
    next_observation, reward, terminated, truncated, info = adapter.step(action)
    assert reward.shape == (3,)
    assert not terminated.any()
    assert not truncated.any()
    np.testing.assert_allclose(next_observation["last_action"], action)
    np.testing.assert_array_equal(next_observation["time"], 1)
    assert set(info["aux_rewards"]) == set(X2_ALL_AUX_REWARD_NAMES)
    for name, value in info["aux_rewards"].items():
        if name == "locomotion_total_reward":
            np.testing.assert_array_equal(value, 1.0)
            continue
        if name == "termination":
            np.testing.assert_array_equal(value, 0.0)
            continue
        if name == "action_magnitude_l2":
            np.testing.assert_array_equal(value, ACTION_DIM * 0.25**2)
            continue
        np.testing.assert_array_equal(value, [1.0, 2.0, 3.0])


def test_action_bounds_and_shape_fail_closed() -> None:
    env = fake_env()
    adapter = X2IsaacLabVectorEnv(env, FakeWrapped(env))
    adapter.reset()
    with pytest.raises(ValueError, match="shape"):
        adapter.step(np.zeros((3, ACTION_DIM + 1), dtype=np.float32))
    bad = np.zeros((3, ACTION_DIM), dtype=np.float32)
    bad[0, 0] = 1.01
    with pytest.raises(ValueError, match="exceeds"):
        adapter.step(bad)


def test_tensor_mode_stays_on_device() -> None:
    env = fake_env(2)
    adapter = X2IsaacLabVectorEnv(env, FakeWrapped(env), to_numpy=False)
    observation, _ = adapter.reset()
    assert all(isinstance(value, torch.Tensor) for value in observation.values())
    assert observation["state"].device.type == "cpu"


def test_phase_conditioned_adapter_emits_command_and_deployable_clock() -> None:
    env = fake_env(2)
    adapter = X2IsaacLabVectorEnv(
        env,
        FakeWrapped(env),
        to_numpy=False,
        include_command_phase=True,
    )
    observation, _ = adapter.reset()
    torch.testing.assert_close(
        observation["command"], torch.tensor([[0.35, 0.0, 0.0]]).repeat(2, 1)
    )
    torch.testing.assert_close(
        observation["gait_phase"],
        torch.tensor([[0.0, 1.0, 1.0, 1.0]]).repeat(2, 1),
    )
    observation, *_ = adapter.step(torch.zeros(2, ACTION_DIM))
    angle = 2.0 * np.pi / 40.0
    torch.testing.assert_close(
        observation["gait_phase"][:, :2],
        torch.tensor([[np.sin(angle), np.cos(angle)]], dtype=torch.float32).repeat(2, 1),
    )


def test_close_blocks_future_use() -> None:
    env = fake_env(1)
    adapter = X2IsaacLabVectorEnv(env, FakeWrapped(env))
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.reset()


def test_joint_order_is_reconstructed_by_name() -> None:
    env = fake_env(1)
    env.scene["robot"].joint_names[0:2] = reversed(
        env.scene["robot"].joint_names[0:2]
    )
    env.scene["robot"].data.joint_pos[0, 0] = 2.0
    env.scene["robot"].data.joint_pos[0, 1] = 1.0
    adapter = X2IsaacLabVectorEnv(env, FakeWrapped(env))
    observation, _ = adapter.reset()
    np.testing.assert_array_equal(observation["state"][0, :2], [1.0, 2.0])


def test_missing_joint_fails_closed() -> None:
    env = fake_env(1)
    env.scene["robot"].joint_names[-1] = "not_an_x2_joint"
    with pytest.raises(RuntimeError, match="joints differ"):
        X2IsaacLabVectorEnv(env, FakeWrapped(env))


def test_autoreset_lane_does_not_leak_history_or_action() -> None:
    env = fake_env(2)
    adapter = X2IsaacLabVectorEnv(env, FakeDoneWrapped(env), history_length=4)
    adapter.reset()
    action = np.full((2, ACTION_DIM), 0.25, dtype=np.float32)
    observation, _, terminated, _, info = adapter.step(action)
    assert terminated.tolist() == [True, False]
    np.testing.assert_array_equal(info["aux_rewards"]["termination"], [1.0, 0.0])
    np.testing.assert_array_equal(observation["last_action"][0], 0.0)
    history = observation["history_actor"][0].reshape(4, STATE_DIM + ACTION_DIM)
    np.testing.assert_array_equal(history[:-1], 0.0)
    np.testing.assert_array_equal(history[-1, -ACTION_DIM:], 0.0)
