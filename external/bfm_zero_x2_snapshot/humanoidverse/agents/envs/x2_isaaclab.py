"""Gym-like BFM-Zero adapter for an existing X2 IsaacLab environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import torch

from humanoidverse.x2_scratch import (
    ACTION_DIM,
    STATE_DIM,
    X2_ALL_AUX_REWARD_NAMES,
    X2_JOINTS_31,
    X2_LOWER_JOINTS_15,
    X2ScratchDataConfig,
)
from humanoidverse.x2_scratch_model import x2_scratch_observation_space


class X2IsaacLabVectorEnv:
    """Expose an IsaacLab X2 ManagerBasedRLEnv using BFM's vector contract.

    ``env`` owns the robot/managers, while ``wrapped`` is the installed
    ``RslRlVecEnvWrapper`` used only for its stable reset/step interface.  The
    BFM observation is reconstructed from articulation state and therefore
    does not inherit Stage219's 93-D policy observation or weights.
    """

    def __init__(
        self,
        env: Any,
        wrapped: Any,
        *,
        history_length: int = 4,
        to_numpy: bool = True,
        include_command_phase: bool = False,
    ) -> None:
        self._env = env
        self._wrapped = wrapped
        self.config = X2ScratchDataConfig(history_length=history_length)
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.to_numpy = to_numpy
        self.include_command_phase = include_command_phase
        self._validate_live_contract()
        self.single_observation_space = x2_scratch_observation_space(
            self.config, include_command_phase=include_command_phase
        )
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space,
            n=self.num_envs,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space,
            n=self.num_envs,
        )
        self._history = torch.zeros(
            self.num_envs,
            self.config.history_length,
            STATE_DIM + ACTION_DIM,
            device=self.device,
        )
        self._last_action = torch.zeros(
            self.num_envs,
            ACTION_DIM,
            device=self.device,
        )
        self._closed = False

    def _validate_live_contract(self) -> None:
        robot = self._robot()
        joint_names = tuple(robot.joint_names)
        if len(joint_names) != len(set(joint_names)) or set(joint_names) != set(X2_JOINTS_31):
            raise RuntimeError(
                "X2 articulation joints differ from the scratch dataset: "
                f"{joint_names}"
            )
        # Isaac/URDF order is interleaved left/right/waist, while the expert
        # trajectories use the documented semantic X2_JOINTS_31 order.  Never
        # let a matching dimension silently stand in for matching semantics.
        self._joint_ids_31 = torch.tensor(
            [joint_names.index(name) for name in X2_JOINTS_31],
            dtype=torch.long,
            device=self.device,
        )
        action_manager = getattr(self._env, "action_manager", None)
        if action_manager is None:
            return
        term = action_manager._terms.get("joint_pos")
        if term is None or tuple(term._joint_names) != X2_LOWER_JOINTS_15:
            raise RuntimeError("X2 live action joint order differs from the scratch dataset")

    @property
    def unwrapped(self) -> Any:
        return self._env

    def _robot(self) -> Any:
        return self._env.scene["robot"]

    def _state(self) -> torch.Tensor:
        data = self._robot().data
        joint_pos = data.joint_pos.index_select(1, self._joint_ids_31)
        default_joint_pos = data.default_joint_pos.index_select(1, self._joint_ids_31)
        joint_vel = data.joint_vel.index_select(1, self._joint_ids_31)
        state = torch.cat(
            (
                joint_pos - default_joint_pos,
                joint_vel,
                data.projected_gravity_b,
                data.root_ang_vel_b,
                data.root_lin_vel_b,
            ),
            dim=-1,
        ).to(torch.float32)
        if state.shape != (self.num_envs, STATE_DIM):
            raise RuntimeError(f"X2 BFM state shape mismatch: {state.shape}")
        if not torch.isfinite(state).all():
            raise RuntimeError("X2 BFM state contains non-finite values")
        return state

    def _append_history(self, state: torch.Tensor) -> None:
        frame = torch.cat((state, self._last_action), dim=-1)
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1].copy_(frame)

    def _observation(self, *, append_history: bool) -> dict[str, Any]:
        state = self._state()
        if append_history:
            self._append_history(state)
        observation: dict[str, torch.Tensor] = {
            "state": state,
            "privileged_state": state.clone(),
            "last_action": self._last_action.clone(),
            "history_actor": self._history.flatten(1).clone(),
            "time": self._env.episode_length_buf.reshape(-1, 1).clone(),
        }
        if self.include_command_phase:
            command_manager = getattr(self._env, "command_manager", None)
            if command_manager is None:
                raise RuntimeError("phase-conditioned X2 observation requires commands")
            command = command_manager.get_command("base_velocity").to(torch.float32)
            moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.1
            phase = torch.remainder(
                self._env.episode_length_buf.to(torch.float32)
                * float(self._env.step_dt)
                / 0.8,
                1.0,
            )
            half_ds_width = 0.30 / 4.0
            right_swing = (phase >= half_ds_width) & (phase < 0.5 - half_ds_width)
            left_swing = (phase >= 0.5 + half_ds_width) & (phase < 1.0 - half_ds_width)
            desired_contact = torch.stack((~left_swing, ~right_swing), dim=-1)
            desired_contact = torch.where(
                moving[:, None], desired_contact, torch.ones_like(desired_contact)
            )
            angle = 2.0 * torch.pi * phase
            clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
            clock = clock * moving.to(clock.dtype)[:, None]
            gait_phase = torch.cat(
                (clock, desired_contact.to(clock.dtype)), dim=-1
            )
            if command.shape != (self.num_envs, 3) or gait_phase.shape != (
                self.num_envs,
                4,
            ):
                raise RuntimeError("phase-conditioned X2 observation shape mismatch")
            observation["command"] = command.clone()
            observation["gait_phase"] = gait_phase
        if self.to_numpy:
            return {
                key: value.detach().cpu().numpy()
                for key, value in observation.items()
            }
        return observation

    def _qpos_qvel(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._robot().data
        joint_pos = data.joint_pos.index_select(1, self._joint_ids_31)
        joint_vel = data.joint_vel.index_select(1, self._joint_ids_31)
        qpos = torch.cat((data.root_pos_w, data.root_quat_w, joint_pos), dim=-1)
        qvel = torch.cat((data.root_lin_vel_w, data.root_ang_vel_w, joint_vel), dim=-1)
        return qpos, qvel

    def _aux_rewards(
        self,
        action: torch.Tensor,
        terminated: torch.Tensor,
        total_reward: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recover unweighted X2 costs from IsaacLab's current reward table."""
        manager = getattr(self._env, "reward_manager", None)
        if manager is None:
            return {}
        term_names = tuple(manager._term_names)
        raw: dict[str, torch.Tensor] = {}
        for name in X2_ALL_AUX_REWARD_NAMES:
            if name == "locomotion_total_reward":
                value = total_reward.reshape(-1).detach().clone().to(torch.float32)
                if value.shape != (self.num_envs,) or not torch.isfinite(value).all():
                    raise RuntimeError("invalid X2 total locomotion reward")
                raw[name] = value
                continue
            if name == "termination":
                raw[name] = terminated.to(torch.float32)
                continue
            if name == "action_magnitude_l2":
                raw[name] = action.square().sum(dim=-1)
                continue
            if name not in term_names:
                raise RuntimeError(f"X2 auxiliary reward term is missing: {name}")
            index = term_names.index(name)
            weight = float(manager.get_term_cfg(name).weight)
            if weight == 0.0:
                raise RuntimeError(f"X2 auxiliary reward term has zero live weight: {name}")
            value = (manager._step_reward[:, index] / weight).detach().clone()
            if value.shape != (self.num_envs,) or not torch.isfinite(value).all():
                raise RuntimeError(f"invalid X2 auxiliary reward term: {name}")
            raw[name] = value
        return raw

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._closed:
            raise RuntimeError("X2 BFM environment is closed")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        del options
        self._wrapped.reset()
        self._history.zero_()
        self._last_action.zero_()
        observation = self._observation(append_history=True)
        qpos, qvel = self._qpos_qvel()
        if self.to_numpy:
            qpos, qvel = qpos.detach().cpu().numpy(), qvel.detach().cpu().numpy()
        return observation, {"qpos": qpos, "qvel": qvel}

    def step(self, action: np.ndarray | torch.Tensor):
        if self._closed:
            raise RuntimeError("X2 BFM environment is closed")
        action_tensor = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self.device,
        )
        if action_tensor.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"X2 BFM action shape mismatch: {action_tensor.shape}")
        if not torch.isfinite(action_tensor).all():
            raise ValueError("X2 BFM action contains non-finite values")
        if torch.abs(action_tensor).max() > 1.0 + 1e-6:
            raise ValueError("X2 BFM action exceeds [-1, 1]")
        action_tensor = action_tensor.clamp(-1.0, 1.0)
        next_observation, reward, done, extras = self._wrapped.step(action_tensor)
        del next_observation
        done = done.reshape(-1).bool()
        time_outs = extras.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(done)
        else:
            time_outs = torch.as_tensor(time_outs, device=self.device).reshape(-1).bool()
        terminated = done & ~time_outs
        truncated = time_outs
        self._last_action.copy_(action_tensor)
        # IsaacLab returns reset observations for lanes that ended during the
        # step.  Their new episode must not inherit an action/history frame
        # from the terminal episode.
        if done.any():
            self._history[done] = 0.0
            self._last_action[done] = 0.0
        observation = self._observation(append_history=True)
        qpos, qvel = self._qpos_qvel()
        info = dict(extras)
        info["qpos"] = qpos
        info["qvel"] = qvel
        info["aux_rewards"] = self._aux_rewards(
            action_tensor,
            terminated,
            torch.as_tensor(reward, device=self.device),
        )
        if self.to_numpy:
            reward = torch.as_tensor(reward).detach().cpu().numpy()
            terminated = terminated.detach().cpu().numpy()
            truncated = truncated.detach().cpu().numpy()
            info = {
                key: (
                    {
                        nested_key: nested_value.detach().cpu().numpy()
                        for nested_key, nested_value in value.items()
                    }
                    if isinstance(value, dict)
                    else value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in info.items()
            }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._closed = True
        # Process lifetime is supervised outside this adapter.  Isaac Kit
        # teardown has historically hung after valid evidence was written, so
        # the caller owns environment/application shutdown.
