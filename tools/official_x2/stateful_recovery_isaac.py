"""Isaac Lab runtime glue for the Stage335 stateful recovery contract.

Import this module only after ``AppLauncher`` has started Isaac Sim.  Pure data
validation remains in :mod:`official_x2.recovery_reset_curriculum` so its unit
tests do not require Kit/PhysX.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv

from gear_sonic.envs.x2_velocity.actions import GaitTemplateLowerBodyJointPositionAction

from .recovery_reset_curriculum import finalize_stateful_recovery


def _stateful_moving_mask(env, command_name: str, threshold: float = 0.1) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > threshold
    if not hasattr(env, "_x2_recovery_force_moving"):
        return moving
    active = env._x2_recovery_force_moving & (
        env.episode_length_buf <= env._x2_recovery_force_moving_until_step
    )
    return moving | active


def stateful_gait_phase_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
    cycle_time_s: float,
    double_support_fraction: float,
) -> torch.Tensor:
    """Normal gait clock plus the bounded Stage326 force-moving event latch."""
    if cycle_time_s <= 0.0:
        raise ValueError("gait cycle time must be positive")
    if not 0.0 < double_support_fraction < 1.0:
        raise ValueError("double support fraction must lie in (0, 1)")
    moving = _stateful_moving_mask(env, command_name)
    elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    phase = torch.remainder(elapsed_s / cycle_time_s, 1.0)
    angle = 2.0 * math.pi * phase
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    clock *= moving.to(clock.dtype).unsqueeze(-1)
    half_ds = double_support_fraction / 4.0
    right_swing = (phase >= half_ds) & (phase < 0.5 - half_ds)
    left_swing = (phase >= 0.5 + half_ds) & (phase < 1.0 - half_ds)
    desired = torch.stack((~left_swing, ~right_swing), dim=-1)
    desired = torch.where(moving.unsqueeze(-1), desired, torch.ones_like(desired))
    return torch.cat((clock, desired.to(clock.dtype)), dim=-1)


class StatefulRecoveryGaitTemplateAction(GaitTemplateLowerBodyJointPositionAction):
    """Keep the source event's moving latch active after a stateful reset."""

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        moving = _stateful_moving_mask(
            self._env, self.cfg.command_name, self.cfg.command_threshold
        )
        # The base implementation is already exact when no event override is
        # active.  Recompute only when a low-magnitude brake command still has
        # an explicit moving phase in the source trace.
        command = self._env.command_manager.get_command(self.cfg.command_name)
        ordinary = torch.linalg.vector_norm(command[:, :2], dim=-1) > self.cfg.command_threshold
        if not bool(torch.any(moving & ~ordinary)):
            return
        phase = torch.remainder(
            self._env.episode_length_buf.to(torch.float32)
            * self._env.step_dt
            / self._template_period_s,
            1.0,
        )
        bins = self._template_q.shape[0]
        phase_position = phase * bins - 0.5
        lower_unwrapped = torch.floor(phase_position)
        blend = phase_position - lower_unwrapped
        lower = lower_unwrapped.to(torch.int64) % bins
        upper = (lower + 1) % bins
        q_bias = (
            (1.0 - blend).unsqueeze(-1) * self._template_q[lower]
            + blend.unsqueeze(-1) * self._template_q[upper]
        )
        q_bias *= moving.to(q_bias.dtype).unsqueeze(-1)
        normalized_bias = self.cfg.template_scale * q_bias / self._scale
        self._normalized_template_bias[:] = normalized_bias
        plant_bias = self._normalized_plant_bias.unsqueeze(0) * self._plant_bias_env_mask
        if self.cfg.plant_bias_moving_only:
            plant_bias *= moving.to(self._raw_actions.dtype).unsqueeze(-1)
        self._preclip_combined_actions[:] = self._raw_actions + normalized_bias + plant_bias
        self._combined_normalized_actions = torch.clamp(
            self._preclip_combined_actions, min=-1.0, max=1.0
        )
        self._processed_actions = self._combined_normalized_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


class StatefulRecoveryRLEnv(ManagerBasedRLEnv):
    """Finalize logical RSI state after all standard manager resets."""

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        finalize_stateful_recovery(self)


def configure_stateful_recovery_cfg(cfg) -> None:
    """Install the stateful action/observation terms without editing Gear-SONIC."""
    cfg.actions.joint_pos.class_type = StatefulRecoveryGaitTemplateAction
    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name)
        gait = getattr(group, "gait_phase", None)
        if gait is None:
            raise RuntimeError(f"stateful recovery requires {group_name}.gait_phase")
        gait.func = stateful_gait_phase_observation


__all__ = [
    "StatefulRecoveryGaitTemplateAction",
    "StatefulRecoveryRLEnv",
    "configure_stateful_recovery_cfg",
    "stateful_gait_phase_observation",
]
