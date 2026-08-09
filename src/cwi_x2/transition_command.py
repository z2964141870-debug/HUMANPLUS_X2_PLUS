"""IsaacLab command/reward terms for within-episode X2 stop transitions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from gear_sonic.envs.x2_velocity.heading_command import (
    GainScheduledUniformVelocityCommand,
    GainScheduledUniformVelocityCommandCfg,
)

from .transition_schedule import smooth_transition_speed, stratified_phase_offsets


class TransitionVelocityCommand(GainScheduledUniformVelocityCommand):
    """Schedule one start, cruise, deceleration, and stop in every episode."""

    cfg: "TransitionVelocityCommandCfg"

    def __init__(self, cfg: "TransitionVelocityCommandCfg", env):
        super().__init__(cfg, env)
        # CommandManager performs the first reset/resample while constructing
        # its terms.  Preserve that first sample instead of overwriting it.
        if not hasattr(self, "cruise_speed"):
            self.cruise_speed = self.vel_command_b[:, 0].clone()
        self.phase_offset_s = stratified_phase_offsets(
            self.num_envs,
            cfg.maximum_phase_offset_s,
            device=self.device,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        resolved = self._resolved_env_ids(env_ids)
        if hasattr(self, "cruise_speed"):
            self.cruise_speed[resolved] = self.vel_command_b[resolved, 0]
        self.is_standing_env[resolved] = False

    def _update_command(self):
        super()._update_command()
        elapsed_s = self._elapsed_s()
        self.vel_command_b[:, 0] = self._scheduled_speed(elapsed_s)

    def _elapsed_s(self) -> torch.Tensor:
        return (
            self._env.episode_length_buf.to(torch.float32)
            * float(self._env.step_dt)
            + self.phase_offset_s
        )

    def _scheduled_speed(self, elapsed_s: torch.Tensor) -> torch.Tensor:
        return smooth_transition_speed(
            elapsed_s,
            self.cruise_speed,
            stand_s=self.cfg.stand_s,
            accelerate_s=self.cfg.accelerate_s,
            cruise_s=self.cfg.cruise_s,
            decelerate_s=self.cfg.decelerate_s,
        )

    def locomotion_intent_features(self, horizon_s: float = 0.6) -> torch.Tensor:
        """Return deployable current speed and normalized future speed delta."""
        if horizon_s < 0.0:
            raise ValueError("locomotion intent horizon must be non-negative")
        elapsed_s = self._elapsed_s()
        current = self._scheduled_speed(elapsed_s)
        future = self._scheduled_speed(elapsed_s + horizon_s)
        scale = 0.5
        return torch.stack((current / scale, (future - current) / scale), dim=-1)

    def terminal_stop_mask(self) -> torch.Tensor:
        """Return environments that completed deceleration into final hold."""
        terminal_start = (
            self.cfg.stand_s
            + self.cfg.accelerate_s
            + self.cfg.cruise_s
            + self.cfg.decelerate_s
        )
        return self._elapsed_s() >= terminal_start


@configclass
class TransitionVelocityCommandCfg(GainScheduledUniformVelocityCommandCfg):
    class_type: type = TransitionVelocityCommand
    stand_s: float = 1.0
    accelerate_s: float = 1.0
    cruise_s: float = 4.0
    decelerate_s: float = 2.0
    maximum_phase_offset_s: float = 0.0


def transition_velocity_cfg(
    source: UniformVelocityCommandCfg,
    *,
    ideal_env_fraction: float,
    ideal_heading_control_stiffness: float,
    response_heading_control_stiffness: float,
    stand_s: float = 1.0,
    accelerate_s: float = 1.0,
    cruise_s: float = 4.0,
    decelerate_s: float = 2.0,
    maximum_phase_offset_s: float = 0.0,
) -> TransitionVelocityCommandCfg:
    """Clone the active command contract and disable mid-episode resampling."""
    return TransitionVelocityCommandCfg(
        asset_name=source.asset_name,
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=source.debug_vis,
        heading_command=source.heading_command,
        heading_control_stiffness=source.heading_control_stiffness,
        rel_standing_envs=0.0,
        rel_heading_envs=source.rel_heading_envs,
        ranges=source.ranges,
        goal_vel_visualizer_cfg=source.goal_vel_visualizer_cfg,
        current_vel_visualizer_cfg=source.current_vel_visualizer_cfg,
        ideal_env_fraction=ideal_env_fraction,
        ideal_heading_control_stiffness=ideal_heading_control_stiffness,
        response_heading_control_stiffness=response_heading_control_stiffness,
        stand_s=stand_s,
        accelerate_s=accelerate_s,
        cruise_s=cruise_s,
        decelerate_s=decelerate_s,
        maximum_phase_offset_s=maximum_phase_offset_s,
    )


def stopped_base_speed_l2(
    env,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual planar speed only after the scheduled command stops."""
    if command_threshold < 0.0:
        raise ValueError("command threshold must be non-negative")
    command = env.command_manager.get_command(command_name)
    stopped = torch.linalg.vector_norm(command[:, :2], dim=-1) <= command_threshold
    robot = env.scene[asset_cfg.name]
    speed_l2 = torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=-1)
    return speed_l2 * stopped.to(speed_l2.dtype)


def terminal_double_support_penalty(
    env,
    command_name: str,
    enter_force_n: float,
    left_sensor_name: str,
    right_sensor_name: str,
) -> torch.Tensor:
    """Penalize missing bilateral ground contact in the terminal hold only.

    This uses simulator ground-filtered forces as a training-only objective;
    it does not append force or contact truth to the deployable actor input.
    """
    if enter_force_n <= 0.0:
        raise ValueError("terminal contact force threshold must be positive")
    command = env.command_manager.get_term(command_name)
    terminal_method = getattr(command, "terminal_stop_mask", None)
    if terminal_method is None:
        raise RuntimeError("terminal contact reward requires TransitionVelocityCommand")

    def vertical_force(sensor_name: str) -> torch.Tensor:
        matrix = env.scene[sensor_name].data.force_matrix_w
        if matrix is None:
            raise RuntimeError(f"ground-filtered sensor {sensor_name!r} has no force matrix")
        return matrix[..., 2].abs().reshape(matrix.shape[0], -1).amax(dim=-1)

    contact = torch.stack(
        (
            vertical_force(left_sensor_name) > enter_force_n,
            vertical_force(right_sensor_name) > enter_force_n,
        ),
        dim=-1,
    )
    missing_fraction = 1.0 - contact.to(dtype=torch.float32).mean(dim=-1)
    return missing_fraction * terminal_method().to(dtype=missing_fraction.dtype)


__all__ = [
    "TransitionVelocityCommand",
    "TransitionVelocityCommandCfg",
    "stopped_base_speed_l2",
    "terminal_double_support_penalty",
    "transition_velocity_cfg",
]
