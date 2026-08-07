"""Ground-filtered reward terms for X2 lower-body locomotion.

The X2 mesh needs self-collision enabled for the validated standing substrate.
Consequently, the broad all-body contact sensor cannot be used to decide
whether a foot is on the floor: ankle self-contact would look like stance.
These terms intentionally read only forces filtered against the ground plane.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .gait import gait_phase_and_desired_contacts, gait_transition_grace_mask


def _ground_vertical_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Return maximum absolute ground-normal force for one filtered sensor."""
    sensor = env.scene[sensor_name]
    forces_w = sensor.data.force_matrix_w
    if forces_w is None:
        raise RuntimeError(
            f"ground-filtered contact sensor {sensor_name!r} has no force_matrix_w; "
            "check filter_prim_paths_expr"
        )
    return forces_w[..., 2].abs().reshape(forces_w.shape[0], -1).amax(dim=-1)


def ground_filtered_feet_slide(
    env: ManagerBasedRLEnv,
    left_sensor_name: str,
    right_sensor_name: str,
    force_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize horizontal foot speed only during real ground contact."""
    if force_threshold <= 0.0:
        raise ValueError("ground contact force threshold must be positive")

    ground_force = torch.stack(
        (
            _ground_vertical_force(env, left_sensor_name),
            _ground_vertical_force(env, right_sensor_name),
        ),
        dim=-1,
    )
    in_contact = ground_force > force_threshold
    robot = env.scene[asset_cfg.name]
    foot_xy_speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    if foot_xy_speed.shape[-1] != 2:
        raise RuntimeError(
            "ground_filtered_feet_slide requires exactly two ordered foot bodies, "
            f"got shape {tuple(foot_xy_speed.shape)}"
        )
    return torch.sum(foot_xy_speed * in_contact, dim=-1)


def base_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame root yaw rate without suppressing roll/pitch recovery."""
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_w[:, 2])


def heading_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalize accumulated world-heading error only in heading-target envs.

    Direct yaw-rate environments are masked out so learning intentional turns
    does not conflict with holding a sampled world heading.  The actor still
    receives only the ordinary deployable yaw-rate command; world heading is
    used as a training reward signal, not appended to its observation.
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    error = torch.atan2(
        torch.sin(command.heading_target - robot.data.heading_w),
        torch.cos(command.heading_target - robot.data.heading_w),
    )
    return torch.square(error) * command.is_heading_env.to(error.dtype)


class GroundFilteredBipedAirTime(ManagerTermBase):
    """Reward alternating single support using ground-only contact state.

    This mirrors IsaacLab's positive biped air-time reward, but maintains its
    own timers from ``force_matrix_w``.  ContactSensor's built-in air-time
    buffers use net contact and are therefore contaminated by X2 self-contact.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        self._air_time = torch.zeros((env.num_envs, 2), device=env.device)
        self._contact_time = torch.zeros((env.num_envs, 2), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._air_time[env_ids] = 0.0
        self._contact_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float,
        force_threshold: float,
        left_sensor_name: str,
        right_sensor_name: str,
    ) -> torch.Tensor:
        if threshold <= 0.0:
            raise ValueError("air-time reward threshold must be positive")
        if force_threshold <= 0.0:
            raise ValueError("ground contact force threshold must be positive")

        ground_force = torch.stack(
            (
                _ground_vertical_force(env, left_sensor_name),
                _ground_vertical_force(env, right_sensor_name),
            ),
            dim=-1,
        )
        in_contact = ground_force > force_threshold

        self._contact_time = torch.where(
            in_contact,
            self._contact_time + env.step_dt,
            torch.zeros_like(self._contact_time),
        )
        self._air_time = torch.where(
            in_contact,
            torch.zeros_like(self._air_time),
            self._air_time + env.step_dt,
        )
        in_mode_time = torch.where(in_contact, self._contact_time, self._air_time)
        single_stance = in_contact.to(torch.int32).sum(dim=-1) == 1
        reward = torch.min(
            torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=-1
        ).values
        reward = torch.clamp(reward, max=threshold)
        moving = torch.linalg.vector_norm(
            env.command_manager.get_command(command_name)[:, :2], dim=-1
        ) > 0.1
        return reward * moving


class GroundContactDwellPenalty(ManagerTermBase):
    """Penalize ground-contact mode changes that occur before a minimum dwell.

    Contact is debounced with separate enter/exit force thresholds.  The term
    does not prescribe a phase or which foot should support; it only rejects
    the 10--16 Hz contact chatter observed in Stage177/178.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        self._contact = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self._mode_time = torch.zeros((env.num_envs, 2), device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contact[env_ids] = False
        self._mode_time[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        min_dwell_s: float,
        enter_force_n: float,
        exit_force_n: float,
        left_sensor_name: str,
        right_sensor_name: str,
    ) -> torch.Tensor:
        if min_dwell_s <= 0.0:
            raise ValueError("minimum contact dwell must be positive")
        if not 0.0 <= exit_force_n < enter_force_n:
            raise ValueError("contact hysteresis requires 0 <= exit < enter")

        force = torch.stack(
            (
                _ground_vertical_force(env, left_sensor_name),
                _ground_vertical_force(env, right_sensor_name),
            ),
            dim=-1,
        )
        next_contact = torch.where(
            self._contact,
            force >= exit_force_n,
            force > enter_force_n,
        )
        initialized = self._initialized.unsqueeze(-1)
        changed = (next_contact != self._contact) & initialized
        shortfall = torch.clamp((min_dwell_s - self._mode_time) / min_dwell_s, min=0.0)
        penalty = torch.sum(shortfall * changed, dim=-1)

        first_contact = force > enter_force_n
        self._contact = torch.where(initialized, next_contact, first_contact)
        self._mode_time = torch.where(
            initialized,
            torch.where(changed, torch.zeros_like(self._mode_time), self._mode_time + env.step_dt),
            torch.zeros_like(self._mode_time),
        )
        self._initialized[:] = True

        moving = torch.linalg.vector_norm(
            env.command_manager.get_command(command_name)[:, :2], dim=-1
        ) > 0.1
        return penalty * moving


class GroundContactPhasePenalty(ManagerTermBase):
    """Penalize disagreement with a deployable DS/SS gait-clock schedule.

    Actual contact is debounced from ground-only force.  A small grace window
    around each scheduled transition avoids demanding an impossible perfectly
    instantaneous force switch and makes timing error, rather than sensor
    threshold noise, the optimized quantity.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        self._contact = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contact[env_ids] = False
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        cycle_time_s: float,
        double_support_fraction: float,
        transition_grace_s: float,
        enter_force_n: float,
        exit_force_n: float,
        left_sensor_name: str,
        right_sensor_name: str,
    ) -> torch.Tensor:
        if not 0.0 <= exit_force_n < enter_force_n:
            raise ValueError("contact hysteresis requires 0 <= exit < enter")

        force = torch.stack(
            (
                _ground_vertical_force(env, left_sensor_name),
                _ground_vertical_force(env, right_sensor_name),
            ),
            dim=-1,
        )
        next_contact = torch.where(self._contact, force >= exit_force_n, force > enter_force_n)
        initialized = self._initialized.unsqueeze(-1)
        first_contact = force > enter_force_n
        self._contact = torch.where(initialized, next_contact, first_contact)
        self._initialized[:] = True

        phase, desired_contact, moving = gait_phase_and_desired_contacts(
            env,
            command_name=command_name,
            cycle_time_s=cycle_time_s,
            double_support_fraction=double_support_fraction,
        )
        outside_grace = gait_transition_grace_mask(
            phase,
            cycle_time_s=cycle_time_s,
            double_support_fraction=double_support_fraction,
            transition_grace_s=transition_grace_s,
        )
        mismatch = (self._contact != desired_contact).to(dtype=force.dtype).mean(dim=-1)
        return mismatch * moving * outside_grace
