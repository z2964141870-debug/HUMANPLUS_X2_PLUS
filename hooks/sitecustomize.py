"""Opt-in import hook that injects upper targets into the legacy X2 evaluator.

This file is imported automatically only when this directory is placed first on
PYTHONPATH.  With both CWI_UPPER_MOTION and CWI_UPPER_MOTION_LIST unset it is a
no-op.  It never edits the old X2 project; it patches the action term in the
current Python process only.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


_TARGET_MODULE = "gear_sonic.envs.x2_velocity.actions"


def _patch_actions_module(module) -> None:
    import torch

    from cwi_x2.upper_motion_contract import UPPER_JOINT_NAMES, load_upper_motion

    cls = module.LowerBodyJointPositionAction
    if getattr(cls, "_cwi_upper_patch_installed", False):
        return
    original_init = cls.__init__
    original_apply = cls.apply_actions

    def patched_init(self, cfg, env):
        original_init(self, cfg, env)
        motion_list = os.environ.get("CWI_UPPER_MOTION_LIST")
        if motion_list:
            motion_paths = [path for path in motion_list.split(";") if path]
        else:
            motion_paths = [os.environ["CWI_UPPER_MOTION"]]
        if not motion_paths:
            raise ValueError("CWI upper motion list is empty")
        clips = [load_upper_motion(path) for path in motion_paths]
        scale = float(os.environ.get("CWI_UPPER_SCALE", "0"))
        start_list = os.environ.get("CWI_UPPER_START_S_LIST")
        if start_list:
            start_values = [float(value) for value in start_list.split(";")]
            if len(start_values) != len(clips):
                raise ValueError(
                    "CWI_UPPER_START_S_LIST must match CWI_UPPER_MOTION_LIST"
                )
        else:
            start_values = [
                float(os.environ.get("CWI_UPPER_START_S", "0"))
            ] * len(clips)
        time_scale = float(os.environ.get("CWI_UPPER_TIME_SCALE", "1"))
        max_excursion = float(os.environ.get("CWI_UPPER_MAX_EXCURSION_RAD", "inf"))
        max_velocity = float(os.environ.get("CWI_UPPER_MAX_VELOCITY_RADPS", "inf"))
        tilt_fallback = float(os.environ.get("CWI_UPPER_TILT_FALLBACK_RAD", "inf"))
        height_fallback = float(os.environ.get("CWI_UPPER_HEIGHT_FALLBACK_M", "-inf"))
        heading_fallback = float(
            os.environ.get("CWI_UPPER_HEADING_FALLBACK_RAD", "inf")
        )
        latch_fallback = os.environ.get("CWI_UPPER_LATCH_FALLBACK", "0") == "1"
        randomize_clip = os.environ.get("CWI_UPPER_RANDOMIZE_CLIP", "0") == "1"
        loop = os.environ.get("CWI_UPPER_LOOP", "0") == "1"
        stop_with_command = (
            os.environ.get("CWI_UPPER_STOP_WITH_COMMAND", "0") == "1"
        )
        zero_fraction = float(os.environ.get("CWI_UPPER_ZERO_FRACTION", "0"))
        deterministic_split = (
            os.environ.get("CWI_UPPER_DETERMINISTIC_SPLIT", "0") == "1"
        )
        if scale < 0.0:
            raise ValueError("CWI_UPPER_SCALE must be non-negative")
        if not 0.0 <= zero_fraction <= 1.0:
            raise ValueError("CWI_UPPER_ZERO_FRACTION must lie in [0, 1]")
        for start_s, clip in zip(start_values, clips, strict=True):
            if start_s < 0.0 or start_s > clip.duration_s:
                raise ValueError(
                    f"CWI upper start={start_s} outside [0,{clip.duration_s}] "
                    f"for {clip.path}"
                )
        if time_scale <= 0.0:
            raise ValueError("CWI_UPPER_TIME_SCALE must be positive")
        if max_excursion <= 0.0 or max_velocity <= 0.0:
            raise ValueError("CWI upper excursion and velocity limits must be positive")
        missing = [name for name in UPPER_JOINT_NAMES if name not in self._asset.joint_names]
        if missing:
            raise RuntimeError(f"X2 articulation is missing upper joints: {missing}")
        ids = [self._asset.joint_names.index(name) for name in UPPER_JOINT_NAMES]
        self._cwi_upper_joint_ids = ids
        self._cwi_upper_clips = [
            torch.as_tensor(clip.q_rad, device=self.device, dtype=torch.float32)
            for clip in clips
        ]
        self._cwi_upper_fps = [float(clip.fps) for clip in clips]
        self._cwi_upper_clip_paths = tuple(str(clip.path) for clip in clips)
        self._cwi_upper_start_by_clip = torch.tensor(
            start_values, device=self.device, dtype=torch.float32
        )
        self._cwi_upper_clip_ids = (
            torch.arange(self._env.num_envs, device=self.device, dtype=torch.int64)
            % len(clips)
        )
        self._cwi_upper_randomize_clip = randomize_clip
        self._cwi_upper_scale = scale
        self._cwi_upper_time_scale = time_scale
        self._cwi_upper_max_excursion = max_excursion
        self._cwi_upper_max_velocity = max_velocity
        self._cwi_upper_tilt_fallback = tilt_fallback
        self._cwi_upper_height_fallback = height_fallback
        self._cwi_upper_heading_fallback = heading_fallback
        self._cwi_upper_latch_fallback = latch_fallback
        self._cwi_upper_loop = loop
        self._cwi_upper_stop_with_command = stop_with_command
        self._cwi_upper_zero_fraction = zero_fraction
        self._cwi_upper_deterministic_split = deterministic_split
        # Keep the default contract bit-exact: with the opt-in fraction left
        # at zero no environment is masked.  A deterministic initial split
        # also guarantees that a short curriculum batch contains both fixed
        # and moving upper-body conditions before the first explicit reset.
        zero_count = int(round(self._env.num_envs * zero_fraction))
        self._cwi_upper_zero_mask = torch.zeros(
            self._env.num_envs, device=self.device, dtype=torch.bool
        )
        self._cwi_upper_zero_mask[:zero_count] = True
        start_s = self._cwi_upper_start_by_clip[self._cwi_upper_clip_ids]
        self._cwi_upper_baseline = self._cwi_sample_upper(
            start_s
        )
        self._cwi_prev_upper_target = self._asset.data.default_joint_pos[
            :, self._cwi_upper_joint_ids
        ].clone()
        self._cwi_upper_fallback_active = torch.zeros(
            self._env.num_envs, device=self.device, dtype=torch.bool
        )
        self._cwi_initial_yaw = torch.zeros(
            self._env.num_envs, device=self.device, dtype=torch.float32
        )
        self._cwi_initial_yaw_valid = torch.zeros(
            self._env.num_envs, device=self.device, dtype=torch.bool
        )
        self._cwi_upper_heading_error = torch.zeros_like(self._cwi_initial_yaw)

    def sample_upper(self, time_s):
        result = torch.empty(
            (self._env.num_envs, len(UPPER_JOINT_NAMES)),
            device=self.device,
            dtype=torch.float32,
        )
        for clip_id, (trajectory, fps) in enumerate(
            zip(self._cwi_upper_clips, self._cwi_upper_fps, strict=True)
        ):
            mask = self._cwi_upper_clip_ids == clip_id
            if not torch.any(mask):
                continue
            last = trajectory.shape[0] - 1
            frame = time_s[mask] * fps
            if self._cwi_upper_loop:
                frame = torch.remainder(frame, last)
            else:
                frame = torch.clamp(frame, 0.0, float(last))
            lower = torch.floor(frame).to(torch.int64)
            upper = torch.clamp(lower + 1, max=last)
            blend = frame - lower.to(frame.dtype)
            result[mask] = (
                (1.0 - blend).unsqueeze(-1) * trajectory[lower]
                + blend.unsqueeze(-1) * trajectory[upper]
            )
        return result

    def reference_time(self):
        return (
            self._env.episode_length_buf.to(torch.float32)
            * self._env.step_dt
            * self._cwi_upper_time_scale
            + self._cwi_upper_start_by_clip[self._cwi_upper_clip_ids]
        )

    def bounded_intent_delta(self, time_s):
        sample = self._cwi_sample_upper(time_s)
        delta = self._cwi_upper_scale * (sample - self._cwi_upper_baseline)
        if self._cwi_upper_max_excursion < float("inf"):
            delta = torch.clamp(
                delta,
                min=-self._cwi_upper_max_excursion,
                max=self._cwi_upper_max_excursion,
            )
        delta = torch.where(
            self._cwi_upper_zero_mask.unsqueeze(-1),
            torch.zeros_like(delta),
            delta,
        )
        return delta

    def upper_intent_features(self, horizon_s=0.6):
        if horizon_s <= 0.0:
            raise ValueError("future upper-intent horizon must be positive")
        time_s = self._cwi_reference_time()
        current = self._cwi_bounded_intent_delta(time_s)
        future = self._cwi_bounded_intent_delta(
            time_s + horizon_s * self._cwi_upper_time_scale
        )
        return torch.cat((current, future - current), dim=-1)

    def patched_apply(self):
        original_apply(self)
        time_s = self._cwi_reference_time()
        default = self._asset.data.default_joint_pos[:, self._cwi_upper_joint_ids]
        target = default + self._cwi_bounded_intent_delta(time_s)
        if self._cwi_upper_stop_with_command:
            command = self._env.command_manager.get_command(self.cfg.command_name)
            moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.1
            target = torch.where(moving.unsqueeze(-1), target, default)
        root_quat = self._asset.data.root_quat_w
        root_up_z = 1.0 - 2.0 * (
            root_quat[:, 1].square() + root_quat[:, 2].square()
        )
        root_tilt = torch.acos(torch.clamp(root_up_z, -1.0, 1.0))
        root_height = self._asset.data.root_pos_w[:, 2]
        root_yaw = torch.atan2(
            2.0
            * (
                root_quat[:, 0] * root_quat[:, 3]
                + root_quat[:, 1] * root_quat[:, 2]
            ),
            1.0
            - 2.0
            * (
                root_quat[:, 2].square()
                + root_quat[:, 3].square()
            ),
        )
        initialize_yaw = ~self._cwi_initial_yaw_valid
        self._cwi_initial_yaw[initialize_yaw] = root_yaw[initialize_yaw]
        self._cwi_initial_yaw_valid[initialize_yaw] = True
        heading_error = torch.atan2(
            torch.sin(root_yaw - self._cwi_initial_yaw),
            torch.cos(root_yaw - self._cwi_initial_yaw),
        )
        self._cwi_upper_heading_error.copy_(heading_error)
        healthy = (root_tilt <= self._cwi_upper_tilt_fallback) & (
            root_height >= self._cwi_upper_height_fallback
        ) & (torch.abs(heading_error) <= self._cwi_upper_heading_fallback)
        if self._cwi_upper_latch_fallback:
            self._cwi_upper_fallback_active |= ~healthy
        else:
            self._cwi_upper_fallback_active = ~healthy
        target = torch.where(
            (~self._cwi_upper_fallback_active).unsqueeze(-1), target, default
        )
        limits = self._asset.data.soft_joint_pos_limits[:, self._cwi_upper_joint_ids]
        # Some validated X2 nominal arm targets sit slightly outside Isaac's
        # reduced "soft" interval.  Never let safety clipping move the known
        # standing default: scale=0 must be bit-exact with the old controller.
        lower = torch.minimum(limits[..., 0], default)
        upper = torch.maximum(limits[..., 1], default)
        target = torch.maximum(torch.minimum(target, upper), lower)
        if self._cwi_upper_max_velocity < float("inf"):
            physics_dt = float(getattr(self._env, "physics_dt", self._env.step_dt))
            max_step = self._cwi_upper_max_velocity * physics_dt
            target = self._cwi_prev_upper_target + torch.clamp(
                target - self._cwi_prev_upper_target,
                min=-max_step,
                max=max_step,
            )
        self._cwi_prev_upper_target.copy_(target)
        self._asset.set_joint_position_target(target, joint_ids=self._cwi_upper_joint_ids)

    def patched_reset(self, env_ids=None):
        original_reset(self, env_ids)
        # Some manager environments may call reset while the original action
        # term is still being constructed.  In that case the CWI state does
        # not exist yet and there is nothing extra to reset.
        if not hasattr(self, "_cwi_prev_upper_target"):
            return
        if env_ids is None:
            env_index = torch.arange(
                self._env.num_envs, device=self.device, dtype=torch.int64
            )
        elif isinstance(env_ids, slice):
            env_index = torch.arange(
                self._env.num_envs, device=self.device, dtype=torch.int64
            )[env_ids]
        else:
            env_index = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.int64
            )
        if self._cwi_upper_randomize_clip and env_index.numel():
            self._cwi_upper_clip_ids[env_index] = torch.randint(
                len(self._cwi_upper_clips),
                (env_index.numel(),),
                device=self.device,
            )
        if (
            env_index.numel()
            and self._cwi_upper_zero_fraction > 0.0
            and not self._cwi_upper_deterministic_split
        ):
            self._cwi_upper_zero_mask[env_index] = (
                torch.rand(env_index.numel(), device=self.device)
                < self._cwi_upper_zero_fraction
            )
        start_s = self._cwi_upper_start_by_clip[self._cwi_upper_clip_ids]
        baseline = self._cwi_sample_upper(start_s)
        self._cwi_upper_baseline[env_index] = baseline[env_index]
        ids = env_index
        default = self._asset.data.default_joint_pos[:, self._cwi_upper_joint_ids]
        self._cwi_prev_upper_target[ids] = default[ids]
        self._cwi_upper_fallback_active[ids] = False
        self._cwi_initial_yaw_valid[ids] = False
        self._cwi_initial_yaw[ids] = 0.0
        self._cwi_upper_heading_error[ids] = 0.0

    cls.__init__ = patched_init
    cls._cwi_sample_upper = sample_upper
    cls._cwi_reference_time = reference_time
    cls._cwi_bounded_intent_delta = bounded_intent_delta
    cls._cwi_upper_intent_features = upper_intent_features
    cls.apply_actions = patched_apply
    original_reset = cls.reset
    cls.reset = patched_reset
    cls._cwi_upper_patch_installed = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        method = getattr(self._wrapped, "create_module", None)
        return None if method is None else method(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_actions_module(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot locate {fullname} for CWI upper-motion patch")
        spec.loader = _PatchLoader(spec.loader)
        return spec


if os.environ.get("CWI_UPPER_MOTION") or os.environ.get("CWI_UPPER_MOTION_LIST"):
    sys.meta_path.insert(0, _PatchFinder())
