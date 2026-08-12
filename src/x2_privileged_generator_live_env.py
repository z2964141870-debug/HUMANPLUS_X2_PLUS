"""Opt-in Isaac environment config for native-basin generator resets."""

from __future__ import annotations

import json
import os

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg

from x2_faithful_live_actions_phase46 import FaithfulWBT29TrackingEnvCfg
from x2_privileged_generator_live import (
    finalize_native_generator_reset,
    reset_from_native_generator_seed,
)


class PrivilegedGeneratorRLEnv(ManagerBasedRLEnv):
    """Finalize native state after SONIC's command manager reset."""

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        finalize_native_generator_reset(self)


class PrivilegedGeneratorTrackingEnvCfg(FaithfulWBT29TrackingEnvCfg):
    """Append the native reset event without changing historical Stage152."""

    def override_settings(self):
        super().override_settings()
        enabled = os.environ.get("X2_PRIVILEGED_GENERATOR_NATIVE_RESET", "false")
        if enabled == "false":
            return
        if enabled != "true":
            raise ValueError("X2_PRIVILEGED_GENERATOR_NATIVE_RESET must be true or false")
        required = (
            "X2_PRIVILEGED_GENERATOR_SEED_PATH",
            "X2_PRIVILEGED_GENERATOR_SEED_SHA256",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"missing native generator reset variables: {missing}")
        fixed = json.loads(os.environ.get("X2_PRIVILEGED_GENERATOR_FIXED_FRAMES", "null"))
        self.events.native_generator_reset = EventTermCfg(
            func=reset_from_native_generator_seed,
            mode="reset",
            params={
                "seed_path": os.environ["X2_PRIVILEGED_GENERATOR_SEED_PATH"],
                "expected_sha256": os.environ["X2_PRIVILEGED_GENERATOR_SEED_SHA256"],
                "reset_fraction": float(
                    os.environ.get("X2_PRIVILEGED_GENERATOR_RESET_FRACTION", "1.0")
                ),
                "sampling_mode": os.environ.get(
                    "X2_PRIVILEGED_GENERATOR_SAMPLING", "evenly_spaced"
                ),
                "fixed_frame_indices": fixed,
                "asset_name": "robot",
            },
        )
