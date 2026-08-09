"""Isaac-only WBT29 action term for the opt-in Phase46 live boundary."""

from __future__ import annotations

import re

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from gear_sonic.envs.manager_env import mdp as observation_terms
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import ModularTrackingEnvCfg


class WBT29JointPositionAction(JointPositionAction):
    """29-joint target term that rejects/drops only head-only scale rows.

    X2's robot-level action-scale builder correctly declares all 31 actuator
    rows.  At the faithful policy boundary the action term selects 29 joints,
    so the two explicit zero-scale head regexes would otherwise be considered
    unmatched by Isaac.  Removing only regexes that match none of the declared
    WBT29 names preserves every body scale and leaves head control to its
    simulator nominal target.
    """

    def __init__(self, cfg, env):
        names = tuple(cfg.joint_names)
        if len(names) != 29 or len(set(names)) != 29:
            raise ValueError("Phase46 action cfg must declare 29 unique exact joints")
        if isinstance(cfg.scale, dict):
            original = dict(cfg.scale)
            cfg.scale = {
                expression: value
                for expression, value in original.items()
                if any(re.fullmatch(expression, name) for name in names)
            }
            removed = sorted(set(original) - set(cfg.scale))
            if removed != ["head_pitch_joint", "head_yaw_joint"]:
                raise ValueError(
                    "Phase46 may remove only the two nominal head scale rows; "
                    f"removed={removed}"
                )
        super().__init__(cfg, env)


@configclass
class WBT29JointPositionActionCfg(JointPositionActionCfg):
    class_type: type = WBT29JointPositionAction


class FaithfulWBT29TrackingEnvCfg(ModularTrackingEnvCfg):
    """Phase46-only environment boundary with native 29-D action feedback.

    The stock X2 post-init assumes a 31-D action term and masks the two head
    entries in the observation. Phase46 already excludes both head joints at
    the action-manager boundary, so applying that mask a second time is an
    error. An empty mask returns the faithful 29-D raw action while leaving all
    historical configurations unchanged.
    """

    def override_settings(self):
        super().override_settings()
        for group_name in ("policy", "critic"):
            group = getattr(self.observations, group_name, None)
            term = getattr(group, "actions", None)
            if term is None:
                continue
            term.func = observation_terms.last_action_with_zeroed_joints
            term.params = {"joint_names": [], "action_name": "joint_pos"}
