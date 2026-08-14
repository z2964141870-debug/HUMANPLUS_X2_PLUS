#!/usr/bin/env python3
"""Print the immutable X2 Isaac articulation joint order without stepping."""

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402

try:
    cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg)
    print(json.dumps(list(env.scene["robot"].joint_names)), flush=True)
    os._exit(0)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
