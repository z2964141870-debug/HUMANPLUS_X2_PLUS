#!/usr/bin/env python3
"""Probe the real IsaacLab command term without training or stepping physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from cwi_x2.transition_command import transition_velocity_cfg  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg  # noqa: E402


def main() -> None:
    cfg = X2LowerVelocityFlatEnvCfg()
    cfg.scene.num_envs = 2
    cfg.scene.env_spacing = 2.5
    cfg.sim.device = args.device
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.30)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity = transition_velocity_cfg(
        cfg.commands.base_velocity,
        ideal_env_fraction=0.0,
        ideal_heading_control_stiffness=1.0,
        response_heading_control_stiffness=0.05,
    )
    print("TRANSITION_PROBE_ENV_CREATE_BEGIN", flush=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    print("TRANSITION_PROBE_ENV_CREATE_DONE", flush=True)
    try:
        print("TRANSITION_PROBE_GET_TERM_BEGIN", flush=True)
        term = env.command_manager.get_term("base_velocity")
        try:
            term.reset(torch.arange(env.num_envs, device=env.device))
        except Exception as error:
            print(f"TRANSITION_PROBE_RESET_ERROR={error!r}", flush=True)
            raise
        print("TRANSITION_PROBE_GET_TERM_DONE", flush=True)
        print(
            "TRANSITION_PROBE_CRUISE="
            + json.dumps([float(value) for value in term.cruise_speed.cpu()]),
            flush=True,
        )
        samples = []
        for time_s in (0.0, 1.0, 1.5, 2.0, 6.0, 7.0, 8.0, 12.0):
            print(f"TRANSITION_PROBE_SAMPLE_BEGIN={time_s}", flush=True)
            env.episode_length_buf.fill_(round(time_s / env.step_dt))
            term._update_command()
            samples.append(
                {
                    "time_s": time_s,
                    "vx": [float(value) for value in term.command[:, 0].cpu()],
                }
            )
            print(f"TRANSITION_PROBE_SAMPLE_DONE={time_s}", flush=True)
        observed = torch.tensor([row["vx"][0] for row in samples])
        expected = torch.tensor([0.0, 0.0, 0.15, 0.30, 0.30, 0.15, 0.0, 0.0])
        print(
            "TRANSITION_PROBE_OBSERVED="
            + json.dumps([float(value) for value in observed]),
            flush=True,
        )
        torch.testing.assert_close(observed, expected, atol=1.0e-6, rtol=0.0)
        result = {"contract": "pass", "samples": samples}
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
