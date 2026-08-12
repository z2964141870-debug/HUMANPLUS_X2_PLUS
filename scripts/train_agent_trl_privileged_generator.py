#!/usr/bin/env python3
"""Opt-in Stage152 entrypoint selecting the post-manager reset environment."""

from __future__ import annotations

import os
import sys

# This wrapper is launched by absolute path after Stage152 changes cwd to the
# SONIC sandbox.  Python otherwise places only this file's directory at
# sys.path[0], so the sandbox-local upstream entrypoint is not importable.
_sandbox_cwd = os.getcwd()
if _sandbox_cwd not in sys.path:
    sys.path.insert(0, _sandbox_cwd)

import train_agent_trl as upstream

# Upstream deliberately removes its script directory so `trl` resolves to the
# HuggingFace package instead of the sibling `gear_sonic/trl` package.  Keep it
# removed.  Since this wrapper imports the Hydra-decorated function, tell Hydra
# to resolve its relative config path as if the upstream file were the main
# module; putting the script directory back on sys.path would reintroduce the
# `trl` name collision.
os.environ.setdefault("HYDRA_MAIN_MODULE", "__main__")


_create_manager_env = upstream.create_manager_env


def _create_privileged_manager_env(config, device, args_cli):
    # Upstream starts AppLauncher before invoking this factory. Importing
    # isaaclab.envs earlier fails because the pxr/Kit runtime is not live yet.
    import isaaclab.envs
    from x2_privileged_generator_live_env import PrivilegedGeneratorRLEnv

    # Isaac Kit's custom importer can load the child module without exposing it
    # as an attribute on the top-level ``isaaclab`` package.  The canonical
    # module object is still registered in sys.modules.
    isaaclab_envs = sys.modules["isaaclab.envs"]
    original = isaaclab_envs.ManagerBasedRLEnv
    isaaclab_envs.ManagerBasedRLEnv = PrivilegedGeneratorRLEnv
    try:
        return _create_manager_env(config, device, args_cli)
    finally:
        isaaclab_envs.ManagerBasedRLEnv = original


upstream.create_manager_env = _create_privileged_manager_env


if __name__ == "__main__":
    upstream.main()
