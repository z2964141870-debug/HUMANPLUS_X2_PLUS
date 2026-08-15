#!/usr/bin/env python3
"""Read-only live IsaacLab probe for the X2 six-link target contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402
from gear_sonic.envs.manager_env.robots.x2 import X2_URDF_BY_COLLISION_PROFILE  # noqa: E402

from humanoidverse.x2_extremity_contract import (  # noqa: E402
    X2_EXTREMITY_LINK_NAMES,
    X2_EXTERNAL_ARM_JOINT_NAMES,
    X2_LOCKED_HEAD_JOINT_NAMES,
    X2_LOWER_POLICY_JOINT_NAMES,
    extremity_future_pose_error,
    resolve_named_indices,
)


NUM_ENVS = 4
SEED = 770611
ZERO_GATE = 1.0e-7
LOCALIZATION_GATE = 1.0e-7
REPORT = args.report.expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cfg():
    cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.seed = SEED
    cfg.sim.device = args.device
    cfg.scene.num_envs = NUM_ENVS
    cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    return cfg


def main() -> dict:
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    if REPORT.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite extremity-contract evidence")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    env = ManagerBasedRLEnv(cfg=build_cfg())
    env.reset(seed=SEED)
    robot = env.scene["robot"]

    body_indices = resolve_named_indices(robot.body_names, X2_EXTREMITY_LINK_NAMES)
    joint_partition = (
        X2_LOWER_POLICY_JOINT_NAMES
        + X2_EXTERNAL_ARM_JOINT_NAMES
        + X2_LOCKED_HEAD_JOINT_NAMES
    )
    joint_indices = resolve_named_indices(robot.joint_names, joint_partition)
    body_ids = torch.tensor(body_indices, dtype=torch.long, device=env.device)
    position = robot.data.body_pos_w.index_select(1, body_ids).clone()
    quaternion = robot.data.body_quat_w.index_select(1, body_ids).clone()

    zero_error = extremity_future_pose_error(position, quaternion, position, quaternion)
    sign_error = extremity_future_pose_error(position, quaternion, position, -quaternion)
    target_position = position.clone()
    left_wrist_index = X2_EXTREMITY_LINK_NAMES.index("left_wrist_roll_link")
    target_position[:, left_wrist_index, 1] += 0.03
    localized = extremity_future_pose_error(
        position,
        quaternion,
        target_position,
        quaternion,
    )
    expected = torch.zeros_like(localized)
    expected[:, 0, left_wrist_index, 1] = 0.03
    localization_error = float((localized - expected).abs().max())
    joint_partition_exact = len(set(joint_partition)) == len(robot.joint_names) == 31
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (position, quaternion, zero_error, sign_error, localized)
    )
    zero_max = float(zero_error.abs().max())
    sign_max = float(sign_error.abs().max())
    passed = (
        tuple(robot.body_names[index] for index in body_indices) == X2_EXTREMITY_LINK_NAMES
        and tuple(robot.joint_names[index] for index in joint_indices) == joint_partition
        and joint_partition_exact
        and zero_max <= ZERO_GATE
        and sign_max <= ZERO_GATE
        and localization_error <= LOCALIZATION_GATE
        and finite
    )
    # Isaac Sim 5.1 can block for minutes while tearing down a headless
    # ManagerBasedRLEnv.  This probe has no stepping loop, weights, or
    # checkpoint writer; the process exits immediately after the report and
    # SHA256 are written below, so keeping teardown out of the evidence path
    # makes the static gate deterministic.
    return {
        "schema": "x2_extremity_contract_probe_v11",
        "decision": (
            "PASS_X2_SIX_LINK_CONTRACT_FOR_STATIC_TRACKING_SMOKE"
            if passed
            else "FAIL_X2_SIX_LINK_CONTRACT_BLOCK_TRACKING"
        ),
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "six_link_order": list(X2_EXTREMITY_LINK_NAMES),
        "six_link_body_indices": list(body_indices),
        "policy_action_15": list(X2_LOWER_POLICY_JOINT_NAMES),
        "external_upper_14": list(X2_EXTERNAL_ARM_JOINT_NAMES),
        "locked_head_2": list(X2_LOCKED_HEAD_JOINT_NAMES),
        "joint_partition_exact": joint_partition_exact,
        "observation_shape_single_frame": list(zero_error.shape),
        "observation_dim_single_frame": int(zero_error[0].numel()),
        "zero_error_max_abs": zero_max,
        "quaternion_sign_error_max_abs": sign_max,
        "single_link_localization_error_max_abs": localization_error,
        "all_finite": finite,
        "sim_steps": 0,
        "policy_weights_loaded": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "permissions": {
            "static_tracking_smoke_unlocked": passed,
            "policy_training_unlocked": False,
            "deployment_unlocked": False,
        },
    }


def write_report(payload: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, REPORT)
    REPORT.with_name(f"{REPORT.name}.sha256").write_text(
        f"{sha256(REPORT)}  {REPORT.name}\n"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


try:
    write_report(main())
except Exception as error:  # pragma: no cover - evidence path
    traceback.print_exc()
    print(json.dumps({"error": type(error).__name__, "message": str(error)}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
