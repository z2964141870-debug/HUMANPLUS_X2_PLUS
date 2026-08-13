#!/usr/bin/env python3
"""Qualify the stable Phase34 closed trajectory as a native DSMS warm start.

This is deliberately a preflight only.  It executes one deterministic direct
replay of the frozen 340 ms suffix, constructs the position-PD targets exactly
equivalent to the recorded motor torques, and does not instantiate IPOPT or an
optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from official_x2.analyze_phase34_full_closed_trace import read_trace, sha256
from official_x2.replay_official_trace_direct_mujoco import JOINTS, pd_gains, yaw_tilt


REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_contract.json"
PHASE34 = REPO / "reports/official_x2/phase34_closed_full_trace_manifest.json"
PHASE37 = REPO / "reports/official_x2/phase37_three_event_teacher_prereg.json"
OUTPUT = REPO / "research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_result.json"
HORIZON_STEPS = 340
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")


def _load_assets() -> tuple[dict[str, Any], dict[str, Path]]:
    phase34 = json.loads(PHASE34.read_text(encoding="utf-8"))
    artifacts = phase34["artifacts"]
    paths = {"mmap": Path(artifacts["mmap"]), "rollout": Path(artifacts["rollout"]), "scene": SCENE}
    expected_hashes = {
        "mmap": artifacts["mmap_sha256"],
        "rollout": artifacts["rollout_sha256"],
        "scene": phase34["provenance"]["scene_sha256"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != expected_hashes[name]:
            raise RuntimeError(f"immutable Phase34 {name} hash mismatch")
    phase34["resolved_asset_hashes"] = expected_hashes
    return phase34, paths


def _joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos, qvel, actuator = [], [], []
    for name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"missing official joint/actuator: {name}")
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qvel.append(int(model.jnt_dofadr[joint_id]))
        actuator.append(actuator_id)
    return np.asarray(qpos), np.asarray(qvel), np.asarray(actuator)


def _qpos_qvel_only_replay(model: mujoco.MjModel, physical: dict[str, Any], anchor: int) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.time = float(physical["time"][anchor])
    data.qpos[:] = physical["qpos"][anchor]
    data.qvel[:] = physical["qvel"][anchor]
    data.ctrl[:] = physical["ctrl"][anchor]
    # DSMS shooting state is qpos/qvel.  Deliberately do not restore solver
    # warm-start/contact workspace here: this tests the actual representation.
    mujoco.mj_forward(model, data)
    qpos_error, qvel_error = 0.0, 0.0
    root_z, root_tilt = [], []
    for step in range(HORIZON_STEPS):
        target_index = anchor + 1 + step
        data.ctrl[:] = physical["ctrl"][target_index]
        mujoco.mj_step(model, data)
        qpos_error = max(qpos_error, float(np.max(np.abs(data.qpos - physical["qpos"][target_index]))))
        qvel_error = max(qvel_error, float(np.max(np.abs(data.qvel - physical["qvel"][target_index]))))
        root_z.append(float(data.qpos[2]))
        root_tilt.append(float(yaw_tilt(data.qpos[3:7])[1]))
    return {
        "steps": HORIZON_STEPS,
        "qpos_absmax": qpos_error,
        "qvel_absmax": qvel_error,
        "root_z_min_m": float(np.min(root_z)),
        "root_tilt_max_rad": float(np.max(root_tilt)),
    }


def _equivalent_pd_targets(model: mujoco.MjModel, physical: dict[str, Any], anchor: int) -> dict[str, Any]:
    qadr, dadr, actuator = _joint_addresses(model)
    gains = pd_gains()
    kp = np.asarray([gains[name][0] for name in JOINTS], dtype=np.float64)
    kd = np.asarray([gains[name][1] for name in JOINTS], dtype=np.float64)
    targets, reconstructed, recorded, applied_force = [], [], [], []
    limit_overshoot = 0.0
    ctrl_range_overshoot = 0.0
    for step in range(HORIZON_STEPS):
        pre = anchor + step
        applied = anchor + 1 + step
        q = physical["qpos"][pre, qadr]
        qd = physical["qvel"][pre, dadr]
        ctrl = physical["ctrl"][applied, actuator]
        target = q + (ctrl + kd * qd) / kp
        raw = kp * (target - q) - kd * qd
        reconstructed.append(raw)
        recorded.append(ctrl)
        applied_force.append(physical["qfrc_actuator"][applied, dadr])
        targets.append(target)
        joint_range = model.jnt_range[model.actuator_trnid[actuator, 0]]
        limit_overshoot = max(limit_overshoot, float(np.max(np.maximum(joint_range[:, 0] - target, target - joint_range[:, 1]))))
        ctrl_range = model.actuator_ctrlrange[actuator]
        ctrl_range_overshoot = max(ctrl_range_overshoot, float(np.max(np.maximum(ctrl_range[:, 0] - ctrl, ctrl - ctrl_range[:, 1]))))
    targets_array = np.asarray(targets)
    reconstructed_array = np.asarray(reconstructed)
    recorded_array = np.asarray(recorded)
    clipped_array = np.clip(recorded_array, model.actuator_ctrlrange[actuator, 0], model.actuator_ctrlrange[actuator, 1])
    reconstruction_error = float(np.max(np.abs(reconstructed_array - recorded_array)))
    applied_force_error = float(np.max(np.abs(clipped_array - np.asarray(applied_force))))
    return {
        "shape": list(targets_array.shape),
        "target_sha256": hashlib.sha256(targets_array.tobytes()).hexdigest(),
        "ctrl_reconstruction_absmax": reconstruction_error,
        "joint_limit_overshoot_rad": max(0.0, limit_overshoot),
        "recorded_ctrl_range_overshoot": max(0.0, ctrl_range_overshoot),
        "recorded_ctrl_overshoot_count": int(np.sum((recorded_array < model.actuator_ctrlrange[actuator, 0]) | (recorded_array > model.actuator_ctrlrange[actuator, 1]))),
        "clipped_ctrl_vs_qfrc_actuator_absmax": applied_force_error,
        "target_absmax_rad": float(np.max(np.abs(targets_array))),
    }


def run() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    phase34, paths = _load_assets()
    phase37 = json.loads(PHASE37.read_text(encoding="utf-8"))
    expected = contract["source"]["selection"]
    observed = phase37["selection"]
    for key, value in expected.items():
        if observed[key] != value:
            raise RuntimeError(f"frozen Phase37 selection drift for {key}: {observed[key]} != {value}")
    if mujoco.mj_versionString() != contract["preflight_gates"]["mujoco_version"]:
        raise RuntimeError(f"MuJoCo 3.3.7 required, got {mujoco.mj_versionString()}")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    if model.nq != 38 or model.nv != 37 or model.nu != 31:
        raise RuntimeError(f"official model shape drift: {model.nq}/{model.nv}/{model.nu}")
    # Official x2.xml actuators must remain raw motors for the warm-start replay.
    raw_motor = bool(np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE))
    physical = read_trace(paths["mmap"])
    anchor = int(expected["anchor_physics_index"])
    if anchor + HORIZON_STEPS >= len(physical["qpos"]):
        raise RuntimeError("frozen native suffix is incomplete")
    replay = _qpos_qvel_only_replay(model, physical, anchor)
    pd = _equivalent_pd_targets(model, physical, anchor)
    gates = contract["preflight_gates"]
    checks = {
        "asset_hashes_exact": True,
        "mujoco_version_exact": mujoco.mj_versionString() == gates["mujoco_version"],
        "raw_motor_model": raw_motor,
        "qpos_replay": replay["qpos_absmax"] <= gates["qpos_qvel_only_replay_qpos_absmax"],
        "qvel_replay": replay["qvel_absmax"] <= gates["qpos_qvel_only_replay_qvel_absmax"],
        "recorded_ctrl_applied_force_match": pd["clipped_ctrl_vs_qfrc_actuator_absmax"] <= gates["recorded_ctrl_applied_force_match_absmax"],
        "pd_equivalent_ctrl": pd["ctrl_reconstruction_absmax"] <= gates["pd_equivalent_target_ctrl_absmax"],
        "pd_equivalent_target_within_joint_limits": pd["joint_limit_overshoot_rad"] <= gates["pd_equivalent_target_joint_limit_overshoot_rad"],
        "root_z_safe": replay["root_z_min_m"] >= gates["initial_root_z_min_m"],
        "root_tilt_safe": replay["root_tilt_max_rad"] <= gates["initial_root_tilt_max_rad"],
    }
    passed = bool(all(checks.values()))
    result = {
        "stage": "Dynamic Retargeting Phase32",
        "scope": "one deterministic 340 ms native-suffix replay; no IPOPT/optimizer/training/GPU/closed ROS",
        "assets": {
            name: {"path": str(path), "sha256": phase34["resolved_asset_hashes"][name]}
            for name, path in paths.items()
        },
        "contract_sha256": sha256(CONTRACT),
        "selection": expected,
        "model": {"mujoco": mujoco.mj_versionString(), "nq": model.nq, "nv": model.nv, "nu": model.nu},
        "replay": replay,
        "pd_equivalent_targets": pd,
        "checks": checks,
        "execution": {"mj_step_calls": HORIZON_STEPS, "optimizer_instances": 0, "optimizer_steps": 0, "gpu": False, "cpu_threads": 1},
        "decision": {
            "native_dsms_warmstart_qualified": passed,
            "single_short_solve_authorized": False,
            "training_unlocked": False,
            "result": "NATIVE_DSMS_PREFLIGHT_PASSED_SOLVE_STILL_LOCKED" if passed else "NATIVE_DSMS_PREFLIGHT_REJECTED_NO_SOLVE",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replay": result["replay"], "pd": result["pd_equivalent_targets"], "checks": result["checks"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
