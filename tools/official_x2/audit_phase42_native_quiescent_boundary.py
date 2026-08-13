#!/usr/bin/env python3
"""Select the minimum-velocity coherent 1s-safe Stage250 native boundary."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from official_x2.audit_stage250_native_dynamic_seed import ISAAC_JOINTS, decode_row


REPO = Path(__file__).resolve().parents[2]
P18 = REPO / "research/dynamic_retargeting_20260811/phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec = importlib.util.spec_from_file_location("p18", P18)
p18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p18)


def margin(point: np.ndarray, centers: np.ndarray, radius: float) -> float:
    equations = ConvexHull(centers).equations
    return float(np.min(
        -(equations[:, :2] @ point + equations[:, 2])
        / np.linalg.norm(equations[:, :2], axis=1)
    ) + radius)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--phase41", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    qpos_address = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in ISAAC_JOINTS
    }
    dof_address = {
        name: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in ISAAC_JOINTS
    }
    trace = json.loads(args.rollout.read_text())["trace"]
    candidates = []
    for index, row in enumerate(trace):
        future = trace[index:index + 51]
        if len(future) < 51 or not all(
            item["stage"] in ("stand", "move", "stop")
            and item["root_z_m"] >= 0.55 and item["root_tilt_rad"] <= 0.30
            for item in future
        ):
            continue
        qpos, qvel = decode_row(model, row, qpos_address, dof_address)
        candidates.append((float(np.linalg.norm(qvel)), float(np.max(np.abs(qvel))), index, row, qpos, qvel))
    qvel_l2, qvel_max, index, row, qpos, qvel = min(candidates, key=lambda item: (item[0], item[1], item[2]))

    floor, helper = p18.physics.foot_geom_contract(model)
    feet = {
        side: sorted(
            geom for geom in helper[side]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[geom] != 0
        )
        for side in p18.SIDES
    }
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    signed = {
        side: data.geom_xpos[feet[side], 2] - model.geom_size[feet[side], 0] - data.geom_xpos[floor, 2]
        for side in p18.SIDES
    }
    centers = np.concatenate([data.geom_xpos[feet[side], :2] for side in p18.SIDES])
    radius = float(model.geom_size[feet["left"][0], 0])
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ISAAC_JOINTS
    ])
    q = np.asarray([qpos[qpos_address[name]] for name in ISAAC_JOINTS])
    limits = model.jnt_range[joint_ids]
    overshoot = np.maximum(limits[:, 0] - q, 0.0) + np.maximum(q - limits[:, 1], 0.0)
    old = json.loads(args.phase41.read_text())
    old_qvel = np.asarray(old["boundary"]["qvel"])
    result = {
        "stage": "BASE Phase42 Stage250 quiescent coherent boundary",
        "execution": {"mj_forward_calls": 1, "mj_step_calls": 0, "gpu": False},
        "selection": {
            "eligible_coherent_anchors": len(candidates),
            "formula": "minimum full generalized qvel L2; then max-abs; then earliest trace index; no solver-result selection",
            "trace_index": index, "stage": row["stage"], "elapsed_s": row["elapsed_s"],
            "qvel_l2": qvel_l2, "qvel_max_abs": qvel_max,
            "phase41_qvel_l2": float(np.linalg.norm(old_qvel)),
            "phase41_qvel_max_abs": float(np.max(np.abs(old_qvel))),
        },
        "boundary": {
            "qpos": qpos.tolist(), "qvel": qvel.tolist(),
            "active12_signed_distance_min_m": {side: float(np.min(signed[side])) for side in p18.SIDES},
            "active12_signed_distance_p95_m": {side: float(np.percentile(np.abs(signed[side]), 95)) for side in p18.SIDES},
            "double_support_com_margin_m": margin(data.subtree_com[pelvis, :2], centers, radius),
            "joint_limit_overshoot_max_rad": float(np.max(overshoot)),
            "root_z_m": float(qpos[2]), "root_tilt_rad": float(row["root_tilt_rad"]),
        },
        "decision": {},
    }
    boundary = result["boundary"]
    result["decision"] = {
        "quiescent_1s_native_boundary_selected": True,
        "velocity_improved_vs_phase41": qvel_l2 < float(np.linalg.norm(old_qvel)),
        "joint_limits_pass": boundary["joint_limit_overshoot_max_rad"] <= 1e-9,
        "double_support_margin_pass": boundary["double_support_com_margin_m"] >= 0.0,
        "physics_unlocked": False, "training_unlocked": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selection": result["selection"], "boundary": boundary, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
