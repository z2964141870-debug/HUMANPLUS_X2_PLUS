#!/usr/bin/env python3
"""Select a coherent Stage250 boundary with a hard low-speed gate then pose proximity."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from official_x2.audit_stage250_native_dynamic_seed import ISAAC_JOINTS, decode_row


REPO = Path(__file__).resolve().parents[2]
P18 = REPO / "research/dynamic_retargeting_20260811/phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec = importlib.util.spec_from_file_location("p18", P18)
p18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p18)
QVEL_MAX_GATE = 0.1


def margin(point, centers, radius):
    equations = ConvexHull(centers).equations
    return float(np.min(
        -(equations[:, :2] @ point + equations[:, 2])
        / np.linalg.norm(equations[:, :2], axis=1)
    ) + radius)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--phase15", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-id", default="PHUMA-LUNGE-R-001")
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
    floor, helper = p18.physics.foot_geom_contract(model)
    feet = {
        side: sorted(
            geom for geom in helper[side]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[geom] != 0
        )
        for side in p18.SIDES
    }
    raw = joblib.load(args.motion)[args.motion_id]
    scale = json.loads(args.phase15.read_text())["intervention"]["hip_roll_scale"]
    initial = p18.init_entry(raw, scale, model, feet, floor)
    names = list(initial["joint_names_mujoco"])
    lower = [names.index(name) for name in p18.LOWER15]
    desired = 0.5 * (
        np.asarray(initial["dof"])[0, lower] + np.asarray(initial["dof"])[-1, lower]
    )

    trace = json.loads(args.rollout.read_text())["trace"]
    coherent = []
    eligible = []
    for index, row in enumerate(trace):
        future = trace[index:index + 51]
        if len(future) < 51 or not all(
            item["stage"] in ("stand", "move", "stop")
            and item["root_z_m"] >= 0.55 and item["root_tilt_rad"] <= 0.30
            for item in future
        ):
            continue
        qpos, qvel = decode_row(model, row, qpos_address, dof_address)
        qlower = np.asarray([qpos[qpos_address[name]] for name in p18.LOWER15])
        record = (
            float(np.linalg.norm(qlower - desired)), float(np.linalg.norm(qvel)),
            float(np.max(np.abs(qvel))), index, row, qpos, qvel,
        )
        coherent.append(record)
        if record[2] <= QVEL_MAX_GATE:
            eligible.append(record)
    if not eligible:
        raise RuntimeError("no coherent Stage250 anchor passes the fixed qvel gate")
    distance, qvel_l2, qvel_max, index, row, qpos, qvel = min(
        eligible, key=lambda item: (item[0], item[1], item[3])
    )
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
    result = {
        "stage": "BASE Phase43 Stage250 balanced coherent boundary",
        "execution": {"mj_forward_calls": 1, "mj_step_calls": 0, "gpu": False},
        "selection": {
            "coherent_anchors": len(coherent), "qvel_gate_radps": QVEL_MAX_GATE,
            "eligible_low_speed_anchors": len(eligible),
            "formula": "hard full-qvel max<=0.1rad/s, then minimum lower15 L2 to Phase15 endpoint mean, then qvel L2, then earliest index",
            "trace_index": index, "stage": row["stage"], "elapsed_s": row["elapsed_s"],
            "lower15_l2_to_phase15_endpoint_mean_rad": distance,
            "qvel_l2": qvel_l2, "qvel_max_abs": qvel_max,
        },
        "boundary": {
            "qpos": qpos.tolist(), "qvel": qvel.tolist(),
            "active12_signed_distance_min_m": {side: float(np.min(signed[side])) for side in p18.SIDES},
            "active12_signed_distance_p95_m": {side: float(np.percentile(np.abs(signed[side]), 95)) for side in p18.SIDES},
            "double_support_com_margin_m": margin(data.subtree_com[0, :2], centers, radius),
            "joint_limit_overshoot_max_rad": float(np.max(overshoot)),
            "root_z_m": float(qpos[2]), "root_tilt_rad": float(row["root_tilt_rad"]),
        },
        "decision": {},
    }
    boundary = result["boundary"]
    result["decision"] = {
        "balanced_1s_native_boundary_selected": True,
        "qvel_gate_pass": qvel_max <= QVEL_MAX_GATE,
        "joint_limits_pass": boundary["joint_limit_overshoot_max_rad"] <= 1e-9,
        "double_support_margin_pass": boundary["double_support_com_margin_m"] >= 0.0,
        "physics_unlocked": False, "training_unlocked": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selection": result["selection"], "boundary": boundary, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
