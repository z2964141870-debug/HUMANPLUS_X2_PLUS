#!/usr/bin/env python3
"""Deterministically time-parameterize the Phase22 contact skeleton; no physics."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent
P18 = HERE / "phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec = importlib.util.spec_from_file_location("p18", P18)
p18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p18)

DT = 0.02
TRANSITION_S = 0.50
HOLD_S = 0.12


def smooth5(u: float) -> float:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def interpolate(a: dict, b: dict, u: float) -> dict:
    s = smooth5(u)
    rotations = Rotation.from_quat(np.asarray([a["quat_xyzw"], b["quat_xyzw"]]))
    quat = Slerp([0.0, 1.0], rotations)([s]).as_quat()[0]
    return {
        "root": (1.0 - s) * np.asarray(a["root"]) + s * np.asarray(b["root"]),
        "quat": quat,
        "lower": (1.0 - s) * np.asarray(a["lower_q"]) + s * np.asarray(b["lower_q"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    boundary = json.loads(args.boundary.read_text())["boundary"]
    skeleton = json.loads(args.skeleton.read_text())
    if not skeleton["decision"]["contact_skeleton_complete"]:
        raise RuntimeError("Phase22 contact skeleton is not complete")

    floor, helper = p18.physics.foot_geom_contract(model)
    feet = {
        side: sorted(
            geom for geom in helper[side]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[geom] != 0
        )
        for side in p18.SIDES
    }
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            names.append(name)
    qadr = np.asarray([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in names
    ])
    lower = np.asarray([names.index(name) for name in p18.LOWER15])
    base_q = np.asarray(boundary["qpos"], dtype=float)[qadr]
    nodes = skeleton["steps"]

    samples = []
    hold_count = int(round(HOLD_S / DT))
    transition_count = int(round(TRANSITION_S / DT))

    def append_state(state: dict, phase: str, node_index: int, active: list, swing: list) -> None:
        samples.append({
            "state": state,
            "phase": phase,
            "node_index": node_index,
            "active": list(active),
            "swing": list(swing),
        })

    first = {
        "root": np.asarray(nodes[0]["root"]),
        "quat": np.asarray(nodes[0]["quat_xyzw"]),
        "lower": np.asarray(nodes[0]["lower_q"]),
    }
    append_state(first, "hold", 0, nodes[0]["active"], nodes[0]["swing"])
    for _ in range(hold_count):
        append_state(first, "hold", 0, nodes[0]["active"], nodes[0]["swing"])
    for index in range(len(nodes) - 1):
        for tick in range(1, transition_count + 1):
            state = interpolate(nodes[index], nodes[index + 1], tick / transition_count)
            append_state(state, "transition", index + 1, [], [])
        endpoint = {
            "root": np.asarray(nodes[index + 1]["root"]),
            "quat": np.asarray(nodes[index + 1]["quat_xyzw"]),
            "lower": np.asarray(nodes[index + 1]["lower_q"]),
        }
        for _ in range(hold_count):
            append_state(
                endpoint, "hold", index + 1,
                nodes[index + 1]["active"], nodes[index + 1]["swing"],
            )

    roots, lower_q, signed_rows = [], [], []
    hold_contact_errors, swing_clearances = [], []
    frame_rows = []
    for frame, sample in enumerate(samples):
        state = sample["state"]
        q = base_q.copy()
        q[lower] = state["lower"]
        data = mujoco.MjData(model)
        data.qpos[:3] = state["root"]
        data.qpos[3:7] = state["quat"][[3, 0, 1, 2]]
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        signed = {
            side: np.asarray([
                data.geom_xpos[g, 2] - model.geom_size[g, 0] - data.geom_xpos[floor, 2]
                for g in feet[side]
            ])
            for side in p18.SIDES
        }
        roots.append(state["root"])
        lower_q.append(state["lower"])
        signed_rows.append(np.r_[signed["left"], signed["right"]])
        if sample["phase"] == "hold":
            for side in sample["active"]:
                hold_contact_errors.append(float(np.min(np.abs(signed[side] - 0.00025))))
            for side in sample["swing"]:
                swing_clearances.append(float(np.min(signed[side])))
        frame_rows.append({
            "frame": frame,
            "time_s": frame * DT,
            "phase": sample["phase"],
            "node_index": sample["node_index"],
            "active": sample["active"],
            "swing": sample["swing"],
            "active12_min_signed_m": {side: float(np.min(signed[side])) for side in p18.SIDES},
        })

    roots = np.asarray(roots)
    lower_q = np.asarray(lower_q)
    signed_rows = np.asarray(signed_rows)
    joint_steps = np.max(np.abs(np.diff(lower_q, axis=0)), axis=1)
    root_acc = np.diff(roots[:, :2], n=2, axis=0) / DT**2
    root_acc_norm = np.linalg.norm(root_acc, axis=1)
    metrics = {
        "duration_s": (len(samples) - 1) * DT,
        "frames": len(samples),
        "all_active12_min_signed_m": float(np.min(signed_rows)),
        "hold_stance_contact_error_max_m": float(max(hold_contact_errors)),
        "single_support_hold_swing_clearance_min_m": float(min(swing_clearances)),
        "joint_step_p95_rad": float(np.percentile(joint_steps, 95)),
        "joint_step_max_rad": float(np.max(joint_steps)),
        "root_horizontal_acceleration_max_mps2": float(np.max(root_acc_norm)),
    }
    gates = {
        "nonpenetration": metrics["all_active12_min_signed_m"] >= -0.0005,
        "hold_stance_contact": metrics["hold_stance_contact_error_max_m"] <= 0.0005,
        "single_support_swing_clearance": metrics["single_support_hold_swing_clearance_min_m"] >= 0.012,
        "joint_step_p95": metrics["joint_step_p95_rad"] <= 0.1,
        "joint_step_max": metrics["joint_step_max_rad"] <= 0.15,
        "root_horizontal_acceleration": metrics["root_horizontal_acceleration_max_mps2"] <= 4.0,
    }
    passed = all(gates.values())
    result = {
        "stage": "Phase23 time-parameterized contact skeleton",
        "execution": {"mj_forward_calls": len(samples), "mj_step_calls": 0, "gpu": False},
        "timing": {"dt_s": DT, "transition_s": TRANSITION_S, "hold_s": HOLD_S},
        "metrics": metrics,
        "gates": gates,
        "frames": frame_rows,
        "decision": {
            "time_parameterized_skeleton_pass": passed,
            "physics_unlocked": False,
            "training_unlocked": False,
            "next": "add forward semantics as the only new layer" if passed else "stop; do not alter timing or keyframes",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"execution": result["execution"], "metrics": metrics, "gates": gates, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
