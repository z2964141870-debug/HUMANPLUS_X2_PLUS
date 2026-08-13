#!/usr/bin/env python3
"""Centroidal/contact-force existence oracle for the frozen Phase27 path."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p18 = load_module("p18", HERE / "phase18_joint_contact_generator/run_phase18_joint_contact_generator.py")
force_oracle = load_module("force_oracle", REPO / "tools/retarget/run_x2_wbt_centroidal_force_feasibility_phase41.py")

DT = 0.02
CONTACT_LOWER = -0.00001
CONTACT_UPPER = 0.0005


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    boundary = json.loads(args.boundary.read_text())["boundary"]
    source = json.loads(args.path.read_text())
    if not source["decision"]["left_half_continuous_path_complete"]:
        raise RuntimeError("Phase27 path is not complete")
    frames = source["frames"]
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
    base_qpos = np.asarray(boundary["qpos"], dtype=float)
    qpos = np.tile(base_qpos, (len(frames), 1))
    for index, frame in enumerate(frames):
        qpos[index, :3] = np.asarray(frame["root"])
        quat = np.asarray(frame["quat_xyzw"])
        qpos[index, 3:7] = quat[[3, 0, 1, 2]]
        q = base_qpos[qadr].copy()
        q[lower] = np.asarray(frame["lower_q"])
        qpos[index, qadr] = q

    qvel = force_oracle.differentiate_qpos(model, qpos, 1.0 / DT)
    com, angular_momentum = force_oracle.model_centroidal_series(model, qpos, qvel)
    com_acceleration = (com[2:] - 2.0 * com[1:-1] + com[:-2]) / DT**2
    hdot = (angular_momentum[2:] - angular_momentum[:-2]) / (2.0 * DT)
    contacts = force_oracle.contact_candidates(model, qpos, floor, feet)
    friction = float(min(
        model.geom_friction[floor, 0],
        *[model.geom_friction[geom, 0] for geoms in feet.values() for geom in geoms],
    ))
    mass = float(np.sum(model.body_mass))
    gravity = np.asarray(model.opt.gravity, dtype=float)

    frame_results = []
    for frame in range(1, len(qpos) - 1):
        candidates = []
        geometry = {}
        geometry_ok = True
        for side in p18.SIDES:
            info = contacts[frame][side]
            distances = np.asarray(info["distance"])
            indices = np.flatnonzero((distances >= CONTACT_LOWER) & (distances <= CONTACT_UPPER))
            if float(np.min(distances)) < CONTACT_LOWER - 1e-12:
                geometry_ok = False
            for sphere in indices:
                candidates.append((side, int(sphere), np.asarray(info["point"])[int(sphere)]))
            geometry[side] = {
                "candidate_indices": indices.tolist(),
                "min_signed_distance_m": float(np.min(distances)),
            }
        if not candidates:
            geometry_ok = False
        force = force_oracle.solve_frame_force_lp(
            mass, gravity, com[frame], com_acceleration[frame - 1],
            hdot[frame - 1], candidates, friction,
        )
        frame_results.append({
            "frame": frame, "time_s": frame * DT,
            "geometry": geometry, "geometry_qualified": geometry_ok,
            "candidate_count": len(candidates),
            "force_feasible": bool(force["feasible"]),
            "solver_status": int(force["solver_status"]),
            "target_force_n": np.asarray(force.get("target_force_n", [])).tolist(),
            "target_moment_nm": np.asarray(force.get("target_moment_nm", [])).tolist(),
            "jointly_feasible": bool(geometry_ok and force["feasible"]),
        })

    geometry_count = sum(item["geometry_qualified"] for item in frame_results)
    force_count = sum(item["force_feasible"] for item in frame_results)
    joint_count = sum(item["jointly_feasible"] for item in frame_results)
    com_norm = np.linalg.norm(com_acceleration, axis=1)
    hdot_norm = np.linalg.norm(hdot, axis=1)
    evaluated = len(frame_results)
    passed = joint_count == evaluated
    result = {
        "stage": "Phase28 left-half centroidal/contact-force feasibility",
        "execution": {"evaluated_frames": evaluated, "mj_step_calls": 0, "gpu": False},
        "model": {"mass_kg": mass, "friction_mu": friction, "contact_band_m": [CONTACT_LOWER, CONTACT_UPPER]},
        "summary": {
            "geometry_qualified_frames": geometry_count,
            "force_lp_feasible_frames": force_count,
            "jointly_feasible_frames": joint_count,
            "com_acceleration_norm_p95_max_mps2": [float(np.percentile(com_norm, 95)), float(np.max(com_norm))],
            "centroidal_hdot_norm_p95_max_nms": [float(np.percentile(hdot_norm, 95)), float(np.max(hdot_norm))],
        },
        "frames": frame_results,
        "decision": {
            "centroidal_force_preflight_pass": passed,
            "dynamics_truth": False, "physics_unlocked": False, "training_unlocked": False,
            "next": "review before raw physics" if passed else "stop; do not run physics or tune force/contact gates",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"execution": result["execution"], "summary": result["summary"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
