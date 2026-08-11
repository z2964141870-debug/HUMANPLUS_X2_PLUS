#!/usr/bin/env python3
"""Zero-physics kinematic authority audit for a shared task-space bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "tools")]
import retarget.run_x2_forefoot_official_physics_screen as physics  # noqa: E402


LOWER15 = [
    *[f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
        "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"
    )],
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_report(matrix: np.ndarray) -> dict:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
    rank = int(np.sum(singular > tolerance))
    return {
        "shape": list(matrix.shape), "rank": rank,
        "nullity_in_lower15": int(matrix.shape[1] - rank),
        "singular_values": singular.tolist(),
        "condition_nonzero": None if rank == 0 else float(singular[0] / singular[rank - 1]),
    }


def run(scene: Path, reset: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    archive = np.load(reset, allow_pickle=False)
    data.qpos[:] = archive["qpos"]
    data.qvel[:] = archive["qvel"]
    mujoco.mj_forward(model, data)

    floor, helper = physics.foot_geom_contract(model)
    feet = {
        side: sorted(
            geom for geom in geoms
            if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and int(model.geom_contype[geom]) != 0
        )
        for side, geoms in helper.items()
    }
    if any(len(geoms) != 12 for geoms in feet.values()):
        raise ValueError(f"active12 contract failed: { {key: len(value) for key, value in feet.items()} }")
    dofs = np.asarray([
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in LOWER15
    ])

    foot_jac, foot_xyz = {}, {}
    for side, geoms in feet.items():
        matrices, positions = [], []
        for geom in geoms:
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacGeom(model, data, jacp, jacr, geom)
            matrices.append(jacp[:, dofs])
            positions.append(data.geom_xpos[geom])
        foot_jac[side] = np.mean(matrices, axis=0)
        foot_xyz[side] = np.mean(positions, axis=0)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    com_jac_full = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSubtreeCom(model, data, com_jac_full, pelvis)
    com_jac = com_jac_full[:, dofs]
    com = data.subtree_com[pelvis].copy()
    right_stance = foot_jac["right"]
    feet6 = np.vstack((right_stance, foot_jac["left"]))
    task8 = np.vstack((right_stance, foot_jac["left"], com_jac[:2]))
    clearance = {
        side: [float(data.geom_xpos[geom, 2] - model.geom_size[geom, 0] - data.geom_xpos[floor, 2]) for geom in geoms]
        for side, geoms in feet.items()
    }
    com_to_right = com[:2] - foot_xyz["right"][:2]
    desired = np.r_[np.zeros(6), -com_to_right]
    linear_delta, *_ = np.linalg.lstsq(task8, desired, rcond=None)
    linear_residual = task8 @ linear_delta - desired
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in LOWER15
    ])
    qpos_addresses = model.jnt_qposadr[joint_ids]
    projected_q = data.qpos[qpos_addresses] + linear_delta
    limits = model.jnt_range[joint_ids]
    overshoot = np.maximum(limits[:, 0] - projected_q, 0.0) + np.maximum(projected_q - limits[:, 1], 0.0)
    result = {
        "stage": "dynamic retargeting Phase12 task-space/load bridge authority audit",
        "execution": {"mj_forward_calls": 1, "mj_step_calls": 0, "optimizer_steps": 0, "gpu": False},
        "assets": {"scene": str(scene), "scene_sha256": sha256(scene), "phase6_reset": str(reset), "phase6_reset_sha256": sha256(reset)},
        "contract": {
            "controlled_joints": LOWER15, "controlled_dof_count": 15, "head_controlled": False,
            "active_sole_spheres": {side: geoms for side, geoms in feet.items()},
            "task_definition": "right stance centroid xyz + left swing centroid xyz + robot COM xy",
        },
        "reset_state": {
            "root_position_m": data.qpos[:3].tolist(), "com_position_m": com.tolist(),
            "foot_centroid_m": {side: value.tolist() for side, value in foot_xyz.items()},
            "com_minus_right_support_xy_m": com_to_right.tolist(),
            "com_minus_right_support_xy_norm_m": float(np.linalg.norm(com_to_right)),
            "sole_clearance_min_mm": {side: float(1000 * min(values)) for side, values in clearance.items()},
        },
        "authority": {
            "right_stance_xyz": rank_report(right_stance),
            "both_feet_xyz": rank_report(feet6),
            "both_feet_xyz_plus_com_xy": rank_report(task8),
        },
        "linearized_full_transfer_with_both_feet_fixed": {
            "desired_com_shift_xy_m": (-com_to_right).tolist(),
            "lower15_delta_rad": linear_delta.tolist(),
            "delta_rms_rad": float(np.sqrt(np.mean(linear_delta**2))),
            "delta_max_abs_rad": float(np.max(np.abs(linear_delta))),
            "task_residual_max_abs_m": float(np.max(np.abs(linear_residual))),
            "joint_limit_violation_count": int(np.sum(overshoot > 1e-9)),
            "joint_limit_overshoot_max_rad": float(np.max(overshoot)),
        },
    }
    combined = result["authority"]["both_feet_xyz_plus_com_xy"]
    transfer = result["linearized_full_transfer_with_both_feet_fixed"]
    result["decision"] = {
        "kinematic_task_authority_available": bool(combined["rank"] == 8 and combined["nullity_in_lower15"] >= 7),
        "linearized_full_right_support_transfer_with_fixed_feet_plausible": bool(
            transfer["delta_max_abs_rad"] <= 0.35
            and transfer["joint_limit_violation_count"] == 0
            and transfer["task_residual_max_abs_m"] <= 1e-6
        ),
        "physics_or_teacher_unlocked": False,
        "next_controller_contract": "online damped task-space update from current state; right stance anchor + explicit left swing clearance + COM/load transfer, then raw official-PD replay",
        "hard_stop": "fail if task Jacobian rank<8, zero-residual replay differs, stable left-off/right-on<100ms, support speed/excursion/root-acc/safety fail",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.scene, args.reset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"authority": result["authority"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
