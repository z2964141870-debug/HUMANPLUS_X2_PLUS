#!/usr/bin/env python3
"""Audit Stage250 official traces as X2-native dynamic-seed candidates.

This is an offline audit.  It decodes the exact 93-D Stage208 observation,
reconstructs an official-MJCF state, computes FK/centroidal diagnostics and a
finite-difference inverse-dynamics *model estimate* of foot GRF.  Estimated
forces are never presented as AimDK, force-sensor, or hardware truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.optimize import nnls


ISAAC_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)

LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)

LOWER_SCALE = np.asarray(
    [0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
     0.4, 0.4, 0.4, 0.4, 0.12, 0.08, 0.4, 0.16, 0.16],
    dtype=np.float64,
)

OBS_SLICES = {
    "base_lin_vel_body": slice(0, 3),
    "base_ang_vel_body": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "command": slice(9, 12),
    "joint_position_offset": slice(12, 43),
    "joint_velocity": slice(43, 74),
    "previous_action": slice(74, 89),
    "gait": slice(89, 93),
}


def default_pose() -> dict[str, float]:
    pose = {name: 0.0 for name in ISAAC_JOINTS}
    for side in ("left", "right"):
        pose[f"{side}_hip_pitch_joint"] = -0.248
        pose[f"{side}_knee_joint"] = 0.5303
        pose[f"{side}_ankle_pitch_joint"] = -0.2823
        pose[f"{side}_shoulder_pitch_joint"] = 0.4
        pose[f"{side}_elbow_joint"] = -1.2
    return pose


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def quaternion_wxyz_from_yaw_gravity(yaw: float, gravity: Iterable[float]) -> np.ndarray:
    """Recover ZYX q from yaw and R^T [0,0,-1]."""
    gx, gy, gz = np.asarray(tuple(gravity), dtype=np.float64)
    pitch = math.asin(float(np.clip(gx, -1.0, 1.0)))
    roll = math.atan2(-gy, -gz)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    return np.asarray([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], dtype=np.float64)


def projected_gravity(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(tuple(quaternion_wxyz), dtype=np.float64)
    return np.asarray([
        2.0 * (-z * x + w * y),
        -2.0 * (z * y + w * x),
        1.0 - 2.0 * (w * w + z * z),
    ], dtype=np.float64)


def yaw_from_quaternion(quaternion_wxyz: Iterable[float]) -> float:
    w, x, y, z = np.asarray(tuple(quaternion_wxyz), dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotation_body_to_world(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(tuple(q), dtype=np.float64)
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def decode_row(
    model: mujoco.MjModel,
    row: dict,
    qpos_adr: dict[str, int],
    dof_adr: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(row["obs"], dtype=np.float64)
    if obs.shape != (93,):
        raise ValueError(f"expected 93-D observation, got {obs.shape}")
    pose = default_pose()
    quaternion = quaternion_wxyz_from_yaw_gravity(
        float(row["root_yaw_rad"]), obs[OBS_SLICES["projected_gravity"]]
    )
    qpos = np.zeros(model.nq, dtype=np.float64)
    qvel = np.zeros(model.nv, dtype=np.float64)
    qpos[:3] = [row["root_x_m"], row["root_y_m"], row["root_z_m"]]
    qpos[3:7] = quaternion
    qvel[:3] = rotation_body_to_world(quaternion) @ obs[OBS_SLICES["base_lin_vel_body"]]
    # This matches the verified direct official-MJCF runner contract: the
    # vendor torso IMU angular velocity is used in free-joint qvel[3:6].
    qvel[3:6] = obs[OBS_SLICES["base_ang_vel_body"]]
    offsets = obs[OBS_SLICES["joint_position_offset"]]
    velocities = obs[OBS_SLICES["joint_velocity"]]
    for index, name in enumerate(ISAAC_JOINTS):
        qpos[qpos_adr[name]] = pose[name] + offsets[index]
        qvel[dof_adr[name]] = velocities[index]
    return qpos, qvel


def pd_gains(name: str) -> tuple[float, float]:
    if name in LOWER_JOINTS:
        return (40.0, 20.0) if "ankle" in name else (300.0, 20.0)
    if "shoulder" in name or "elbow" in name:
        return 40.0, 5.0
    if "wrist" in name:
        return 30.0, 3.0
    return 50.0, 5.0


def normalized_angle_error(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def reconstruct_trace(model: mujoco.MjModel, rows: list[dict]) -> dict:
    valid = [row for row in rows if len(row.get("obs", [])) == 93 and len(row.get("action", [])) == 15]
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dof_adr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    qpos, qvel = zip(*(decode_row(model, row, qpos_adr, dof_adr) for row in valid))
    return {
        "rows": valid,
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "qpos_adr": qpos_adr,
        "dof_adr": dof_adr,
    }


def contact_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    body_name = f"{side}_ankle_roll_link"
    return [
        gid for gid in range(model.ngeom)
        if model.body(int(model.geom_bodyid[gid])).name == body_name
        and int(model.geom_contype[gid]) != 0
        and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        and float(model.geom_pos[gid, 2]) < -0.05
    ]


def foot_points(model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]) -> dict[str, np.ndarray]:
    centers = data.geom_xpos[geom_ids].copy()
    radii = model.geom_size[geom_ids, 0]
    bottom = centers.copy()
    bottom[:, 2] -= radii
    local_x = model.geom_pos[geom_ids, 0]
    rear = bottom[local_x <= np.median(local_x)].mean(axis=0)
    front = bottom[local_x > np.median(local_x)].mean(axis=0)
    return {
        "center": bottom.mean(axis=0),
        "rear": rear,
        "front": front,
        "min_z": np.asarray([np.min(bottom[:, 2])]),
    }


def friction_pyramid_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points: list[tuple[np.ndarray, int]],
    mu: float = 0.6,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    columns: list[np.ndarray] = []
    labels: list[tuple[str, int]] = []
    rays = (
        np.asarray([mu, mu, 1.0]), np.asarray([mu, -mu, 1.0]),
        np.asarray([-mu, mu, 1.0]), np.asarray([-mu, -mu, 1.0]),
    )
    for point_index, (point, body_id) in enumerate(points):
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
        for ray_index, ray in enumerate(rays):
            columns.append(jacp.T @ ray)
            labels.append(("left" if point_index < 2 else "right", ray_index))
    return np.column_stack(columns), labels


def contiguous_cycles(left: np.ndarray, right: np.ndarray) -> int:
    states = np.where(left & right, 0, np.where(left, 1, np.where(right, 2, 3)))
    compressed = [int(states[0])] if states.size else []
    for state in states[1:]:
        if int(state) != compressed[-1]:
            compressed.append(int(state))
    return sum(
        1 for i in range(1, len(compressed) - 1)
        if compressed[i - 1] == 0 and compressed[i] in (1, 2) and compressed[i + 1] == 0
    )


def analyze_case(model: mujoco.MjModel, payload: dict) -> dict:
    reconstructed = reconstruct_trace(model, payload["trace"])
    rows = reconstructed["rows"]
    qpos = reconstructed["qpos"]
    qvel = reconstructed["qvel"]
    qpos_adr = reconstructed["qpos_adr"]
    dof_adr = reconstructed["dof_adr"]
    dt = 0.02
    data = mujoco.MjData(model)
    pelvis_id = int(model.body("pelvis").id)
    left_body = int(model.body("left_ankle_roll_link").id)
    right_body = int(model.body("right_ankle_roll_link").id)
    geom_ids = {side: contact_geom_ids(model, side) for side in ("left", "right")}
    if any(len(ids) < 4 for ids in geom_ids.values()):
        raise RuntimeError(f"missing official sole collision spheres: {geom_ids}")

    com, com_velocity, centroidal_angmom = [], [], []
    sole_center = {"left": [], "right": []}
    sole_min_z = {"left": [], "right": []}
    point_cache: list[dict[str, dict[str, np.ndarray]]] = []
    gravity_error, yaw_error = [], []
    limit_excess = []
    for index, row in enumerate(rows):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        com.append(data.subtree_com[pelvis_id].copy())
        com_velocity.append(data.subtree_linvel[pelvis_id].copy())
        centroidal_angmom.append(data.subtree_angmom[pelvis_id].copy())
        points = {side: foot_points(model, data, geom_ids[side]) for side in ("left", "right")}
        point_cache.append(points)
        for side in ("left", "right"):
            sole_center[side].append(points[side]["center"])
            sole_min_z[side].append(float(points[side]["min_z"][0]))
        observed_gravity = np.asarray(row["obs"][6:9], dtype=np.float64)
        gravity_error.append(float(np.max(np.abs(projected_gravity(qpos[index, 3:7]) - observed_gravity))))
        yaw_error.append(abs(normalized_angle_error(yaw_from_quaternion(qpos[index, 3:7]), row["root_yaw_rad"])))
        excess = 0.0
        for name in ISAAC_JOINTS:
            jid = int(model.joint(name).id)
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
                value = qpos[index, qpos_adr[name]]
                excess = max(excess, float(max(low - value, value - high, 0.0)))
        limit_excess.append(excess)

    com = np.asarray(com)
    com_velocity = np.asarray(com_velocity)
    centroidal_angmom = np.asarray(centroidal_angmom)
    sole_center = {side: np.asarray(values) for side, values in sole_center.items()}
    sole_min_z = {side: np.asarray(values) for side, values in sole_min_z.items()}
    sole_velocity = {side: np.gradient(values, dt, axis=0) for side, values in sole_center.items()}

    fd_errors = {"root_linear": [], "root_angular": [], "joint": []}
    velocity_from_qpos = np.zeros(model.nv)
    for index in range(len(rows) - 1):
        mujoco.mj_differentiatePos(model, velocity_from_qpos, dt, qpos[index], qpos[index + 1])
        midpoint = 0.5 * (qvel[index] + qvel[index + 1])
        error = velocity_from_qpos - midpoint
        fd_errors["root_linear"].append(float(np.linalg.norm(error[:3])))
        fd_errors["root_angular"].append(float(np.linalg.norm(error[3:6])))
        fd_errors["joint"].append(float(np.sqrt(np.mean(np.square(error[6:])))))

    qacc = np.gradient(qvel, dt, axis=0)
    total_mass = float(model.body_subtreemass[pelvis_id])
    normal_forces = {"left": np.full(len(rows), np.nan), "right": np.full(len(rows), np.nan)}
    dynamics_root_residual = np.full(len(rows), np.nan)
    dynamics_full_residual = np.full(len(rows), np.nan)
    for index in range(1, len(rows) - 1):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        data.ctrl[:] = 0.0
        # mj_forward computes FK, passive forces and Jacobians, but also
        # overwrites qacc with the forward-dynamics result.  Install the
        # finite-difference acceleration only *after* that call before RNE.
        mujoco.mj_forward(model, data)
        passive = data.qfrc_passive.copy()
        data.qacc[:] = qacc[index]
        rne = np.zeros(model.nv, dtype=np.float64)
        mujoco.mj_rne(model, data, 1, rne)

        targets = default_pose()
        action = np.asarray(rows[index]["action"], dtype=np.float64)
        for action_index, name in enumerate(LOWER_JOINTS):
            targets[name] += float(action[action_index] * LOWER_SCALE[action_index])
        for name in ISAAC_JOINTS:
            kp, kd = pd_gains(name)
            aid = int(model.actuator(f"motor_{name}").id)
            torque = kp * (targets[name] - qpos[index, qpos_adr[name]]) - kd * qvel[index, dof_adr[name]]
            low, high = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(torque, low, high))
        mujoco.mj_forward(model, data)
        actuator = data.qfrc_actuator.copy()

        points = [
            (point_cache[index]["left"]["rear"], left_body),
            (point_cache[index]["left"]["front"], left_body),
            (point_cache[index]["right"]["rear"], right_body),
            (point_cache[index]["right"]["front"], right_body),
        ]
        matrix, labels = friction_pyramid_matrix(model, data, points)
        required = rne - actuator - passive
        coefficients, _ = nnls(matrix[:6], required[:6])
        estimated = matrix @ coefficients
        root_scale = max(float(np.linalg.norm(required[:6])), total_mass * 9.81 * 0.1)
        full_scale = max(float(np.linalg.norm(required)), total_mass * 9.81)
        dynamics_root_residual[index] = float(np.linalg.norm(required[:6] - estimated[:6]) / root_scale)
        dynamics_full_residual[index] = float(np.linalg.norm(required - estimated) / full_scale)
        for side in ("left", "right"):
            normal_forces[side][index] = float(sum(
                coefficients[column] for column, label in enumerate(labels) if label[0] == side
            ))

    # A force/contact proxy is only used where the sole is geometrically close
    # to the ground.  It remains a model estimate, never a measured label.
    force_threshold = 0.08 * total_mass * 9.81
    contacts = {
        side: (sole_min_z[side] <= 0.025) & (np.nan_to_num(normal_forces[side]) >= force_threshold)
        for side in ("left", "right")
    }
    move = np.asarray([row["stage"] == "move" for row in rows])
    generator_contact = {
        "left": np.asarray([row["obs"][91] > 0.5 for row in rows]),
        "right": np.asarray([row["obs"][92] > 0.5 for row in rows]),
    }
    pitch = np.asarray([math.asin(float(np.clip(row["obs"][6], -1.0, 1.0))) for row in rows])
    yaw = np.unwrap(np.asarray([row["root_yaw_rad"] for row in rows], dtype=np.float64))
    waist_pitch_index = ISAAC_JOINTS.index("waist_pitch_joint")

    contact_metrics = {}
    for side in ("left", "right"):
        stance = move & contacts[side]
        swing = move & ~contacts[side]
        intended_stance = move & generator_contact[side]
        intended_swing = move & ~generator_contact[side]
        slip = np.linalg.norm(sole_velocity[side][:, :2], axis=1)
        contact_metrics[side] = {
            "estimated_contact_fraction_move": float(np.mean(contacts[side][move])),
            "generator_agreement_move": float(np.mean(contacts[side][move] == generator_contact[side][move])),
            "stance_slip_mps": quantiles(slip[stance]),
            "swing_clearance_m": quantiles(np.maximum(sole_min_z[side][swing], 0.0)),
            "generator_stance_slip_mps": quantiles(slip[intended_stance]),
            "generator_swing_clearance_m": quantiles(
                np.maximum(sole_min_z[side][intended_swing], 0.0)
            ),
            "sole_min_z_m": quantiles(sole_min_z[side][move]),
            "estimated_normal_force_n": quantiles(normal_forces[side][move]),
        }

    action = np.asarray([row["action"] for row in rows], dtype=np.float64)
    action_delta = np.linalg.norm(np.diff(action, axis=0), axis=1)
    result = {
        "schema_contract": {
            "rows_total": len(payload["trace"]),
            "rows_decoded": len(rows),
            "obs_widths": sorted(set(len(row.get("obs", [])) for row in payload["trace"])),
            "action_widths": sorted(set(len(row.get("action", [])) for row in payload["trace"])),
            "joint_order": list(ISAAC_JOINTS),
            "action_order": list(LOWER_JOINTS),
            "gravity_reconstruction_abs_max": max(gravity_error),
            "yaw_reconstruction_abs_max_rad": max(yaw_error),
            "joint_limit_excess_max_rad": max(limit_excess),
            "summary_contracts": {
                key: payload["summary"].get(key) for key in (
                    "observation_contract", "action_contract", "default_pose_profile",
                    "pd_profile", "model", "stationary_model", "template", "full_gate_pass",
                )
            },
        },
        "continuity": {
            "qpos_fd_vs_recorded_qvel": {name: quantiles(values) for name, values in fd_errors.items()},
            "action_delta_l2": quantiles(action_delta),
            "com_fd_vs_model_velocity_mps": quantiles(
                np.linalg.norm(np.gradient(com, dt, axis=0) - com_velocity, axis=1)
            ),
        },
        "centroidal": {
            "mass_kg": total_mass,
            "com_height_m": quantiles(com[move, 2]),
            "com_speed_mps": quantiles(np.linalg.norm(com_velocity[move], axis=1)),
            "centroidal_angular_momentum_norm": quantiles(np.linalg.norm(centroidal_angmom[move], axis=1)),
        },
        "contact": {
            "source": "50Hz finite-difference + official MJCF inverse-dynamics/friction-pyramid model estimate",
            "measured_or_closed_ros_force": False,
            "force_threshold_n": force_threshold,
            "left": contact_metrics["left"],
            "right": contact_metrics["right"],
            "estimated_ds_ss_ds_cycles_move_raw": contiguous_cycles(
                contacts["left"][move], contacts["right"][move]
            ),
            "generator_ds_ss_ds_cycles_move": contiguous_cycles(
                generator_contact["left"][move], generator_contact["right"][move]
            ),
            "warning": (
                "force split uses an underdetermined 16-ray/6-root-equation NNLS; "
                "raw force-contact switches are not authoritative contact labels"
            ),
        },
        "dynamics": {
            "root_generalized_residual_relative": quantiles(dynamics_root_residual[move]),
            "full_generalized_residual_relative": quantiles(dynamics_full_residual[move]),
            "caveat": "qacc is finite-differenced at 50Hz; PD torque is reconstructed, not recorded; GRF is a model estimate",
        },
        "posture_and_turn": {
            "move_signed_root_pitch_rad": quantiles(pitch[move]),
            "move_waist_pitch_rad": quantiles(
                np.asarray([row["obs"][12 + waist_pitch_index] for row in rows])[move]
            ),
            "move_yaw_progress_rad": float(yaw[move][-1] - yaw[move][0]),
            "summary_yaw_progress_rad": payload["summary"].get("move_yaw_progress_rad"),
            "summary_forward_m": payload["summary"].get("move_forward_displacement_m"),
            "summary_lateral_m": payload["summary"].get("move_lateral_displacement_m"),
        },
    }

    # Qualification is deliberately split: these target-native rollouts are
    # useful as warm-start trajectories, but their inferred force must not be
    # promoted to ground-truth dynamic supervision.
    kinematic_seed = bool(
        payload["summary"].get("full_gate_pass")
        and result["schema_contract"]["gravity_reconstruction_abs_max"] <= 1e-6
        and result["schema_contract"]["joint_limit_excess_max_rad"] <= 1e-6
        and result["continuity"]["qpos_fd_vs_recorded_qvel"]["joint"]["p95"] <= 1.0
    )
    has_contact_cycle = result["contact"]["generator_ds_ss_ds_cycles_move"] >= 1
    low_slip = all(
        result["contact"][side]["generator_stance_slip_mps"]["p95"] is not None
        and result["contact"][side]["generator_stance_slip_mps"]["p95"] <= 0.20
        for side in ("left", "right")
    )
    result["qualification"] = {
        "native_kinematic_control_seed": kinematic_seed,
        "contact_consistent_seed": bool(kinematic_seed and has_contact_cycle and low_slip),
        "dynamics_truth_seed": False,
        "reasons": [
            "Target-native full-gate rollout may warm-start X2 trajectory optimization if continuity passes.",
            "GRF/contact/centroidal outputs are official-MJCF model estimates from 50Hz state, not measured labels.",
            "Persistent negative pelvis pitch must be preserved as a known bias, not treated as a natural-pose target.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--stationary-actor", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    actor_contract = {}
    for slot, path in (("main", args.actor), ("stationary", args.stationary_actor)):
        session = ort.InferenceSession(str(path.resolve()), providers=["CPUExecutionProvider"])
        input_shape = list(session.get_inputs()[0].shape)
        output_shape = list(session.get_outputs()[0].shape)
        if input_shape[-1] != 93 or output_shape[-1] != 15:
            raise RuntimeError(f"{slot} actor shape mismatch: {input_shape} -> {output_shape}")
        actor_contract[slot] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "input_shape": input_shape, "output_shape": output_shape,
        }
    cases = {}
    for path in args.trace:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases[path.stem] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            **analyze_case(model, payload),
        }
    missing_fields = [
        "1kHz simulator qpos/qvel or exact per-sample timestamps for high-confidence qacc",
        "actual applied actuator torque after saturation (31D)",
        "MuJoCo contact identities, positions, impulses/forces at every physics substep",
        "full root quaternion and free-base linear/angular velocity in explicitly named frames",
        "controller target before/after clipping plus Kp/Kd per joint",
    ]
    output = {
        "stage": "BASE Phase27",
        "scene": {"path": str(args.scene.resolve()), "sha256": sha256_file(args.scene)},
        "actors": actor_contract,
        "method_boundary": {
            "offline_only": True,
            "new_rollout": False,
            "training": False,
            "official_fk": True,
            "grf_is_model_estimate": True,
            "grf_is_closed_ros_or_hardware_truth": False,
        },
        "cases": cases,
        "minimum_record_extension_for_dynamics_truth": missing_fields,
        "decision": {
            "usable_as_native_warm_start": all(
                case["qualification"]["native_kinematic_control_seed"] for case in cases.values()
            ),
            "usable_as_contact_consistent_dynamic_seed": all(
                case["qualification"]["contact_consistent_seed"] for case in cases.values()
            ),
            "usable_as_dynamics_ground_truth": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["decision"], indent=2))


if __name__ == "__main__":
    main()
