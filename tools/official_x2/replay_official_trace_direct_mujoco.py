#!/usr/bin/env python3
"""Replay an official-gate target trace directly in the vendor X2 MJCF.

This removes ROS and the closed-source simulator wrapper from the loop while
retaining the vendor MJCF, 1 kHz physics, motor limits, and the same position-PD
contract.  Agreement with an official trace is the acceptance test before this
direct environment may be used for policy optimization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np


LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")
ARM_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
JOINTS = LOWER_JOINTS + HEAD_JOINTS + ARM_JOINTS
LOWER_SCALE = np.asarray(
    [0.4, 0.4, 0.4, 0.4, 0.12, 0.08,
     0.4, 0.4, 0.4, 0.4, 0.12, 0.08, 0.4, 0.16, 0.16],
    dtype=np.float64,
)


def default_pose() -> dict[str, float]:
    pose = {name: 0.0 for name in JOINTS}
    for side in ("left", "right"):
        pose[f"{side}_hip_pitch_joint"] = -0.248
        pose[f"{side}_knee_joint"] = 0.5303
        pose[f"{side}_ankle_pitch_joint"] = -0.2823
        pose[f"{side}_shoulder_pitch_joint"] = 0.4
        pose[f"{side}_elbow_joint"] = -1.2
    return pose


def official_start_pose() -> dict[str, float]:
    """Logical nominal configuration from the vendor default.yaml."""
    pose = {name: 0.0 for name in JOINTS}
    for side in ("left", "right"):
        pose[f"{side}_hip_pitch_joint"] = -0.24
        pose[f"{side}_knee_joint"] = 0.45
        pose[f"{side}_ankle_pitch_joint"] = -0.21
        pose[f"{side}_shoulder_pitch_joint"] = 0.196
    return pose


def pd_gains() -> dict[str, tuple[float, float]]:
    gains = {name: (300.0, 20.0) for name in LOWER_JOINTS}
    for side in ("left", "right"):
        gains[f"{side}_ankle_pitch_joint"] = (40.0, 20.0)
        gains[f"{side}_ankle_roll_joint"] = (40.0, 20.0)
    for name in ARM_JOINTS:
        gains[name] = (30.0, 3.0) if "wrist" in name else (40.0, 5.0)
    for name in HEAD_JOINTS:
        gains[name] = (50.0, 5.0)
    return gains


def yaw_tilt(quat_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat_wxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Tilt is the angle between body and world vertical, independent of yaw.
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    tilt = math.acos(float(np.clip(body_z_world_z, -1.0, 1.0)))
    return yaw, tilt


def wrapped_error(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument(
        "--command-delay-steps", type=int, default=0,
        help="Whole 1 kHz physics steps of target delay used only for attribution.",
    )
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    official_rows = payload["trace"]
    summary = payload["summary"]
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    if not math.isclose(model.opt.timestep, 0.001, abs_tol=1.0e-12):
        raise RuntimeError(f"vendor timestep changed: {model.opt.timestep}")
    substeps = round(args.control_dt / model.opt.timestep)
    if not math.isclose(substeps * model.opt.timestep, args.control_dt, abs_tol=1.0e-12):
        raise ValueError("control dt must be an integer multiple of MJCF timestep")
    if args.command_delay_steps < 0:
        raise ValueError("command delay must be non-negative")

    pose = default_pose()
    gains = pd_gains()
    qpos_adr = {name: model.joint(name).qposadr for name in JOINTS}
    dof_adr = {name: model.joint(name).dofadr for name in JOINTS}
    actuator_id = {name: model.actuator(f"motor_{name}").id for name in JOINTS}

    # The first official row is the first state observed by the adapter, after
    # simulator startup.  Seeding its available free-base quantities avoids
    # attributing wrapper startup latency to the direct physics model.
    first = official_rows[0]
    data.qpos[:3] = [first["root_x_m"], first["root_y_m"], first["root_z_m"]]
    half_yaw = 0.5 * float(first["root_yaw_rad"])
    data.qpos[3:7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    data.qvel[:3] = [first["root_vx_w_mps"], first["root_vy_w_mps"], 0.0]
    data.qvel[5] = first["root_yaw_rate_radps"]
    for name, value in official_start_pose().items():
        data.qpos[qpos_adr[name]] = value
    mujoco.mj_forward(model, data)
    start_q = {name: float(data.qpos[qpos_adr[name]]) for name in JOINTS}
    current_target = dict(start_q)
    target_queue = [dict(current_target) for _ in range(args.command_delay_steps + 1)]

    direct_rows: list[dict[str, float | str]] = []
    errors: dict[str, list[float]] = {key: [] for key in ("x", "y", "z", "yaw", "tilt")}
    for official in official_rows:
        yaw, tilt = yaw_tilt(data.qpos[3:7])
        direct = {
            "stage": official["stage"],
            "elapsed_s": float(official["elapsed_s"]),
            "root_x_m": float(data.qpos[0]),
            "root_y_m": float(data.qpos[1]),
            "root_z_m": float(data.qpos[2]),
            "root_yaw_rad": yaw,
            "root_tilt_rad": tilt,
            "joint_pos": {name: float(data.qpos[qpos_adr[name]]) for name in JOINTS},
            "joint_vel": {name: float(data.qvel[dof_adr[name]]) for name in JOINTS},
        }
        direct_rows.append(direct)
        errors["x"].append(direct["root_x_m"] - official["root_x_m"])
        errors["y"].append(direct["root_y_m"] - official["root_y_m"])
        errors["z"].append(direct["root_z_m"] - official["root_z_m"])
        errors["yaw"].append(wrapped_error(yaw, official["root_yaw_rad"]))
        errors["tilt"].append(tilt - official["root_tilt_rad"])

        if official["stage"] == "prepare":
            alpha = min(1.0, float(official["elapsed_s"]) / float(summary["prepare_seconds"]))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            current_target = {
                name: start_q[name] + alpha * (pose[name] - start_q[name]) for name in JOINTS
            }
        elif len(official["action"]) == len(LOWER_JOINTS):
            current_target = dict(pose)
            for index, name in enumerate(LOWER_JOINTS):
                current_target[name] = pose[name] + float(official["action"][index]) * LOWER_SCALE[index]
        else:
            current_target = dict(pose)

        for _ in range(substeps):
            target_queue.append(dict(current_target))
            applied = target_queue.pop(0)
            for name in JOINTS:
                kp, kd = gains[name]
                torque = kp * (applied[name] - data.qpos[qpos_adr[name]]) - kd * data.qvel[dof_adr[name]]
                aid = actuator_id[name]
                low, high = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = np.clip(torque, low, high)
            mujoco.mj_step(model, data)

    metrics: dict[str, object] = {
        "scene": str(args.scene.resolve()),
        "official_trace": str(args.trace.resolve()),
        "physics_dt_s": float(model.opt.timestep),
        "control_dt_s": args.control_dt,
        "command_delay_steps": args.command_delay_steps,
        "row_count": len(direct_rows),
        "rmse": {key: float(np.sqrt(np.mean(np.square(values)))) for key, values in errors.items()},
        "max_abs": {key: float(np.max(np.abs(values))) for key, values in errors.items()},
        "final_direct": direct_rows[-1],
        "final_official": {
            key: official_rows[-1][key]
            for key in ("stage", "elapsed_s", "root_x_m", "root_y_m", "root_z_m", "root_yaw_rad", "root_tilt_rad")
        },
        "direct_trace": direct_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rmse": metrics["rmse"], "max_abs": metrics["max_abs"]}, indent=2))


if __name__ == "__main__":
    main()
