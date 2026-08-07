#!/usr/bin/env python3
"""Run the Stage219 two-skill controller directly in the vendor X2 MJCF.

This is a straight-line contract probe.  It duplicates the verified official
adapter's 50 Hz observation/action semantics while stepping the vendor MJCF at
1 kHz, eliminating ROS scheduling and enabling a future batched training path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from replay_official_trace_direct_mujoco import (
    JOINTS, LOWER_JOINTS, LOWER_SCALE, default_pose, official_start_pose,
    pd_gains, yaw_tilt,
)


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


def rotation_body_to_world(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def projected_gravity(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        2.0 * (-z*x + w*y),
        -2.0 * (z*y + w*x),
        1.0 - 2.0 * (w*w + z*z),
    ], dtype=np.float32)


def phase_features(elapsed: float, moving: bool, period: float, double_support: float) -> np.ndarray:
    if not moving:
        return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    phase = (elapsed / period) % 1.0
    half_ds = double_support / 4.0
    right_swing = half_ds <= phase < 0.5 - half_ds
    left_swing = 0.5 + half_ds <= phase < 1.0 - half_ds
    return np.asarray([
        math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase),
        float(not left_swing), float(not right_swing),
    ], dtype=np.float32)


def template_bias(template: np.ndarray, elapsed: float, period: float) -> np.ndarray:
    phase = (elapsed / period) % 1.0
    position = phase * template.shape[0] - 0.5
    lower_unwrapped = math.floor(position)
    blend = position - lower_unwrapped
    lower = lower_unwrapped % template.shape[0]
    upper = (lower + 1) % template.shape[0]
    return (1.0 - blend) * template[lower] + blend * template[upper]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stationary-model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--initial-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vx", type=float, default=0.30)
    parser.add_argument("--prepare-seconds", type=float, default=0.2)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--move-seconds", type=float, default=4.0)
    parser.add_argument("--stop-seconds", type=float, default=8.0)
    parser.add_argument("--obs-noise-scale", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--physical-observation-delay-steps", type=int, default=0)
    parser.add_argument(
        "--physical-observation-lag-fraction", type=float, default=0.0,
        help="Blend current physical state toward the previous 20 ms sample (0..1).",
    )
    parser.add_argument(
        "--predict-physical-observation", action="store_true",
        help="First-order predict delayed q/gravity back to the current control time.",
    )
    args = parser.parse_args()
    if args.obs_noise_scale < 0.0:
        raise ValueError("observation noise scale must be non-negative")
    if args.physical_observation_delay_steps < 0:
        raise ValueError("physical observation delay must be non-negative")
    if not 0.0 <= args.physical_observation_lag_fraction <= 1.0:
        raise ValueError("physical observation lag fraction must be in [0, 1]")
    rng = np.random.default_rng(args.noise_seed)

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    if not math.isclose(model.opt.timestep, 0.001, abs_tol=1e-12):
        raise RuntimeError(f"unexpected vendor timestep {model.opt.timestep}")
    control_substeps = 20
    main_session = ort.InferenceSession(str(args.model.resolve()), providers=["CPUExecutionProvider"])
    stationary_session = ort.InferenceSession(
        str(args.stationary_model.resolve()), providers=["CPUExecutionProvider"]
    )
    archive = np.load(args.template, allow_pickle=False)
    template = archive["q_cycle_zero_mean_rad"].astype(np.float32)
    period = float(archive["period_s"])
    double_support = float(archive["double_support_fraction"])
    if tuple(archive["joint_names_15"].tolist()) != LOWER_JOINTS:
        raise RuntimeError("gait-template joint order mismatch")

    pose = default_pose()
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in JOINTS}
    dof_adr = {name: int(model.joint(name).dofadr[0]) for name in JOINTS}
    actuator_id = {name: int(model.actuator(f"motor_{name}").id) for name in JOINTS}
    gains = pd_gains()
    initial_payload = json.loads(args.initial_trace.read_text(encoding="utf-8"))
    initial = initial_payload["trace"][0]
    data.qpos[:3] = [initial["root_x_m"], initial["root_y_m"], initial["root_z_m"]]
    half_yaw = 0.5 * initial["root_yaw_rad"]
    data.qpos[3:7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    data.qvel[:3] = [initial["root_vx_w_mps"], initial["root_vy_w_mps"], 0.0]
    data.qvel[5] = initial["root_yaw_rate_radps"]
    for name, value in official_start_pose().items():
        data.qpos[qpos_adr[name]] = value
    mujoco.mj_forward(model, data)
    prepare_start = {name: float(data.qpos[qpos_adr[name]]) for name in JOINTS}

    previous = {"main": np.zeros(15, np.float32), "stationary": np.zeros(15, np.float32)}
    lateral_state = "off"
    lateral_bias = np.zeros(2, dtype=np.float32)
    move_origin: tuple[float, float] | None = None
    move_heading: float | None = None
    raw_trace: list[dict[str, object]] = []
    issued_actions: list[np.ndarray] = []
    physical_history: list[np.ndarray] = []
    delayed_physical_history: list[np.ndarray] = []

    def state_values() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        q = data.qpos[3:7].copy()
        rotation = rotation_body_to_world(q)
        lin_body = (rotation.T @ data.qvel[:3]).astype(np.float32)
        ang_body = data.qvel[3:6].astype(np.float32).copy()
        yaw, tilt = yaw_tilt(q)
        return lin_body, ang_body, projected_gravity(q), yaw, tilt

    def policy_action(
        phase_elapsed: float, command_vx: float, slot: str, moving: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal lateral_state, lateral_bias
        lin_body, ang_body, gravity, yaw, _ = state_values()
        command = np.asarray([command_vx, 0.0, 0.0], dtype=np.float32)
        joint_pos = np.asarray(
            [data.qpos[qpos_adr[name]] - pose[name] for name in ISAAC_JOINTS], dtype=np.float32
        )
        joint_vel = np.asarray([data.qvel[dof_adr[name]] for name in ISAAC_JOINTS], dtype=np.float32)
        physical_now = np.concatenate([lin_body, ang_body, gravity, joint_pos, joint_vel])
        physical_history.append(physical_now)
        history_index = max(0, len(physical_history) - 1 - args.physical_observation_delay_steps)
        physical = physical_history[history_index]
        if args.physical_observation_delay_steps == 0 and len(physical_history) >= 2:
            lag = args.physical_observation_lag_fraction
            physical = (1.0 - lag) * physical + lag * physical_history[-2]
        delayed_physical_history.append(physical.copy())
        if args.predict_physical_observation:
            tau = 0.02 * (
                args.physical_observation_delay_steps + args.physical_observation_lag_fraction
            )
            physical = physical.copy()
            acceleration = np.zeros_like(physical)
            if len(delayed_physical_history) >= 2:
                acceleration = (delayed_physical_history[-1] - delayed_physical_history[-2]) / 0.02
            velocity_delayed = physical[40:71].copy()
            physical[0:6] += tau * acceleration[0:6]
            physical[40:71] += tau * acceleration[40:71]
            omega = physical[3:6].copy()
            gravity_delayed = physical[6:9].copy()
            gravity_predicted = gravity_delayed - tau * np.cross(omega, gravity_delayed)
            gravity_norm = np.linalg.norm(gravity_predicted)
            if gravity_norm > 1.0e-8:
                physical[6:9] = gravity_predicted / gravity_norm
            physical[9:40] += tau * velocity_delayed + 0.5 * tau * tau * acceleration[40:71]
        lin_body = physical[0:3]
        ang_body = physical[3:6]
        gravity = physical[6:9]
        joint_pos = physical[9:40]
        joint_vel = physical[40:71]
        phase = phase_features(phase_elapsed, moving, period, double_support)
        obs = np.concatenate(
            [lin_body, ang_body, gravity, command, joint_pos, joint_vel, previous[slot], phase]
        ).astype(np.float32)
        if args.obs_noise_scale > 0.0:
            # One-sigma values come from the aligned official-vs-direct stand
            # audit.  Commands and gait phase are local controller state and
            # are therefore deliberately left exact.
            std = np.zeros(93, dtype=np.float32)
            std[0:3] = 0.0085
            std[3:6] = 0.031
            std[6:9] = 0.0015
            std[12:43] = 0.0042
            std[43:74] = 0.041
            std[74:89] = 0.0255
            noise = np.clip(rng.normal(size=93), -3.0, 3.0).astype(np.float32)
            obs += args.obs_noise_scale * std * noise
        session = stationary_session if slot == "stationary" else main_session
        raw = session.run(["actions"], {"obs": obs[None]})[0][0].astype(np.float32)
        if slot == "stationary":
            main_obs = obs.copy()
            main_obs[74:89] = previous["main"]
            main_raw = main_session.run(["actions"], {"obs": main_obs[None]})[0][0].astype(np.float32)
            raw = 0.5 * main_raw + 0.5 * raw
        combined = raw.copy()
        if moving:
            combined += 0.15 * template_bias(template, phase_elapsed, period) / LOWER_SCALE
            combined[[5, 11]] += 0.20
            assert move_origin is not None and move_heading is not None
            dx = float(data.qpos[0]) - move_origin[0]
            dy = float(data.qpos[1]) - move_origin[1]
            cross_track = -math.sin(move_heading) * dx + math.cos(move_heading) * dy
            if lateral_state == "off":
                if cross_track <= -0.12:
                    lateral_state = "right"
                elif cross_track >= 0.12:
                    lateral_state = "left"
            elif lateral_state == "right" and cross_track >= -0.04:
                lateral_state = "off"
            elif lateral_state == "left" and cross_track <= 0.04:
                lateral_state = "off"
            target = (
                np.asarray([0.5, 0.5], np.float32) if lateral_state == "right" else
                np.asarray([0.5, -0.5], np.float32) if lateral_state == "left" else
                np.zeros(2, np.float32)
            )
            lateral_bias += np.clip(target - lateral_bias, -0.02, 0.02)
            combined[[2, 8]] += lateral_bias
        combined = np.clip(combined, -1.0, 1.0)
        previous[slot] = raw.copy()
        if slot == "stationary":
            previous["main"] = raw.copy()
        return combined, obs, raw

    total_steps = round(
        (args.prepare_seconds + args.stand_seconds + args.move_seconds + args.stop_seconds) / 0.02
    )
    for step in range(total_steps):
        elapsed = step * 0.02
        if elapsed < args.prepare_seconds:
            stage = "prepare"
            stage_elapsed = elapsed
            alpha = min(1.0, elapsed / args.prepare_seconds)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            targets = {
                name: prepare_start[name] + alpha * (pose[name] - prepare_start[name]) for name in JOINTS
            }
            action = obs = None
        elif elapsed < args.prepare_seconds + args.stand_seconds:
            stage = "stand"
            stage_elapsed = elapsed - args.prepare_seconds
            action, obs, _ = policy_action(0.0, 0.0, "stationary", False)
            targets = dict(pose)
            for index, name in enumerate(LOWER_JOINTS):
                targets[name] += float(action[index] * LOWER_SCALE[index])
        elif elapsed < args.prepare_seconds + args.stand_seconds + args.move_seconds:
            stage = "move"
            stage_elapsed = elapsed - args.prepare_seconds - args.stand_seconds
            if move_origin is None:
                _, _, _, move_heading, _ = state_values()
                move_origin = (float(data.qpos[0]), float(data.qpos[1]))
            action, obs, _ = policy_action(stage_elapsed, args.vx, "main", True)
            targets = dict(pose)
            for index, name in enumerate(LOWER_JOINTS):
                targets[name] += float(action[index] * LOWER_SCALE[index])
        else:
            stage = "stop"
            stage_elapsed = elapsed - args.prepare_seconds - args.stand_seconds - args.move_seconds
            action, obs, _ = policy_action(args.move_seconds, 0.0, "stationary", False)
            targets = dict(pose)
            for index, name in enumerate(LOWER_JOINTS):
                targets[name] += float(action[index] * LOWER_SCALE[index])

        lin_body, _, _, yaw, tilt = state_values()
        raw_trace.append({
            "stage": stage, "elapsed_s": stage_elapsed,
            "root_x_m": float(data.qpos[0]), "root_y_m": float(data.qpos[1]),
            "root_z_m": float(data.qpos[2]), "root_tilt_rad": tilt,
            "root_yaw_rad": yaw, "root_vx_b_mps": float(lin_body[0]),
            "root_vy_b_mps": float(lin_body[1]),
            "obs": [] if obs is None else obs.tolist(),
            "action": [] if action is None else action.tolist(),
        })
        if action is not None:
            issued_actions.append(action.copy())
        for _ in range(control_substeps):
            for name in JOINTS:
                kp, kd = gains[name]
                torque = kp * (targets[name] - data.qpos[qpos_adr[name]]) - kd * data.qvel[dof_adr[name]]
                aid = actuator_id[name]
                low, high = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = float(np.clip(torque, low, high))
            mujoco.mj_step(model, data)

    move_rows = [row for row in raw_trace if row["stage"] == "move"]
    stop_rows = [row for row in raw_trace if row["stage"] == "stop"]
    move_start = move_rows[0]
    move_end = move_rows[-1]
    actions = np.asarray(issued_actions, dtype=np.float64)
    summary = {
        "domain": "vendor_mjcf_direct_closed_loop",
        "model": str(args.model.resolve()),
        "obs_noise_scale": args.obs_noise_scale,
        "noise_seed": args.noise_seed,
        "physical_observation_delay_steps": args.physical_observation_delay_steps,
        "physical_observation_lag_fraction": args.physical_observation_lag_fraction,
        "predict_physical_observation": args.predict_physical_observation,
        "survived_full_horizon": min(row["root_z_m"] for row in raw_trace) >= 0.45
        and max(row["root_tilt_rad"] for row in raw_trace) <= 0.8,
        "move_forward_displacement_m": move_end["root_x_m"] - move_start["root_x_m"],
        "move_lateral_displacement_m": move_end["root_y_m"] - move_start["root_y_m"],
        "move_tilt_max_rad": max(row["root_tilt_rad"] for row in move_rows),
        "stop_root_z_min_m": min(row["root_z_m"] for row in stop_rows),
        "stop_tilt_max_rad": max(row["root_tilt_rad"] for row in stop_rows),
        "stop_tail_speed_mean_mps": float(np.mean([
            math.hypot(row["root_vx_b_mps"], row["root_vy_b_mps"]) for row in stop_rows[-50:]
        ])),
        "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.999)),
        "action_delta_l2_p95": float(np.quantile(np.linalg.norm(np.diff(actions, axis=0), axis=1), 0.95)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "trace": raw_trace}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
