#!/usr/bin/env python3
"""One-shot extended direct-official Stage250 straight qualification capture.

The vendor ``scene.xml`` is loaded unchanged.  This is a direct MuJoCo runner,
not the closed AimDK ROS wrapper.  It first proves the historical ONNX/action
contract and a same-state 10-tick deterministic fork, then may record exactly
one 50 Hz straight episode with every 1 kHz physics substep.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from official_x2.audit_stage250_native_dynamic_seed import (
    ISAAC_JOINTS,
    LOWER_JOINTS,
    LOWER_SCALE,
    contiguous_cycles,
    decode_row,
    projected_gravity,
    quantiles,
    rotation_body_to_world,
    sha256_file,
)
from official_x2.replay_official_trace_direct_mujoco import (
    JOINTS,
    default_pose,
    pd_gains,
    yaw_tilt,
)
from official_x2.run_testonly_official_mjcf_suffix_probe import (
    integration_state,
    phase_features,
    restore_integration_state,
    template_bias,
)


CONTROL_DT = 0.02
STAND_TICKS = 100
MOVE_TICKS = 200
STOP_TICKS = 400
TOTAL_TICKS = STAND_TICKS + MOVE_TICKS + STOP_TICKS
VX = 0.30


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def wrapped(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def stage_at_tick(tick: int) -> tuple[str, float, float]:
    if tick < STAND_TICKS:
        return "stand", tick * CONTROL_DT, 0.0
    if tick < STAND_TICKS + MOVE_TICKS:
        return "move", (tick - STAND_TICKS) * CONTROL_DT, VX
    return "stop", (tick - STAND_TICKS - MOVE_TICKS) * CONTROL_DT, 0.0


def _session_action(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    return session.run(["actions"], {"obs": obs[None].astype(np.float32)})[0][0].astype(np.float32)


class Stage250Contract:
    def __init__(self, main: ort.InferenceSession, stationary: ort.InferenceSession, template_path: Path):
        self.main = main
        self.stationary = stationary
        archive = np.load(template_path, allow_pickle=False)
        if tuple(archive["joint_names_15"].tolist()) != LOWER_JOINTS:
            raise RuntimeError("template joint order mismatch")
        if not np.allclose(archive["action_scale_rad"], LOWER_SCALE, atol=1e-6, rtol=0.0):
            raise RuntimeError("template action scale mismatch")
        self.template = archive["q_cycle_zero_mean_rad"].astype(np.float32)
        self.period = float(archive["period_s"])
        self.double_support = float(archive["double_support_fraction"])
        self.previous = {
            "main": np.zeros(15, dtype=np.float32),
            "stationary": np.zeros(15, dtype=np.float32),
        }
        self.issued = {
            "main": np.zeros(15, dtype=np.float32),
            "stationary": np.zeros(15, dtype=np.float32),
        }
        self.heading_target: float | None = None
        self.heading_origin: tuple[float, float] | None = None
        self.lateral_state = "off"
        self.lateral_bias = np.zeros(2, dtype=np.float32)
        self.last_stage: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "previous": {key: value.copy() for key, value in self.previous.items()},
            "issued": {key: value.copy() for key, value in self.issued.items()},
            "heading_target": self.heading_target,
            "heading_origin": self.heading_origin,
            "lateral_state": self.lateral_state,
            "lateral_bias": self.lateral_bias.copy(),
            "last_stage": self.last_stage,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.previous = {key: np.asarray(value, dtype=np.float32).copy() for key, value in state["previous"].items()}
        self.issued = {key: np.asarray(value, dtype=np.float32).copy() for key, value in state["issued"].items()}
        self.heading_target = state["heading_target"]
        self.heading_origin = state["heading_origin"]
        self.lateral_state = str(state["lateral_state"])
        self.lateral_bias = np.asarray(state["lateral_bias"], dtype=np.float32).copy()
        self.last_stage = state["last_stage"]

    def prepare_stage(
        self, stage: str, root_xy: tuple[float, float], root_yaw: float
    ) -> None:
        """Reproduce Stage250's explicit stand→move→stop handoffs."""
        if stage == self.last_stage:
            return
        if stage == "move":
            # Legacy step-command evaluation rebases heading and cross-track
            # at the first move tick (move_accelerate_seconds == 0).
            self.heading_target = float(root_yaw)
            self.heading_origin = tuple(root_xy)
            self.issued["main"] = self.issued["stationary"].copy()
        elif stage == "stop":
            # The stationary actor receives the last moving history/action.
            self.previous["stationary"] = self.previous["main"].copy()
            self.issued["stationary"] = self.issued["main"].copy()
        self.last_stage = stage

    def action_from_obs(
        self,
        *,
        obs: np.ndarray,
        stage: str,
        stage_elapsed: float,
        root_xy: tuple[float, float],
        root_yaw: float,
    ) -> dict[str, np.ndarray | float | str]:
        slot = "main" if stage == "move" else "stationary"
        if self.heading_target is None:
            self.heading_target = float(root_yaw)
            self.heading_origin = tuple(root_xy)
        primary_session = self.main if slot == "main" else self.stationary
        raw_primary = _session_action(primary_session, obs)
        raw_main = raw_primary.copy()
        raw_effective = raw_primary.copy()
        if slot == "stationary":
            main_obs = obs.copy()
            main_obs[74:89] = self.previous["main"]
            raw_main = _session_action(self.main, main_obs)
            raw_effective = 0.5 * raw_main + 0.5 * raw_primary
        first_clip = np.clip(raw_effective, -1.0, 1.0).astype(np.float32)
        moving = stage == "move"
        normalized_template = (
            0.15 * template_bias(self.template, stage_elapsed, self.period) / LOWER_SCALE
            if moving else np.zeros(15, dtype=np.float32)
        ).astype(np.float32)
        preclip = first_clip + normalized_template
        if moving:
            preclip[[5, 11]] += 0.20
            assert self.heading_origin is not None and self.heading_target is not None
            dx = float(root_xy[0]) - self.heading_origin[0]
            dy = float(root_xy[1]) - self.heading_origin[1]
            cross_track = -math.sin(self.heading_target) * dx + math.cos(self.heading_target) * dy
            if self.lateral_state == "off":
                if cross_track <= -0.12:
                    self.lateral_state = "right"
                elif cross_track >= 0.12:
                    self.lateral_state = "left"
            elif self.lateral_state == "right" and cross_track >= -0.04:
                self.lateral_state = "off"
            elif self.lateral_state == "left" and cross_track <= 0.04:
                self.lateral_state = "off"
            if self.lateral_state == "right":
                target = np.asarray([0.5, 0.5], dtype=np.float32)
            elif self.lateral_state == "left":
                target = np.asarray([0.5, -0.5], dtype=np.float32)
            else:
                target = np.zeros(2, dtype=np.float32)
            self.lateral_bias += np.clip(target - self.lateral_bias, -0.02, 0.02)
            preclip[[2, 8]] += self.lateral_bias
        final = np.clip(preclip, -1.0, 1.0).astype(np.float32)
        self.previous[slot] = first_clip.copy()
        self.issued[slot] = final.copy()
        if slot == "stationary":
            self.previous["main"] = first_clip.copy()
        return {
            "slot": slot,
            "raw_primary": raw_primary,
            "raw_main": raw_main,
            "raw_effective": raw_effective,
            "first_clip": first_clip,
            "template": normalized_template,
            "final": final,
            "lateral_bias": self.lateral_bias.copy(),
        }


def historical_action_preflight(
    payload: dict,
    main: ort.InferenceSession,
    stationary: ort.InferenceSession,
    template: Path,
) -> dict:
    contract = Stage250Contract(main, stationary, template)
    errors, previous_errors, raw_max, first_clip_max = [], [], [], []
    decoded = [row for row in payload["trace"] if len(row.get("obs", [])) == 93]
    for row in decoded:
        obs = np.asarray(row["obs"], dtype=np.float32)
        stage = str(row["stage"])
        slot = "main" if stage == "move" else "stationary"
        contract.prepare_stage(
            stage,
            (float(row["root_x_m"]), float(row["root_y_m"])),
            float(row["root_yaw_rad"]),
        )
        previous_errors.append(float(np.max(np.abs(obs[74:89] - contract.previous[slot]))))
        result = contract.action_from_obs(
            obs=obs,
            stage=stage,
            stage_elapsed=float(row["elapsed_s"]),
            root_xy=(float(row["root_x_m"]), float(row["root_y_m"])),
            root_yaw=float(row["root_yaw_rad"]),
        )
        expected = np.asarray(row["action"], dtype=np.float32)
        errors.append(float(np.max(np.abs(result["final"] - expected))))
        raw_max.append(float(np.max(np.abs(result["raw_effective"]))))
        first_clip_max.append(float(np.max(np.abs(result["first_clip"]))))
    return {
        "decoded_rows": len(decoded),
        "previous_action_abs_max_error": max(previous_errors),
        "historical_final_action_abs_max_error": max(errors),
        "raw_actor_abs_max": max(raw_max),
        "first_clip_abs_max": max(first_clip_max),
        "consistent_within_float32_5e7": max(max(previous_errors), max(errors)) <= 5e-7,
    }


class DirectRunner:
    def __init__(self, args: argparse.Namespace, initial_row: dict):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
        if not math.isclose(self.model.opt.timestep, 0.001, abs_tol=1e-12):
            raise RuntimeError("official scene timestep is not 1ms")
        self.data = mujoco.MjData(self.model)
        self.main = ort.InferenceSession(str(args.actor.resolve()), providers=["CPUExecutionProvider"])
        self.stationary = ort.InferenceSession(str(args.stationary_actor.resolve()), providers=["CPUExecutionProvider"])
        self.contract = Stage250Contract(self.main, self.stationary, args.template)
        self.default = default_pose()
        self.gains = pd_gains()
        self.qpos_adr = {name: int(self.model.joint(name).qposadr[0]) for name in JOINTS}
        self.dof_adr = {name: int(self.model.joint(name).dofadr[0]) for name in JOINTS}
        self.actuator_id = {name: int(self.model.actuator(f"motor_{name}").id) for name in JOINTS}
        qpos, qvel = decode_row(self.model, initial_row, self.qpos_adr, self.dof_adr)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.substeps = round(CONTROL_DT / self.model.opt.timestep)
        self.floor_id = int(self.model.geom("floor").id)
        self.foot_side = {}
        for gid in range(self.model.ngeom):
            body = self.model.body(int(self.model.geom_bodyid[gid])).name
            if body == "left_ankle_roll_link":
                self.foot_side[gid] = "left"
            elif body == "right_ankle_roll_link":
                self.foot_side[gid] = "right"

    def snapshot(self) -> tuple[np.ndarray, dict[str, Any], int]:
        return integration_state(self.model, self.data), self.contract.snapshot(), self.tick

    def restore(self, snapshot: tuple[np.ndarray, dict[str, Any], int]) -> None:
        restore_integration_state(self.model, self.data, snapshot[0])
        self.contract.restore(snapshot[1])
        self.tick = int(snapshot[2])

    def observation(self, stage: str, elapsed: float, vx: float) -> tuple[np.ndarray, float, float]:
        quaternion = self.data.qpos[3:7].copy()
        rotation = rotation_body_to_world(quaternion)
        lin_body = (rotation.T @ self.data.qvel[:3]).astype(np.float32)
        ang_body = self.data.qvel[3:6].astype(np.float32)
        gravity = projected_gravity(quaternion).astype(np.float32)
        yaw, _ = yaw_tilt(quaternion)
        self.contract.prepare_stage(
            stage,
            (float(self.data.qpos[0]), float(self.data.qpos[1])),
            yaw,
        )
        if self.contract.heading_target is None:
            heading_error = 0.0
        else:
            heading_error = wrapped(self.contract.heading_target - yaw)
        command_wz = float(np.clip(0.5 * heading_error, -0.1, 0.1))
        command = np.asarray([vx, 0.0, command_wz], dtype=np.float32)
        q = np.asarray(
            [self.data.qpos[self.qpos_adr[name]] - self.default[name] for name in ISAAC_JOINTS],
            dtype=np.float32,
        )
        dq = np.asarray([self.data.qvel[self.dof_adr[name]] for name in ISAAC_JOINTS], dtype=np.float32)
        slot = "main" if stage == "move" else "stationary"
        gait = phase_features(
            elapsed if stage == "move" else 0.0,
            stage == "move",
            self.contract.period,
            self.contract.double_support,
        )
        obs = np.concatenate((lin_body, ang_body, gravity, command, q, dq, self.contract.previous[slot], gait)).astype(np.float32)
        return obs, yaw, command_wz

    def _contact_rows(self) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
        contacts = []
        side_values = {"left": [], "right": []}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            side = None
            foot_geom = None
            if geom1 == self.floor_id and geom2 in self.foot_side:
                side, foot_geom = self.foot_side[geom2], geom2
            elif geom2 == self.floor_id and geom1 in self.foot_side:
                side, foot_geom = self.foot_side[geom1], geom1
            slip = None
            if side is not None and foot_geom is not None:
                body_id = int(self.model.geom_bodyid[foot_geom])
                velocity = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(
                    self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
                )
                center = self.data.xipos[body_id]
                point_velocity = velocity[3:6] + np.cross(velocity[0:3], contact.pos - center)
                slip = float(np.linalg.norm(point_velocity[:2]))
                side_values[side].append(float(force[0]))
            contacts.append({
                "geom1": geom1,
                "geom2": geom2,
                "geom1_name": self.model.geom(geom1).name,
                "geom2_name": self.model.geom(geom2).name,
                "side": side,
                "position_world_m": np.asarray(contact.pos).tolist(),
                "frame": np.asarray(contact.frame).tolist(),
                "distance_m": float(contact.dist),
                "dim": int(contact.dim),
                "efc_address": int(contact.efc_address),
                "wrench_contact_frame": force.tolist(),
                "impulse_approx_wrench_dt": (force * self.model.opt.timestep).tolist(),
                "foot_point_horizontal_slip_mps": slip,
            })
        return contacts, side_values

    def step(self, writer: gzip.GzipFile | None = None) -> dict[str, Any]:
        stage, elapsed, vx = stage_at_tick(self.tick)
        obs, yaw, command_wz = self.observation(stage, elapsed, vx)
        result = self.contract.action_from_obs(
            obs=obs,
            stage=stage,
            stage_elapsed=elapsed,
            root_xy=(float(self.data.qpos[0]), float(self.data.qpos[1])),
            root_yaw=yaw,
        )
        targets = dict(self.default)
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] += float(result["final"][index] * LOWER_SCALE[index])
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for name in JOINTS:
            kp, kd = self.gains[name]
            aid = self.actuator_id[name]
            torque = kp * (targets[name] - self.data.qpos[self.qpos_adr[name]]) - kd * self.data.qvel[self.dof_adr[name]]
            low, high = self.model.actuator_ctrlrange[aid]
            ctrl[aid] = float(np.clip(torque, low, high))
        control_row = {
            "tick": self.tick,
            "stage": stage,
            "elapsed_s": elapsed,
            "obs93": obs.tolist(),
            "gait": obs[89:93].tolist(),
            "command": [vx, 0.0, command_wz],
            "raw_actor_primary": result["raw_primary"].tolist(),
            "raw_actor_main_for_blend": result["raw_main"].tolist(),
            "raw_actor_effective": result["raw_effective"].tolist(),
            "first_clip": result["first_clip"].tolist(),
            "template_normalized": result["template"].tolist(),
            "final_clip_action": result["final"].tolist(),
            "pd_target_joint_order": list(JOINTS),
            "pd_target_rad": [targets[name] for name in JOINTS],
        }
        block_contact = {"left": False, "right": False}
        block_force = {"left": [], "right": []}
        block_slip = {"left": [], "right": []}
        for substep in range(self.substeps):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            contacts, side_forces = self._contact_rows()
            for side in ("left", "right"):
                if side_forces[side]:
                    block_contact[side] = True
                    block_force[side].append(sum(side_forces[side]))
            for contact in contacts:
                side = contact["side"]
                slip = contact["foot_point_horizontal_slip_mps"]
                if side is not None and slip is not None:
                    block_slip[side].append(float(slip))
            if writer is not None:
                substep_row = {
                    "physics_step": self.tick * self.substeps + substep + 1,
                    "control_tick": self.tick,
                    "substep": substep,
                    "time_s": float(self.data.time),
                    "stage": stage,
                    "qpos": self.data.qpos.tolist(),
                    "qvel": self.data.qvel.tolist(),
                    "root_position_m": self.data.qpos[:3].tolist(),
                    "root_quaternion_wxyz": self.data.qpos[3:7].tolist(),
                    "root_velocity_free_joint": self.data.qvel[:6].tolist(),
                    "contacts": contacts,
                    "ctrl": self.data.ctrl.tolist(),
                    "actuator_force": self.data.actuator_force.tolist(),
                    "qfrc_actuator": self.data.qfrc_actuator.tolist(),
                    "qfrc_constraint": self.data.qfrc_constraint.tolist(),
                    "raw_actor_effective": result["raw_effective"].tolist(),
                    "first_clip": result["first_clip"].tolist(),
                    "template_normalized": result["template"].tolist(),
                    "final_clip_action": result["final"].tolist(),
                    "pd_target_rad": [targets[name] for name in JOINTS],
                }
                writer.write((json.dumps(substep_row, separators=(",", ":")) + "\n").encode("utf-8"))
        quaternion = self.data.qpos[3:7].copy()
        rotation = rotation_body_to_world(quaternion)
        body_velocity = rotation.T @ self.data.qvel[:3]
        yaw_after, tilt = yaw_tilt(quaternion)
        control_row.update({
            "root_x_m": float(self.data.qpos[0]),
            "root_y_m": float(self.data.qpos[1]),
            "root_z_m": float(self.data.qpos[2]),
            "root_yaw_rad": yaw_after,
            "root_tilt_rad": tilt,
            "root_vx_b_mps": float(body_velocity[0]),
            "root_vy_b_mps": float(body_velocity[1]),
            "realized_contact": block_contact,
            "realized_normal_force_n": {
                side: float(np.mean(values)) if values else 0.0 for side, values in block_force.items()
            },
            "realized_slip_mps": {
                side: float(np.max(values)) if values else None for side, values in block_slip.items()
            },
        })
        self.tick += 1
        return control_row


def fork_preflight(args: argparse.Namespace, initial_row: dict) -> dict:
    first = DirectRunner(args, initial_row)
    second = DirectRunner(args, initial_row)
    snapshot = first.snapshot()
    second.restore(snapshot)
    errors = {key: 0.0 for key in ("qpos", "qvel", "obs", "action")}
    event_equal = True
    for _ in range(10):
        row1, row2 = first.step(), second.step()
        errors["qpos"] = max(errors["qpos"], float(np.max(np.abs(first.data.qpos - second.data.qpos))))
        errors["qvel"] = max(errors["qvel"], float(np.max(np.abs(first.data.qvel - second.data.qvel))))
        errors["obs"] = max(errors["obs"], float(np.max(np.abs(np.asarray(row1["obs93"]) - row2["obs93"]))))
        errors["action"] = max(errors["action"], float(np.max(np.abs(np.asarray(row1["final_clip_action"]) - row2["final_clip_action"]))))
        event_equal = event_equal and row1["stage"] == row2["stage"]
    return {"ticks": 10, "max_errors": errors, "event_equal": event_equal, "exact": all(value == 0.0 for value in errors.values()) and event_equal}


def episode_summary(rows: list[dict]) -> dict:
    groups = {stage: [row for row in rows if row["stage"] == stage] for stage in ("stand", "move", "stop")}
    stand, move, stop = groups["stand"], groups["move"], groups["stop"]
    thresholds = {
        "stand_z": 0.45, "stand_tilt": 0.25, "stand_drift": 0.10, "stand_tail": 0.03,
        "startup_z": 0.60, "startup_tilt": 0.30, "startup_forward": 0.10, "startup_back": 0.03,
        "move_z": 0.45, "move_tilt": 0.40, "move_forward": 0.50, "move_lateral": 0.30, "move_heading": 0.30,
        "stop_z": 0.45, "stop_tilt": 0.30, "stop_drift": 0.15, "stop_tail": 0.03,
    }
    def speed(row: dict) -> float:
        return math.hypot(row["root_vx_b_mps"], row["root_vy_b_mps"])
    stand_drift = math.hypot(stand[-1]["root_x_m"] - stand[0]["root_x_m"], stand[-1]["root_y_m"] - stand[0]["root_y_m"])
    stand_tail = float(np.mean([speed(row) for row in stand[-50:]]))
    startup = move[:50]
    move_origin = np.asarray([move[0]["root_x_m"], move[0]["root_y_m"]])
    heading = move[0]["root_yaw_rad"]
    heading_axis = np.asarray([math.cos(heading), math.sin(heading)])
    lateral_axis = np.asarray([-math.sin(heading), math.cos(heading)])
    startup_disp = np.asarray([startup[-1]["root_x_m"], startup[-1]["root_y_m"]]) - move_origin
    startup_path = [float(np.dot(np.asarray([r["root_x_m"], r["root_y_m"]]) - move_origin, heading_axis)) for r in startup]
    move_disp = np.asarray([move[-1]["root_x_m"], move[-1]["root_y_m"]]) - move_origin
    move_forward = float(np.dot(move_disp, heading_axis))
    move_lateral = float(np.dot(move_disp, lateral_axis))
    heading_max = max(abs(wrapped(row["root_yaw_rad"] - heading)) for row in move)
    stop_origin = np.asarray([stop[0]["root_x_m"], stop[0]["root_y_m"]])
    stop_drift = float(np.linalg.norm(np.asarray([stop[-1]["root_x_m"], stop[-1]["root_y_m"]]) - stop_origin))
    stop_tail = float(np.mean([speed(row) for row in stop[-50:]]))
    metrics = {
        "stand_root_z_min_m": min(r["root_z_m"] for r in stand),
        "stand_tilt_max_rad": max(r["root_tilt_rad"] for r in stand),
        "stand_xy_drift_m": stand_drift,
        "stand_tail_speed_mps": stand_tail,
        "startup_root_z_min_m": min(r["root_z_m"] for r in startup),
        "startup_tilt_max_rad": max(r["root_tilt_rad"] for r in startup),
        "startup_forward_m": float(np.dot(startup_disp, heading_axis)),
        "startup_backward_excursion_m": max(0.0, -min(startup_path)),
        "move_root_z_min_m": min(r["root_z_m"] for r in move),
        "move_tilt_max_rad": max(r["root_tilt_rad"] for r in move),
        "move_forward_m": move_forward,
        "move_lateral_m": move_lateral,
        "move_heading_max_rad": heading_max,
        "stop_root_z_min_m": min(r["root_z_m"] for r in stop),
        "stop_tilt_max_rad": max(r["root_tilt_rad"] for r in stop),
        "stop_xy_drift_m": stop_drift,
        "stop_tail_speed_mps": stop_tail,
    }
    gates = {
        "stand": metrics["stand_root_z_min_m"] >= thresholds["stand_z"] and metrics["stand_tilt_max_rad"] <= thresholds["stand_tilt"] and stand_drift <= thresholds["stand_drift"] and stand_tail <= thresholds["stand_tail"],
        "startup": metrics["startup_root_z_min_m"] >= thresholds["startup_z"] and metrics["startup_tilt_max_rad"] <= thresholds["startup_tilt"] and metrics["startup_forward_m"] >= thresholds["startup_forward"] and metrics["startup_backward_excursion_m"] <= thresholds["startup_back"],
        "move": metrics["move_root_z_min_m"] >= thresholds["move_z"] and metrics["move_tilt_max_rad"] <= thresholds["move_tilt"] and move_forward >= thresholds["move_forward"] and abs(move_lateral) <= thresholds["move_lateral"] and heading_max <= thresholds["move_heading"],
        "stop": metrics["stop_root_z_min_m"] >= thresholds["stop_z"] and metrics["stop_tilt_max_rad"] <= thresholds["stop_tilt"] and stop_drift <= thresholds["stop_drift"] and stop_tail <= thresholds["stop_tail"],
    }
    gates["full"] = all(gates.values())

    contact = {}
    realized = {}
    for side in ("left", "right"):
        contact[side] = np.asarray([bool(row["realized_contact"][side]) for row in move])
        slips = [row["realized_slip_mps"][side] for row in move if row["realized_slip_mps"][side] is not None]
        forces = [row["realized_normal_force_n"][side] for row in move]
        realized[side] = {
            "contact_fraction": float(np.mean(contact[side])),
            "contact_slip_mps": quantiles(slips),
            "normal_force_n": quantiles(forces),
        }
    realized["ds_ss_ds_cycles"] = contiguous_cycles(contact["left"], contact["right"])
    realized["provenance"] = "mujoco.mj_contactForce at every 1ms substep; model contact, not hardware force"
    return {"metrics": metrics, "gates": gates, "realized_contact": realized}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--stationary-actor", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--historical-trace", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.historical_trace.read_text(encoding="utf-8"))
    initial_row = next(row for row in payload["trace"] if len(row.get("obs", [])) == 93)
    main_session = ort.InferenceSession(str(args.actor.resolve()), providers=["CPUExecutionProvider"])
    stationary_session = ort.InferenceSession(str(args.stationary_actor.resolve()), providers=["CPUExecutionProvider"])
    historical = historical_action_preflight(payload, main_session, stationary_session, args.template)
    fork = fork_preflight(args, initial_row)
    assets = {
        key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for key, path in (
            ("scene", args.scene), ("actor", args.actor), ("stationary_actor", args.stationary_actor),
            ("template", args.template), ("historical_trace", args.historical_trace),
        )
    }
    preflight_pass = historical["consistent_within_float32_5e7"] and fork["exact"]
    manifest: dict[str, Any] = {
        "stage": "BASE Phase28",
        "contract": {
            "unchanged_official_scene": True,
            "control_hz": 50,
            "physics_hz": 1000,
            "stand_move_stop_ticks": [STAND_TICKS, MOVE_TICKS, STOP_TICKS],
            "single_rollout_max": 1,
            "domain": "direct official MuJoCo; not closed AimDK ROS",
        },
        "assets": assets,
        "historical_action_preflight": historical,
        "fork_preflight": fork,
        "preflight_pass": preflight_pass,
        "rollout_executed": False,
        "qualified_native_dynamic_seed": False,
    }
    if args.preflight_only or not preflight_pass:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"preflight_pass": preflight_pass, "rollout_executed": False}, indent=2))
        return

    args.cache_dir.mkdir(parents=True, exist_ok=False)
    substep_path = args.cache_dir / "stage250_straight_substeps.jsonl.gz"
    tick_path = args.cache_dir / "stage250_straight_control_ticks.json"
    runner = DirectRunner(args, initial_row)
    control_rows = []
    with gzip.open(substep_path, "wb", compresslevel=6) as writer:
        for _ in range(TOTAL_TICKS):
            control_rows.append(runner.step(writer))
    tick_path.write_text(json.dumps(control_rows, separators=(",", ":")) + "\n", encoding="utf-8")
    summary = episode_summary(control_rows)
    manifest.update({
        "rollout_executed": True,
        "rollout_count": 1,
        "episode": summary,
        "cache": {
            "substeps": {"path": str(substep_path.resolve()), "sha256": sha256_file(substep_path), "bytes": substep_path.stat().st_size, "rows": TOTAL_TICKS * 20},
            "control_ticks": {"path": str(tick_path.resolve()), "sha256": sha256_file(tick_path), "bytes": tick_path.stat().st_size, "rows": TOTAL_TICKS},
        },
        "qualified_native_dynamic_seed": bool(summary["gates"]["full"]),
        "qualification_boundary": (
            "full nominal gate is necessary; realized contact metrics remain official-MJCF model evidence, not hardware truth"
        ),
    })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"preflight_pass": True, "rollout_executed": True, "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
