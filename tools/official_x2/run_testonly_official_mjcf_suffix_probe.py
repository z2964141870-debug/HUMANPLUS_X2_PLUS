#!/usr/bin/env python3
"""Deterministic suffix probe in the unchanged official X2 MJCF.

Evidence boundary
-----------------
This is deliberately a *test-only*, same-process MuJoCo runner.  It loads the
vendor ``scene.xml`` without modification and reproduces the Stage351
matched-event controller contract at 50 Hz, but it does not run the closed
AimDK ROS simulator.  AimDK exposes reset, not lossless mid-event state
injection; consequently a pass here proves only that the physical integration
state plus the Phase6 controller snapshot are sufficient in this isolated
runner.  It must not be labelled an official-ROS suffix replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import mujoco
import numpy as np
import onnxruntime as ort

try:  # Package import used by tests.
    from .controller_snapshot_contract import (
        capture_controller_state,
        restore_controller_state,
    )
    from .analyze_sagittal_posture import signed_pitch_from_xyzw_rad
    from .replay_official_trace_direct_mujoco import (
        JOINTS,
        LOWER_JOINTS,
        LOWER_SCALE,
        default_pose,
        official_start_pose,
        pd_gains,
        yaw_tilt,
    )
    from .skill_handoff_contract import matched_event_speed
except ImportError:  # Direct CLI execution from tools/official_x2.
    from controller_snapshot_contract import (
        capture_controller_state,
        restore_controller_state,
    )
    from analyze_sagittal_posture import signed_pitch_from_xyzw_rad
    from replay_official_trace_direct_mujoco import (
        JOINTS,
        LOWER_JOINTS,
        LOWER_SCALE,
        default_pose,
        official_start_pose,
        pd_gains,
        yaw_tilt,
    )
    from skill_handoff_contract import matched_event_speed


CONTROL_DT_S = 0.02
SNAPSHOT_NEXT_STOP_ELAPSED_S = 1.80
CHECKPOINT_TICKS = (10, 25, 50)
PHYSICAL_TOLERANCE = 1.0e-12
PITCH_TOLERANCE_DEG = 1.0e-12

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


def rotation_body_to_world(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            2.0 * (-z*x + w*y),
            -2.0 * (z*y + w*x),
            1.0 - 2.0 * (w*w + z*z),
        ],
        dtype=np.float32,
    )


def phase_features(
    elapsed_s: float,
    moving: bool,
    period_s: float,
    double_support_fraction: float,
) -> np.ndarray:
    if not moving:
        return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    phase = (elapsed_s / period_s) % 1.0
    half_ds = double_support_fraction / 4.0
    right_swing = half_ds <= phase < 0.5 - half_ds
    left_swing = 0.5 + half_ds <= phase < 1.0 - half_ds
    return np.asarray(
        [
            math.sin(2 * math.pi * phase),
            math.cos(2 * math.pi * phase),
            float(not left_swing),
            float(not right_swing),
        ],
        dtype=np.float32,
    )


def template_bias(template: np.ndarray, elapsed_s: float, period_s: float) -> np.ndarray:
    phase = (elapsed_s / period_s) % 1.0
    position = phase * template.shape[0] - 0.5
    lower_unwrapped = math.floor(position)
    blend = position - lower_unwrapped
    lower = lower_unwrapped % template.shape[0]
    upper = (lower + 1) % template.shape[0]
    return (1.0 - blend) * template[lower] + blend * template[upper]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state


def physical_state_sha256(state: np.ndarray, asset_manifest_sha256: str) -> str:
    state = np.asarray(state, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"official-x2-mjstate-integration-v1\0")
    digest.update(asset_manifest_sha256.encode("ascii"))
    digest.update(state.tobytes(order="C"))
    return digest.hexdigest()


def restore_integration_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: np.ndarray,
) -> None:
    """Restore integration state while making qpos-derived caches usable.

    ``mj_forward`` refreshes kinematics and sensors but may overwrite members
    of ``mjSTATE_INTEGRATION`` (notably warm-start state).  Restore the exact
    integration vector once more after forward; qpos/qvel are unchanged, so
    the derived kinematics remain valid for the next controller decision.
    """
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, np.asarray(state, dtype=np.float64), spec)
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(model, data, np.asarray(state, dtype=np.float64), spec)


class MatchedEventRunner:
    """Minimal Stage351 contract runner with Phase6 snapshot-compatible state."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
        self.data = mujoco.MjData(self.model)
        if not math.isclose(self.model.opt.timestep, 0.001, abs_tol=1.0e-12):
            raise RuntimeError(f"unexpected official MJCF timestep {self.model.opt.timestep}")
        self.substeps = round(CONTROL_DT_S / self.model.opt.timestep)
        self.session = ort.InferenceSession(
            str(args.model.resolve()), providers=["CPUExecutionProvider"]
        )
        self.stationary_session = ort.InferenceSession(
            str(args.stationary_model.resolve()), providers=["CPUExecutionProvider"]
        )
        archive = np.load(args.template, allow_pickle=False)
        if tuple(archive["joint_names_15"].tolist()) != LOWER_JOINTS:
            raise RuntimeError("gait-template joint order mismatch")
        self.template = archive["q_cycle_zero_mean_rad"].astype(np.float32)
        self.period = float(archive["period_s"])
        self.double_support = float(archive["double_support_fraction"])
        self.default = default_pose()
        self.gains = pd_gains()
        self.qpos_adr = {name: int(self.model.joint(name).qposadr[0]) for name in JOINTS}
        self.dof_adr = {name: int(self.model.joint(name).dofadr[0]) for name in JOINTS}
        self.actuator_id = {
            name: int(self.model.actuator(f"motor_{name}").id) for name in JOINTS
        }

        initial_payload = json.loads(args.initial_trace.read_text(encoding="utf-8"))
        initial = initial_payload["trace"][0]
        self.data.qpos[:3] = [
            initial["root_x_m"], initial["root_y_m"], initial["root_z_m"]
        ]
        half_yaw = 0.5 * float(initial["root_yaw_rad"])
        self.data.qpos[3:7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
        self.data.qvel[:3] = [
            initial["root_vx_w_mps"], initial["root_vy_w_mps"], 0.0
        ]
        self.data.qvel[5] = initial["root_yaw_rate_radps"]
        for name, value in official_start_pose().items():
            self.data.qpos[self.qpos_adr[name]] = value
        mujoco.mj_forward(self.model, self.data)

        # Mutable controller state: names intentionally match the Phase6
        # serialization contract.  No hidden RNG, filter, or wall clock is used.
        self.previous_actions = {
            slot: np.zeros(15, dtype=np.float32)
            for slot in ("main", "stationary", "recovery")
        }
        self.issued_actions = {
            slot: np.zeros(15, dtype=np.float32)
            for slot in ("main", "stationary", "recovery")
        }
        self.sequence_step = 0
        self.control_steps = 0
        self.stop_hold_latch_s = None
        self.stop_emergency_latch = False
        self.heading_target_rad = None
        self.heading_origin_xy = None
        self.move_heading_initialized = False
        self.stop_policy_initialized = False
        self.lateral_recovery_state = "off"
        self.lateral_recovery_bias = np.zeros(2, dtype=np.float32)
        self.heading_recovery_active = False
        self.heading_action_recovery_active = False
        self.heading_action_recovery_steps = 0
        self.last_move_targets = None
        self.stop_hold_targets = None
        self.prepare_start_q = {
            name: float(self.data.qpos[self.qpos_adr[name]]) for name in JOINTS
        }
        self.predicted_physical_step = -1
        self.previous_physical_observation = None
        self.predicted_physical_observation = None
        self.upper_previous_target = np.asarray(
            [self.default[name] for name in JOINTS if "shoulder" in name or "elbow" in name or "wrist" in name],
            dtype=np.float32,
        )
        self.upper_last_target = self.upper_previous_target.copy()
        self.upper_fallback_steps = 0
        self.upper_fallback_active = False
        self.upper_fallback_first_step = None
        self.finished = False

    def state_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        quaternion = self.data.qpos[3:7].copy()
        rotation = rotation_body_to_world(quaternion)
        lin_body = (rotation.T @ self.data.qvel[:3]).astype(np.float32)
        ang_body = self.data.qvel[3:6].astype(np.float32).copy()
        yaw, tilt = yaw_tilt(quaternion)
        return lin_body, ang_body, projected_gravity(quaternion), yaw, tilt

    def _session_obs(
        self,
        session: ort.InferenceSession,
        obs: np.ndarray,
        *,
        intent_vx: float = 0.0,
        future_vx: float = 0.0,
    ) -> np.ndarray:
        width = int(session.get_inputs()[0].shape[-1])
        if width == 93:
            return obs
        if width == 123:
            upper = np.zeros(28, dtype=np.float32)
            locomotion = np.asarray(
                [intent_vx / 0.5, (future_vx - intent_vx) / 0.5], dtype=np.float32
            )
            return np.concatenate((obs, upper, locomotion)).astype(np.float32)
        raise RuntimeError(f"unsupported actor input width {width}")

    def policy_targets(
        self,
        phase_elapsed_s: float,
        command_vx: float,
        *,
        policy_slot: str,
        force_moving: bool = False,
        future_vx: float | None = None,
        template_multiplier: float = 1.0,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        lin_body, ang_body, gravity, current_yaw, _ = self.state_values()
        if self.heading_target_rad is None:
            self.heading_target_rad = current_yaw
            self.heading_origin_xy = (float(self.data.qpos[0]), float(self.data.qpos[1]))
        command = np.asarray([command_vx, 0.0, 0.0], dtype=np.float32)
        joint_pos = np.asarray(
            [self.data.qpos[self.qpos_adr[name]] - self.default[name] for name in ISAAC_JOINTS],
            dtype=np.float32,
        )
        joint_vel = np.asarray(
            [self.data.qvel[self.dof_adr[name]] for name in ISAAC_JOINTS], dtype=np.float32
        )
        moving = force_moving or abs(command_vx) > 0.1
        phase = phase_features(
            phase_elapsed_s if moving else 0.0,
            moving,
            self.period,
            self.double_support,
        )
        obs = np.concatenate(
            (
                lin_body,
                ang_body,
                gravity,
                command,
                joint_pos,
                joint_vel,
                self.previous_actions[policy_slot],
                phase,
            )
        ).astype(np.float32)
        session = self.session if policy_slot == "main" else self.stationary_session
        model_obs = self._session_obs(
            session,
            obs,
            intent_vx=command_vx,
            future_vx=command_vx if future_vx is None else future_vx,
        )
        raw = session.run(["actions"], {"obs": model_obs[None]})[0][0].astype(np.float32)
        if policy_slot == "stationary" and self.args.stationary_blend < 1.0:
            main_obs = obs.copy()
            main_obs[74:89] = self.previous_actions["main"]
            main_model_obs = self._session_obs(self.session, main_obs)
            main_raw = self.session.run(
                ["actions"], {"obs": main_model_obs[None]}
            )[0][0].astype(np.float32)
            raw = (
                (1.0 - self.args.stationary_blend) * main_raw
                + self.args.stationary_blend * raw
            )
        residual = np.clip(raw, -1.0, 1.0)
        bias = (
            0.15
            * template_multiplier
            * template_bias(self.template, phase_elapsed_s, self.period)
            / LOWER_SCALE
            if moving
            else np.zeros(15, dtype=np.float32)
        )
        action = np.clip(residual + bias, -1.0, 1.0).astype(np.float32)
        self.previous_actions[policy_slot] = residual.copy()
        self.issued_actions[policy_slot] = action.copy()
        targets = dict(self.default)
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] += float(action[index] * LOWER_SCALE[index])
        return targets, obs, action

    def _move_speed(self, elapsed_s: float) -> float:
        return matched_event_speed(
            elapsed_s=elapsed_s,
            cruise_speed_mps=self.args.vx,
            accelerate_s=self.args.move_accelerate_seconds,
            cruise_s=self.args.move_seconds - self.args.move_accelerate_seconds,
            decelerate_s=self.args.stop_intent_decelerate_seconds,
        )

    def _controller_event(self, stage: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "sequence_step": self.sequence_step,
            "control_steps": self.control_steps,
            "heading_target_rad": self.heading_target_rad,
            "heading_origin_xy": self.heading_origin_xy,
            "move_heading_initialized": self.move_heading_initialized,
            "stop_policy_initialized": self.stop_policy_initialized,
            "stop_hold_latch_s": self.stop_hold_latch_s,
            "stop_emergency_latch": self.stop_emergency_latch,
            "lateral_recovery_state": self.lateral_recovery_state,
            "lateral_recovery_bias": self.lateral_recovery_bias.tolist(),
            "heading_recovery_active": self.heading_recovery_active,
            "heading_action_recovery_active": self.heading_action_recovery_active,
            "heading_action_recovery_steps": self.heading_action_recovery_steps,
            "previous_action_sha256": hashlib.sha256(
                np.concatenate([self.previous_actions[key] for key in ("main", "stationary", "recovery")]).tobytes()
            ).hexdigest(),
            "issued_action_sha256": hashlib.sha256(
                np.concatenate([self.issued_actions[key] for key in ("main", "stationary", "recovery")]).tobytes()
            ).hexdigest(),
        }

    def tick(self) -> dict[str, Any]:
        elapsed = self.sequence_step * CONTROL_DT_S
        self.sequence_step += 1
        prepare_end = self.args.prepare_seconds
        stand_end = prepare_end + self.args.stand_seconds
        move_end = stand_end + self.args.move_seconds

        obs: np.ndarray | None = None
        action: np.ndarray | None = None
        if elapsed < prepare_end:
            stage = "prepare"
            phase_elapsed = elapsed
            alpha = min(1.0, elapsed / max(self.args.prepare_seconds, 1.0e-6))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            targets = {
                name: self.prepare_start_q[name]
                + alpha * (self.default[name] - self.prepare_start_q[name])
                for name in JOINTS
            }
        elif elapsed < stand_end:
            stage = "stand"
            phase_elapsed = elapsed - prepare_end
            targets, obs, action = self.policy_targets(
                0.0, 0.0, policy_slot="stationary"
            )
        elif elapsed < move_end:
            stage = "move"
            phase_elapsed = elapsed - stand_end
            if not self.move_heading_initialized:
                self.issued_actions["main"] = self.issued_actions["stationary"].copy()
                _, _, _, yaw, _ = self.state_values()
                if self.heading_target_rad is None:
                    self.heading_target_rad = yaw
                    self.heading_origin_xy = (
                        float(self.data.qpos[0]), float(self.data.qpos[1])
                    )
                self.move_heading_initialized = True
            command_vx = self._move_speed(phase_elapsed)
            future_vx = self._move_speed(phase_elapsed + 1.0)
            targets, obs, action = self.policy_targets(
                phase_elapsed,
                command_vx,
                policy_slot="main",
                future_vx=future_vx,
                template_multiplier=self.args.move_template_multiplier,
            )
            self.last_move_targets = dict(targets)
            self.control_steps += 1
        else:
            stop_elapsed = elapsed - move_end
            phase_elapsed = stop_elapsed
            if not self.stop_policy_initialized:
                self.previous_actions["recovery"] = self.previous_actions["main"].copy()
                self.issued_actions["recovery"] = self.issued_actions["main"].copy()
                self.stop_policy_initialized = True
            if stop_elapsed < self.args.stop_transition_seconds:
                stage = "stop_curriculum"
                command_vx = self._move_speed(self.args.move_seconds + stop_elapsed)
                future_vx = self._move_speed(
                    self.args.move_seconds + stop_elapsed + 1.0
                )
                multiplier = float(
                    np.clip(abs(command_vx) / max(abs(self.args.vx), 1.0e-6), 0.0, 1.0)
                )
                targets, obs, action = self.policy_targets(
                    self.args.move_seconds + stop_elapsed,
                    command_vx,
                    policy_slot="main",
                    force_moving=True,
                    future_vx=future_vx,
                    template_multiplier=multiplier,
                )
            else:
                stage = "stop_stationary"
                if self.stop_hold_latch_s is None:
                    self.previous_actions["stationary"] = self.previous_actions["main"].copy()
                    self.issued_actions["stationary"] = self.issued_actions["main"].copy()
                    self.stop_hold_latch_s = stop_elapsed
                targets, obs, action = self.policy_targets(
                    0.0, 0.0, policy_slot="stationary"
                )

        for _ in range(self.substeps):
            for name in JOINTS:
                kp, kd = self.gains[name]
                kp *= self.args.pd_kp_multiplier
                kd *= self.args.pd_kd_multiplier
                torque = (
                    kp * (targets[name] - self.data.qpos[self.qpos_adr[name]])
                    - kd * self.data.qvel[self.dof_adr[name]]
                )
                actuator = self.actuator_id[name]
                low, high = self.model.actuator_ctrlrange[actuator]
                self.data.ctrl[actuator] = float(np.clip(torque, low, high))
            mujoco.mj_step(self.model, self.data)

        quaternion = self.data.qpos[3:7].copy()
        pitch_deg = math.degrees(
            signed_pitch_from_xyzw_rad(
                (quaternion[1], quaternion[2], quaternion[3], quaternion[0])
            )
        )
        return {
            "stage": stage,
            "elapsed_s": float(phase_elapsed),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "root": self.data.qpos[:7].copy(),
            "action": None if action is None else action.copy(),
            "root_pitch_deg": pitch_deg,
            "controller_event": self._controller_event(stage),
            "obs": None if obs is None else obs.copy(),
        }


def compare_rows(source: Mapping[str, Any], restored: Mapping[str, Any]) -> dict[str, Any]:
    errors = {
        field: float(np.max(np.abs(np.asarray(source[field]) - np.asarray(restored[field]))))
        for field in ("qpos", "qvel", "root")
    }
    if source["action"] is None or restored["action"] is None:
        action_exact = source["action"] is None and restored["action"] is None
        action_error = 0.0 if action_exact else float("inf")
    else:
        action_exact = bool(np.array_equal(source["action"], restored["action"]))
        action_error = float(
            np.max(np.abs(np.asarray(source["action"]) - np.asarray(restored["action"])))
        )
    event_exact = source["controller_event"] == restored["controller_event"]
    pitch_error = abs(float(source["root_pitch_deg"]) - float(restored["root_pitch_deg"]))
    passed = bool(
        max(errors.values()) <= PHYSICAL_TOLERANCE
        and action_exact
        and event_exact
        and pitch_error <= PITCH_TOLERANCE_DEG
    )
    return {
        **{f"{key}_abs_max": value for key, value in errors.items()},
        "action_abs_max": action_error,
        "action_bitwise_exact": action_exact,
        "controller_event_exact": event_exact,
        "root_pitch_abs_error_deg": pitch_error,
        "passed": passed,
    }


def phase_pitch_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {key: [] for key in ("stand", "start", "move", "stop")}
    for row in rows:
        stage = str(row["stage"])
        if stage == "stand":
            key = "stand"
        elif stage == "move" and float(row["elapsed_s"]) < 1.0:
            key = "start"
        elif stage == "move":
            key = "move"
        elif stage.startswith("stop"):
            key = "stop"
        else:
            continue
        grouped[key].append(float(row["root_pitch_deg"]))
    result: dict[str, Any] = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        result[key] = {
            "count": int(array.size),
            "mean_deg": None if not array.size else float(np.mean(array)),
            "p05_deg": None if not array.size else float(np.quantile(array, 0.05)),
            "p95_deg": None if not array.size else float(np.quantile(array, 0.95)),
        }
    return result


def jsonable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in row.items()
        if key != "obs"
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source = MatchedEventRunner(args)
    move_end_step = round(
        (args.prepare_seconds + args.stand_seconds + args.move_seconds) / CONTROL_DT_S
    )
    snapshot_step = move_end_step + round(SNAPSHOT_NEXT_STOP_ELAPSED_S / CONTROL_DT_S)
    prefix: list[dict[str, Any]] = []
    while source.sequence_step < snapshot_step:
        prefix.append(source.tick())

    state = integration_state(source.model, source.data)
    state_sha = physical_state_sha256(state, args.asset_manifest_sha256)
    controller_state = capture_controller_state(source, physical_state_sha256=state_sha)
    # Snapshot restore must be exact before any suffix step; this is a stronger
    # condition than the explicitly tolerated future floating-point comparison.
    restored = MatchedEventRunner(args)
    restore_integration_state(restored.model, restored.data, state)
    restored_state = integration_state(restored.model, restored.data)
    physical_roundtrip_bitwise_exact = bool(np.array_equal(state, restored_state))
    restored_state_sha = physical_state_sha256(restored_state, args.asset_manifest_sha256)
    restore_controller_state(
        restored,
        controller_state,
        expected_physical_state_sha256=restored_state_sha,
    )
    controller_roundtrip_exact = bool(
        capture_controller_state(restored, physical_state_sha256=restored_state_sha)
        == controller_state
    )

    source_suffix = [source.tick() for _ in range(max(CHECKPOINT_TICKS))]
    restored_suffix = [restored.tick() for _ in range(max(CHECKPOINT_TICKS))]
    comparisons = [
        compare_rows(source_row, restored_row)
        for source_row, restored_row in zip(source_suffix, restored_suffix)
    ]
    first_failure = next(
        (
            {"tick": index + 1, **comparison}
            for index, comparison in enumerate(comparisons)
            if not comparison["passed"]
        ),
        None,
    )
    checkpoint_results = {
        str(tick): comparisons[tick - 1] for tick in CHECKPOINT_TICKS
    }
    suffix_pass = bool(
        physical_roundtrip_bitwise_exact
        and controller_roundtrip_exact
        and first_failure is None
    )
    snapshot_payload = {
        "schema": "official_x2_testonly_matched_event_snapshot_v1",
        "evidence_boundary": "unchanged_official_mjcf_same_process_not_ros",
        "asset_manifest_sha256": args.asset_manifest_sha256,
        "physical_state_spec": "mjSTATE_INTEGRATION",
        "physical_state_sha256": state_sha,
        "physical_state_float64": state.tolist(),
        "controller_state": controller_state,
        "snapshot_before_next_tick": {
            "sequence_step": snapshot_step,
            "next_stage": "stop_curriculum",
            "next_stop_elapsed_s": SNAPSHOT_NEXT_STOP_ELAPSED_S,
        },
    }
    result = {
        "pre_registration": {
            "hypothesis": (
                "The complete mjSTATE_INTEGRATION vector plus the Phase6 controller "
                "snapshot is sufficient for deterministic matched-event suffixes."
            ),
            "intervention": "restore both physical and controller states into a fresh runner",
            "control": "uninterrupted source runner from the same snapshot boundary",
            "checkpoints_ticks": list(CHECKPOINT_TICKS),
            "tolerances": {
                "physical_snapshot_roundtrip": "bitwise exact",
                "qpos_qvel_root_abs_max": PHYSICAL_TOLERANCE,
                "action": "bitwise exact",
                "controller_event": "exact",
                "signed_root_pitch_abs_deg": PITCH_TOLERANCE_DEG,
            },
            "failure_rule": "first failing future tick identifies the first divergent field",
        },
        "evidence_boundary": {
            "official_scene_xml_unchanged": True,
            "same_process_test_only": True,
            "official_aimdk_ros_closed_loop": False,
            "reason": "public AimDK simulator API has no lossless mid-event full-state injection",
        },
        "snapshot": snapshot_payload,
        "roundtrip": {
            "physical_bitwise_exact": physical_roundtrip_bitwise_exact,
            "controller_exact": controller_roundtrip_exact,
        },
        "checkpoint_results": checkpoint_results,
        "first_failure": first_failure,
        "deterministic_suffix_pass": suffix_pass,
        "signed_root_pitch_phase_summary": phase_pitch_summary(prefix + source_suffix),
        "source_suffix": [jsonable_row(row) for row in source_suffix],
        "restored_suffix": [jsonable_row(row) for row in restored_suffix],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stationary-model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--initial-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-output", type=Path, required=True)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--prepare-seconds", type=float, default=0.2)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--move-seconds", type=float, default=5.2)
    parser.add_argument("--stop-seconds", type=float, default=8.0)
    parser.add_argument("--move-accelerate-seconds", type=float, default=1.0)
    parser.add_argument("--stop-intent-decelerate-seconds", type=float, default=2.0)
    parser.add_argument("--stop-transition-seconds", type=float, default=2.0)
    parser.add_argument("--move-template-multiplier", type=float, default=1.0)
    parser.add_argument("--stationary-blend", type=float, default=0.5)
    parser.add_argument("--pd-kp-multiplier", type=float, default=1.2)
    parser.add_argument("--pd-kd-multiplier", type=float, default=1.2)
    args = parser.parse_args()
    args.clock_mode = "step"
    assets = {
        "scene": sha256_file(args.scene.resolve()),
        "model": sha256_file(args.model.resolve()),
        "stationary_model": sha256_file(args.stationary_model.resolve()),
        "template": sha256_file(args.template.resolve()),
    }
    args.asset_hashes = assets
    args.asset_manifest_sha256 = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return args


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.snapshot_output.write_text(
        json.dumps(result["snapshot"], indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "deterministic_suffix_pass": result["deterministic_suffix_pass"],
                "roundtrip": result["roundtrip"],
                "checkpoint_results": result["checkpoint_results"],
                "first_failure": result["first_failure"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
