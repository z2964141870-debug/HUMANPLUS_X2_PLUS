#!/usr/bin/env python3
"""BASE Phase26 test-only dynamic bridge teacher in the unchanged official MJCF.

This runner deliberately does not claim AimDK ROS equivalence.  It reuses the
Phase8 same-process ``mjSTATE_INTEGRATION`` + controller snapshot boundary,
then compares a zero-residual bridge with one bounded, low-dimensional CEM
search from the exact same snapshot.  No policy weights or vendor assets are
modified.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from official_x2.audit_phase25_role_divergence import calibrate, classify
from official_x2.controller_snapshot_contract import capture_controller_state, restore_controller_state
from official_x2.outcome_aware_state_role_v2 import load_manifest, resolve_rows
from official_x2.replay_official_trace_direct_mujoco import JOINTS, LOWER_JOINTS, LOWER_SCALE, yaw_tilt
from official_x2.run_testonly_official_mjcf_suffix_probe import (
    CONTROL_DT_S,
    MatchedEventRunner,
    integration_state,
    physical_state_sha256,
    restore_integration_state,
)


SNAPSHOT_STOP_ELAPSED_S = 1.80
BRIDGE_SECONDS = 1.00
STRICT_POST_HANDOFF_SECONDS = 1.00
REPORT_POST_HANDOFF_SECONDS = 2.00
KNOTS = 2
MODE_NAMES = ("sagittal_crouch", "hip_roll_anti", "ankle_roll_anti", "waist_pitch", "waist_roll")
LOW = np.full(len(MODE_NAMES) * KNOTS, -0.20, dtype=np.float64)
HIGH = np.full(len(MODE_NAMES) * KNOTS, 0.20, dtype=np.float64)
ESSENTIAL_GROUPS = (
    "base_lin_vel", "base_ang_vel", "projected_gravity",
    "joint_position", "joint_velocity", "previous_action", "root_posture",
)
ROOT_Z_MIN_M = 0.55
ROOT_TILT_MAX_RAD = 0.30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mode_matrix() -> np.ndarray:
    modes = np.zeros((len(MODE_NAMES), 15), dtype=np.float64)
    # Normalized action modes, not direct root teleportation.
    modes[0, [0, 6]] = 0.5
    modes[0, [3, 9]] = 1.0
    modes[0, [4, 10]] = -0.5
    modes[1, [1, 7]] = [1.0, -1.0]
    modes[2, [5, 11]] = [1.0, -1.0]
    modes[3, 13] = 1.0
    modes[4, 14] = 1.0
    norms = np.linalg.norm(modes, axis=1, keepdims=True)
    return modes / np.where(norms > 0, norms, 1.0)


MODES = mode_matrix()


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def residual_at(raw: np.ndarray, fraction: float) -> np.ndarray:
    knots = np.clip(np.asarray(raw, dtype=np.float64), LOW, HIGH).reshape(KNOTS, len(MODE_NAMES))
    position = float(np.clip(fraction, 0.0, 1.0)) * (KNOTS - 1)
    lower = min(KNOTS - 1, int(math.floor(position)))
    upper = min(KNOTS - 1, lower + 1)
    blend = smoothstep(position - lower)
    coeff = (1.0 - blend) * knots[lower] + blend * knots[upper]
    return np.asarray(coeff @ MODES, dtype=np.float32)


def build_support(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    indices = {
        role: [int(row["manifest_row_index"]) for row in manifest["rows"] if row.get("eligible") and row.get("state_role") == role]
        for role in ("success_safe", "critical_from_failure")
    }
    rows = {role: resolve_rows(manifest, values) for role, values in indices.items()}
    extractors = {
        "base_lin_vel": lambda row: row["observation_93d"][0:3],
        "base_ang_vel": lambda row: row["observation_93d"][3:6],
        "projected_gravity": lambda row: row["observation_93d"][6:9],
        "joint_position": lambda row: row["observation_93d"][12:43],
        "joint_velocity": lambda row: row["observation_93d"][43:74],
        "previous_action": lambda row: row["observation_93d"][74:89],
        "root_posture": lambda row: [row["physical_state"]["root_position_m"][2], root_tilt(row["physical_state"]["root_quaternion_xyzw"])],
    }
    return {
        name: calibrate(
            np.asarray([fn(row) for row in rows["success_safe"]], dtype=np.float64),
            np.asarray([fn(row) for row in rows["critical_from_failure"]], dtype=np.float64),
        )
        for name, fn in extractors.items()
    }


def root_tilt(quaternion_xyzw: list[float] | np.ndarray) -> float:
    x, y, _z, _w = map(float, quaternion_xyzw)
    return float(math.acos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


class BridgeRunner(MatchedEventRunner):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.recovery_session = ort.InferenceSession(
            str(args.recovery_model.resolve()), providers=["CPUExecutionProvider"]
        )

    @property
    def move_end_s(self) -> float:
        return self.args.prepare_seconds + self.args.stand_seconds + self.args.move_seconds

    def step_physics(self, targets: dict[str, float]) -> None:
        for _ in range(self.substeps):
            for name in JOINTS:
                kp, kd = self.gains[name]
                kp *= self.args.pd_kp_multiplier
                kd *= self.args.pd_kd_multiplier
                torque = kp * (targets[name] - self.data.qpos[self.qpos_adr[name]]) - kd * self.data.qvel[self.dof_adr[name]]
                actuator = self.actuator_id[name]
                low, high = self.model.actuator_ctrlrange[actuator]
                self.data.ctrl[actuator] = float(np.clip(torque, low, high))
            mujoco.mj_step(self.model, self.data)

    def query(self, policy_slot: str) -> dict[str, np.ndarray | float]:
        lin, ang, gravity, _yaw, tilt = self.state_values()
        joint_pos = np.asarray(
            [self.data.qpos[self.qpos_adr[name]] - self.default[name] for name in self.args.isaac_joints],
            dtype=np.float64,
        )
        joint_vel = np.asarray(
            [self.data.qvel[self.dof_adr[name]] for name in self.args.isaac_joints], dtype=np.float64
        )
        return {
            "base_lin_vel": np.asarray(lin, dtype=np.float64),
            "base_ang_vel": np.asarray(ang, dtype=np.float64),
            "projected_gravity": np.asarray(gravity, dtype=np.float64),
            "joint_position": joint_pos,
            "joint_velocity": joint_vel,
            "previous_action": np.asarray(self.previous_actions[policy_slot], dtype=np.float64),
            "root_posture": np.asarray([self.data.qpos[2], tilt], dtype=np.float64),
        }

    def recovery_targets(self) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        saved_session = self.stationary_session
        self.stationary_session = self.recovery_session
        self.previous_actions["stationary"] = self.previous_actions["recovery"].copy()
        self.issued_actions["stationary"] = self.issued_actions["recovery"].copy()
        try:
            targets, obs, action = self.policy_targets(0.0, 0.0, policy_slot="stationary")
            self.previous_actions["recovery"] = self.previous_actions["stationary"].copy()
            self.issued_actions["recovery"] = self.issued_actions["stationary"].copy()
            return targets, obs, action
        finally:
            self.stationary_session = saved_session


def support_scores(state: dict[str, np.ndarray | float], support: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: classify(np.asarray(state[name]), support[name]) for name in ESSENTIAL_GROUPS}


def integration_row(runner: BridgeRunner, stage: str, action: np.ndarray, scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    _yaw, tilt = yaw_tilt(runner.data.qpos[3:7])
    return {
        "stage": stage,
        "sequence_step": int(runner.sequence_step),
        "qpos": runner.data.qpos.copy(), "qvel": runner.data.qvel.copy(),
        "action": np.asarray(action, dtype=np.float32).copy(),
        "root_z_m": float(runner.data.qpos[2]), "root_tilt_rad": float(tilt),
        "scores": scores,
    }


def run_from_snapshot(
    runner: BridgeRunner,
    state: np.ndarray,
    controller_state: dict[str, Any],
    state_sha: str,
    support: dict[str, dict[str, Any]],
    raw: np.ndarray,
    *,
    recovery_seconds: float,
) -> dict[str, Any]:
    restore_integration_state(runner.model, runner.data, state)
    restore_controller_state(runner, controller_state, expected_physical_state_sha256=state_sha)
    bridge_ticks = round(BRIDGE_SECONDS / CONTROL_DT_S)
    bridge_rows = []
    last_targets: dict[str, float] | None = None
    for tick in range(bridge_ticks):
        elapsed = runner.sequence_step * CONTROL_DT_S
        stop_elapsed = elapsed - runner.move_end_s
        if stop_elapsed < runner.args.stop_transition_seconds:
            command_vx = runner._move_speed(runner.args.move_seconds + stop_elapsed)
            future_vx = runner._move_speed(runner.args.move_seconds + stop_elapsed + 1.0)
            multiplier = float(np.clip(abs(command_vx) / max(abs(runner.args.vx), 1e-6), 0.0, 1.0))
        else:
            command_vx = future_vx = multiplier = 0.0
        targets, _obs, issued = runner.policy_targets(
            runner.args.move_seconds + stop_elapsed, command_vx,
            policy_slot="main", force_moving=True, future_vx=future_vx,
            template_multiplier=multiplier,
        )
        delta = residual_at(raw, (tick + 1) / bridge_ticks)
        policy_residual = np.clip(runner.previous_actions["main"] + delta, -1.0, 1.0).astype(np.float32)
        final_action = np.clip(issued + delta, -1.0, 1.0).astype(np.float32)
        runner.previous_actions["main"] = policy_residual.copy()
        runner.issued_actions["main"] = final_action.copy()
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] = runner.default[name] + float(final_action[index] * LOWER_SCALE[index])
        runner.step_physics(targets)
        runner.sequence_step += 1
        runner.control_steps += 1
        last_targets = dict(targets)
        state_now = runner.query("main")
        bridge_rows.append(integration_row(runner, "bridge", final_action, support_scores(state_now, support)))

    if last_targets is None:
        raise RuntimeError("empty bridge")
    runner.previous_actions["recovery"] = runner.previous_actions["main"].copy()
    runner.issued_actions["recovery"] = runner.issued_actions["main"].copy()
    runner.stop_hold_targets = dict(last_targets)
    runner.stop_hold_latch_s = runner.sequence_step * CONTROL_DT_S - runner.move_end_s
    endpoint = support_scores(runner.query("main"), support)

    recovery_rows = []
    recovery_ticks = round(recovery_seconds / CONTROL_DT_S)
    for tick in range(recovery_ticks):
        targets, _obs, proposal = runner.recovery_targets()
        blend_fraction = smoothstep((tick * CONTROL_DT_S) / 0.5)
        blended = {
            name: (1.0 - blend_fraction) * runner.stop_hold_targets[name] + blend_fraction * targets[name]
            for name in JOINTS
        }
        actual = np.asarray(
            [(blended[name] - runner.default[name]) / LOWER_SCALE[index] for index, name in enumerate(LOWER_JOINTS)],
            dtype=np.float32,
        )
        actual = np.clip(actual, -1.0, 1.0)
        runner.previous_actions["recovery"] = actual.copy()
        runner.issued_actions["recovery"] = actual.copy()
        runner.previous_actions["stationary"] = actual.copy()
        runner.issued_actions["stationary"] = actual.copy()
        runner.step_physics(blended)
        runner.sequence_step += 1
        state_now = runner.query("recovery")
        recovery_rows.append(integration_row(runner, "recovery", actual, support_scores(state_now, support)))

    return {
        "endpoint_scores": endpoint,
        "bridge_rows": bridge_rows,
        "recovery_rows": recovery_rows,
        "raw": np.asarray(raw, dtype=np.float64).copy(),
    }


def score_rollout(rollout: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    endpoint = rollout["endpoint_scores"]
    endpoint_values = [float(endpoint[name]["success_normalized"]) for name in ESSENTIAL_GROUPS]
    strict_ticks = rollout["recovery_rows"][: round(STRICT_POST_HANDOFF_SECONDS / CONTROL_DT_S)]
    union_normalized = {
        name: [float(row["scores"][name]["union_distance"] / row["scores"][name]["union_threshold"]) for row in strict_ticks]
        for name in ESSENTIAL_GROUPS
    }
    endpoint_pass = all(value <= 1.0 for value in endpoint_values)
    support_pass = all(max(values, default=float("inf")) <= 1.0 for values in union_normalized.values())
    all_rows = rollout["bridge_rows"] + strict_ticks
    safety_pass = all(row["root_z_m"] >= ROOT_Z_MIN_M and row["root_tilt_rad"] <= ROOT_TILT_MAX_RAD for row in all_rows)
    endpoint_cost = float(np.mean(np.square(np.minimum(endpoint_values, 5.0))))
    post_cost = float(np.mean([min(max(values, default=5.0), 5.0) ** 2 for values in union_normalized.values()]))
    safety_violation = sum(
        max(0.0, ROOT_Z_MIN_M - row["root_z_m"]) ** 2 * 100.0
        + max(0.0, row["root_tilt_rad"] - ROOT_TILT_MAX_RAD) ** 2 * 100.0
        for row in all_rows
    )
    regularization = 0.05 * float(np.mean(np.square(rollout["raw"])))
    cost = endpoint_cost + post_cost + safety_violation + regularization
    diagnostics = {
        "endpoint_success_normalized": dict(zip(ESSENTIAL_GROUPS, endpoint_values)),
        "post_1s_union_normalized_max": {name: max(values, default=None) for name, values in union_normalized.items()},
        "endpoint_success_joint_support": endpoint_pass,
        "post_1s_union_support": support_pass,
        "root_safety_through_post_1s": safety_pass,
        "strict_bridge_success": bool(endpoint_pass and support_pass and safety_pass),
        "cost": cost,
    }
    return cost, diagnostics


def exact_rows(left: list[dict[str, Any]], right: list[dict[str, Any]], ticks: int) -> bool:
    for a, b in zip(left[:ticks], right[:ticks]):
        for key in ("qpos", "qvel", "action"):
            if not np.array_equal(a[key], b[key]):
                return False
    return True


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    support = build_support(args.manifest)
    source = BridgeRunner(args)
    snapshot_step = round((source.move_end_s + SNAPSHOT_STOP_ELAPSED_S) / CONTROL_DT_S)
    while source.sequence_step < snapshot_step:
        source.tick()
    state = integration_state(source.model, source.data)
    state_sha = physical_state_sha256(state, args.asset_manifest_sha256)
    controller_state = capture_controller_state(source, physical_state_sha256=state_sha)
    scratch = BridgeRunner(args)
    restore_integration_state(scratch.model, scratch.data, state)
    restored_state = integration_state(scratch.model, scratch.data)
    restore_controller_state(scratch, controller_state, expected_physical_state_sha256=physical_state_sha256(restored_state, args.asset_manifest_sha256))
    roundtrip = {
        "physical_bitwise_exact": bool(np.array_equal(state, restored_state)),
        "controller_exact": capture_controller_state(scratch, physical_state_sha256=state_sha) == controller_state,
    }

    zero = np.zeros_like(LOW)
    preflight_a = run_from_snapshot(scratch, state, controller_state, state_sha, support, zero, recovery_seconds=STRICT_POST_HANDOFF_SECONDS)
    preflight_b = run_from_snapshot(scratch, state, controller_state, state_sha, support, zero, recovery_seconds=STRICT_POST_HANDOFF_SECONDS)
    fork_exact = exact_rows(preflight_a["bridge_rows"] + preflight_a["recovery_rows"], preflight_b["bridge_rows"] + preflight_b["recovery_rows"], 10)
    if not all(roundtrip.values()) or not fork_exact:
        return {
            "stage": "BASE Phase26 test-only bridge teacher",
            "evidence_boundary": evidence_boundary(),
            "preflight": {"roundtrip": roundtrip, "two_fork_first_10_ticks_exact": fork_exact},
            "cem_executed": False,
            "decision": "STOPPED_BEFORE_CEM_NONDETERMINISTIC_SNAPSHOT",
        }

    baseline_cost, baseline_diag = score_rollout(preflight_a)
    if args.preflight_only:
        return {
            "stage": "BASE Phase26 test-only bridge preflight",
            "evidence_boundary": evidence_boundary(),
            "assets": args.asset_hashes,
            "snapshot": {
                "physical_state_sha256": state_sha,
                "controller_state": controller_state,
                "sequence_step": snapshot_step,
                "next_stop_elapsed_s": SNAPSHOT_STOP_ELAPSED_S,
            },
            "preflight": {
                "roundtrip": roundtrip,
                "two_fork_first_10_ticks_exact": fork_exact,
                "zero_bridge_cost": baseline_cost,
                "zero_bridge_diagnostics": baseline_diag,
            },
            "cem_executed": False,
            "decision": "PREFLIGHT_PASS_CEM_NOT_RUN",
        }
    rng = np.random.default_rng(args.seed)
    mean = np.zeros_like(LOW)
    std = np.full_like(LOW, 0.10)
    best = {"cost": baseline_cost, "raw": zero.copy(), "diag": baseline_diag}
    iterations = []
    evaluations = []
    for iteration in range(args.iterations):
        population = np.clip(rng.normal(mean, std, size=(args.population, len(mean))), LOW, HIGH)
        population[0] = mean
        if iteration == 0:
            population[0] = zero
        scored = []
        for raw in population:
            rollout = run_from_snapshot(scratch, state, controller_state, state_sha, support, raw, recovery_seconds=STRICT_POST_HANDOFF_SECONDS)
            cost, diag = score_rollout(rollout)
            record = {"iteration": iteration, "raw": raw.copy(), "cost": cost, "diagnostics": diag}
            evaluations.append(record)
            scored.append(record)
            if cost < best["cost"]:
                best = {"cost": cost, "raw": raw.copy(), "diag": diag}
        scored.sort(key=lambda item: item["cost"])
        elites = scored[: args.elites]
        values = np.asarray([item["raw"] for item in elites])
        mean = 0.25 * mean + 0.75 * np.mean(values, axis=0)
        std = np.maximum(0.03, 0.25 * std + 0.75 * np.std(values, axis=0))
        iterations.append({
            "iteration": iteration, "best_cost": float(scored[0]["cost"]),
            "best_strict_success": bool(scored[0]["diagnostics"]["strict_bridge_success"]),
        })

    baseline_full = run_from_snapshot(scratch, state, controller_state, state_sha, support, zero, recovery_seconds=REPORT_POST_HANDOFF_SECONDS)
    best_full = run_from_snapshot(scratch, state, controller_state, state_sha, support, best["raw"], recovery_seconds=REPORT_POST_HANDOFF_SECONDS)
    baseline_full_cost, baseline_full_diag = score_rollout(baseline_full)
    best_full_cost, best_full_diag = score_rollout(best_full)
    return {
        "stage": "BASE Phase26 test-only official-MJCF dynamic bridge teacher",
        "pre_registration": {
            "snapshot_stop_elapsed_s": SNAPSHOT_STOP_ELAPSED_S,
            "bridge_seconds": BRIDGE_SECONDS,
            "strict_post_handoff_seconds": STRICT_POST_HANDOFF_SECONDS,
            "report_post_handoff_seconds": REPORT_POST_HANDOFF_SECONDS,
            "variables": f"{len(MODE_NAMES)} fixed normalized lower-body modes x {KNOTS} C2 knots",
            "bounds": [-0.20, 0.20],
            "seed_population_iterations_elites": [args.seed, args.population, args.iterations, args.elites],
            "success": "endpoint all groups in success-safe LOO-p95 AND next 1s all groups in union LOO-p95 AND root-z/tilt safe",
            "no_handoff_is_success": False,
        },
        "evidence_boundary": evidence_boundary(),
        "assets": args.asset_hashes,
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "snapshot": {
            "physical_state_sha256": state_sha,
            "controller_state": controller_state,
            "sequence_step": snapshot_step,
            "next_stop_elapsed_s": SNAPSHOT_STOP_ELAPSED_S,
        },
        "preflight": {"roundtrip": roundtrip, "two_fork_first_10_ticks_exact": fork_exact},
        "cem_executed": True,
        "search": {"iterations": iterations, "evaluation_count": len(evaluations)},
        "baseline": {"cost": baseline_full_cost, "diagnostics": baseline_full_diag, "rollout": baseline_full},
        "best": {"cost": best_full_cost, "raw": best["raw"], "diagnostics": best_full_diag, "rollout": best_full},
        "decision": {
            "strict_dynamic_bridge_found": bool(best_full_diag["strict_bridge_success"]),
            "improves_cost_over_same_snapshot_zero_control": bool(best_full_cost < baseline_full_cost),
            "handoff_executed_for_both": True,
            "training_unlocked": False,
        },
    }


def evidence_boundary() -> dict[str, Any]:
    return {
        "official_scene_xml_unchanged": True,
        "same_process_test_only_mjcf": True,
        "official_aimdk_ros_closed_loop": False,
        "reason": "public AimDK ROS simulator has no lossless mid-event state injection",
        "root_is_never_prescribed_or_teleported": True,
        "contact_com_dynamics_are_model_estimates_not_hardware_truth": True,
        "ppo_or_policy_training": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stationary-model", type=Path, required=True)
    parser.add_argument("--recovery-model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--initial-trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--prepare-seconds", type=float, default=0.2)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--move-seconds", type=float, default=5.2)
    parser.add_argument("--move-accelerate-seconds", type=float, default=1.0)
    parser.add_argument("--stop-intent-decelerate-seconds", type=float, default=2.0)
    parser.add_argument("--stop-transition-seconds", type=float, default=2.0)
    parser.add_argument("--move-template-multiplier", type=float, default=1.0)
    parser.add_argument("--stationary-blend", type=float, default=0.5)
    parser.add_argument("--pd-kp-multiplier", type=float, default=1.2)
    parser.add_argument("--pd-kd-multiplier", type=float, default=1.2)
    args = parser.parse_args()
    args.stop_seconds = 8.0
    args.clock_mode = "step"
    args.isaac_joints = (
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
    args.asset_hashes = {
        name: sha256(path.resolve()) for name, path in {
            "scene": args.scene, "model": args.model, "stationary_model": args.stationary_model,
            "recovery_model": args.recovery_model, "template": args.template,
            "initial_trace": args.initial_trace,
        }.items()
    }
    args.asset_manifest_sha256 = hashlib.sha256(json.dumps(args.asset_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"preflight": result["preflight"], "cem_executed": result["cem_executed"], "decision": result.get("decision")}, indent=2))


if __name__ == "__main__":
    main()
