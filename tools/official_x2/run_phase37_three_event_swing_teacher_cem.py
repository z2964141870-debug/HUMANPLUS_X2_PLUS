#!/usr/bin/env python3
"""One-shot three-event swing teacher on the frozen Phase36 exact window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from official_x2.analyze_phase34_full_closed_trace import read_trace, sha256
from official_x2.audit_phase35_contact_core import core_and_edge_masks
from official_x2.audit_stage250_native_dynamic_seed import contact_geom_ids, foot_points, quantiles
from official_x2.replay_official_trace_direct_mujoco import JOINTS, pd_gains, yaw_tilt
from official_x2.run_phase36_exact_swing_teacher_cem import (
    exact_prefix,
    geom_contract,
    integration_state,
    longest_false_window,
    restore_integration_state,
    seed_state,
    select_window,
    smoothstep,
)


CONTROL_TICKS = 18
SUBSTEPS = 20
FOLLOW_TICKS = 4
TEACHER_STEPS = CONTROL_TICKS * SUBSTEPS
SEED = 3601
POPULATION = 24
ITERATIONS = 5
ELITES = 6
PARAMETER_NAMES = (
    "prep_stance_hip_roll", "prep_waist_roll",
    "swing_hip_pitch", "swing_knee", "swing_ankle_pitch",
    "touchdown_knee", "touchdown_ankle_pitch",
    "liftoff_shift_s", "touchdown_shift_s",
)
BOUNDS = np.asarray([0.05, 0.04, 0.08, 0.12, 0.06, 0.06, 0.04, 0.02, 0.02], dtype=np.float64)
NOMINAL_LIFTOFF_S = 0.10
NOMINAL_TOUCHDOWN_S = 0.26
TEACHER_SECONDS = 0.36
MIN_SWING_OFF_TICKS = 100
MIN_CLEARANCE_M = 0.012
MAX_STANCE_CORE_SLIP_MPS = 0.10
MIN_ROOT_Z_M = 0.55
MAX_ROOT_TILT_RAD = 0.30
TERMINAL_CONTACT_TICKS = 40


def three_event_residual(raw: np.ndarray, time_s: float, swing_side: str) -> dict[str, float]:
    value = np.clip(np.asarray(raw, dtype=np.float64), -BOUNDS, BOUNDS)
    liftoff = NOMINAL_LIFTOFF_S + float(value[7])
    touchdown = NOMINAL_TOUCHDOWN_S + float(value[8])
    if touchdown - liftoff < 0.10:
        midpoint = 0.5 * (liftoff + touchdown)
        liftoff, touchdown = midpoint - 0.05, midpoint + 0.05
    stance = "right" if swing_side == "left" else "left"

    # Anticipatory support-load shift: C2 ramp through the pre-liftoff event,
    # hold during swing, then C2 release during touchdown.
    load = smoothstep(time_s / max(liftoff, 1e-6))
    if time_s >= touchdown:
        load = 1.0 - smoothstep((time_s - touchdown) / max(TEACHER_SECONDS - touchdown, 1e-6))

    # Independent swing event: C2 bell between liftoff and touchdown.
    phase = np.clip((time_s - liftoff) / max(touchdown - liftoff, 1e-6), 0.0, 1.0)
    lift = 4.0 * smoothstep(float(phase)) * (1.0 - smoothstep(float(phase)))

    # Independent touchdown blend, decaying exactly to zero at horizon.
    touchdown_blend = 0.0
    if time_s >= touchdown:
        touchdown_blend = 1.0 - smoothstep((time_s - touchdown) / max(TEACHER_SECONDS - touchdown, 1e-6))
    return {
        f"{stance}_hip_roll_joint": float(value[0] * load),
        "waist_roll_joint": float(value[1] * load),
        f"{swing_side}_hip_pitch_joint": float(value[2] * lift),
        f"{swing_side}_knee_joint": float(value[3] * lift + value[5] * touchdown_blend),
        f"{swing_side}_ankle_pitch_joint": float(value[4] * lift + value[6] * touchdown_blend),
    }


def contact_state(model: mujoco.MjModel, data: mujoco.MjData,
                  floor: int, geom_side: dict[int, str]) -> dict[str, bool]:
    result = {"left": False, "right": False}
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if floor not in (geom1, geom2):
            continue
        side = geom_side.get(geom2 if geom1 == floor else geom1)
        if side:
            result[side] = True
    return result


def ctrl_with_position_residual(baseline: np.ndarray, residual: dict[str, float],
                                actuator: dict[str, int], gains: dict[str, tuple[float, float]]) -> np.ndarray:
    """Preserve recorded ctrl exactly at zero; only add bounded PD-equivalent delta."""
    result = np.asarray(baseline, dtype=np.float64).copy()
    for name, delta_rad in residual.items():
        result[actuator[name]] = float(baseline[actuator[name]] + gains[name][0] * delta_rad)
    return result


def run_candidate(model: mujoco.MjModel, physical: dict[str, Any], state: np.ndarray,
                  selection: dict[str, Any], raw: np.ndarray) -> dict[str, Any]:
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    restore_integration_state(model, data, state)
    floor, geom_side, bodies = geom_contract(model)
    sole_geoms = {side: contact_geom_ids(model, side) for side in bodies}
    actuator = {name: int(model.actuator(f"motor_{name}").id) for name in JOINTS}
    gains = pd_gains()
    anchor = selection["anchor_physics_index"]
    total = (CONTROL_TICKS + FOLLOW_TICKS) * SUBSTEPS
    contacts = {side: np.zeros(total, dtype=bool) for side in bodies}
    clearance = {side: np.zeros(total) for side in bodies}
    stance_point_speed = np.full(total, np.nan)
    root_z, root_tilt = np.zeros(total), np.zeros(total)
    residual_ctrl_absmax = 0.0
    recorded_qpos_absmax = 0.0
    recorded_qvel_absmax = 0.0
    for step in range(total):
        source_qpos, source_qvel = data.qpos.copy(), data.qvel.copy()
        baseline = physical["ctrl"][anchor + 1 + step]
        data.ctrl[:] = baseline
        if step < TEACHER_STEPS:
            residual = three_event_residual(raw, (step + 1) / 1000.0, selection["swing_side"])
            data.ctrl[:] = ctrl_with_position_residual(baseline, residual, actuator, gains)
            residual_ctrl_absmax = max(
                residual_ctrl_absmax,
                max((abs(gains[name][0] * delta) for name, delta in residual.items()), default=0.0),
            )
        mujoco.mj_step(model, data)
        recorded_index = anchor + 1 + step
        recorded_qpos_absmax = max(
            recorded_qpos_absmax,
            float(np.max(np.abs(data.qpos - physical["qpos"][recorded_index]))),
        )
        recorded_qvel_absmax = max(
            recorded_qvel_absmax,
            float(np.max(np.abs(data.qvel - physical["qvel"][recorded_index]))),
        )
        realized = contact_state(model, data, floor, geom_side)
        for side in bodies:
            contacts[side][step] = realized[side]
            clearance[side][step] = max(0.0, float(foot_points(model, data, sole_geoms[side])["min_z"][0]))
        root_z[step] = float(data.qpos[2])
        root_tilt[step] = yaw_tilt(data.qpos[3:7])[1]

        scratch.qpos[:] = source_qpos
        scratch.qvel[:] = source_qvel
        mujoco.mj_forward(model, scratch)
        stance, speeds = selection["stance_side"], []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if floor not in (geom1, geom2) or geom_side.get(geom2 if geom1 == floor else geom1) != stance:
                continue
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, scratch, mujoco.mjtObj.mjOBJ_BODY,
                                     bodies[stance], velocity, 0)
            center = scratch.xipos[bodies[stance]]
            point_velocity = velocity[3:6] + np.cross(velocity[0:3], np.asarray(contact.pos) - center)
            speeds.append(float(np.linalg.norm(point_velocity[:2])))
        if speeds:
            stance_point_speed[step] = max(speeds)

    swing = selection["swing_side"]
    off_start, off_end = longest_false_window(contacts[swing][:TEACHER_STEPS])
    off_ticks = off_end - off_start
    max_clearance = float(np.max(clearance[swing][off_start:off_end])) if off_ticks else 0.0
    stance_core, _ = core_and_edge_masks(contacts[selection["stance_side"]])
    slip = quantiles(stance_point_speed[stance_core & np.isfinite(stance_point_speed)])
    slip_p95 = slip["p95"] if slip["p95"] is not None else float("inf")
    root_safe = bool(np.min(root_z) >= MIN_ROOT_Z_M and np.max(root_tilt) <= MAX_ROOT_TILT_RAD)
    terminal = bool(np.all(contacts[swing][-TERMINAL_CONTACT_TICKS:]))
    follow = slice(-FOLLOW_TICKS * SUBSTEPS, None)
    follow_safe = bool(np.all((contacts["left"] | contacts["right"])[follow])
                       and np.min(root_z[follow]) >= MIN_ROOT_Z_M
                       and np.max(root_tilt[follow]) <= MAX_ROOT_TILT_RAD)
    hard = {
        "swing_off_ge_100ms": off_ticks >= MIN_SWING_OFF_TICKS,
        "clearance_ge_12mm": max_clearance >= MIN_CLEARANCE_M,
        "stance_core_slip_p95_le_0p10": slip_p95 <= MAX_STANCE_CORE_SLIP_MPS,
        "root_tilt_safe": root_safe,
        "terminal_recontact_40ms": terminal,
        "followup_short_window_safe": follow_safe,
    }
    liftoff = NOMINAL_LIFTOFF_S + float(np.clip(raw[7], -BOUNDS[7], BOUNDS[7]))
    touchdown = NOMINAL_TOUCHDOWN_S + float(np.clip(raw[8], -BOUNDS[8], BOUNDS[8]))
    desired = slice(round(liftoff * 1000), round(touchdown * 1000))
    contact_fraction = float(np.mean(contacts[swing][:TEACHER_STEPS][desired]))
    cost = float(
        8.0 * max(0.0, (MIN_SWING_OFF_TICKS - off_ticks) / MIN_SWING_OFF_TICKS) ** 2
        + 3.0 * contact_fraction
        + 2.0 * max(0.0, (MIN_CLEARANCE_M - max_clearance) / MIN_CLEARANCE_M) ** 2
        + 2.0 * max(0.0, (slip_p95 - MAX_STANCE_CORE_SLIP_MPS) / MAX_STANCE_CORE_SLIP_MPS) ** 2
        + 4.0 * (not root_safe) + 3.0 * (not terminal) + 2.0 * (not follow_safe)
        + 0.02 * float(np.mean(np.square(np.asarray(raw) / BOUNDS)))
    )
    return {
        "hard_gates": hard, "strict_success": bool(all(hard.values())), "cost": cost,
        "longest_swing_off_ticks": int(off_ticks), "longest_swing_off_s": float(off_ticks / 1000.0),
        "swing_off_interval_ticks": [int(off_start), int(off_end)],
        "max_swing_clearance_m": max_clearance, "stance_core_slip_mps": slip,
        "root_z_min_m": float(np.min(root_z)), "root_tilt_max_rad": float(np.max(root_tilt)),
        "terminal_swing_contact_fraction_40ms": float(np.mean(contacts[swing][-TERMINAL_CONTACT_TICKS:])),
        "desired_swing_region_contact_fraction": contact_fraction,
        "position_residual_ctrl_absmax": float(residual_ctrl_absmax),
        "qpos_absmax_vs_recorded_suffix": float(recorded_qpos_absmax),
        "qvel_absmax_vs_recorded_suffix": float(recorded_qvel_absmax),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if mujoco.mj_versionString() != "3.3.7":
        raise RuntimeError(f"exact vendor MuJoCo 3.3.7 required, got {mujoco.mj_versionString()}")
    phase34 = json.loads(args.phase34_audit.read_text(encoding="utf-8"))
    assets = phase34["assets"]
    paths = {name: Path(assets[name]) for name in ("mmap", "rollout", "scene")}
    for name, path in paths.items():
        if sha256(path) != assets[f"{name}_sha256"]:
            raise RuntimeError(f"immutable {name} hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    physical = read_trace(paths["mmap"])
    rollout = json.loads(paths["rollout"].read_text(encoding="utf-8"))
    selection = select_window(model, physical, rollout)
    expected = {"telemetry_index": 114, "swing_side": "right", "stance_side": "left",
                "aligned_physics_index": 2310, "anchor_physics_index": 2309}
    if any(selection[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Phase36 frozen window drift: {selection}")
    state = seed_state(model, physical, selection["anchor_physics_index"])
    prefix = exact_prefix(model, physical, state, selection["anchor_physics_index"])
    zero = np.zeros(len(PARAMETER_NAMES), dtype=np.float64)
    baseline = run_candidate(model, physical, state, selection, zero)
    common = {
        "stage": "BASE Phase37", "scope": "one new three-event exact-state teacher hypothesis; no Phase36 budget extension/training/WBT/hardware/Git/cloud",
        "assets": {name: {"path": str(path), "sha256": assets[f"{name}_sha256"]} for name, path in paths.items()},
        "selection": selection, "snapshot_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "contract": {
            "teacher_seconds": TEACHER_SECONDS, "followup_seconds": FOLLOW_TICKS * 0.02,
            "events": {"prep_nominal_s": [0.0, NOMINAL_LIFTOFF_S],
                       "swing_nominal_s": [NOMINAL_LIFTOFF_S, NOMINAL_TOUCHDOWN_S],
                       "touchdown_nominal_s": [NOMINAL_TOUCHDOWN_S, TEACHER_SECONDS]},
            "parameter_names": list(PARAMETER_NAMES), "bounds": BOUNDS.tolist(),
            "seed_population_iterations_elites": [SEED, POPULATION, ITERATIONS, ELITES],
            "hard_gates": {"swing_off_ms": 100, "clearance_m": 0.012,
                           "stance_core_slip_p95_mps": 0.10, "root_z_m": 0.55,
                           "root_tilt_rad": 0.30, "terminal_recontact_ms": 40},
        },
        "preflight": {"exact_recorded_ctrl_prefix": prefix}, "baseline": baseline,
    }
    zero_full_exact = (baseline["qpos_absmax_vs_recorded_suffix"] <= 1e-12
                       and baseline["qvel_absmax_vs_recorded_suffix"] <= 1e-12)
    common["preflight"]["zero_candidate_full_440ms_exact"] = {
        "qpos_absmax": baseline["qpos_absmax_vs_recorded_suffix"],
        "qvel_absmax": baseline["qvel_absmax_vs_recorded_suffix"],
        "pass_1e12": bool(zero_full_exact),
    }
    if args.preflight_only or not prefix["exact_le_1e12"] or not zero_full_exact:
        return {**common, "cem_executed": False,
                "decision": ("PREFLIGHT_PASS_CEM_NOT_RUN" if prefix["exact_le_1e12"] and zero_full_exact
                             else "STOPPED_ZERO_CANDIDATE_NOT_EXACT")}

    rng = np.random.default_rng(SEED)
    mean, std = np.zeros_like(BOUNDS), 0.45 * BOUNDS
    best_raw, best_result = zero.copy(), baseline
    iterations = []
    evaluation_count = 0
    for iteration in range(ITERATIONS):
        population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), -BOUNDS, BOUNDS)
        population[0] = best_raw
        if iteration == 0:
            population[0] = zero
        scored = []
        for raw in population:
            result = run_candidate(model, physical, state, selection, raw)
            evaluation_count += 1
            scored.append((result["cost"], raw.copy(), result))
            if ((result["strict_success"] and not best_result["strict_success"])
                    or (result["strict_success"] == best_result["strict_success"]
                        and result["cost"] < best_result["cost"])):
                best_raw, best_result = raw.copy(), result
        scored.sort(key=lambda item: (not item[2]["strict_success"], item[0]))
        elites = scored[:ELITES]
        values = np.asarray([item[1] for item in elites])
        mean = 0.25 * mean + 0.75 * np.mean(values, axis=0)
        std = np.maximum(0.05 * BOUNDS, 0.25 * std + 0.75 * np.std(values, axis=0))
        iterations.append({"iteration": iteration, "best_cost": float(scored[0][0]),
                           "strict_success": bool(scored[0][2]["strict_success"])})
    best_result = run_candidate(model, physical, state, selection, best_raw)
    return {
        **common, "cem_executed": True,
        "search": {"evaluation_count": evaluation_count, "iterations": iterations},
        "best": {"parameters": dict(zip(PARAMETER_NAMES, best_raw.tolist())), "metrics": best_result},
        "decision": {"strict_teacher_found": bool(best_result["strict_success"]),
                     "teacher_route_supported": bool(best_result["strict_success"]),
                     "training_unlocked": False,
                     "result": "FOUND_THREE_EVENT_DYNAMIC_TEACHER" if best_result["strict_success"] else "THREE_EVENT_TEACHER_ROUTE_STOPPED"},
        "boundary": "single exact-state direct vendor-MJCF oracle, not closed ROS replay or hardware truth",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase34-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "preflight": result["preflight"],
                      "baseline": result["baseline"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
