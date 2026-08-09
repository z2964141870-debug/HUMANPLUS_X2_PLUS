#!/usr/bin/env python3
"""One-window exact-state CEM swing teacher from the frozen Phase34 trace."""

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
from official_x2.audit_phase35_contact_core import align_telemetry, core_and_edge_masks
from official_x2.audit_stage250_native_dynamic_seed import contact_geom_ids, foot_points, quantiles
from official_x2.replay_official_trace_direct_mujoco import JOINTS, pd_gains, yaw_tilt


CONTROL_TICKS = 12
SUBSTEPS = 20
FOLLOW_TICKS = 4
KNOTS = 3
MODE_NAMES = ("swing_hip_pitch", "swing_knee", "swing_ankle_pitch", "stance_hip_roll", "waist_roll")
MODE_BOUNDS_RAD = np.asarray([0.08, 0.12, 0.06, 0.05, 0.04], dtype=np.float64)
SEED = 3601
POPULATION = 24
ITERATIONS = 5
ELITES = 6
MIN_SWING_OFF_TICKS = 100
MIN_CLEARANCE_M = 0.012
MAX_STANCE_CORE_SLIP_MPS = 0.10
MIN_ROOT_Z_M = 0.55
MAX_ROOT_TILT_RAD = 0.30
TERMINAL_CONTACT_TICKS = 40


def integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, specification), dtype=np.float64)
    mujoco.mj_getState(model, data, state, specification)
    return state


def restore_integration_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, state, specification)
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(model, data, state, specification)


def longest_false_window(values: np.ndarray) -> tuple[int, int]:
    values = np.asarray(values, dtype=bool)
    best = (0, 0)
    start: int | None = None
    for index, value in enumerate(np.r_[values, True]):
        if not bool(value) and start is None:
            start = index
        elif bool(value) and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def coefficients(raw: np.ndarray, control_tick: int) -> np.ndarray:
    bounds = np.tile(MODE_BOUNDS_RAD, KNOTS)
    knots = np.clip(np.asarray(raw, dtype=np.float64), -bounds, bounds).reshape(KNOTS, len(MODE_NAMES))
    position = control_tick / (CONTROL_TICKS - 1) * (KNOTS - 1)
    left = min(KNOTS - 1, int(math.floor(position)))
    right = min(KNOTS - 1, left + 1)
    blend = smoothstep(position - left)
    return (1.0 - blend) * knots[left] + blend * knots[right]


def mode_joints(swing_side: str) -> tuple[str, ...]:
    stance = "right" if swing_side == "left" else "left"
    return (
        f"{swing_side}_hip_pitch_joint", f"{swing_side}_knee_joint",
        f"{swing_side}_ankle_pitch_joint", f"{stance}_hip_roll_joint", "waist_roll_joint",
    )


def geom_contract(model: mujoco.MjModel) -> tuple[int, dict[int, str], dict[str, int]]:
    floor = int(model.geom("floor").id)
    bodies = {side: int(model.body(f"{side}_ankle_roll_link").id) for side in ("left", "right")}
    geom_side = {}
    for geom_id in range(model.ngeom):
        for side, body_id in bodies.items():
            if int(model.geom_bodyid[geom_id]) == body_id:
                geom_side[geom_id] = side
    return floor, geom_side, bodies


def contact_sides(contact_rows: list[dict[str, Any]], floor: int, geom_side: dict[int, str]) -> dict[str, bool]:
    result = {"left": False, "right": False}
    for contact in contact_rows:
        geom1, geom2 = int(contact["geom1"]), int(contact["geom2"])
        if floor not in (geom1, geom2):
            continue
        side = geom_side.get(geom2 if geom1 == floor else geom1)
        if side:
            result[side] = True
    return result


def select_window(model: mujoco.MjModel, physical: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    aligned = align_telemetry(physical["qpos"], rollout["trace"])
    floor, geom_side, _bodies = geom_contract(model)
    for telemetry_index, (physics_index, row) in enumerate(zip(aligned, rollout["trace"])):
        if row["stage"] != "move" or len(row.get("obs", [])) != 93:
            continue
        realized = contact_sides(physical["contacts"][physics_index], floor, geom_side)
        for side, obs_index in (("left", 91), ("right", 92)):
            if row["obs"][obs_index] <= 0.5 and realized[side]:
                anchor = int(physics_index - 1)
                required = (CONTROL_TICKS + FOLLOW_TICKS) * SUBSTEPS
                if anchor >= 0 and anchor + required < len(physical["qpos"]):
                    return {
                        "telemetry_index": telemetry_index, "move_elapsed_s": float(row["elapsed_s"]),
                        "swing_side": side, "stance_side": "right" if side == "left" else "left",
                        "aligned_physics_index": int(physics_index), "anchor_physics_index": anchor,
                        "anchor_time_s": float(physical["time"][anchor]),
                    }
    raise RuntimeError("no generator-swing/realized-contact mismatch window with sufficient suffix")


def seed_state(model: mujoco.MjModel, physical: dict[str, Any], anchor: int) -> np.ndarray:
    data = mujoco.MjData(model)
    data.time = physical["time"][anchor]
    data.qpos[:] = physical["qpos"][anchor]
    data.qvel[:] = physical["qvel"][anchor]
    data.ctrl[:] = physical["ctrl"][anchor]
    data.qacc_warmstart[:] = physical["warmstart"][anchor]
    return integration_state(model, data)


def contact_state_from_data(model: mujoco.MjModel, data: mujoco.MjData,
                            floor: int, geom_side: dict[int, str]) -> dict[str, bool]:
    result = {"left": False, "right": False}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if floor not in (geom1, geom2):
            continue
        side = geom_side.get(geom2 if geom1 == floor else geom1)
        if side:
            result[side] = True
    return result


def run_candidate(model: mujoco.MjModel, physical: dict[str, Any], state: np.ndarray,
                  selection: dict[str, Any], raw: np.ndarray) -> dict[str, Any]:
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    restore_integration_state(model, data, state)
    floor, geom_side, bodies = geom_contract(model)
    sole_geoms = {side: contact_geom_ids(model, side) for side in bodies}
    actuator = {name: int(model.actuator(f"motor_{name}").id) for name in JOINTS}
    gains = pd_gains()
    joints = mode_joints(selection["swing_side"])
    anchor = selection["anchor_physics_index"]
    total = (CONTROL_TICKS + FOLLOW_TICKS) * SUBSTEPS
    contacts = {side: np.zeros(total, dtype=bool) for side in bodies}
    clearance = {side: np.zeros(total) for side in bodies}
    stance_point_speed = np.full(total, np.nan)
    root_z, root_tilt = np.zeros(total), np.zeros(total)
    saturation_count = 0
    for step in range(total):
        source_qpos, source_qvel = data.qpos.copy(), data.qvel.copy()
        baseline_ctrl = physical["ctrl"][anchor + 1 + step].copy()
        data.ctrl[:] = baseline_ctrl
        if step < CONTROL_TICKS * SUBSTEPS:
            control_tick = min(CONTROL_TICKS - 1, step // SUBSTEPS)
            residual = coefficients(raw, control_tick)
            for name, delta_rad in zip(joints, residual):
                actuator_id = actuator[name]
                proposed = baseline_ctrl[actuator_id] + gains[name][0] * float(delta_rad)
                low, high = model.actuator_ctrlrange[actuator_id]
                clipped = float(np.clip(proposed, low, high))
                saturation_count += int(abs(clipped - proposed) > 1e-12)
                data.ctrl[actuator_id] = clipped
        mujoco.mj_step(model, data)
        state_contact = contact_state_from_data(model, data, floor, geom_side)
        for side in bodies:
            contacts[side][step] = state_contact[side]
            clearance[side][step] = max(0.0, float(foot_points(model, data, sole_geoms[side])["min_z"][0]))
        root_z[step] = float(data.qpos[2])
        root_tilt[step] = yaw_tilt(data.qpos[3:7])[1]

        # The post-step contact workspace belongs to the pre-integration state.
        scratch.qpos[:] = source_qpos
        scratch.qvel[:] = source_qvel
        mujoco.mj_forward(model, scratch)
        speeds = []
        stance = selection["stance_side"]
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if floor not in (geom1, geom2):
                continue
            if geom_side.get(geom2 if geom1 == floor else geom1) != stance:
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
    teacher_contact = contacts[swing][: CONTROL_TICKS * SUBSTEPS]
    off_start, off_end = longest_false_window(teacher_contact)
    off_ticks = off_end - off_start
    max_clearance = float(np.max(clearance[swing][off_start:off_end])) if off_ticks else 0.0
    stance_core, _edge = core_and_edge_masks(contacts[selection["stance_side"]])
    slip_values = stance_point_speed[stance_core & np.isfinite(stance_point_speed)]
    slip = quantiles(slip_values)
    slip_p95 = slip["p95"] if slip["p95"] is not None else float("inf")
    terminal = bool(np.all(contacts[swing][-TERMINAL_CONTACT_TICKS:]))
    root_safe = bool(np.min(root_z) >= MIN_ROOT_Z_M and np.max(root_tilt) <= MAX_ROOT_TILT_RAD)
    subsequent_safe = bool(
        np.all((contacts["left"] | contacts["right"])[-FOLLOW_TICKS * SUBSTEPS:]) and
        np.min(root_z[-FOLLOW_TICKS * SUBSTEPS:]) >= MIN_ROOT_Z_M and
        np.max(root_tilt[-FOLLOW_TICKS * SUBSTEPS:]) <= MAX_ROOT_TILT_RAD
    )
    hard = {
        "swing_off_ge_100ms": off_ticks >= MIN_SWING_OFF_TICKS,
        "clearance_ge_12mm": max_clearance >= MIN_CLEARANCE_M,
        "stance_core_slip_p95_le_0p10": slip_p95 <= MAX_STANCE_CORE_SLIP_MPS,
        "root_tilt_safe": root_safe,
        "terminal_recontact_40ms": terminal,
        "followup_short_window_safe": subsequent_safe,
    }
    target_start, target_end = 40, 180
    target_contact_fraction = float(np.mean(teacher_contact[target_start:target_end]))
    residual_scale = np.tile(MODE_BOUNDS_RAD, KNOTS)
    cost = float(
        8.0 * max(0.0, (MIN_SWING_OFF_TICKS - off_ticks) / MIN_SWING_OFF_TICKS) ** 2
        + 3.0 * target_contact_fraction
        + 2.0 * max(0.0, (MIN_CLEARANCE_M - max_clearance) / MIN_CLEARANCE_M) ** 2
        + 2.0 * max(0.0, (slip_p95 - MAX_STANCE_CORE_SLIP_MPS) / MAX_STANCE_CORE_SLIP_MPS) ** 2
        + 4.0 * (not root_safe) + 3.0 * (not terminal) + 2.0 * (not subsequent_safe)
        + 0.02 * np.mean(np.square(np.asarray(raw) / residual_scale))
    )
    return {
        "hard_gates": hard, "strict_success": bool(all(hard.values())), "cost": cost,
        "longest_swing_off_ticks": int(off_ticks), "longest_swing_off_s": float(off_ticks / 1000.0),
        "swing_off_interval_ticks": [int(off_start), int(off_end)],
        "max_swing_clearance_m": max_clearance,
        "stance_core_slip_mps": slip,
        "root_z_min_m": float(np.min(root_z)), "root_tilt_max_rad": float(np.max(root_tilt)),
        "terminal_swing_contact_fraction_40ms": float(np.mean(contacts[swing][-TERMINAL_CONTACT_TICKS:])),
        "target_region_swing_contact_fraction": target_contact_fraction,
        "actuator_residual_saturation_count": int(saturation_count),
    }


def exact_prefix(model: mujoco.MjModel, physical: dict[str, Any], state: np.ndarray,
                 anchor: int, ticks: int = 10) -> dict[str, Any]:
    data = mujoco.MjData(model)
    restore_integration_state(model, data, state)
    qpos_error, qvel_error = [], []
    for tick in range(ticks):
        index = anchor + 1 + tick
        data.ctrl[:] = physical["ctrl"][index]
        mujoco.mj_step(model, data)
        qpos_error.append(float(np.max(np.abs(data.qpos - physical["qpos"][index]))))
        qvel_error.append(float(np.max(np.abs(data.qvel - physical["qvel"][index]))))
    return {"ticks": ticks, "qpos_absmax": max(qpos_error), "qvel_absmax": max(qvel_error),
            "exact_le_1e12": max(qpos_error + qvel_error) <= 1e-12}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if mujoco.mj_versionString() != "3.3.7":
        raise RuntimeError(f"exact vendor MuJoCo 3.3.7 required, got {mujoco.mj_versionString()}")
    phase34 = json.loads(args.phase34_audit.read_text(encoding="utf-8"))
    assets = phase34["assets"]
    paths = {name: Path(assets[name]) for name in ("mmap", "rollout", "scene")}
    for name, path in paths.items():
        expected = assets[f"{name}_sha256"]
        if sha256(path) != expected:
            raise RuntimeError(f"immutable {name} hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    physical = read_trace(paths["mmap"])
    rollout = json.loads(paths["rollout"].read_text(encoding="utf-8"))
    selection = select_window(model, physical, rollout)
    state = seed_state(model, physical, selection["anchor_physics_index"])
    prefix = exact_prefix(model, physical, state, selection["anchor_physics_index"])
    zero = np.zeros(KNOTS * len(MODE_NAMES), dtype=np.float64)
    baseline = run_candidate(model, physical, state, selection, zero)
    contract = {
        "control_ticks": CONTROL_TICKS, "control_dt_s": 0.020,
        "followup_ticks": FOLLOW_TICKS, "knots": KNOTS, "mode_names": list(MODE_NAMES),
        "mode_bounds_rad": MODE_BOUNDS_RAD.tolist(),
        "seed_population_iterations_elites": [SEED, POPULATION, ITERATIONS, ELITES],
        "hard_gates": {
            "minimum_swing_off_ms": MIN_SWING_OFF_TICKS,
            "minimum_clearance_m": MIN_CLEARANCE_M,
            "maximum_stance_core_slip_p95_mps": MAX_STANCE_CORE_SLIP_MPS,
            "minimum_root_z_m": MIN_ROOT_Z_M, "maximum_root_tilt_rad": MAX_ROOT_TILT_RAD,
            "terminal_recontact_ms": TERMINAL_CONTACT_TICKS,
        },
        "future_input": "recorded closed Phase34 ctrl at each exact 1ms step plus bounded position-equivalent residual",
    }
    common = {
        "stage": "BASE Phase36", "scope": "one exact-state direct-official-MJCF CEM window; no training/WBT/hardware/Git/cloud",
        "assets": {name: {"path": str(path), "sha256": assets[f"{name}_sha256"]} for name, path in paths.items()},
        "selection": selection, "snapshot_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "contract": contract, "preflight": {"exact_recorded_ctrl_prefix": prefix}, "baseline": baseline,
    }
    if args.preflight_only or not prefix["exact_le_1e12"]:
        return {**common, "cem_executed": False,
                "decision": "PREFLIGHT_PASS_CEM_NOT_RUN" if prefix["exact_le_1e12"] else "STOPPED_EXACT_FORK_FAILED"}

    rng = np.random.default_rng(SEED)
    bounds = np.tile(MODE_BOUNDS_RAD, KNOTS)
    mean = np.zeros_like(bounds)
    std = 0.45 * bounds
    best_raw, best_result = zero.copy(), baseline
    iterations = []
    evaluation_count = 0
    for iteration in range(ITERATIONS):
        population = np.clip(rng.normal(mean, std, size=(POPULATION, len(mean))), -bounds, bounds)
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
        std = np.maximum(0.05 * bounds, 0.25 * std + 0.75 * np.std(values, axis=0))
        iterations.append({"iteration": iteration, "best_cost": float(scored[0][0]),
                           "strict_success": bool(scored[0][2]["strict_success"])})
    best_result = run_candidate(model, physical, state, selection, best_raw)
    return {
        **common, "cem_executed": True,
        "search": {"evaluation_count": evaluation_count, "iterations": iterations},
        "best": {"raw_knots": best_raw.reshape(KNOTS, len(MODE_NAMES)).tolist(), "metrics": best_result},
        "decision": {
            "strict_teacher_found": bool(best_result["strict_success"]),
            "representation_supported": bool(best_result["strict_success"]),
            "training_unlocked": False,
            "result": "FOUND_ONE_WINDOW_DYNAMIC_TEACHER" if best_result["strict_success"] else "NO_FEASIBLE_TEACHER_IN_FIXED_REPRESENTATION",
        },
        "boundary": "direct exact-state vendor-MJCF oracle for one fixed window; not closed ROS replay or hardware truth",
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
