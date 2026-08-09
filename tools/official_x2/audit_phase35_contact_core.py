#!/usr/bin/env python3
"""Offline contact-core audit of the immutable Phase34 closed trace.

This stage deliberately does not run physics.  It separates impact/contact
switch transients from the load-bearing interior of each realized contact
window and preserves the Phase34 mmap byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from official_x2.analyze_phase34_full_closed_trace import read_trace, root_score, sha256
from official_x2.audit_stage250_native_dynamic_seed import (
    contact_geom_ids,
    contiguous_cycles,
    foot_points,
    quantiles,
)


PHYSICS_HZ = 1000
# Existing temporal contact contract: about 66.7 ms confirmation and 100 ms
# minimum dwell.  Conversion to the 1 kHz closed trace is fixed, not fitted.
CONFIRM_TICKS = 67
MIN_DWELL_TICKS = 100
EDGE_TRIM_TICKS = 50
SLIP_GATE_MPS = 0.10


def intervals(values: np.ndarray) -> list[tuple[int, int, bool]]:
    values = np.asarray(values, dtype=bool)
    if not len(values):
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return [(int(start), int(end), bool(values[start])) for start, end in zip(starts, ends)]


def hysteresis(values: np.ndarray, confirm_ticks: int = CONFIRM_TICKS) -> np.ndarray:
    """Confirm an edge for a fixed duration and backdate it to its first tick."""
    raw = np.asarray(values, dtype=bool)
    if not len(raw):
        return raw.copy()
    if confirm_ticks < 1:
        raise ValueError("confirm_ticks must be positive")
    result = np.full(len(raw), raw[0], dtype=bool)
    state = bool(raw[0])
    pending: int | None = None
    for tick in range(1, len(raw)):
        if bool(raw[tick]) == state:
            pending = None
            result[tick] = state
            continue
        if pending is None:
            pending = tick
        if tick - pending + 1 >= confirm_ticks:
            state = bool(raw[tick])
            result[pending : tick + 1] = state
            pending = None
        else:
            result[tick] = state
    return result


def enforce_min_dwell(values: np.ndarray, minimum: int = MIN_DWELL_TICKS) -> np.ndarray:
    """Remove Boolean islands shorter than the fixed minimum dwell."""
    if minimum < 1:
        raise ValueError("minimum must be positive")
    result = np.asarray(values, dtype=bool).copy()
    for _ in range(len(result) + 1):
        runs = intervals(result)
        short = [(a, b, state) for a, b, state in runs if b - a < minimum]
        if not short or len(runs) == 1:
            return result
        changed = False
        for index, (start, end, state) in enumerate(runs):
            if end - start >= minimum:
                continue
            if index == 0:
                replacement = runs[index + 1][2]
            elif index == len(runs) - 1:
                replacement = runs[index - 1][2]
            else:
                if runs[index - 1][2] != runs[index + 1][2]:
                    raise AssertionError("Boolean island neighbours differ")
                replacement = runs[index - 1][2]
            result[start:end] = replacement
            changed = changed or replacement != state
        if not changed:
            return result
    raise RuntimeError("minimum-dwell cleanup did not converge")


def canonical_contact(values: np.ndarray) -> np.ndarray:
    return enforce_min_dwell(hysteresis(np.asarray(values, dtype=bool)))


def core_and_edge_masks(values: np.ndarray, trim_ticks: int = EDGE_TRIM_TICKS) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=bool)
    core = np.zeros(len(values), dtype=bool)
    edge = np.zeros(len(values), dtype=bool)
    for start, end, state in intervals(values):
        if not state:
            continue
        left, right = min(end, start + trim_ticks), max(start, end - trim_ticks)
        edge[start:left] = True
        edge[right:end] = True
        if right > left:
            core[left:right] = True
    return core, edge


def weighted_quantiles(values: Iterable[float], weights: Iterable[float]) -> dict[str, float | int | None]:
    values = np.asarray(list(values), dtype=np.float64)
    weights = np.asarray(list(weights), dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        return {"count": 0, "weight_sum": 0.0, "mean": None, "p50": None, "p95": None, "max": None}
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    def percentile(probability: float) -> float:
        return float(values[min(int(np.searchsorted(cdf, probability, side="left")), len(values) - 1)])
    return {
        "count": int(len(values)), "weight_sum": float(np.sum(weights)),
        "mean": float(np.sum(values * weights) / np.sum(weights)),
        "p50": percentile(0.50), "p95": percentile(0.95), "max": float(np.max(values)),
    }


def align_telemetry(qpos: np.ndarray, telemetry: list[dict[str, Any]]) -> np.ndarray:
    first = min(range(min(100, len(qpos))), key=lambda index: root_score(qpos[index], telemetry[0]))
    aligned = [first]
    for row in telemetry[1:]:
        lower, upper = aligned[-1] + 1, min(len(qpos), aligned[-1] + 101)
        aligned.append(min(range(lower, upper), key=lambda index: root_score(qpos[index], row)))
    return np.asarray(aligned, dtype=int)


def analyze(model: mujoco.MjModel, physical: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    qpos, qvel = physical["qpos"], physical["qvel"]
    telemetry = rollout["trace"]
    aligned = align_telemetry(qpos, telemetry)
    move_rows = np.asarray([row["stage"] == "move" for row in telemetry])
    move_aligned = aligned[move_rows]
    start, end = int(move_aligned[0]), int(move_aligned[-1] + 1)
    length = end - start

    floor = int(model.geom("floor").id)
    body_ids = {side: int(model.body(f"{side}_ankle_roll_link").id) for side in ("left", "right")}
    geom_side: dict[int, str] = {}
    for geom_id in range(model.ngeom):
        for side, body_id in body_ids.items():
            if int(model.geom_bodyid[geom_id]) == body_id:
                geom_side[geom_id] = side

    raw = {side: np.zeros(length, dtype=bool) for side in body_ids}
    max_point_speed = {side: np.full(length, np.nan) for side in body_ids}
    weighted_tick_speed = {side: np.full(length, np.nan) for side in body_ids}
    body_speed = {side: np.zeros(length) for side in body_ids}
    sole_xy = {side: np.zeros((length, 2)) for side in body_ids}
    total_normal = {side: np.zeros(length) for side in body_ids}
    point_speeds: dict[str, list[list[float]]] = {side: [[] for _ in range(length)] for side in body_ids}
    point_weights: dict[str, list[list[float]]] = {side: [[] for _ in range(length)] for side in body_ids}

    data = mujoco.MjData(model)
    sole_geom_ids = {side: contact_geom_ids(model, side) for side in body_ids}
    for local, physical_index in enumerate(range(start, end)):
        source = max(0, physical_index - 1)
        data.qpos[:] = qpos[source]
        data.qvel[:] = qvel[source]
        mujoco.mj_forward(model, data)
        for side, body_id in body_ids.items():
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            body_speed[side][local] = float(np.linalg.norm(velocity[3:5]))
            sole_xy[side][local] = foot_points(model, data, sole_geom_ids[side])["center"][:2]
        for contact in physical["contacts"][physical_index]:
            geom1, geom2 = contact["geom1"], contact["geom2"]
            if floor not in (geom1, geom2):
                continue
            other = geom2 if geom1 == floor else geom1
            side = geom_side.get(other)
            if side is None:
                continue
            raw[side][local] = True
            force = max(0.0, float(contact["wrench"][0]))
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     body_ids[side], velocity, 0)
            center = data.xipos[body_ids[side]]
            point_velocity = velocity[3:6] + np.cross(velocity[0:3], contact["position"] - center)
            speed = float(np.linalg.norm(point_velocity[:2]))
            point_speeds[side][local].append(speed)
            point_weights[side][local].append(force)
            total_normal[side][local] += force

    filtered = {side: canonical_contact(raw[side]) for side in body_ids}
    generator = {
        "left": np.asarray([row["obs"][91] > .5 for row in np.asarray(telemetry, dtype=object)[move_rows]]),
        "right": np.asarray([row["obs"][92] > .5 for row in np.asarray(telemetry, dtype=object)[move_rows]]),
    }
    realized_50 = {
        side: filtered[side][move_aligned - start] for side in body_ids
    }
    result: dict[str, Any] = {}
    core_masks: dict[str, np.ndarray] = {}
    for side in body_ids:
        for tick in range(length):
            speeds, forces = point_speeds[side][tick], point_weights[side][tick]
            if speeds:
                max_point_speed[side][tick] = max(speeds)
                positive = np.asarray(forces) > 0.0
                if np.any(positive):
                    weighted_tick_speed[side][tick] = float(
                        np.average(np.asarray(speeds)[positive], weights=np.asarray(forces)[positive]))
        # A load-bearing core belongs to an actually continuous collision
        # window.  The debounced schedule is intentionally used only for the
        # phase-cycle audit: applying it here would relabel short flight gaps
        # as stance and count swing-foot velocity as slip.
        core, edge = core_and_edge_masks(raw[side])
        core_masks[side] = core
        impulse_speeds, impulse_weights = [], []
        for tick in np.flatnonzero(core):
            impulse_speeds.extend(point_speeds[side][tick])
            impulse_weights.extend([force / PHYSICS_HZ for force in point_weights[side][tick]])
        excursions = []
        for window_start, window_end, state in intervals(raw[side]):
            if not state:
                continue
            a, b = window_start + EDGE_TRIM_TICKS, window_end - EDGE_TRIM_TICKS
            if b <= a:
                continue
            xy = sole_xy[side][a:b]
            excursions.append({
                "duration_s": float((b - a) / PHYSICS_HZ),
                "net_m": float(np.linalg.norm(xy[-1] - xy[0])),
                "max_from_start_m": float(np.max(np.linalg.norm(xy - xy[0], axis=1))),
            })
        raw_p95 = quantiles(max_point_speed[side][raw[side]])["p95"]
        core_point = quantiles(max_point_speed[side][core])
        core_weighted_tick = quantiles(weighted_tick_speed[side][core])
        core_body = quantiles(body_speed[side][core])
        impulse = weighted_quantiles(impulse_speeds, impulse_weights)
        core_clean = bool(
            core_point["count"] and core_weighted_tick["p95"] <= SLIP_GATE_MPS
            and core_body["p95"] <= SLIP_GATE_MPS
        )
        result[side] = {
            "raw_contact_windows": int(sum(state for _, _, state in intervals(raw[side]))),
            "debounced_contact_windows": int(sum(state for _, _, state in intervals(filtered[side]))),
            "raw_contact_fraction": float(np.mean(raw[side])),
            "debounced_contact_fraction": float(np.mean(filtered[side])),
            "core_ticks": int(np.sum(core)), "edge_ticks": int(np.sum(edge)),
            "raw_max_contact_point_speed_mps": quantiles(max_point_speed[side][raw[side]]),
            "edge_max_contact_point_speed_mps": quantiles(max_point_speed[side][edge & raw[side]]),
            "core_max_contact_point_speed_mps": core_point,
            "core_force_weighted_point_speed_per_tick_mps": core_weighted_tick,
            "core_impulse_weighted_contact_point_speed_mps": impulse,
            "core_sole_body_linear_speed_mps": core_body,
            "core_normal_force_n": quantiles(total_normal[side][core]),
            "core_stance_excursion": {
                "window_count": len(excursions),
                "net_m": quantiles(value["net_m"] for value in excursions),
                "max_from_start_m": quantiles(value["max_from_start_m"] for value in excursions),
                "duration_s": quantiles(value["duration_s"] for value in excursions),
            },
            "core_slip_gate_p95_le_0p10": core_clean,
            "raw_failure_inflated_by_transient_or_point_switch": bool(
                raw_p95 is not None and raw_p95 > SLIP_GATE_MPS and core_clean),
        }

    realized_cycles = contiguous_cycles(realized_50["left"], realized_50["right"])
    generator_cycles = contiguous_cycles(generator["left"], generator["right"])
    cycle_exact = realized_cycles == generator_cycles
    both_core_clean = all(result[side]["core_slip_gate_p95_le_0p10"] for side in body_ids)
    trainable_50hz = bool(rollout["summary"].get("full_gate_pass") and cycle_exact and both_core_clean)
    return {
        "fixed_contract": {
            "physics_hz": PHYSICS_HZ, "confirm_ticks": CONFIRM_TICKS,
            "minimum_dwell_ticks": MIN_DWELL_TICKS, "edge_trim_ticks": EDGE_TRIM_TICKS,
            "slip_gate_mps": SLIP_GATE_MPS,
        },
        "coverage": {"move_physics_ticks": length, "move_telemetry_rows": int(np.sum(move_rows))},
        "contact_schedule": {
            "realized_ds_ss_ds_cycles_after_debounce_50hz": int(realized_cycles),
            "generator_ds_ss_ds_cycles_50hz": int(generator_cycles),
            "exact_cycle_count_consistency": bool(cycle_exact),
            "realized_generator_agreement": {
                side: float(np.mean(realized_50[side] == generator[side])) for side in body_ids
            },
        },
        "feet": result,
        "qualification": {
            "closed_full_gate": bool(rollout["summary"].get("full_gate_pass")),
            "both_feet_load_bearing_core_slip_gate": bool(both_core_clean),
            "debounced_cycle_count_matches_generator": bool(cycle_exact),
            "trainable_50hz_contact_seed": trainable_50hz,
            "hardware_contact_truth": False,
            "boundary": "official closed-MuJoCo model contact only; no hardware GRF/COP/wrench",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase34-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase34 = json.loads(args.phase34_audit.read_text(encoding="utf-8"))
    assets = phase34["assets"]
    mmap_path, rollout_path, scene_path = map(Path, (assets["mmap"], assets["rollout"], assets["scene"]))
    if sha256(mmap_path) != assets["mmap_sha256"] or sha256(rollout_path) != assets["rollout_sha256"]:
        raise RuntimeError("immutable Phase34 source hash mismatch")
    physical = read_trace(mmap_path)
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    analysis = analyze(model, physical, rollout)
    payload = {
        "stage": "BASE Phase35", "scope": "offline-only Phase34 contact-core audit; 0 physics/0 training",
        "source": {"phase34_audit": str(args.phase34_audit), "phase34_audit_sha256": sha256(args.phase34_audit),
                   "mmap": str(mmap_path), "mmap_sha256": assets["mmap_sha256"],
                   "rollout": str(rollout_path), "rollout_sha256": assets["rollout_sha256"],
                   "scene": str(scene_path), "scene_sha256": assets["scene_sha256"]},
        "analysis": analysis,
    }
    payload["decision"] = ("QUALIFIED_TRAINABLE_50HZ_CONTACT_SEED" if
                           analysis["qualification"]["trainable_50hz_contact_seed"] else
                           "NOT_QUALIFIED_TRAINABLE_50HZ_CONTACT_SEED")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], **analysis}, indent=2))


if __name__ == "__main__":
    main()
