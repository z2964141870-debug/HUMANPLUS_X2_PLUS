#!/usr/bin/env python3
"""Stream and qualify the one-shot Phase34 full closed AimDK trace."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import mmap
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from official_x2.audit_stage250_native_dynamic_seed import (
    contact_geom_ids,
    contiguous_cycles,
    foot_points,
    quantiles,
)
from official_x2.decode_phase32_mujoco_trace import FileHeader, StateRecord
from official_x2.replay_official_trace_direct_mujoco import (
    ARM_JOINTS,
    HEAD_JOINTS,
    JOINTS,
    LOWER_SCALE,
    LOWER_JOINTS,
    default_pose,
    pd_gains,
    yaw_tilt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_score(qpos: np.ndarray, telemetry: dict[str, Any]) -> float:
    yaw, tilt = yaw_tilt(qpos[3:7])
    yaw_error = math.atan2(math.sin(yaw - telemetry["root_yaw_rad"]),
                           math.cos(yaw - telemetry["root_yaw_rad"]))
    xyz = np.asarray([telemetry["root_x_m"], telemetry["root_y_m"], telemetry["root_z_m"]])
    return float(np.linalg.norm(qpos[:3] - xyz) + abs(yaw_error) + abs(tilt - telemetry["root_tilt_rad"]))


def read_trace(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        header = FileHeader.from_buffer_copy(mapped[:ctypes.sizeof(FileHeader)])
        if bytes(header.magic).split(b"\0", 1)[0] != b"X2MJSHIMV1":
            raise RuntimeError("trace magic mismatch")
        if header.header_size != ctypes.sizeof(FileHeader) or header.record_size != ctypes.sizeof(StateRecord):
            raise RuntimeError("trace ABI mismatch")
        if mapped.size() != header.header_size + header.max_records * header.record_size:
            raise RuntimeError("trace preallocation size mismatch")
        committed = int(header.committed_records)
        call_counts = {1: 0, 2: 0, 3: 0}
        last_reset = -1
        sequences = []
        data_pointers: dict[int, int] = {}
        for index in range(committed):
            offset = header.header_size + index * header.record_size
            row = StateRecord.from_buffer_copy(mapped, offset)
            if row.committed != 1:
                raise RuntimeError(f"uncommitted row within committed range: {index}")
            sequences.append(int(row.sequence))
            call_counts[int(row.call_kind)] = call_counts.get(int(row.call_kind), 0) + 1
            data_pointers[int(row.data_pointer)] = data_pointers.get(int(row.data_pointer), 0) + 1
            if row.call_kind == 1:
                last_reset = int(row.sequence)

        qpos, qvel, qacc, warmstart, ctrl = [], [], [], [], []
        qfrc_actuator, qfrc_constraint, times, sequences_step = [], [], [], []
        contacts: list[list[dict[str, Any]]] = []
        for index in range(committed):
            offset = header.header_size + index * header.record_size
            row = StateRecord.from_buffer_copy(mapped, offset)
            if row.call_kind != 3 or row.sequence <= last_reset:
                continue
            qpos.append(np.asarray(row.qpos[:row.nq], dtype=np.float64))
            qvel.append(np.asarray(row.qvel[:row.nv], dtype=np.float64))
            qacc.append(np.asarray(row.qacc[:row.nv], dtype=np.float64))
            warmstart.append(np.asarray(row.qacc_warmstart[:row.nv], dtype=np.float64))
            ctrl.append(np.asarray(row.ctrl[:row.nu], dtype=np.float64))
            qfrc_actuator.append(np.asarray(row.qfrc_actuator[:row.nv], dtype=np.float64))
            qfrc_constraint.append(np.asarray(row.qfrc_constraint[:row.nv], dtype=np.float64))
            times.append(float(row.time_s))
            sequences_step.append(int(row.sequence))
            contact_rows = []
            for contact_index in range(row.contacts_stored):
                contact = row.contacts[contact_index]
                contact_rows.append({
                    "geom1": int(contact.geom1), "geom2": int(contact.geom2),
                    "position": np.asarray(contact.position, dtype=np.float64),
                    "wrench": np.asarray(contact.wrench, dtype=np.float64),
                })
            contacts.append(contact_rows)
        mapped.close()
    return {
        "header": {
            "format_version": int(header.format_version), "header_size": int(header.header_size),
            "record_size": int(header.record_size), "max_records": int(header.max_records),
            "committed_records": committed, "dropped_records": int(header.dropped_records),
            "max_time_s": float(header.max_time_s),
            "mujoco_version": bytes(header.mujoco_version).split(b"\0", 1)[0].decode(),
            "call_counts": {str(key): value for key, value in call_counts.items()},
            "data_pointer_counts": {str(key): value for key, value in data_pointers.items()},
            "sequence_contiguous": sequences == list(range(committed)),
            "last_reset_sequence": last_reset,
        },
        "time": np.asarray(times), "sequence": np.asarray(sequences_step),
        "qpos": np.asarray(qpos), "qvel": np.asarray(qvel), "qacc": np.asarray(qacc),
        "warmstart": np.asarray(warmstart), "ctrl": np.asarray(ctrl),
        "qfrc_actuator": np.asarray(qfrc_actuator),
        "qfrc_constraint": np.asarray(qfrc_constraint), "contacts": contacts,
    }


def target_for_row(row: dict[str, Any]) -> dict[str, float]:
    target = default_pose()
    physical = row.get("physical_lower_target_rad", [])
    action = row.get("action", [])
    if len(physical) == len(LOWER_JOINTS):
        for name, value in zip(LOWER_JOINTS, physical):
            target[name] = float(value)
    elif len(action) == len(LOWER_JOINTS):
        for index, name in enumerate(LOWER_JOINTS):
            target[name] += float(action[index] * LOWER_SCALE[index])
    else:
        raise ValueError("row does not contain a reconstructable lower target")
    for name, value in zip(ARM_JOINTS, row["upper_target_rad"]):
        target[name] = float(value)
    for name in HEAD_JOINTS:
        target[name] = 0.0
    return target


def analyze(model: mujoco.MjModel, physical: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    qpos, qvel = physical["qpos"], physical["qvel"]
    telemetry = rollout["trace"]
    if qpos.shape[0] < 14200 or len(telemetry) != 710:
        raise RuntimeError("full episode coverage missing")

    # Align row0 by root state, then monotonically match every callback within
    # the next 100 physics steps.  The callback clock is nominally 50 Hz, but
    # ROS scheduling creates real 15--24 ms intervals in the full episode.
    first_search = range(min(100, len(qpos)))
    base_index = min(first_search, key=lambda index: root_score(qpos[index], telemetry[0]))
    aligned_list = [base_index]
    for row in telemetry[1:]:
        lower = aligned_list[-1] + 1
        upper = min(len(qpos), lower + 100)
        if lower >= upper:
            raise RuntimeError("telemetry extends beyond physical trace")
        aligned_list.append(min(range(lower, upper), key=lambda index: root_score(qpos[index], row)))
    aligned = np.asarray(aligned_list, dtype=int)
    alignment_scores = np.asarray([root_score(qpos[index], row) for index, row in zip(aligned, telemetry)])
    time_offsets = physical["time"][aligned] - np.asarray([
        (0.0 if row["stage"] == "prepare" else 0.2 if row["stage"] == "stand" else
         2.2 if row["stage"] == "move" else 6.2) + row["elapsed_s"]
        for row in telemetry
    ])

    floor = int(model.geom("floor").id)
    body_ids = {side: int(model.body(f"{side}_ankle_roll_link").id) for side in ("left", "right")}
    geom_side = {}
    for gid in range(model.ngeom):
        for side, body_id in body_ids.items():
            if int(model.geom_bodyid[gid]) == body_id:
                geom_side[gid] = side
    contact_bool = {side: np.zeros(len(qpos), dtype=bool) for side in ("left", "right")}
    normal_force = {side: np.zeros(len(qpos)) for side in ("left", "right")}
    contact_positions: list[dict[str, list[np.ndarray]]] = []
    for index, contact_rows in enumerate(physical["contacts"]):
        positions = {"left": [], "right": []}
        for contact in contact_rows:
            g1, g2 = contact["geom1"], contact["geom2"]
            if floor not in (g1, g2):
                continue
            other = g2 if g1 == floor else g1
            side = geom_side.get(other)
            if side is None:
                continue
            contact_bool[side][index] = True
            normal_force[side][index] += max(0.0, float(contact["wrench"][0]))
            positions[side].append(contact["position"])
        contact_positions.append(positions)

    data = mujoco.MjData(model)
    sole_ids = {side: contact_geom_ids(model, side) for side in ("left", "right")}
    sole_min_z = {side: np.zeros(len(qpos)) for side in ("left", "right")}
    slip = {side: np.full(len(qpos), np.nan) for side in ("left", "right")}
    for index in range(len(qpos)):
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            sole_min_z[side][index] = float(foot_points(model, data, sole_ids[side])["min_z"][0])
        for side in ("left", "right"):
            if contact_positions[index][side]:
                # mj_step's force/contact workspace belongs to the state used
                # for forward dynamics, immediately before the integration.
                source = max(0, index - 1)
                data.qpos[:] = qpos[source]
                data.qvel[:] = qvel[source]
                mujoco.mj_forward(model, data)
                velocity = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                         body_ids[side], velocity, 0)
                center = data.xipos[body_ids[side]]
                point_speeds = []
                for point in contact_positions[index][side]:
                    point_velocity = velocity[3:6] + np.cross(velocity[0:3], point - center)
                    point_speeds.append(float(np.linalg.norm(point_velocity[:2])))
                slip[side][index] = max(point_speeds)

    stage = np.asarray([row["stage"] for row in telemetry])
    move_rows = stage == "move"
    move_indices = aligned[move_rows]
    realized_50 = {side: contact_bool[side][aligned] for side in ("left", "right")}
    generator = {
        "left": np.asarray([len(row.get("obs", [])) == 93 and row["obs"][91] > .5 for row in telemetry]),
        "right": np.asarray([len(row.get("obs", [])) == 93 and row["obs"][92] > .5 for row in telemetry]),
    }
    contact_result = {}
    for side in ("left", "right"):
        stance = realized_50[side][move_rows]
        move_slip = slip[side][move_indices]
        move_clearance = np.maximum(sole_min_z[side][move_indices], 0.0)
        contact_result[side] = {
            "realized_contact_fraction_move_50hz": float(np.mean(stance)),
            "generator_agreement_move_50hz": float(np.mean(stance == generator[side][move_rows])),
            "realized_stance_contact_point_slip_mps": quantiles(move_slip[stance]),
            "realized_swing_sole_clearance_m": quantiles(move_clearance[~stance]),
            "normal_force_n_at_realized_contact": quantiles(normal_force[side][move_indices][stance]),
            "physics_1khz_contact_fraction_move": float(np.mean(contact_bool[side][move_indices[0]:move_indices[-1] + 20])),
        }
    realized_cycles = contiguous_cycles(realized_50["left"][move_rows], realized_50["right"][move_rows])
    generator_cycles = contiguous_cycles(generator["left"][move_rows], generator["right"][move_rows])

    # Exact recorded acceleration and generalized-force closure at aligned 50Hz rows.
    residual_relative, acceleration_error = [], []
    for index in aligned:
        if index <= 0:
            continue
        # Post-step qacc/forces were evaluated at the immediately preceding
        # integration state; evaluating RNE at post-step qpos is a time-index
        # error and creates a false residual.
        data.qpos[:] = qpos[index - 1]
        data.qvel[:] = qvel[index - 1]
        data.ctrl[:] = physical["ctrl"][index]
        mujoco.mj_forward(model, data)
        passive = data.qfrc_passive.copy()
        data.qacc[:] = physical["qacc"][index]
        inverse = np.zeros(model.nv, dtype=np.float64)
        mujoco.mj_rne(model, data, 1, inverse)
        expected = passive + physical["qfrc_actuator"][index] + physical["qfrc_constraint"][index]
        residual_relative.append(float(np.linalg.norm(inverse - expected) / max(np.linalg.norm(inverse), 1.0)))
        if index > 0:
            fd_acc = (qvel[index] - qvel[index - 1]) / float(model.opt.timestep)
            acceleration_error.append(float(np.max(np.abs(fd_acc - physical["qacc"][index]))))

    # Offline action-publication-to-realized-ctrl lag curve.  This is diagnostic,
    # reports every lag 0..20 and does not tune the physical contract.
    qadr = {name: int(model.joint(name).qposadr[0]) for name in JOINTS}
    dadr = {name: int(model.joint(name).dofadr[0]) for name in JOINTS}
    actuator = {name: int(model.actuator(f"motor_{name}").id) for name in JOINTS}
    gains = pd_gains()
    lag_curve = []
    for lag in range(21):
        errors = []
        for telemetry_index, row in enumerate(telemetry):
            if len(row.get("physical_lower_target_rad", [])) != len(LOWER_JOINTS) and len(row.get("action", [])) != len(LOWER_JOINTS):
                continue
            step = int(aligned[telemetry_index] + lag)
            if step <= 0 or step >= len(qpos):
                continue
            target = target_for_row(row)
            predicted = np.zeros(model.nu)
            for name in JOINTS:
                kp, kd = gains[name]
                aid = actuator[name]
                torque = kp * (target[name] - qpos[step - 1, qadr[name]]) - kd * qvel[step - 1, dadr[name]]
                low, high = model.actuator_ctrlrange[aid]
                predicted[aid] = np.clip(torque, low, high)
            errors.append(float(np.max(np.abs(predicted - physical["ctrl"][step]))))
        values = np.asarray(errors)
        lag_curve.append({"lag_physics_ticks": lag, "lag_ms": lag,
                          "ctrl_absmax_error_p50": float(np.quantile(values, .5)),
                          "ctrl_absmax_error_p95": float(np.quantile(values, .95)),
                          "fraction_within_1e5": float(np.mean(values <= 1e-5))})

    best_lag = min(lag_curve, key=lambda row: row["ctrl_absmax_error_p50"])
    signed_pitch = np.asarray([row["root_pitch_rad"] for row in telemetry])
    full_gate = bool(rollout["summary"].get("full_gate_pass"))
    dynamics_closure = float(np.quantile(residual_relative, .95)) <= 1e-8
    acceleration_consistency = float(np.quantile(acceleration_error, .95)) <= 1e-8
    has_cycle = realized_cycles >= 1
    low_slip = all(contact_result[side]["realized_stance_contact_point_slip_mps"]["p95"] is not None and
                   contact_result[side]["realized_stance_contact_point_slip_mps"]["p95"] <= .20
                   for side in ("left", "right"))
    alignment_pass = float(np.max(alignment_scores)) <= 1e-8
    native_dynamic = full_gate and dynamics_closure and acceleration_consistency and has_cycle and alignment_pass
    return {
        "coverage": {
            "physics_steps_after_last_reset": len(qpos),
            "physics_time_first_s": float(physical["time"][0]),
            "physics_time_last_s": float(physical["time"][-1]),
            "telemetry_rows": len(telemetry),
        },
        "telemetry_alignment": {
            "base_physics_step_index": int(base_index),
            "physics_time_minus_stage_absolute_time_s": quantiles(time_offsets),
            "root_alignment_score": quantiles(alignment_scores),
            "max_score": float(np.max(alignment_scores)),
            "pass_1e8": alignment_pass,
        },
        "action_to_ctrl_alignment": {"curve_0_to_20ms": lag_curve, "best_median_lag": best_lag,
                                     "interpretation": "diagnostic realized-PD alignment; not a gate or tuned delay"},
        "contact": {
            "source": "closed official MuJoCo collision pairs and mj_contactForce at 1kHz; model truth, not hardware force",
            "realized_ds_ss_ds_cycles_move_50hz": realized_cycles,
            "generator_ds_ss_ds_cycles_move_50hz": generator_cycles,
            "left": contact_result["left"], "right": contact_result["right"],
            "low_slip_gate_p95_le_0p20": low_slip,
        },
        "dynamics": {
            "qacc_semimplicit_fd_absmax": quantiles(np.asarray(acceleration_error)),
            "generalized_equation_relative_residual": quantiles(np.asarray(residual_relative)),
            "acceleration_consistency_gate": acceleration_consistency,
            "dynamics_closure_gate": dynamics_closure,
            "source": "recorded qacc/qfrc_actuator/qfrc_constraint plus official-model passive and inverse dynamics",
        },
        "posture": {"move_signed_root_pitch_rad": quantiles(signed_pitch[move_rows])},
        "qualification": {
            "closed_full_gate": full_gate,
            "x2_native_closed_sim_dynamic_seed": native_dynamic,
            "contact_clean_silver_seed": native_dynamic and low_slip,
            "hardware_dynamics_truth": False,
            "reason": "native dynamic seed requires full gate, exact 1kHz dynamics closure, realized contact cycle, and exact telemetry alignment; contact-clean Silver additionally requires both stance-slip p95<=0.20m/s",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmap", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    physical = read_trace(args.mmap)
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    result = {
        "stage": "BASE Phase34",
        "scope": "one closed full episode plus offline qualification; no training/WBT/hardware/Git/cloud",
        "assets": {"mmap": str(args.mmap), "mmap_sha256": sha256(args.mmap),
                   "rollout": str(args.rollout), "rollout_sha256": sha256(args.rollout),
                   "scene": str(args.scene), "scene_sha256": sha256(args.scene)},
        "trace_header": physical["header"],
        "analysis": analyze(model, physical, rollout),
    }
    result["decision"] = ("QUALIFIED_X2_NATIVE_CLOSED_SIM_DYNAMIC_SEED" if
                          result["analysis"]["qualification"]["x2_native_closed_sim_dynamic_seed"]
                          else "NOT_QUALIFIED_X2_NATIVE_DYNAMIC_SEED")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "coverage": result["analysis"]["coverage"],
                      "contact": result["analysis"]["contact"],
                      "dynamics": result["analysis"]["dynamics"],
                      "alignment": result["analysis"]["telemetry_alignment"]}, indent=2))


if __name__ == "__main__":
    main()
