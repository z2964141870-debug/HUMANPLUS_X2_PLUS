#!/usr/bin/env python3
"""Derive realized-contact/clearance metrics from the frozen Phase28 cache."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from official_x2.audit_stage250_native_dynamic_seed import (
    contact_geom_ids,
    contiguous_cycles,
    quantiles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--substeps", type=Path, required=True)
    parser.add_argument("--control-ticks", type=Path, required=True)
    parser.add_argument("--historical-trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    geom_ids = {side: contact_geom_ids(model, side) for side in ("left", "right")}
    by_tick: dict[int, dict] = {}
    slip = {side: [] for side in ("left", "right")}
    normal_force = {side: [] for side in ("left", "right")}
    impulse = {side: [] for side in ("left", "right")}
    with gzip.open(args.substeps, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tick = int(row["control_tick"])
            aggregate = by_tick.setdefault(tick, {
                "stage": row["stage"],
                "contact": {"left": False, "right": False},
                "qpos": row["qpos"],
            })
            aggregate["qpos"] = row["qpos"]
            for contact in row["contacts"]:
                side = contact["side"]
                if side not in ("left", "right"):
                    continue
                aggregate["contact"][side] = True
                force = float(contact["wrench_contact_frame"][0])
                normal_force[side].append(force)
                impulse[side].append(force * model.opt.timestep)
                value = contact["foot_point_horizontal_slip_mps"]
                if value is not None:
                    slip[side].append(float(value))
    contact = {side: [] for side in ("left", "right")}
    clearance = {side: [] for side in ("left", "right")}
    move_ticks = [tick for tick in sorted(by_tick) if by_tick[tick]["stage"] == "move"]
    for tick in move_ticks:
        aggregate = by_tick[tick]
        data.qpos[:] = np.asarray(aggregate["qpos"], dtype=np.float64)
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            is_contact = bool(aggregate["contact"][side])
            contact[side].append(is_contact)
            if not is_contact:
                centers = data.geom_xpos[geom_ids[side]]
                bottom = centers[:, 2] - model.geom_size[geom_ids[side], 0]
                clearance[side].append(float(max(0.0, np.min(bottom))))
    audit = {
        "physics_substep_contact_source": "mujoco.mj_contactForce on unchanged official scene.xml",
        "hardware_or_closed_ros_force": False,
        "contact_impulse_definition": "normal force × 0.001 s; derived approximation, not a separately exposed impulse",
        "move": {
            "ds_ss_ds_cycles": contiguous_cycles(
                np.asarray(contact["left"], dtype=bool), np.asarray(contact["right"], dtype=bool)
            ),
            "left": {
                "contact_tick_fraction": float(np.mean(contact["left"])),
                "contact_point_slip_mps": quantiles(slip["left"]),
                "swing_sole_clearance_m": quantiles(clearance["left"]),
                "contact_normal_force_n": quantiles(normal_force["left"]),
                "contact_normal_impulse_ns": quantiles(impulse["left"]),
            },
            "right": {
                "contact_tick_fraction": float(np.mean(contact["right"])),
                "contact_point_slip_mps": quantiles(slip["right"]),
                "swing_sole_clearance_m": quantiles(clearance["right"]),
                "contact_normal_force_n": quantiles(normal_force["right"]),
                "contact_normal_impulse_ns": quantiles(impulse["right"]),
            },
        },
        "interpretation": "episode had already collapsed during stand; these are realized failure-domain contacts, not a valid locomotion seed",
    }
    direct_ticks = json.loads(args.control_ticks.read_text(encoding="utf-8"))
    historical_payload = json.loads(args.historical_trace.read_text(encoding="utf-8"))
    historical_ticks = [row for row in historical_payload["trace"] if len(row.get("obs", [])) == 93]
    first_divergence = None
    for index in range(min(len(direct_ticks), len(historical_ticks) - 1)):
        direct = direct_ticks[index]
        historical = historical_ticks[index + 1]
        z_error = abs(float(direct["root_z_m"]) - float(historical["root_z_m"]))
        tilt_error = abs(float(direct["root_tilt_rad"]) - float(historical["root_tilt_rad"]))
        if z_error > 0.01 or tilt_error > 0.05:
            first_divergence = {
                "control_tick": index,
                "elapsed_from_initial_state_s": (index + 1) * 0.02,
                "root_z_abs_error_m": z_error,
                "root_tilt_abs_error_rad": tilt_error,
                "diagnostic_thresholds_not_gates": {"root_z_m": 0.01, "root_tilt_rad": 0.05},
            }
            break
    audit["historical_state_divergence"] = {
        "initial_obs93_abs_max_error": float(np.max(np.abs(
            np.asarray(direct_ticks[0]["obs93"], dtype=np.float64)
            - np.asarray(historical_ticks[0]["obs"], dtype=np.float64)
        ))),
        "initial_action_abs_max_error": float(np.max(np.abs(
            np.asarray(direct_ticks[0]["final_clip_action"], dtype=np.float64)
            - np.asarray(historical_ticks[0]["action"], dtype=np.float64)
        ))),
        "first_material_divergence": first_divergence,
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["extended_realized_contact_audit"] = audit
    manifest["qualified_native_dynamic_seed"] = False
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["move"], indent=2))


if __name__ == "__main__":
    main()
