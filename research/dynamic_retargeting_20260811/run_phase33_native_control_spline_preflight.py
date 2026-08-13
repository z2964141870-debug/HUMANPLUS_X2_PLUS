#!/usr/bin/env python3
"""Preflight the fixed 50 Hz linear control spline on the native suffix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/home/humanplus/projects/ZHY/dsms_workspace/shooting-for-contact")
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault("TRAJOPT_ROOT_DIR", str(UPSTREAM))

from src.spline import SplineConfig, make_spline
from official_x2.analyze_phase34_full_closed_trace import read_trace, sha256
from official_x2.audit_stage250_native_dynamic_seed import contact_geom_ids
from official_x2.replay_official_trace_direct_mujoco import yaw_tilt


PHASE32 = REPO / "research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_result.json"
CONTRACT = REPO / "research/dynamic_retargeting_20260811/phase33_native_control_spline_contract.json"
OUTPUT = REPO / "research/dynamic_retargeting_20260811/phase33_native_control_spline_result.json"
STEPS = 340
KNOTS = 18


def _contact_sides(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    floor = int(model.geom("floor").id)
    geom_side = {geom: side for side in ("left", "right") for geom in contact_geom_ids(model, side)}
    result = {"left": False, "right": False}
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if floor in (geom1, geom2):
            side = geom_side.get(geom2 if geom1 == floor else geom1)
            if side is not None:
                result[side] = True
    return result


def _state_tangent_absmax(model: mujoco.MjModel, qpos_a: np.ndarray, qvel_a: np.ndarray,
                          qpos_b: np.ndarray, qvel_b: np.ndarray) -> float:
    position = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, position, 1.0, qpos_a, qpos_b)
    return float(max(np.max(np.abs(position)), np.max(np.abs(qvel_b - qvel_a))))


def run() -> dict:
    source = json.loads(PHASE32.read_text(encoding="utf-8"))
    if not source["decision"]["native_dsms_warmstart_qualified"]:
        raise RuntimeError("Phase32 native warm-start is not qualified")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    paths = {name: Path(row["path"]) for name, row in source["assets"].items()}
    for name, path in paths.items():
        if sha256(path) != source["assets"][name]["sha256"]:
            raise RuntimeError(f"Phase32 {name} hash drift")
    if mujoco.mj_versionString() != "3.3.7":
        raise RuntimeError(f"MuJoCo 3.3.7 required, got {mujoco.mj_versionString()}")
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    physical = read_trace(paths["mmap"])
    anchor = int(source["selection"]["anchor_physics_index"])
    recorded = physical["ctrl"][anchor + 1:anchor + 1 + STEPS].copy()
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    recorded = np.clip(recorded, low, high)
    spline = make_spline(SplineConfig(M=KNOTS, spline_type="linear"), STEPS, model.nu)
    points = spline.fit(recorded).reshape(KNOTS, model.nu)
    head = [int(model.actuator(name).id) for name in ("motor_head_yaw_joint", "motor_head_pitch_joint")]
    points[:, head] = 0.0
    points = np.clip(points, low, high)
    compressed = spline.evaluate(points)

    data = mujoco.MjData(model)
    data.time = float(physical["time"][anchor])
    data.qpos[:] = physical["qpos"][anchor]
    data.qvel[:] = physical["qvel"][anchor]
    data.ctrl[:] = compressed[0]
    mujoco.mj_forward(model, data)
    qpos_error, tangent_error = 0.0, 0.0
    root_z, root_tilt, any_contact = [], [], []
    node_errors = []
    for step in range(STEPS):
        data.ctrl[:] = compressed[step]
        mujoco.mj_step(model, data)
        target = anchor + 1 + step
        qpos_error = max(qpos_error, float(np.max(np.abs(data.qpos - physical["qpos"][target]))))
        root_z.append(float(data.qpos[2]))
        root_tilt.append(float(yaw_tilt(data.qpos[3:7])[1]))
        contact = _contact_sides(model, data)
        any_contact.append(contact["left"] or contact["right"])
        if (step + 1) % 20 == 0:
            error = _state_tangent_absmax(model, physical["qpos"][target], physical["qvel"][target], data.qpos, data.qvel)
            node_errors.append(error)
            tangent_error = max(tangent_error, error)

    range_overshoot = float(max(0.0, np.max(np.maximum(low - points, points - high))))
    gates = contract["hard_gates"]
    metrics = {
        "control_fit_rmse": float(np.sqrt(np.mean(np.square(compressed - recorded)))),
        "control_fit_absmax": float(np.max(np.abs(compressed - recorded))),
        "control_point_range_overshoot": range_overshoot,
        "qpos_absmax_vs_recorded": qpos_error,
        "node_state_tangent_absmax": tangent_error,
        "node_state_tangent_errors": node_errors,
        "root_z_min_m": float(np.min(root_z)),
        "root_tilt_max_rad": float(np.max(root_tilt)),
        "at_least_one_foot_contact_fraction": float(np.mean(any_contact)),
    }
    checks = {
        "node_state_tangent": metrics["node_state_tangent_absmax"] <= gates["node_state_tangent_absmax"],
        "qpos": metrics["qpos_absmax_vs_recorded"] <= gates["qpos_absmax_vs_recorded"],
        "root_z": metrics["root_z_min_m"] >= gates["root_z_min_m"],
        "root_tilt": metrics["root_tilt_max_rad"] <= gates["root_tilt_max_rad"],
        "support": metrics["at_least_one_foot_contact_fraction"] >= gates["at_least_one_foot_contact_fraction"],
        "bounds": metrics["control_point_range_overshoot"] <= gates["control_point_range_overshoot"],
    }
    passed = bool(all(checks.values()))
    return {
        "stage": "Dynamic Retargeting Phase33",
        "scope": "fixed 18-knot linear compression replay only; no optimizer/training/GPU/closed ROS",
        "source_phase32_sha256": sha256(PHASE32),
        "contract_sha256": sha256(CONTRACT),
        "representation": contract["representation"],
        "metrics": metrics,
        "checks": checks,
        "execution": {"mj_step_calls": STEPS, "optimizer_instances": 0, "optimizer_steps": 0, "cpu_threads": 1, "gpu": False},
        "decision": {
            "control_spline_warmstart_qualified": passed,
            "single_short_solve_authorized": False,
            "training_unlocked": False,
            "result": "CONTROL_SPLINE_PREFLIGHT_PASSED_SOLVE_STILL_LOCKED" if passed else "CONTROL_SPLINE_REJECTED_NO_SOLVE",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "checks": result["checks"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
