#!/usr/bin/env python3
"""Test fixed 18-knot official position-PD target compression."""

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
from official_x2.replay_official_trace_direct_mujoco import JOINTS, pd_gains, yaw_tilt


PHASE32 = REPO / "research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_result.json"
PHASE33 = REPO / "research/dynamic_retargeting_20260811/phase33_native_control_spline_result.json"
CONTRACT = REPO / "research/dynamic_retargeting_20260811/phase33b_native_pd_spline_contract.json"
OUTPUT = REPO / "research/dynamic_retargeting_20260811/phase33b_native_pd_spline_result.json"
STEPS, KNOTS = 340, 18


def _addresses(model):
    qadr, dadr, actuator = [], [], []
    for name in JOINTS:
        joint = int(model.joint(name).id)
        qadr.append(int(model.jnt_qposadr[joint])); dadr.append(int(model.jnt_dofadr[joint]))
        actuator.append(int(model.actuator(f"motor_{name}").id))
    return np.asarray(qadr), np.asarray(dadr), np.asarray(actuator)


def _servo_model(scene: Path):
    model = mujoco.MjModel.from_xml_path(str(scene))
    qadr, dadr, actuator = _addresses(model)
    gains = pd_gains()
    kp = np.asarray([gains[name][0] for name in JOINTS]); kd = np.asarray([gains[name][1] for name in JOINTS])
    torque_range = model.actuator_ctrlrange.copy()
    target_range = model.jnt_range[model.actuator_trnid[:, 0]].copy()
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
    model.actuator_gainprm[:] = 0.0; model.actuator_biasprm[:] = 0.0
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd
    model.actuator_gear[:] = 0.0; model.actuator_gear[:, 0] = 1.0
    model.actuator_ctrlrange[:] = target_range; model.actuator_ctrllimited[:] = 1
    model.actuator_forcerange[:] = torque_range; model.actuator_forcelimited[:] = 1
    head = [int(model.actuator(name).id) for name in ("motor_head_yaw_joint", "motor_head_pitch_joint")]
    model.actuator_ctrlrange[head] = 0.0
    return model, qadr, dadr, actuator, kp, kd, target_range, head


def _contacts(model, data):
    floor = int(model.geom("floor").id)
    side_by_geom = {geom: side for side in ("left", "right") for geom in contact_geom_ids(model, side)}
    found = {"left": False, "right": False}
    for index in range(data.ncon):
        row = data.contact[index]; g1, g2 = int(row.geom1), int(row.geom2)
        if floor in (g1, g2):
            side = side_by_geom.get(g2 if g1 == floor else g1)
            if side: found[side] = True
    return found


def _tangent(model, qa, va, qb, vb):
    dq = np.zeros(model.nv); mujoco.mj_differentiatePos(model, dq, 1.0, qa, qb)
    return float(max(np.max(np.abs(dq)), np.max(np.abs(vb - va))))


def run():
    source = json.loads(PHASE32.read_text())
    prior = json.loads(PHASE33.read_text())
    if not source["decision"]["native_dsms_warmstart_qualified"] or prior["decision"]["control_spline_warmstart_qualified"]:
        raise RuntimeError("Phase33b is only valid after Phase32 pass and Phase33 torque-spline rejection")
    contract = json.loads(CONTRACT.read_text())
    paths = {name: Path(row["path"]) for name, row in source["assets"].items()}
    for name, path in paths.items():
        if sha256(path) != source["assets"][name]["sha256"]: raise RuntimeError(f"asset drift: {name}")
    model, qadr, dadr, actuator, kp, kd, limits, head = _servo_model(paths["scene"])
    physical = read_trace(paths["mmap"]); anchor = int(source["selection"]["anchor_physics_index"])
    targets = []
    for step in range(STEPS):
        pre, applied = anchor + step, anchor + 1 + step
        targets.append(physical["qpos"][pre, qadr] + (physical["ctrl"][applied, actuator] + kd * physical["qvel"][pre, dadr]) / kp)
    targets = np.asarray(targets)
    spline = make_spline(SplineConfig(M=KNOTS, spline_type="linear"), STEPS, model.nu)
    points = spline.fit(targets).reshape(KNOTS, model.nu); points[:, head] = 0.0
    points = np.clip(points, limits[:, 0], limits[:, 1]); compressed = spline.evaluate(points)
    data = mujoco.MjData(model); data.time = float(physical["time"][anchor])
    data.qpos[:] = physical["qpos"][anchor]; data.qvel[:] = physical["qvel"][anchor]; data.ctrl[:] = compressed[0]
    mujoco.mj_forward(model, data)
    qerr = terr = 0.0; node_errors = []; zs = []; tilts = []; support = []
    for step in range(STEPS):
        data.ctrl[:] = compressed[step]; mujoco.mj_step(model, data); target = anchor + 1 + step
        qerr = max(qerr, float(np.max(np.abs(data.qpos - physical["qpos"][target]))))
        zs.append(float(data.qpos[2])); tilts.append(float(yaw_tilt(data.qpos[3:7])[1]))
        c = _contacts(model, data); support.append(c["left"] or c["right"])
        if (step + 1) % 20 == 0:
            e = _tangent(model, physical["qpos"][target], physical["qvel"][target], data.qpos, data.qvel)
            node_errors.append(e); terr = max(terr, e)
    overshoot = float(max(0.0, np.max(np.maximum(limits[:, 0] - points, points - limits[:, 1]))))
    metrics = {"target_fit_rmse_rad": float(np.sqrt(np.mean((compressed-targets)**2))), "target_fit_absmax_rad": float(np.max(np.abs(compressed-targets))),
               "control_point_range_overshoot": overshoot, "qpos_absmax_vs_recorded": qerr, "node_state_tangent_absmax": terr,
               "node_state_tangent_errors": node_errors, "root_z_min_m": float(np.min(zs)), "root_tilt_max_rad": float(np.max(tilts)),
               "at_least_one_foot_contact_fraction": float(np.mean(support))}
    g = contract["hard_gates"]
    checks = {"node_state_tangent": terr <= g["node_state_tangent_absmax"], "qpos": qerr <= g["qpos_absmax_vs_recorded"],
              "root_z": metrics["root_z_min_m"] >= g["root_z_min_m"], "root_tilt": metrics["root_tilt_max_rad"] <= g["root_tilt_max_rad"],
              "support": metrics["at_least_one_foot_contact_fraction"] >= g["at_least_one_foot_contact_fraction"], "bounds": overshoot <= g["control_point_range_overshoot"]}
    passed = bool(all(checks.values()))
    return {"stage":"Dynamic Retargeting Phase33b", "scope":"fixed equivalent-PD 18-knot replay; no optimizer/training/GPU/closed ROS",
            "source_phase32_sha256":sha256(PHASE32), "source_phase33_sha256":sha256(PHASE33), "contract_sha256":sha256(CONTRACT),
            "metrics":metrics, "checks":checks, "execution":{"mj_step_calls":STEPS,"optimizer_instances":0,"optimizer_steps":0,"cpu_threads":1,"gpu":False},
            "decision":{"pd_spline_warmstart_qualified":passed,"single_short_solve_authorized":False,"training_unlocked":False,
                        "result":"PD_SPLINE_PREFLIGHT_PASSED_SOLVE_STILL_LOCKED" if passed else "PD_SPLINE_REJECTED_NO_SOLVE"}}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=OUTPUT); args=parser.parse_args()
    result=run(); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"metrics":result["metrics"],"checks":result["checks"],"decision":result["decision"]},indent=2))


if __name__ == "__main__": main()
