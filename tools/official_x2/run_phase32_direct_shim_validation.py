#!/usr/bin/env python3
"""Run deterministic toy or official-scene MuJoCo prefix for Phase32."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from official_x2.audit_phase29_closed_wrapper_divergence import ISAAC_JOINTS, decode_row
from official_x2.replay_official_trace_direct_mujoco import JOINTS, pd_gains


ROOT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
SCENE = ROOT / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"
HISTORICAL = ROOT / "results/official_native_strict_20260807/stage250_video_straight.json"
CONTROLS = ROOT / "cache/phase28_stage250_extended_direct/stage250_straight_control_ticks.json"


def toy() -> tuple[mujoco.MjModel, mujoco.MjData, list[list[float]]]:
    xml = """<mujoco><option timestep='0.001'/><worldbody><body><joint name='j' type='hinge'/>
      <geom type='capsule' size='.03 .1' mass='1'/></body></worldbody>
      <actuator><motor joint='j' gear='1' ctrllimited='true' ctrlrange='-2 2'/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.2
    data.qvel[0] = -0.1
    mujoco.mj_forward(model, data)
    controls = [[0.3 * math.sin(index * 0.07)] for index in range(300)]
    return model, data, controls


def official() -> tuple[mujoco.MjModel, mujoco.MjData, list[list[float]]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE.resolve()))
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    row = next(row for row in historical["trace"] if len(row.get("obs", [])) == 93)
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dof_adr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    qpos, qvel = decode_row(model, row, qpos_adr, dof_adr)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    frozen = json.loads(CONTROLS.read_text(encoding="utf-8"))[:15]
    gains = pd_gains()
    joint_qpos = {name: int(model.joint(name).qposadr[0]) for name in JOINTS}
    joint_dof = {name: int(model.joint(name).dofadr[0]) for name in JOINTS}
    actuator = {name: int(model.actuator(f"motor_{name}").id) for name in JOINTS}
    controls = []
    for tick, tick_row in enumerate(frozen):
        if tick_row["pd_target_joint_order"] != list(JOINTS):
            raise RuntimeError("joint contract drift")
        targets = dict(zip(JOINTS, map(float, tick_row["pd_target_rad"])))
        ctrl = np.zeros(model.nu, dtype=np.float64)
        for name in JOINTS:
            kp, kd = gains[name]
            aid = actuator[name]
            torque = kp * (targets[name] - data.qpos[joint_qpos[name]]) - kd * data.qvel[joint_dof[name]]
            low, high = model.actuator_ctrlrange[aid]
            ctrl[aid] = np.clip(torque, low, high)
        # The validation intentionally freezes the exact resulting torque for each
        # 20-substep block, matching the Phase30 prefix contract.
        controls.extend([ctrl.tolist()] * 20)
        # Advance a private copy for the next tick's frozen-PD torque construction.
        private = mujoco.MjData(model)
        private.qpos[:] = data.qpos
        private.qvel[:] = data.qvel
        mujoco.mj_forward(model, private)
        for _ in range(20):
            private.ctrl[:] = ctrl
            mujoco.mj_step(model, private)
        data.qpos[:] = private.qpos
        data.qvel[:] = private.qvel
        data.time = private.time
        mujoco.mj_forward(model, data)
    # Restore the exact initial state after constructing the open-loop control list.
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.time = 0.0
    mujoco.mj_forward(model, data)
    return model, data, controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "official"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if mujoco.mj_versionString() != "3.3.7":
        raise RuntimeError(f"expected exact vendor 3.3.7, got {mujoco.mj_versionString()}")
    model, data, controls = toy() if args.mode == "toy" else official()
    rows = []
    started = time.perf_counter()
    for index, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        rows.append({"step": index + 1, "time_s": float(data.time),
                     "qpos": np.asarray(data.qpos).tolist(),
                     "qvel": np.asarray(data.qvel).tolist()})
    elapsed = time.perf_counter() - started
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "libmujoco.so" in line})
    payload = {"mode": args.mode, "mujoco_version": mujoco.mj_versionString(),
               "mapped_mujoco": mapped, "nq": model.nq, "nv": model.nv,
               "steps": len(rows), "step_wall_s": elapsed, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("mode", "mujoco_version", "steps", "step_wall_s")}, indent=2))


if __name__ == "__main__":
    main()
