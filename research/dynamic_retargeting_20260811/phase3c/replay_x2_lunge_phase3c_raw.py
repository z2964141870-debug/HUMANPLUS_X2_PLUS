#!/usr/bin/env python3
"""Independent raw-motor replay of the corrected Phase3b returned iterate."""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
REPO = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
CONTROL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml")
ARTIFACT = ROOT / "phase3b/x2_lunge_phase3b_prefix.npz"
OUT = ROOT / "phase3b/phase3b_raw_motor_replay.json"
sys.path[:0] = [str(REPO / "tools"), str(REPO)]

import mujoco
import retarget.run_x2_forefoot_official_physics_screen as physics


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--output", type=Path, default=OUT); ap.add_argument("--status-ok", action="store_true")
    args = ap.parse_args(); artifact, output = args.artifact, args.output
    z = np.load(artifact, allow_pickle=True); target = np.asarray(z["input"], dtype=np.float64)
    stitched = np.asarray(z["state"], dtype=np.float64); defects = np.asarray(z["defects"], dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    contract = physics.build_control_contract(model, CONTROL)
    data.qpos[:] = stitched[0, :model.nq]; data.qvel[:] = stitched[0, model.nq:model.nq+model.nv]
    mujoco.mj_forward(model, data)
    floor, raw_feet = physics.foot_geom_contract(model)
    # The helper also returns one contype=0 visual ankle mesh per side.  Only
    # the 12 active 5-mm sole spheres participate in official collision.
    feet = {s: {g for g in gs if int(model.geom_contype[g]) != 0} for s, gs in raw_feet.items()}
    if any(len(gs) != 12 for gs in feet.values()):
        raise RuntimeError(f"expected 12 active sole spheres per foot, got { {s:len(gs) for s,gs in feet.items()} }")
    geom_side = {g: s for s, gs in feet.items() for g in gs}
    prev_geom = data.geom_xpos.copy(); initial_centroid = {s: np.mean(data.geom_xpos[list(gs), :2], axis=0) for s, gs in feet.items()}
    slip = {"left": [], "right": []}; excursion = {"left": [], "right": []}
    realized = {"left": [], "right": []}; min_clearance = {"left": np.inf, "right": np.inf}
    root_vel = [data.qvel[:2].copy()]; tilt = [physics.root_tilt(data.qpos[3:7])]; root_z = [float(data.qpos[2])]
    saturation = 0; torque_samples = 0; fell_at = None
    for k in range(len(target)):
        q = data.qpos[contract.qpos_addresses]; dq = data.qvel[contract.qvel_addresses]
        raw = contract.kp * (target[k] - q) - contract.kd * dq
        saturation += int(np.sum((raw < contract.torque_low) | (raw > contract.torque_high))); torque_samples += len(raw)
        data.ctrl[:] = np.clip(raw, contract.torque_low, contract.torque_high); mujoco.mj_step(model, data)
        contacts = {"left": set(), "right": set()}
        for ci in range(data.ncon):
            c = data.contact[ci]; a, b = int(c.geom1), int(c.geom2)
            other = b if a == floor else a if b == floor else -1
            if other in geom_side: contacts[geom_side[other]].add(other)
        for side, geoms in feet.items():
            realized[side].append(bool(contacts[side]))
            for g in contacts[side]: slip[side].append(float(np.linalg.norm(data.geom_xpos[g, :2] - prev_geom[g, :2]) / model.opt.timestep))
            centroid = np.mean(data.geom_xpos[list(geoms), :2], axis=0)
            excursion[side].append(float(np.linalg.norm(centroid - initial_centroid[side])))
            clearance = [float(data.geom_xpos[g, 2] - model.geom_size[g, 0] - data.geom_xpos[floor, 2]) for g in geoms]
            min_clearance[side] = min(min_clearance[side], min(clearance))
        prev_geom[:] = data.geom_xpos; root_vel.append(data.qvel[:2].copy())
        current_tilt = physics.root_tilt(data.qpos[3:7]); tilt.append(current_tilt); root_z.append(float(data.qpos[2]))
        if fell_at is None and (data.qpos[2] < 0.42 or current_tilt > 0.90): fell_at = (k + 1) * model.opt.timestep
    # Frozen Phase30 source intent: L stance [0,10]/30s; R stance [0,25]/30s.
    times = (np.arange(len(target)) + 1) * model.opt.timestep
    intended = {"left": times <= 10.0/30.0 + 1e-12, "right": times <= 25.0/30.0 + 1e-12}
    rv = np.asarray(root_vel); root_acc = np.diff(rv, axis=0) / model.opt.timestep
    q_terminal_error = float(np.max(np.abs(data.qpos - stitched[-1, :model.nq])))
    result = {
        "schema": "x2_phase3b_raw_motor_replay_v1", "artifact": str(artifact), "artifact_sha256": sha(artifact),
        "scene_sha256": sha(SCENE), "control_sha256": sha(CONTROL), "mujoco": mujoco.__version__,
        "duration_s": len(target) * model.opt.timestep, "fall_time_s": fell_at,
        "survive_full_prefix": fell_at is None, "root_z_min_m": min(root_z), "tilt_max_rad": max(tilt),
        "max_defect": float(np.max(np.abs(defects))), "terminal_qpos_vs_stitched_max_abs": q_terminal_error,
        "torque_saturation_fraction": saturation / max(torque_samples, 1),
        "root_horiz_accel_p95_max_mps2": [float(np.percentile(np.linalg.norm(root_acc, axis=1), 95)), float(np.max(np.linalg.norm(root_acc, axis=1)))],
        "head_q_max_abs_rad": float(np.max(np.abs(data.qpos[contract.qpos_addresses[-2:]]))),
        "contact": {}, "penetration_min_mm": {s: 1000.0 * min_clearance[s] for s in feet},
    }
    for side in feet:
        mask = intended[side]; actual = np.asarray(realized[side], dtype=bool)
        result["contact"][side] = {
            "intended_fraction": float(np.mean(mask)), "realized_during_intent_fraction": float(np.mean(actual[mask])),
            "contradiction_fraction": float(np.mean(~actual[mask])),
            "stance_speed_p95_mps": float(np.percentile(slip[side], 95)) if slip[side] else None,
            "stance_excursion_max_m": float(np.max(np.asarray(excursion[side])[mask])),
        }
    both_absent = ~(np.asarray(realized["left"]) | np.asarray(realized["right"]))
    result["unintended_flight_fraction"] = float(np.mean(both_absent))
    gates = {
        "ipopt_status": bool(args.status_ok), "max_defect": result["max_defect"] <= 1e-2,
        "survive_full_prefix": result["survive_full_prefix"],
        "penetration": min(result["penetration_min_mm"].values()) >= -0.5,
        "stance_speed": all(result["contact"][s]["stance_speed_p95_mps"] is not None and result["contact"][s]["stance_speed_p95_mps"] <= .10 for s in feet),
        "stance_excursion": all(result["contact"][s]["stance_excursion_max_m"] <= .03 for s in feet),
        "flight": result["unintended_flight_fraction"] <= .02,
        "root_accel": result["root_horiz_accel_p95_max_mps2"][0] <= 4.0,
        "head": result["head_q_max_abs_rad"] <= .02,
    }
    result["gates"] = gates; result["pass"] = all(gates.values())
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
