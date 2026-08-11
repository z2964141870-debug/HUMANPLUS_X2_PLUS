#!/usr/bin/env python3
"""One-shot reset-compatible contact projection and raw official-PD bridge."""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
REPO = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
CONTROL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml")
PHASE3B = ROOT / "phase3b"
OUT = ROOT / "phase6_reset_bridge"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(PHASE3B), str(REPO / "tools"), str(REPO)]

import mujoco
import retarget.run_x2_forefoot_official_physics_screen as physics
import x2_lunge_dsms_phase3b_corrected as base


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def sole_clearances(model, data, feet, floor):
    floor_z = float(data.geom_xpos[floor, 2])
    return {s: np.asarray([data.geom_xpos[g, 2] - model.geom_size[g, 0] - floor_z
                           for g in sorted(gs)], dtype=np.float64) for s, gs in feet.items()}


def actual_contacts(model, data, floor, geom_side):
    out = {"left": set(), "right": set()}
    for ci in range(data.ncon):
        c = data.contact[ci]; a, b = int(c.geom1), int(c.geom2)
        other = b if a == floor else a if b == floor else -1
        if other in geom_side: out[geom_side[other]].add(other)
    return out


def collidable_sole_spheres(model, feet):
    """The official helper includes one non-colliding foot mesh per side.

    Only the 12 contype-enabled sphere geoms per foot have a meaningful
    sphere-to-plane clearance and can produce effective floor contact.
    """
    return {s:{g for g in gs if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
               and int(model.geom_contype[g]) != 0} for s,gs in feet.items()}


def project_initial(Xref, contract):
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    floor, helper_feet = physics.foot_geom_contract(model)
    feet = collidable_sole_spheres(model, helper_feet)
    assert all(len(gs) == 12 for gs in feet.values()), {s:len(gs) for s,gs in feet.items()}
    names = ([f"{s}_{j}_joint" for s in ("left", "right")
              for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
             + ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"])
    qadr = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names], dtype=int)
    jids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names], dtype=int)
    upper_names = ["torso_link", "head_pitch_link", "left_wrist_roll_link", "right_wrist_roll_link"]
    upper_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in upper_names], dtype=int)
    qref = Xref[0, :model.nq].copy()

    def set_z(z):
        data.qpos[:] = qref
        data.qpos[:3] = z[:3]
        data.qpos[qadr] = z[3:]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    zref = np.r_[qref[:3], qref[qadr]]
    set_z(zref); initial_clear = sole_clearances(model, data, feet, floor)
    upper_ref = data.xpos[upper_ids].copy()
    # Fixed scales encode preservation priorities; constraints, not penalty tuning,
    # determine contact feasibility.
    def objective(z):
        set_z(z)
        root = (z[:3] - zref[:3]) / np.array([.03, .03, .03])
        joints = (z[3:] - zref[3:]) / .20
        upper = (data.xpos[upper_ids] - upper_ref).ravel() / .04
        return float(np.dot(root, root) + np.dot(joints, joints) + .25*np.dot(upper, upper))

    def clearance_vec(z):
        set_z(z); c = sole_clearances(model, data, feet, floor)
        return np.r_[c["left"], c["right"]]

    def contact_band(z):
        set_z(z); c = sole_clearances(model, data, feet, floor)
        # >=0 means at least one sphere on each foot is 0.05 mm into contact.
        return np.asarray([-0.00005 - np.min(c["left"]), -0.00005 - np.min(c["right"])])

    lo = zref.copy(); hi = zref.copy()
    lo[:2] -= .05; hi[:2] += .05; lo[2] -= .08; hi[2] += .08
    jr = model.jnt_range[jids]
    lo[3:] = np.maximum(jr[:, 0], zref[3:] - .35)
    hi[3:] = np.minimum(jr[:, 1], zref[3:] + .35)
    result = minimize(objective, zref, method="SLSQP", bounds=list(zip(lo, hi)),
                      constraints=[{"type":"ineq", "fun":lambda z: clearance_vec(z)+.0005},
                                   {"type":"ineq", "fun":contact_band}],
                      options={"maxiter":500, "ftol":1e-12, "disp":True})
    set_z(result.x); projected_clear = sole_clearances(model, data, feet, floor)
    geom_side = {g:s for s,gs in feet.items() for g in gs}
    contacts = actual_contacts(model, data, floor, geom_side)
    upper_delta = np.linalg.norm(data.xpos[upper_ids]-upper_ref, axis=1)
    qproj = qref.copy(); qproj[:3] = result.x[:3]; qproj[qadr] = result.x[3:]
    report = {
        "solver_success": bool(result.success), "solver_status": int(result.status),
        "solver_message": str(result.message), "iterations": int(result.nit), "objective": float(result.fun),
        "root_delta_m": (qproj[:3]-qref[:3]).tolist(),
        "selected_joint_delta_rms_rad": float(np.sqrt(np.mean((qproj[qadr]-qref[qadr])**2))),
        "selected_joint_delta_max_rad": float(np.max(np.abs(qproj[qadr]-qref[qadr]))),
        "selected_joint_deltas_rad": dict(zip(names, (qproj[qadr]-qref[qadr]).tolist())),
        "upper_keypoint_names": upper_names, "upper_keypoint_error_m": upper_delta.tolist(),
        "upper_keypoint_error_rms_m": float(np.sqrt(np.mean(upper_delta**2))),
        "upper_keypoint_error_max_m": float(np.max(upper_delta)),
        "initial_clearance_mm": {s:(1000*c).tolist() for s,c in initial_clear.items()},
        "initial_min_clearance_mm": {s:float(1000*np.min(c)) for s,c in initial_clear.items()},
        "projected_clearance_mm": {s:(1000*c).tolist() for s,c in projected_clear.items()},
        "projected_min_clearance_mm": {s:float(1000*np.min(c)) for s,c in projected_clear.items()},
        "projected_contact_geom_count": {s:len(contacts[s]) for s in contacts},
        "all_nonpenetrating": bool(min(np.min(c) for c in projected_clear.values()) >= -.0005-1e-8),
        "effective_contact_each_foot": bool(all(len(contacts[s]) > 0 for s in contacts)),
        "velocity_reset": "all qvel zero"
    }
    report["sole_geom_contract"] = {"helper_count":{s:len(gs) for s,gs in helper_feet.items()},
                                    "collidable_sphere_count":{s:len(gs) for s,gs in feet.items()},
                                    "excluded_noncolliding_geom_ids":{s:sorted(helper_feet[s]-feet[s]) for s in feet}}
    report["pass"] = bool(report["solver_success"] and report["all_nonpenetrating"] and report["effective_contact_each_foot"])
    return qproj, qref, qadr, names, report


def replay(qproj, qref0, Xref, contract, projection):
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    floor, helper_feet = physics.foot_geom_contract(model)
    feet = collidable_sole_spheres(model, helper_feet)
    geom_side = {g:s for s,gs in feet.items() for g in gs}
    data.qpos[:] = qproj; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)
    act_qadr = np.asarray(contract.qpos_addresses, dtype=int)
    delta = qproj[act_qadr] - qref0[act_qadr]
    steps = 340; dt = float(model.opt.timestep)
    prev_geom = data.geom_xpos.copy()
    init_centroid = {s:np.mean(data.geom_xpos[list(gs),:2],axis=0) for s,gs in feet.items()}
    slip={s:[] for s in feet}; excursion={s:[] for s in feet}; actual={s:[] for s in feet}
    min_clear={s:np.inf for s in feet}; root_vel=[data.qvel[:2].copy()]
    root_z=[float(data.qpos[2])]; tilt=[physics.root_tilt(data.qpos[3:7])]
    saturation=0; torque_n=0; fell_at=None

    def sample():
        c=sole_clearances(model,data,feet,floor)
        contacts=actual_contacts(model,data,floor,geom_side)
        for s,gs in feet.items():
            actual[s].append(bool(contacts[s])); min_clear[s]=min(min_clear[s],float(np.min(c[s])))
            centroid=np.mean(data.geom_xpos[list(gs),:2],axis=0)
            excursion[s].append(float(np.linalg.norm(centroid-init_centroid[s])))
            for g in contacts[s]: slip[s].append(float(np.linalg.norm(data.geom_xpos[g,:2]-prev_geom[g,:2])/dt))

    sample()  # strict t0 accounting
    ref_act = Xref[:, act_qadr]
    for k in range(steps):
        t=k*dt; phase=min(t/.02,len(ref_act)-1.0); i=int(np.floor(phase)); j=min(i+1,len(ref_act)-1); a=phase-i
        qref=(1-a)*ref_act[i]+a*ref_act[j]
        u=np.clip(t/.34,0.,1.); smooth=u*u*(3.-2.*u)
        target=qref+(1.-smooth)*delta
        # Head stays at the official neutral target.
        target[-2:]=0.0
        q=data.qpos[act_qadr]; dq=data.qvel[contract.qvel_addresses]
        raw=contract.kp*(target-q)-contract.kd*dq
        saturation += int(np.sum((raw<contract.torque_low)|(raw>contract.torque_high))); torque_n += len(raw)
        data.ctrl[:]=np.clip(raw,contract.torque_low,contract.torque_high)
        mujoco.mj_step(model,data); sample(); prev_geom[:]=data.geom_xpos
        root_vel.append(data.qvel[:2].copy()); root_z.append(float(data.qpos[2])); tilt.append(physics.root_tilt(data.qpos[3:7]))
        if fell_at is None and (data.qpos[2]<.42 or tilt[-1]>.90): fell_at=(k+1)*dt
    times=np.arange(steps+1)*dt
    intended={"left":times<=10/30+1e-12,"right":times<=25/30+1e-12}
    accel=np.diff(np.asarray(root_vel),axis=0)/dt; anorm=np.linalg.norm(accel,axis=1)
    out={
        "schema":"x2_phase6_reset_bridge_result_v1", "scene_sha256":sha(SCENE), "control_sha256":sha(CONTROL),
        "mujoco":mujoco.__version__, "projection":projection, "duration_s":steps*dt,
        "fall_time_s":fell_at, "survive_full_prefix":fell_at is None,
        "root_z_min_m":float(min(root_z)), "tilt_max_rad":float(max(tilt)),
        "torque_saturation_fraction":saturation/max(torque_n,1),
        "penetration_min_mm":{s:1000*min_clear[s] for s in feet},
        "root_horiz_accel_p95_max_mps2":[float(np.percentile(anorm,95)),float(np.max(anorm))],
        "head_q_max_abs_rad":float(np.max(np.abs(data.qpos[act_qadr[-2:]]))), "contact":{}
    }
    for s in feet:
        mask=intended[s]; arr=np.asarray(actual[s],bool)
        out["contact"][s]={
            "realized_during_intent_fraction":float(np.mean(arr[mask])),
            "contradiction_fraction":float(np.mean(~arr[mask])),
            "stance_speed_p95_mps":float(np.percentile(slip[s],95)) if slip[s] else None,
            "stance_excursion_max_m":float(np.max(np.asarray(excursion[s])[mask]))
        }
    flight=~(np.asarray(actual["left"])|np.asarray(actual["right"])); out["unintended_flight_fraction"]=float(np.mean(flight))
    gates={
        "projection":bool(projection["pass"]), "survive_full_prefix":out["survive_full_prefix"],
        "penetration":min(out["penetration_min_mm"].values())>=-.5,
        "stance_speed":all(out["contact"][s]["stance_speed_p95_mps"] is not None and out["contact"][s]["stance_speed_p95_mps"]<=.10 for s in feet),
        "stance_excursion":all(out["contact"][s]["stance_excursion_max_m"]<=.03 for s in feet),
        "flight":out["unintended_flight_fraction"]<=.02,
        "root_accel":out["root_horiz_accel_p95_max_mps2"][0]<=4., "head":out["head_q_max_abs_rad"]<=.02
    }
    out["gates"]=gates; out["pass"]=all(gates.values())
    return out


def main():
    entry, adapted, ref, Xref, contract=base.load_reference(18)
    qproj,qref0,qadr,names,projection=project_initial(Xref,contract)
    result=replay(qproj,qref0,Xref,contract,projection) if projection["pass"] else {"projection":projection,"pass":False,"stop":"projection failed"}
    np.savez_compressed(OUT/"phase6_projected_initial_state.npz",qpos=qproj,qvel=np.zeros(37),reference_qpos=qref0,
                        selected_qpos_addresses=qadr,selected_joint_names=np.asarray(names))
    (OUT/"phase6_reset_bridge_result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
