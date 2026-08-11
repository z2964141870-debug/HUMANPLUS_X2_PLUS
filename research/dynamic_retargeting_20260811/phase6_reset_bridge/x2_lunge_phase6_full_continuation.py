#!/usr/bin/env python3
"""Zero-optimization deterministic full continuation of frozen Phase6 reset/bridge."""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
PHASE6 = ROOT / "phase6_reset_bridge"
PHASE3B = ROOT / "phase3b"
REPO = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
CONTROL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml")
RESET = PHASE6 / "phase6_projected_initial_state.npz"
OUT = PHASE6 / "phase6_full_continuation.json"
sys.path[:0] = [str(PHASE6), str(PHASE3B), str(REPO / "tools"), str(REPO)]

import mujoco
import retarget.run_x2_forefoot_official_physics_screen as physics
import x2_lunge_dsms_phase3b_corrected as base
from x2_lunge_phase6_reset_bridge import collidable_sole_spheres, actual_contacts, sole_clearances


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def source_intent(side: str, t: np.ndarray) -> np.ndarray:
    # Frozen Phase30 source contact intervals, inclusive source-frame endpoints.
    intervals={"left":[(0,10),(20,92),(105,175)], "right":[(0,25),(83,115),(171,175)]}[side]
    return np.logical_or.reduce([(t >= a/30-1e-12) & (t <= b/30+1e-12) for a,b in intervals])


def main():
    reset=np.load(RESET,allow_pickle=False); qproj=np.asarray(reset["qpos"],dtype=np.float64)
    qvel0=np.asarray(reset["qvel"],dtype=np.float64)
    _,_,_,Xref,contract=base.load_reference(10000)
    model=mujoco.MjModel.from_xml_path(str(SCENE)); data=mujoco.MjData(model)
    assert len(qproj)==model.nq and len(qvel0)==model.nv
    assert np.max(np.abs(qvel0))==0.0
    floor,helper_feet=physics.foot_geom_contract(model); feet=collidable_sole_spheres(model,helper_feet)
    assert all(len(gs)==12 for gs in feet.values())
    geom_side={g:s for s,gs in feet.items() for g in gs}
    data.qpos[:]=qproj; data.qvel[:]=qvel0; mujoco.mj_forward(model,data)
    act_qadr=np.asarray(contract.qpos_addresses,dtype=int)
    ref_act=Xref[:,act_qadr]
    delta=qproj[act_qadr]-Xref[0,act_qadr]
    duration=(len(Xref)-1)*.02; steps=int(round(duration/model.opt.timestep)); dt=float(model.opt.timestep)
    assert abs(steps*dt-duration)<1e-12

    actual={s:[] for s in feet}; slip={s:[] for s in feet}; clearance={s:[] for s in feet}
    previous_contact={s:None for s in feet}
    prev_geom=data.geom_xpos.copy(); root_pos=[]; root_vel=[]; root_z=[]; tilt=[]
    fall_step=None; fall_causes=[]; saturation=0; torque_n=0; saturation_by_step=[]

    def sample(step):
        nonlocal fall_step,fall_causes
        contacts=actual_contacts(model,data,floor,geom_side); clears=sole_clearances(model,data,feet,floor)
        for s in feet:
            on=bool(contacts[s]); actual[s].append(on); clearance[s].append(float(np.min(clears[s])))
            previous_contact[s]=on
            if step>0 and on:
                for g in contacts[s]: slip[s].append((step,float(np.linalg.norm(data.geom_xpos[g,:2]-prev_geom[g,:2])/dt)))
        root_pos.append(data.qpos[:3].copy()); root_vel.append(data.qvel[:2].copy())
        z=float(data.qpos[2]); ang=float(physics.root_tilt(data.qpos[3:7])); root_z.append(z); tilt.append(ang)
        if fall_step is None and (z<.42 or ang>.90):
            fall_step=step; fall_causes=[]
            if z<.42: fall_causes.append("root_z_below_0.42m")
            if ang>.90: fall_causes.append("tilt_above_0.90rad")

    sample(0)
    for k in range(steps):
        t=k*dt; phase=min(t/.02,len(ref_act)-1.0); i=int(np.floor(phase)); j=min(i+1,len(ref_act)-1); a=phase-i
        qref=(1-a)*ref_act[i]+a*ref_act[j]
        u=np.clip(t/.34,0.,1.); smooth=u*u*(3.-2.*u)
        target=qref+(1.-smooth)*delta; target[-2:]=0.0
        q=data.qpos[act_qadr]; dq=data.qvel[contract.qvel_addresses]
        raw=contract.kp*(target-q)-contract.kd*dq
        sat_k=int(np.sum((raw<contract.torque_low)|(raw>contract.torque_high)))
        saturation += sat_k; torque_n += len(raw); saturation_by_step.append(sat_k)
        data.ctrl[:]=np.clip(raw,contract.torque_low,contract.torque_high)
        mujoco.mj_step(model,data); sample(k+1); prev_geom[:]=data.geom_xpos

    time=np.arange(steps+1)*dt
    eval_end=(fall_step+1) if fall_step is not None else len(time)
    rp=np.asarray(root_pos[:eval_end]); rv=np.asarray(root_vel[:eval_end]); acc=np.diff(rv,axis=0)/dt
    anorm=np.linalg.norm(acc,axis=1) if len(acc) else np.zeros(1)
    result={
        "schema":"x2_phase6_full_continuation_v1", "method":"frozen projected reset + same 0.34s smooth bridge; no optimization/search",
        "scene_sha256":sha(SCENE), "control_sha256":sha(CONTROL), "reset_sha256":sha(RESET), "mujoco":mujoco.__version__,
        "requested_duration_s":duration, "simulated_duration_s":steps*dt,
        "survive_full_motion":fall_step is None, "fall_time_s":None if fall_step is None else fall_step*dt,
        "fall_causes":fall_causes, "fall_state":{"root_z_m":root_z[fall_step],"tilt_rad":tilt[fall_step]} if fall_step is not None else None,
        "comparison_original_free_root_fall_s":.575,
        "survival_improvement_s":None if fall_step is None else fall_step*dt-.575,
        "survival_ratio_vs_original":None if fall_step is None else fall_step*dt/.575,
        "pre_fall_samples":eval_end, "root_pre_fall":{
            "z_min_m":float(np.min(root_z[:eval_end])), "tilt_max_rad":float(np.max(tilt[:eval_end])),
            "xy_net_displacement_m":float(np.linalg.norm(rp[-1,:2]-rp[0,:2])),
            "xy_excursion_max_m":float(np.max(np.linalg.norm(rp[:,:2]-rp[0,:2],axis=1))),
            "horiz_accel_p95_mps2":float(np.percentile(anorm,95)), "horiz_accel_max_mps2":float(np.max(anorm))},
        "torque_saturation_fraction_pre_fall":float(np.sum(saturation_by_step[:max(eval_end-1,0)]))/max((eval_end-1)*model.nu,1),
        "torque_saturation_fraction_full":saturation/max(torque_n,1), "active12_contact_pre_fall":{}
    }
    for s in feet:
        arr=np.asarray(actual[s][:eval_end],bool); intent=source_intent(s,time[:eval_end])
        sl=np.asarray([v for step,v in slip[s] if step<eval_end],dtype=float)
        result["active12_contact_pre_fall"][s]={
            "sole_sphere_count":len(feet[s]), "contact_fraction":float(np.mean(arr)),
            "realized_during_intent_fraction":float(np.mean(arr[intent])) if np.any(intent) else None,
            "contradiction_during_intent_fraction":float(np.mean(~arr[intent])) if np.any(intent) else None,
            "contact_during_nonintent_fraction":float(np.mean(arr[~intent])) if np.any(~intent) else None,
            "min_clearance_mm":float(1000*np.min(clearance[s][:eval_end])),
            "slip_p95_mps":float(np.percentile(sl,95)) if len(sl) else None,
            "slip_max_mps":float(np.max(sl)) if len(sl) else None,
            "contact_switches":int(np.sum(arr[1:] != arr[:-1]))
        }
    flight=~(np.asarray(actual["left"][:eval_end])|np.asarray(actual["right"][:eval_end]))
    result["active12_contact_pre_fall"]["unintended_flight_fraction"]=float(np.mean(flight))
    OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
