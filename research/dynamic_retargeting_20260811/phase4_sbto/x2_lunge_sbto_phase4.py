#!/usr/bin/env python3
"""One preregistered progressive-window SBTO/CEM run on raw official X2."""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path
import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace/phase4_sbto")
REPO = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
CONTROL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml")
LUNGE = Path("/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/silver_contact/phase30_time_dilation/x2_phase30_time_dilation_1p46.pkl")
import sys
sys.path[:0] = [str(REPO / "tools"), str(REPO)]
import retarget.run_x2_forefoot_official_physics_screen as physics

SEED = 20260811
CP_FRAMES = np.array([0, 4, 9, 13, 17], dtype=np.float64)
WINDOWS = [7, 12, 18]
CANDIDATES, ELITES, ITERS = 40, 8, 4


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_ref():
    e = joblib.load(LUNGE)["PHUMA-LUNGE-R-001"]
    fps0, fps = float(e["fps"]), 50.0
    q0 = np.asarray(e["dof"], float); p0 = np.asarray(e["root_trans_offset"], float)
    r0 = np.asarray(e["root_rot"], float); duration = (len(q0)-1)/fps0
    t0 = np.arange(len(q0))/fps0; n = int(round(duration*fps))+1; t = np.linspace(0,duration,n)
    q = np.column_stack([np.interp(t,t0,q0[:,i]) for i in range(q0.shape[1])])
    p = np.column_stack([np.interp(t,t0,p0[:,i]) for i in range(3)])
    r = Slerp(t0, Rotation.from_quat(r0))(t).as_quat()
    dq = np.gradient(q, 1/fps, axis=0, edge_order=1); dp = np.gradient(p,1/fps,axis=0,edge_order=1)
    dr = np.zeros((n,3)); inc=(Rotation.from_quat(r[:-1]).inv()*Rotation.from_quat(r[1:])).as_rotvec()*fps
    dr[0],dr[-1]=inc[0],inc[-1]
    if n>2: dr[1:-1]=.5*(inc[:-1]+inc[1:])
    return {"q":q,"dq":dq,"p":p,"r":r,"dp":dp,"dr":dr,"names":list(e["joint_names_mujoco"]),"fps":fps}


class Runner:
    def __init__(self, ref):
        self.ref=ref; self.m=mujoco.MjModel.from_xml_path(str(SCENE)); self.d=mujoco.MjData(self.m)
        self.c=physics.build_control_contract(self.m, CONTROL)
        if mujoco.__version__ != "3.3.7" or int(self.m.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_EULER):
            raise RuntimeError("frozen MuJoCo/Euler contract changed")
        if tuple(ref["names"]) != self.c.joint_names: raise RuntimeError("joint order mismatch")
        entry_index={name:i for i,name in enumerate(ref["names"])}
        self.act_ref_idx=np.asarray([entry_index[name] for name in self.c.actuator_joint_names],dtype=np.int64)
        self.floor,self.feet=physics.foot_geom_contract(self.m); self.side={g:s for s,gs in self.feet.items() for g in gs}
        # Clearance is meaningful only for collision-active sole spheres.
        # foot_geom_contract also returns one contype=0 visual mesh per foot;
        # treating its mesh geom_size[0] as a sphere radius polluted the old
        # penetration auditor without affecting physics or contacts.
        self.penetration_feet={s:{g for g in gs if int(self.m.geom_contype[g]) != 0} for s,gs in self.feet.items()}
        if any(len(gs) != 12 for gs in self.penetration_feet.values()):
            raise RuntimeError(f"active sole sphere contract changed: {self.penetration_feet}")
        self.opt_idx=np.arange(self.m.nu-2); self.steps_per_frame=20
        self.initial_qpos=self._make_qpos(0); self.initial_qvel=self._make_qvel(0)
        self.cp_scale=np.array([.10 if any(x in n for x in ("hip","knee","ankle")) else .05 if "waist" in n else .06 for n in self.c.actuator_joint_names[:-2]])
        self.cp_bound=np.array([.25 if any(x in n for x in ("hip","knee","ankle")) else .15 for n in self.c.actuator_joint_names[:-2]])

    def _make_qpos(self,i):
        x=np.zeros(self.m.nq); x[:3]=self.ref["p"][i]; x[3:7]=self.ref["r"][i][[3,0,1,2]]; x[self.c.qpos_addresses]=self.ref["q"][i,self.act_ref_idx]; return x
    def _make_qvel(self,i):
        x=np.zeros(self.m.nv); x[:3]=self.ref["dp"][i]; x[3:6]=self.ref["dr"][i]; x[self.c.qvel_addresses]=self.ref["dq"][i,self.act_ref_idx]; return x
    def reset(self):
        mujoco.mj_resetData(self.m,self.d); self.d.qpos[:]=self.initial_qpos; self.d.qvel[:]=self.initial_qvel; mujoco.mj_forward(self.m,self.d)
    def targets(self, cp, frames):
        corr=np.column_stack([np.interp(np.arange(frames),CP_FRAMES,cp[:,j]) for j in range(len(self.opt_idx))])
        out=self.ref["q"][:frames,self.act_ref_idx].copy(); out[:,:-2]+=corr; out[:,-2:]=0
        ranges=np.array([self.m.jnt_range[self.m.actuator_trnid[a,0]] for a in range(self.m.nu)])
        return np.clip(out,ranges[:,0],ranges[:,1])
    def run(self, cp, frames, detail=False):
        self.reset(); targets=self.targets(cp,frames); steps=(frames-1)*20
        prev=self.d.geom_xpos.copy(); init={s:np.mean(self.d.geom_xpos[list(gs),:2],axis=0) for s,gs in self.feet.items()}
        realized={s:[] for s in self.feet}; slips={s:[] for s in self.feet}; excursions={s:[] for s in self.feet}; clear={s:np.inf for s in self.feet}
        rootv=[self.d.qvel[:2].copy()]; rootz=[float(self.d.qpos[2])]; tilts=[physics.root_tilt(self.d.qpos[3:7])]
        sat=0; effort=0; qerr=[]; rooterr=[]; fell=None
        for k in range(steps):
            fi=min(k//20,frames-1); target=targets[fi]; q=self.d.qpos[self.c.qpos_addresses]; dq=self.d.qvel[self.c.qvel_addresses]
            raw=self.c.kp*(target-q)-self.c.kd*dq; sat+=int(np.sum((raw<self.c.torque_low)|(raw>self.c.torque_high))); effort+=len(raw)
            self.d.ctrl[:]=np.clip(raw,self.c.torque_low,self.c.torque_high); mujoco.mj_step(self.m,self.d)
            contacts={s:set() for s in self.feet}
            for ci in range(self.d.ncon):
                cc=self.d.contact[ci]; a,b=int(cc.geom1),int(cc.geom2); other=b if a==self.floor else a if b==self.floor else -1
                if other in self.side: contacts[self.side[other]].add(other)
            for s,gs in self.feet.items():
                realized[s].append(bool(contacts[s]))
                for g in contacts[s]: slips[s].append(float(np.linalg.norm(self.d.geom_xpos[g,:2]-prev[g,:2])/.001))
                cen=np.mean(self.d.geom_xpos[list(gs),:2],axis=0); excursions[s].append(float(np.linalg.norm(cen-init[s])))
                for g in self.penetration_feet[s]: clear[s]=min(clear[s],float(self.d.geom_xpos[g,2]-self.m.geom_size[g,0]-self.d.geom_xpos[self.floor,2]))
            prev[:]=self.d.geom_xpos; rootv.append(self.d.qvel[:2].copy()); rootz.append(float(self.d.qpos[2])); tilts.append(physics.root_tilt(self.d.qpos[3:7]))
            qerr.append(float(np.mean((q-self.ref["q"][fi,self.act_ref_idx])**2))); rooterr.append(float(np.sum((self.d.qpos[:2]-self.ref["p"][fi,:2])**2)))
            if fell is None and (self.d.qpos[2]<.42 or tilts[-1]>.90): fell=(k+1)*.001
        actual={s:np.asarray(v,bool) for s,v in realized.items()}; flight=~(actual["left"]|actual["right"])
        rv=np.asarray(rootv); acc=np.linalg.norm(np.diff(rv,axis=0)/.001,axis=1)
        intended={"left":(np.arange(1,steps+1)*.001<=10/30+1e-12),"right":(np.arange(1,steps+1)*.001<=25/30+1e-12)}
        # Fixed ranking loss; the hard gates below remain the only verdict.
        miss=sum(np.mean(~actual[s][intended[s]]) for s in self.feet)
        pen=sum((max(0.,-.0005-clear[s])/.01)**2 for s in self.feet)
        speed=sum((max(0.,(np.percentile(slips[s],95) if slips[s] else 1.)-.10)/.20)**2 for s in self.feet)
        exc=sum((max(0.,np.max(np.asarray(excursions[s])[intended[s]])-.03)/.03)**2 for s in self.feet)
        accel=(max(0.,np.percentile(acc,95)-4)/10)**2
        loss=20*miss+30*np.mean(flight)+20*pen+5*speed+3*exc+3*accel+2*np.mean(qerr)+5*np.mean(rooterr)+.2*np.mean((cp/self.cp_scale)**2)+(100 if fell is not None else 0)
        metrics={"loss":float(loss),"duration_s":steps*.001,"fall_time_s":fell,"survive_full_prefix":fell is None,
          "root_z_min_m":min(rootz),"tilt_max_rad":max(tilts),"torque_saturation_fraction":sat/max(effort,1),
          "penetration_min_mm":{s:1000*clear[s] for s in self.feet},"unintended_flight_fraction":float(np.mean(flight)),
          "root_horiz_accel_p95_max_mps2":[float(np.percentile(acc,95)),float(np.max(acc))],
          "head_q_max_abs_rad":float(np.max(np.abs(self.d.qpos[self.c.qpos_addresses[-2:]]))),"contact":{}}
        for s in self.feet:
            metrics["contact"][s]={"realized_during_intent_fraction":float(np.mean(actual[s][intended[s]])),"contradiction_fraction":float(np.mean(~actual[s][intended[s]])),
              "stance_speed_p95_mps":float(np.percentile(slips[s],95)) if slips[s] else None,"stance_excursion_max_m":float(np.max(np.asarray(excursions[s])[intended[s]]))}
        return metrics

    def baseline(self):
        self.reset(); n=(len(self.ref["q"])-1)*20; fell=None
        for k in range(n):
            fi=k//20; q=self.d.qpos[self.c.qpos_addresses]; dq=self.d.qvel[self.c.qvel_addresses]
            raw=self.c.kp*(self.ref["q"][fi,self.act_ref_idx]-q)-self.c.kd*dq; self.d.ctrl[:]=np.clip(raw,self.c.torque_low,self.c.torque_high); mujoco.mj_step(self.m,self.d)
            if self.d.qpos[2]<.42 or physics.root_tilt(self.d.qpos[3:7])>.90: fell=(k+1)*.001; break
        return fell


def gates(m):
    g={"survive_full_prefix":m["survive_full_prefix"],"penetration":min(m["penetration_min_mm"].values())>=-.5,
       "stance_speed":all(m["contact"][s]["stance_speed_p95_mps"] is not None and m["contact"][s]["stance_speed_p95_mps"]<=.10 for s in ("left","right")),
       "stance_excursion":all(m["contact"][s]["stance_excursion_max_m"]<=.03 for s in ("left","right")),
       "flight":m["unintended_flight_fraction"]<=.02,"root_accel":m["root_horiz_accel_p95_max_mps2"][0]<=4.,"head":m["head_q_max_abs_rad"]<=.02}
    return g,all(g.values())


def preflight():
    ref=load_ref(); r=Runner(ref); zero=np.zeros((len(CP_FRAMES),len(r.opt_idx)))
    a=r.run(zero,18); b=r.run(zero,18); baseline=r.baseline()
    check={"scene_sha256":sha(SCENE),"control_sha256":sha(CONTROL),"motion_sha256":sha(LUNGE),"mujoco":mujoco.__version__,
      "official_euler":int(r.m.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_EULER),"reference_roundtrip_max_abs":float(np.max(np.abs(r.initial_qpos[r.c.qpos_addresses]-ref["q"][0,r.act_ref_idx]))),
      "zero_correction_max_abs":float(np.max(np.abs(zero))),"deterministic_score_max_abs":abs(a["loss"]-b["loss"]),"baseline_fall_time_s":baseline}
    check["pass"]=bool(check["official_euler"] and check["reference_roundtrip_max_abs"]<=1e-12 and check["zero_correction_max_abs"]==0 and check["deterministic_score_max_abs"]<=1e-12 and baseline is not None and abs(baseline-.575)<=.02)
    (ROOT/"phase4_sbto_preflight.json").write_text(json.dumps(check,indent=2)+"\n"); print(json.dumps(check,indent=2)); return check


def search():
    ref=load_ref(); r=Runner(ref); rng=np.random.default_rng(SEED); mean=np.zeros((len(CP_FRAMES),len(r.opt_idx))); std=np.tile(r.cp_scale,(len(CP_FRAMES),1)); history=[]; t0=time.perf_counter()
    nominal=r.run(mean,18)
    for frames in WINDOWS:
        active=int(np.searchsorted(CP_FRAMES,frames-1,side="right")); active=max(active,2)
        for it in range(ITERS):
            samples=np.repeat(mean[None],CANDIDATES,axis=0); samples[0]=mean
            samples[1:,:active]=mean[:active]+rng.normal(size=(CANDIDATES-1,active,len(r.opt_idx)))*std[:active]
            samples[:,:active]=np.clip(samples[:,:active],-r.cp_bound,r.cp_bound)
            scores=np.array([r.run(x,frames)["loss"] for x in samples]); elite=np.argsort(scores)[:ELITES]
            mean[:active]=np.mean(samples[elite,:active],axis=0); std[:active]=np.maximum(np.std(samples[elite,:active],axis=0),.008)
            history.append({"window_frames":frames,"iteration":it+1,"best_loss":float(scores[elite[0]]),"mean_loss":float(np.mean(scores)),"active_control_points":active})
            print(json.dumps(history[-1]),flush=True)
    final=r.run(mean,18,True); gg,passed=gates(final); final["gates"]=gg; final["pass"]=passed
    np.savez_compressed(ROOT/"x2_lunge_phase4_sbto_prefix.npz",control_point_frames=CP_FRAMES,correction=mean,targets=r.targets(mean,18),reference_q_actuator_order=ref["q"][:18,r.act_ref_idx],actuator_joint_names=np.asarray(r.c.actuator_joint_names))
    result={"schema":"x2_phase4_sbto_result_v1","seed":SEED,"configuration":{"windows":WINDOWS,"candidates":CANDIDATES,"elites":ELITES,"iterations":ITERS,"control_points":CP_FRAMES.tolist()},
      "wall_s":time.perf_counter()-t0,"nominal":nominal,"history":history,"final_raw_replay":final,"pass":passed,"stopped":True,"rl_run":False}
    (ROOT/"phase4_sbto_result.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"wall_s":result["wall_s"],"pass":passed,"gates":gg,"metrics":final},indent=2))


def audit_penetration():
    """Zero-search deterministic replay correcting only the sole geom set."""
    result_path=ROOT/"phase4_sbto_result.json"; artifact_path=ROOT/"x2_lunge_phase4_sbto_prefix.npz"
    old_bytes=result_path.read_bytes(); old=json.loads(old_bytes); z=np.load(artifact_path,allow_pickle=True)
    correction=np.asarray(z["correction"],float); ref=load_ref(); r=Runner(ref)
    nominal=r.run(np.zeros_like(correction),18,True); final=r.run(correction,18,True)
    gg,passed=gates(final); final["gates"]=gg; final["pass"]=passed
    # Everything except penetration must remain bitwise/numerically invariant.
    def strip_pen(x):
        y=json.loads(json.dumps(x)); y.pop("penetration_min_mm",None); y.pop("loss",None); y.pop("gates",None); y.pop("pass",None); return y
    if strip_pen(old["nominal"]) != strip_pen(nominal) or strip_pen(old["final_raw_replay"]) != strip_pen(final):
        raise RuntimeError("auditor-only replay changed a non-penetration metric")
    correction_record={
      "schema":"x2_phase4_sbto_penetration_audit_v1","date":"2026-08-11",
      "reason":"foot_geom_contract included one contype=0 visual mesh per foot; old auditor treated geom_size[0] as sphere radius",
      "physics_search_rerun":False,"cem_rerun":False,"deterministic_replays":2,
      "result_pre_audit_sha256":hashlib.sha256(old_bytes).hexdigest(),"artifact_sha256":sha(artifact_path),
      "active_geom_counts":{s:len(gs) for s,gs in r.penetration_feet.items()},
      "active_geom_ids":{s:sorted(map(int,gs)) for s,gs in r.penetration_feet.items()},
      "excluded_geom_ids":{s:sorted(map(int,set(r.feet[s])-set(r.penetration_feet[s]))) for s in r.feet},
      "penetration_pre_mm":{"nominal":old["nominal"]["penetration_min_mm"],"candidate":old["final_raw_replay"]["penetration_min_mm"]},
      "penetration_corrected_mm":{"nominal":nominal["penetration_min_mm"],"candidate":final["penetration_min_mm"]},
      "scalar_loss_pre":{"nominal":old["nominal"]["loss"],"candidate":old["final_raw_replay"]["loss"]},
      "scalar_loss_recomputed_with_correct_auditor":{"nominal":nominal["loss"],"candidate":final["loss"]},
      "corrected_gates":gg,"corrected_pass":passed,
      "non_penetration_metrics_exact":True}
    old["nominal_pre_penetration_audit"]=old["nominal"]
    old["final_raw_replay_pre_penetration_audit"]=old["final_raw_replay"]
    old["nominal"]=nominal; old["final_raw_replay"]=final; old["pass"]=passed
    old["provenance_correction"]=correction_record
    (ROOT/"phase4_sbto_penetration_audit.json").write_text(json.dumps(correction_record,indent=2)+"\n")
    result_path.write_text(json.dumps(old,indent=2)+"\n")
    print(json.dumps(correction_record,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("preflight","search","audit")); a=ap.parse_args()
    preflight() if a.mode=="preflight" else audit_penetration() if a.mode=="audit" else search()
