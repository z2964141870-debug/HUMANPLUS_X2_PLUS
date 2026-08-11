#!/usr/bin/env python3
"""Independent reset-aware active12 SBTO; one preregistered search, no RL."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS","1")
ROOT=Path("/home/humanplus/projects/ZHY/dsms_workspace")
OUT=ROOT/"phase8_reset_sbto"; OUT.mkdir(parents=True,exist_ok=True)
PHASE3B=ROOT/"phase3b"; PHASE6=ROOT/"phase6_reset_bridge"
REPO=Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
SCENE=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
CONTROL=Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml")
RESET=PHASE6/"phase6_projected_initial_state.npz"
sys.path[:0]=[str(PHASE3B),str(PHASE6),str(REPO/"tools"),str(REPO)]
import mujoco
import retarget.run_x2_forefoot_official_physics_screen as physics
import x2_lunge_dsms_phase3b_corrected as base
from x2_lunge_phase6_reset_bridge import collidable_sole_spheres,actual_contacts,sole_clearances

SEED=20260811; CP_FRAMES=np.array([0,8,18,27,35],float); WINDOWS=[14,25,36]
CANDIDATES=40; ELITES=8; ITERS=4; HORIZON=.70
NATIVE={"minimum_mm":-4.487,"p01_mm":{"left":-2.116,"right":-2.491},"p05_mm":{"left":-.683,"right":-.827}}

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def longest_true_ms(x):
 best=cur=0
 for v in np.asarray(x,bool): cur=cur+1 if v else 0; best=max(best,cur)
 return best

class Runner:
 def __init__(self):
  _,_,_,self.Xref,self.c=base.load_reference(10000)
  z=np.load(RESET,allow_pickle=False); self.q0=np.asarray(z["qpos"],float); self.v0=np.asarray(z["qvel"],float)
  self.m=mujoco.MjModel.from_xml_path(str(SCENE)); self.d=mujoco.MjData(self.m)
  if sha(RESET)!="141c6d7d9c006634eb00ddf25d2f593080abd0d8cfee8376767c14674d8cc2ea": raise RuntimeError("reset hash changed")
  if mujoco.__version__!="3.3.7" or int(self.m.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_EULER): raise RuntimeError("physics contract changed")
  self.floor,helper=physics.foot_geom_contract(self.m); self.feet=collidable_sole_spheres(self.m,helper)
  if any(len(v)!=12 for v in self.feet.values()): raise RuntimeError("active12 contract changed")
  self.side={g:s for s,gs in self.feet.items() for g in gs}; self.aq=np.asarray(self.c.qpos_addresses); self.av=np.asarray(self.c.qvel_addresses)
  self.ref=self.Xref[:,self.aq]; self.delta=self.q0[self.aq]-self.Xref[0,self.aq]; self.opt=np.arange(self.m.nu-2)
  names=self.c.actuator_joint_names[:-2]
  self.scale=np.array([.10 if any(k in n for k in ("hip","knee","ankle")) else .05 if "waist" in n else .06 for n in names])
  self.bound=np.array([.25 if any(k in n for k in ("hip","knee","ankle")) else .15 for n in names])
 def reset(self):
  mujoco.mj_resetData(self.m,self.d); self.d.qpos[:]=self.q0; self.d.qvel[:]=self.v0; mujoco.mj_forward(self.m,self.d)
 def nominal(self,t):
  phase=min(t/.02,len(self.ref)-1.); i=int(np.floor(phase)); j=min(i+1,len(self.ref)-1); a=phase-i
  q=(1-a)*self.ref[i]+a*self.ref[j]; u=np.clip(t/.34,0.,1.); smooth=u*u*(3-2*u)
  out=q+(1-smooth)*self.delta; out[-2:]=0.; return out
 def correction(self,cp,t):
  frame=t/.02
  if frame<=CP_FRAMES[-1]: return np.array([np.interp(frame,CP_FRAMES,cp[:,j]) for j in range(len(self.opt))])
  if t<=1.: return cp[-1]*max(0.,1-(t-HORIZON)/.30)
  return np.zeros(len(self.opt))
 def run(self,cp,duration,full_metrics=True,stop_at_fall=False):
  self.reset(); dt=float(self.m.opt.timestep); steps=int(round(duration/dt)); prev=self.d.geom_xpos.copy()
  init={s:np.mean(self.d.geom_xpos[sorted(gs),:2],axis=0) for s,gs in self.feet.items()}
  actual={s:[] for s in self.feet}; clear={s:[] for s in self.feet}; slip={s:[] for s in self.feet}; exc={s:[] for s in self.feet}
  rv=[self.d.qvel[:2].copy()]; rz=[float(self.d.qpos[2])]; tilt=[physics.root_tilt(self.d.qpos[3:7])]; qerr=[]; sat=0; nt=0; fall=None
  def sample(k):
   con=actual_contacts(self.m,self.d,self.floor,self.side); cl=sole_clearances(self.m,self.d,self.feet,self.floor)
   for s,gs in self.feet.items():
    actual[s].append(bool(con[s])); clear[s].append(float(np.min(cl[s])))
    cen=np.mean(self.d.geom_xpos[sorted(gs),:2],axis=0); exc[s].append(float(np.linalg.norm(cen-init[s])))
    if k>0:
     for g in con[s]: slip[s].append(float(np.linalg.norm(self.d.geom_xpos[g,:2]-prev[g,:2])/dt))
  sample(0)
  for k in range(steps):
   t=k*dt; target=self.nominal(t); target[:-2]+=self.correction(cp,t); target[-2:]=0
   q=self.d.qpos[self.aq]; dq=self.d.qvel[self.av]; raw=self.c.kp*(target-q)-self.c.kd*dq
   sat+=int(np.sum((raw<self.c.torque_low)|(raw>self.c.torque_high))); nt+=len(raw)
   self.d.ctrl[:]=np.clip(raw,self.c.torque_low,self.c.torque_high); mujoco.mj_step(self.m,self.d); sample(k+1); prev[:]=self.d.geom_xpos
   rv.append(self.d.qvel[:2].copy()); rz.append(float(self.d.qpos[2])); tilt.append(physics.root_tilt(self.d.qpos[3:7])); qerr.append(float(np.mean((q-self.nominal(t))**2)))
   if fall is None and (self.d.qpos[2]<.42 or tilt[-1]>.90):
    fall=(k+1)*dt
    if stop_at_fall: break
  simulated_steps=len(rv)-1; tt=np.arange(simulated_steps+1)*dt; intent={"left":tt<=10/30+1e-12,"right":tt<=25/30+1e-12}
  act={s:np.asarray(v,bool) for s,v in actual.items()}; nonleft=~intent["left"]; right_support=act["right"][nonleft]
  switch=(~act["left"] & act["right"] & nonleft); longest=longest_true_ms(switch)
  acc=np.linalg.norm(np.diff(np.asarray(rv),axis=0)/dt,axis=1)
  # Penetration is deliberately absent here. It is report-only below.
  support_miss=sum(np.mean(~act[s][intent[s]]) for s in self.feet if np.any(intent[s]))
  stuck=float(np.mean(act["left"][nonleft])) if np.any(nonleft) else 1.
  left_clear=np.asarray(clear["left"])[nonleft]; lift=float(np.mean((np.maximum(0.,.005-left_clear)/.01)**2)) if len(left_clear) else 1.
  flight=float(np.mean(~(act["left"]|act["right"])))
  speed=sum((max(0.,(np.percentile(slip[s],95) if slip[s] else 1.)-.10)/.20)**2 for s in self.feet)
  accel=(max(0.,np.percentile(acc,95)-4)/10)**2
  loss=25*support_miss+30*stuck+15*lift+40*flight+5*speed+4*accel+2*np.mean(qerr)+.2*np.mean((cp/self.scale)**2)+2*sat/max(nt,1)+(200 if fall is not None and fall<duration-1e-12 else 0)
  out={"loss":float(loss),"duration_s":duration,"simulated_duration_s":simulated_steps*dt,"fall_time_s":fall,"survive_requested":fall is None or fall>=duration-1e-12,
   "root_z_min_m":min(rz),"tilt_max_rad":max(tilt),"torque_saturation_fraction":sat/max(nt,1),"head_q_max_abs_rad":float(np.max(np.abs(self.d.qpos[self.aq[-2:]]))),
   "root_horiz_accel_p95_max_mps2":[float(np.percentile(acc,95)),float(np.max(acc))],"penetration_min_mm":{s:1000*min(clear[s]) for s in self.feet},"contact":{},
   "left_nonintent_contact_fraction":stuck,"right_support_during_left_nonintent_fraction":float(np.mean(right_support)) if len(right_support) else None,
   "left_off_right_on_longest_ms":longest,"unintended_flight_fraction":flight}
  for s in self.feet:
   out["contact"][s]={"intent_realized_fraction":float(np.mean(act[s][intent[s]])) if np.any(intent[s]) else None,
    "intent_contradiction_fraction":float(np.mean(~act[s][intent[s]])) if np.any(intent[s]) else None,
    "stance_speed_p95_mps":float(np.percentile(slip[s],95)) if slip[s] else None,"stance_excursion_max_m":float(np.max(np.asarray(exc[s])[intent[s]])) if np.any(intent[s]) else None,
    "switches":int(np.sum(act[s][1:]!=act[s][:-1]))}
  return out

def verdict(x,horizon=True):
 legacy=min(x["penetration_min_mm"].values())>=-.5; native=min(x["penetration_min_mm"].values())>=NATIVE["minimum_mm"]
 g={"survive":x["survive_requested"],"liftoff_30ms":x["left_off_right_on_longest_ms"]>=30,"left_stuck":x["left_nonintent_contact_fraction"]<=.10,
  "right_support_conversion":x["right_support_during_left_nonintent_fraction"] is not None and x["right_support_during_left_nonintent_fraction"]>=.95,
  "stance_speed":all(x["contact"][s]["stance_speed_p95_mps"] is not None and x["contact"][s]["stance_speed_p95_mps"]<=.10 for s in self_sides(x)),
  "stance_excursion":all(x["contact"][s]["stance_excursion_max_m"]<=.03 for s in self_sides(x)),"flight":x["unintended_flight_fraction"]<=.02,
  "root_accel":x["root_horiz_accel_p95_max_mps2"][0]<=4.,"head":x["head_q_max_abs_rad"]<=.02,"native_calibrated_penetration":native}
 return {"primary":g,"primary_pass":all(g.values()),"legacy_absolute_penetration_ge_minus_0p5mm":legacy,"penetration_in_search_loss":False}

def self_sides(x): return ("left","right")

def preflight():
 r=Runner(); zero=np.zeros((5,29)); a=r.run(zero,.70); b=r.run(zero,.70); full=r.run(zero,1.0)
 start=r.run(zero,.001)
 c={"reset_sha256":sha(RESET),"scene_sha256":sha(SCENE),"control_sha256":sha(CONTROL),"mujoco":mujoco.__version__,"official_euler":int(r.m.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_EULER),
  "active_spheres":{s:len(gs) for s,gs in r.feet.items()},"zero_correction_max_abs":0.,"deterministic_loss_delta":abs(a["loss"]-b["loss"]),"nominal_fall_time_s":full["fall_time_s"],
  "initial_contact":{"left":start["contact"]["left"]["intent_realized_fraction"],"right":start["contact"]["right"]["intent_realized_fraction"]},"penetration_in_search_loss":False}
 c["pass"]=bool(c["reset_sha256"].startswith("141c6d") and c["official_euler"] and all(v==12 for v in c["active_spheres"].values()) and c["deterministic_loss_delta"]==0 and abs(c["nominal_fall_time_s"]-.683)<=.002)
 (OUT/"phase8_preflight.json").write_text(json.dumps(c,indent=2)+"\n"); print(json.dumps(c,indent=2)); return c

def search():
 r=Runner(); rng=np.random.default_rng(SEED); mean=np.zeros((5,29)); std=np.tile(r.scale,(5,1)); history=[]; final_best=None; t0=time.perf_counter(); nominal=r.run(mean,HORIZON)
 for frames in WINDOWS:
  active=max(2,int(np.searchsorted(CP_FRAMES,frames-1,side="right")))
  for it in range(ITERS):
   samples=np.repeat(mean[None],CANDIDATES,axis=0); samples[0]=mean; samples[1:,:active]=mean[:active]+rng.normal(size=(CANDIDATES-1,active,29))*std[:active]
   samples[:,:active]=np.clip(samples[:,:active],-r.bound,r.bound); duration=(frames-1)*.02
   vals=[r.run(x,duration) for x in samples]; scores=np.asarray([v["loss"] for v in vals]); elite=np.argsort(scores)[:ELITES]
   mean[:active]=np.mean(samples[elite,:active],axis=0); std[:active]=np.maximum(np.std(samples[elite,:active],axis=0),.008)
   history.append({"window_frames":frames,"iteration":it+1,"best_loss":float(scores[elite[0]]),"mean_loss":float(np.mean(scores)),"active_control_points":active}); print(json.dumps(history[-1]),flush=True)
   if frames==WINDOWS[-1] and it==ITERS-1: final_best=samples[elite[0]].copy()
 candidate=r.run(final_best,HORIZON); candidate["verdict"]=verdict(candidate); nominal["verdict"]=verdict(nominal)
 full=r.run(final_best,(len(r.Xref)-1)*.02); full["fall_improvement_vs_phase6_s"]=None if full["fall_time_s"] is None else full["fall_time_s"]-.683
 np.savez_compressed(OUT/"phase8_reset_sbto_candidate.npz",correction=final_best,control_point_frames=CP_FRAMES,reset_qpos=r.q0,reset_qvel=r.v0)
 out={"schema":"x2_phase8_reset_sbto_result_v1","wall_s":time.perf_counter()-t0,"configuration":{"seed":SEED,"windows":WINDOWS,"candidates":CANDIDATES,"elites":ELITES,"iterations":ITERS},
  "native_penetration_calibration":NATIVE,"nominal_0p70":nominal,"candidate_0p70":candidate,"candidate_full_continuation":full,"history":history,"search_reruns":0,"rl_run":False,"stopped":True}
 (OUT/"phase8_result.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"wall_s":out["wall_s"],"candidate":candidate,"full_fall":full["fall_time_s"],"improvement":full["fall_improvement_vs_phase6_s"]},indent=2))

def audit_prefall():
 z=np.load(OUT/"phase8_reset_sbto_candidate.npz",allow_pickle=False); cp=np.asarray(z["correction"],float); r=Runner()
 pre=r.run(cp,(len(r.Xref)-1)*.02,stop_at_fall=True); pre["fall_improvement_vs_phase6_s"]=None if pre["fall_time_s"] is None else pre["fall_time_s"]-.683
 audit={"schema":"x2_phase8_candidate_prefall_audit_v1","search_rerun":False,"deterministic_replay":1,"candidate_sha256":sha(OUT/"phase8_reset_sbto_candidate.npz"),"metrics_until_first_fall":pre}
 (OUT/"phase8_candidate_prefall_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
 p=OUT/"phase8_result.json"; result=json.loads(p.read_text()); result["candidate_full_continuation_prefall_audit"]=pre; result["postfall_full_metrics_truth_boundary"]="candidate_full_continuation contact/slip fields include post-fall motion and are non-causal; use prefall audit"
 p.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(audit,indent=2))

if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("preflight","search","audit")); a=ap.parse_args(); preflight() if a.mode=="preflight" else audit_prefall() if a.mode=="audit" else search()
