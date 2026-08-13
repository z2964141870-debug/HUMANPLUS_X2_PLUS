#!/usr/bin/env python3
"""Run exactly one 50-iteration native-basin DSMS short solve and replay."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
import run_phase34_native_dsms_nlp_preflight as p34

REPO=Path(__file__).resolve().parents[2];CONTRACT=REPO/'research/dynamic_retargeting_20260811/phase35_native_dsms_short_solve_contract.json';OUTPUT=REPO/'research/dynamic_retargeting_20260811/phase35_native_dsms_short_solve_result.json';ART=REPO/'research/dynamic_retargeting_20260811/phase35_native_dsms_candidate.npz'

def longest_false(a):
 best=(0,0);start=None
 for i,v in enumerate(np.r_[np.asarray(a,bool),True]):
  if not v and start is None:start=i
  elif v and start is not None:
   if i-start>best[1]-best[0]:best=(start,i)
   start=None
 return best

def run():
 c=json.loads(CONTRACT.read_text());pre,problem,X0,U0,physical,anchor=p34.build(return_runtime=True)
 if not pre['decision']['native_contact_nlp_preflight_qualified']:raise RuntimeError('Phase34 preflight gate failed')
 t=time.perf_counter();X,U,info=problem.solve(X0,U0);wall=time.perf_counter()-t
 ends=problem.dyn.rollout_batch(X[:problem.N],U.reshape(problem.N,problem.K,problem.nu))[:,problem.K]
 defects=np.asarray([problem.dyn.state_diff(X[i+1],ends[i]) for i in range(problem.N)])
 model=problem.dyn.model;data=mujoco.MjData(model);data.qpos[:]=problem.x_init[:model.nq];data.qvel[:]=problem.x_init[model.nq:];mujoco.mj_forward(model,data)
 floor=int(model.geom('floor').id); geoms={s:p34.contact_geom_ids(model,s) for s in ('left','right')};side={g:s for s,gs in geoms.items() for g in gs};body=int(model.body('left_ankle_roll_link').id)
 contacts={s:[] for s in geoms};clear=[];left_speed=[];zs=[];tilts=[];states=[problem.x_init.copy()]
 for u in U:
  data.ctrl[:]=u;mujoco.mj_step(model,data);states.append(np.r_[data.qpos.copy(),data.qvel.copy()]);found={'left':False,'right':False}
  for j in range(data.ncon):
   row=data.contact[j];g1,g2=int(row.geom1),int(row.geom2)
   if floor in (g1,g2):
    s=side.get(g2 if g1==floor else g1)
    if s:found[s]=True
  for s in found:contacts[s].append(found[s])
  clear.append(max(0.,p34.min_sole_z(model,data,geoms['right'])));v=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,v,0);left_speed.append(float(np.linalg.norm(v[3:5])) if found['left'] else np.nan)
  zs.append(float(data.qpos[2]));tilts.append(float(p34.yaw_tilt(data.qpos[3:7])[1]))
 states=np.asarray(states);ref=problem.X_ref[:len(states)];sem=states[:,7:38]-ref[:,7:38];off=longest_false(contacts['right']);sl=np.asarray(left_speed);sl=sl[np.isfinite(sl)]
 status=int(info.get('status',-999));msg=info.get('status_msg','');msg=msg.decode() if isinstance(msg,(bytes,bytearray)) else str(msg)
 metrics={'solver_status':status,'solver_status_msg':msg,'wall_s':wall,'objective':float(info.get('obj_val',np.nan)),'max_defect':float(np.max(np.abs(defects))),
  'root_z_min_m':float(np.min(zs)),'root_tilt_max_rad':float(np.max(tilts)),'right_swing_off_s':float((off[1]-off[0])*.001),'right_clearance_m':float(np.max(clear[off[0]:off[1]])) if off[1]>off[0] else 0.,
  'left_contact_fraction':float(np.mean(contacts['left'])),'left_foot_speed_p95_mps':float(np.percentile(sl,95)) if len(sl) else None,'terminal_right_contact_40ms':bool(np.all(contacts['right'][-40:])),
  'joint_semantic_rms_rad':float(np.sqrt(np.mean(sem**2)))}
 g=c['hard_gates'];checks={'solver_status':status in (0,1),'defect':metrics['max_defect']<=g['max_defect'],'root_z':metrics['root_z_min_m']>=g['root_z_min_m'],'root_tilt':metrics['root_tilt_max_rad']<=g['root_tilt_max_rad'],
  'swing_off':metrics['right_swing_off_s']>=g['right_swing_off_s'],'clearance':metrics['right_clearance_m']>=g['right_clearance_m'],'left_contact':metrics['left_contact_fraction']>=g['left_contact_fraction'],
  'left_speed':metrics['left_foot_speed_p95_mps'] is not None and metrics['left_foot_speed_p95_mps']<=g['left_foot_speed_p95_mps'],'terminal':metrics['terminal_right_contact_40ms']==g['terminal_right_contact_40ms'],'semantic':metrics['joint_semantic_rms_rad']<=g['joint_semantic_rms_rad']}
 passed=all(checks.values());np.savez_compressed(ART,state=states,input=U,shooting_nodes=X,defects=defects,reference=ref)
 return {'stage':'Dynamic Retargeting Phase35','scope':'one native 0.34s DSMS solve; 1 CPU thread/50 iterations/no retry/no GPU/training','contract_sha256':p34.sha256(CONTRACT),'phase34_result_sha256':p34.sha256(p34.OUTPUT),'metrics':metrics,'checks':checks,
  'artifact':{'path':str(ART),'sha256':p34.sha256(ART)},'execution':{'solve_count':1,'retry_count':0,'max_iterations':50,'cpu_threads':1,'gpu':False},
  'decision':{'dynamic_teacher_qualified':passed,'longer_solve_authorized':False,'training_unlocked':False,'result':'NATIVE_SHORT_DSMS_TEACHER_PASSED' if passed else 'NATIVE_SHORT_DSMS_REJECTED_STOP'}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT);a=ap.parse_args();r=run();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'metrics':r['metrics'],'checks':r['checks'],'decision':r['decision']},indent=2))
if __name__=='__main__':main()
