#!/usr/bin/env python3
"""Build the native short-horizon DSMS NLP without invoking IPOPT."""
from __future__ import annotations
import argparse, ctypes, dataclasses, hashlib, json, mmap, os, sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]; UP=Path('/home/humanplus/projects/ZHY/dsms_workspace/shooting-for-contact')
sys.path[:0]=[str(UP),str(UP/'examples/g1_gait'),str(REPO/'tools'),str(REPO)]
os.environ.setdefault('TRAJOPT_ROOT_DIR',str(UP))
import mujoco
import research.dynamic_retargeting_20260811.phase3b.x2_lunge_dsms_phase3b_corrected as b
from src.dynamics import DynamicsConfig
from src.multi_shooting import MultiShootingConfig
from src.spline import SplineConfig
from examples.g1_gait.g1_gait import G1GaitTO
from official_x2.decode_phase32_mujoco_trace import FileHeader,StateRecord
from official_x2.replay_official_trace_direct_mujoco import JOINTS,pd_gains,yaw_tilt

P32=REPO/'research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_result.json'
P33B=REPO/'research/dynamic_retargeting_20260811/phase33b_native_pd_spline_result.json'
CONTRACT=REPO/'research/dynamic_retargeting_20260811/phase34_native_dsms_nlp_contract.json'
OUTPUT=REPO/'research/dynamic_retargeting_20260811/phase34_native_dsms_nlp_result.json'
STEPS=340; NODES=17; KNOTS=18

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()

def read_trace(path):
    with Path(path).open('rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);header=FileHeader.from_buffer_copy(mm[:ctypes.sizeof(FileHeader)])
        last=-1
        for i in range(int(header.committed_records)):
            row=StateRecord.from_buffer_copy(mm,header.header_size+i*header.record_size)
            if row.call_kind==1:last=int(row.sequence)
        qpos=[];qvel=[];ctrl=[]
        for i in range(int(header.committed_records)):
            row=StateRecord.from_buffer_copy(mm,header.header_size+i*header.record_size)
            if row.call_kind==3 and row.sequence>last:
                qpos.append(np.asarray(row.qpos[:row.nq],float));qvel.append(np.asarray(row.qvel[:row.nv],float));ctrl.append(np.asarray(row.ctrl[:row.nu],float))
        mm.close()
    return {'qpos':np.asarray(qpos),'qvel':np.asarray(qvel),'ctrl':np.asarray(ctrl)}

def contact_geom_ids(model,side):
    body=int(model.body(f'{side}_ankle_roll_link').id)
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g])==body and int(model.geom_contype[g])!=0]

def min_sole_z(model,data,geoms):
    return float(min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in geoms))

@dataclasses.dataclass
class Cost:
    w_base_pos:float=40.; w_base_ori:float=10.; w_base_linvel:float=1.; w_base_angvel:float=1.
    w_joint_pos:float=.1; w_joint_vel:float=.01; term_scale:float=10.; Rtau:float=1e-7; Rrate:float=1e-2
    exact_vel_grad:bool=False; ee_anchor:str='torso_link'
    def ee_track_expanded(self):
        return ((1,'pelvis',10.,10.,1.,.01,.01,False),(1,'left_ankle_roll_link',100.,300.,1.,20.,.01,False),(1,'right_ankle_roll_link',100.,300.,1.,20.,.01,False),(1,'left_wrist_yaw_link',1.,1.,1.,.01,.01,True),(1,'right_wrist_yaw_link',1.,1.,1.,.01,.01,True))

def curve(t):
    def s(x): x=float(np.clip(x,0,1)); return x*x*(3-2*x)
    if t<.08:return .00025
    if t<.14:return .00025+(.012-.00025)*s((t-.08)/.06)
    if t<=.26:return .012
    return .012+(.00025-.012)*s((t-.26)/.08)

def build(return_runtime=False):
    p32=json.loads(P32.read_text()); p33=json.loads(P33B.read_text()); c=json.loads(CONTRACT.read_text())
    if not p32['decision']['native_dsms_warmstart_qualified'] or p33['decision']['pd_spline_warmstart_qualified']:
        raise RuntimeError('Phase34 preflight requires Phase32 pass and Phase33b reference-matching rejection')
    paths={k:Path(v['path']) for k,v in p32['assets'].items()}; physical=read_trace(paths['mmap']); anchor=int(p32['selection']['anchor_physics_index'])
    raw=mujoco.MjModel.from_xml_path(str(paths['scene'])); qadr=[];dadr=[];act=[]
    for name in JOINTS:
        jid=int(raw.joint(name).id);qadr.append(int(raw.jnt_qposadr[jid]));dadr.append(int(raw.jnt_dofadr[jid]));act.append(int(raw.actuator(f'motor_{name}').id))
    qadr=np.asarray(qadr);dadr=np.asarray(dadr);act=np.asarray(act);g=pd_gains();kp=np.asarray([g[n][0] for n in JOINTS]);kd=np.asarray([g[n][1] for n in JOINTS])
    Xref=np.column_stack([physical['qpos'][anchor:anchor+STEPS+1:20],physical['qvel'][anchor:anchor+STEPS+1:20]])
    targets=np.asarray([physical['qpos'][anchor+i,qadr]+(physical['ctrl'][anchor+1+i,act]+kd*physical['qvel'][anchor+i,dadr])/kp for i in range(STEPS)])
    dyn=DynamicsConfig(model_path=str(paths['scene']),sim_dt=.001,integrator=None,actuator_mode='position',n_threads=1,fd_eps=1e-6,fd_centered=True,friction_cone=None)
    ms=MultiShootingConfig(N=NODES,node_dt=.02,spline=SplineConfig(M=KNOTS,spline_type='linear'),ipopt_options={'linear_solver':'mumps','max_iter':50,'print_level':5,'hessian_approximation':'limited-memory'},keep_best_sol=True,keep_best_sol_rho=1e2)
    problem=G1GaitTO(dyn,ms,Cost(),Xref,.02); points=problem.spline.fit(targets); points=np.clip(points.reshape(KNOTS,raw.nu),problem.dyn.u_lb,problem.dyn.u_ub); points[:,[-2,-1]]=0.; U=problem.spline.evaluate(points)
    x=Xref[0].copy(); X=[x.copy()]
    for u in U: x=problem.dyn.dynamics(x,u);X.append(x.copy())
    X=np.asarray(X)
    # Replace only the Cartesian target: exact native state remains the soft state reference.
    d=mujoco.MjData(problem.dyn.model); sole=contact_geom_ids(problem.dyn.model,'right')
    desired=[]
    for k,x in enumerate(problem.X_ref):
        d.qpos[:]=x[:problem.dyn.nq];d.qvel[:]=x[problem.dyn.nq:];mujoco.mj_forward(problem.dyn.model,d)
        current=min_sole_z(problem.dyn.model,d,sole);want=curve(k*.001);problem.ee.pos_ref[k,2,2]+=want-current;desired.append(want)
    z=problem.pack(X[::20],points); constraints=problem.constraints(z); objective=problem.objective(z); gradient=problem.gradient(z)
    roots=X[:,2];tilts=np.asarray([yaw_tilt(q[3:7])[1] for q in X[:,:problem.dyn.nq]])
    metrics={'variables':problem.n_vars,'constraints':problem.n_constraints,'constraints_returned':len(constraints),'constraints_absmax':float(np.max(np.abs(constraints))),
             'initial_state_absmax':float(np.max(np.abs(X[0]-problem.x_init))),'objective':float(objective),'gradient_absmax':float(np.max(np.abs(gradient))),
             'warmstart_root_z_min_m':float(np.min(roots)),'warmstart_root_tilt_max_rad':float(np.max(tilts)),'desired_clearance_max_m':float(np.max(desired))}
    gates=c['hard_gates'];checks={'constraints':metrics['constraints_absmax']<=gates['constraints_absmax'],'initial':metrics['initial_state_absmax']<=gates['initial_state_absmax'],
      'root_z':metrics['warmstart_root_z_min_m']>=gates['warmstart_root_z_min_m'],'root_tilt':metrics['warmstart_root_tilt_max_rad']<=gates['warmstart_root_tilt_max_rad'],
      'objective_finite':bool(np.isfinite(objective)),'gradient_finite':bool(np.all(np.isfinite(gradient))),'constraint_count':len(constraints)==problem.n_constraints}
    passed=all(checks.values())
    result={'stage':'Dynamic Retargeting Phase34','scope':'NLP build/objective/gradient/zero-defect warm-start only; IPOPT not constructed','contract_sha256':sha256(CONTRACT),
      'source':{'phase32_sha256':sha256(P32),'phase33b_sha256':sha256(P33B)},'metrics':metrics,'checks':checks,
      'execution':{'mj_step_calls':STEPS,'optimizer_instances':0,'optimizer_steps':0,'cpu_threads':1,'gpu':False},
      'decision':{'native_contact_nlp_preflight_qualified':passed,'ipopt_solve_authorized':False,'training_unlocked':False,'result':'NLP_PREFLIGHT_PASSED_SOLVE_STILL_LOCKED' if passed else 'NLP_PREFLIGHT_REJECTED_NO_SOLVE'}}
    if return_runtime:return result,problem,X,U,physical,anchor
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT);a=ap.parse_args();r=build();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'metrics':r['metrics'],'checks':r['checks'],'decision':r['decision']},indent=2))
if __name__=='__main__':main()
