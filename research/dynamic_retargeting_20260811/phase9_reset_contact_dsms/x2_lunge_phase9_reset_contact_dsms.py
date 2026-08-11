#!/usr/bin/env python3
"""One reset-projected, phase-aware 0.70 s DSMS solve and raw replay."""
from __future__ import annotations

import dataclasses, hashlib, json, os, sys, time
from pathlib import Path
os.environ["OMP_NUM_THREADS"]="4"; os.environ["MKL_NUM_THREADS"]="4"; os.environ["OPENBLAS_NUM_THREADS"]="4"
import numpy as np

ROOT=Path("/home/humanplus/projects/ZHY/dsms_workspace")
PHASE3B=ROOT/"phase3b"; PHASE6=ROOT/"phase6_reset_bridge"; OUT=ROOT/"phase9_reset_contact_dsms"
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(PHASE6),str(PHASE3B)]
import x2_lunge_dsms_phase3b_corrected as b
from x2_lunge_phase6_reset_bridge import collidable_sole_spheres, actual_contacts, sole_clearances
import mujoco
from src.dynamics import DynamicsConfig
from src.multi_shooting import MultiShootingConfig
from src.spline import SplineConfig
from examples.g1_gait.g1_gait import G1GaitTO
from utils.file_utils import save_trajectory

RESET=PHASE6/"phase6_projected_initial_state.npz"
HORIZON=.70; STANCE_CLEAR=.00025; LIFT_CLEAR=.030; LIFT_START=10/30; LIFT_RAMP=.10


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for z in iter(lambda:f.read(1<<20),b""): h.update(z)
    return h.hexdigest()


@dataclasses.dataclass
class Config(b.TrackConfig):
    def ee_track_expanded(self):
        return ((1,"pelvis",1.,1.,1.,.01,.01,False),
                (1,"left_ankle_roll_link",100.,300.,1.,20.,.01,False),
                (1,"right_ankle_roll_link",100.,300.,1.,20.,.01,False),
                (1,"left_wrist_yaw_link",1.,1.,1.,.01,.01,True),
                (1,"right_wrist_yaw_link",1.,1.,1.,.01,.01,True))


def load_reset_reference():
    n=int(round(HORIZON/.02))+1
    _,_,_,X,contract=b.load_reference(n)
    z=np.load(RESET,allow_pickle=False)
    X[0,:38]=z["qpos"]; X[0,38:]=z["qvel"]
    return X,contract,z["qpos"].copy(),z["qvel"].copy()


def desired_left_clearance(t):
    u=np.clip((t-LIFT_START)/LIFT_RAMP,0.,1.); s=u*u*(3-2*u)
    return STANCE_CLEAR+(LIFT_CLEAR-STANCE_CLEAR)*s


def patch_targets(p):
    m=p.dyn.model; d=mujoco.MjData(m); floor,helper=b.physics.foot_geom_contract(m); feet=collidable_sole_spheres(m,helper)
    assert all(len(gs)==12 for gs in feet.values())
    names=("left_ankle_roll_link","right_ankle_roll_link")
    bids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in names]
    # G1GaitTO has already expanded the 50 Hz reference to the 1 kHz simulation grid.
    n=len(p.X_ref); t=np.arange(n)*.001; body=np.zeros((n,2,3)); center=np.zeros((n,2,2)); clear=np.zeros((n,2))
    for k,x in enumerate(p.X_ref):
        d.qpos[:]=x[:m.nq]; d.qvel[:]=x[m.nq:]; mujoco.mj_forward(m,d)
        c=sole_clearances(m,d,feet,floor)
        for j,s in enumerate(("left","right")):
            body[k,j]=d.xpos[bids[j]]; center[k,j]=np.mean(d.geom_xpos[list(feet[s]),:2],axis=0); clear[k,j]=np.min(c[s])
    left_des=desired_left_clearance(t); right_des=np.full(n,STANCE_CLEAR)
    for j,(row,des) in enumerate(((1,left_des),(2,right_des))):
        p.ee.pos_ref[:,row,:2]=body[:,j,:2]+(center[0,j]-center[:,j])
        p.ee.pos_ref[:,row,2]=body[:,j,2]+(des-clear[:,j])
        p.ee.v_ref[:,row,:]=0.
        if j==0: p.ee.v_ref[:,row,:3]=np.gradient(p.ee.pos_ref[:,row,:3],.001,axis=0,edge_order=1)
    sample_t=np.asarray([0.,LIFT_START,LIFT_START+.05,LIFT_START+LIFT_RAMP,HORIZON])
    p.phase9_diag={"active_spheres":{s:len(gs) for s,gs in feet.items()},
        "clearance_samples":{"time_s":sample_t.tolist(),"left_m":desired_left_clearance(sample_t).tolist(),
                             "right_m":np.full(len(sample_t),STANCE_CLEAR).tolist()},
        "left_at_10over30_m":float(np.interp(LIFT_START,t,left_des)),"left_terminal_m":float(left_des[-1]),
        "right_terminal_m":float(right_des[-1]),"lift_start_s":LIFT_START,"lift_ramp_s":LIFT_RAMP}


def build(X,max_iter):
    dyn=DynamicsConfig(model_path=str(b.SCENE),sim_dt=.001,integrator=None,actuator_mode="position",n_threads=4,
                       fd_eps=1e-6,fd_centered=True,friction_cone=None)
    N=len(X)-1; spline=SplineConfig(M=N+1,spline_type="linear")
    ms=MultiShootingConfig(N=N,node_dt=.02,spline=spline,ipopt_options={"linear_solver":"mumps","print_level":5,
        "max_iter":max_iter,"tol":1e-2,"acceptable_tol":1e-2,"acceptable_constr_viol_tol":1e-2,
        "acceptable_iter":5,"hessian_approximation":"limited-memory","mu_strategy":"adaptive"},
        keep_best_sol=True,keep_best_sol_rho=1e2)
    p=G1GaitTO(dyn,ms,Config(),X,.02); patch_targets(p); return p


def preflight():
    X,c,q,dq=load_reset_reference(); p=build(X,1); X0,U0=p.warm_start(p.N_sim)
    z=p.pack(X0[::p.K][:p.N+1],p.spline.fit(U0)); returned=len(p.constraints(z))
    head_idx=[i for i,n in enumerate(c.actuator_joint_names) if n.startswith("head_")]
    head_bounds=[[float(p._lb[p.n_states+a]),float(p._ub[p.n_states+a])] for a in head_idx]
    out={"schema":"phase9_preflight_v1","scene_sha256":sha(b.SCENE),"control_sha256":sha(b.CONTROL),
         "reset_sha256":sha(RESET),"mujoco":mujoco.__version__,"threads":4,
         "horizon_s":(len(X)-1)*.02,"nodes":p.N,"reset_qpos_exact":bool(np.array_equal(X[0,:p.dyn.nq],q)),
         "reset_qvel_zero_exact":bool(np.max(np.abs(X[0,p.dyn.nq:]))==0.),"target":p.phase9_diag,
         "constraints_registered":int(p.n_constraints),"constraints_returned":returned,
         "constraints_exact":returned==p.n_constraints,"head_actuator_indices":head_idx,"head_bounds":head_bounds,
         "head_bounds_zero":bool(all(v==[0.,0.] for v in head_bounds))}
    out["pass"]=bool(out["threads"]==4 and abs(out["horizon_s"]-.70)<1e-12 and out["reset_qpos_exact"] and
                     out["reset_qvel_zero_exact"] and out["constraints_exact"] and out["head_bounds_zero"] and
                     out["target"]["active_spheres"]=={"left":12,"right":12} and out["target"]["left_terminal_m"]>=.0299)
    (OUT/"phase9_preflight.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return out


def solve():
    Xref,contract,_,_=load_reset_reference(); p=build(Xref,300); X0,U0=p.warm_start(p.N_sim)
    t0=time.perf_counter(); X,U,info=p.solve(X0,U0); wall=time.perf_counter()-t0
    fine=p.stitched_trajectory(X,U); ends=p.dyn.rollout_batch(X[:p.N],U.reshape(p.N,p.K,p.nu))[:,p.K]
    defects=np.asarray([p.dyn.state_diff(X[i+1],ends[i]) for i in range(p.N)])
    path=OUT/"x2_lunge_phase9_prefix.npz"
    save_trajectory(str(path),time=np.arange(len(fine))*.001,state=fine,input=U,model=str(b.SCENE),reference=p.X_ref,
                    spline_type="linear",defects=defects,node_dt=np.asarray(.02))
    msg=info.get("status_msg",""); msg=msg.decode() if isinstance(msg,(bytes,bytearray)) else str(msg)
    out={"status":int(info.get("status",-999)),"status_msg":msg,"status_ok":info.get("status") in (0,1),
         "wall_s":wall,"objective":float(info.get("obj_val",np.nan)),"max_defect":float(np.max(np.abs(defects))),
         "defect_p95":float(np.percentile(np.abs(defects),95)),"artifact":str(path),"target":p.phase9_diag}
    (OUT/"phase9_solve_result.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return out


def replay():
    art=np.load(OUT/"x2_lunge_phase9_prefix.npz",allow_pickle=True); target=np.asarray(art["input"],float)
    stitched=np.asarray(art["state"],float); defects=np.asarray(art["defects"],float)
    Xref,contract,qreset,vreset=load_reset_reference(); m=mujoco.MjModel.from_xml_path(str(b.SCENE)); d=mujoco.MjData(m)
    floor,helper=b.physics.foot_geom_contract(m); feet=collidable_sole_spheres(m,helper); geom_side={g:s for s,gs in feet.items() for g in gs}
    d.qpos[:]=qreset; d.qvel[:]=vreset; mujoco.mj_forward(m,d); prev=d.geom_xpos.copy()
    actual={s:[] for s in feet}; slip={s:[] for s in feet}; minclear={s:[] for s in feet}; rootv=[]; rootp=[]; tilts=[]; zs=[]; heads=[]
    fall=None; causes=[]; sat=[]
    def sample(step):
        nonlocal fall,causes
        contacts=actual_contacts(m,d,floor,geom_side); cs=sole_clearances(m,d,feet,floor)
        for s in feet:
            actual[s].append(bool(contacts[s])); minclear[s].append(float(np.min(cs[s])))
            if step and contacts[s]:
                slip[s].extend((step,float(np.linalg.norm(d.geom_xpos[g,:2]-prev[g,:2])/.001)) for g in contacts[s])
        rootv.append(d.qvel[:2].copy()); rootp.append(d.qpos[:3].copy()); z=float(d.qpos[2]); a=float(b.physics.root_tilt(d.qpos[3:7])); zs.append(z); tilts.append(a)
        heads.append(float(np.max(np.abs(d.qpos[[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ("head_yaw_joint","head_pitch_joint")]]))))
        if fall is None and (z<.42 or a>.90):
            fall=step*.001; causes=(["root_z"] if z<.42 else [])+(["tilt"] if a>.90 else [])
    sample(0)
    for k,u in enumerate(target):
        q=d.qpos[contract.qpos_addresses]; dq=d.qvel[contract.qvel_addresses]; raw=contract.kp*(u-q)-contract.kd*dq
        sat.append(int(np.sum((raw<contract.torque_low)|(raw>contract.torque_high)))); d.ctrl[:]=np.clip(raw,contract.torque_low,contract.torque_high)
        mujoco.mj_step(m,d); sample(k+1); prev[:]=d.geom_xpos
    t=np.arange(len(target)+1)*.001; eval_end=(int(round(fall/.001))+1) if fall is not None else len(t)
    lift=(t[:eval_end]>LIFT_START+1e-12); stance=(t[:eval_end]<=LIFT_START+1e-12)
    rv=np.asarray(rootv[:eval_end]); acc=np.linalg.norm(np.diff(rv,axis=0)/.001,axis=1)
    # Robust semantic comparison directly at 1 kHz by interpolating 50 Hz reference.
    tref=np.arange(len(Xref))*.02; tf=np.arange(len(stitched))*.001
    refq=np.column_stack([np.interp(tf,tref,Xref[:,7+j]) for j in range(31)])
    sem=stitched[:,7:38]-refq
    out={"schema":"phase9_raw_replay_v1","duration_s":len(target)*.001,"fall_time_s":fall,"fall_causes":causes,
         "beats_phase6_0p683":bool(fall is None or fall>.683),"survive_0p70":fall is None,"root_z_min_m":float(np.min(zs[:eval_end])),
         "tilt_max_rad":float(np.max(tilts[:eval_end])),"root_xy_excursion_m":float(np.max(np.linalg.norm(np.asarray(rootp[:eval_end])[:,:2]-rootp[0][:2],axis=1))),
         "root_accel_p95_max_mps2":[float(np.percentile(acc,95)),float(np.max(acc))],
         "max_defect":float(np.max(np.abs(defects))),"torque_saturation_fraction":float(np.sum(sat))/(len(sat)*m.nu),
         "semantic_joint_rms_rad":float(np.sqrt(np.mean(sem**2))),"semantic_joint_max_rad":float(np.max(np.abs(sem))),
         "head_max_abs_rad":float(np.max(heads[:eval_end])),"active12":{}}
    for s in feet:
        arr=np.asarray(actual[s][:eval_end],bool); sl=np.asarray([v for step,v in slip[s] if step<eval_end])
        lift_fraction=float(np.mean(arr[lift])) if np.any(lift) else None
        lift_switches=int(np.sum(arr[np.flatnonzero(lift)[0]:][1:] != arr[np.flatnonzero(lift)[0]:][:-1])) if np.any(lift) else 0
        out["active12"][s]={"min_clearance_mm":1000*float(np.min(minclear[s][:eval_end])),"contact_fraction":float(np.mean(arr)),
            "stance_contact_fraction":float(np.mean(arr[stance])),"liftoff_contact_fraction":lift_fraction,
            "contact_switches":int(np.sum(arr[1:]!=arr[:-1])),
            "liftoff_contact_switches":lift_switches,"slip_p95_mps":float(np.percentile(sl,95)) if len(sl) else None,
            "slip_max_mps":float(np.max(sl)) if len(sl) else None}
    gates={"solver_status":json.loads((OUT/"phase9_solve_result.json").read_text())["status_ok"],"defect":out["max_defect"]<=.01,
           "beats_phase6":out["beats_phase6_0p683"],"left_liftoff":out["active12"]["left"]["liftoff_contact_fraction"]<=.20,
           "left_switch":out["active12"]["left"]["liftoff_contact_switches"]>=1,"right_stance":out["active12"]["right"]["contact_fraction"]>=.98,
           "penetration":min(v["min_clearance_mm"] for v in out["active12"].values())>=-.5,
           "slip":all(v["slip_p95_mps"] is not None and v["slip_p95_mps"]<=.10 for v in out["active12"].values()),
           "root_accel":out["root_accel_p95_max_mps2"][0]<=4.,"head":out["head_max_abs_rad"]<=.02,
           "semantic":out["semantic_joint_rms_rad"]<=.15}
    out["gates"]=gates; out["pass"]=all(gates.values()); (OUT/"phase9_raw_replay.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return out


if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "preflight"
    {"preflight":preflight,"solve":solve,"replay":replay}[mode]()
