#!/usr/bin/env python3
"""One offline 7-knot nonlinear contact skeleton solve for X2."""
from __future__ import annotations
import argparse, copy, importlib.util, json, time
from pathlib import Path
import joblib, mujoco, numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import qr
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.spatial.transform import Rotation, Slerp

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
P18=HERE/"phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec=importlib.util.spec_from_file_location("p18",P18); p18=importlib.util.module_from_spec(spec); spec.loader.exec_module(p18)
KNOT_ACTIVE=(("left","right"),("left","right"),("left",),("left","right"),("right",),("left","right"),("left","right"))
LEFT_SEG=((0,1,2,3),(5,6)); RIGHT_SEG=((0,1),(3,4,5,6))

class Eval:
    def __init__(self,model,base,frames,feet,floor,lower,dofs,qadr,pelvis,upper,boundary,cfg):
        self.m=model; self.base=base; self.frames=frames; self.feet=feet; self.floor=floor; self.lower=lower; self.dofs=dofs; self.qadr=qadr; self.pelvis=pelvis; self.upper=upper; self.boundary=boundary; self.cfg=cfg; self.cache=None; self.eq_keep=None
        self.base_upper=[]; d=mujoco.MjData(model)
        for f in frames: p18.set_frame(model,d,base,f,qadr); self.base_upper.append(d.xpos[upper].copy())
        self.base_upper=np.asarray(self.base_upper)
        self.fixed_geom={}
        for k,f in enumerate(frames):
            p18.set_frame(model,d,base,f,qadr)
            for s in p18.SIDES: self.fixed_geom[(k,s)]=feet[s][int(np.argmin(d.geom_xpos[feet[s],2]-model.geom_size[feet[s],0]-d.geom_xpos[floor,2]))]
    def state(self,x):
        if self.cache is not None and np.array_equal(x,self.cache[0]): return self.cache[1]
        xx=x.reshape(7,21); rows=[]; d=mujoco.MjData(self.m); sel=np.r_[np.arange(6),self.dofs]
        for k,f in enumerate(self.frames):
            root=np.asarray(self.base["root_trans_offset"])[f]+xx[k,:3]; quat=p18.quat_apply(np.asarray(self.base["root_rot"])[f],xx[k,3:6]); q=np.asarray(self.base["dof"])[f].copy(); q[self.lower]+=xx[k,6:]
            d.qpos[:3]=root; d.qpos[3:7]=quat[[3,0,1,2]]; d.qpos[self.qadr]=q; mujoco.mj_forward(self.m,d)
            com=d.subtree_com[self.pelvis].copy(); cj=np.zeros((3,self.m.nv)); mujoco.mj_jacSubtreeCom(self.m,d,cj,self.pelvis)
            foot={}; fj={}; signed={}; sj={}
            for s in p18.SIDES:
                foot[s]=np.mean(d.geom_xpos[self.feet[s]],axis=0); full=np.zeros((3,self.m.nv)); signed[s]=[]; sj[s]=[]
                for g in self.feet[s]:
                    jp=np.zeros((3,self.m.nv)); jr=np.zeros((3,self.m.nv)); mujoco.mj_jacGeom(self.m,d,jp,jr,g); full+=jp/len(self.feet[s]); signed[s].append(d.geom_xpos[g,2]-self.m.geom_size[g,0]-d.geom_xpos[self.floor,2]); sj[s].append(jp[2,sel])
                fj[s]=full[:,sel]; signed[s]=np.asarray(signed[s]); sj[s]=np.asarray(sj[s])
            up=[]; uj=[]
            for body in self.upper:
                jp=np.zeros((3,self.m.nv)); jr=np.zeros((3,self.m.nv)); mujoco.mj_jacBody(self.m,d,jp,jr,body); up.append(d.xpos[body].copy()); uj.append(jp[:,sel])
            rows.append({"root":root,"quat":quat,"q":q,"com":com,"cj":cj[:,sel],"foot":foot,"fj":fj,"signed":signed,"sj":sj,"upper":np.asarray(up),"uj":np.asarray(uj)})
        self.cache=(x.copy(),rows); return rows
    def objective(self,x):
        rows=self.state(x); xx=x.reshape(7,21); W=self.cfg["secondary_weights_sqrt"]
        value=W["correction"]**2*np.sum(xx**2)+W["knot_velocity"]**2*np.sum(np.diff(xx,axis=0)**2)+W["knot_acceleration"]**2*np.sum(np.diff(xx,n=2,axis=0)**2)
        for k,r in enumerate(rows): value+=W["upper_keypoint"]**2*np.sum((r["upper"]-self.base_upper[k])**2)
        return float(value)
    def objective_jac(self,x):
        rows=self.state(x); xx=x.reshape(7,21); W=self.cfg["secondary_weights_sqrt"]; g=2*W["correction"]**2*xx
        v=np.diff(xx,axis=0); g[:-1]-=2*W["knot_velocity"]**2*v; g[1:]+=2*W["knot_velocity"]**2*v
        a=np.diff(xx,n=2,axis=0); g[:-2]+=2*W["knot_acceleration"]**2*a; g[1:-1]-=4*W["knot_acceleration"]**2*a; g[2:]+=2*W["knot_acceleration"]**2*a
        for k,r in enumerate(rows):
            for j in range(len(self.upper)): g[k]+=2*W["upper_keypoint"]**2*r["uj"][j].T@(r["upper"][j]-self.base_upper[k,j])
        return g.ravel()
    def eq(self,x,jac=False):
        rows=self.state(x); vals=[]; J=[]
        def add(k,val,grad):
            vals.extend(np.atleast_1d(val)); block=np.zeros((np.atleast_1d(val).size,147)); block[:,k*21:(k+1)*21]=np.atleast_2d(grad); J.extend(block)
        xx=x.reshape(7,21); target=self.boundary
        for k in (0,6): add(k,xx[k]-target[k],np.eye(21))
        for k,r in enumerate(rows):
            act=KNOT_ACTIVE[k]
            for s in act:
                idx=self.feet[s].index(self.fixed_geom[(k,s)]); add(k,r["signed"][s][idx]-.00025,r["sj"][s][idx])
            center=np.mean([r["foot"][s][:2] for s in act],axis=0); centerj=np.mean([r["fj"][s][:2] for s in act],axis=0); add(k,r["com"][:2]-center,r["cj"][:2]-centerj)
        for side,segs in (("left",LEFT_SEG),("right",RIGHT_SEG)):
            for seg in segs:
                a=seg[0]
                for b in seg[1:]:
                    vals.extend(rows[b]["foot"][side][:2]-rows[a]["foot"][side][:2]); block=np.zeros((2,147)); block[:,b*21:(b+1)*21]=rows[b]["fj"][side][:2]; block[:,a*21:(a+1)*21]=-rows[a]["fj"][side][:2]; J.extend(block)
        array=np.asarray(J) if jac else np.asarray(vals)
        return array if self.eq_keep is None else array[self.eq_keep]
    def ineq(self,x,jac=False):
        rows=self.state(x); vals=[]; J=[]
        for k,r in enumerate(rows):
            for s in p18.SIDES:
                threshold=.012 if (k==2 and s=="right") or (k==4 and s=="left") else -.00001
                vals.extend(r["signed"][s]-threshold); block=np.zeros((len(self.feet[s]),147)); block[:,k*21:(k+1)*21]=r["sj"][s]; J.extend(block)
        return np.asarray(J) if jac else np.asarray(vals)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); p.add_argument("--motion",type=Path,required=True); p.add_argument("--phase15",type=Path,required=True); p.add_argument("--boundary",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--motion-id",default="PHUMA-LUNGE-R-001"); a=p.parse_args(); cfg=json.loads((HERE/"phase21_contact_keyframe_skeleton_contract.json").read_text()); model=mujoco.MjModel.from_xml_path(str(a.scene)); floor,helper=p18.physics.foot_geom_contract(model); feet={s:sorted(g for g in helper[s] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]!=0) for s in p18.SIDES}; raw=joblib.load(a.motion)[a.motion_id]; scale=json.loads(a.phase15.read_text())["intervention"]["hip_roll_scale"]; base=p18.init_entry(raw,scale,model,feet,floor); fractions=np.asarray(cfg["knot_fractions"]); frames=np.rint(fractions*(len(raw["dof"])-1)).astype(int); names=list(base["joint_names_mujoco"]); lower=np.asarray([names.index(x) for x in p18.LOWER15]); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in p18.LOWER15]); dofs=model.jnt_dofadr[jids]; qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in names]); pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis"); upper=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,x) for x in p18.UPPER_BODIES]
    b=np.asarray(json.loads(a.boundary.read_text())["boundary"]["qpos"]); target=np.zeros((7,21)); bq=np.asarray([b[model.jnt_qposadr[j]] for j in jids]); bquat=b[3:7][[1,2,3,0]]
    for k in (0,6):
        f=frames[k]; target[k,:3]=b[:3]-np.asarray(base["root_trans_offset"])[f]; target[k,3:6]=(Rotation.from_quat(bquat)*Rotation.from_quat(np.asarray(base["root_rot"])[f]).inv()).as_rotvec(); target[k,6:]=bq-np.asarray(base["dof"])[f,lower]
    limits=model.jnt_range[jids]; lo=np.tile(np.r_[[-.4,-.4,-.2],[-.8]*3,[-10.0]*15],7); hi=np.tile(np.r_[[.4,.4,.2],[.8]*3,[10.0]*15],7)
    for k,f in enumerate(frames): lo[k*21+6:(k+1)*21]=limits[:,0]-np.asarray(base["dof"])[f,lower]; hi[k*21+6:(k+1)*21]=limits[:,1]-np.asarray(base["dof"])[f,lower]
    # Explicit check avoids starting an optimizer with impossible native endpoints.
    boundary_ok=all(np.all(target[k]>=lo[k*21:(k+1)*21]-1e-9) and np.all(target[k]<=hi[k*21:(k+1)*21]+1e-9) for k in (0,6)); started=time.perf_counter()
    if not boundary_ok:
        labels=["root_x","root_y","root_z","root_rx","root_ry","root_rz",*p18.LOWER15]; violations=[]
        for k in (0,6):
            for j,label in enumerate(labels):
                low,high=lo[k*21+j],hi[k*21+j]; value=target[k,j]
                if value<low-1e-9 or value>high+1e-9: violations.append({"knot":k,"field":label,"target_correction":float(value),"lower_bound":float(low),"upper_bound":float(high),"overshoot":float(max(low-value,value-high,0))})
        result={"stage":"Phase21 contact keyframe skeleton","preflight":{"boundary_within_bounds":False,"target_max_abs":float(np.max(np.abs(target[[0,6]]))),"violation_count":len(violations),"violations":violations},"execution":{"slsqp_started":False,"mj_step_calls":0,"gpu":False},"decision":{"stage_a_passed":False,"physics_unlocked":False,"training_unlocked":False,"next":"boundary contract incompatible with frozen correction bounds"}}; a.output.write_text(json.dumps(result,indent=2)+"\n"); return
    ev=Eval(model,base,frames,feet,floor,lower,dofs,qadr,pelvis,upper,target,cfg); x0=np.zeros(147); x0[:21]=target[0]; x0[-21:]=target[6]
    raw_eq=ev.eq(x0); raw_jac=ev.eq(x0,True); _q,_r,piv=qr(raw_jac.T,pivoting=True,mode="economic"); eq_rank=int(np.linalg.matrix_rank(raw_jac)); ev.eq_keep=np.sort(piv[:eq_rank])
    res=minimize(ev.objective,x0,jac=ev.objective_jac,method="SLSQP",bounds=Bounds(lo,hi),constraints=[NonlinearConstraint(lambda x:ev.eq(x),0,0,jac=lambda x:ev.eq(x,True)),NonlinearConstraint(lambda x:ev.ineq(x),0,np.inf,jac=lambda x:ev.ineq(x,True))],options={"maxiter":cfg["solver"]["maximum_iterations"],"ftol":1e-9,"disp":True})
    xx=res.x.reshape(7,21); times=frames.astype(float); dense=np.arange(len(raw["dof"])); candidate=copy.deepcopy(base); root=np.asarray(base["root_trans_offset"],float).copy(); q=np.asarray(base["dof"],float).copy(); root[:]=CubicSpline(times,np.asarray(base["root_trans_offset"])[frames]+xx[:,:3],axis=0)(dense); q[:,lower]=CubicSpline(times,np.asarray(base["dof"])[frames][:,lower]+xx[:,6:],axis=0)(dense); knot_quat=np.asarray([p18.quat_apply(np.asarray(base["root_rot"])[frames[k]],xx[k,3:6]) for k in range(7)]); quat=Slerp(times,Rotation.from_quat(knot_quat))(dense).as_quat(); candidate["root_trans_offset"]=root.astype(np.asarray(base["root_trans_offset"]).dtype); candidate["root_rot"]=quat.astype(np.asarray(base["root_rot"]).dtype); candidate["dof"]=q.astype(np.asarray(base["dof"]).dtype); active=p18.schedule("DS-L-DS-R-DS",len(q)); gates=json.loads((HERE/"phase18_joint_contact_generator/prereg_phase18_joint_contact_generator.json").read_text()); metrics,checks,passed=p18.audit(model,base,candidate,active,feet,floor,gates)
    # Report both the independent solve system and every original equality.
    keep=ev.eq_keep; independent_eq=float(np.max(np.abs(ev.eq(res.x)))); ev.eq_keep=None; full_eq=float(np.max(np.abs(ev.eq(res.x)))); ev.eq_keep=keep
    result={"stage":"Phase21 contact keyframe skeleton","preflight":{"boundary_within_bounds":True,"frames":frames.tolist(),"raw_equality_rows":len(raw_eq),"independent_equality_rank":eq_rank},"execution":{"slsqp_started":True,"success":bool(res.success),"status":int(res.status),"message":str(res.message),"iterations":int(res.nit),"wall_time_s":time.perf_counter()-started,"mj_step_calls":0,"gpu":False},"constraints":{"independent_eq_max_abs":independent_eq,"full_original_eq_max_abs":full_eq,"ineq_min":float(np.min(ev.ineq(res.x)))},"dense_metrics":metrics,"dense_checks":checks,"decision":{"stage_a_passed":bool(res.success and passed and full_eq<=1e-6),"physics_unlocked":False,"training_unlocked":False,"next":"review only" if res.success and passed and full_eq<=1e-6 else "stop without alternate timing/weights/template"}}; a.output.write_text(json.dumps(result,indent=2)+"\n");
    if result["decision"]["stage_a_passed"]: joblib.dump({a.motion_id:candidate},a.candidate,compress=True)
if __name__=="__main__": main()
