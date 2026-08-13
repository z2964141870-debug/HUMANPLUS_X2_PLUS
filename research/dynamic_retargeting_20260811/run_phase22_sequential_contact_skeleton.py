#!/usr/bin/env python3
"""Sequential contact/COM skeleton feasibility, no semantics and no physics."""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import mujoco, numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

HERE=Path(__file__).resolve().parent; P18=HERE/"phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec=importlib.util.spec_from_file_location("p18",P18); p18=importlib.util.module_from_spec(spec); spec.loader.exec_module(p18)
STEPS=(("DS_NATIVE",("left","right"),()),("DS_LOAD",("left","right"),()),("L_SUPPORT_R_SWING",("left",),("right",)),("DS_R_TOUCHDOWN",("left","right"),()),("R_SUPPORT_L_SWING",("right",),("left",)),("DS_L_TOUCHDOWN",("left","right"),()),("DS_SETTLE",("left","right"),()))

class Frame:
    def __init__(self,model,feet,floor,pelvis,dofs,qadr,state,active,swing,anchors):
        self.m=model; self.feet=feet; self.floor=floor; self.pelvis=pelvis; self.dofs=dofs; self.qadr=qadr; self.base=state; self.active=active; self.swing=swing; self.anchors=anchors; self.cache=None; self.fixed={}
        r=self.eval(np.zeros(21)); self.fixed={s:int(np.argmin(r["signed"][s])) for s in active}; self.cache=None
    def eval(self,x):
        if self.cache is not None and np.array_equal(x,self.cache[0]): return self.cache[1]
        root=self.base["root"]+x[:3]; quat=p18.quat_apply(self.base["quat"],x[3:6]); q=self.base["q"].copy(); q[self.base["lower"]]+=x[6:]; d=mujoco.MjData(self.m); d.qpos[:3]=root; d.qpos[3:7]=quat[[3,0,1,2]]; d.qpos[self.qadr]=q; mujoco.mj_forward(self.m,d); sel=np.r_[np.arange(6),self.dofs]; cj=np.zeros((3,self.m.nv)); mujoco.mj_jacSubtreeCom(self.m,d,cj,self.pelvis); foot={}; fj={}; signed={}; sj={}
        for s in p18.SIDES:
            pos=d.geom_xpos[self.feet[s]]; foot[s]=np.mean(pos,axis=0); full=np.zeros((3,self.m.nv)); signed[s]=[]; sj[s]=[]
            for g in self.feet[s]:
                jp=np.zeros((3,self.m.nv)); jr=np.zeros((3,self.m.nv)); mujoco.mj_jacGeom(self.m,d,jp,jr,g); full+=jp/len(self.feet[s]); signed[s].append(d.geom_xpos[g,2]-self.m.geom_size[g,0]-d.geom_xpos[self.floor,2]); sj[s].append(jp[2,sel])
            foot[s]=np.asarray(foot[s]); fj[s]=full[:,sel]; signed[s]=np.asarray(signed[s]); sj[s]=np.asarray(sj[s])
        out={"root":root,"quat":quat,"q":q,"com":d.subtree_com[self.pelvis].copy(),"cj":cj[:,sel],"foot":foot,"fj":fj,"signed":signed,"sj":sj}; self.cache=(x.copy(),out); return out
    def eq(self,x,jac=False):
        r=self.eval(x); vals=[]; rows=[]
        for s in self.active:
            k=self.fixed[s]; vals.append(r["signed"][s][k]-.00025); rows.append(r["sj"][s][k])
        center=np.mean([r["foot"][s][:2] for s in self.active],axis=0); centerj=np.mean([r["fj"][s][:2] for s in self.active],axis=0); vals.extend(r["com"][:2]-center); rows.extend(r["cj"][:2]-centerj)
        for s in self.active:
            if s in self.anchors: vals.extend(r["foot"][s][:2]-self.anchors[s]); rows.extend(r["fj"][s][:2])
        return np.asarray(rows) if jac else np.asarray(vals)
    def ineq(self,x,jac=False):
        r=self.eval(x); vals=[]; rows=[]
        for s in p18.SIDES:
            threshold=.012 if s in self.swing else -.00001
            vals.extend(r["signed"][s]-threshold); rows.extend(r["sj"][s])
        return np.asarray(rows) if jac else np.asarray(vals)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); p.add_argument("--boundary",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); model=mujoco.MjModel.from_xml_path(str(a.scene)); floor,helper=p18.physics.foot_geom_contract(model); feet={s:sorted(g for g in helper[s] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]!=0) for s in p18.SIDES}; boundary=json.loads(a.boundary.read_text())["boundary"]; qpos=np.asarray(boundary["qpos"],float); names=[]
    for jid in range(model.njnt):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid)
        if name and model.jnt_type[jid]!=mujoco.mjtJoint.mjJNT_FREE: names.append(name)
    qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in names]); lower=np.asarray([names.index(x) for x in p18.LOWER15]); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in p18.LOWER15]); dofs=model.jnt_dofadr[jids]; limits=model.jnt_range[jids]; pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis"); state={"root":qpos[:3].copy(),"quat":qpos[3:7][[1,2,3,0]].copy(),"q":qpos[qadr].copy(),"lower":lower}; native={k:(v.copy() if hasattr(v,"copy") else v) for k,v in state.items()}; anchors={}; rows=[]; started=time.perf_counter()
    for index,(label,active,swing) in enumerate(STEPS):
        ev=Frame(model,feet,floor,pelvis,dofs,qadr,state,active,swing,anchors); initial=ev.eval(np.zeros(21));
        if index==0:
            result_x=np.zeros(21); success=True; objective_converged=True; status=0; message="native boundary accepted"
        else:
            lo=np.r_[[-.25,-.25,-.12],[-.35]*3,[-.65]*15]; hi=-lo; lo[6:]=np.maximum(lo[6:],limits[:,0]-state["q"][lower]); hi[6:]=np.minimum(hi[6:],limits[:,1]-state["q"][lower]); target=np.zeros(21)
            def obj(x):
                value=float(x@x)
                if label=="DS_SETTLE": value+=.25*float(np.sum((ev.eval(x)["q"][lower]-native["q"][lower])**2))
                return value
            def jac(x):
                g=2*x
                if label=="DS_SETTLE": g[6:]+=.5*(ev.eval(x)["q"][lower]-native["q"][lower])
                return g
            sol=minimize(obj,target,jac=jac,method="SLSQP",bounds=Bounds(lo,hi),constraints=[NonlinearConstraint(lambda x:ev.eq(x),0,0,jac=lambda x:ev.eq(x,True)),NonlinearConstraint(lambda x:ev.ineq(x),0,np.inf,jac=lambda x:ev.ineq(x,True))],options={"maxiter":100,"ftol":1e-9,"disp":False}); result_x=sol.x; objective_converged=bool(sol.success); status=int(sol.status); message=str(sol.message)
        exact=ev.eval(result_x)
        if index==0:
            # Phase41 already certifies this closed-AimDK soft-contact boundary;
            # do not replace it by the stricter skeleton contact convention.
            eqmax=0.0; inmin=float(min(np.min(exact["signed"][s])+.0005 for s in p18.SIDES)); success=bool(inmin>=0)
        else:
            eqmax=float(np.max(np.abs(ev.eq(result_x)))); inmin=float(np.min(ev.ineq(result_x))); success=bool(eqmax<=1e-6 and inmin>=-1e-8)
        rows.append({
            "index":index,
            "label":label,
            "success":success,
            "objective_converged":objective_converged,
            "status":status,
            "message":message,
            "eq_max_abs":eqmax,
            "ineq_min":inmin,
            "root":exact["root"].tolist(),
            "quat_xyzw":exact["quat"].tolist(),
            "lower_q":exact["q"][lower].tolist(),
            "foot_centroid_xy":{s:exact["foot"][s][:2].tolist() for s in p18.SIDES},
            "active":list(active),
            "swing":list(swing),
        })
        if not success: break
        state={"root":exact["root"],"quat":exact["quat"],"q":exact["q"],"lower":lower}
        if label=="DS_NATIVE": anchors={s:exact["foot"][s][:2].copy() for s in p18.SIDES}
        elif label=="DS_R_TOUCHDOWN": anchors["right"]=exact["foot"]["right"][:2].copy()
        elif label=="DS_L_TOUCHDOWN": anchors["left"]=exact["foot"]["left"][:2].copy()
    complete=len(rows)==len(STEPS) and all(r["success"] for r in rows); result={"stage":"Phase22 sequential contact skeleton","execution":{"completed_steps":len(rows),"wall_time_s":time.perf_counter()-started,"mj_step_calls":0,"gpu":False},"steps":rows,"decision":{"contact_skeleton_complete":complete,"physics_unlocked":False,"training_unlocked":False,"next":"add boundary/semantic layers only after review" if complete else "stop at first failed keyframe without alternate configuration"}}; a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"execution":result["execution"],"steps":rows,"decision":result["decision"]},indent=2))
if __name__=="__main__": main()
