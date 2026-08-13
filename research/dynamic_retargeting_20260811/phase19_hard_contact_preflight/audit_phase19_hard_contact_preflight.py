#!/usr/bin/env python3
"""First-order hard-contact feasibility audit; no physics integration."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import joblib, mujoco, numpy as np
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
P18=REPO/"research/dynamic_retargeting_20260811/phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec=importlib.util.spec_from_file_location("phase18_generator",P18); p18=importlib.util.module_from_spec(spec); spec.loader.exec_module(p18)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(scene,motion,phase15,motion_id):
    model=mujoco.MjModel.from_xml_path(str(scene)); floor,helper=p18.physics.foot_geom_contract(model)
    feet={s:sorted(g for g in helper[s] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]!=0) for s in p18.SIDES}
    raw=joblib.load(motion)[motion_id]; scale=json.loads(Path(phase15).read_text())["intervention"]["hip_roll_scale"]
    entry=p18.init_entry(raw,scale,model,feet,floor); active=p18.schedule("DS-L-DS-R-DS",len(entry["dof"])); names=list(entry["joint_names_mujoco"])
    qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in names]); lower_cols=np.asarray([names.index(x) for x in p18.LOWER15]); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in p18.LOWER15]); dofs=model.jnt_dofadr[jids]; limits=model.jnt_range[jids]; pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    rows=[]; data=mujoco.MjData(model); q=np.asarray(entry["dof"],float)
    for i in range(len(q)):
        p18.set_frame(model,data,entry,i,qadr); sel=np.r_[np.arange(6),dofs]; com=data.subtree_com[pelvis].copy(); cj=np.zeros((3,model.nv)); mujoco.mj_jacSubtreeCom(model,data,cj,pelvis)
        active_s=[s for s in p18.SIDES if active[s][i]]; foot={}; fj={}; sd={}; sj={}
        for s in p18.SIDES:
            foot[s]=np.mean(data.geom_xpos[feet[s]],axis=0); full=np.zeros((3,model.nv))
            for g in feet[s]:
                jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacGeom(model,data,jp,jr,g); full+=jp/len(feet[s])
            fj[s]=full[:,sel]; distances=data.geom_xpos[feet[s],2]-model.geom_size[feet[s],0]-data.geom_xpos[floor,2]; k=int(np.argmin(distances)); sd[s]=float(distances[k])
            jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacGeom(model,data,jp,jr,feet[s][k]); sj[s]=jp[2,sel]
        Aeq=[]; beq=[]
        for s in active_s: Aeq.append(sj[s]); beq.append(-sd[s])
        center=np.mean([foot[s][:2] for s in active_s],axis=0); center_j=np.mean([fj[s][:2] for s in active_s],axis=0)
        for axis in range(2): Aeq.append(cj[axis,sel]-center_j[axis]); beq.append(-(com[axis]-center[axis]))
        Aub=[]; bub=[]
        for s in p18.SIDES:
            if s not in active_s: Aub.append(-sj[s]); bub.append(sd[s]-.012)
        bounds=[(-.25,.25)]*2+[(-.12,.12)]+[(-.35,.35)]*3
        for k,c in enumerate(lower_cols): bounds.append((max(-.65,limits[k,0]-q[i,c]),min(.65,limits[k,1]-q[i,c])))
        sol=linprog(np.zeros(21),A_ub=np.asarray(Aub) if Aub else None,b_ub=np.asarray(bub) if Aub else None,A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,method="highs")
        residual=None if not sol.success else float(np.max(np.abs(np.asarray(Aeq)@sol.x-np.asarray(beq))))
        rows.append({"frame":i,"time_s":i/float(entry["fps"]),"support":active_s,"feasible":bool(sol.success),"status":int(sol.status),"equality_rank":int(np.linalg.matrix_rank(np.asarray(Aeq))),"equality_count":len(Aeq),"residual_max":residual,"delta_max_abs":None if not sol.success else float(np.max(np.abs(sol.x)))})
    single=[r for r in rows if len(r["support"])==1]; double=[r for r in rows if len(r["support"])==2]
    return {"stage":"Phase19 first-order hard contact feasibility preflight","execution":{"mj_forward_calls":len(rows),"mj_step_calls":0,"lp_solves":len(rows),"gpu":False},"assets":{"scene_sha256":sha(scene),"motion_sha256":sha(motion),"phase15_sha256":sha(phase15)},"contract":{"template":"DS-L-DS-R-DS","variables":"root xyz + local rotation + lower15 increment","hard_equalities":"active sole signed distance=0; COM xy equals active foot-centroid center","hard_inequality":"swing sole signed distance >=12mm","bounds":"root xyz/rotation and lower15 Phase18 frozen bounds + joint limits","truth":"local first-order geometry certificate only; not temporal, force, COP, or physics feasibility"},"summary":{"frames":len(rows),"feasible":sum(r["feasible"] for r in rows),"single_support_frames":len(single),"single_support_feasible":sum(r["feasible"] for r in single),"double_support_frames":len(double),"double_support_feasible":sum(r["feasible"] for r in double),"max_delta_feasible":max((r["delta_max_abs"] for r in rows if r["feasible"]),default=None)},"rows":rows,"decision":{}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); p.add_argument("--motion",type=Path,required=True); p.add_argument("--phase15",type=Path,required=True); p.add_argument("--motion-id",default="PHUMA-LUNGE-R-001"); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); result=run(a.scene,a.motion,a.phase15,a.motion_id)
    s=result["summary"]; result["decision"]={"local_hard_geometry_has_full_frame_coverage":s["feasible"]==s["frames"],"temporal_hierarchy_implementation_unlocked":s["feasible"]==s["frames"],"physics_unlocked":False,"training_unlocked":False,"next":"implement temporal hard-constraint hierarchy" if s["feasible"]==s["frames"] else "stop and identify infeasible support/frame class before temporal implementation"}
    a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"summary":s,"decision":result["decision"]},indent=2))
if __name__=="__main__": main()
