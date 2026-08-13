#!/usr/bin/env python3
"""Phase20 block-nullspace temporal hard-contact generator (offline only)."""
from __future__ import annotations
import argparse, copy, importlib.util, json, time
from pathlib import Path
import joblib, mujoco, numpy as np
from scipy.sparse.linalg import lsqr

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
P18=HERE/"phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec=importlib.util.spec_from_file_location("p18",P18); p18=importlib.util.module_from_spec(spec); spec.loader.exec_module(p18)

def frame_geometry(model,data,entry,i,qadr,feet,floor,active,pelvis,dofs):
    p18.set_frame(model,data,entry,i,qadr); sel=np.r_[np.arange(6),dofs]
    com=data.subtree_com[pelvis].copy(); cj=np.zeros((3,model.nv)); mujoco.mj_jacSubtreeCom(model,data,cj,pelvis)
    foot={}; fj={}; signed={}; sj={}
    for s in p18.SIDES:
        pos=data.geom_xpos[feet[s]]; foot[s]=np.mean(pos,axis=0); full=np.zeros((3,model.nv))
        for g in feet[s]:
            jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacGeom(model,data,jp,jr,g); full+=jp/len(feet[s])
        fj[s]=full[:,sel]; d=pos[:,2]-model.geom_size[feet[s],0]-data.geom_xpos[floor,2]; k=int(np.argmin(d)); signed[s]=float(d[k])
        jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacGeom(model,data,jp,jr,feet[s][k]); sj[s]=jp[2,sel]
    act=[s for s in p18.SIDES if active[s][i]]; J=[]; c=[]
    for s in act: J.append(sj[s]); c.append(signed[s])
    center=np.mean([foot[s][:2] for s in act],axis=0); centerj=np.mean([fj[s][:2] for s in act],axis=0)
    for axis in range(2): J.append(cj[axis,sel]-centerj[axis]); c.append(com[axis]-center[axis])
    for s in p18.SIDES:
        if s not in act and signed[s]<.012: J.append(sj[s]); c.append(signed[s]-.012)
    J=np.asarray(J); c=np.asarray(c); U,S,Vt=np.linalg.svd(J,full_matrices=True); rank=int(np.sum(S>1e-9)); projection=np.linalg.lstsq(J,-c,rcond=None)[0]; Z=Vt[rank:].T
    return {"J":J,"c":c,"p":projection,"Z":Z,"foot":foot,"fj":fj,"active":act,"rank":rank}

def solve(model,initial,active,feet,floor,cfg):
    n=len(initial["dof"]); names=list(initial["joint_names_mujoco"]); fps=float(initial["fps"]); q0=np.asarray(initial["dof"],float); root0=np.asarray(initial["root_trans_offset"],float); quat0=np.asarray(initial["root_rot"],float)
    lower=np.asarray([names.index(x) for x in p18.LOWER15]); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in p18.LOWER15]); dofs=model.jnt_dofadr[jids]; limits=model.jnt_range[jids]
    qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in names]); pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    dr=np.zeros((n,3)); drot=np.zeros((n,3)); dq=np.zeros((n,15)); candidate=copy.deepcopy(initial); iterations=[]; previous_violation=None; nondecrease=0; started=time.perf_counter(); W=cfg["method"]["secondary_weights_sqrt"]
    for outer in range(cfg["method"]["outer_iterations"]):
        data=mujoco.MjData(model); frames=[frame_geometry(model,data,candidate,i,qadr,feet,floor,active,pelvis,dofs) for i in range(n)]; dims=[f["Z"].shape[1] for f in frames]; offsets=np.cumsum([0]+dims); system=p18.System(int(offsets[-1])); base=np.c_[np.asarray(candidate["root_trans_offset"],float),drot,np.asarray(candidate["dof"],float)[:,lower]]; correction=np.c_[dr,drot,dq]
        projected=np.asarray([base[i]+frames[i]["p"] for i in range(n)])
        for i,f in enumerate(frames):
            Z=f["Z"]; off=int(offsets[i])
            for k in range(21): system.add({off+j:Z[k,j] for j in range(Z.shape[1]) if Z[k,j]!=0},correction[i,k]+f["p"][k],W["correction"])
        for i in range(1,n):
            for k in range(21):
                coeff={};
                for frame,sign in ((i,1),(i-1,-1)):
                    Z=frames[frame]["Z"]; off=int(offsets[frame]); coeff.update({off+j:coeff.get(off+j,0)+sign*Z[k,j] for j in range(Z.shape[1]) if Z[k,j]!=0})
                system.add(coeff,projected[i,k]-projected[i-1,k],W["state_velocity"])
            for s in p18.SIDES:
                if active[s][i] and active[s][i-1]:
                    pred=[]; coeffs=[]
                    for frame,sign in ((i,1),(i-1,-1)):
                        f=frames[frame]; pred.append(sign*(f["foot"][s]+f["fj"][s]@f["p"])); coeffs.append((frame,sign*f["fj"][s]@f["Z"]))
                    residual=sum(pred)
                    for k in range(3):
                        coeff={}
                        for frame,mat in coeffs:
                            off=int(offsets[frame]); coeff.update({off+j:coeff.get(off+j,0)+mat[k,j] for j in range(mat.shape[1]) if mat[k,j]!=0})
                        system.add(coeff,residual[k],W["stance_foot_velocity"])
        for i in range(1,n-1):
            for k in range(21):
                coeff={}
                for frame,sign in ((i-1,1),(i,-2),(i+1,1)):
                    Z=frames[frame]["Z"]; off=int(offsets[frame]); coeff.update({off+j:coeff.get(off+j,0)+sign*Z[k,j] for j in range(Z.shape[1]) if Z[k,j]!=0})
                system.add(coeff,projected[i-1,k]-2*projected[i,k]+projected[i+1,k],W["state_acceleration"])
        A,b=system.finish(); ls=lsqr(A,b,atol=cfg["method"]["lsqr"]["atol"],btol=cfg["method"]["lsqr"]["btol"],iter_lim=cfg["method"]["lsqr"]["iteration_limit"]); z=cfg["method"]["lsqr"]["step_scale"]*ls[0]
        dx=[]
        for i,f in enumerate(frames):
            # Hierarchy is strict: never shrink the hard projection merely
            # because the secondary null-space step is large.  If p itself
            # consumes the trust region, postpone all smoothing for this frame.
            hard=f["p"]; secondary=f["Z"]@z[offsets[i]:offsets[i+1]]
            hard_max=float(np.max(np.abs(hard)))
            if hard_max>.10:
                step=hard*(.10/hard_max)
            else:
                remaining=.10-hard_max
                secondary_max=float(np.max(np.abs(secondary)))
                step=hard+secondary*min(1.0,remaining/max(secondary_max,1e-12))
            dx.append(step)
        dx=np.asarray(dx); dr+=dx[:,:3]; drot+=dx[:,3:6]; dq+=dx[:,6:]
        bounds={"xy":.25,"z":.12,"rot":.35,"q":.65}; norm=np.linalg.norm(dr[:,:2],axis=1); dr[:,:2]*=np.minimum(1,bounds["xy"]/np.maximum(norm,1e-12))[:,None]; dr[:,2]=np.clip(dr[:,2],-bounds["z"],bounds["z"]); rn=np.linalg.norm(drot,axis=1); drot*=np.minimum(1,bounds["rot"]/np.maximum(rn,1e-12))[:,None]; dq=np.clip(dq,-bounds["q"],bounds["q"]); dq=np.maximum(dq,limits[:,0]-q0[:,lower]); dq=np.minimum(dq,limits[:,1]-q0[:,lower])
        candidate=copy.deepcopy(initial); candidate["root_trans_offset"]=(root0+dr).astype(np.asarray(initial["root_trans_offset"]).dtype); candidate["root_rot"]=np.asarray([p18.quat_apply(quat0[i],drot[i]) for i in range(n)],dtype=np.asarray(initial["root_rot"]).dtype); qq=q0.copy(); qq[:,lower]+=dq; candidate["dof"]=qq.astype(np.asarray(initial["dof"]).dtype)
        metrics,checks,passed=p18.audit(model,initial,candidate,active,feet,floor,{"hard_gates":json.loads((HERE/"phase18_joint_contact_generator/prereg_phase18_joint_contact_generator.json").read_text())["hard_gates"]}); violation=max(metrics["stance_contact_abs_distance_p95_m"]/.0005,max(0,-(metrics["single_support_com_margin_min_m"] or 0))/.001,max(0,.012-(metrics["swing_clearance_min_m"] or -1))/.012)
        nondecrease=nondecrease+1 if previous_violation is not None and violation>=previous_violation-1e-6 else 0; previous_violation=violation; iterations.append({"outer":outer+1,"lsqr_iterations":int(ls[2]),"hard_violation_score":float(violation),"metrics":metrics,"complete_pass":passed})
        if passed or nondecrease>=2: break
    return candidate,iterations,time.perf_counter()-started

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); p.add_argument("--motion",type=Path,required=True); p.add_argument("--phase15",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--motion-id",default="PHUMA-LUNGE-R-001"); a=p.parse_args(); cfg=json.loads((HERE/"phase20_temporal_hard_contact_contract.json").read_text()); model=mujoco.MjModel.from_xml_path(str(a.scene)); floor,helper=p18.physics.foot_geom_contract(model); feet={s:sorted(g for g in helper[s] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]!=0) for s in p18.SIDES}; raw=joblib.load(a.motion)[a.motion_id]; scale=json.loads(a.phase15.read_text())["intervention"]["hip_roll_scale"]; initial=p18.init_entry(raw,scale,model,feet,floor); active=p18.schedule("DS-L-DS-R-DS",len(raw["dof"])); candidate,iters,wall=solve(model,initial,active,feet,floor,cfg); passed=bool(iters[-1]["complete_pass"]); result={"stage":"Phase20 temporal hard-contact hierarchy","execution":{"mj_step_calls":0,"gpu":False,"outer_iterations":len(iters),"wall_time_s":wall},"iterations":iters,"decision":{"stage_a_passed":passed,"physics_unlocked":False,"training_unlocked":False,"next":"review only" if passed else "stop without tuning"}}; a.output.write_text(json.dumps(result,indent=2)+"\n");
    if passed: joblib.dump({a.motion_id:candidate},a.candidate,compress=True)
if __name__=="__main__": main()
