#!/usr/bin/env python3
"""Offline X2 contact/foot-placement/root/lower-q Stage-A generator.

No mj_step, policy forward, GPU, or learned optimizer is used.  Five frozen
contact templates are tried serially and the first complete geometry pass is
frozen.  Phase30 contact labels are intentionally ignored as physical truth.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, time
from pathlib import Path
import sys
import joblib, mujoco, numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import retarget.run_x2_forefoot_official_physics_screen as physics  # noqa:E402
from official_x2.replay_official_trace_direct_mujoco import official_start_pose  # noqa:E402

SIDES = ("left", "right")
LOWER15 = [
    *[f"{s}_{j}_joint" for s in SIDES for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")],
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
]
UPPER_BODIES = ("torso_link", "head_pitch_link", "left_wrist_roll_link", "right_wrist_roll_link")
TEMPLATES = ("DS", "DS-L-DS", "DS-R-DS", "DS-L-DS-R-DS", "DS-R-DS-L-DS")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    edge = np.diff(padded)
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def schedule(name: str, n: int) -> dict[str, np.ndarray]:
    active = {s: np.ones(n, dtype=bool) for s in SIDES}
    if name == "DS": return active
    if name in ("DS-L-DS", "DS-R-DS"):
        a, b = round(.2*n), round(.8*n); support = "left" if "-L-" in name else "right"
        active["right" if support == "left" else "left"][a:b] = False
        return active
    cuts = [round(x*n) for x in (.15, .35, .50, .70)]
    order = ("left", "right") if name == "DS-L-DS-R-DS" else ("right", "left")
    active["right" if order[0] == "left" else "left"][cuts[0]:cuts[1]] = False
    active["right" if order[1] == "left" else "left"][cuts[2]:cuts[3]] = False
    return active


class System:
    def __init__(self, n: int): self.n=n; self.r=[]; self.c=[]; self.v=[]; self.b=[]
    def add(self, coeff: dict[int,float], residual: float, weight: float):
        row=len(self.b); self.b.append(-weight*float(residual))
        for col,val in coeff.items():
            if val: self.r.append(row); self.c.append(int(col)); self.v.append(weight*float(val))
    def finish(self):
        return sparse.coo_matrix((self.v,(self.r,self.c)),shape=(len(self.b),self.n)).tocsr(), np.asarray(self.b)


def quat_apply(base_xyzw: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return (Rotation.from_rotvec(delta) * Rotation.from_quat(base_xyzw)).as_quat()


def init_entry(entry: dict, scale: float, model: mujoco.MjModel, feet: dict[str,list[int]], floor: int) -> dict:
    out=copy.deepcopy(entry); q=np.asarray(entry["dof"],float).copy(); names=list(entry["joint_names_mujoco"])
    neutral=official_start_pose()
    for name in ("left_hip_roll_joint","right_hip_roll_joint"):
        c=names.index(name); q[:,c]=neutral.get(name,0.0)+scale*(q[:,c]-neutral.get(name,0.0))
    root=np.asarray(entry["root_trans_offset"],float).copy(); quat=np.asarray(entry["root_rot"],float)
    qadr=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]
    d=mujoco.MjData(model); correction=[]
    for i in range(len(q)):
        d.qpos[:3]=root[i]; d.qpos[3:7]=quat[i][[3,0,1,2]]; d.qpos[qadr]=q[i]; mujoco.mj_forward(model,d)
        clear=min(np.min(d.geom_xpos[feet[s],2]-model.geom_size[feet[s],0]-d.geom_xpos[floor,2]) for s in SIDES)
        correction.append(.00505-clear)
    correction=np.convolve(np.pad(correction,(4,4),mode="edge"),np.ones(9)/9,mode="valid")
    root[:,2]+=correction; out["dof"]=q.astype(np.asarray(entry["dof"]).dtype); out["root_trans_offset"]=root.astype(np.asarray(entry["root_trans_offset"]).dtype)
    return out


def set_frame(model, data, entry, i, qadr):
    data.qpos[:3]=entry["root_trans_offset"][i]; data.qpos[3:7]=np.asarray(entry["root_rot"])[i][[3,0,1,2]]
    data.qpos[qadr]=entry["dof"][i]; mujoco.mj_forward(model,data)


def solve_template(model, initial, active, feet, floor, cfg):
    started=time.perf_counter(); n=len(initial["dof"]); names=list(initial["joint_names_mujoco"]); fps=float(initial["fps"])
    lower_cols=np.asarray([names.index(x) for x in LOWER15]); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in LOWER15])
    dofs=model.jnt_dofadr[jids]; qadrl=model.jnt_qposadr[jids]; qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in names])
    pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis"); upper=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,x) for x in UPPER_BODIES]
    width=21; segs={s:segments(active[s]) for s in SIDES}; anchor_offset=n*width
    lookup={s:np.full(n,-1,int) for s in SIDES}; anchor_index={}; count=0
    for s in SIDES:
        for j,(a,b) in enumerate(segs[s]): lookup[s][a:b]=j; anchor_index[(s,j)]=count; count+=1
    total=anchor_offset+2*count; root0=np.asarray(initial["root_trans_offset"],float); quat0=np.asarray(initial["root_rot"],float); q0=np.asarray(initial["dof"],float)
    dr=np.zeros((n,3)); drot=np.zeros((n,3)); dq=np.zeros((n,15)); candidate=copy.deepcopy(initial); data=mujoco.MjData(model)
    upper_ref=[]; foot_ref={s:[] for s in SIDES}
    for i in range(n):
        set_frame(model,data,initial,i,qadr); upper_ref.append(data.xpos[upper].copy())
        for s in SIDES: foot_ref[s].append(np.mean(data.geom_xpos[feet[s]],axis=0))
    upper_ref=np.asarray(upper_ref); foot_ref={s:np.asarray(v) for s,v in foot_ref.items()}
    anchors={s:np.asarray([np.mean(foot_ref[s][a:b,:2],axis=0) for a,b in segs[s]]) for s in SIDES}
    iters=[]; W=cfg["weights_sqrt"]
    for iteration in range(cfg["solver"]["iterations"]):
        sysm=System(total); data=mujoco.MjData(model)
        for i in range(n):
            set_frame(model,data,candidate,i,qadr); cols=np.arange(i*width,(i+1)*width); sel=np.r_[np.arange(6),dofs]
            comjac=np.zeros((3,model.nv)); mujoco.mj_jacSubtreeCom(model,data,comjac,pelvis); com=data.subtree_com[pelvis].copy()
            active_s=[s for s in SIDES if active[s][i]]; center=[]
            for s in active_s:
                j=int(lookup[s][i]); center.append(anchors[s][j])
            center=np.mean(center,axis=0)
            for axis in range(2):
                coeff={int(c):float(v) for c,v in zip(cols,comjac[axis,sel]) if v}
                for s in active_s:
                    ac=anchor_offset+2*anchor_index[(s,int(lookup[s][i]))]+axis; coeff[ac]=coeff.get(ac,0)-1/len(active_s)
                sysm.add(coeff,com[axis]-center[axis],W["com_xy"])
            for s in SIDES:
                positions=data.geom_xpos[feet[s]]; centroid=np.mean(positions,axis=0); jac=np.zeros((3,model.nv))
                for g in feet[s]:
                    jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacGeom(model,data,jp,jr,g); jac+=jp/len(feet[s])
                signed=np.min(positions[:,2]-model.geom_size[feet[s],0]-data.geom_xpos[floor,2])
                if active[s][i]:
                    j=int(lookup[s][i]); ac=anchor_offset+2*anchor_index[(s,j)]
                    for axis in range(2):
                        coeff={int(c):float(v) for c,v in zip(cols,jac[axis,sel]) if v}; coeff[ac+axis]=-1
                        sysm.add(coeff,centroid[axis]-anchors[s][j,axis],W["stance_xyz"])
                    sysm.add({int(c):float(v) for c,v in zip(cols,jac[2,sel]) if v},signed,W["stance_xyz"])
                elif signed < cfg["hard_gates"]["swing_clearance_min_m"]:
                    sysm.add({int(c):float(-v) for c,v in zip(cols,jac[2,sel]) if v},cfg["hard_gates"]["swing_clearance_min_m"]-signed,W["swing_clearance"])
            for body,ref in zip(upper,upper_ref[i]):
                jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacBody(model,data,jp,jr,body)
                for axis in range(3): sysm.add({int(c):float(v) for c,v in zip(cols,jp[axis,sel]) if v},data.xpos[body,axis]-ref[axis],W["upper_keypoint"])
            for k in range(6): sysm.add({int(cols[k]):1},np.r_[dr[i],drot[i]][k],W["root_reference"])
            for k in range(15): sysm.add({int(cols[6+k]):1},dq[i,k],W["joint_reference"])
        state=np.c_[np.asarray(candidate["root_trans_offset"]),drot,np.asarray(candidate["dof"])[:,lower_cols]]
        for i in range(1,n):
            for k in range(width): sysm.add({(i-1)*width+k:-1,i*width+k:1},state[i,k]-state[i-1,k],W["velocity"])
        for i in range(1,n-1):
            for k in range(width): sysm.add({(i-1)*width+k:1,i*width+k:-2,(i+1)*width+k:1},state[i-1,k]-2*state[i,k]+state[i+1,k],W["acceleration"])
        for s in SIDES:
            for j,(a,b) in enumerate(segs[s]):
                ref=np.mean(foot_ref[s][a:b,:2],axis=0); ac=anchor_offset+2*anchor_index[(s,j)]
                for axis in range(2): sysm.add({ac+axis:1},anchors[s][j,axis]-ref[axis],W["anchor_reference"])
        A,b=sysm.finish(); sol=lsqr(A,b,atol=cfg["solver"]["lsqr_atol"],btol=cfg["solver"]["lsqr_btol"],iter_lim=cfg["solver"]["lsqr_iteration_limit"])
        step=cfg["solver"]["step_scale"]*sol[0]; traj=step[:anchor_offset].reshape(n,width); dr+=traj[:,:3]; drot+=traj[:,3:6]; dq+=traj[:,6:]
        bounds=cfg["bounds"]; norm=np.linalg.norm(dr[:,:2],axis=1); dr[:,:2]*=np.minimum(1,bounds["root_xy_correction_m"]/np.maximum(norm,1e-12))[:,None]
        dr[:,2]=np.clip(dr[:,2],-bounds["root_z_correction_m"],bounds["root_z_correction_m"]); rn=np.linalg.norm(drot,axis=1); drot*=np.minimum(1,bounds["root_rotation_correction_rad"]/np.maximum(rn,1e-12))[:,None]
        dq=np.clip(dq,-bounds["joint_correction_rad"],bounds["joint_correction_rad"]); limits=model.jnt_range[jids]
        dq=np.maximum(dq,limits[:,0]-q0[:,lower_cols]); dq=np.minimum(dq,limits[:,1]-q0[:,lower_cols])
        for s in SIDES:
            for j in range(len(segs[s])):
                ac=anchor_offset+2*anchor_index[(s,j)]; anchors[s][j]+=step[ac:ac+2]
        candidate=copy.deepcopy(initial); candidate["root_trans_offset"]=(root0+dr).astype(np.asarray(initial["root_trans_offset"]).dtype)
        candidate["root_rot"]=np.asarray([quat_apply(quat0[i],drot[i]) for i in range(n)],dtype=np.asarray(initial["root_rot"]).dtype)
        qq=q0.copy(); qq[:,lower_cols]+=dq; candidate["dof"]=qq.astype(np.asarray(initial["dof"]).dtype)
        iters.append({"iteration":iteration+1,"lsqr_iterations":int(sol[2]),"residual_norm":float(sol[3]),"step_l2":float(np.linalg.norm(step))})
    return candidate,{"wall_time_s":time.perf_counter()-started,"iterations":iters,"segments":{s:segs[s] for s in SIDES}}


def audit(model, original, candidate, active, feet, floor, cfg):
    n=len(candidate["dof"]); fps=float(candidate["fps"]); names=list(candidate["joint_names_mujoco"]); qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in names]); pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis"); upper=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,x) for x in UPPER_BODIES]
    data=mujoco.MjData(model); data0=mujoco.MjData(model); foot={s:[] for s in SIDES}; dist={s:[] for s in SIDES}; com=[]; upper_err=[]
    for i in range(n):
        set_frame(model,data,candidate,i,qadr); set_frame(model,data0,original,i,qadr); com.append(data.subtree_com[pelvis].copy()); upper_err.extend(np.linalg.norm(data.xpos[upper]-data0.xpos[upper],axis=1))
        for s in SIDES:
            foot[s].append(np.mean(data.geom_xpos[feet[s]],axis=0)); dist[s].append(np.min(data.geom_xpos[feet[s],2]-model.geom_size[feet[s],0]-data.geom_xpos[floor,2]))
    com=np.asarray(com); foot={s:np.asarray(v) for s,v in foot.items()}; dist={s:np.asarray(v) for s,v in dist.items()}; radius=float(model.geom_size[feet["left"][0],0]); margins=[]
    stance_d=[]; speeds=[]; excursions=[]; swing=[]; longest_ss=0
    for s in SIDES:
        for a,b in segments(active[s]):
            stance_d.extend(np.abs(dist[s][a:b])); excursions.append(float(np.max(np.linalg.norm(foot[s][a:b,:2]-foot[s][a,:2],axis=1))))
            if b-a>1: speeds.extend(np.linalg.norm(np.diff(foot[s][a:b,:2],axis=0),axis=1)*fps)
        swing.extend(dist[s][~active[s]])
    for i in range(n):
        act=[s for s in SIDES if active[s][i]]
        if len(act)==1: longest_ss=max(longest_ss,max((b-a for a,b in segments(active[act[0]] & ~active["right" if act[0]=="left" else "left"]) if a<=i<b),default=0))
        if len(act)==1:
            pts=data_pts=np.asarray([])
            set_frame(model,data,candidate,i,qadr); pts=data.geom_xpos[feet[act[0]],:2]; eq=ConvexHull(pts).equations; margins.append(float(np.min(-(eq[:,:2]@com[i,:2]+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1))+radius))
    q=np.asarray(candidate["dof"],float); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in names]); limits=model.jnt_range[jids]; violation=np.maximum(limits[:,0]-q,0)+np.maximum(q-limits[:,1],0)
    root=np.asarray(candidate["root_trans_offset"],float); acc=np.linalg.norm(np.diff(root[:,:2],n=2,axis=0),axis=1)*fps*fps
    actual_contact=(dist["left"]<=.0005)|(dist["right"]<=.0005); gates=cfg["hard_gates"]
    metrics={"joint_limit_overshoot_max_rad":float(np.max(violation)),"joint_step_p95_rad":float(np.percentile(np.max(np.abs(np.diff(q,axis=0)),axis=1),95)),"single_support_dwell_max_s":longest_ss/fps,"stance_contact_abs_distance_p95_m":float(np.percentile(stance_d,95)),"single_support_com_margin_min_m":None if not margins else float(min(margins)),"stance_speed_p95_mps":float(np.percentile(speeds,95)),"stance_excursion_max_m":float(max(excursions)),"swing_clearance_min_m":None if not swing else float(min(swing)),"flight_fraction":float(np.mean(~actual_contact)),"root_horizontal_accel_p95_mps2":float(np.percentile(acc,95)),"upper_keypoint_p95_m":float(np.percentile(upper_err,95)),"head_lock_max_rad":float(np.max(np.abs(q[:,-2:])))}
    checks={"joint_limits":metrics["joint_limit_overshoot_max_rad"]<=1e-9,"joint_step":metrics["joint_step_p95_rad"]<=gates["joint_step_p95_rad"],"single_support_dwell":metrics["single_support_dwell_max_s"]>=gates["single_support_dwell_min_s"],"stance_contact":metrics["stance_contact_abs_distance_p95_m"]<=gates["stance_contact_abs_distance_p95_m"],"com_margin":metrics["single_support_com_margin_min_m"] is not None and metrics["single_support_com_margin_min_m"]>=0,"stance_speed":metrics["stance_speed_p95_mps"]<=gates["stance_speed_p95_mps"],"stance_excursion":metrics["stance_excursion_max_m"]<=gates["stance_excursion_max_m"],"swing_clearance":metrics["swing_clearance_min_m"] is not None and metrics["swing_clearance_min_m"]>=gates["swing_clearance_min_m"],"flight":metrics["flight_fraction"]<=gates["flight_fraction_max"],"root_accel":metrics["root_horizontal_accel_p95_mps2"]<=gates["root_horizontal_accel_p95_mps2"],"upper":metrics["upper_keypoint_p95_m"]<=gates["upper_keypoint_p95_m"],"head":metrics["head_lock_max_rad"]<=gates["head_lock_max_rad"]}
    return metrics,checks,bool(all(checks.values()))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); p.add_argument("--motion",type=Path,required=True); p.add_argument("--phase15",type=Path,required=True); p.add_argument("--motion-id",default="PHUMA-LUNGE-R-001"); p.add_argument("--preflight-only",action="store_true"); p.add_argument("--output",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); a=p.parse_args()
    cfg=json.loads((Path(__file__).with_name("prereg_phase18_joint_contact_generator.json")).read_text()); model=mujoco.MjModel.from_xml_path(str(a.scene)); floor,helper=physics.foot_geom_contract(model); feet={s:sorted(g for g in helper[s] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]!=0) for s in SIDES}
    entry=joblib.load(a.motion)[a.motion_id]; scale=json.loads(a.phase15.read_text())["intervention"]["hip_roll_scale"]; initial=init_entry(entry,scale,model,feet,floor)
    pre={"active12":all(len(feet[s])==12 for s in SIDES),"frames":len(entry["dof"]),"joint_order_31":len(entry["joint_names_mujoco"])==31,"lower15_present":all(x in entry["joint_names_mujoco"] for x in LOWER15),"templates":list(TEMPLATES),"pass":True}; pre["pass"]=all(v for k,v in pre.items() if k not in ("frames","templates"))
    if a.preflight_only:
        a.output.write_text(json.dumps({"preflight":pre},indent=2)+"\n"); return
    rows=[]; frozen=None
    for name in TEMPLATES:
        active=schedule(name,len(entry["dof"])); cand,diag=solve_template(model,initial,active,feet,floor,cfg); metrics,checks,passed=audit(model,initial,cand,active,feet,floor,cfg)
        rows.append({"template":name,"diagnostics":diag,"metrics":metrics,"checks":checks,"complete_pass":passed}); print(name,passed,metrics,flush=True)
        if passed: frozen=cand; break
    if frozen is not None: a.candidate.parent.mkdir(parents=True,exist_ok=True); joblib.dump({a.motion_id:frozen},a.candidate,compress=True)
    result={"stage":"dynamic retargeting Phase18 Stage-A joint generator","execution":{"mj_step_calls":0,"policy_forwards":0,"optimizer_updates":0,"gpu":False},"assets":{"scene":str(a.scene),"scene_sha256":sha(a.scene),"motion":str(a.motion),"motion_sha256":sha(a.motion),"phase15":str(a.phase15),"phase15_sha256":sha(a.phase15)},"truth_boundary":{"source_contact_labels_are_truth":False,"human_upper_and_torso_are_soft_targets":True,"official_geometry_and_limits_are_hard":True},"preflight":pre,"templates":rows,"decision":{"stage_a_passed":frozen is not None,"frozen_template":None if frozen is None else rows[-1]["template"],"physics_unlocked":False,"training_unlocked":False,"next":"review Stage-A only; Stage-B remains unauthorized" if frozen is not None else "stop generator without physics; do not scan weights or timings"}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
