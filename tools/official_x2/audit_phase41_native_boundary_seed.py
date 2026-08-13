#!/usr/bin/env python3
"""Select one coherent Phase34 native boundary for the keyframe skeleton."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import joblib, mujoco, numpy as np
from scipy.spatial import ConvexHull
from official_x2.audit_stage250_native_dynamic_seed import ISAAC_JOINTS, decode_row

REPO=Path(__file__).resolve().parents[2]; P18=REPO/"research/dynamic_retargeting_20260811/phase18_joint_contact_generator/run_phase18_joint_contact_generator.py"
spec=importlib.util.spec_from_file_location("p18",P18); p18=importlib.util.module_from_spec(spec); spec.loader.exec_module(p18)
def margin(point,centers,radius):
    eq=ConvexHull(centers).equations; return float(np.min(-(eq[:,:2]@point+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1))+radius)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); p.add_argument("--rollout",type=Path,required=True); p.add_argument("--motion",type=Path,required=True); p.add_argument("--phase15",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--motion-id",default="PHUMA-LUNGE-R-001"); a=p.parse_args()
    model=mujoco.MjModel.from_xml_path(str(a.scene)); data=mujoco.MjData(model); qpa={x:int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)]) for x in ISAAC_JOINTS}; dva={x:int(model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)]) for x in ISAAC_JOINTS}
    floor,helper=p18.physics.foot_geom_contract(model); feet={s:sorted(g for g in helper[s] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]!=0) for s in p18.SIDES}; pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    raw=joblib.load(a.motion)[a.motion_id]; scale=json.loads(a.phase15.read_text())["intervention"]["hip_roll_scale"]; initial=p18.init_entry(raw,scale,model,feet,floor); names=list(initial["joint_names_mujoco"]); lower=[names.index(x) for x in p18.LOWER15]; desired=.5*(np.asarray(initial["dof"])[0,lower]+np.asarray(initial["dof"])[-1,lower])
    trace=json.loads(a.rollout.read_text())["trace"]; candidates=[]
    for i,row in enumerate(trace):
        future=trace[i:i+51]
        if len(future)<51 or not all(x["stage"] in ("stand","move","stop") and x["root_z_m"]>=.55 and x["root_tilt_rad"]<=.30 for x in future): continue
        qpos,qvel=decode_row(model,row,qpa,dva); qlower=np.asarray([qpos[qpa[x]] for x in p18.LOWER15]); candidates.append((float(np.linalg.norm(qlower-desired)),i,row,qpos,qvel))
    distance,index,row,qpos,qvel=min(candidates,key=lambda x:(x[0],x[1])); data.qpos[:]=qpos; data.qvel[:]=qvel; mujoco.mj_forward(model,data); signed={s:(data.geom_xpos[feet[s],2]-model.geom_size[feet[s],0]-data.geom_xpos[floor,2]) for s in p18.SIDES}; centers=np.concatenate([data.geom_xpos[feet[s],:2] for s in p18.SIDES]); radius=float(model.geom_size[feet["left"][0],0]); jids=np.asarray([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x) for x in ISAAC_JOINTS]); q=np.asarray([qpos[qpa[x]] for x in ISAAC_JOINTS]); limits=model.jnt_range[jids]; over=np.maximum(limits[:,0]-q,0)+np.maximum(q-limits[:,1],0)
    result={"stage":"Phase41 coherent native boundary seed","execution":{"mj_forward_calls":1,"mj_step_calls":0,"gpu":False},"selection":{"eligible_coherent_anchors":len(candidates),"formula":"minimum lower15 L2 to mean of Phase15 first/last lower15; earliest index tie-break; no solver-result selection","trace_index":index,"stage":row["stage"],"elapsed_s":row["elapsed_s"],"lower15_l2_to_phase15_endpoint_mean_rad":distance},"boundary":{"qpos":qpos.tolist(),"qvel":qvel.tolist(),"active12_signed_distance_min_m":{s:float(np.min(signed[s])) for s in p18.SIDES},"active12_signed_distance_p95_m":{s:float(np.percentile(np.abs(signed[s]),95)) for s in p18.SIDES},"double_support_com_margin_m":margin(data.subtree_com[pelvis,:2],centers,radius),"joint_limit_overshoot_max_rad":float(np.max(over)),"root_z_m":float(qpos[2]),"root_tilt_rad":float(row["root_tilt_rad"])},"decision":{}}
    b=result["boundary"]; result["decision"]={"coherent_1s_native_boundary_selected":True,"joint_limits_pass":b["joint_limit_overshoot_max_rad"]<=1e-9,"double_support_margin_pass":b["double_support_com_margin_m"]>=0,"physics_unlocked":False,"training_unlocked":False}; a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"selection":result["selection"],"boundary":b,"decision":result["decision"]},indent=2))
if __name__=="__main__": main()
