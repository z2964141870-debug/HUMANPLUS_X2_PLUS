#!/usr/bin/env python3
"""Compare BASE bridge endpoints with coherent 1s-safe Stage250 states."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from official_x2.analyze_phase34_full_closed_trace import JOINTS, default_pose, yaw_tilt

GROUPS=("joint_position","joint_velocity","issued_action","projected_gravity","root_posture")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gravity(qwxyz):
    q=np.asarray(qwxyz); r=Rotation.from_quat(q[[1,2,3,0]]); return r.as_matrix().T@np.array([0.,0.,-1.])
def stage_features(row):
    obs=np.asarray(row["obs"],float); neutral=np.asarray([default_pose()[x] for x in JOINTS])
    return {"joint_position":neutral+obs[12:43],"joint_velocity":obs[43:74],"issued_action":np.asarray(row["action"],float),"projected_gravity":obs[6:9],"root_posture":np.asarray([row["root_z_m"],row["root_tilt_rad"]])}
def bridge_features(row):
    qpos=np.asarray(row["qpos"],float); qvel=np.asarray(row["qvel"],float)
    return {"joint_position":qpos[7:38],"joint_velocity":qvel[6:37],"issued_action":np.asarray(row["action"],float),"projected_gravity":gravity(qpos[3:7]),"root_posture":np.asarray([qpos[2],yaw_tilt(qpos[3:7])[1]])}
def loo_threshold(values):
    d=[]
    for i in range(len(values)):
        delta=np.linalg.norm(values-values[i],axis=1); delta[i]=np.inf; d.append(np.min(delta))
    return float(np.percentile(d,95))
def run(rollout_path,phase26_path,phase39_path):
    rollout=json.loads(Path(rollout_path).read_text()); trace=[r for r in rollout["trace"] if r["stage"] in ("stand","move","stop") and r["root_z_m"]>=.55 and r["root_tilt_rad"]<=.30]
    # A coherent anchor must retain 50 subsequent telemetry samples in the safe set.
    source_indices={id(r):i for i,r in enumerate(rollout["trace"])}; safe_ids={id(r) for r in trace}; anchors=[]
    for r in trace:
        i=source_indices[id(r)]; future=rollout["trace"][i:i+51]
        if len(future)==51 and all(x["stage"] in ("stand","move","stop") and x["root_z_m"]>=.55 and x["root_tilt_rad"]<=.30 for x in future): anchors.append(r)
    feats=[stage_features(r) for r in trace]; arrays={g:np.asarray([x[g] for x in feats]) for g in GROUPS}; thresholds={g:loo_threshold(arrays[g]) for g in GROUPS}
    anchor_features=[stage_features(r) for r in anchors]
    p26=json.loads(Path(phase26_path).read_text()); p39=json.loads(Path(phase39_path).read_text())
    endpoints={"phase26_best":p26["best"]["rollout"]["bridge_rows"][-1],"phase39_sequence":p39["rollout"]["bridge_rows"][-1]}
    out={}
    for name,row in endpoints.items():
        x=bridge_features(row); per=[]
        for i,a in enumerate(anchor_features):
            distance={g:float(np.linalg.norm(x[g]-a[g])) for g in GROUPS}; score={g:distance[g]/max(thresholds[g],1e-12) for g in GROUPS}; per.append((max(score.values()),i,score,distance))
        composite,i,score,distance=min(per,key=lambda x:x[0]); target=anchors[i]
        out[name]={"coherent_minimax_normalized":composite,"all_groups_within_stage250_loo_p95":composite<=1,"per_group_distance":distance,"per_group_normalized":score,"nearest_anchor":{"trace_index":source_indices[id(target)],"stage":target["stage"],"elapsed_s":target["elapsed_s"],"root_z_m":target["root_z_m"],"root_tilt_rad":target["root_tilt_rad"]}}
    return {"stage":"BASE Phase40 Stage250 coherent success-basin audit","execution":{"physics_steps":0,"optimizer_steps":0,"gpu":False},"assets":{"rollout_sha256":sha(rollout_path),"phase26_sha256":sha(phase26_path),"phase39_sha256":sha(phase39_path)},"contract":{"reference":"Phase34 full-gate closed AimDK Stage250 trace","features":list(GROUPS),"normalization":"Phase34 safe-state within-trace leave-one-out nearest-neighbor p95 per group","coherence":"one shared reference row with next 50 telemetry samples safe; no cross-time nearest-neighbor stitching","root_xy_yaw_excluded":True},"reference":{"safe_rows":len(trace),"coherent_1s_anchor_rows":len(anchors),"loo_p95":thresholds},"endpoints":out,"decision":{"phase26_in_stage250_coherent_basin":out["phase26_best"]["all_groups_within_stage250_loo_p95"],"phase39_in_stage250_coherent_basin":out["phase39_sequence"]["all_groups_within_stage250_loo_p95"],"physics_unlocked":False,"training_unlocked":False}}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--rollout",type=Path,required=True); p.add_argument("--phase26",type=Path,required=True); p.add_argument("--phase39",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); r=run(a.rollout,a.phase26,a.phase39); a.output.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps({"reference":r["reference"],"endpoints":r["endpoints"],"decision":r["decision"]},indent=2))
if __name__=="__main__": main()
