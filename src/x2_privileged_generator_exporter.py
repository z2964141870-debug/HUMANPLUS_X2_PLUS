"""Strict rollout exporter for dynamically generated X2 references."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

class PhysicalGeneratorRolloutBuffer:
 def __init__(self):self.rows=[]
 def append(self,*,time_s,qpos,qvel,pd_target31,contact_lr,normal_force_lr):
  arrays=[np.asarray(qpos,float),np.asarray(qvel,float),np.asarray(pd_target31,float),np.asarray(contact_lr,bool),np.asarray(normal_force_lr,float)]
  if [a.shape for a in arrays]!=[(38,),(37,),(31,),(2,),(2,)]:raise ValueError('X2 rollout row shape mismatch')
  if not all(np.all(np.isfinite(a)) for a in (arrays[0],arrays[1],arrays[2],arrays[4])):raise ValueError('non-finite rollout row')
  if self.rows and float(time_s)<=self.rows[-1][0]:raise ValueError('rollout time must be strictly increasing')
  self.rows.append((float(time_s),*arrays))
 def arrays(self):
  if not self.rows:raise ValueError('empty rollout')
  return {k:np.asarray(v) for k,v in zip(('time_s','qpos','qvel','pd_target31','contact_lr','normal_force_lr'),zip(*self.rows))}
 def save(self,path:Path,manifest_path:Path,*,source_hashes:dict[str,str]):
  data=self.arrays();np.savez_compressed(path,**data);h=hashlib.sha256(path.read_bytes()).hexdigest();manifest={'schema':'x2_privileged_physical_generator_rollout_v1','frames':len(data['time_s']),'artifact':str(path),'artifact_sha256':h,'source_hashes':dict(source_hashes),'contact_semantics':'realized simulator collision/force only; no reference contact labels'};manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');return manifest
