"""Immutable state-only seed hook for an X2 privileged physical generator."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch
from x2_faithful_any2any_phase23 import WBT29PolicyContract

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

@dataclass(frozen=True)
class NativeGeneratorSeedSpec:
 path:Path;sha256:str;optimizer_eligible:bool=False

class NativeGeneratorSeedHook:
 def __init__(self,spec:NativeGeneratorSeedSpec,contract:WBT29PolicyContract):
  self.spec=spec;self.contract=contract
  if spec.optimizer_eligible:raise ValueError('native basin seed is reset/reference-only, never optimizer data')
 def load(self):
  if sha256(self.spec.path)!=self.spec.sha256:raise ValueError('native seed hash drift')
  z=np.load(self.spec.path,allow_pickle=False);names=tuple(z['joint_names'].tolist())
  if names!=self.contract.official31:raise ValueError('native seed official31 order drift')
  required={'time_s','qpos','qvel','root_pose','root_velocity','joint_names','joint_pos31','joint_vel31'}
  if not required.issubset(z.files):raise ValueError('native seed fields missing')
  self.data={k:np.asarray(z[k]).copy() for k in required};return {'frames':len(self.data['time_s']),'sha256':self.spec.sha256,'optimizer_eligible':False,'contact_labels_present':False}
 def future_source29(self,frames:torch.Tensor,horizon:int=10)->torch.Tensor:
  if not hasattr(self,'data'):self.load()
  idx=frames.to(dtype=torch.long,device='cpu').reshape(-1,1)+torch.arange(horizon).reshape(1,-1)
  idx=torch.clamp(idx,0,len(self.data['time_s'])-1)
  q=torch.from_numpy(self.data['joint_pos31'])[idx];dq=torch.from_numpy(self.data['joint_vel31'])[idx]
  q29=self.contract.official_to_source(q);dq29=self.contract.official_to_source(dq)
  return torch.cat((q29,dq29),dim=-1)
 def reset_state(self,frames:torch.Tensor)->dict[str,torch.Tensor]:
  """Return explicit simulator reset tensors; never synthesize contact state."""
  if not hasattr(self,'data'):self.load()
  idx=torch.clamp(frames.to(dtype=torch.long,device='cpu'),0,len(self.data['time_s'])-1)
  q31=torch.from_numpy(self.data['joint_pos31'])[idx];dq31=torch.from_numpy(self.data['joint_vel31'])[idx]
  return {'root_pose':torch.from_numpy(self.data['root_pose'])[idx],
          'root_velocity':torch.from_numpy(self.data['root_velocity'])[idx],
          'joint_pos31':q31,'joint_vel31':dq31,
          'joint_pos_source29':self.contract.official_to_source(q31),
          'joint_vel_source29':self.contract.official_to_source(dq31)}
