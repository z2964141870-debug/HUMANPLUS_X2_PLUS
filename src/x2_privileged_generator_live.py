"""Default-off reset batch adapter for future live privileged-generator wiring."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch
from x2_native_generator_seed import NativeGeneratorSeedHook

@dataclass(frozen=True)
class PrivilegedGeneratorResetBatch:
 frames:torch.Tensor;root_pose:torch.Tensor;root_velocity:torch.Tensor;joint_pos31:torch.Tensor;joint_vel31:torch.Tensor;future_source29:torch.Tensor

def deterministic_reset_batch(hook:NativeGeneratorSeedHook,num_envs:int,*,device:str='cpu')->PrivilegedGeneratorResetBatch:
 """Build an evenly spaced immutable reset batch; performs no env mutation."""
 if num_envs<1:raise ValueError('num_envs must be positive')
 if not hasattr(hook,'data'):hook.load()
 frames=torch.linspace(0,len(hook.data['time_s'])-1,num_envs,dtype=torch.float64).round().long()
 state=hook.reset_state(frames);future=hook.future_source29(frames)
 return PrivilegedGeneratorResetBatch(frames.to(device),state['root_pose'].to(device),state['root_velocity'].to(device),state['joint_pos31'].to(device),state['joint_vel31'].to(device),future.to(device))

def validate_live_boundary(batch:PrivilegedGeneratorResetBatch,num_envs:int)->dict[str,bool]:
 return {'frames':batch.frames.shape==(num_envs,), 'root_pose':batch.root_pose.shape==(num_envs,7),'root_velocity':batch.root_velocity.shape==(num_envs,6),'joint_pos31':batch.joint_pos31.shape==(num_envs,31),'joint_vel31':batch.joint_vel31.shape==(num_envs,31),'future_10x58':batch.future_source29.shape==(num_envs,10,58),'finite':all(torch.isfinite(x).all().item() for x in (batch.root_pose,batch.root_velocity,batch.joint_pos31,batch.joint_vel31,batch.future_source29))}


def _seed_sha256(seed_path: str) -> str:
 h = hashlib.sha256()
 with open(seed_path, 'rb') as f:
  for b in iter(lambda: f.read(1 << 20), b''):
   h.update(b)
 return h.hexdigest()


def reset_from_native_generator_seed(env, env_ids, *, seed_path: str,
        expected_sha256: str, fixed_frame_indices=None,
        reset_fraction: float | None = None,
        sampling_mode: str = 'evenly_spaced',
        asset_name: str = 'robot',
        device: str | None = None):
 """Live-zero native reset: write explicit seed state into the simulator and
 record a fail-closed reset ledger. Performs no contact synthesis, no hard
 position projection, no optimizer/physics step. Default-off by design."""
 dev = torch.device(device) if device is not None else getattr(env, 'device', torch.device('cpu'))
 env_ids = torch.as_tensor(env_ids, device=dev)
 n = int(env_ids.numel())

 # reset_fraction gate: an explicit zero fraction means a strict no-op that must
 # return before touching the seed file (used to validate the early-exit path).
 if reset_fraction is not None and float(reset_fraction) <= 0.0:
  env._x2_native_generator_reset_last = {
   'strict_no_op': True,
   'selected_env_count': n,
   'no_op': True,
   'finalizer_applied': False,
  }
  return

 if n == 0:
  env._x2_native_generator_reset_last = {
   'strict_no_op': True, 'selected_env_count': 0, 'no_op': True, 'finalizer_applied': False,
  }
  return

 actual_sha256 = _seed_sha256(seed_path)
 if actual_sha256 != expected_sha256:
  raise ValueError('native seed sha256 drift; refusing live reset')

 scene = env.scene
 try:
  robot = scene[asset_name]
 except (KeyError, TypeError):
  robot = getattr(scene, asset_name)

 with np.load(seed_path, allow_pickle=False) as payload:
  required = {'joint_names', 'root_pose', 'root_velocity', 'joint_pos31', 'joint_vel31'}
  missing = sorted(required.difference(payload.files))
  if missing:
   raise ValueError(f'native seed missing fields: {missing}')
  total_frames = int(payload['joint_pos31'].shape[0])
  if sampling_mode != 'evenly_spaced':
   raise ValueError(f'unsupported native reset sampling_mode: {sampling_mode}')
  if fixed_frame_indices is None:
   frames = torch.linspace(0, total_frames - 1, n, device=dev).round().long()
  else:
   frames = torch.as_tensor(fixed_frame_indices, device=dev).to(dtype=torch.long)
   if frames.numel() != n:
    raise ValueError(f'fixed_frame_indices count {frames.numel()} != env_ids count {n}')
  idx = torch.clamp(frames, 0, total_frames - 1).to(dtype=torch.long)
  cpu_idx = idx.detach().cpu().numpy()

  payload_joint_names = list(payload['joint_names'].tolist())
  robot_joint_names = list(robot.joint_names)
  if set(payload_joint_names) != set(robot_joint_names):
   raise ValueError('native seed joint_names do not match target asset')
  reorder = [payload_joint_names.index(name) for name in robot_joint_names]
  dtype = robot.data.default_joint_pos.dtype
  root_pose = torch.from_numpy(payload['root_pose'][cpu_idx]).to(device=dev, dtype=dtype)
  root_velocity = torch.from_numpy(payload['root_velocity'][cpu_idx]).to(device=dev, dtype=dtype)
  joint_pos31 = torch.from_numpy(payload['joint_pos31'][cpu_idx][:, reorder]).to(
   device=dev, dtype=dtype
  )
  joint_vel31 = torch.from_numpy(payload['joint_vel31'][cpu_idx][:, reorder]).to(
   device=dev, dtype=dtype
  )

 # Seed root poses are local to each replicated environment. Isaac expects
 # world-frame root positions at the write boundary.
 origins = getattr(scene, 'env_origins', None)
 if origins is not None:
  root_pose = root_pose.clone()
  root_pose[:, :3] += origins[env_ids].to(device=dev, dtype=root_pose.dtype)

 robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
 robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
 robot.write_joint_state_to_sim(joint_pos31, joint_vel31, env_ids=env_ids)

 soft_limits = robot.data.soft_joint_pos_limits
 if soft_limits.ndim == 3:
  soft_limits = soft_limits[env_ids]
 else:
  soft_limits = soft_limits.unsqueeze(0).expand(n, -1, -1)
 soft_limits = soft_limits.to(device=dev, dtype=joint_pos31.dtype)
 soft_overshoot = torch.maximum(
  (soft_limits[..., 0] - joint_pos31).clamp_min(0),
  (joint_pos31 - soft_limits[..., 1]).clamp_min(0),
 )

 env._x2_native_generator_reset_last = {
  'strict_no_op': False,
  'selected_env_count': n,
  'selected_env_ids': env_ids,
  'no_op': False,
  'contact_labels_written': False,
  'hard_position_projection_applied': False,
  'frame_indices': idx,
  'seed_path': str(Path(seed_path).resolve()),
  'seed_sha256': actual_sha256,
  'sampling_mode': sampling_mode,
  'asset_name': asset_name,
  'root_pose': root_pose,
  'root_velocity': root_velocity,
  'joint_pos': joint_pos31,
  'joint_vel': joint_vel31,
  'soft_position_overshoot_max_rad': float(soft_overshoot.max().item()),
  'finalizer_applied': False,
  '_env_ids': env_ids,
  '_root_pose': root_pose,
  '_root_velocity': root_velocity,
  '_joint_pos31': joint_pos31,
  '_joint_vel31': joint_vel31,
 }


def finalize_native_generator_reset(env) -> dict:
 """Close out the live native reset: confirm the ledger, mark the finalizer, and
 report the selected env count. Re-applies the recorded explicit state so the
 simulator holds the seed state at the finalizer boundary; no physics step, no
 contact synthesis, no optimizer claim."""
 ledger = getattr(env, '_x2_native_generator_reset_last', None) or {}
 no_op = bool(ledger.get('no_op', True))
 selected = int(ledger.get('selected_env_count', 0))
 if not no_op and '_env_ids' in ledger:
  scene = env.scene
  asset_name = ledger.get('asset_name', 'robot')
  try:
   robot = scene[asset_name]
  except (KeyError, TypeError):
   robot = getattr(scene, asset_name)
  eids = ledger['_env_ids']
  robot.write_root_pose_to_sim(ledger['_root_pose'], env_ids=eids)
  robot.write_root_velocity_to_sim(ledger['_root_velocity'], env_ids=eids)
  robot.write_joint_state_to_sim(
   ledger['_joint_pos31'], ledger['_joint_vel31'], env_ids=eids
  )
 ledger['finalizer_applied'] = True
 env._x2_native_generator_reset_last = ledger
 return {'selected_env_count': selected, 'no_op': no_op}
