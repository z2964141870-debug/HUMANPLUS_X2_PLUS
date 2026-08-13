#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import torch
from x2_faithful_any2any_phase23 import WBT29PolicyContract
from x2_native_generator_seed import NativeGeneratorSeedHook,NativeGeneratorSeedSpec,sha256
R=Path(__file__).resolve().parents[2];SEED=R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed.npz';P36=R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_result.json';P26=R/'reports/retarget/x2_faithful_zero_update_phase26.json';OUT=R/'research/dynamic_retargeting_20260811/phase37_native_generator_hook_result.json'
def run():
 p36=json.loads(P36.read_text());c=json.loads(P26.read_text())['contract'];contract=WBT29PolicyContract.build(c['official31_order'],c['sim_target29_order'],c['policy_source29_order'],c['head_excluded2'])
 hook=NativeGeneratorSeedHook(NativeGeneratorSeedSpec(SEED,p36['assets']['output_sha256']),contract);manifest=hook.load();frames=torch.tensor([0,50,199]);future=hook.future_source29(frames);z=hook.data;official=torch.from_numpy(z['joint_pos31'][[0,50,199]]);source=contract.official_to_source(official);target=contract.source_to_target(source);roundtrip=contract.target_to_source(target)
 reset=hook.reset_state(frames);checks={'manifest_frames':manifest['frames']==200,'future_shape':list(future.shape)==[3,10,58],'finite':bool(torch.isfinite(future).all()),'source_target_roundtrip_exact':bool(torch.equal(source,roundtrip)),'endpoint_clamped':bool(torch.equal(future[-1,-1],future[-1,0])),'reset_shapes':list(reset['root_pose'].shape)==[3,7] and list(reset['joint_pos31'].shape)==[3,31] and list(reset['joint_pos_source29'].shape)==[3,29],'reset_source_exact':bool(torch.equal(reset['joint_pos_source29'],source)),'contact_absent':manifest['contact_labels_present'] is False,'optimizer_ineligible':manifest['optimizer_eligible'] is False}
 r={'stage':'Dynamic Retargeting Phase37','scope':'CPU immutable native-seed hook and 10x58 future-reference probe; no Isaac/optimizer/physics','assets':{'seed_sha256':sha256(SEED),'phase36_sha256':sha256(P36),'phase26_sha256':sha256(P26)},'manifest':manifest,'future_shape':list(future.shape),'checks':checks,'execution':{'physics_steps':0,'optimizer_steps':0,'gpu':False},'decision':{'native_generator_hook_ready':all(checks.values()),'live_zero_update_authorized':False,'training_unlocked':False}}
 OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return r
if __name__=='__main__':run()
