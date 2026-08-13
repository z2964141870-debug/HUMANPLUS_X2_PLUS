#!/usr/bin/env python3
import json
from pathlib import Path
from x2_faithful_any2any_phase23 import WBT29PolicyContract
from x2_native_generator_seed import NativeGeneratorSeedHook,NativeGeneratorSeedSpec,sha256
from x2_privileged_generator_live import deterministic_reset_batch,validate_live_boundary
R=Path(__file__).resolve().parents[2];S=R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed.npz';P36=R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_result.json';P26=R/'reports/retarget/x2_faithful_zero_update_phase26.json';OUT=R/'research/dynamic_retargeting_20260811/phase39_privileged_generator_reset_batch.json'
def run():
 p=json.loads(P26.read_text())['contract'];c=WBT29PolicyContract.build(p['official31_order'],p['sim_target29_order'],p['policy_source29_order'],p['head_excluded2']);spec=NativeGeneratorSeedSpec(S,json.loads(P36.read_text())['assets']['output_sha256']);hook=NativeGeneratorSeedHook(spec,c);batch=deterministic_reset_batch(hook,16);checks=validate_live_boundary(batch,16);checks['endpoints_exact']=int(batch.frames[0])==0 and int(batch.frames[-1])==199;checks['monotonic_frames']=bool((batch.frames[1:]>=batch.frames[:-1]).all())
 r={'stage':'Dynamic Retargeting Phase39','scope':'CPU reset-batch construction only; no live env mutation/Isaac/physics/optimizer','assets':{'seed_sha256':sha256(S),'adapter_sha256':sha256(R/'src/x2_privileged_generator_live.py')},'frames':batch.frames.tolist(),'checks':checks,'execution':{'env_instances':0,'env_resets':0,'physics_steps':0,'optimizer_steps':0,'gpu':False},'decision':{'reset_batch_adapter_ready':all(checks.values()),'live_zero_executed':False,'training_unlocked':False}};OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return r
if __name__=='__main__':run()
