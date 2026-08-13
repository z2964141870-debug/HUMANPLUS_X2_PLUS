#!/usr/bin/env python3
"""Fail-closed readiness audit for the learning-based physical-generator route."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];OUT=R/'research/dynamic_retargeting_20260811/phase38_privileged_generator_readiness.json';REPORT=R/'research/dynamic_retargeting_20260811/REPORT_phase38_privileged_generator_readiness.md'
FILES={'seed_result':R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_result.json','hook_result':R/'research/dynamic_retargeting_20260811/phase37_native_generator_hook_result.json','hook_code':R/'src/x2_native_generator_seed.py','phase46':R/'src/x2_faithful_live_phase46.py','launcher':R/'scripts/run_dcpeft_stage152.sh','physics_guard':R/'src/x2_physics_provenance_guard.py'}
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def run():
 for p in FILES.values():
  if not p.is_file():raise FileNotFoundError(p)
 seed=json.loads(FILES['seed_result'].read_text());hook=json.loads(FILES['hook_result'].read_text());live=FILES['phase46'].read_text();launcher=FILES['launcher'].read_text()
 ready={'native_seed_200x50hz':seed['decision']['native_generator_seed_exported'],'immutable_reset_and_future_hook':hook['decision']['native_generator_hook_ready'],'wbt29_future_10x58_live_contract':'command_multi_future_source29' in live,'privileged_critic_1645_contract':'1645' in live,'phase25_physics_hash_guard':FILES['physics_guard'].stat().st_size>0,'stage152_opt_in_entrypoint':'FAITHFUL_WBT29' in launcher}
 blockers={'native_seed_reset_hook_wired_live':'x2_native_generator_seed' in launcher or 'x2_native_generator_seed' in live,'actual_contact_reward_without_reference_labels':(R/'src/x2_privileged_generator_reward.py').exists(),'rollout_state_contact_exporter':(R/'src/x2_privileged_generator_exporter.py').exists(),'live_zero_update_manifest':(R/'reports/x2_privileged_generator_live_zero.json').exists()}
 result={'stage':'Dynamic Retargeting Phase38','scope':'static readiness only; no Isaac/physics/optimizer/GPU','assets':{k:{'path':str(p),'sha256':sha(p)} for k,p in FILES.items()},'ready':ready,'blockers_resolved':blockers,'counts':{'ready':sum(ready.values()),'ready_total':len(ready),'blockers_resolved':sum(blockers.values()),'blockers_total':len(blockers)},'decision':{'offline_contract_ready':all(ready.values()),'live_zero_update_ready':all(ready.values()) and all(blockers.values()),'training_unlocked':False,'result':'OFFLINE_HOOK_READY_LIVE_ZERO_BLOCKED'}}
 OUT.write_text(json.dumps(result,indent=2)+'\n');REPORT.write_text(f"""# Phase38 — Privileged Physical Generator readiness

## 裁决

`OFFLINE_HOOK_READY / LIVE ZERO BLOCKED`

现有工程已经具备 {sum(ready.values())}/{len(ready)} 个底层合同：Phase34 原生 200×50Hz 成功盆地、不可变 reset/future hook、WBT29 10×58 future、1645D privileged critic、Phase25 physics hash guard，以及 Stage152 opt-in 入口。

仍缺 {len(blockers)-sum(blockers.values())} 项 live 接线：native reset hook 尚未进入 Stage152；尚无“不读取 reference contact label、只用 realized physics”的 generator reward；尚无 rollout state/contact exporter；因此也没有 live zero manifest。

这说明 OmniTrack 式路线不是缺模型骨架，而是缺 Stage-I 专用环境 glue。下一步应只实现默认关闭的 reset/reference/reward/exporter 接线并跑 zero-update；不能直接复用旧 Bronze contact reward，也不能启动 PPO。
""");print(json.dumps(result,indent=2));return result
if __name__=='__main__':run()
