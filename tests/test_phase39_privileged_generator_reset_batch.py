import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/dynamic_retargeting_20260811/phase39_privileged_generator_reset_batch.json'
def test_result_if_present_is_no_live_mutation():
 if not O.exists():return
 r=json.loads(O.read_text());assert r['execution']=={'env_instances':0,'env_resets':0,'physics_steps':0,'optimizer_steps':0,'gpu':False};assert r['decision']['live_zero_executed'] is False and r['decision']['training_unlocked'] is False;assert r['decision']['reset_batch_adapter_ready']==all(r['checks'].values())
