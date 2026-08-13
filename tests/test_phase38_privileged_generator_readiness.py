import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/dynamic_retargeting_20260811/phase38_privileged_generator_readiness.json'
def test_readiness_if_present_is_fail_closed():
 if not O.exists():return
 r=json.loads(O.read_text());assert r['decision']['training_unlocked'] is False;assert r['decision']['offline_contract_ready']==all(r['ready'].values());assert r['decision']['live_zero_update_ready']==(all(r['ready'].values()) and all(r['blockers_resolved'].values()))
