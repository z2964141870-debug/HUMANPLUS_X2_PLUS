import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research/dynamic_retargeting_20260811/phase37_native_generator_hook_result.json'
def test_result_if_present_is_state_only():
 if not O.exists():return
 r=json.loads(O.read_text());assert r['manifest']['contact_labels_present'] is False and r['manifest']['optimizer_eligible'] is False;assert r['execution']['physics_steps']==0 and r['execution']['optimizer_steps']==0;assert r['decision']['training_unlocked'] is False;assert r['decision']['native_generator_hook_ready']==all(r['checks'].values())
