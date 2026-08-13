import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'research/dynamic_retargeting_20260811/phase34_native_dsms_nlp_contract.json';O=R/'research/dynamic_retargeting_20260811/phase34_native_dsms_nlp_result.json'
def test_contract_is_preflight_only():
 c=json.loads(C.read_text());assert c['fixed']['cpu_threads']==1 and c['fixed']['gpu'] is False;assert c['ipopt_authorized'] is False and c['training_authorized'] is False
def test_result_if_present():
 if not O.exists():return
 r=json.loads(O.read_text());assert r['execution']['optimizer_instances']==0 and r['decision']['ipopt_solve_authorized'] is False;assert r['decision']['native_contact_nlp_preflight_qualified']==all(r['checks'].values())
