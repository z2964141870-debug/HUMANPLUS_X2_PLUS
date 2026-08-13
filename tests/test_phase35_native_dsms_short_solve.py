import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'research/dynamic_retargeting_20260811/phase35_native_dsms_short_solve_contract.json';O=R/'research/dynamic_retargeting_20260811/phase35_native_dsms_short_solve_result.json'
def test_contract_is_one_low_resource_solve():
 c=json.loads(C.read_text());assert c['solver']=={'linear_solver':'mumps','max_iterations':50,'cpu_threads':1,'nice':10,'gpu':False,'retry_count':0};assert c['training_authorized'] is False and c['longer_solve_authorized'] is False
def test_result_if_present_is_fail_closed():
 if not O.exists():return
 r=json.loads(O.read_text());assert r['execution']['solve_count']==1 and r['execution']['retry_count']==0;assert r['decision']['training_unlocked'] is False and r['decision']['longer_solve_authorized'] is False;assert r['decision']['dynamic_teacher_qualified']==all(r['checks'].values())
