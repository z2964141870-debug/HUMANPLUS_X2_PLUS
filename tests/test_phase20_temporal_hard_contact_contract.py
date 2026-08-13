import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_phase20_contract_is_fail_closed_and_low_resource():
    p=ROOT/"research/dynamic_retargeting_20260811/phase20_temporal_hard_contact_contract.json"
    r=json.loads(p.read_text()); assert r["status"]=="IMPLEMENTATION_PREREGISTERED_NOT_RUN"
    assert r["method"]["outer_iterations"]==10
    assert r["resource"]=={"single_process":True,"cpu_threads":1,"gpu":False,"wall_time_limit_minutes":30}
    assert r["stage_b_physics_authorized"] is False; assert r["training_authorized"] is False
    assert "weight scan" in r["forbidden"] and "parallel workers" in r["forbidden"]
