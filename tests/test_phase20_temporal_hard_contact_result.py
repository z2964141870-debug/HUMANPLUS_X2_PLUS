import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"research/dynamic_retargeting_20260811/phase20_result.json"
def test_phase20_valid_run_is_fail_closed():
    r=json.loads(RESULT.read_text()); assert r["execution"]["mj_step_calls"]==0 and r["execution"]["gpu"] is False
    assert r["execution"]["outer_iterations"]==10; assert r["decision"]["stage_a_passed"] is False
    assert r["decision"]["physics_unlocked"] is False; assert r["decision"]["training_unlocked"] is False
def test_phase20_improves_geometry_but_does_not_meet_gates():
    r=json.loads(RESULT.read_text()); first=r["iterations"][0]; last=r["iterations"][-1]
    assert last["hard_violation_score"]<first["hard_violation_score"]
    assert last["metrics"]["single_support_com_margin_min_m"]>0
    assert last["metrics"]["stance_contact_abs_distance_p95_m"]>0.0005
    assert last["metrics"]["stance_speed_p95_mps"]>0.1
