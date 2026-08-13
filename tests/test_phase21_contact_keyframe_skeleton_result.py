import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"research/dynamic_retargeting_20260811/phase21_result.json"
def test_phase21_final_solve_is_fail_closed():
    r=json.loads(RESULT.read_text()); assert r["preflight"]["boundary_within_bounds"] is True
    assert r["preflight"]["raw_equality_rows"]==84 and r["preflight"]["independent_equality_rank"]==76
    assert r["execution"]["iterations"]==150 and r["execution"]["mj_step_calls"]==0
    assert r["decision"]["stage_a_passed"] is False; assert r["decision"]["physics_unlocked"] is False; assert r["decision"]["training_unlocked"] is False
def test_phase21_failure_is_not_a_near_miss():
    r=json.loads(RESULT.read_text()); assert r["constraints"]["full_original_eq_max_abs"]>0.8
    assert r["dense_metrics"]["single_support_com_margin_min_m"]<-.5
    assert r["dense_metrics"]["stance_speed_p95_mps"]>1.0
