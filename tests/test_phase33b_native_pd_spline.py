import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"research/dynamic_retargeting_20260811/phase33b_native_pd_spline_contract.json"
RESULT=ROOT/"research/dynamic_retargeting_20260811/phase33b_native_pd_spline_result.json"

def test_contract_is_single_variable_and_optimizer_free():
    c=json.loads(CONTRACT.read_text()); assert "only_variable_vs_phase33" in c
    assert c["representation"]["control_knots"]==18 and c["resources"]["optimizer_steps"]==0
    assert c["next_solve_authorized"] is False

def test_result_if_present_is_fail_closed():
    if not RESULT.exists(): return
    r=json.loads(RESULT.read_text()); assert r["execution"]["optimizer_steps"]==0
    assert r["decision"]["single_short_solve_authorized"] is False and r["decision"]["training_unlocked"] is False
    assert r["decision"]["pd_spline_warmstart_qualified"]==all(r["checks"].values())
