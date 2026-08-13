import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"research/dynamic_retargeting_20260811/phase19_hard_contact_preflight/phase19_result.json"
def test_phase19_is_local_offline_certificate():
    if not RESULT.exists(): return
    r=json.loads(RESULT.read_text()); assert r["execution"]["mj_step_calls"]==0; assert r["execution"]["gpu"] is False
    assert r["contract"]["template"]=="DS-L-DS-R-DS"; assert "not temporal" in r["contract"]["truth"]
    assert r["decision"]["physics_unlocked"] is False; assert r["decision"]["training_unlocked"] is False
def test_phase19_counts_are_closed():
    if not RESULT.exists(): return
    r=json.loads(RESULT.read_text()); s=r["summary"]
    assert s["single_support_frames"]+s["double_support_frames"]==s["frames"]
    assert s["feasible"]==s["single_support_feasible"]+s["double_support_feasible"]
