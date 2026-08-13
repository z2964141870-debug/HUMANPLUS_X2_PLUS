import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"reports/official_x2/phase41_native_boundary_seed.json"
def test_phase41_boundary_is_coherent_and_offline():
    if not RESULT.exists(): return
    r=json.loads(RESULT.read_text()); assert r["execution"]=={"mj_forward_calls":1,"mj_step_calls":0,"gpu":False}
    assert r["selection"]["eligible_coherent_anchors"]>0
    assert r["decision"]["coherent_1s_native_boundary_selected"] is True
    assert r["decision"]["joint_limits_pass"] is True
    assert r["decision"]["physics_unlocked"] is False and r["decision"]["training_unlocked"] is False
