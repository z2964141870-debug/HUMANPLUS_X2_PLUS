import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"research/dynamic_retargeting_20260811/phase22_result.json"
def test_phase22_is_serial_and_fail_closed():
    contract=json.loads((ROOT/"research/dynamic_retargeting_20260811/phase22_sequential_contact_skeleton_contract.json").read_text()); assert contract["resource"]=={"single_process":True,"cpu_threads":1,"gpu":False}; assert contract["physics_authorized"] is False and contract["training_authorized"] is False
    if RESULT.exists():
        r=json.loads(RESULT.read_text()); assert r["execution"]["mj_step_calls"]==0 and r["execution"]["gpu"] is False
        assert r["decision"]["physics_unlocked"] is False and r["decision"]["training_unlocked"] is False
        if not r["decision"]["contact_skeleton_complete"]: assert len(r["steps"])<7 and r["steps"][-1]["success"] is False

def test_phase22_complete_result_is_reusable_and_hard_feasible():
    if not RESULT.exists():
        return
    r=json.loads(RESULT.read_text())
    assert r["decision"]["contact_skeleton_complete"] is True
    assert [x["label"] for x in r["steps"]] == [
        "DS_NATIVE", "DS_LOAD", "L_SUPPORT_R_SWING", "DS_R_TOUCHDOWN",
        "R_SUPPORT_L_SWING", "DS_L_TOUCHDOWN", "DS_SETTLE",
    ]
    assert all(x["success"] for x in r["steps"])
    assert max(x["eq_max_abs"] for x in r["steps"][1:]) <= 1e-6
    assert min(x["ineq_min"] for x in r["steps"][1:]) >= -1e-8
    for x in r["steps"]:
        assert len(x["root"]) == 3
        assert len(x["quat_xyzw"]) == 4
        assert len(x["lower_q"]) == 15
        assert set(x["foot_centroid_xy"]) == {"left", "right"}
        assert all(len(v) == 2 for v in x["foot_centroid_xy"].values())
