import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase23_time_parameterized_skeleton_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase23_result.json"


def test_phase23_contract_is_single_run_and_no_physics():
    c = json.loads(CONTRACT.read_text())
    assert c["resource"] == {"single_process": True, "cpu_threads": 1, "gpu": False}
    assert c["transition_duration_s"] == 0.5 and c["hold_duration_s"] == 0.12
    assert c["physics_authorized"] is False and c["training_authorized"] is False


def test_phase23_result_respects_contract_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"]["mj_step_calls"] == 0 and r["execution"]["gpu"] is False
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
    assert len(r["frames"]) == r["metrics"]["frames"]
    assert r["decision"]["time_parameterized_skeleton_pass"] == all(r["gates"].values())
    if not r["decision"]["time_parameterized_skeleton_pass"]:
        assert any(value is False for value in r["gates"].values())
