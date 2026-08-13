import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase25_load_transfer_skeleton_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase25_result.json"


def test_phase25_contract_is_quasistatic_and_fail_closed():
    c = json.loads(CONTRACT.read_text())
    assert "not COP" in c["meaning"]
    assert c["resource"] == {"single_process": True, "cpu_threads": 1, "gpu": False}
    assert c["physics_authorized"] is False and c["training_authorized"] is False


def test_phase25_result_truth_boundary_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"]["mj_step_calls"] == 0 and r["execution"]["gpu"] is False
    assert r["decision"]["dynamics_truth"] is False
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
    if r["decision"]["load_transfer_skeleton_complete"]:
        assert len(r["steps"]) == 8 and all(step["success"] for step in r["steps"])
    else:
        assert len(r["steps"]) < 8 and r["steps"][-1]["success"] is False
