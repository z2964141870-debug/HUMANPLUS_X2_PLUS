import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase31_balanced_boundary_geometry_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase31_result.json"


def test_phase31_is_boundary_only_and_no_physics():
    c = json.loads(CONTRACT.read_text())
    assert "Phase43" in c["only_variable"]
    assert c["resource"] == {"single_process": True, "cpu_threads": 1, "gpu": False}
    assert c["physics_authorized"] is False and c["training_authorized"] is False


def test_phase31_result_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"]["mj_step_calls"] == 0 and r["execution"]["gpu"] is False
    assert r["decision"]["dynamics_truth"] is False
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
    if r["decision"]["balanced_boundary_geometry_complete"]:
        assert len(r["steps"]) == 9 and all(step["success"] for step in r["steps"])
    else:
        assert len(r["steps"]) < 9 and r["steps"][-1]["success"] is False
