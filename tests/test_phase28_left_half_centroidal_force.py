import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase28_left_half_centroidal_force_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase28_result.json"


def test_phase28_contract_is_contact_implicit_and_no_physics():
    c = json.loads(CONTRACT.read_text())
    assert "no prescribed" in c["contact_implicit_candidates"]
    assert c["resource"] == {"single_process": True, "cpu_threads": 1, "gpu": False}
    assert c["physics_authorized"] is False and c["training_authorized"] is False


def test_phase28_result_truth_boundary_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"]["mj_step_calls"] == 0 and r["execution"]["gpu"] is False
    assert r["decision"]["dynamics_truth"] is False
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
    evaluated = r["execution"]["evaluated_frames"]
    passed = r["summary"]["jointly_feasible_frames"] == evaluated
    assert r["decision"]["centroidal_force_preflight_pass"] == passed
