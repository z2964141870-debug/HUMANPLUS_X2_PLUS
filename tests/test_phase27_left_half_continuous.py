import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase27_left_half_continuous_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase27_result.json"


def test_phase27_contract_is_left_only_and_no_physics():
    c = json.loads(CONTRACT.read_text())
    assert len(c["segments"]) == 3
    assert c["resource"] == {"single_process": True, "cpu_threads": 1, "gpu": False}
    assert c["physics_authorized"] is False and c["training_authorized"] is False


def test_phase27_result_boundary_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"]["mj_step_calls"] == 0 and r["execution"]["gpu"] is False
    assert r["decision"]["dynamics_truth"] is False
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
    if r["decision"]["left_half_continuous_path_complete"]:
        assert r["failed_frame"] is None and all(r["gates"].values())
    else:
        assert r["failed_frame"] is not None or not all(r["gates"].values())
