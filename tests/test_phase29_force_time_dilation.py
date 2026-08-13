import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase29_force_time_dilation_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase29_result.json"


def test_phase29_contract_is_one_scale_and_no_physics():
    c = json.loads(CONTRACT.read_text())
    assert "dt=0.04s" in c["candidate"]
    assert c["resource"] == {"single_process": True, "cpu_threads": 1, "gpu": False}
    assert c["physics_authorized"] is False and c["training_authorized"] is False


def test_phase29_result_is_strict_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["intervention"] == {"time_scale": 2.0, "dt_s": 0.04, "only_variable": "time"}
    assert r["candidate"]["execution"]["mj_step_calls"] == 0
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
    expected = r["candidate"]["summary"]["jointly_feasible_frames"] == r["candidate"]["execution"]["evaluated_frames"]
    assert r["decision"]["time_dilation_force_preflight_pass"] == expected
