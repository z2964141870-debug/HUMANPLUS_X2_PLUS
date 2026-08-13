import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/official_x2/phase43_native_balanced_boundary.json"


def test_phase43_boundary_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"] == {"mj_forward_calls": 1, "mj_step_calls": 0, "gpu": False}
    assert r["selection"]["coherent_anchors"] == 650
    assert r["selection"]["eligible_low_speed_anchors"] > 0
    assert r["selection"]["qvel_max_abs"] <= r["selection"]["qvel_gate_radps"]
    assert r["decision"]["qvel_gate_pass"] is True
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
