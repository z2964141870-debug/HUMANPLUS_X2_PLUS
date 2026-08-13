import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/official_x2/phase42_native_quiescent_boundary.json"


def test_phase42_boundary_if_present():
    if not RESULT.exists():
        return
    r = json.loads(RESULT.read_text())
    assert r["execution"] == {"mj_forward_calls": 1, "mj_step_calls": 0, "gpu": False}
    assert r["selection"]["eligible_coherent_anchors"] == 650
    assert r["selection"]["qvel_l2"] < r["selection"]["phase41_qvel_l2"]
    assert r["decision"]["velocity_improved_vs_phase41"] is True
    assert r["decision"]["physics_unlocked"] is False
    assert r["decision"]["training_unlocked"] is False
