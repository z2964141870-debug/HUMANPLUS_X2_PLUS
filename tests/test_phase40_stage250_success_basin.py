import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"reports/official_x2/phase40_stage250_success_basin.json"
def test_phase40_uses_coherent_native_success_rows_only():
    if not RESULT.exists(): return
    r=json.loads(RESULT.read_text()); assert r["execution"]=={"physics_steps":0,"optimizer_steps":0,"gpu":False}
    assert r["contract"]["coherence"].startswith("one shared reference row")
    assert r["contract"]["root_xy_yaw_excluded"] is True
    assert r["reference"]["coherent_1s_anchor_rows"]>0
    assert r["decision"]["physics_unlocked"] is False and r["decision"]["training_unlocked"] is False
