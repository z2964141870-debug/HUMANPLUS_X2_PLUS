import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_phase21_contract_uses_native_boundaries_and_sparse_knots():
    r=json.loads((ROOT/"research/dynamic_retargeting_20260811/phase21_contact_keyframe_skeleton_contract.json").read_text())
    assert r["status"]=="PREREGISTERED_NOT_RUN"; assert len(r["knot_fractions"])==7
    assert r["inputs"]["start_end_boundary"].startswith("Phase34 coherent safe")
    assert "Phase30 contact labels" in r["inputs"]["not_truth"]
    assert r["solver"]["single_configuration"] is True and r["solver"]["no_parameter_scan"] is True
    assert r["resource"]=={"single_process":True,"cpu_threads":1,"gpu":False,"wall_time_limit_minutes":30}
    assert r["stage_b_physics_authorized"] is False and r["training_authorized"] is False
