import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULT=ROOT/"research/dynamic_retargeting_20260811/phase20_temporal_backend.json"
def test_phase20_backend_prefers_low_resource_nullspace():
    if not RESULT.exists(): return
    r=json.loads(RESULT.read_text()); assert r["execution"]=={"physics_steps":0,"solver_calls":0,"gpu":False}
    assert r["problem"]["local_nullity_min"]>=17
    assert r["memory_estimate_bytes"]["block_nullspace_bases"]<1024*1024
    assert r["memory_estimate_bytes"]["dense_qp_hessian"]>100*1024*1024
    assert r["decision"]["block_nullspace_sparse_lsqr_preferred"] is True
