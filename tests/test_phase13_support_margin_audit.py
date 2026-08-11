import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase13_support_margin_audit/phase13_result.json"


def test_phase13_zero_physics_truth_boundary() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["optimizer_steps"] == 0
    assert result["execution"]["gpu"] is False
    assert "quasi-static" in result["truth_boundary"]["interpretation"]
    assert result["counts"] == {"frames": 175, "single_support": 137, "double_support": 38, "flight": 0}


def test_phase13_systematic_support_mismatch_stops_route() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    single = result["single_support"]
    assert single["outside_count"] == 137
    assert single["outside_fraction"] == 1.0
    assert single["outside_gap_p50_m"] > 0.20
    assert single["outside_gap_p95_m"] > 0.39
    assert all(side["outside_fraction"] == 1.0 for side in single["per_side"].values())
    assert result["decision"]["original_contact_schedule_is_quasi_statically_supported"] is False
    assert result["decision"]["physics_or_teacher_unlocked"] is False
