import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase14_contact_schedule_repair/phase14_result.json"


def test_phase14_keeps_geometry_frozen_and_uses_zero_physics() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["contract"]["q_root_foot_placement_modified"] is False
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["optimizer_steps"] == 0
    assert result["execution"]["gpu"] is False


def test_phase14_rejects_label_only_repair() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    summary = result["summary"]
    assert summary["changed_frames"] == 137
    assert summary["unresolved_frames"] == 30
    assert summary["repaired_counts"] == {"left": 0, "right": 0, "left+right": 145}
    assert summary["double_support_margin_min_m"] < -0.024
    assert result["decision"]["label_only_quasi_static_repair_exists"] is False
    assert result["decision"]["label_only_repair_is_small"] is False
    assert result["decision"]["physics_or_teacher_unlocked"] is False
