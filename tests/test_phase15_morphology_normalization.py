import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase15_morphology_normalization/phase15_result.json"


def test_phase15_is_deterministic_two_joint_counterfactual() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["intervention"]["only_joint_values_changed"] == ["left_hip_roll_joint", "right_hip_roll_joint"]
    assert result["intervention"]["root_xy_orientation_other_joints_unchanged"] is True
    assert result["execution"]["parameter_search"] is False
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["optimizer_steps"] == 0


def test_phase15_improves_width_but_not_single_support() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    geometry = result["geometry"]
    assert 0.34 < result["intervention"]["hip_roll_scale"] < 0.35
    assert geometry["candidate_foot_separation_p50_m"] < 0.47
    assert geometry["double_support_outside_count"] == 0
    assert geometry["single_support_outside_count"] == 137
    assert geometry["single_support_outside_fraction"] == 1.0
    assert result["preservation"]["root_z_correction_max_abs_m"] > 0.10
    assert result["decision"]["deterministic_hip_roll_normalization_fixes_single_support"] is False
    assert result["decision"]["physics_or_teacher_unlocked"] is False
