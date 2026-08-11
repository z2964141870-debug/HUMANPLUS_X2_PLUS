import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase12_taskspace_load_bridge/phase12_preflight.json"


def test_phase12_is_zero_physics_and_name_mapped() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["optimizer_steps"] == 0
    assert result["execution"]["gpu"] is False
    assert result["contract"]["controlled_dof_count"] == 15
    assert result["contract"]["head_controlled"] is False
    assert all(len(value) == 12 for value in result["contract"]["active_sole_spheres"].values())


def test_phase12_rejects_fixed_feet_full_transfer_before_physics() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    authority = result["authority"]["both_feet_xyz_plus_com_xy"]
    transfer = result["linearized_full_transfer_with_both_feet_fixed"]
    assert authority["rank"] == 8
    assert authority["nullity_in_lower15"] == 7
    assert result["decision"]["kinematic_task_authority_available"] is True
    assert transfer["delta_max_abs_rad"] > 4.0
    assert transfer["joint_limit_violation_count"] == 2
    assert result["decision"]["linearized_full_right_support_transfer_with_fixed_feet_plausible"] is False
    assert result["decision"]["physics_or_teacher_unlocked"] is False
