import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase16_local_support_reachability/phase16_result.json"


def test_phase16_is_swing_free_local_zero_physics_audit() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert "swing foot unconstrained" in result["contract"]["task_per_side"]
    assert result["contract"]["lower15_delta_max_abs_rad"] == 0.35
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["optimizer_steps"] == 0


def test_phase16_has_no_locally_reachable_support_window() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    summary = result["summary"]
    assert summary["left_feasible_frames"] == 0
    assert summary["right_feasible_frames"] == 0
    assert summary["either_feasible_frames"] == 0
    assert summary["per_side_distribution"]["left"]["delta_max_abs_min_rad"] > 0.74
    assert summary["per_side_distribution"]["right"]["delta_max_abs_min_rad"] > 1.43
    assert all(value["joint_limit_violation_frame_fraction"] == 1.0 for value in summary["per_side_distribution"].values())
    assert result["decision"]["has_at_least_100ms_local_single_support_window"] is False
    assert result["decision"]["physics_or_teacher_unlocked"] is False
