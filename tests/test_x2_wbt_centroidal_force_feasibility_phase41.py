import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_centroidal_force_feasibility_phase41.json"


def test_phase41_is_one_fixed_model_force_oracle():
    report = json.loads(REPORT.read_text())
    assert report["pre_registration"]["window_frames"] == [0, 25]
    assert report["pre_registration"]["parameter_scan"] is False
    assert report["truth_boundary"]["force_oracle_configuration_count"] == 1
    assert report["truth_boundary"]["mujoco_integration_steps"] == 0
    assert report["truth_boundary"]["ppo_or_policy_training"] is False


def test_phase41_preserves_truth_boundary_and_hard_unlock():
    report = json.loads(REPORT.read_text())
    assert report["truth_boundary"]["official_mujoco_model_estimate_not_hardware_truth"] is True
    result = report["result"]
    allowed = report["decision"]["alternating_teacher_allowed"]
    assert allowed == (result["jointly_feasible_frames"] == result["evaluated_frames"])
    assert report["decision"]["alternating_teacher_run"] is False
    assert report["decision"]["full_trajectory_true_silver"] is False
    assert report["decision"]["physics_or_ppo_allowed"] is False


def test_phase41_noncontact_forces_are_structurally_excluded():
    report = json.loads(REPORT.read_text())
    assert report["pre_registration"]["noncontact_foot_force_zero"] is True
    for frame in report["frames"]:
        for side, geometry in frame["contact_geometry"].items():
            if not geometry["intended"]:
                assert geometry["active_sphere_indices"] == []
