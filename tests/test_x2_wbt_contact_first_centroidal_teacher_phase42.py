import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_contact_first_centroidal_teacher_phase42.json"


def test_phase42_is_one_preregistered_contact_first_configuration():
    report = json.loads(REPORT.read_text())
    prereg = report["pre_registration"]
    assert prereg["solver"] == "daqp"
    assert prereg["outer_iterations_max"] == 10
    assert prereg["window_frames"] == [0, 25]
    assert prereg["parameter_scan"] is False
    assert report["truth_boundary"]["configuration_count"] == 1
    assert report["truth_boundary"]["mujoco_integration_steps"] == 0


def test_phase42_fail_closed_between_layers():
    report = json.loads(REPORT.read_text())
    passed = report["decision"]["layer_A_feasible"]
    if not passed:
        assert report["decision"]["layer_B_whole_body_ik_run"] is False
        assert report["decision"]["layer_C_phase41_oracle_run"] is False
    assert report["decision"]["physics_or_ppo_allowed"] is False
    assert report["decision"]["full_trajectory_jointly_feasible"] is False


def test_phase42_truth_boundary_is_model_estimate_only():
    report = json.loads(REPORT.read_text())
    assert report["truth_boundary"]["official_model_estimates_not_hardware_truth"] is True
    assert report["truth_boundary"]["held_out_read"] is False
    assert report["truth_boundary"]["policy_physics_or_ppo"] is False
