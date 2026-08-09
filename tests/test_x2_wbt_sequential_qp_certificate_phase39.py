import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_sequential_qp_certificate_phase39.json"


def test_phase39_is_one_fixed_qp_contract_on_same_window():
    report = json.loads(REPORT.read_text())
    prereg = report["pre_registration"]
    assert prereg["solver"] == "daqp"
    assert prereg["outer_iterations_max"] == 20
    assert prereg["window_frames"] == [0, 25]
    assert prereg["parameter_scan"] is False
    assert report["truth_boundary"]["sequential_qp_configuration_count"] == 1


def test_phase39_records_active_switches_and_all_sphere_rule():
    report = json.loads(REPORT.read_text())
    assert "all 12 active sole spheres" in report["pre_registration"]["all_sphere_rule"]
    assert len(report["summary"]["active_switches_lr"]) == 2
    for iteration in report["iterations"]:
        assert len(iteration["active_sphere_indices_lr"]) == 25


def test_phase39_never_claims_full_silver_or_physics():
    report = json.loads(REPORT.read_text())
    assert report["decision"]["full_trajectory_true_silver"] is False
    assert report["decision"]["physics_proposal_allowed"] is False
    if report["decision"]["local_window_feasible"]:
        assert report["final_metrics"]["feasible"] is True
