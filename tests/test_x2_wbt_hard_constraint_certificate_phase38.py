import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_hard_constraint_certificate_phase38.json"


def test_phase38_is_one_local_hard_constraint_run():
    report = json.loads(REPORT.read_text())
    assert report["truth_boundary"]["offline_slsqp_run_count"] == 1
    assert report["truth_boundary"]["mujoco_integration_steps"] == 0
    assert report["truth_boundary"]["held_out_read"] is False
    assert report["pre_registration"]["window_frames"] == [0, 25]
    assert report["pre_registration"]["parameter_or_window_scan"] is False


def test_phase38_never_claims_full_silver_or_physics_permission():
    report = json.loads(REPORT.read_text())
    assert report["decision"]["full_trajectory_true_silver"] is False
    assert report["decision"]["physics_proposal_allowed"] is False
    if report["decision"]["local_window_feasible"]:
        assert report["result"]["max_constraint_violation"] <= 1.0e-7


def test_phase38_solver_audit_is_explicit():
    report = json.loads(REPORT.read_text())
    audit = report["solver_audit"]
    assert audit["scipy_slsqp"]["nonlinear_constraints"] is True
    assert "quadprog" in audit["qpsolvers"]["backends"]
    assert audit["selected"].startswith("scipy.optimize.SLSQP")
