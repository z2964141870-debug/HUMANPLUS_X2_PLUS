import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/retarget/run_x2_wbt_rootxy_sequential_qp_phase40.py"
REPORT = REPO / "reports/retarget/x2_wbt_rootxy_sequential_qp_phase40.json"


def test_inscribed_polygon_guarantees_euclidean_bound():
    # Pure contract test: do not import the physics runner into the lightweight
    # report-test environment.
    sides, radius = 32, 0.20
    theta = 2.0 * np.pi * np.arange(sides) / sides
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    scale = float(np.cos(np.pi / sides))
    assert directions.shape == (32, 2)
    # Every vertex of u.x <= R*cos(pi/N) lies exactly on radius R.
    vertex = np.linalg.solve(directions[:2], np.full(2, radius * scale))
    assert np.linalg.norm(vertex) <= radius + 1e-12
    source = TOOL.read_text()
    assert "POLYGON_SIDES = 32" in source
    assert "ROOT_XY_MAX_M = 0.20" in source


def test_phase40_is_single_rootxy_ab_and_keeps_phase39_contract():
    report = json.loads(REPORT.read_text())
    prereg = report["pre_registration"]
    assert prereg["single_structural_variable"] == "per-frame root XY correction"
    assert prereg["root_xy_relative_source_max_m"] == 0.20
    assert prereg["solver"] == "daqp"
    assert prereg["outer_iterations_max"] == 20
    assert prereg["window_frames"] == [0, 25]
    assert prereg["parameter_scan"] is False
    assert report["truth_boundary"]["phase39_A_rerun"] is False
    assert report["truth_boundary"]["configuration_count_B"] == 1


def test_phase40_never_claims_silver_or_physics():
    report = json.loads(REPORT.read_text())
    assert report["decision"]["full_trajectory_true_silver"] is False
    assert report["decision"]["physics_proposal_allowed"] is False
    assert report["truth_boundary"]["mujoco_integration_steps"] == 0
    assert report["truth_boundary"]["policy_optimizer_ppo_training"] is False
    if report["decision"]["local_window_feasible"]:
        assert report["final_metrics"]["feasible"] is True
        assert report["final_metrics"]["root_xy_correction_max_m"] <= 0.2000001
        assert report["final_metrics"]["root_horizontal_acceleration_p95_max_mps2"][1] <= 4.0000001
