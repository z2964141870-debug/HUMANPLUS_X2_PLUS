import argparse

import mujoco
import numpy as np

from official_x2.audit_phase29_closed_wrapper_divergence import (
    DEFAULT_BINARY,
    DEFAULT_CONTROL,
    DEFAULT_HISTORICAL,
    DEFAULT_HISTORICAL_SCENE,
    DEFAULT_LAUNCHER,
    DEFAULT_MODULE,
    DEFAULT_PHASE28_RUNNER,
    DEFAULT_RESOLVED_CONFIG,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_SCENE,
    DEFAULT_SIM_CONFIG,
    DEFAULT_SUBSTEPS,
    DEFAULT_VENDOR_MJ,
    build_report,
    quat_geodesic,
    vector_summary,
)


def _args():
    return argparse.Namespace(
        historical=DEFAULT_HISTORICAL,
        substeps=DEFAULT_SUBSTEPS,
        control=DEFAULT_CONTROL,
        scene=DEFAULT_SCENE,
        binary=DEFAULT_BINARY,
        module=DEFAULT_MODULE,
        vendor_mujoco=DEFAULT_VENDOR_MJ,
        historical_scene=DEFAULT_HISTORICAL_SCENE,
        launcher=DEFAULT_LAUNCHER,
        robot_config=DEFAULT_ROBOT_CONFIG,
        sim_config=DEFAULT_SIM_CONFIG,
        resolved_config=DEFAULT_RESOLVED_CONFIG,
        phase28_runner=DEFAULT_PHASE28_RUNNER,
    )


def test_quaternion_geodesic_is_sign_invariant():
    q = np.asarray([0.9, 0.1, -0.2, 0.3], dtype=np.float64)
    q /= np.linalg.norm(q)
    assert quat_geodesic(q, q) == 0.0
    assert quat_geodesic(q, -q) == 0.0


def test_vector_summary_reports_expected_norms():
    result = vector_summary(np.asarray([3.0, 4.0]))
    assert result["abs_max"] == 4.0
    assert result["l2"] == 5.0
    assert np.isclose(result["rmse"], np.sqrt(12.5))


def test_phase29_keeps_closed_wrapper_boundary_and_initial_contract():
    report = build_report(_args())
    assert report["alignment"]["historical_prepare_rows"] == 10
    assert report["alignment"]["initial_visible_state_exact_by_construction"]
    assert report["alignment"]["initial_action_abs_max_error"] <= 1e-6
    assert report["compiled_provenance"]["python_direct_mj_version_string"] == mujoco.mj_versionString()
    assert report["compiled_provenance"]["model_dimensions"]["na"] == 0
    assert report["root_cause_decision"]["status"] == "BLOCKED_BY_CLOSED_WRAPPER"
    assert not report["root_cause_decision"]["unique_single_variable_identified"]
    assert not report["root_cause_decision"]["minimal_probe_preregistered_not_run"]["execution_authorized"]


def test_phase29_detects_divergence_before_action_can_be_causally_attributed():
    report = build_report(_args())
    rows = report["first_0p2s"]["aligned_control_boundaries"]
    assert rows[0]["elapsed_s"] == 0.02
    assert rows[0]["state_error"]["joint_velocity"]["abs_max"] > 0.5
    assert rows[0]["action_error"]["abs_max"] > 0.1
    matrix = report["observable_matrix"]
    assert matrix["qacc_warmstart_solver_efc_state"]["historical"] is False
    assert matrix["contact_and_mj_contactForce"]["historical"] is False
    assert matrix["actuator_activation"]["direct"].startswith("structurally absent")
