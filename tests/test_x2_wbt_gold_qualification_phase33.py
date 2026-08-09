import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_gold_qualification_phase33.json"


def test_phase33_exact_single_candidate_and_paired_contract():
    report = json.loads(REPORT.read_text())
    prereg = report["preregistration"]
    assert report["preflight"]["motion_id"] == "PHUMA-LUNGE-R-001"
    assert prereg["comparison_type"].startswith("paired qualification")
    assert prereg["same_official_scene_control_pd_action_limits_sole_contract"] is True
    assert prereg["numeric_initial_state_equal"] is False
    assert prereg["parameter_scan_or_reference_change"] is False


def test_phase33_conditional_free_and_same_runner():
    report = json.loads(REPORT.read_text())
    candidate_pass = report["gate"]["candidate_prescribed"]["pass"]
    free_present = "phase30_candidate_free" in report["paired_replay"]
    assert free_present == candidate_pass
    assert report["truth_boundary"]["candidate_free_executed_only_after_all_prescribed_gates"] == free_present
    assert set(report["paired_replay"]) >= {"phase28_original_prescribed", "phase30_candidate_prescribed"}


def test_phase33_truth_and_metrics_complete():
    report = json.loads(REPORT.read_text())
    assert report["truth_boundary"]["physics_used_for_qualification_not_tuning"] is True
    assert report["truth_boundary"]["CEM_PPO_training_real_robot"] is False
    for key in ("phase28_original_prescribed", "phase30_candidate_prescribed"):
        replay = report["paired_replay"][key]
        for field in ("q_tracking", "body_tracking", "root_tracking", "contact", "slip_p95_max_mps", "torque", "replay_joint_motion"):
            assert field in replay


def test_phase33_timebase_adapter_unit_gate():
    report = json.loads(REPORT.read_text())
    unit = report["preflight"]["timebase_adapter_unit"]
    assert all(value["pass"] for value in unit.values())
    for value in unit.values():
        assert value["checks"]["official_output_fps_50"] is True
        assert value["checks"]["duration_preserved_within_half_control_step"] is True
        assert value["checks"]["q_endpoints_exact"] is True
        assert value["checks"]["root_endpoints_exact"] is True
        assert value["checks"]["slerp_path_preserved"] is True
        assert value["checks"]["contact_phase_transition_count_preserved"] is True
