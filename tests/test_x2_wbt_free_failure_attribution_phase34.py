import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_free_failure_attribution_phase34.json"


def test_phase34_is_read_only_and_honest_about_missing_trace():
    report = json.loads(REPORT.read_text())
    assert report["scope"]["new_physics_steps"] == 0
    assert report["scope"]["optimizer_or_training"] is False
    assert report["scope"]["phase33_per_frame_replay_trace_available"] is False
    assert len(report["existing_replay_evidence"]["missing"]) == 3


def test_phase34_source_contact_infeasibility_is_evidenced():
    report = json.loads(REPORT.read_text())
    contact = report["source_reference"]["full_contact_schedule"]
    assert contact["flight_ratio"] > 0.95
    assert contact["right_ratio"] == 0.0
    free = report["existing_replay_evidence"]["free_root_0_to_0p575_aggregate"]
    assert free["contact"]["agreement"]["mean"] < 0.1


def test_phase34_has_exactly_one_next_experiment():
    report = json.loads(REPORT.read_text())
    assert isinstance(report["single_next_falsifiable_experiment"], str)
    assert "exact-joint kinematic oracle" in report["single_next_falsifiable_experiment"]
    assert "warmup" in report["single_next_falsifiable_experiment"]
