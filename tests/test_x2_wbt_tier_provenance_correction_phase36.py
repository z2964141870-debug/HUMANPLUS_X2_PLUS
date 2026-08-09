import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_tier_provenance_correction_phase36.json"


def test_phase36_is_append_only_and_thresholds_unchanged():
    report = json.loads(REPORT.read_text())
    assert report["truth_boundary"]["physics_integration_steps"] == 0
    assert report["truth_boundary"]["optimizer_training_ppo"] is False
    assert report["truth_boundary"]["historical_reports_overwritten"] is False
    assert report["contract_correction"]["correct_official_collision_signed_distance_threshold_m"] == 0.0
    assert report["contract_correction"]["active_sole_spheres_per_foot"] == 12


def test_phase36_fixed_panel_complete_and_collision_contract_exact():
    report = json.loads(REPORT.read_text())
    assert report["phase28_summary"]["completed"] == 24
    for phase in ("phase28_corrected", "phase29_corrected", "phase30_corrected"):
        for row in report[phase]:
            assert row["corrected"]["contact"]["collision_signed_distance_exact"] is True


def test_phase36_revokes_false_phase30_silver_and_optimizer_qualification():
    report = json.loads(REPORT.read_text())
    lunge = next(row for row in report["phase30_corrected"] if row["id"] == "PHUMA-LUNGE-R-001")
    assert lunge["historical_tier"] == "Silver"
    assert lunge["corrected"]["tier"] != "Silver"
    assert lunge["corrected"]["contact"]["unintended_flight_fraction"] > 0.95
    assert lunge["corrected"]["contact"]["official_geometry_contact_ratio"]["right"] == 0.0
    assert report["decision"]["phase30_lunge_silver_revoked"] is True
    assert report["decision"]["faithful_any2any_optimizer_or_ppo_qualified"] is False
