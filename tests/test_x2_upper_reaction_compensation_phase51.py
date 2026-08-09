import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_upper_reaction_compensation_phase51.json"


def test_phase51_uses_model_estimates_and_never_calls_them_hardware_truth():
    r = json.loads(REPORT.read_text())
    assert r["model_contract"]["centroidal_values_are_model_estimates_not_hardware_truth"] is True
    assert r["scope"].startswith("offline model calculation")


def test_phase51_has_one_minimal_zero_upper_fallback_contract():
    r = json.loads(REPORT.read_text())
    b2 = r["preregistered_B2_if_unlocked"]
    assert b2["direct_leg12_write"] is False
    assert b2["zero_upper_exact_fallback"] is True
    assert b2["episodes"] == 1 and b2["retries"] == 0
    assert b2["gain"] == 1.0 and b2["time_shift_s"] == 0.0 and b2["parameter_scan"] is False


def test_phase51_execution_is_fail_closed_by_offline_evidence():
    r = json.loads(REPORT.read_text())
    unlocked = any(row["proven_offline"] for row in r["candidate_interfaces"].values())
    assert r["decision"]["physics_execution_unlocked"] is unlocked
    if not unlocked:
        assert r["decision"]["status"].startswith("STOP_OFFLINE")
