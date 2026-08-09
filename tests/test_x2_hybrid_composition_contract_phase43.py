import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_hybrid_composition_contract_phase43.json"


def test_phase43_ownership_is_disjoint_and_never_readds_gmr_lower():
    report = json.loads(REPORT.read_text())
    contract = report["contract"]
    assert len(contract["native_owned"]["lower12"]) == 12
    assert len(contract["native_owned"]["waist3"]) == 3
    assert len(contract["human_owned"]["upper14"]) == 14
    assert "GMR lower" in contract["forbidden"]
    assert report["roundtrip"]["B_native_fields_exact"] is True


def test_phase43_zero_change_roundtrip_and_truth_boundary():
    report = json.loads(REPORT.read_text())
    assert report["roundtrip"]["A_official31_wbt29_official31_max_error"] == 0.0
    assert report["roundtrip"]["head_excluded_from_wbt29"] is True
    assert report["truth_boundary"]["gait_phase_is_intended_contact_not_realized_collision"] is True
    assert report["truth_boundary"]["physics_or_replay_run"] is False
    assert report["truth_boundary"]["ppo_or_optimizer"] is False


def test_phase43_fail_closed_on_phase27_and_realized_contact():
    report = json.loads(REPORT.read_text())
    expected = bool(
        report["checks"]["base_phase27_available"]
        and report["checks"]["base_phase27_qualified"]
        and report["checks"]["stage250_realized_contact_available"]
        and all(value for key, value in report["checks"].items() if key not in (
            "base_phase27_available", "base_phase27_qualified", "stage250_realized_contact_available",
            "base_phase27_warm_start_usable", "base_phase27_dynamics_ground_truth",
        ))
    )
    assert report["decision"]["composition_contract_ready"] == expected
    assert report["decision"]["physics_ab_allowed"] == expected
