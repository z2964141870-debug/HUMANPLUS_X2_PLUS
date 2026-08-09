import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_half_upper_closed_phase53.json"


def test_phase53_is_exactly_one_half_amplitude_episode():
    r = json.loads(REPORT.read_text())
    assert r["execution"] == {"new_physics_episodes": 1, "retries": 0, "A_reused": True, "B1_reused": True}
    assert r["static"]["only_changed"] == {
        "upper_scale": [0.25, 0.125],
        "upper_max_excursion_rad": [0.12, 0.06],
        "upper_max_velocity_radps": [0.2, 0.1],
    }
    assert r["static"]["frozen_contract_exact_except_preregistered_half_scale_excursion_slew"] is True


def test_phase53_decision_is_all_preregistered_gates_without_scan():
    r = json.loads(REPORT.read_text())
    expected = "RETAIN_UPPER_AMPLITUDE_ROUTE" if all(r["gates"].values()) else "STOP_UPPER_AMPLITUDE_ROUTE"
    assert r["decision"]["status"] == expected
    assert "0.25/0.75" in r["decision"]["conclusion"] if expected.startswith("STOP") else True


def test_phase53_contact_truth_boundary_is_explicit():
    r = json.loads(REPORT.read_text())
    assert r["evidence_boundary"]["contact_is_model_truth_not_hardware"] is True
    assert r["truth_boundary"].endswith("not hardware GRF/COP")
