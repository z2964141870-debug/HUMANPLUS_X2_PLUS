import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_hybrid_physical_ab_phase44.json"


def test_phase44_is_default_fail_closed_and_never_runs_without_seed():
    report = json.loads(REPORT.read_text())
    if not report["qualification"]["execution_unlock"]:
        assert report["decision"]["status"] == "BLOCKED_BY_NATIVE_SEED"
        assert report["decision"]["physical_ab_executed"] is False
        assert report["executions"] == []
    assert report["truth_boundary"]["ppo_or_optimizer"] is False
    assert report["truth_boundary"]["prescribed_is_not_free_balance"] is True


def test_phase44_upper_contract_is_bounded_and_direct_write_is_arm_only():
    report = json.loads(REPORT.read_text())
    upper = report["upper_contract"]
    assert upper["raw_shape"] == [200, 14]
    assert upper["bounded_shape"] == [200, 14]
    assert upper["bounded_qstep_max_rad"] <= upper["max_velocity_radps"] * 0.02 + 1e-12
    assert len(report["ab_contract"]["composition_direct_write_mask"]) == 14
    assert len(report["ab_contract"]["lower_direct_write_forbidden"]) == 15
    assert report["static_checks"]["gmr_lower_root_contact_absent"] is True


def test_phase44_requires_A_reproduction_before_B():
    report = json.loads(REPORT.read_text())
    assert report["ab_contract"]["A_must_reproduce_seed_before_B"] is True
    arms = [item["arm"] for item in report["executions"]]
    if "B" in arms:
        assert arms.index("A") < arms.index("B")
