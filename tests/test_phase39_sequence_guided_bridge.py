import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/official_x2/phase39_sequence_guided_bridge.json"


def test_phase39_single_candidate_contract_and_stop() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    contract = result["pre_registration"]
    assert contract["single_candidate_no_retry"] is True
    assert contract["search_or_optimizer"] is False
    assert contract["root_prescribed_or_teleported"] is False
    assert len(result["target"]["source_ticks"]) == 51
    assert result["target"]["source_ticks"] == list(range(354, 405))
    assert result["decision"] == {
        "strict_dynamic_bridge_found": False,
        "training_unlocked": False,
        "route": "STOP_AND_REVIEW",
    }


def test_phase39_history_pass_does_not_hide_physical_failure() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    endpoint = result["diagnostics"]["endpoint_success_normalized"]
    timeline = result["timeline"]
    assert endpoint["previous_action"] <= 1.0
    assert endpoint["projected_gravity"] > 10.0
    assert result["diagnostics"]["endpoint_success_joint_support"] is False
    assert result["diagnostics"]["post_1s_union_support"] is False
    assert result["diagnostics"]["root_safety_through_post_1s"] is False
    assert timeline["bridge_root_z_min_m"] >= 0.55
    assert timeline["bridge_root_tilt_max_rad"] <= 0.30
    assert timeline["first_recovery_tilt_over_0p30_s"] == 0.34
    assert timeline["first_recovery_height_below_0p55_s"] == 0.58
