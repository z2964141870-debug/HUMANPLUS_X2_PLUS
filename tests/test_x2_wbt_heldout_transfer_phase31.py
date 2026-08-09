import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_heldout_transfer_phase31.json"


def test_phase31_fixed_heldout_and_frozen_contract():
    report = json.loads(REPORT.read_text())
    contract = report["frozen_contract"]
    assert contract["motions"] == ["AMASS-TURN-R-001", "PHUMA-RAISE-R-001", "AMASS-KICK-R-001"]
    assert contract["held_metric_used_for_tuning"] is False
    assert contract["fallback_motion_substitution"] is False
    assert contract["per_clip_tuning"] is False
    assert contract["phase29"]["gn_iterations"] == 8
    assert contract["phase30_time_dilation"] == 1.46
    assert all(row["split"] == "held_out" for row in report["motions"])


def test_phase31_explicit_adapters_and_complete_chain():
    report = json.loads(REPORT.read_text())
    assert report["summary"]["completed"] == 3
    for row in report["motions"]:
        assert row["source_adapter"]
        assert row["repair_diagnostics"]["no_frame_or_segment_deleted"] is True
        assert row["repair_diagnostics"]["root_xy_bound_respected"] is True
        assert row["retime_resample"]["source_contact_phase_count_before"] == row["retime_resample"]["source_contact_phase_count_after"]
        assert row["retime_resample"]["source_contact_transition_count_before"] == row["retime_resample"]["source_contact_transition_count_after"]


def test_phase31_truth_and_gate_consistency():
    report = json.loads(REPORT.read_text())
    truth = report["truth_boundary"]
    assert truth["physics_steps"] == 0
    assert truth["ppo_updates"] == 0
    assert truth["silver_is_not_gold"] is True
    assert report["decision"]["independent_data_gate_pass"] == (report["summary"]["retime_silver_count"] >= 1)
