from __future__ import annotations

import json
from pathlib import Path

import retarget.run_x2_native_activation_prefix_phase21 as phase21


REPO = Path(__file__).resolve().parents[1]


def test_phase21_capture_qualification_records_all_five_timepoints():
    q = phase21.qualification(
        phase21.DEFAULT_SOURCE, phase21.DEFAULT_MANIFEST, phase21.DEFAULT_CONTROLLER
    )
    assert q["passed"] is True
    assert q["subscription_ready_s"] < q["joint_mode_s"] < q["first_complete_snapshot_s"]
    assert q["four_second_prefix_achieved_s"] <= q["rl_mode_s"]
    assert q["scoreable_joint_prefix_s"] >= 4.0
    assert q["rl_segment_s"] >= 20.0


def test_phase21_saved_report_obeys_conditional_free_gate():
    path = REPO / "reports/retarget/x2_native_activation_prefix_phase21.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["capture_qualification"]["passed"] is True
    assert report["truth_boundary"]["actual_state_is_reference_command_is_control"] is True
    if not report["gate"]["prescribed"]["pass"]:
        assert report["gate"]["free"] is None
        assert set(report["replay"]) == {phase21.phase12.MODE_PRESCRIBED}
        if "source_joint_step" in report["gate"]["prescribed"]["failed"]:
            assert report["decision"]["status"] == "PHASE21_SOURCE_PREFIX_REJECTED_PRESCRIBED_FAILED"
