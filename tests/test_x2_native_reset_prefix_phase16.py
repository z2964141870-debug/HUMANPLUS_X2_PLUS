from __future__ import annotations

import official_x2.audit_x2_native_reset_prefix_phase16 as phase16
import json
from pathlib import Path


def test_phase16_single_capture_records_exact_mode_sequence_but_prefix_is_short():
    report = phase16.audit(
        phase16.DEFAULT_SOURCE, phase16.DEFAULT_MANIFEST, phase16.DEFAULT_CONTROLLER_LOG
    )
    capture = report["capture"]
    assert [row["inferred_mode"] for row in capture["mode_events"]] == [
        "JOINT_DEFAULT", "RL_DEFAULT"
    ]
    assert capture["mode_header_populated"] == [False, False]
    assert capture["raw_joint_to_rl_interval_s"] >= 4.0
    assert capture["complete_reference_prefix_before_rl_s"] < 1.0
    assert capture["recorded_complete_snapshot_rl_segment_s"] >= 20.0


def test_phase16_contract_failure_stops_before_physics():
    report = phase16.audit(
        phase16.DEFAULT_SOURCE, phase16.DEFAULT_MANIFEST, phase16.DEFAULT_CONTROLLER_LOG
    )
    gate = report["pre_registered_contract"]
    assert gate["passed"] is False
    assert gate["checks"]["complete_reference_prefix_at_least_4s"] is False
    assert report["decision"]["status"] == "PHASE16_RESET_PREFIX_CONTRACT_REJECTED_NO_PHYSICS"
    assert report["truth_boundary"]["physics_replay_executed"] is False
    assert report["truth_boundary"]["free_root_executed"] is False


def test_phase17_failure_report_proves_no_mode_or_physics_after_ready_timeout():
    path = Path(__file__).resolve().parents[1] / "reports/retarget/x2_native_reset_ready_phase17.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    attempt = report["capture_attempt"]
    assert attempt["attempt_count"] == 1
    assert attempt["ready_receipt_s"] is None
    assert attempt["joint_mode_receipt_s"] is None
    assert attempt["rl_mode_receipt_s"] is None
    assert attempt["npz_written"] is False
    assert report["qualification_gate"]["passed"] is False
    assert report["truth_boundary"]["second_capture_not_run"] is True
    assert report["truth_boundary"]["prescribed_physics_executed"] is False
