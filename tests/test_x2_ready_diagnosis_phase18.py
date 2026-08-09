from __future__ import annotations

import official_x2.audit_x2_ready_logs_phase18 as phase18


def test_phase18_logs_do_not_identify_missing_readiness_component():
    report = phase18.audit(
        phase18.DEFAULT_RECORDER_LOG,
        phase18.DEFAULT_CONTROLLER_LOG,
        phase18.DEFAULT_SIM_LOG,
        phase18.DEFAULT_INNER,
    )
    assert report["questions"]["a_specific_snapshot_ready_missing_component"]["answer"] is None
    assert report["questions"]["b_only_more_than_5s_needed"]["answer"] is None
    assert report["questions"]["c_process_health"]["controller_loaded"] is True
    assert report["questions"]["c_process_health"]["simulator_initialized"] is True


def test_phase18_evidence_gate_forbids_domain217_capture():
    report = phase18.audit(
        phase18.DEFAULT_RECORDER_LOG,
        phase18.DEFAULT_CONTROLLER_LOG,
        phase18.DEFAULT_SIM_LOG,
        phase18.DEFAULT_INNER,
    )
    assert report["launcher_wait"]["nominal_budget_s"] == 5.0
    assert report["evidence_gate"]["wait_only_explanation_proven"] is False
    assert report["evidence_gate"]["domain217_capture_allowed"] is False
    assert report["truth_boundary"]["no_domain217_capture"] is True
    assert report["decision"]["status"] == "PHASE18_WAIT_ONLY_CAUSE_UNPROVEN_NO_CAPTURE"
