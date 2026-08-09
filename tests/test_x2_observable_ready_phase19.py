from __future__ import annotations

import official_x2.audit_x2_observable_ready_phase19 as phase19


def test_phase19_diagnostic_identifies_complete_zero_mode_telemetry_absence():
    report = phase19.audit(
        phase19.DEFAULT_READINESS,
        phase19.DEFAULT_RECORDER_LOG,
        phase19.DEFAULT_CONTROLLER_LOG,
        phase19.DEFAULT_SIM_LOG,
    )
    ready = report["readiness"]
    assert ready["diagnostic_elapsed_ns"] >= 15_000_000_000
    assert ready["state_joint_seen_count"] == 0
    assert all(row["seen"] == 0 for row in ready["state_by_group"].values())
    assert ready["odom_seen"] is False
    assert ready["imu_seen"] is False


def test_phase19_stops_before_mode_and_physics():
    report = phase19.audit(
        phase19.DEFAULT_READINESS,
        phase19.DEFAULT_RECORDER_LOG,
        phase19.DEFAULT_CONTROLLER_LOG,
        phase19.DEFAULT_SIM_LOG,
    )
    assert report["qualification"]["passed"] is False
    assert report["checks"]["joint_mode_not_published"] is True
    assert report["checks"]["rl_mode_not_published"] is True
    assert report["checks"]["npz_not_written"] is True
    assert report["truth_boundary"]["no_prescribed_or_free_physics"] is True
    assert report["decision"]["status"] == "PHASE19_ZERO_MODE_TELEMETRY_ABSENT_NO_CAPTURE"
