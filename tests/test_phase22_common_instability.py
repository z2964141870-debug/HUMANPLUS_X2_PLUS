from pathlib import Path

from official_x2.audit_phase22_common_instability import build_report


def test_phase22_real_panel_has_common_sequence_and_no_contact_overclaim():
    report = build_report(
        Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"),
        Path("manifests/x2_phase19_outcome_aware_state_role.json"),
        Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"),
        Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"),
    )
    assert report["trace_count"] == 15
    assert report["analysis_only"] is True
    assert report["causal_claim"] is False
    assert report["interpretation"]["physical_contact_observed"] is False
    assert report["interpretation"]["post_fall_smaller_drift_is_recovery"] is False
    for label in ("source", "f000", "f005"):
        group = report["groups"][label]
        assert group["event_times_after_handoff_s"]["heading_official_threshold"]["max"] < 0.0
        assert group["event_times_after_handoff_s"]["lateral_official_threshold"]["max"] < 0.0
        assert group["event_times_after_handoff_s"]["root_tilt_gt_0p30"]["count"] == 5
        assert group["event_times_after_handoff_s"]["root_z_lt_0p45"]["count"] == 5
        assert set(group["support_phase_at_tilt"]) == {"generator_double_support"}


def test_phase22_f005_does_not_turn_postfall_drift_into_recovery_claim():
    report = build_report(
        Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"),
        Path("manifests/x2_phase19_outcome_aware_state_role.json"),
        Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"),
        Path("reports/official_x2/stage335_stop_recovery_state_extraction.json"),
    )
    source = report["groups"]["source"]["event_times_after_handoff_s"]
    candidate = report["groups"]["f005"]["event_times_after_handoff_s"]
    assert candidate["root_tilt_gt_0p30"]["median"] < source["root_tilt_gt_0p30"]["median"]
    assert candidate["root_z_lt_0p45"]["median"] < source["root_z_lt_0p45"]["median"]
    assert report["next_single_falsifiable_intervention"]["training_required"] is False
