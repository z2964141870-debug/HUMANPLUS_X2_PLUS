from pathlib import Path


def test_state_feedback_finalizer_binds_controller_and_three_way_decision():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "tools/retarget/finalize_x2_official_state_feedback_brake_failure_cases.py"
    ).read_text(encoding="utf-8")
    assert 'summary.get("stop_brake_gain")' in text
    assert 'summary.get("stop_brake_limit_mps")' in text
    assert 'summary.get("stop_hold_latch_s") is not None' in text
    assert 'token.split("=", 1)' in text
    assert 'float(command_env.get("STOP_BRAKE_LIMIT", "nan"))' in text
    assert '"resource_parameters": all(resource_parameter_checks.values())' in text
    assert "PASS_OFFICIAL_STATE_FEEDBACK_BRAKE_PANEL_PREREG_ONLY" in text
    assert "PARTIAL_OFFICIAL_STATE_FEEDBACK_BRAKE_SKILL_ONLY" in text
    assert "FAIL_OFFICIAL_STATE_FEEDBACK_BRAKE_INTEGRATION_STOP" in text
    assert 'resource.get("raw_returncode") in (0, 2)' in text
    assert 'signed_prereg.get("schema") == "x2_official_state_feedback_brake_refinalize_v1"' in text
    assert "verify_sidecar(inherited_path)" in text
    assert 'sha256(inherited_path) != inherited["sha256"]' in text
