from pathlib import Path


def test_panel_finalizer_reconstructs_exact_24_case_matrix_and_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/retarget/finalize_x2_official_state_feedback_brake_panel.py").read_text(
        encoding="utf-8"
    )
    assert 'if len(rows) != 24' in text
    assert 'if len(args.case) != 24 or len(args.resource) != 24' in text
    assert '[row["case"] for row in source["cases"]]' in text
    assert 'summary.get("stop_brake_gain")' in text
    assert 'summary.get("stop_hold_latch_s") is not None' in text
    assert 'summary.get("action_bias_mode") == spec["supervisor"]' in text
    assert 'PASS_OFFICIAL_STATE_FEEDBACK_BRAKE_24_OF_24_LOCAL_CANDIDATE' in text
    assert 'PARTIAL_OFFICIAL_STATE_FEEDBACK_BRAKE_PANEL_STOP' in text
    assert 'FAIL_OFFICIAL_STATE_FEEDBACK_BRAKE_PANEL_STOP' in text


def test_panel_shell_runs_original_matrix_order_and_one_attempt_each():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_x2_official_state_feedback_brake_panel.sh").read_text(
        encoding="utf-8"
    )
    assert "for speed in medium low" in text
    assert "for motion in straight turn_right turn_left" in text
    assert 'if [ "$motion" = straight ]; then repeats=6; else repeats=3; fi' in text
    assert "MAX_ATTEMPTS=1" in text
    assert "STOP_CONTROLLER=brake_then_policy" in text
    assert "STOP_BRAKE_GAIN=1.5" in text
    assert "domain=$((120 + index))" in text
    assert 'test "$index" -eq 24' in text
