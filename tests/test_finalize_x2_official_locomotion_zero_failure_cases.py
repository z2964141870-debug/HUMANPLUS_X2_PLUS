from pathlib import Path


def test_finalizer_has_fail_closed_three_way_decision():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/retarget/finalize_x2_official_locomotion_zero_failure_cases.py").read_text(
        encoding="utf-8"
    )
    assert "PASS_OFFICIAL_LOCOMOTION_ZERO_PANEL_PREREG_ONLY" in text
    assert "PARTIAL_OFFICIAL_LOCOMOTION_ZERO_BRAKE_SKILL_ONLY" in text
    assert "FAIL_OFFICIAL_LOCOMOTION_ZERO_INTEGRATION_STOP" in text
    assert 'resource.get("raw_returncode") in (0, 2)' in text
    assert 'summary["stop_controller"] != "locomotion_zero"' in text
