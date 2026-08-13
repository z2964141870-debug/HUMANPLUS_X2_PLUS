from pathlib import Path


def test_repair_is_exactly_six_left_cases_and_no_promotion():
    root = Path(__file__).resolve().parents[1]
    shell = (root / "scripts/run_x2_official_state_feedback_brake_panel_repair.sh").read_text()
    finalizer = (root / "tools/retarget/finalize_x2_official_state_feedback_brake_panel_repair.py").read_text()
    assert "for speed in medium low" in shell
    assert "for repeat in 1 2 3; do" in shell
    assert "MIRROR_POLICY=true" in shell
    assert 'test "$repair_index" -eq 6' in shell
    assert "x2_sfbrake_repair" in finalizer
    assert 'sum(row["repaired"] for row in rows) != 6' in finalizer
    assert "COMPLETE_REPAIRED_PANEL_DIAGNOSTIC_NO_PROMOTION" in finalizer
    assert '"baidu_access_or_upload": False' in finalizer


def test_attempt0_script_remains_frozen_with_bad_literal_as_provenance():
    root = Path(__file__).resolve().parents[1]
    old = (root / "scripts/run_x2_official_state_feedback_brake_panel.sh").read_text()
    assert "MIRROR_POLICY=1" in old
