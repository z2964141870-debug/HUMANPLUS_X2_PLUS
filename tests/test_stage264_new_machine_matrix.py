from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_freezes_the_24_case_shape_and_reuses_only_verified_passes():
    text = (ROOT / "tools/official_x2/run_stage264_new_machine_matrix.sh").read_text()
    assert '"medium:0.30" "low:0.25"' in text
    assert '"straight:6" "turn_right:3" "turn_left:3"' in text
    assert 'summary["full_gate_pass"] is True' in text
    assert "MAX_ATTEMPTS=1" in text


def test_summarizer_requires_exactly_24_expected_cases():
    source = (ROOT / "tools/official_x2/summarize_stage264_new_machine_matrix.py").read_text()
    assert 'passes == len(rows) == 24' in source
    assert '"sha256": sha256(path)' in source
    assert '"natural_posture_pass": False' in source
    assert 'if args.allow_partial:' in source
    assert '"complete_matrix": len(rows) == 24' in source
