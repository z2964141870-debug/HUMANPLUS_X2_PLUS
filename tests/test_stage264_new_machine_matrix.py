from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_freezes_shape_and_can_preserve_failures_while_collecting_matrix():
    text = (ROOT / "tools/official_x2/run_stage264_new_machine_matrix.sh").read_text()
    assert '"medium:0.30" "low:0.25"' in text
    assert '"straight:6" "turn_right:3" "turn_left:3"' in text
    assert 'CONTINUE_AFTER_FAILURE="${CONTINUE_AFTER_FAILURE:-false}"' in text
    assert "reusing hash-preserved gate failure" in text
    assert '[[ ! -e "$output" ]]' in text
    assert "failure_count=$((failure_count + 1))" in text
    assert "if (( failure_count > 0 ))" in text
    assert "SIMULATOR_HTTP_PORT=31822" in text
    assert "MAX_ATTEMPTS=1" in text


def test_gate_runner_supports_an_opt_in_non_ephemeral_simulator_http_port():
    text = (ROOT / "tools/official_x2/run_official_gate_case.sh").read_text()
    assert 'SIMULATOR_HTTP_PORT="${SIMULATOR_HTTP_PORT:-}"' in text
    assert "SIMULATOR_HTTP_PORT >= 1024 && SIMULATOR_HTTP_PORT < 32768" in text
    assert "expected exactly one AimRT HTTP port 51822" in text
    assert "/simulator/default.yaml:ro" in text


def test_summarizer_requires_exactly_24_expected_cases():
    source = (ROOT / "tools/official_x2/summarize_stage264_new_machine_matrix.py").read_text()
    assert 'passes == len(rows) == 24' in source
    assert '"sha256": sha256(path)' in source
    assert '"natural_posture_pass": False' in source
    assert 'if args.allow_partial:' in source
    assert '"complete_matrix": len(rows) == 24' in source
    assert '"failures": failures' in source
    assert '"moving_to_stationary_action_jump_l2"' in source
    assert 'len(move_trace) != 200 or len(stop_trace) != 400' in source
