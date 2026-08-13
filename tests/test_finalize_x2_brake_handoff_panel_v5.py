from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_finalizer_has_independent_recomputation_and_no_training_unlock():
    text = (ROOT / "tools/retarget/finalize_x2_brake_handoff_panel_v5.py").read_text()
    assert "summarize_rows(panel_rows" in text
    assert "strategy_key(validation_summaries" in text
    assert "handoff_gates(" in text
    assert "optimizer_steps\": 0" in text
    assert "global_filesystem_delta_bytes_informational_only" in text


def test_runner_does_not_optimize_or_apply_posture_residual():
    text = (ROOT / "scripts/run_x2_brake_handoff_panel_v5.py").read_text()
    for forbidden in ("torch.optim", ".backward(", "optimizer.step", "state_feedback_residual"):
        assert forbidden not in text
    assert "_reward_ignored" in text
    assert "top_validation_ids" in text
