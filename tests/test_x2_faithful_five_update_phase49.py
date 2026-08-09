from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/x2_faithful_five_update_phase49.py"
LAUNCHER = ROOT / "scripts/run_x2_faithful_phase49_segment.sh"
GATE = ROOT / "tools/retarget/gate_x2_faithful_phase49_update.py"


def test_phase49_segment_is_one_update_with_five_update_scheduler_horizon():
    text = SOURCE.read_text()
    ast.parse(text)
    assert "_StopAfterOneOuterUpdate" in text
    assert '"configured_outer_updates": 5' in text
    assert '"transitions_per_update": 1536' in text
    assert '"optimizer_steps_per_update": 20' in text
    assert "kl >= 0.02" in text


def test_phase49_resume_and_scope_are_fail_closed():
    text = SOURCE.read_text()
    assert "Phase49 resume step drift" in text
    assert "held-out key entered Phase49 optimizer" in text
    assert "Phase49 frozen parameter hash changed" in text
    launcher = LAUNCHER.read_text()
    assert "ITERS=5" in launcher
    assert 'CHECKPOINT="$(realpath "${CHECKPOINT}")"' in launcher
    assert 'RESUME="${RESUME}"' in launcher
    assert 'PRESERVE_LORA_CHECKPOINT="${PRESERVE}"' in launcher
    assert "SAVE_LAST_FREQUENCY=1" in launcher
    regression = (ROOT / "scripts/run_x2_faithful_phase49_regression.sh").read_text()
    assert 'CHECKPOINT="$(realpath "${CHECKPOINT}")"' in regression


def test_phase49_gate_uses_frozen_source_and_all_hard_stops():
    text = GATE.read_text()
    ast.parse(text)
    assert "x2_faithful_phase47_regression_pre_train" in text
    assert "x2_faithful_phase47_regression_pre_held_out" in text
    assert '"kl_below_0p02"' in text
    assert '"survival_drop_le_0p1s"' in text
    assert '"termination_increase_le_0p05"' in text
    assert '"no_tracking_error_increases_over_10pct"' in text
    assert '"long_training_authorized": False' in text


def test_phase49_final_report_records_hard_stop_before_update5():
    import json
    report = json.loads(
        (ROOT / "reports/retarget/x2_faithful_exact_s7_five_update_phase49.json").read_text()
    )
    assert report["decision"] == "HARD_STOP_AT_UPDATE4_UPDATE5_NOT_RUN"
    assert report["executed_updates"] == 4
    assert report["executed_optimizer_steps"] == 80
    assert report["hard_stop"]["update5_executed"] is False
    assert report["best_checkpoint"]["update"] == 2
    assert report["truth_boundary"]["long_training_authorized"] is False
