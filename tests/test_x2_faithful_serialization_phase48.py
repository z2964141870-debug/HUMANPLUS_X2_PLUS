from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/retarget/audit_x2_faithful_serialization_phase48.py"


def test_phase48_is_serialization_only_and_uses_live_writer():
    text = SOURCE.read_text()
    ast.parse(text)
    assert "ModelSaveCallback.save_checkpoint(" in text
    assert 'writer_state.__dict__.setdefault("log_history", [])' in text
    assert '"optimizer_steps": 0' in text
    assert '"physics_steps": 0' in text
    assert ".backward(" not in text
    assert "optimizer.step(" not in text
    assert "mj_step(" not in text


def test_phase48_preregisters_fresh_five_update_without_running_it():
    text = SOURCE.read_text()
    assert '"authorized_in_phase48": False' in text
    assert '"initialization": "fresh_from_original_SONIC_source_checkpoint_not_Phase47_final"' in text
    assert '"outer_updates_max": 5' in text
    assert '"optimizer_steps_total_max": 100' in text
    assert '"held_optimizer_samples": 0' in text
    assert '"final_reload_gate_max_abs": 1e-6' in text
