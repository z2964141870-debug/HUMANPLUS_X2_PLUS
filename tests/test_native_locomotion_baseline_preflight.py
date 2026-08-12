from pathlib import Path


def test_preflight_freezes_stage264_as_stage219_plus_stage250_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/official_x2/audit_native_locomotion_baseline_preflight.py").read_text()
    assert "stage219_s2600_actor.onnx" in text
    assert "Stage250 deployment contract" in text
    assert "not a separate checkpoint" in text
    assert "allow_gate_replay" in text
    assert "allow_continued_training" in text


def test_preflight_requires_exact_checkpoint_and_template_hashes():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/official_x2/audit_native_locomotion_baseline_preflight.py").read_text()
    assert "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb" in text
    assert "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d" in text
    assert 'path.is_file()' in text
