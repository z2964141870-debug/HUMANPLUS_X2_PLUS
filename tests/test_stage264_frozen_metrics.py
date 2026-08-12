from pathlib import Path


def test_stage264_metric_card_binds_pitch_contact_and_videos():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/official_x2/summarize_stage264_frozen_metrics.py").read_text()
    assert "signed_root_pitch" in text
    assert "strict_slip_gate_pass" in text
    assert "side_video" in text and "turn_video" in text
    assert "new_machine_gate_replayed" in text


def test_stage264_metric_card_does_not_claim_natural_posture_or_silver():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/official_x2/summarize_stage264_frozen_metrics.py").read_text()
    assert '"natural_posture_pass": False' in text
    assert '"contact_clean_silver": False' in text
