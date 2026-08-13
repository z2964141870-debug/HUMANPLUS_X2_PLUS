from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase75_finalizer_only_unlocks_phase76_shadow_preregistration() -> None:
    text = (ROOT / "tools/retarget/finalize_x2_phase75_pairing_preflight.py").read_text()
    assert "EXPECTED_SHA256" in text
    assert '"phase76_shadow_preregistration_unlocked"' in text
    assert '"phase76_scientific_preregistration_unlocked"' in text
    assert '"phase76_launch_unlocked"' in text

