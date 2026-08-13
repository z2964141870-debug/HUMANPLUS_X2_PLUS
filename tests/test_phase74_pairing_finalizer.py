from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase74_finalizer_boundaries_are_technical_only() -> None:
    text = (ROOT / "tools/retarget/finalize_x2_phase74_pairing_preflight.py").read_text()
    assert "PASS_INITIAL_PAIRING_PREFLIGHT_ONLY" in text
    assert '"phase75_shadow_preregistration_unlocked": valid' in text
    assert '"phase75_scientific_preregistration_unlocked": False' in text
    assert '"phase75_launch_unlocked": False' in text
    assert '"training_unlocked": False' in text
    assert '"deployment_unlocked": False' in text
    assert "len(set(source_initial_hashes)) == 3" in text
    assert 'row.get("unknown_mutable_tensor_fields_empty") is True' in text
