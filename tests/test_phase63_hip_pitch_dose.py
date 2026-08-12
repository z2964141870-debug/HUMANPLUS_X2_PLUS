import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase63_runner_has_frozen_small_dose_partition():
    source = (ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    for token in (
        "CWI_PHASE63_HIP_DOSE_SCREEN",
        "def phase63_hip_pitch_doses",
        '("hip_pitch_m004", -0.004)',
        '("hip_pitch_m020", -0.020)',
        "biases[start:stop, [0, 6]] = value",
        '"checkpoint_modified": False',
        '"posture_variant": (',
    ):
        assert token in source


def test_phase63_preregistration_is_diagnostic_and_fail_closed():
    prereg = json.loads(
        (ROOT / "reports/retarget/x2_hip_pitch_dose_phase63_prereg.json").read_text()
    )
    assert prereg["screen"]["normalized_action_doses"] == [
        0.0, -0.004, -0.006, -0.008, -0.01, -0.012, -0.016, -0.02
    ]
    assert prereg["screen"]["optimizer_steps"] == 0
    assert prereg["decision_boundary"]["training_unlocked_by_this_screen"] is False
    assert prereg["decision_boundary"]["deployment_unlocked_by_this_screen"] is False


def test_phase63_finalizer_selects_smallest_passing_dose_only():
    source = (ROOT / "tools/retarget/finalize_x2_hip_pitch_dose_phase63.py").read_text()
    assert 'selected = passed[0] if passed else None' in source
    assert '"training_unlocked": False' in source
    assert '"deployment_unlocked": False' in source
    assert '"task2_complete": False' in source
