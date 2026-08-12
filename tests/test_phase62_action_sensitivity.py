from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase62_screen_is_one_batched_run_without_weights():
    source = (ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    for token in (
        'CWI_PHASE62_ACTION_SENSITIVITY',
        'choices=("train", "eval", "screen")',
        'epsilon = 0.02',
        '("hip_pitch_neg", (0, 6), -epsilon)',
        '("knee_pos", (3, 9), epsilon)',
        '("ankle_pitch_pos", (4, 10), epsilon)',
        '("waist_pitch_neg", (13,), -epsilon)',
        '"checkpoint_modified": False',
        'ideal_env_fraction=1.0 if ACTION_SCREEN or RESIDUAL_DIAGNOSTIC else 0.75',
        '"phase": PHASE, "mode": args.mode',
    ):
        assert token in source


def test_phase62_finalizer_is_diagnostic_only():
    source = (ROOT / "tools/retarget/finalize_x2_action_sensitivity_phase62.py").read_text()
    assert '"training_unlocked": False' in source
    assert '"deployment_unlocked": False' in source
    assert 'for joint in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch")' in source
    assert '"legacy_eval_mode_tag_accepted": legacy_eval_tag' in source
