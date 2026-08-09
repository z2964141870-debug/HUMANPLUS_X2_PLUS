from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase21_trainer_isolated_role_aware_contract():
    source = (ROOT / "scripts/train_x2_stage221_official_ankle_match.py").read_text()
    assert '"--outcome_aware_reset_manifest"' in source
    assert '"--outcome_aware_reset_fraction"' in source
    assert '"state_role_name": "balanced"' in source
    assert "legacy recovery dataset and Phase20 manifest are mutually exclusive" in source
    assert "Phase20 outcome-aware reset is isolated to --profile stand_backend" in source
    assert "stage306_s2652_transition_head_actor.onnx" not in source


def test_phase21_runner_is_a_single_frozen_pair():
    runner = (ROOT / "scripts/run_phase21_outcome_aware_paired_u5.sh").read_text()
    assert "SEED=47" in runner
    assert "UPDATES=5" in runner
    assert 'NUM_ENVS="${NUM_ENVS:-64}"' in runner
    assert "run_one f000 0.00" in runner
    assert "run_one f005 0.05" in runner
    assert "--profile stand_backend" in runner
    assert "--weights_only_resume" in runner
    assert "--outcome_aware_reset_manifest" in runner
    assert "--outcome_aware_reset_fraction \"$fraction\"" in runner
    assert "--recovery_reset_dataset" not in runner
    assert "stage306_s2652_transition_head_actor.onnx" not in runner
    assert "model_155.pt" in runner


def test_phase21_official_gate_has_frozen_roles_and_three_models():
    runner = (ROOT / "tools/official_x2/run_phase21_outcome_aware_paired_gate.sh").read_text()
    assert "labels=(source f000 f005)" in runner
    assert "REPEATS=5" in runner
    assert "MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx" in runner
    assert "STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx" in runner
    assert 'RECOVERY_MODEL_PATH="/models/$recovery"' in runner
    assert "CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS=0.5" in runner
    assert "STOP_CONTROLLER=curriculum_then_policy" in runner
    assert "PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2" in runner
    assert 'expected_counts = {"main": 360, "stationary": 100, "recovery": 300}' in runner
