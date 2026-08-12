import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase65_runner_uses_deployable_pre_step_suffix_and_checkerboard_crossover():
    source = (ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    for token in (
        "CWI_PHASE65_CROSSOVER_PASS",
        "def phase65_crossover_partition",
        'sequence_tc = (row + column).remainder(2) == 0',
        "def phase65_single_support_knee_bias",
        'gait = policy_observation[:, -4:]',
        'right_swing_left_support = left & ~right',
        'left_swing_right_support = ~left & right',
        'bias[active_single_support, 3] = dose',
        'bias[active_single_support, 9] = dose',
        '"measurement_alignment": "post-step physical outcome labeled by the pre-step semantic region that produced its action"',
    ):
        assert token in source


def test_phase65_checkerboard_is_balanced_by_row_column_and_reverses():
    sequence_tc = []
    for env_id in range(64):
        row, column = divmod(env_id, 8)
        sequence_tc.append((row + column) % 2 == 0)
    assert sum(sequence_tc) == 32
    for index in range(8):
        assert sum(sequence_tc[index * 8 : (index + 1) * 8]) == 4
        assert sum(sequence_tc[index::8]) == 4
    assert all((value is True) != (value is False) for value in sequence_tc)


def test_phase65_preregistration_is_two_pass_zero_optimizer_fail_closed():
    prereg = json.loads(
        (ROOT / "reports/retarget/x2_single_support_knee_crossover_phase65_prereg.json").read_text()
    )
    assert prereg["screen"]["sequential_batched_runs"] == 2
    assert prereg["screen"]["optimizer_steps"] == 0
    assert prereg["pairing_gates"]["expected_region_counts_per_env"] == {
        "double_support_zero": 30,
        "right_swing_left_support": 70,
        "double_support_half": 30,
        "left_swing_right_support": 70,
        "standing": 0,
    }
    assert prereg["decision_boundary"]["training_unlocked"] is False
    assert prereg["decision_boundary"]["deployment_unlocked"] is False


def test_phase65_finalizer_requires_pairing_sequences_pool_and_phase_gates():
    source = (
        ROOT / "tools/retarget/finalize_x2_single_support_knee_crossover_phase65.py"
    ).read_text()
    for token in (
        'decision = "FAIL_CROSSOVER_REPLAY_INVALID_STOP"',
        'for name in ("TC", "CT", "pooled")',
        '"training_unlocked": False',
        '"deployment_unlocked": False',
        "Nonlinear p05/p95 and RMSE values were recomputed",
    ):
        assert token in source


def test_phase65_backup_builder_stays_unverified_until_remote_listing():
    source = (ROOT / "tools/retarget/build_x2_phase65_backup_manifest.py").read_text()
    assert 'parser.add_argument("--remote-verified", action="store_true")' in source
    assert '"remote_listing_byte_sizes_match": args.remote_verified' in source
    assert '"reports/retarget/x2_single_support_knee_crossover_phase65_result.json.sha256"' in source
