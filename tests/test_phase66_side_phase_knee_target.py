import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase66_runner_uses_latin_side_phase_cells_and_final_target_layer():
    source = (ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    for token in (
        "CWI_PHASE66_SIDE_PHASE_PASS",
        "def phase66_side_phase_partition",
        'slots = (row + column).remainder(8)',
        "def phase66_physical_knee_target_offset",
        'target_offset_rad: float = -0.003',
        'requested[active & (target_side == 0), 3] = target_offset_rad',
        'requested[active & (target_side == 1), 9] = target_offset_rad',
        'phase66_source_target = term._processed_actions.clone()',
        'term._processed_actions[:] = proposed',
    ):
        assert token in source


def test_phase66_latin_cells_cover_each_row_and_column():
    cells = {slot: [] for slot in range(8)}
    for env_id in range(64):
        row, column = divmod(env_id, 8)
        cells[(row + column) % 8].append((row, column))
    for coordinates in cells.values():
        assert {row for row, _ in coordinates} == set(range(8))
        assert {column for _, column in coordinates} == set(range(8))


def test_phase66_prereg_is_diagnostic_zero_optimizer_and_uses_live_counts():
    prereg = json.loads(
        (ROOT / "reports/retarget/x2_side_phase_knee_target_phase66_prereg.json").read_text()
    )
    assert prereg["screen"]["optimizer_steps"] == 0
    assert prereg["pairing_gates"]["expected_live_region_counts_per_env"] == {
        "double_support_zero": 29,
        "right_swing_left_support": 70,
        "double_support_half": 30,
        "left_swing_right_support": 71,
        "standing": 0,
    }
    assert len(prereg["screen"]["allocation"]["condition_order"]) == 8
    assert prereg["decision_boundary"]["training_unlocked"] is False


def test_phase66_finalizer_requires_same_index_pairing_and_mirrored_roles():
    source = (ROOT / "tools/retarget/finalize_x2_side_phase_knee_target_phase66.py").read_text()
    for token in (
        'decision = "FAIL_SIDE_PHASE_REPLAY_INVALID_STOP"',
        '"stance": all(row_by_name[name]["passed"] for name in role_rule["stance_pair"])',
        '"swing": all(row_by_name[name]["passed"] for name in role_rule["swing_pair"])',
        '"training_unlocked": False',
        '"deployment_unlocked": False',
    ):
        assert token in source
