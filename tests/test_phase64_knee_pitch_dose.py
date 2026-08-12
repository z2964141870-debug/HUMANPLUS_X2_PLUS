import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase64_runner_has_latin_square_mirrored_knee_partition():
    source = (ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    for token in (
        "CWI_PHASE64_KNEE_DOSE_SCREEN",
        "def phase64_knee_pitch_mirrored_doses",
        'latin_slot = (row + column).remainder(8)',
        '("A_knee_m008", -0.008)',
        '("B_knee_m008", -0.008)',
        'biases[mask, 3] = value',
        'biases[mask, 9] = value',
        '"8x8_latin_square_slot_equals_row_plus_column_modulo_8"',
    ):
        assert token in source


def test_phase64_latin_square_cells_cover_every_row_and_column():
    cells = {slot: [] for slot in range(8)}
    for env_id in range(64):
        row, column = divmod(env_id, 8)
        cells[(row + column) % 8].append((row, column))
    for coordinates in cells.values():
        assert len(coordinates) == 8
        assert {row for row, _ in coordinates} == set(range(8))
        assert {column for _, column in coordinates} == set(range(8))


def test_phase64_preregistration_remains_diagnostic_and_fail_closed():
    prereg = json.loads(
        (ROOT / "reports/retarget/x2_knee_pitch_dose_phase64_prereg.json").read_text()
    )
    assert prereg["screen"]["normalized_action_doses"] == [0.0, -0.008, -0.01, -0.012]
    assert prereg["screen"]["optimizer_steps"] == 0
    assert prereg["decision_boundary"]["training_unlocked"] is False
    assert prereg["decision_boundary"]["deployment_unlocked"] is False
    assert "every phase-local gate" in prereg["selection_rule"]


def test_phase64_finalizer_requires_both_lanes_and_direct_pool():
    source = (ROOT / "tools/retarget/finalize_x2_knee_pitch_dose_phase64.py").read_text()
    assert 'for comparison_name in ("A", "B", "pooled")' in source
    assert 'selected = passing[0] if passing else None' in source
    assert '"training_unlocked": False' in source
    assert '"deployment_unlocked": False' in source


def test_phase64_backup_builder_is_fail_closed_before_remote_verification():
    source = (ROOT / "tools/retarget/build_x2_phase64_backup_manifest.py").read_text()
    assert 'parser.add_argument("--remote-verified", action="store_true")' in source
    assert '"remote_listing_byte_sizes_match": args.remote_verified' in source
    assert '"reports/retarget/x2_knee_pitch_dose_phase64_result.json.sha256"' in source
