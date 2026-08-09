from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from retarget.run_x2_joint_reference_repair_phase2 import (
    RepairParameters,
    bounded_root_correction,
    phase_contract,
    search_grid,
    shifted,
)


def test_shifted_is_clamped_not_wrapped():
    values = np.arange(5)
    assert shifted(values, 2).tolist() == [2, 3, 4, 4, 4]
    assert shifted(values, -2).tolist() == [0, 0, 0, 1, 2]


def test_phase_contract_does_not_invent_swing_when_ambiguous():
    kin = {
        "left_contact_probability": np.ones(6) * 0.7,
        "right_contact_probability": np.ones(6) * 0.7,
        "left_foot": np.column_stack([np.zeros((6, 2)), np.ones(6) * 0.02]),
        "right_foot": np.column_stack([np.zeros((6, 2)), np.ones(6) * 0.02]),
    }
    phase = phase_contract(kin)
    assert phase["double_support"].all()
    assert not phase["left_swing"].any()
    assert not phase["right_swing"].any()


def test_root_correction_is_strictly_bounded():
    frames = 12
    root = np.zeros((frames, 3))
    kin = {
        "left_foot": np.tile([1.0, 1.0, 0.2], (frames, 1)),
        "right_foot": np.tile([-1.0, -1.0, -0.2], (frames, 1)),
        "ground_z_m": 0.0,
    }
    phase = {
        "stance_left": np.ones(frames, dtype=bool),
        "stance_right": np.zeros(frames, dtype=bool),
        "double_support": np.zeros(frames, dtype=bool),
    }
    _, correction = bounded_root_correction(root, kin, phase, 1.0, 1.0)
    assert np.max(np.linalg.norm(correction[:, :2], axis=1)) <= 0.0400001
    assert np.max(np.abs(correction[:, 2])) <= 0.0250001


def test_search_grid_is_bounded_and_excludes_noop():
    grid = search_grid()
    assert len(grid) == 48
    assert all(0.0 < value.root_support_gain <= 0.5 for value in grid)
    assert all(abs(value.contact_phase_shift_frames) <= 2 for value in grid)
    assert all(value.swing_clearance_m <= 0.020 for value in grid)
    assert not any(value.is_noop for value in grid)


def test_noop_contract_requires_every_intervention_off():
    assert RepairParameters(0, 0, 0, 0, 0, stance_lock_gain=0).is_noop
    assert not RepairParameters(0, 0, 0, 0, 0).is_noop


def test_generated_report_preserves_gate_and_truth_boundaries():
    report_path = Path(__file__).resolve().parents[1] / (
        "reports/retarget/x2_joint_reference_repair_phase2.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["truth_boundary"]["oracle_deployable"] is False
    assert report["truth_boundary"]["training_or_checkpoint_load"] is False
    assert "hardware GRF" in report["truth_boundary"]["not_measured"]
    assert set(report["offline_search"]) == {
        "walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand"
    }
    assert all(value["offline_pass_count"] > 0 for value in report["offline_search"].values())
    variants = set(report["physics_aggregate"])
    assert variants == {
        "current_v4_exact30", "toe_to_forefoot_exact30_smooth9", "joint_repair_oracle"
    }
    assert all(
        set(modes) == {"prescribed_root_trackability", "free_root_balance"}
        for modes in report["physics_aggregate"].values()
    )
