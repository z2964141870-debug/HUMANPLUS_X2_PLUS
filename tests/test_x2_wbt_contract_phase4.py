from __future__ import annotations

import json
from pathlib import Path

import joblib
import mujoco
import numpy as np

from retarget.run_x2_wbt_contract_phase4 import (
    DEFAULT_CONTACT_CACHE,
    DEFAULT_CURRENT,
    REPO,
    TurnOrientationVector,
    build_turn_orientation_teacher,
    compare_contact_labels,
    contact_frame_labels,
    event_frames,
    phase3,
)


def test_contact_frame_labels_uses_majority_without_smoothing():
    times = np.arange(0.0, 0.1, 0.001)
    active = {"left": np.zeros(len(times), dtype=bool), "right": np.zeros(len(times), dtype=bool)}
    active["left"][(times >= 0.02) & (times < 0.05)] = True
    labels = contact_frame_labels(active, times, frame_count=4, fps=30.0)
    assert labels["left"].dtype == bool
    assert labels["left"].tolist() == [False, True, False, False]


def test_contact_comparison_reports_confusion_and_event_error():
    official = {"left": np.array([1, 1, 0, 0, 1], bool), "right": np.ones(5, bool)}
    fk = {"left": np.array([1, 0, 0, 1, 1], bool), "right": np.ones(5, bool)}
    result = compare_contact_labels(fk, official)
    assert result["macro_agreement"] == 0.8
    assert result["per_side"]["left"]["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert event_frames(official["left"]) == {"touchdown": [0, 4], "liftoff": [2]}


def test_turn_zero_vector_is_explicit_noop_contract():
    assert TurnOrientationVector((0.0, 0.0, 0.0), 0.0, 0.0).is_zero
    assert not TurnOrientationVector((0.0, 0.1, 0.0), 0.0, 0.0).is_zero


def test_nonzero_turn_representation_is_finite_and_velocity_bounded():
    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_phase4_test")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_phase4_test")
    motions = joblib.load(DEFAULT_CURRENT)
    contacts = joblib.load(DEFAULT_CONTACT_CACHE)
    key = next(key for key, entry in motions.items() if entry["panel_role"] == "turn_right")
    official = {
        side: np.asarray(contacts[key]["official_contact"][side], dtype=bool)
        for side in ("left", "right")
    }
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    vector = TurnOrientationVector((0.0, 0.03, 0.0), 0.04, 0.20)
    candidate, diagnostics = build_turn_orientation_teacher(
        model, motions[key], official, vector, phase2
    )
    assert diagnostics["exact_zero_update"] is False
    assert np.all(np.isfinite(candidate["dof"]))
    assert np.allclose(np.linalg.norm(candidate["root_rot"], axis=1), 1.0, atol=1e-5)
    assert diagnostics["joint_step_max_rad"] <= 0.180001


def test_generated_report_keeps_turn_nonzero_panel_locked():
    report = json.loads(
        (REPO / "reports/retarget/x2_wbt_contract_phase4.json").read_text(encoding="utf-8")
    )
    assert report["truth_boundary"]["turn_nonzero_search_executed"] is False
    assert report["decision"]["checks"]["raw_reference_qvel_safe_to_adopt"] is False
    assert report["decision"]["checks"]["turn_orientation_zero_update_exact"] is True
    assert set(report["decision"]["official_geometry_misaligned_roles"]) == {
        "walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand"
    }
