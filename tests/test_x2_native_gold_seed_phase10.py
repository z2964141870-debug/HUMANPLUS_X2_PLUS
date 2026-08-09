from __future__ import annotations

import json

import joblib
import numpy as np

from retarget.audit_x2_native_gold_seed_phase10 import (
    DEFAULT_OUTPUT,
    FPS,
    REPO,
    lock_head,
    phase_statistics,
    pose_aa_from_dof,
    split_plan,
)


def test_split_has_nonoverlap_and_temporal_embargo():
    plan = split_plan(3000)
    assert plan["train"] == [(0, 400), (400, 800), (800, 1200), (1200, 1600)]
    assert plan["held_out"] == [(1800, 2200), (2200, 2600), (2600, 3000)]
    assert plan["train"][-1][1] < plan["held_out"][0][0]


def test_pose_aa_keeps_root_orientation_and_axis_times_joint_angle():
    q = np.array([[0.5, -0.25]], dtype=np.float64)
    root_quat = np.array([[0.0, 0.0, np.sin(0.1), np.cos(0.1)]])
    names = ["a", "b"]
    contract = {"a": {"axis_xyz": [0, 1, 0]}, "b": {"axis_xyz": [1, 0, 0]}}
    pose = pose_aa_from_dof(q, root_quat, names, contract)
    assert pose.shape == (1, 3, 3)
    assert np.allclose(pose[0, 0], [0, 0, 0.2])
    assert np.allclose(pose[0, 1], [0, 0.5, 0])
    assert np.allclose(pose[0, 2], [-0.25, 0, 0])


def test_head_lock_preserves_body_and_zeros_head_velocity():
    q = np.arange(12, dtype=np.float64).reshape(3, 4)
    dq = q + 100
    names = ["body_a", "head_yaw", "body_b", "head_pitch"]
    contract = {
        "head_yaw": {"model_nominal_rad": 0.1},
        "head_pitch": {"model_nominal_rad": -0.2},
    }
    locked_q, locked_dq, _ = lock_head(q, dq, names, contract, ["head_yaw", "head_pitch"])
    assert np.array_equal(locked_q[:, [0, 2]], q[:, [0, 2]])
    assert np.all(locked_q[:, 1] == 0.1) and np.all(locked_q[:, 3] == -0.2)
    assert np.all(locked_dq[:, [1, 3]] == 0.0)


def test_phase_statistics_counts_ds_ss_ds():
    contact = {
        "left": np.ones(9, dtype=bool),
        "right": np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool),
    }
    metrics = phase_statistics(contact)
    assert metrics["single_support_ratio"] == 3 / 9
    assert metrics["ds_ss_ds_cycle_count"] == 1


def test_exported_report_and_assets_keep_truth_and_split_contract():
    report = json.loads(
        (REPO / "reports/retarget/x2_native_gold_seed_phase10.json").read_text(encoding="utf-8")
    )
    assert report["decision"]["exported"] is True
    assert report["truth_boundary"]["not_gmr_amass_silver_or_gmr_replacement"] is True
    assert report["truth_boundary"]["contact_is_model_geometry_not_grf_cop_wrench"] is True
    assert report["split"]["policy"]["random_adjacent_frame_split"] is False
    train = joblib.load(DEFAULT_OUTPUT / "train/official_native_dance_train.pkl")
    held = joblib.load(DEFAULT_OUTPUT / "held_out/official_native_dance_held_out.pkl")
    assert len(train) == 4 and len(held) == 3
    for entry in [*train.values(), *held.values()]:
        assert entry["dof"].shape == (400, 31)
        assert entry["pose_aa"].shape == (400, 32, 3)
        assert entry["smpl_joints"].shape == (400, 24, 3)
        assert entry["fps"] == FPS
