from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from x2_native_gold_motionlib_adapter import (
    apply_recorded_state_adapter,
    select_wbt29,
    wbt29_indices,
)


REPO = Path(__file__).resolve().parents[1]


class FakeMotionLib:
    pass


def test_wbt29_indices_and_selection_are_name_driven():
    official = [f"j{i}" for i in range(31)]
    wbt = [f"j{i}" for i in range(0, 31, 2)][:15] + [f"j{i}" for i in range(1, 29, 2)]
    assert len(wbt) == 29
    values = torch.arange(31).reshape(1, 31)
    selected = select_wbt29(values, official, wbt)
    assert selected.tolist()[0] == [official.index(name) for name in wbt]
    assert wbt29_indices(official, wbt).shape == (29,)


def test_state_adapter_restores_auxiliary_state_without_pose_change(tmp_path):
    frames = 4
    key = "clip"
    source = {
        key: {
            "dof": np.zeros((frames, 31), np.float32),
            "dof_vel": np.arange(frames * 31, dtype=np.float32).reshape(frames, 31),
            "root_lin_vel_w_mps": np.ones((frames, 3), np.float32),
            "root_ang_vel": np.ones((frames, 3), np.float32) * 2,
            "model_contact": {
                "left": np.array([1, 1, 0, 0], bool),
                "right": np.array([1, 0, 0, 1], bool),
            },
        }
    }
    path = tmp_path / "source.pkl"
    joblib.dump(source, path)
    motion = FakeMotionLib()
    motion.curr_motion_keys = [key]
    motion._motion_num_frames = torch.tensor([frames])
    motion.dof_pos = torch.randn(frames, 31)
    motion.dof_vel = torch.zeros(frames, 31)
    motion.body_pos_w = torch.randn(frames, 32, 3)
    motion.body_quat_w = torch.randn(frames, 32, 4)
    motion.body_lin_vel_w = torch.zeros(frames, 32, 3)
    motion.body_ang_vel_w = torch.zeros(frames, 32, 3)
    motion.feet_l = torch.zeros(frames, 1)
    motion.feet_r = torch.zeros(frames, 1)
    q_before = motion.dof_pos.clone()
    pos_before = motion.body_pos_w.clone()
    quat_before = motion.body_quat_w.clone()
    result = apply_recorded_state_adapter(motion, path)
    assert torch.equal(q_before, motion.dof_pos)
    assert torch.equal(pos_before, motion.body_pos_w)
    assert torch.equal(quat_before, motion.body_quat_w)
    assert torch.equal(motion.dof_vel, torch.from_numpy(source[key]["dof_vel"]))
    assert motion.feet_l[:, 0].tolist() == [True, True, False, False]
    assert all(result["unchanged_fields"].values())


def test_phase11_report_passes_without_physics_or_training():
    report = json.loads(
        (REPO / "reports/retarget/x2_native_gold_motionlib_phase11.json").read_text(encoding="utf-8")
    )
    assert report["zero_update_gate"]["pass"] is True
    assert report["truth_boundary"]["official_physics_replay_executed"] is False
    assert report["truth_boundary"]["source_trace_stability_is_not_replay_stability"] is True
    assert report["decision"]["status"] == "PHASE11_MOTIONLIB_INGESTION_PASSED"
