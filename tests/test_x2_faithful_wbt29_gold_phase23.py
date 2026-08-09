from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from x2_faithful_any2any_phase23 import (
    GoldSplitSpec,
    ImmutableGoldMotionLibHook,
    POLICY_TERM_ORDER,
    WBT29PolicyContract,
    _sha256,
)


def contract() -> WBT29PolicyContract:
    official = tuple(f"joint_{index}" for index in range(31))
    head = ("joint_29", "joint_30")
    target = tuple(f"joint_{index}" for index in range(29))
    source = target[::2] + target[1::2]
    return WBT29PolicyContract.build(official, target, source, head)


def test_wbt29_action_observation_and_history_are_source_semantic():
    item = contract()
    q31 = torch.arange(2 * 10 * 31).reshape(2, 10, 31)
    source = item.official_to_source(q31)
    assert source.shape == (2, 10, 29)
    assert torch.equal(item.source_to_target(source), q31[..., :29])

    nominal = torch.ones_like(q31) * -7
    scattered = item.source_action_to_official(source, nominal)
    assert torch.equal(scattered[..., :29], q31[..., :29])
    assert torch.equal(scattered[..., 29:], nominal[..., 29:])

    terms = OrderedDict(
        (
            ("gravity_dir", torch.zeros(2, 10, 3)),
            ("base_ang_vel", torch.zeros(2, 10, 3)),
            ("joint_pos", q31),
            ("joint_vel", q31 + 1),
            ("actions", torch.zeros(2, 10, 29)),
        )
    )
    adapted = item.adapt_policy_history(terms)
    assert tuple(adapted) == POLICY_TERM_ORDER
    assert torch.equal(adapted["joint_pos"], source)
    assert item.flatten_policy_history(adapted).shape == (2, 930)


def test_policy_contract_rejects_head_or_wrong_term_order():
    item = contract()
    terms = OrderedDict(
        (
            ("base_ang_vel", torch.zeros(1, 10, 3)),
            ("gravity_dir", torch.zeros(1, 10, 3)),
            ("joint_pos", torch.zeros(1, 10, 31)),
            ("joint_vel", torch.zeros(1, 10, 31)),
            ("actions", torch.zeros(1, 10, 29)),
        )
    )
    with pytest.raises(ValueError, match="ordered exactly"):
        item.adapt_policy_history(terms)
    with pytest.raises(ValueError, match="expected source action 29"):
        item.source_action_to_official(torch.zeros(1, 31), torch.zeros(1, 31))


class FakeMotionLib:
    pass


def write_split(path: Path, split: str, prefix: str) -> None:
    frames = 3
    entry = {
        "split": split,
        "dof": np.zeros((frames, 31), np.float32),
        "dof_vel": np.ones((frames, 31), np.float32),
        "root_lin_vel_w_mps": np.ones((frames, 3), np.float32) * 2,
        "root_ang_vel": np.ones((frames, 3), np.float32) * 3,
        "model_contact": {
            "left": np.array([True, False, True]),
            "right": np.array([True, True, False]),
        },
    }
    joblib.dump({f"{prefix}000": entry}, path)


def test_immutable_gold_hook_restores_state_and_enforces_hash_and_split(tmp_path):
    path = tmp_path / "train.pkl"
    write_split(path, "train", "train_")
    spec = GoldSplitSpec("train", path, _sha256(path), "train_", True)
    hook = ImmutableGoldMotionLibHook(spec)
    motion = FakeMotionLib()
    motion.curr_motion_keys = ["train_000"]
    motion._motion_num_frames = torch.tensor([3])
    motion.dof_pos = torch.randn(3, 31)
    motion.dof_vel = torch.zeros(3, 31)
    motion.body_pos_w = torch.randn(3, 32, 3)
    motion.body_quat_w = torch.randn(3, 32, 4)
    motion.body_lin_vel_w = torch.zeros(3, 32, 3)
    motion.body_ang_vel_w = torch.zeros(3, 32, 3)
    motion.feet_l = torch.zeros(3, 1, dtype=torch.bool)
    motion.feet_r = torch.zeros(3, 1, dtype=torch.bool)
    q = motion.dof_pos.clone()
    result = hook.attach(motion)
    assert result["optimizer_eligible"] is True
    assert torch.equal(q, motion.dof_pos)
    assert torch.all(motion.dof_vel == 1)
    assert motion.feet_l[:, 0].tolist() == [True, False, True]

    bad = ImmutableGoldMotionLibHook(
        GoldSplitSpec("held_out", path, _sha256(path), "held_", False)
    )
    with pytest.raises(ValueError, match="unexpected sampler keys"):
        bad.validate_source()
    changed = joblib.load(path)
    changed["train_000"]["dof"][0, 0] = 1
    joblib.dump(changed, path)
    with pytest.raises(ValueError, match="hash differs"):
        hook.validate_source()


def test_phase23_report_declares_cpu_only_and_split_isolation():
    import json

    report = json.loads(
        (REPO / "reports/retarget/x2_faithful_wbt29_gold_phase23.json").read_text()
    )
    assert report["decision"]["status"] == "B1_B2_IMPLEMENTATION_READY_CPU_PROBED"
    assert report["truth_boundary"]["isaac_physics_training_optimizer_network_mutation"] is False
    assert report["truth_boundary"]["live_train_eval_entrypoint_wired"] is False
    assert report["contract"]["policy_action_dim"] == 29
    assert report["gold_splits"]["held_out"]["hook"]["optimizer_eligible"] is False
    assert report["split_isolation"]["pass"] is True
