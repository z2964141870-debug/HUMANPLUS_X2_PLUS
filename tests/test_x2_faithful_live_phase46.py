from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch

from x2_faithful_live_phase46 import (
    EXACT_S7_ACTOR,
    EXACT_S7_CRITIC,
    HEAD2,
    SOURCE_CANONICAL,
    SOURCE29,
    command_multi_future_source29,
    validate_split,
)


REPO = Path(__file__).resolve().parents[1]


class _Robot:
    joint_names = list(SOURCE29) + list(HEAD2)


class _Command:
    num_envs = 2
    num_future_frames = 10
    robot = _Robot()
    joint_pos_multi_future = torch.zeros(2, 310)
    joint_vel_multi_future = torch.zeros(2, 310)


class _Manager:
    @staticmethod
    def get_term(name):
        assert name == "motion"
        return _Command()


class _Env:
    command_manager = _Manager()


def test_wbt29_partition_and_exact_s7_scope():
    assert len(SOURCE29) == 29 and len(set(SOURCE29)) == 29
    assert set(SOURCE29).isdisjoint(HEAD2)
    assert EXACT_S7_CRITIC == tuple(
        f"critic_module.module.{index}" for index in (2, 4, 6, 8, 10)
    )
    assert "critic_module.module.0" not in EXACT_S7_CRITIC
    assert "critic_module.module.12" not in EXACT_S7_CRITIC
    assert len(EXACT_S7_ACTOR) == 7


def test_reference_adapter_is_29d_and_preserves_sonic_flatten_layout(monkeypatch):
    for index, name in enumerate(SOURCE29):
        monkeypatch.setitem(SOURCE_CANONICAL, name, (0.0, float(index), 1.0))
    flat = command_multi_future_source29(_Env(), "motion", False)
    nonflat = command_multi_future_source29(_Env(), "motion", True)
    assert flat.shape == (2, 580)
    assert nonflat.shape == (2, 10, 58)
    assert torch.equal(flat, nonflat.reshape(2, 580))
    assert torch.equal(flat[:, :29], torch.arange(29).expand(2, 29))


def test_bronze_source_hash_keys_and_split_are_frozen():
    source = REPO / "artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl"
    payload = joblib.load(source)

    class Motion:
        curr_motion_keys = list(payload)

    report = validate_split(
        Motion(),
        source_path=source,
        expected_sha256="06b759e7926bd841537cf515d7cc46793d47503a9fab5da95c649a2f0b6ec4a6",
        expected_keys=["AMASS-STAND-001", "AMASS-UPPER-001", "PHUMA-LUNGE-R-001"],
        expected_split="train",
        recorded_state_adapter=False,
    )
    assert report["sampler_keys"] == list(payload)


def test_stage152_opt_in_is_fail_closed_and_default_compatible():
    launcher = (REPO / "scripts/run_dcpeft_stage152.sh").read_text()
    phase46 = (REPO / "scripts/run_x2_faithful_live_zero_phase46.sh").read_text()
    action_boundary = (REPO / "src/x2_faithful_live_actions_phase46.py").read_text()
    assert 'FAITHFUL_WBT29="${FAITHFUL_WBT29:-false}"' in launcher
    assert "x2_faithful_live_phase46.Phase46LiveZeroTrainer" in launcher
    assert "x2_faithful_live_actions_phase46.WBT29JointPositionActionCfg" in launcher
    assert "x2_faithful_live_actions_phase46.FaithfulWBT29TrackingEnvCfg" in launcher
    assert "WBT14_SOURCE_BODIES=" in launcher
    assert 'TRACKED_BODY_NAMES="${WBT14_SOURCE_BODIES}"' in launcher
    assert 'term.params = {"joint_names": [], "action_name": "joint_pos"}' in action_boundary
    assert 'CRITIC_LORA_PREFIXES="${CRITIC_LORA_PREFIXES:-[critic_module]}"' in launcher
    assert "+algo.config.any2any_lora.critic_lora_prefixes=${CRITIC_LORA_PREFIXES}" in launcher
    assert "critic_module.module.0" not in phase46
    assert "critic_module.module.12" not in phase46
    assert "ITERS=0" in phase46
    assert "TRAIN_BOUNDARY_KEYS='[]'" in phase46
    assert "PHASE46_SPLIT" in phase46


def test_live_zero_summary_records_no_optimizer_or_control_step():
    summary = json.loads((REPO / "reports/x2_wbt_faithful_live_zero_phase46.json").read_text())
    assert summary["decision"] == "PASS_LIVE_ZERO_UPDATE_ONLY"
    assert summary["optimizer_steps"] == 0
    assert summary["environment_control_steps"] == 0
    assert summary["runtime_contract"]["policy_action"] == 29
    assert summary["runtime_contract"]["sim_articulation"] == 31
    assert summary["runtime_contract"]["critic_s7_layers"] == [2, 4, 6, 8, 10]
    assert summary["equivalence"]["action_B0_max_abs"] == 0.0
    assert summary["equivalence"]["value_B0_max_abs"] == 0.0
