from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.x2_native_locomotion_posture_phase60 import (
    actual_support_com_outside_distance,
    signed_backward_pitch_penalty,
    signed_root_pitch_rad,
)


class _Scene(dict):
    pass


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_command(self, _name):
        return self.command


class _PhysxView:
    def __init__(self, masses):
        self.masses = masses

    def get_masses(self):
        return self.masses


class _Robot:
    def __init__(self):
        self.data = type("Data", (), {})()
        self.data.projected_gravity_b = torch.tensor(
            [[torch.sin(torch.tensor(-0.20)), 0.0, -1.0], [torch.sin(torch.tensor(0.10)), 0.0, -1.0]]
        )
        self.data.body_com_pos_w = torch.tensor(
            [[[0.30, 0.0, 0.7]], [[0.00, 0.0, 0.7]]]
        )
        self.data.body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 0.068], [0.0, -0.20, 0.068]],
                [[0.0, 0.10, 0.068], [0.0, -0.10, 0.068]],
            ]
        )
        self.data.body_quat_w = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]] * 2, [[1.0, 0.0, 0.0, 0.0]] * 2]
        )
        self.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
        self.root_physx_view = _PhysxView(torch.ones(2, 1))

    def find_bodies(self, _names, preserve_order):
        assert preserve_order
        return [0, 1], _names


def _env():
    robot = _Robot()
    sensor = lambda z: type("Sensor", (), {"data": type("Data", (), {"force_matrix_w": z})()})()
    left_force = torch.tensor([[[[0.0, 0.0, 50.0]]], [[[0.0, 0.0, 50.0]]]])
    right_force = torch.tensor([[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 50.0]]]])
    return type(
        "Env",
        (),
        {
            "scene": _Scene(
                robot=robot,
                left_foot_ground_contact=sensor(left_force),
                right_foot_ground_contact=sensor(right_force),
            ),
            "command_manager": _CommandManager(torch.tensor([[0.35, 0.0, 0.0], [0.35, 0.0, 0.0]])),
        },
    )()


def test_signed_pitch_is_directional_and_does_not_force_upright():
    env = _env()
    pitch = signed_root_pitch_rad(env.scene["robot"])
    assert pitch.tolist() == pytest.approx([-0.20, 0.10], abs=1.0e-6)
    penalty = signed_backward_pitch_penalty(
        env, "base_velocity", tolerance_rad=0.05, normalization_rad=0.15
    )
    assert penalty.tolist() == pytest.approx([1.0, 0.0], abs=1.0e-6)


def test_actual_contact_support_distinguishes_single_outside_and_double_inside():
    outside, count = actual_support_com_outside_distance(env=_env(), force_threshold_n=10.0)
    assert count.tolist() == [1, 2]
    assert outside[0] == pytest.approx(0.156, abs=1.0e-6)
    assert outside[1] == pytest.approx(0.0, abs=1.0e-6)


def test_phase60_prereg_has_exactly_three_groups_and_fail_closed_limits():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "reports/retarget/x2_native_posture_phase60_prereg.json").read_text())
    assert list(cfg["groups"]) == ["A", "B", "C"]
    assert cfg["groups"]["A"]["role"].endswith("no optimizer")
    assert cfg["groups"]["B"]["actual_support_com_weight"] == 0.0
    assert cfg["groups"]["C"]["actual_support_com_weight"] < 0.0
    assert cfg["reward_contract"]["waist_pitch_direct_constraint"] is False
    assert cfg["training"]["optimizer_steps_per_group"] == 20
    assert cfg["evaluation"]["seeds"] == [40, 41, 42]
    assert "does not satisfy Task2" in cfg["stop"]


def test_phase60_runner_is_fresh_lora_and_fixed_upper_only():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    assert 'POSTURE_VARIANT not in {"A", "B", "C"}' in runner
    assert '"CWI_UPPER_ZERO_FRACTION": "1.0" if PEFT_PHASE60' in runner
    assert 'POSTURE_REWARD_WEIGHTS[POSTURE_VARIANT]' in runner
    assert 'Phase60 group A is frozen evaluation-only' in runner
    assert 'args.checkpoint.resolve() != ORIGINAL.resolve()' in runner
