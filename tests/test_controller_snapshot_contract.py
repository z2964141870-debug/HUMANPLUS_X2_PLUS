from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from official_x2.controller_snapshot_contract import (
    audit_trace_controller_state,
    capture_controller_state,
    restore_controller_state,
    validate_controller_state,
)


PHYSICAL_SHA = "a" * 64


def _controller():
    return SimpleNamespace(
        args=SimpleNamespace(clock_mode="step", vx=0.3, stop_controller="brake_blend_to_policy"),
        previous_actions={slot: np.arange(15, dtype=np.float32) / 20 for slot in ("main", "stationary", "recovery")},
        issued_actions={slot: -np.arange(15, dtype=np.float32) / 20 for slot in ("main", "stationary", "recovery")},
        sequence_step=231,
        control_steps=200,
        stop_hold_latch_s=0.86,
        stop_emergency_latch=False,
        heading_target_rad=0.2,
        heading_origin_xy=(1.0, -2.0),
        move_heading_initialized=True,
        stop_policy_initialized=True,
        lateral_recovery_state="left",
        lateral_recovery_bias=np.asarray([0.04, -0.02], dtype=np.float32),
        heading_recovery_active=False,
        heading_action_recovery_active=True,
        heading_action_recovery_steps=7,
        last_move_targets={"left_hip_pitch_joint": 0.1},
        stop_hold_targets=None,
        prepare_start_q={"left_hip_pitch_joint": 0.0},
        predicted_physical_step=231,
        previous_physical_observation=np.arange(71, dtype=np.float32),
        predicted_physical_observation=np.arange(71, dtype=np.float32) + 0.1,
        upper_previous_target=np.arange(14, dtype=np.float32) / 100,
        upper_last_target=np.arange(14, dtype=np.float32) / 90,
        upper_fallback_steps=2,
        upper_fallback_active=False,
        upper_fallback_first_step=None,
        finished=False,
    )


def test_controller_snapshot_roundtrip_is_lossless_and_copy_isolated():
    source = _controller()
    snapshot = capture_controller_state(source, physical_state_sha256=PHYSICAL_SHA)
    target = _controller()
    target.sequence_step = 0
    target.previous_actions["main"].fill(0.0)
    target.lateral_recovery_bias.fill(0.0)
    restore_controller_state(target, snapshot, expected_physical_state_sha256=PHYSICAL_SHA)
    restored = capture_controller_state(target, physical_state_sha256=PHYSICAL_SHA)
    assert restored == snapshot
    target.previous_actions["main"][0] = 99.0
    assert snapshot["previous_actions"]["main"][0] != 99.0


def test_restoring_identical_controller_snapshot_is_exact_noop():
    controller = _controller()
    before = capture_controller_state(controller, physical_state_sha256=PHYSICAL_SHA)
    restore_controller_state(
        controller,
        before,
        expected_physical_state_sha256=PHYSICAL_SHA,
    )
    after = capture_controller_state(controller, physical_state_sha256=PHYSICAL_SHA)
    assert after == before


def test_controller_snapshot_fails_closed_on_physical_or_args_mismatch():
    source = _controller()
    snapshot = capture_controller_state(source, physical_state_sha256=PHYSICAL_SHA)
    with pytest.raises(ValueError, match="physical snapshot hash"):
        restore_controller_state(source, snapshot, expected_physical_state_sha256="b" * 64)
    changed = _controller()
    changed.args.vx = 0.4
    with pytest.raises(ValueError, match="arguments"):
        restore_controller_state(changed, snapshot, expected_physical_state_sha256=PHYSICAL_SHA)


def test_controller_snapshot_rejects_wall_clock_and_bad_shapes():
    source = _controller()
    source.args.clock_mode = "wall"
    with pytest.raises(ValueError, match="clock_mode='step'"):
        capture_controller_state(source, physical_state_sha256=PHYSICAL_SHA)
    source.args.clock_mode = "step"
    snapshot = capture_controller_state(source, physical_state_sha256=PHYSICAL_SHA)
    broken = deepcopy(snapshot)
    broken["issued_actions"]["main"] = [0.0] * 14
    with pytest.raises(ValueError, match="issued_actions.main"):
        validate_controller_state(broken)


def test_existing_stage326_trace_is_explicitly_not_suffix_ready():
    trace = Path(
        "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
        "official_native_strict_20260807/"
        "stage326_s2652_preview0p5_split_intent_brake_stiff1p2_fixed_r1.json"
    )
    audit = audit_trace_controller_state(trace)
    assert audit["row_count"] == 710
    assert audit["valid_controller_state_rows"] == 0
    assert audit["missing_controller_state_rows"] == 710
    assert audit["suffix_replay_ready"] is False
