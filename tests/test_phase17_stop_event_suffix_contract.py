import copy
from types import SimpleNamespace

import numpy as np
import pytest

from official_x2.controller_snapshot_contract import capture_controller_state
from official_x2.recovery_suffix_aggregation import canonical_sha256, physical_state_payload
from official_x2.stop_event_suffix_contract import (
    build_stop_event_sidecar,
    build_stop_event_snapshot,
    stop_event_capture_eligible,
    validate_stop_event_snapshot,
)


class Dummy:
    def __init__(self):
        self.args = SimpleNamespace(
            clock_mode="step", vx=0.3, stop_controller="brake_blend_to_policy"
        )
        self.previous_actions = {name: np.zeros(15) for name in ("main", "stationary", "recovery")}
        self.issued_actions = {name: np.zeros(15) for name in ("main", "stationary", "recovery")}
        self.sequence_step = 470
        self.control_steps = 469
        self.stop_policy_initialized = True
        self.stop_hold_targets = None
        self.stop_hold_latch_s = 1.0
        self.stop_emergency_latch = False
        self.lateral_recovery_state = "off"
        self.lateral_recovery_bias = np.zeros(2)
        self.heading_target_rad = 0.0
        self.heading_origin_xy = (0.0, 0.0)
        self.move_heading_initialized = True
        self.heading_recovery_active = False
        self.heading_action_recovery_active = False
        self.heading_action_recovery_steps = 0
        self.last_move_targets = None
        self.prepare_start_q = {}
        self.previous_physical_observation = None
        self.upper_last_target = np.zeros(14)
        self.upper_previous_target = np.zeros(14)
        self.upper_fallback_steps = 0
        self.upper_fallback_active = False
        self.upper_fallback_first_step = None
        self.actor_symmetry_projection_count = 0
        self.actor_symmetry_equivariance_rmse_sum = 0.0
        self.actor_symmetry_equivariance_abs_max = 0.0
        self.actor_symmetry_applied_delta_rmse_sum = 0.0
        self.actor_symmetry_applied_delta_abs_max = 0.0
        self.predicted_physical_observation = None
        self.predicted_physical_step = -1
        self.finished = False


def fixture_snapshot():
    physical = physical_state_payload(
        joint_names=[f"j{i}" for i in range(31)],
        joint_position_rad=np.zeros(31), joint_velocity_radps=np.zeros(31),
        root_position_m=[0, 0, 0.62], root_quaternion_xyzw=[0, 0, 0, 1],
        root_linear_velocity_world_mps=[0.1, 0, 0], root_angular_velocity_world_radps=[0, 0, 0],
    )
    controller = capture_controller_state(Dummy(), physical_state_sha256=canonical_sha256(physical))
    obs = np.zeros(93); obs[8] = -1; obs[9] = 0.1; obs[90] = 1
    return build_stop_event_snapshot(
        episode_id="e1", source_tick=470, stop_elapsed_s=0.0,
        authority_slot="brake_main", observation_policy_slot="main",
        physical_state=physical, observation=obs,
        actor_proposal_action=np.zeros(15), actual_issued_action=np.ones(15) * 0.1,
        physical_lower_target_rad=np.ones(15) * 0.02,
        default_lower_target_rad=np.zeros(15), lower_action_scale_rad=np.ones(15) * 0.2,
        command=obs[9:12], gait_phase=obs[89:93], controller_state=controller,
    )


def test_default_off_and_event_horizon_are_fail_closed():
    assert not stop_event_capture_eligible(output_enabled=False, stage="stop", stop_elapsed_s=0, horizon_s=4)
    assert stop_event_capture_eligible(output_enabled=True, stage="stop", stop_elapsed_s=4, horizon_s=4)
    assert not stop_event_capture_eligible(output_enabled=True, stage="move", stop_elapsed_s=0, horizon_s=4)
    with pytest.raises(ValueError):
        stop_event_capture_eligible(output_enabled=True, stage="stop", stop_elapsed_s=0, horizon_s=4.1)


def test_snapshot_distinguishes_actor_proposal_from_actual_physical_issue():
    snapshot = fixture_snapshot()
    assert snapshot["actor_proposal_action"] != snapshot["actual_issued_action"]
    assert snapshot["actual_previous_action_input"] == snapshot["observation_93d"][74:89]
    validate_stop_event_snapshot(snapshot)
    broken = copy.deepcopy(snapshot); broken["actual_previous_action_input"][0] = 1
    with pytest.raises(ValueError): validate_stop_event_snapshot(broken)


def test_sidecar_binds_outcome_contract_and_rows():
    snapshot = fixture_snapshot()
    sidecar = build_stop_event_sidecar(
        [snapshot], source_trace_path="/results/e1.json", source_trace_sha256="a" * 64,
        adapter_path="/repo/adapter.py", adapter_sha256="b" * 64,
        model_assets={"moving": {"path": "/models/m.onnx", "sha256": "c" * 64}},
        episode_outcome={"stop_gate_pass": True, "full_gate_pass": True, "survived_stop_height_gate": True},
        horizon_s=4.0, frozen_control_contract={"stop_controller": "brake_blend_to_policy"},
    )
    assert sidecar["row_count"] == 1
    assert sidecar["episode_outcome"]["stop_gate_pass"] is True
