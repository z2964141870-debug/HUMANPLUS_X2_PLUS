import copy

import numpy as np

from official_x2.audit_phase17_stop_event_suffix import (
    classify_outcome,
    query_groups,
    stop_event_dedup_key,
)
from official_x2.validate_phase17_stop_event_episode import classify_summary


def _row():
    obs = np.zeros(93, dtype=float)
    obs[8] = -1.0
    return {
        "authority_slot": "stationary",
        "stop_elapsed_s": 0.0,
        "observation_93d": obs.tolist(),
        "actual_issued_action": np.zeros(15).tolist(),
        "physical_state": {
            "root_position_m": [0.0, 0.0, 0.62],
            "root_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "root_linear_velocity_world_mps": [0.1, 0.0, 0.0],
        },
    }


def test_complete_success_requires_full_gate():
    assert classify_outcome({"full_gate_pass": True, "stop_gate_pass": True}) == "success"
    assert classify_outcome({"full_gate_pass": False, "stop_gate_pass": True, "survived_stop_height_gate": True}) == "critical"
    assert classify_outcome({"full_gate_pass": False, "stop_gate_pass": False, "survived_stop_height_gate": False}) == "failure"
    assert classify_summary({"full_gate_pass": True, "stop_gate_pass": True}) == "success"
    assert classify_summary({"full_gate_pass": False, "stop_gate_pass": True, "survived_stop_height_gate": True}) == "critical"


def test_query_uses_same_93d_group_and_signed_root_convention():
    groups, meta = query_groups([_row()], ["r1:success"])
    assert groups["previous_action"].shape == (1, 15)
    assert groups["projected_gravity"].shape == (1, 3)
    np.testing.assert_allclose(groups["root_state"][0], [0.62, 0.0, 0.0, 0.1])
    assert meta[0]["episode"] == "r1:success"


def test_dedup_key_is_quantized_and_authority_aware():
    row = _row()
    close = copy.deepcopy(row)
    close["physical_state"]["root_position_m"][2] += 0.001
    assert stop_event_dedup_key(row) == stop_event_dedup_key(close)
    other_slot = copy.deepcopy(row)
    other_slot["authority_slot"] = "brake_main"
    assert stop_event_dedup_key(row) != stop_event_dedup_key(other_slot)
