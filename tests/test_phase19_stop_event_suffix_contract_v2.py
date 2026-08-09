import copy
from pathlib import Path

import numpy as np
import pytest

from official_x2.stop_event_suffix_contract_v2 import (
    ISAAC_JOINTS,
    SELF_CONSISTENCY_ATOL,
    finite_gate_outcome,
    reconstruct_actor_observation,
    validate_actor_observation_state,
)


def _actor_state(*, moving=False, elapsed=0.0, offset=0.0, command_vx=0.0):
    q = np.zeros(31, dtype=np.float32)
    dq = np.zeros(31, dtype=np.float32)
    phase_elapsed = elapsed + (offset if moving or abs(command_vx) > 0.1 else 0.0)
    if moving:
        phase = (phase_elapsed / 0.8) % 1.0
        half_ds = 0.2 / 4.0
        right_swing = half_ds <= phase < 0.5 - half_ds
        left_swing = 0.5 + half_ds <= phase < 1.0 - half_ds
        gait = np.asarray(
            [
                np.sin(2 * np.pi * phase),
                np.cos(2 * np.pi * phase),
                not left_swing,
                not right_swing,
            ],
            dtype=np.float32,
        )
    else:
        gait = np.asarray([0, 0, 1, 1], dtype=np.float32)
    state = {
        "schema": "aimdk_x2_actor_observation_source_v2",
        "policy_slot": "recovery",
        "sequence_step": 500,
        "torso_imu_source": {
            "topic": "/aima/hal/imu/torso/state",
            "frame_id": "torso_imu",
            "sample_time_s": 10.0,
            "orientation_xyzw": [0, 0, 0, 1],
            "angular_velocity_radps": [0.01, -0.02, 0.03],
            "linear_acceleration_mps2": [0, 0, 9.81],
            "orientation_covariance": [0] * 9,
            "angular_velocity_covariance": [0] * 9,
            "linear_acceleration_covariance": [0] * 9,
            "filter_contract": "vendor torso IMU topic; no adapter-side filtering",
        },
        "odom_source": {
            "topic": "/aima/hal/odom/state",
            "sample_time_s": 10.0,
            "root_quaternion_xyzw": [0, 0, 0, 1],
            "root_linear_velocity_world_mps": [0.1, -0.2, 0.0],
        },
        "joint_source": {
            "joint_names": list(ISAAC_JOINTS),
            "position_rad": q.tolist(),
            "velocity_radps": dq.tolist(),
            "default_position_rad": q.tolist(),
        },
        "predictor_state": {
            "prediction_seconds": 0.0,
            "sequence_step": 500,
            "raw_physical_observation_71d": np.concatenate(
                (
                    np.asarray([0.1, -0.2, 0.0, 0.01, -0.02, 0.03, 0, 0, -1], dtype=np.float32),
                    np.zeros(62, dtype=np.float32),
                )
            ).tolist(),
            "previous_physical_before": None,
            "predicted_physical_before": None,
            "predicted_step_before": -1,
            "previous_physical_after": None,
            "predicted_physical_after": None,
            "predicted_step_after": -1,
        },
        "gait_generator_state": {
            "phase_elapsed_input_s": elapsed,
            "actor_phase_elapsed_s": phase_elapsed,
            "phase_offset_s": offset,
            "period_s": 0.8,
            "double_support_fraction": 0.2,
            "moving": moving,
            "force_moving": moving,
            "command_vx_before_mirror": command_vx,
            "gait_output": gait.tolist(),
            "latch_state": {"stop_hold_latch_s": 1.0, "stop_emergency_latch": False},
        },
        "command_velocity_mps_radps": [command_vx, 0, 0],
        "previous_action_pre_inference": np.arange(15, dtype=np.float32).tolist(),
        "mirror_policy": False,
        "actor_observation_93d": [],
        "model_input": [],
        "model_input_width": 93,
        "capture_boundary": "pre_actor_inference",
    }
    obs = reconstruct_actor_observation(state)
    state["actor_observation_93d"] = obs.tolist()
    state["model_input"] = obs.tolist()
    return state


def test_v2_sources_reconstruct_exact_actor_input_and_model_prefix():
    state = _actor_state()
    metrics = validate_actor_observation_state(state)
    assert metrics["reconstruction_abs_max"] <= SELF_CONSISTENCY_ATOL
    assert metrics["model_prefix_abs_max"] == 0.0


def test_v2_rejects_root_substitution_for_torso_imu_and_gait_tampering():
    state = _actor_state()
    broken = copy.deepcopy(state)
    broken["torso_imu_source"]["angular_velocity_radps"][0] += 0.1
    with pytest.raises(ValueError, match="reconstruct"):
        validate_actor_observation_state(broken)
    broken = copy.deepcopy(state)
    broken["gait_generator_state"]["gait_output"][2] = 0.0
    with pytest.raises(ValueError, match="gait output mismatch"):
        validate_actor_observation_state(broken)


def test_exact_floating_phase_boundary_is_preserved_not_integer_rebuilt():
    # At the right-swing boundary phase=0.05, exact float comparison changes
    # contact.  V2 retains that clock rather than reconstructing it from a
    # rounded episode index.
    state = _actor_state(moving=True, elapsed=0.04, command_vx=0.3)
    validate_actor_observation_state(state)
    after = _actor_state(moving=True, elapsed=0.040001, command_vx=0.3)
    validate_actor_observation_state(after)
    assert state["actor_observation_93d"][92] == 1.0
    assert after["actor_observation_93d"][92] == 0.0
    rounded = copy.deepcopy(state)
    rounded["gait_generator_state"]["phase_elapsed_input_s"] = 0.040001
    rounded["gait_generator_state"]["actor_phase_elapsed_s"] = 0.040001
    with pytest.raises(ValueError, match="gait output mismatch"):
        validate_actor_observation_state(rounded)


def test_adapter_v2_channel_is_explicit_default_off():
    source = Path("tools/official_x2/stage208_official_mujoco_adapter.py").read_text()
    assert '"--stop-event-snapshot-v2-output"' in source
    method = source[source.index("def _maybe_record_stop_event_suffix_snapshot_v2"):]
    assert "output_enabled=self.args.stop_event_snapshot_v2_output is not None" in method


def test_v2_hash_outcome_ignores_nonfinite_disabled_thresholds():
    projected = finite_gate_outcome(
        {"full_gate_pass": True, "stop_gate_pass": False, "disabled_heading": float("inf")}
    )
    assert projected["full_gate_pass"] is True
    assert projected["stop_gate_pass"] is False
    assert "disabled_heading" not in projected
