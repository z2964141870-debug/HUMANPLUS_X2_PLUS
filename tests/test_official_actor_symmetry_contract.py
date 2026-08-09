import numpy as np
import pytest

from official_x2.actor_symmetry_contract import (
    mirror_named_vector,
    mirror_stage208_observation,
    project_actor_action,
)


ISAAC = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)
LOWER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)


def test_named_and_full_observation_reflection_are_involutions():
    rng = np.random.default_rng(7)
    q = rng.normal(size=len(LOWER)).astype(np.float32)
    assert np.array_equal(mirror_named_vector(mirror_named_vector(q, LOWER), LOWER), q)

    observation = rng.normal(size=93).astype(np.float32)
    mirrored = mirror_stage208_observation(
        observation,
        isaac_joint_names=ISAAC,
        action_joint_names=LOWER,
    )
    restored = mirror_stage208_observation(
        mirrored,
        isaac_joint_names=ISAAC,
        action_joint_names=LOWER,
    )
    assert np.array_equal(restored, observation)
    assert np.array_equal(mirrored[91:93], observation[[92, 91]])
    assert np.array_equal(mirrored[89:91], -observation[89:91])


def test_projection_is_noop_at_zero_and_equivariant_average_at_one():
    rng = np.random.default_rng(9)
    raw = rng.normal(size=len(LOWER)).astype(np.float32)
    mirrored_prediction = rng.normal(size=len(LOWER)).astype(np.float32)
    no_op, no_op_metrics = project_actor_action(
        raw,
        mirrored_prediction,
        action_joint_names=LOWER,
        alpha=0.0,
        mask_mode="roll_yaw",
    )
    assert np.array_equal(no_op, raw)
    assert no_op_metrics["applied_delta_abs_max"] == 0.0

    projected, _ = project_actor_action(
        raw,
        mirrored_prediction,
        action_joint_names=LOWER,
        alpha=1.0,
        mask_mode="all",
    )
    mirrored_back = mirror_named_vector(mirrored_prediction, LOWER)
    assert np.allclose(projected, 0.5 * (raw + mirrored_back), atol=1e-7, rtol=0.0)


def test_roll_yaw_mask_does_not_modify_pitch_or_knee_components():
    raw = np.arange(len(LOWER), dtype=np.float32)
    mirrored_prediction = -raw
    projected, _ = project_actor_action(
        raw,
        mirrored_prediction,
        action_joint_names=LOWER,
        alpha=1.0,
        mask_mode="roll_yaw",
    )
    for index, name in enumerate(LOWER):
        if "_roll_" not in name and "_yaw_" not in name:
            assert projected[index] == raw[index]


def test_contract_rejects_bad_shapes_and_alpha():
    with pytest.raises(ValueError):
        mirror_stage208_observation(
            np.zeros(92, dtype=np.float32),
            isaac_joint_names=ISAAC,
            action_joint_names=LOWER,
        )
    with pytest.raises(ValueError):
        project_actor_action(
            np.zeros(15), np.zeros(15), action_joint_names=LOWER,
            alpha=1.1, mask_mode="all",
        )
