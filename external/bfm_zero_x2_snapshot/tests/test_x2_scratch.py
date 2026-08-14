from __future__ import annotations

import numpy as np
import pytest

from humanoidverse.x2_scratch import (
    ACTION_DIM,
    COMMAND_DIM,
    GAIT_PHASE_DIM,
    STATE_DIM,
    X2_AUX_REWARD_SCALES,
    X2_DEFAULT_JOINT_POS_31,
    X2_ISAAC_JOINTS_31,
    X2_JOINTS_31,
    X2_SCRATCH_ACTION_SCALE_15,
    X2ScratchDataConfig,
    angular_velocity_from_xyzw,
    build_x2_scratch_replay_buffers,
    convert_x2_motion_entry,
    convert_stage219_critic_rollout,
    deployable_gait_phase_features,
    normalized_lower_physical_targets,
    projected_gravity_from_xyzw,
    reconstruct_stage219_direct_action_labels,
    root_quaternion_wxyz_from_projected_gravity_yaw,
    rotate_body_to_world_wxyz,
    rotate_world_to_body_xyzw,
    stack_causal_history,
)
from humanoidverse.x2_scratch_model import (
    parameter_summary,
    x2_scratch_model_config,
    x2_scratch_agent_config,
    x2_scratch_observation_space,
)


def synthetic_entry(frames: int = 16) -> dict:
    dof = np.repeat(X2_DEFAULT_JOINT_POS_31[None], frames, axis=0)
    action = dof.copy()
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    root_trans_offset = np.zeros((frames, 3), dtype=np.float32)
    return {
        "dof": dof,
        "action": action,
        "root_rot": root_rot,
        "root_trans_offset": root_trans_offset,
        "fps": 50,
        "joint_names_mujoco": list(X2_JOINTS_31),
    }


def test_identity_quaternion_contract() -> None:
    quaternion = synthetic_entry()["root_rot"]
    np.testing.assert_array_equal(
        projected_gravity_from_xyzw(quaternion),
        np.tile(np.asarray((0.0, 0.0, -1.0), dtype=np.float32), (len(quaternion), 1)),
    )
    np.testing.assert_array_equal(
        angular_velocity_from_xyzw(quaternion, 0.02),
        np.zeros((len(quaternion), 3), dtype=np.float32),
    )


def test_convert_static_motion_shapes_and_causality() -> None:
    config = X2ScratchDataConfig(history_length=4)
    episode, audit = convert_x2_motion_entry(synthetic_entry(), motion_id=7, config=config)
    observation = episode["observation"]
    assert observation["state"].shape == (16, STATE_DIM)
    assert observation["privileged_state"].shape == (16, STATE_DIM)
    assert observation["last_action"].shape == (16, ACTION_DIM)
    assert observation["history_actor"].shape == (16, config.history_actor_dim)
    assert episode["action"].shape == (16, ACTION_DIM)
    assert episode["truncated"].sum() == 1
    assert bool(episode["truncated"][-1])
    assert not episode["terminated"].any()
    assert np.all(episode["motion_id"] == 7)
    assert audit["action_clip_fraction"] == 0.0
    assert audit["finite"] is True


def test_history_is_zero_left_padded_and_causal() -> None:
    state = np.arange(12, dtype=np.float32).reshape(4, 3)
    action = np.arange(8, dtype=np.float32).reshape(4, 2)
    history = stack_causal_history(state, action, length=3).reshape(4, 3, 5)
    np.testing.assert_array_equal(history[0, :2], 0.0)
    np.testing.assert_array_equal(history[0, 2], np.concatenate((state[0], action[0])))
    np.testing.assert_array_equal(history[3, 0], np.concatenate((state[1], action[1])))
    np.testing.assert_array_equal(history[3, 2], np.concatenate((state[3], action[3])))


def test_wrong_joint_order_fails_closed() -> None:
    entry = synthetic_entry()
    entry["joint_names_mujoco"][0], entry["joint_names_mujoco"][1] = (
        entry["joint_names_mujoco"][1],
        entry["joint_names_mujoco"][0],
    )
    with pytest.raises(ValueError, match="joint order"):
        convert_x2_motion_entry(entry, motion_id=0)


def test_physical_lower_target_uses_direct_scratch_contract() -> None:
    target = np.repeat(X2_DEFAULT_JOINT_POS_31[None, :ACTION_DIM], 2, axis=0)
    target[1, 0] += 0.5
    action = normalized_lower_physical_targets(target)
    np.testing.assert_array_equal(action[0], 0.0)
    assert action[1, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="15 columns"):
        normalized_lower_physical_targets(target[:, :-1])


def test_reconstruct_root_orientation_and_body_vector() -> None:
    gravity = np.asarray(
        ((0.0, 0.0, -1.0), (-0.20829715, 0.05412683, -0.97656673)),
        dtype=np.float32,
    )
    yaw = np.asarray((0.0, -0.09307115), dtype=np.float32)
    quaternion = root_quaternion_wxyz_from_projected_gravity_yaw(gravity, yaw)
    np.testing.assert_allclose(quaternion[0], (1.0, 0.0, 0.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1.0e-7)
    vector = np.asarray(((1.0, 2.0, 3.0), (0.3, -0.03, -0.08)), dtype=np.float32)
    world = rotate_body_to_world_wxyz(vector, quaternion)
    np.testing.assert_allclose(world[0], vector[0], atol=1.0e-7)
    np.testing.assert_allclose(np.linalg.norm(world, axis=1), np.linalg.norm(vector, axis=1), atol=1.0e-6)


def test_world_linear_velocity_is_preserved_in_identity_body_frame() -> None:
    entry = synthetic_entry()
    entry["root_trans_offset"][:, 0] = np.arange(16, dtype=np.float32) * 0.007
    episode, audit = convert_x2_motion_entry(entry, motion_id=0)
    expected = np.tile(np.asarray((0.35, 0.0, 0.0), dtype=np.float32), (16, 1))
    np.testing.assert_allclose(
        episode["observation"]["state"][:, -3:], expected, atol=1.0e-6
    )
    np.testing.assert_allclose(
        audit["base_linear_velocity_mean"], expected[0], atol=1.0e-6
    )
    np.testing.assert_allclose(
        rotate_world_to_body_xyzw(expected, entry["root_rot"]), expected
    )


def test_excessive_action_clipping_fails_closed() -> None:
    entry = synthetic_entry()
    entry["action"][:, 4] += 2.0
    with pytest.raises(ValueError, match="clip fraction"):
        convert_x2_motion_entry(entry, motion_id=0)


def test_fps_mismatch_fails_closed() -> None:
    entry = synthetic_entry()
    entry["fps"] = 30
    with pytest.raises(ValueError, match="does not match"):
        convert_x2_motion_entry(entry, motion_id=0)


def test_reduced_model_config_contract() -> None:
    observation_space = x2_scratch_observation_space(X2ScratchDataConfig())
    assert observation_space["state"].shape == (STATE_DIM,)
    assert observation_space["last_action"].shape == (ACTION_DIM,)
    config = x2_scratch_model_config(device="cpu", hidden_dim=128, hidden_layers=2, z_dim=32)
    assert config.device == "cpu"
    assert config.archi.z_dim == 32
    assert config.archi.actor.hidden_dim == 128
    assert config.archi.actor.hidden_layers == 2

    phase_space = x2_scratch_observation_space(
        X2ScratchDataConfig(), include_command_phase=True
    )
    assert phase_space["command"].shape == (COMMAND_DIM,)
    assert phase_space["gait_phase"].shape == (GAIT_PHASE_DIM,)
    phase_config = x2_scratch_model_config(
        device="cpu",
        hidden_dim=128,
        hidden_layers=2,
        z_dim=32,
        include_command_phase=True,
    )
    assert phase_config.archi.actor.input_filter.key[-2:] == (
        "command",
        "gait_phase",
    )


def test_parameter_summary_uses_all_bfm_components() -> None:
    config = x2_scratch_model_config(device="cpu", hidden_dim=64, hidden_layers=2, z_dim=16)
    model = config.build(x2_scratch_observation_space(), ACTION_DIM)
    summary = parameter_summary(model)
    assert summary["total"] > 0
    for name in ("actor", "forward", "backward", "critic", "aux_critic", "discriminator"):
        assert summary[name] > 0


def test_residual_model_rejects_one_hidden_layer() -> None:
    with pytest.raises(ValueError, match="capacity"):
        x2_scratch_model_config(device="cpu", hidden_dim=64, hidden_layers=1, z_dim=16)


def test_model_device_is_logical_not_indexed() -> None:
    with pytest.raises(ValueError, match="device"):
        x2_scratch_model_config(device="cuda:0")


def test_build_official_expert_and_seed_replay_buffers() -> None:
    episodes = [
        convert_x2_motion_entry(synthetic_entry(16), motion_id=index)[0]
        for index in range(2)
    ]
    replay = build_x2_scratch_replay_buffers(
        episodes,
        seq_length=8,
        z_dim=16,
        seed=4,
    )
    assert replay["audit"] == {
        "expert_frames": 32,
        "seed_transitions": 30,
        "motion_count": 2,
        "seq_length": 8,
        "z_dim": 16,
        "device": "cpu",
    }
    assert replay["expert_slicer"].storage["observation"]["state"].shape == (32, STATE_DIM)
    assert replay["train"].storage["z"].shape == (30, 16)
    np.testing.assert_allclose(
        np.linalg.norm(replay["train"].storage["z"].numpy(), axis=-1),
        4.0,
        atol=1.0e-5,
    )


def test_scratch_agent_config_is_full_and_bounded() -> None:
    config = x2_scratch_agent_config(
        device="cpu",
        hidden_dim=64,
        hidden_layers=2,
        z_dim=16,
        batch_size=16,
    )
    assert config.train.batch_size == 16
    assert config.model.archi.z_dim == 16
    assert config.compile is False
    assert config.cudagraphs is False
    with pytest.raises(ValueError, match="multiple of 8"):
        x2_scratch_agent_config(batch_size=10)
    tracking = x2_scratch_agent_config(
        rollout_expert_trajectories=True,
        rollout_expert_trajectories_length=128,
        rollout_expert_trajectories_percentage=0.5,
    )
    assert tracking.train.rollout_expert_trajectories is True
    assert tracking.train.rollout_expert_trajectories_length == 128
    assert tracking.train.rollout_expert_trajectories_percentage == 0.5


def test_x2_auxiliary_reward_config_is_explicitly_opt_in() -> None:
    plain = x2_scratch_agent_config()
    auxiliary = x2_scratch_agent_config(use_x2_aux_rewards=True)
    assert plain.aux_rewards == []
    assert auxiliary.aux_rewards == list(X2_AUX_REWARD_SCALES)
    assert auxiliary.aux_rewards_scaling == X2_AUX_REWARD_SCALES


def test_convert_stage219_closed_loop_rollout_preserves_proprioception() -> None:
    rollout = np.zeros((12, 3, 93), dtype=np.float32)
    rollout[..., 0] = 0.35
    rollout[..., 6:9] = (0.0, 0.0, -1.0)
    rollout[..., 12:43] = 0.1
    rollout[..., 43:74] = 0.2
    rollout[1:, :, 74:89] = 0.25
    episodes, audit = convert_stage219_critic_rollout(rollout, motion_id_offset=4)
    assert audit["episode_count"] == 3
    assert audit["total_frames"] == 36
    assert audit["all_finite"] is True
    assert audit["forward_velocity_mean_mps"] == pytest.approx(0.35)
    episode = episodes[1]
    assert episode["observation"]["state"].shape == (12, STATE_DIM)
    np.testing.assert_allclose(episode["observation"]["state"][:, -3], 0.35)
    np.testing.assert_allclose(episode["action"][:-1], 0.25)
    assert episode["motion_id"][0, 0] == 5
    assert episode["truncated"][-1, 0]


def test_stage219_joint_order_is_explicitly_reindexed() -> None:
    rollout = np.zeros((12, 1, 93), dtype=np.float32)
    rollout[..., 12:43] = np.arange(31, dtype=np.float32)
    episodes, audit = convert_stage219_critic_rollout(rollout)
    expected = np.asarray(
        [X2_ISAAC_JOINTS_31.index(name) for name in X2_JOINTS_31],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        episodes[0]["observation"]["state"][0, :31], expected
    )
    assert audit["joint_order_reindexed"] is True


def test_phase_conditioned_stage219_conversion_preserves_deployable_suffix() -> None:
    rollout = np.zeros((12, 2, 93), dtype=np.float32)
    rollout[..., 9:12] = (0.35, -0.02, 0.1)
    rollout[..., 89:93] = (0.5, -0.5, 1.0, 0.0)
    episodes, audit = convert_stage219_critic_rollout(
        rollout, include_command_phase=True
    )
    np.testing.assert_array_equal(
        episodes[0]["observation"]["command"], rollout[:, 0, 9:12]
    )
    np.testing.assert_array_equal(
        episodes[1]["observation"]["gait_phase"], rollout[:, 1, 89:93]
    )
    assert audit["command_phase_observation_contract"] is True


def test_deployable_gait_phase_features_match_stage181_clock_contract() -> None:
    steps = np.asarray((0, 3, 4, 10, 16, 20, 24, 36), dtype=np.int64)
    command = np.zeros((len(steps), 3), dtype=np.float32)
    command[:, 0] = 0.35
    gait = deployable_gait_phase_features(steps, command)
    np.testing.assert_allclose(
        gait[:, 0], np.sin(2.0 * np.pi * steps / 40.0), atol=2.0e-7
    )
    np.testing.assert_allclose(
        gait[:, 1], np.cos(2.0 * np.pi * steps / 40.0), atol=2.0e-7
    )
    np.testing.assert_array_equal(gait[0, 2:], (1.0, 1.0))
    np.testing.assert_array_equal(gait[4, 2:], (1.0, 0.0))
    np.testing.assert_array_equal(gait[6, 2:], (0.0, 1.0))

    command[:, 0] = 0.0
    standing = deployable_gait_phase_features(steps, command)
    np.testing.assert_array_equal(standing[:, :2], 0.0)
    np.testing.assert_array_equal(standing[:, 2:], 1.0)


def test_reconstruct_stage219_direct_action_labels_aligns_next_last_action() -> None:
    rollout = np.zeros((10, 2, 93), dtype=np.float32)
    rollout[1:, :, 74:89] = 0.2
    cycle = np.zeros((40, ACTION_DIM), dtype=np.float32)
    scale = np.full(ACTION_DIM, 0.4, dtype=np.float32)
    labels = reconstruct_stage219_direct_action_labels(rollout, cycle, scale)
    assert labels.shape == (9, 2, ACTION_DIM)
    expected = 0.2 * scale / X2_SCRATCH_ACTION_SCALE_15
    np.testing.assert_allclose(
        labels, np.broadcast_to(expected, labels.shape), rtol=1.0e-6
    )

    cycle[:, 0] = 0.4
    labels = reconstruct_stage219_direct_action_labels(
        rollout, cycle, scale, template_scale=0.15
    )
    np.testing.assert_allclose(labels[..., 0], 0.28, rtol=1.0e-6)
