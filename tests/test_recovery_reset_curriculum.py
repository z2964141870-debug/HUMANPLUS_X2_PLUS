from types import SimpleNamespace

import numpy as np
import pytest
import torch

from official_x2.recovery_reset_curriculum import (
    audit_stateful_recovery_sidecar,
    audit_recovery_dataset,
    balanced_sample_indices,
    finalize_stateful_recovery,
    gait_phase_from_steps,
    joint_reorder_indices,
    load_recovery_dataset,
    load_stateful_recovery_sidecar,
    reset_from_recovery_dataset,
)


DATASET = (
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/"
    "stage335_stage306_stiff_fixed_stop_recovery_states.npz"
)
SHA256 = "4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
SOURCE_REPORT = (
    "/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/reports/official_x2/"
    "stage335_stop_recovery_state_extraction.json"
)


def test_stage335_dataset_exact_contract():
    audit = audit_recovery_dataset(DATASET, SHA256)
    assert audit.state_count == 90
    assert audit.eventual_pass_count == 63
    assert audit.eventual_fail_count == 27
    assert audit.source_counts == {0: 21, 1: 8, 2: 21, 3: 21, 4: 19}
    assert audit.root_z_range_m[0] >= 0.60
    assert audit.max_abs_tilt_rad < 0.50
    assert audit.max_body_speed_mps < 0.60
    assert audit.gravity_norm_max_error < 1.0e-6


def test_stage335_hash_is_enforced():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_recovery_dataset(DATASET, "0" * 64)


def test_joint_reorder_and_balanced_sampling():
    assert joint_reorder_indices(["b", "a"], ["a", "b"]).tolist() == [1, 0]
    with pytest.raises(ValueError, match="joint-name contract mismatch"):
        joint_reorder_indices(["a", "b"], ["a", "c"])
    labels = np.asarray([True, True, False])
    samples = balanced_sample_indices(labels, 101, np.random.default_rng(7))
    selected = labels[samples]
    assert int(selected.sum()) in {50, 51}


class _FakeAsset:
    def __init__(self, arrays, num_envs=4):
        self.device = "cpu"
        self.joint_names = arrays["joint_names"].tolist()
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros(num_envs, 31),
            soft_joint_pos_limits=torch.stack(
                (-torch.full((num_envs, 31), 10.0), torch.full((num_envs, 31), 10.0)), dim=-1
            ),
            soft_joint_vel_limits=torch.full((num_envs, 31), 20.0),
            joint_vel_limits=torch.full((num_envs, 31), 25.0),
        )
        self.root_pose = None
        self.root_velocity = None
        self.joint_pos = None
        self.joint_vel = None

    def write_root_pose_to_sim(self, value, env_ids):
        self.root_pose = value.clone()

    def write_root_velocity_to_sim(self, value, env_ids):
        self.root_velocity = value.clone()

    def write_joint_state_to_sim(self, position, velocity, env_ids):
        self.joint_pos = position.clone()
        self.joint_vel = velocity.clone()


class _FakeScene(dict):
    def __init__(self, asset, num_envs=4):
        super().__init__(robot=asset)
        self.env_origins = torch.zeros(num_envs, 3)


def test_physical_reset_event_overwrites_selected_states_only():
    arrays = load_recovery_dataset(DATASET, SHA256)
    asset = _FakeAsset(arrays)
    scene = _FakeScene(asset)
    env = SimpleNamespace(scene=scene, device="cpu")
    torch.manual_seed(3)
    reset_from_recovery_dataset(
        env,
        torch.arange(4),
        DATASET,
        1.0,
        sampling_mode="balanced",
        expected_sha256=SHA256,
    )
    sample_ids = env._x2_recovery_reset_last["sample_indices"].numpy()
    labels = arrays["eventual_pass"][sample_ids]
    assert int(labels.sum()) == 2
    assert asset.root_pose.shape == (4, 7)
    assert asset.root_velocity.shape == (4, 6)
    assert asset.joint_pos.shape == (4, 31)
    assert torch.allclose(asset.root_pose[:, 2], torch.as_tensor(arrays["root_z_m"][sample_ids]).float())
    assert torch.allclose(asset.joint_pos, torch.as_tensor(arrays["joint_pos_rel_rad"][sample_ids]))
    assert torch.allclose(asset.joint_vel, torch.as_tensor(arrays["joint_vel_radps"][sample_ids]))


def test_zero_fraction_is_exact_noop():
    arrays = load_recovery_dataset(DATASET, SHA256)
    asset = _FakeAsset(arrays)
    scene = _FakeScene(asset)
    env = SimpleNamespace(scene=scene, device="cpu")
    reset_from_recovery_dataset(env, torch.arange(4), DATASET, 0.0)
    assert env._x2_recovery_reset_last["selected_env_ids"].numel() == 0
    assert asset.root_pose is None
    assert asset.joint_pos is None


def test_first_reset_uses_hard_velocity_limits_when_soft_limits_are_uninitialized():
    arrays = load_recovery_dataset(DATASET, SHA256)
    asset = _FakeAsset(arrays)
    asset.data.soft_joint_vel_limits.zero_()
    scene = _FakeScene(asset)
    env = SimpleNamespace(scene=scene, device="cpu")
    torch.manual_seed(3)
    reset_from_recovery_dataset(
        env,
        torch.arange(4),
        DATASET,
        1.0,
        sampling_mode="balanced",
        expected_sha256=SHA256,
    )
    sample_ids = env._x2_recovery_reset_last["sample_indices"].numpy()
    assert env._x2_recovery_reset_last["velocity_limit_fallback_count"] == 4 * 31
    assert torch.allclose(asset.joint_vel, torch.as_tensor(arrays["joint_vel_radps"][sample_ids]))


def test_stateful_sidecar_is_lossless_and_clock_reconstructs_phase():
    audit = audit_stateful_recovery_sidecar(DATASET, SOURCE_REPORT, SHA256)
    assert audit.state_count == 90
    assert audit.moving_phase_count == 35
    assert audit.stationary_phase_count == 55
    assert audit.low_command_forced_moving_count == 12
    assert audit.max_observation_crosscheck_error <= 1.0e-7
    assert audit.previous_issued_action_missing_count == 0
    arrays = load_recovery_dataset(DATASET, SHA256)
    sidecar = load_stateful_recovery_sidecar(DATASET, SOURCE_REPORT, SHA256)
    reconstructed = gait_phase_from_steps(
        sidecar["episode_clock_steps"], sidecar["force_moving"]
    )
    assert np.max(np.abs(reconstructed - arrays["gait_phase"])) < 1.0e-5
    assert np.isfinite(sidecar["previous_issued_action"]).all()


class _FakeActionTerm:
    def __init__(self, num_envs=4):
        self._raw_actions = torch.zeros(num_envs, 15)
        self._processed_actions = torch.zeros(num_envs, 15)
        self._combined_normalized_actions = torch.zeros(num_envs, 15)
        self._preclip_combined_actions = torch.zeros(num_envs, 15)
        self._normalized_template_bias = torch.zeros(num_envs, 15)
        self._scale = torch.full((num_envs, 15), 0.25)
        self._offset = torch.zeros(num_envs, 15)
        self._clip = torch.stack(
            (-torch.ones(num_envs, 15), torch.ones(num_envs, 15)), dim=-1
        )


class _FakeActionManager:
    def __init__(self, num_envs=4):
        self._action = torch.zeros(num_envs, 15)
        self._prev_action = torch.zeros(num_envs, 15)
        self.term = _FakeActionTerm(num_envs)

    def get_term(self, name):
        assert name == "joint_pos"
        return self.term


class _FakeCommandTerm:
    def __init__(self, num_envs=4):
        self.vel_command_b = torch.zeros(num_envs, 3)
        self.is_standing_env = torch.ones(num_envs, dtype=torch.bool)
        self.is_heading_env = torch.zeros(num_envs, dtype=torch.bool)
        self.heading_target = torch.zeros(num_envs)
        self.time_left = torch.zeros(num_envs)
        self.command_counter = torch.zeros(num_envs, dtype=torch.long)


class _FakeCommandManager:
    def __init__(self, num_envs=4):
        self.term = _FakeCommandTerm(num_envs)

    def get_term(self, name):
        assert name == "base_velocity"
        return self.term


def _fake_stateful_env(num_envs=4):
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        action_manager=_FakeActionManager(num_envs),
        command_manager=_FakeCommandManager(num_envs),
        _x2_recovery_reset_last={},
    )


def test_post_manager_stateful_finalizer_restores_all_snapshot_buffers():
    arrays = load_recovery_dataset(DATASET, SHA256)
    sidecar = load_stateful_recovery_sidecar(DATASET, SOURCE_REPORT, SHA256)
    env = _fake_stateful_env()
    selected = torch.tensor([1, 3])
    sample_ids = torch.tensor([0, 4])
    env._x2_recovery_stateful_pending = {
        "selected_env_ids": selected,
        "sample_indices": sample_ids,
        "dataset_path": DATASET,
        "expected_dataset_sha256": SHA256,
        "source_report_path": SOURCE_REPORT,
        "expected_source_report_sha256": None,
    }
    result = finalize_stateful_recovery(env)
    expected_previous = torch.as_tensor(arrays["previous_action"][[0, 4]])
    expected_issued = torch.as_tensor(sidecar["previous_issued_action"][[0, 4]])
    assert result["selected_env_count"] == 2
    assert torch.allclose(env.action_manager._action[selected], expected_previous)
    assert torch.allclose(env.action_manager._prev_action[selected], expected_previous)
    assert torch.allclose(env.action_manager.term._raw_actions[selected], expected_previous)
    assert torch.allclose(
        env.action_manager.term._combined_normalized_actions[selected], expected_issued
    )
    assert torch.allclose(
        env.action_manager.term._processed_actions[selected], expected_issued * 0.25
    )
    assert torch.equal(
        env.episode_length_buf[selected],
        torch.as_tensor(sidecar["episode_clock_steps"][[0, 4]]),
    )
    assert torch.allclose(
        env.command_manager.term.vel_command_b[selected],
        torch.as_tensor(sidecar["command_velocity_mps_radps"][[0, 4]]),
    )
    assert env._x2_recovery_reset_last["stateful_finalized"] is True


def test_stateful_finalizer_without_pending_payload_is_strict_noop():
    env = _fake_stateful_env()
    env.episode_length_buf[:] = torch.arange(4)
    env.action_manager._action[:] = 0.37
    env.command_manager.term.vel_command_b[:] = 0.19
    before = (
        env.episode_length_buf.clone(),
        env.action_manager._action.clone(),
        env.command_manager.term.vel_command_b.clone(),
    )
    result = finalize_stateful_recovery(env)
    assert result == {"selected_env_count": 0, "no_op": True}
    assert torch.equal(env.episode_length_buf, before[0])
    assert torch.equal(env.action_manager._action, before[1])
    assert torch.equal(env.command_manager.term.vel_command_b, before[2])
