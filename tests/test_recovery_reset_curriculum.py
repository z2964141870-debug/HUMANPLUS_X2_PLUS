from types import SimpleNamespace

import numpy as np
import pytest
import torch

from official_x2.recovery_reset_curriculum import (
    audit_recovery_dataset,
    balanced_sample_indices,
    joint_reorder_indices,
    load_recovery_dataset,
    reset_from_recovery_dataset,
)


DATASET = (
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/"
    "stage335_stage306_stiff_fixed_stop_recovery_states.npz"
)
SHA256 = "4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"


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
