import pytest
import torch

from humanoidverse.x2_extremity_contract import (
    X2_EXTREMITY_LINK_NAMES,
    X2_EXTERNAL_ARM_JOINT_NAMES,
    X2_LOCKED_HEAD_JOINT_NAMES,
    X2_LOWER_POLICY_JOINT_NAMES,
    extremity_future_pose_error,
    resolve_named_indices,
)


def identity_pose(batch: int = 2):
    position = torch.zeros(batch, 6, 3)
    quaternion = torch.zeros(batch, 6, 4)
    quaternion[..., 0] = 1.0
    return position, quaternion


def test_link_contract_resolves_order() -> None:
    available = ("unused",) + tuple(reversed(X2_EXTREMITY_LINK_NAMES))
    indices = resolve_named_indices(available, X2_EXTREMITY_LINK_NAMES)
    assert tuple(available[index] for index in indices) == X2_EXTREMITY_LINK_NAMES


def test_link_contract_rejects_missing_name() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        resolve_named_indices(X2_EXTREMITY_LINK_NAMES[:-1], X2_EXTREMITY_LINK_NAMES)


def test_joint_partition_is_exactly_31_unique_joints() -> None:
    partition = (
        X2_LOWER_POLICY_JOINT_NAMES
        + X2_EXTERNAL_ARM_JOINT_NAMES
        + X2_LOCKED_HEAD_JOINT_NAMES
    )
    assert (len(X2_LOWER_POLICY_JOINT_NAMES), len(X2_EXTERNAL_ARM_JOINT_NAMES)) == (15, 14)
    assert len(partition) == len(set(partition)) == 31


def test_identical_target_has_exact_zero_error() -> None:
    position, quaternion = identity_pose()
    encoded = extremity_future_pose_error(position, quaternion, position, quaternion)
    assert encoded.shape == (2, 1, 6, 6)
    assert torch.count_nonzero(encoded) == 0


def test_quaternion_sign_does_not_change_rotation_error() -> None:
    position, quaternion = identity_pose()
    encoded = extremity_future_pose_error(position, quaternion, position, -quaternion)
    torch.testing.assert_close(encoded, torch.zeros_like(encoded))


def test_single_link_translation_is_localized() -> None:
    position, quaternion = identity_pose(batch=1)
    target = position.clone()
    target[:, 2, 1] = 0.03
    encoded = extremity_future_pose_error(position, quaternion, target, quaternion)
    assert encoded[0, 0, 2, 1].item() == pytest.approx(0.03)
    untouched = encoded.clone()
    untouched[0, 0, 2, 1] = 0.0
    assert torch.count_nonzero(untouched) == 0


def test_future_horizon_is_preserved() -> None:
    position, quaternion = identity_pose(batch=1)
    target_position = position[:, None].repeat(1, 4, 1, 1)
    target_quaternion = quaternion[:, None].repeat(1, 4, 1, 1)
    target_position[:, :, 0, 2] += torch.arange(4).reshape(1, 4) * 0.01
    encoded = extremity_future_pose_error(
        position,
        quaternion,
        target_position,
        target_quaternion,
    )
    assert encoded.shape == (1, 4, 6, 6)
    torch.testing.assert_close(encoded[0, :, 0, 2], torch.tensor((0.0, 0.01, 0.02, 0.03)))


def test_position_error_is_expressed_in_current_pelvis_frame() -> None:
    position, quaternion = identity_pose(batch=1)
    half = 2.0**-0.5
    quaternion[:, 0] = torch.tensor((half, 0.0, 0.0, half))
    target = position.clone()
    target[:, 2, 0] = 0.03
    encoded = extremity_future_pose_error(position, quaternion, target, quaternion)
    assert encoded[0, 0, 2, 0].item() == pytest.approx(0.0, abs=1.0e-7)
    assert encoded[0, 0, 2, 1].item() == pytest.approx(-0.03)


def test_rotation_error_uses_shortest_rotation_vector() -> None:
    position, quaternion = identity_pose(batch=1)
    target_quaternion = quaternion.clone()
    half = 2.0**-0.5
    target_quaternion[:, 2] = torch.tensor((half, 0.0, 0.0, half))
    encoded = extremity_future_pose_error(
        position,
        quaternion,
        position,
        target_quaternion,
    )
    assert encoded[0, 0, 2, 5].item() == pytest.approx(torch.pi / 2.0)
