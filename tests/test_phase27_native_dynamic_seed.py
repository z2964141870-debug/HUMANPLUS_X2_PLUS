import math

import numpy as np

from official_x2.audit_stage250_native_dynamic_seed import (
    ISAAC_JOINTS,
    LOWER_JOINTS,
    OBS_SLICES,
    contiguous_cycles,
    projected_gravity,
    quaternion_wxyz_from_yaw_gravity,
    yaw_from_quaternion,
)


def test_observation_partition_is_exact_93d():
    widths = [sl.stop - sl.start for sl in OBS_SLICES.values()]
    assert widths == [3, 3, 3, 3, 31, 31, 15, 4]
    assert sum(widths) == 93
    assert len(ISAAC_JOINTS) == 31
    assert len(LOWER_JOINTS) == 15


def test_yaw_gravity_quaternion_round_trip():
    for yaw in (-2.0, -0.3, 0.0, 1.4):
        for roll, pitch in ((0.0, 0.0), (0.2, -0.15), (-0.1, 0.25)):
            gravity = np.asarray(
                [math.sin(pitch), -math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch)]
            )
            quaternion = quaternion_wxyz_from_yaw_gravity(yaw, gravity)
            assert np.allclose(projected_gravity(quaternion), gravity, atol=1e-12)
            error = math.atan2(
                math.sin(yaw_from_quaternion(quaternion) - yaw),
                math.cos(yaw_from_quaternion(quaternion) - yaw),
            )
            assert abs(error) <= 1e-12


def test_ds_ss_ds_cycle_counter():
    left = np.asarray([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=bool)
    right = np.asarray([1, 1, 1, 1, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert contiguous_cycles(left, right) == 2


def test_no_false_cycle_without_double_support_return():
    left = np.asarray([1, 1, 0, 0, 0], dtype=bool)
    right = np.asarray([1, 1, 1, 1, 1], dtype=bool)
    assert contiguous_cycles(left, right) == 0
