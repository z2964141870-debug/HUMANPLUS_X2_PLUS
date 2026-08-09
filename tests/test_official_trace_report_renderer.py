import math

import numpy as np

from official_x2.render_official_trace_report import (
    root_quaternion_wxyz,
    root_roll_pitch_from_projected_gravity,
)


def test_projected_gravity_round_trip_for_known_roll_pitch():
    roll = math.radians(7.0)
    pitch = math.radians(-11.0)
    gravity = np.asarray(
        [math.sin(pitch), -math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch)]
    )
    actual_roll, actual_pitch = root_roll_pitch_from_projected_gravity(gravity)
    assert actual_roll == pytest.approx(roll, abs=1.0e-12)
    assert actual_pitch == pytest.approx(pitch, abs=1.0e-12)


def test_root_quaternion_is_unit_and_preserves_yaw_for_upright_row():
    row = {"root_yaw_rad": 0.4, "obs": [0.0] * 6 + [0.0, 0.0, -1.0] + [0.0] * 84}
    quat = root_quaternion_wxyz(row)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1.0e-12)
    assert quat[0] == pytest.approx(math.cos(0.2), abs=1.0e-12)
    assert quat[3] == pytest.approx(math.sin(0.2), abs=1.0e-12)


import pytest
