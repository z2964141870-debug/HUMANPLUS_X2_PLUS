#!/usr/bin/env python3
import math
import unittest

import numpy as np

from x2_sonic_input_adapter import (
    DEFAULT_ANGLES_MJ,
    X2SonicAdapterConfig,
    X2SonicInputAdapter,
)


class AdapterTest(unittest.TestCase):
    def test_upright_identity_and_pose_scale(self):
        adapter = X2SonicInputAdapter(X2SonicAdapterConfig(root_tilt_scale=0.0, pose_scale=0.5))
        raw = DEFAULT_ANGLES_MJ + 1.0
        out_jp, out_q, telemetry = adapter.process(raw, np.asarray([1.0, 0.0, 0.0, 0.0]), 50.0)
        np.testing.assert_allclose(out_q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(out_jp, DEFAULT_ANGLES_MJ + 0.5, atol=1e-12)
        self.assertAlmostEqual(telemetry["raw_root_tilt_rad"], 0.0, places=8)
        self.assertAlmostEqual(telemetry["adapted_root_tilt_rad"], 0.0, places=8)
        self.assertTrue(telemetry["finite"])

    def test_root_tilt_is_reduced_but_yaw_kept(self):
        # 30-degree pitch followed by 45-degree yaw, scalar-first quaternion.
        pitch, yaw = math.radians(30.0), math.radians(45.0)
        q = np.asarray([
            math.cos(pitch / 2) * math.cos(yaw / 2),
            -math.sin(pitch / 2) * math.sin(yaw / 2),
            math.sin(pitch / 2) * math.cos(yaw / 2),
            math.cos(pitch / 2) * math.sin(yaw / 2),
        ])
        adapter = X2SonicInputAdapter(X2SonicAdapterConfig(root_tilt_scale=0.5, pose_scale=1.0))
        _, out_q, telemetry = adapter.process(DEFAULT_ANGLES_MJ, q, 50.0)
        self.assertLess(telemetry["adapted_root_tilt_rad"], telemetry["raw_root_tilt_rad"])
        # Slerping with the yaw-only quaternion must preserve the yaw command.
        out_yaw = math.atan2(2 * (out_q[0] * out_q[3] + out_q[1] * out_q[2]), 1 - 2 * (out_q[2] ** 2 + out_q[3] ** 2))
        self.assertAlmostEqual(out_yaw, yaw, places=6)

    def test_rate_limit_uses_adapted_previous_output(self):
        adapter = X2SonicInputAdapter(X2SonicAdapterConfig(root_tilt_scale=0.0, pose_scale=1.0, joint_speed_limit_radps=1.0))
        first, _, t0 = adapter.process(DEFAULT_ANGLES_MJ, [1, 0, 0, 0], 50.0)
        second, _, t1 = adapter.process(DEFAULT_ANGLES_MJ + 1.0, [1, 0, 0, 0], 50.0)
        np.testing.assert_allclose(first, DEFAULT_ANGLES_MJ)
        np.testing.assert_allclose(second - first, 1.0 / 50.0, atol=1e-12)
        self.assertFalse(t0["joint_rate_limited"])
        self.assertTrue(t1["joint_rate_limited"])


if __name__ == "__main__":
    unittest.main()
