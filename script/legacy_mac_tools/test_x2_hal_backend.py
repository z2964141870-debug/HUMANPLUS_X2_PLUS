import unittest

import numpy as np

from x2_hal_backend import HalTeleopBackend, LinearTargetInterpolator


class InterpolatorTest(unittest.TestCase):
    def test_interpolates_between_policy_ticks(self):
        interp = LinearTargetInterpolator(transition_s=0.02)
        interp.update(np.zeros(29), 10.0)
        interp.update(np.ones(29), 10.02)
        np.testing.assert_allclose(interp.sample(10.03), 0.5, atol=1e-12)
        np.testing.assert_allclose(interp.sample(10.04), 1.0, atol=1e-12)

    def test_dry_backend_clamps_and_omits_ros(self):
        backend = HalTeleopBackend(dry_run=True)
        target = np.zeros(31)
        target[5] = 99.0
        target[29:] = [0.2, -0.2]
        self.assertTrue(backend.send(target))
        self.assertAlmostEqual(backend.last_qpos62[12], 0.3625)
        np.testing.assert_allclose(backend.last_qpos62[36:38], [0.2, -0.2])
        backend.close()

    def test_measured_hanging_pose_is_preserved_on_arm(self):
        backend = HalTeleopBackend(dry_run=True)
        target = np.zeros(31)
        target[13] = 0.3491
        target[16] = -0.12798977
        target[23] = 0.13757658
        backend.send(target)
        np.testing.assert_allclose(backend.last_qpos62[20:31][[0, 3, 10]], target[[13, 16, 23]])
        backend.close()

    def test_rejects_bad_shape(self):
        backend = HalTeleopBackend(dry_run=True)
        with self.assertRaises(ValueError):
            backend.send(np.zeros(29))

    def test_rejects_bad_gain_scale(self):
        backend = HalTeleopBackend(dry_run=True)
        with self.assertRaises(ValueError):
            backend.send(np.zeros(31), gain_scale=1.1)

    def test_rejects_unknown_gain_profile(self):
        backend = HalTeleopBackend(dry_run=True)
        with self.assertRaises(ValueError):
            backend.send(np.zeros(31), gain_profile="unknown")


if __name__ == "__main__":
    unittest.main()
