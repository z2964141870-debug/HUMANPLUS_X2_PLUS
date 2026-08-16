import unittest

import numpy as np

from phase_aware_root_tilt_adapter import allowance, slerp, yaw_quat


def tilt(q):
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return np.arccos(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))


class PhaseAwareRootTiltTest(unittest.TestCase):
    def test_low_and_descending_reference_suppress_allowance(self):
        root_pos = np.asarray([[0.0, 0.0, 0.44], [0.0, 0.0, 0.43], [0.0, 0.0, 0.42]])
        scales, stats = allowance(root_pos, 50.0, 0.1, 0.50, 0.58, -0.05, 0.0)
        self.assertTrue(np.all(scales <= 1.0e-8))
        self.assertEqual(stats["fraction_zero"], 1.0)

    def test_high_flat_reference_keeps_max_allowance(self):
        root_pos = np.asarray([[0.0, 0.0, 0.65], [0.0, 0.0, 0.65], [0.0, 0.0, 0.65]])
        scales, _ = allowance(root_pos, 50.0, 0.1, 0.50, 0.58, -0.05, 0.0)
        self.assertTrue(np.allclose(scales, 0.1))

    def test_slerp_never_introduces_tilt_at_zero(self):
        q = np.asarray([0.95, 0.1, -0.2, 0.2])
        out = slerp(yaw_quat(q), q, 0.0)
        self.assertAlmostEqual(float(tilt(out)), 0.0, places=7)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
