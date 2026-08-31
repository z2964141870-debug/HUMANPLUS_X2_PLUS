import unittest

import numpy as np

from phase_contact_root_tilt_adapter import support_mask


class PhaseContactRootTiltTest(unittest.TestCase):
    def test_one_stationary_near_floor_foot_is_support(self):
        positions = np.asarray([
            [[0.0, 0.0, 0.04], [0.2, 0.0, 0.20]],
            [[0.0, 0.0, 0.04], [0.2, 0.0, 0.20]],
        ])
        self.assertTrue(np.all(support_mask(positions, 50.0, 0.08, 0.25)))

    def test_two_high_or_fast_feet_have_no_support(self):
        positions = np.asarray([
            [[0.0, 0.0, 0.20], [0.2, 0.0, 0.20]],
            [[0.3, 0.0, 0.20], [0.5, 0.0, 0.20]],
        ])
        self.assertTrue(np.all(~support_mask(positions, 50.0, 0.08, 0.25)))


if __name__ == "__main__":
    unittest.main()
