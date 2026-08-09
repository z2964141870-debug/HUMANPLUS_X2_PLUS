import numpy as np

from official_x2.extract_stop_recovery_states import gravity_to_roll_pitch


def projected_gravity(roll: float, pitch: float) -> np.ndarray:
    return np.asarray(
        [
            np.sin(pitch),
            -np.sin(roll) * np.cos(pitch),
            -np.cos(roll) * np.cos(pitch),
        ]
    )


def test_gravity_to_roll_pitch_roundtrip():
    for roll, pitch in ((0.0, 0.0), (0.2, -0.1), (-0.3, 0.25)):
        recovered = gravity_to_roll_pitch(projected_gravity(roll, pitch))
        assert np.allclose(recovered, (roll, pitch), atol=1.0e-7)
