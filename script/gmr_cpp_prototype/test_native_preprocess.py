from __future__ import annotations

import numpy as np

from native_preprocess import NativeFramePreprocessor


class Rotation:
    def __init__(self, quaternion):
        quaternion = np.asarray(quaternion, dtype=np.float64)
        self.quaternion = quaternion / np.linalg.norm(quaternion)

    @classmethod
    def from_quat(cls, quaternion, scalar_first=True):
        if not scalar_first:
            quaternion = np.asarray(quaternion)[[3, 0, 1, 2]]
        return cls(quaternion)

    @classmethod
    def from_rotvec(cls, rotvec):
        rotvec = np.asarray(rotvec, dtype=np.float64)
        angle = np.linalg.norm(rotvec)
        if angle < 1e-15:
            return cls([1.0, 0.0, 0.0, 0.0])
        half = 0.5 * angle
        xyz = rotvec * (np.sin(half) / angle)
        return cls(np.concatenate([[np.cos(half)], xyz]))

    def __mul__(self, other):
        w1, x1, y1, z1 = self.quaternion
        w2, x2, y2, z2 = other.quaternion
        return Rotation(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def apply(self, vector):
        w, x, y, z = self.quaternion
        vector = np.asarray(vector, dtype=np.float64)
        twice_cross = 2.0 * np.cross([x, y, z], vector)
        return vector + w * twice_cross + np.cross([x, y, z], twice_cross)

    def as_quat(self, scalar_first=True):
        if scalar_first:
            return self.quaternion.copy()
        return self.quaternion[[1, 2, 3, 0]].copy()


class Reference:
    def __init__(self, body_names, scales, position_offsets, rotation_offsets, root_name):
        self.body_names = body_names
        self.scales = scales
        self.position_offsets = position_offsets
        self.rotation_offsets = rotation_offsets
        self.root_name = root_name

    def __call__(self, human_data, ground_offset):
        root_position = human_data[self.root_name][0]
        scaled_root = self.scales[self.root_name] * root_position
        output = {}
        for name in self.body_names:
            position, quaternion = human_data[name]
            if name == self.root_name:
                scaled_position = scaled_root
            else:
                scaled_position = (position - root_position) * self.scales[name] + scaled_root
            updated = (
                Rotation.from_quat(quaternion, scalar_first=True) * self.rotation_offsets[name]
            )
            output[name] = (
                scaled_position + updated.apply(self.position_offsets[name])
                - np.array([0.0, 0.0, ground_offset]),
                updated.as_quat(scalar_first=True),
            )
        return output


def main():
    rng = np.random.default_rng(7)
    names = ("pelvis", "left_foot", "right_wrist")
    scales = {name: float(rng.uniform(0.7, 1.1)) for name in names}
    position_offsets = {name: rng.normal(0.0, 0.1, 3) for name in names}
    rotation_offsets = {
        name: Rotation.from_rotvec(rng.normal(0.0, 0.2, 3)) for name in names
    }
    native = NativeFramePreprocessor(
        names, scales, position_offsets, rotation_offsets, "pelvis"
    )
    reference = Reference(names, scales, position_offsets, rotation_offsets, "pelvis")
    maximum_error = 0.0
    for _ in range(1000):
        human_data = {
            name: (
                rng.normal(0.0, 1.0, 3),
                Rotation.from_rotvec(rng.normal(0.0, 1.0, 3)).as_quat(
                    scalar_first=True
                ),
            )
            for name in names
        }
        expected = reference(human_data, 0.13)
        actual = native(human_data, 0.13)
        for name in names:
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(expected[name][0] - actual[name][0]))),
                float(np.max(np.abs(expected[name][1] - actual[name][1]))),
            )
    print(f"native_preprocess_max_error={maximum_error:.3e}")
    if maximum_error > 1e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
