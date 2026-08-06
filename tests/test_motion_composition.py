from __future__ import annotations

import numpy as np

from dcpeft_motion_composition import compose_gait_with_upper


def _motion(length: int, arm_wave: bool) -> dict:
    names = ["left_hip_pitch_joint", "left_shoulder_pitch_joint"]
    q = np.zeros((length, 2), dtype=np.float32)
    q[:, 0] = -0.2
    if arm_wave:
        q[:, 1] = np.sin(np.linspace(0.0, 4.0 * np.pi, length)).astype(np.float32)
    pose = np.zeros((length, 3, 3), dtype=np.float32)
    pose[:, 1, 1] = q[:, 0]
    pose[:, 2, 1] = q[:, 1]
    return {
        "dof": q,
        "pose_aa": pose,
        "joint_names_mujoco": names,
        "fps": 50,
        "root_trans_offset": np.zeros((length, 3), dtype=np.float32),
    }


def test_composition_changes_only_arms_and_preserves_boundaries() -> None:
    base = _motion(101, arm_wave=False)
    upper = _motion(201, arm_wave=True)
    result, report = compose_gait_with_upper(base, upper, alpha=0.6)
    assert np.array_equal(result["dof"][:, 0], base["dof"][:, 0])
    assert np.array_equal(result["dof"][[0, -1], 1], base["dof"][[0, -1], 1])
    assert np.max(np.abs(result["dof"][:, 1])) > 0.1
    assert report["lower_waist_exact"] is True
    assert report["pose_dof_norm_max_error"] < 1.0e-6
