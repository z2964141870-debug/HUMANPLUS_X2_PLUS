import numpy as np

from official_x2.analyze_phase34_full_closed_trace import root_score, target_for_row
from official_x2.replay_official_trace_direct_mujoco import ARM_JOINTS, LOWER_JOINTS


def test_root_score_zero_for_identical_pose():
    qpos = np.asarray([1.0, 2.0, .65, 1.0, 0.0, 0.0, 0.0])
    row = {"root_x_m": 1.0, "root_y_m": 2.0, "root_z_m": .65,
           "root_yaw_rad": 0.0, "root_tilt_rad": 0.0}
    assert root_score(qpos, row) == 0.0


def test_target_contract_maps_lower_and_upper_without_reordering():
    row = {"physical_lower_target_rad": list(range(len(LOWER_JOINTS))),
           "upper_target_rad": list(range(100, 100 + len(ARM_JOINTS)))}
    target = target_for_row(row)
    assert [target[name] for name in LOWER_JOINTS] == list(range(len(LOWER_JOINTS)))
    assert [target[name] for name in ARM_JOINTS] == list(range(100, 100 + len(ARM_JOINTS)))
    assert target["head_yaw_joint"] == target["head_pitch_joint"] == 0.0


def test_target_contract_reconstructs_legacy_empty_physical_from_action():
    row = {"physical_lower_target_rad": [], "action": [0.0] * len(LOWER_JOINTS),
           "upper_target_rad": [0.0] * len(ARM_JOINTS)}
    target = target_for_row(row)
    assert target["left_hip_pitch_joint"] == -0.248
    assert target["left_knee_joint"] == 0.5303
