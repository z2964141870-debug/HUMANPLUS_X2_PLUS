from pathlib import Path

import mujoco
import numpy as np

from retarget.build_x2_official_contract import (
    actuated_joint_names,
    contact_mirror_permutation,
    mirror_joint_values,
    verify_mirror_fk,
)


SCENE = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/"
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"
)


def test_official_x2_joint_count_and_mirror_roundtrip():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    names = actuated_joint_names(model)
    assert len(names) == 31
    values = np.arange(31, dtype=np.float64) / 10.0
    assert np.array_equal(mirror_joint_values(mirror_joint_values(values, names), names), values)


def test_official_x2_foot_contact_contract():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    permutation = contact_mirror_permutation(model)
    assert permutation == [2, 3, 0, 1, 5, 4, 7, 6, 9, 8, 10, 11]


def test_official_x2_random_fk_mirror():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    report = verify_mirror_fk(model, seeds=3)
    assert report["body_position_max_abs_m"] < 3.0e-3
    assert report["body_rotation_max_abs"] < 7.0e-3
    assert report["foot_contact_position_max_abs_m"] < 1.0e-8
