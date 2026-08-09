from pathlib import Path

import joblib
import mujoco
import numpy as np

from retarget.run_x2_forefoot_official_physics_screen import (
    DEFAULT_CONTROL,
    DEFAULT_CURRENT,
    DEFAULT_FOREFOOT,
    DEFAULT_SCENE,
    build_control_contract,
    interpolate_motion,
    validate_motion_pair,
)


def test_official_pd_and_period_contract():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    contract = build_control_contract(model, DEFAULT_CONTROL)
    index = {name: i for i, name in enumerate(contract.actuator_joint_names)}
    assert model.opt.timestep == 0.001
    assert contract.control_dt == 0.02
    assert len(contract.actuator_joint_names) == 31
    assert contract.kp[index["left_hip_pitch_joint"]] == 120.0
    assert contract.kp[index["left_hip_roll_joint"]] == 100.0
    assert contract.kp[index["waist_pitch_joint"]] == 200.0
    assert contract.kp[index["head_yaw_joint"]] == 20.0
    assert contract.kd[index["head_yaw_joint"]] == 1.0
    assert np.array_equal(contract.torque_low, -contract.torque_high)


def test_exact30_pair_has_identical_panel_and_shape_contract():
    current = joblib.load(DEFAULT_CURRENT)
    forefoot = joblib.load(DEFAULT_FOREFOOT)
    report = validate_motion_pair(current, forefoot)
    assert report["matched"] is True
    assert len(report["motions"]) == 9
    assert {row["panel_role"] for row in report["motions"]} >= {
        "walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand"
    }


def test_interpolation_preserves_exact_source_frames():
    entry = next(iter(joblib.load(DEFAULT_CURRENT).values()))
    times = np.arange(len(entry["dof"]), dtype=np.float64) / float(entry["fps"])
    pos, quat_wxyz, dof = interpolate_motion(entry, times)
    assert np.allclose(pos, entry["root_trans_offset"], atol=1.0e-12)
    assert np.allclose(dof, entry["dof"], atol=1.0e-12)
    expected_wxyz = np.asarray(entry["root_rot"])[:, [3, 0, 1, 2]]
    # Quaternions q and -q are equivalent; use absolute dot products.
    assert np.allclose(np.abs(np.sum(quat_wxyz * expected_wxyz, axis=1)), 1.0, atol=1.0e-12)
