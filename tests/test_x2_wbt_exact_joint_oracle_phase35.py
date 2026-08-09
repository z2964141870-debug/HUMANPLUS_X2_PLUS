import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_exact_joint_oracle_phase35.json"


def test_phase35_is_single_oracle_with_frozen_contract():
    report = json.loads(REPORT.read_text())
    assert report["preflight"]["new_physics_mode_count"] == 1
    assert report["preflight"]["existing_free_pd_not_repeated"] is True
    assert report["truth_boundary"]["root_reference_pd_warmup_contact_labels_unchanged"] is True
    assert report["truth_boundary"]["training_ppo_optimizer_cem_real_robot_base_port51822"] is False


def test_phase35_trace_is_frame_complete_and_head_nominal():
    report = json.loads(REPORT.read_text())
    trace = np.load(report["trace"]["path"])
    count = len(trace["time_s"])
    for key in (
        "root_position_m", "root_quaternion_wxyz", "root_linear_velocity_mps",
        "root_angular_velocity_radps", "root_tilt_rad", "official_sole_collision_lr",
        "q_error_wbt29_rad", "contact_linear_impulse_lr_Ns", "contact_torque_impulse_lr_Nms",
    ):
        assert len(trace[key]) == count
    assert trace["q_error_wbt29_rad"].shape[1] == 29
    assert trace["head_joint_names"].shape == (2,)
    assert np.max(np.abs(trace["qfrc_actuator_wbt29"])) == 0.0


def test_phase35_terminal_and_attribution_are_pre_registered():
    report = json.loads(REPORT.read_text())
    oracle = report["oracle"]
    assert oracle["terminal"] is not None or oracle["duration_fraction"] >= 0.999
    assert oracle["pre_registered_attribution"]["decision"] in {
        "PD_OR_CLOSED_LOOP_PRIMARY", "REFERENCE_GEOMETRY_CONTACT_PRIMARY", "MIXED_OR_UNRESOLVED"
    }
    assert report["truth_boundary"]["kinematic_oracle_not_executable_controller"] is True
