import math

import numpy as np
import pytest

from official_x2.analyze_sagittal_posture import (
    JOINT_POS_OBS_OFFSET,
    LOWER_SCALE_RAD,
    WAIST_PITCH_ISAAC_INDEX,
    WAIST_PITCH_LOWER_INDEX,
    analyze_trace_payload,
    projected_gravity_from_xyzw,
    signed_pitch_from_projected_gravity_rad,
    signed_pitch_from_xyzw_rad,
    split_trace_phases,
)


def test_quaternion_and_projected_gravity_pitch_agree_with_signed_contract():
    pitch = math.radians(-11.0)
    quaternion_xyzw = [0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)]
    gravity = projected_gravity_from_xyzw(quaternion_xyzw)
    assert signed_pitch_from_xyzw_rad(quaternion_xyzw) == pytest.approx(pitch)
    assert signed_pitch_from_projected_gravity_rad(gravity) == pytest.approx(pitch)


def test_phase_split_uses_fixed_startup_window_and_excludes_prepare():
    rows = [
        {"stage": "prepare", "elapsed_s": 0.0},
        {"stage": "stand", "elapsed_s": 0.0},
        {"stage": "move", "elapsed_s": 0.0},
        {"stage": "move", "elapsed_s": 0.98},
        {"stage": "move", "elapsed_s": 1.0},
        {"stage": "stop", "elapsed_s": 0.0},
    ]
    phases = split_trace_phases(rows, startup_seconds=1.0)
    assert [len(phases[name]) for name in ("stand", "start", "move", "stop")] == [1, 2, 1, 1]


def test_trace_audit_keeps_root_actual_waist_and_issued_target_separate():
    pitch = math.radians(-10.0)
    obs = [0.0] * 93
    obs[6:9] = [math.sin(pitch), 0.0, -math.cos(pitch)]
    waist_actual = math.radians(-4.0)
    obs[JOINT_POS_OBS_OFFSET + WAIST_PITCH_ISAAC_INDEX] = waist_actual
    action = [0.0] * 15
    action[WAIST_PITCH_LOWER_INDEX] = -0.5
    issued_target = -0.5 * LOWER_SCALE_RAD[WAIST_PITCH_LOWER_INDEX]
    row = {"stage": "stand", "elapsed_s": 0.0, "obs": obs, "action": action}
    payload = {
        "summary": {"default_pose_profile": "stage208"},
        "trace": [row],
    }
    result = analyze_trace_payload(payload)["phases"]["stand"]
    assert result["root_pelvis_pitch_deg"]["mean"] == pytest.approx(-10.0)
    assert result["waist_pitch_actual_deg"]["mean"] == pytest.approx(math.degrees(waist_actual))
    assert result["waist_pitch_issued_target_deg"]["mean"] == pytest.approx(math.degrees(issued_target))
    assert result["waist_pitch_actual_minus_issued_target_deg"]["mean"] == pytest.approx(
        math.degrees(waist_actual - issued_target)
    )


def test_legacy_trace_start_window_is_preregistered_one_second():
    obs = [0.0] * 93
    obs[8] = -1.0
    action = [0.0] * 15
    payload = {
        "summary": {"default_pose_profile": "stage208"},
        "trace": [
            {"stage": "move", "elapsed_s": 0.98, "obs": obs, "action": action},
            {"stage": "move", "elapsed_s": 1.00, "obs": obs, "action": action},
        ],
    }
    result = analyze_trace_payload(payload)
    assert result["contract"]["startup_seconds"] == 1.0
    assert result["phases"]["start"]["rows_total"] == 1
    assert result["phases"]["move"]["rows_total"] == 1
