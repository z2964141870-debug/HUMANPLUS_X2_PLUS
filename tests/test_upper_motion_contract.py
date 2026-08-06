from __future__ import annotations

import numpy as np

from cwi_x2.upper_motion_contract import (
    UpperMotionClip,
    UpperSafetyLimits,
    bounded_upper_target_step,
    relative_upper_target,
    sample_linear,
    wrapped_angle_delta_rad,
)


def test_sample_linear_clamps_and_interpolates(tmp_path):
    q = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    out = sample_linear(q, 2.0, np.array([-1.0, 0.25, 1.5]))
    np.testing.assert_allclose(out, [[0.0, 0.0], [0.5, 1.0], [2.0, 4.0]])


def test_relative_target_is_exact_default_at_start(tmp_path):
    clip = UpperMotionClip(
        path=tmp_path / "fake.pkl",
        key="fake",
        fps=10.0,
        joint_names=("a", "b"),
        q_rad=np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float32),
    )
    default = np.array([0.25, -0.5], dtype=np.float32)
    target = relative_upper_target(clip, default, 0.0, start_s=0.1, scale=0.75)
    np.testing.assert_array_equal(target, default)


def test_relative_target_scale_and_limits(tmp_path):
    clip = UpperMotionClip(
        path=tmp_path / "fake.pkl",
        key="fake",
        fps=1.0,
        joint_names=("a", "b"),
        q_rad=np.array([[0.0, 0.0], [2.0, -2.0]], dtype=np.float32),
    )
    target = relative_upper_target(
        clip,
        np.zeros(2, dtype=np.float32),
        1.0,
        start_s=0.0,
        scale=0.5,
        lower_limits_rad=np.array([-0.5, -0.5]),
        upper_limits_rad=np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(target, [0.5, -0.5])


def test_relative_target_time_scale_slows_reference_clock(tmp_path):
    clip = UpperMotionClip(
        path=tmp_path / "fake.pkl",
        key="fake",
        fps=1.0,
        joint_names=("a",),
        q_rad=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
    )
    target = relative_upper_target(
        clip,
        np.zeros(1, dtype=np.float32),
        2.0,
        start_s=0.0,
        scale=1.0,
        time_scale=0.5,
    )
    np.testing.assert_allclose(target, [1.0])


def test_relative_target_rejects_nonpositive_time_scale(tmp_path):
    clip = UpperMotionClip(
        path=tmp_path / "fake.pkl",
        key="fake",
        fps=1.0,
        joint_names=("a",),
        q_rad=np.array([[0.0], [1.0]], dtype=np.float32),
    )
    with np.testing.assert_raises_regex(ValueError, "time scale"):
        relative_upper_target(
            clip,
            np.zeros(1, dtype=np.float32),
            0.0,
            start_s=0.0,
            scale=1.0,
            time_scale=0.0,
        )


def test_bounded_step_applies_excursion_and_velocity_limits():
    limits = UpperSafetyLimits(
        max_excursion_rad=0.12,
        max_velocity_radps=0.20,
        tilt_fallback_rad=0.35,
        height_fallback_m=0.58,
    )
    out = bounded_upper_target_step(
        desired_q_rad=np.array([0.5, -0.5], dtype=np.float32),
        default_q_rad=np.zeros(2, dtype=np.float32),
        previous_q_rad=np.zeros(2, dtype=np.float32),
        dt_s=0.02,
        limits=limits,
    )
    np.testing.assert_allclose(out, [0.004, -0.004], atol=1.0e-7)


def test_bounded_step_falls_back_toward_default_when_unhealthy():
    limits = UpperSafetyLimits(
        max_excursion_rad=0.12,
        max_velocity_radps=0.20,
        tilt_fallback_rad=0.35,
        height_fallback_m=0.58,
    )
    out = bounded_upper_target_step(
        desired_q_rad=np.array([0.1, -0.1], dtype=np.float32),
        default_q_rad=np.zeros(2, dtype=np.float32),
        previous_q_rad=np.array([0.05, -0.05], dtype=np.float32),
        dt_s=0.02,
        limits=limits,
        healthy=False,
    )
    np.testing.assert_allclose(out, [0.046, -0.046], atol=1.0e-7)


def test_bounded_step_supports_per_environment_health_mask():
    limits = UpperSafetyLimits(0.12, 0.20, 0.35, 0.58)
    desired = np.full((2, 2), 0.1, dtype=np.float32)
    previous = np.zeros_like(desired)
    out = bounded_upper_target_step(
        desired,
        np.zeros_like(desired),
        previous,
        dt_s=0.02,
        limits=limits,
        healthy=np.array([True, False]),
    )
    np.testing.assert_allclose(out[0], [0.004, 0.004], atol=1.0e-7)
    np.testing.assert_array_equal(out[1], [0.0, 0.0])


def test_wrapped_angle_delta_uses_shortest_arc():
    delta = wrapped_angle_delta_rad(
        np.array([-np.pi + 0.1, np.pi - 0.1]),
        np.array([np.pi - 0.1, -np.pi + 0.1]),
    )
    np.testing.assert_allclose(delta, [0.2, -0.2], atol=1.0e-6)
