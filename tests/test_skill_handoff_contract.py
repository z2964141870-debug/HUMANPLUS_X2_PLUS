import pytest

from official_x2.skill_handoff_contract import (
    blend_curriculum_recovery_targets,
    curriculum_stop_policy_slot,
    matched_event_speed,
    normalized_action_from_physical_targets,
    should_emergency_latch,
    stop_policy_slot,
)


def test_matched_event_speed_hits_training_boundaries() -> None:
    common = dict(
        cruise_speed_mps=0.3,
        accelerate_s=1.0,
        cruise_s=4.2,
        decelerate_s=2.0,
    )
    assert matched_event_speed(elapsed_s=0.0, **common) == 0.0
    assert matched_event_speed(elapsed_s=0.5, **common) == pytest.approx(0.15)
    assert matched_event_speed(elapsed_s=1.0, **common) == pytest.approx(0.3)
    assert matched_event_speed(elapsed_s=5.2, **common) == pytest.approx(0.3)
    assert matched_event_speed(elapsed_s=6.2, **common) == pytest.approx(0.15)
    assert matched_event_speed(elapsed_s=7.2, **common) == 0.0
    assert matched_event_speed(elapsed_s=8.0, **common) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accelerate_s": 0.0, "cruise_s": 1.0, "decelerate_s": 1.0},
        {"accelerate_s": 1.0, "cruise_s": -0.1, "decelerate_s": 1.0},
        {"accelerate_s": 1.0, "cruise_s": 1.0, "decelerate_s": 0.0},
    ],
)
def test_matched_event_speed_rejects_invalid_durations(kwargs) -> None:
    with pytest.raises(ValueError):
        matched_event_speed(elapsed_s=0.0, cruise_speed_mps=0.3, **kwargs)


def test_stop_policy_slot_separates_nominal_stand_and_recovery() -> None:
    assert stop_policy_slot(None) == "stationary"
    assert stop_policy_slot("") == "stationary"
    assert stop_policy_slot("recovery.onnx") == "recovery"


def test_curriculum_stop_role_is_exactly_old_route_without_recovery() -> None:
    assert curriculum_stop_policy_slot(
        stop_elapsed_s=1.999, transition_s=2.0, recovery_model=None
    ) == "main"
    assert curriculum_stop_policy_slot(
        stop_elapsed_s=2.0, transition_s=2.0, recovery_model=None
    ) == "stationary"
    assert curriculum_stop_policy_slot(
        stop_elapsed_s=8.0, transition_s=2.0, recovery_model=""
    ) == "stationary"


def test_curriculum_stop_recovery_gets_authority_only_after_handoff() -> None:
    slots = [
        curriculum_stop_policy_slot(
            stop_elapsed_s=step * 0.02,
            transition_s=2.0,
            recovery_model="f005.onnx",
        )
        for step in range(400)
    ]
    assert slots[:100] == ["main"] * 100
    assert slots[100:] == ["recovery"] * 300
    assert "stationary" not in slots


def test_zero_duration_curriculum_target_blend_is_exact_old_target() -> None:
    handoff = {"left_knee": 0.2, "right_knee": -0.1}
    recovery = {"left_knee": 0.7, "right_knee": 0.4}
    assert blend_curriculum_recovery_targets(
        handoff,
        recovery,
        elapsed_after_handoff_s=0.0,
        blend_seconds=0.0,
    ) == recovery


def test_c2_curriculum_target_blend_is_continuous_and_bounded() -> None:
    handoff = {"joint": 0.0}
    recovery = {"joint": 1.0}
    samples = [
        blend_curriculum_recovery_targets(
            handoff,
            recovery,
            elapsed_after_handoff_s=step * 0.02,
            blend_seconds=0.5,
        )["joint"]
        for step in range(26)
    ]
    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    assert samples[12] < 0.5 < samples[13]
    assert samples == sorted(samples)
    assert max(right - left for left, right in zip(samples, samples[1:])) <= 0.075


def test_curriculum_target_blend_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        blend_curriculum_recovery_targets(
            {"joint": 0.0}, {"joint": 1.0},
            elapsed_after_handoff_s=-0.02, blend_seconds=0.5,
        )


def test_physical_recovery_target_roundtrips_to_actual_normalized_action() -> None:
    names = ("left_knee", "right_knee")
    defaults = {"left_knee": 0.5, "right_knee": 0.5}
    scales = (0.4, 0.4)
    handoff = {"left_knee": 0.3, "right_knee": 0.7}
    proposal = {"left_knee": 0.9, "right_knee": 0.1}
    targets = blend_curriculum_recovery_targets(
        handoff,
        proposal,
        elapsed_after_handoff_s=0.2,
        blend_seconds=0.5,
    )
    actual = normalized_action_from_physical_targets(
        targets, defaults, names, scales
    )
    reconstructed = {
        name: defaults[name] + value * scale
        for name, value, scale in zip(names, actual, scales)
    }
    assert reconstructed == pytest.approx(targets)
    # This is the value the next recovery observation must receive.
    next_previous_action = list(actual)
    assert next_previous_action == pytest.approx(actual)


def test_physical_target_inverse_fails_closed_outside_action_contract() -> None:
    with pytest.raises(ValueError, match="outside normalized action bounds"):
        normalized_action_from_physical_targets(
            {"joint": 2.0}, {"joint": 0.0}, ("joint",), (0.4,)
        )
    with pytest.raises(ValueError, match="do not match"):
        blend_curriculum_recovery_targets(
            {"left": 0.0}, {"right": 1.0},
            elapsed_after_handoff_s=0.0, blend_seconds=0.5,
        )


def test_emergency_latch_is_disabled_by_default() -> None:
    assert not should_emergency_latch(
        stop_elapsed_s=2.0,
        speed_mps=0.01,
        tilt_rad=0.40,
        tilt_threshold_rad=None,
        speed_max_mps=0.10,
        min_elapsed_s=0.80,
    )


def test_emergency_latch_requires_all_preregistered_conditions() -> None:
    common = dict(
        tilt_threshold_rad=0.15,
        speed_max_mps=0.10,
        min_elapsed_s=0.80,
    )
    assert should_emergency_latch(
        stop_elapsed_s=0.80, speed_mps=0.10, tilt_rad=0.15, **common
    )
    assert not should_emergency_latch(
        stop_elapsed_s=0.79, speed_mps=0.10, tilt_rad=0.15, **common
    )
    assert not should_emergency_latch(
        stop_elapsed_s=0.80, speed_mps=0.11, tilt_rad=0.15, **common
    )
    assert not should_emergency_latch(
        stop_elapsed_s=0.80, speed_mps=0.10, tilt_rad=0.14, **common
    )
