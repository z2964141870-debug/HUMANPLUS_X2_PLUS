import pytest

from official_x2.skill_handoff_contract import (
    matched_event_speed,
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
