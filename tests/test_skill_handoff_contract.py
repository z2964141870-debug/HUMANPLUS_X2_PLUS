from official_x2.skill_handoff_contract import should_emergency_latch, stop_policy_slot


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
