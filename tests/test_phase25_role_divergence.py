import numpy as np

from official_x2.audit_phase25_role_divergence import (
    calibrate,
    classify,
    first_stable,
    robust_scale,
)


def test_robust_scale_falls_back_for_constant_features() -> None:
    values = np.ones((8, 3))
    assert np.all(robust_scale(values) == 1.0)


def test_role_calibration_uses_leave_one_out_and_separates_clusters() -> None:
    rng = np.random.default_rng(25)
    success = rng.normal(0.0, 0.1, size=(20, 3))
    critical = rng.normal(3.0, 0.1, size=(20, 3))
    calibration = calibrate(success, critical)
    assert calibration["balanced_accuracy"] > 0.95
    assert classify(np.zeros(3), calibration)["assigned_role"] == "success_safe"
    assert classify(np.full(3, 3.0), calibration)["assigned_role"] == "critical_from_failure"


def test_first_stable_requires_five_consecutive_ticks() -> None:
    series = [{"offset_s": i * 0.02, "critical": i >= 3} for i in range(10)]
    assert first_stable(series, lambda row: row["critical"]) == 0.06
    intermittent = [{"offset_s": i * 0.02, "critical": i % 2 == 0} for i in range(10)]
    assert first_stable(intermittent, lambda row: row["critical"]) is None
