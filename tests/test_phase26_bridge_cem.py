import numpy as np

from official_x2.run_phase26_testonly_bridge_cem import (
    HIGH,
    LOW,
    MODE_NAMES,
    MODES,
    residual_at,
    smoothstep,
)


def test_bridge_modes_are_low_dimensional_and_lower_body_only() -> None:
    assert MODES.shape == (len(MODE_NAMES), 15)
    assert len(MODE_NAMES) == 5
    assert np.allclose(np.linalg.norm(MODES, axis=1), 1.0)


def test_residual_knots_are_bounded_and_deterministic() -> None:
    raw = np.linspace(-1.0, 1.0, len(LOW))
    start = residual_at(raw, 0.0)
    end = residual_at(raw, 1.0)
    assert np.array_equal(start, residual_at(raw, 0.0))
    assert np.max(np.abs(start)) <= np.linalg.norm(np.clip(raw, LOW, HIGH)[: len(MODE_NAMES)]) + 1e-6
    assert np.max(np.abs(end)) <= np.linalg.norm(np.clip(raw, LOW, HIGH)[len(MODE_NAMES) :]) + 1e-6


def test_smoothstep_has_exact_endpoints() -> None:
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == 0.5
