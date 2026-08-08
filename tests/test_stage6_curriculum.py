from __future__ import annotations

import pytest

from scripts.train_stage6_future_intent import (
    gain_randomization_range,
    optional_positive_float,
)


def test_gain_randomization_is_opt_in(monkeypatch):
    monkeypatch.delenv("CWI_STAGE6_GAIN_MIN", raising=False)
    monkeypatch.delenv("CWI_STAGE6_GAIN_MAX", raising=False)
    assert gain_randomization_range() is None


def test_gain_randomization_parses_joint_range(monkeypatch):
    monkeypatch.setenv("CWI_STAGE6_GAIN_MIN", "0.90")
    monkeypatch.setenv("CWI_STAGE6_GAIN_MAX", "1.20")
    assert gain_randomization_range() == (0.9, 1.2)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [("0.9", None), (None, "1.2"), ("0", "1.2"), ("1.2", "0.9")],
)
def test_gain_randomization_rejects_partial_or_invalid_ranges(
    monkeypatch, lower, upper
):
    monkeypatch.delenv("CWI_STAGE6_GAIN_MIN", raising=False)
    monkeypatch.delenv("CWI_STAGE6_GAIN_MAX", raising=False)
    if lower is not None:
        monkeypatch.setenv("CWI_STAGE6_GAIN_MIN", lower)
    if upper is not None:
        monkeypatch.setenv("CWI_STAGE6_GAIN_MAX", upper)
    with pytest.raises(ValueError):
        gain_randomization_range()


def test_optional_positive_float(monkeypatch):
    monkeypatch.delenv("CWI_STAGE6_LEARNING_RATE", raising=False)
    assert optional_positive_float("CWI_STAGE6_LEARNING_RATE") is None
    monkeypatch.setenv("CWI_STAGE6_LEARNING_RATE", "0.0001")
    assert optional_positive_float("CWI_STAGE6_LEARNING_RATE") == pytest.approx(1.0e-4)
    monkeypatch.setenv("CWI_STAGE6_LEARNING_RATE", "0")
    with pytest.raises(ValueError, match="must be positive"):
        optional_positive_float("CWI_STAGE6_LEARNING_RATE")
