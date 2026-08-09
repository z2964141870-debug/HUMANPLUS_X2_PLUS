"""Pure contracts for official-X2 stand/recovery skill handoff."""

from __future__ import annotations


def stop_policy_slot(recovery_model: str | None) -> str:
    """Keep nominal stand and post-brake recovery ownership explicit."""

    return "recovery" if recovery_model else "stationary"


def should_emergency_latch(
    *,
    stop_elapsed_s: float,
    speed_mps: float,
    tilt_rad: float,
    tilt_threshold_rad: float | None,
    speed_max_mps: float,
    min_elapsed_s: float,
) -> bool:
    """Return the one-way emergency-latch predicate; ``None`` is strict off."""

    return bool(
        tilt_threshold_rad is not None
        and stop_elapsed_s >= min_elapsed_s
        and speed_mps <= speed_max_mps
        and tilt_rad >= tilt_threshold_rad
    )
