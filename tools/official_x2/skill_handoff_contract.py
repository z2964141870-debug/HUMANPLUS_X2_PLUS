"""Pure contracts for official-X2 stand/recovery skill handoff."""

from __future__ import annotations

import math


def matched_event_speed(
    *,
    elapsed_s: float,
    cruise_speed_mps: float,
    accelerate_s: float,
    cruise_s: float,
    decelerate_s: float,
) -> float:
    """Scalar C1 start/cruise/stop schedule shared with the training event.

    The schedule is intentionally a pure function so the official AimDK
    adapter can be checked without ROS or MuJoCo.  It mirrors
    ``smooth_transition_speed`` from the IsaacLab command term.
    """

    values = (elapsed_s, cruise_speed_mps, accelerate_s, cruise_s, decelerate_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("matched event values must be finite")
    if accelerate_s <= 0.0 or decelerate_s <= 0.0 or cruise_s < 0.0:
        raise ValueError("matched event requires positive ramps and non-negative cruise")

    def smoothstep(value: float) -> float:
        clipped = min(max(value, 0.0), 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    acceleration = smoothstep(elapsed_s / accelerate_s)
    deceleration_start = accelerate_s + cruise_s
    deceleration = 1.0 - smoothstep((elapsed_s - deceleration_start) / decelerate_s)
    return cruise_speed_mps * acceleration * deceleration


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
