"""Pure contracts for official-X2 stand/recovery skill handoff."""

from __future__ import annotations

from typing import Mapping, Sequence

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


def curriculum_stop_policy_slot(
    *,
    stop_elapsed_s: float,
    transition_s: float,
    recovery_model: str | None,
) -> str:
    """Owner of a matched curriculum-stop tick.

    The moving actor owns the complete positive-to-zero transition.  Only
    after that event may a dedicated recovery actor take authority.  With no
    recovery model this is exactly the historical main -> stationary route.
    """
    if stop_elapsed_s < 0.0 or transition_s <= 0.0:
        raise ValueError("curriculum stop requires non-negative time and positive transition")
    if stop_elapsed_s < transition_s:
        return "main"
    return stop_policy_slot(recovery_model)


def blend_curriculum_recovery_targets(
    handoff_targets: Mapping[str, float],
    recovery_targets: Mapping[str, float],
    *,
    elapsed_after_handoff_s: float,
    blend_seconds: float,
) -> dict[str, float]:
    """C2 physical-target blend without touching policy recurrent state.

    A zero duration is an exact copy of the historical recovery targets.  For
    a positive duration, the first recovery tick repeats the last actually
    issued transition target and a quintic smoothstep reaches the live
    recovery target at the requested endpoint.
    """
    if elapsed_after_handoff_s < 0.0 or blend_seconds < 0.0:
        raise ValueError("handoff elapsed time and blend duration must be non-negative")
    if set(handoff_targets) != set(recovery_targets):
        raise ValueError("handoff and recovery target joints do not match")
    if blend_seconds == 0.0:
        return {name: float(value) for name, value in recovery_targets.items()}
    x = min(1.0, elapsed_after_handoff_s / blend_seconds)
    alpha = x * x * x * (10.0 + x * (-15.0 + 6.0 * x))
    return {
        name: float(handoff_targets[name])
        + alpha * (float(recovery_targets[name]) - float(handoff_targets[name]))
        for name in recovery_targets
    }


def normalized_action_from_physical_targets(
    targets: Mapping[str, float],
    defaults: Mapping[str, float],
    joint_names: Sequence[str],
    action_scale_rad: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> list[float]:
    """Invert a no-template position target into the action actually issued.

    This is valid for the stationary/recovery branch where the gait template
    is zero.  It keeps next-step last_action synchronized with the smoothed
    physical target instead of feeding the actor an unexecuted proposal.
    """
    if len(joint_names) != len(action_scale_rad):
        raise ValueError("joint names and action scales must have equal length")
    missing = [name for name in joint_names if name not in targets or name not in defaults]
    if missing:
        raise ValueError(f"missing physical-target contract joints: {missing}")
    action = []
    for name, scale in zip(joint_names, action_scale_rad):
        if scale <= 0.0:
            raise ValueError("action scales must be positive")
        value = (float(targets[name]) - float(defaults[name])) / float(scale)
        if value < -1.0 - tolerance or value > 1.0 + tolerance:
            raise ValueError(f"physical target for {name} is outside normalized action bounds")
        action.append(max(-1.0, min(1.0, value)))
    return action


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
