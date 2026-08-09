#!/usr/bin/env python3
"""Lossless controller-state contract for deterministic official X2 suffix replay.

This module deliberately does not reset MuJoCo or run a policy.  It captures
and restores only the mutable state that affects the next Stage208 controller
decision.  A valid suffix replay additionally needs a matched physical-state
snapshot; the physical snapshot hash is therefore mandatory and the helper
fails closed when it is absent.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "aimdk_x2_stage208_controller_state_v1"
ACTION_SLOTS = ("main", "stationary", "recovery")
ARRAY_FIELDS = {
    "lateral_recovery_bias": 2,
    "upper_previous_target": None,
    "upper_last_target": None,
    "previous_physical_observation": 71,
    "predicted_physical_observation": 71,
}
SCALAR_FIELDS = (
    "sequence_step",
    "control_steps",
    "stop_hold_latch_s",
    "stop_emergency_latch",
    "heading_target_rad",
    "move_heading_initialized",
    "stop_policy_initialized",
    "lateral_recovery_state",
    "heading_recovery_active",
    "heading_action_recovery_active",
    "heading_action_recovery_steps",
    "predicted_physical_step",
    "upper_fallback_steps",
    "upper_fallback_active",
    "upper_fallback_first_step",
    "finished",
)
MAPPING_FIELDS = (
    "last_move_targets",
    "stop_hold_targets",
    "prepare_start_q",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_attribute(controller: Any, name: str) -> Any:
    if not hasattr(controller, name):
        raise ValueError(f"controller lacks required mutable state: {name}")
    return getattr(controller, name)


def capture_controller_state(
    controller: Any,
    *,
    physical_state_sha256: str,
) -> dict[str, Any]:
    """Capture every mutable value needed before the next matched-event tick."""
    if not physical_state_sha256 or len(physical_state_sha256) != 64:
        raise ValueError("a 64-character physical_state_sha256 is required")
    args = _require_attribute(controller, "args")
    if getattr(args, "clock_mode", None) != "step":
        raise ValueError("deterministic suffix replay requires clock_mode='step'")

    state: dict[str, Any] = {
        "schema": SCHEMA,
        "physical_state_sha256": physical_state_sha256,
        "runtime_args_sha256": _sha256_json(vars(args)),
        "clock_mode": "step",
        "previous_actions": {
            slot: np.asarray(_require_attribute(controller, "previous_actions")[slot], dtype=np.float32).tolist()
            for slot in ACTION_SLOTS
        },
        "issued_actions": {
            slot: np.asarray(_require_attribute(controller, "issued_actions")[slot], dtype=np.float32).tolist()
            for slot in ACTION_SLOTS
        },
        "heading_origin_xy": _jsonable(_require_attribute(controller, "heading_origin_xy")),
    }
    for name in SCALAR_FIELDS:
        state[name] = _jsonable(_require_attribute(controller, name))
    for name in MAPPING_FIELDS:
        state[name] = _jsonable(_require_attribute(controller, name))
    for name in ARRAY_FIELDS:
        value = _require_attribute(controller, name)
        state[name] = None if value is None else np.asarray(value, dtype=np.float32).tolist()
    validate_controller_state(state)
    return state


def validate_controller_state(state: Mapping[str, Any]) -> None:
    """Validate shape, finiteness and fail-closed replay prerequisites."""
    if state.get("schema") != SCHEMA:
        raise ValueError(f"unsupported controller-state schema: {state.get('schema')!r}")
    if state.get("clock_mode") != "step":
        raise ValueError("controller state is not on the deterministic step clock")
    for digest_name in ("physical_state_sha256", "runtime_args_sha256"):
        digest = state.get(digest_name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid {digest_name}")
    for action_field in ("previous_actions", "issued_actions"):
        slots = state.get(action_field)
        if not isinstance(slots, Mapping) or set(slots) != set(ACTION_SLOTS):
            raise ValueError(f"invalid {action_field} slots")
        for slot in ACTION_SLOTS:
            value = np.asarray(slots[slot], dtype=np.float64)
            if value.shape != (15,) or not np.isfinite(value).all():
                raise ValueError(f"invalid {action_field}.{slot}")
    heading_origin = state.get("heading_origin_xy")
    if heading_origin is not None:
        value = np.asarray(heading_origin, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("invalid heading_origin_xy")
    if state.get("lateral_recovery_state") not in {"off", "left", "right"}:
        raise ValueError("invalid lateral_recovery_state")
    for name, expected_size in ARRAY_FIELDS.items():
        value = state.get(name)
        if value is None:
            if name in {"lateral_recovery_bias", "upper_previous_target", "upper_last_target"}:
                raise ValueError(f"required array {name} is missing")
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"invalid array {name}")
        if expected_size is not None and array.shape != (expected_size,):
            raise ValueError(f"invalid {name} shape: {array.shape}")
    for name in ("sequence_step", "control_steps", "heading_action_recovery_steps"):
        if not isinstance(state.get(name), int) or state[name] < 0:
            raise ValueError(f"invalid non-negative counter {name}")


def restore_controller_state(
    controller: Any,
    state: Mapping[str, Any],
    *,
    expected_physical_state_sha256: str,
) -> None:
    """Restore a validated snapshot; refuse config or physical-state mismatch."""
    validate_controller_state(state)
    if state["physical_state_sha256"] != expected_physical_state_sha256:
        raise ValueError("physical snapshot hash does not match controller snapshot")
    current_args_sha = _sha256_json(vars(_require_attribute(controller, "args")))
    if state["runtime_args_sha256"] != current_args_sha:
        raise ValueError("runtime controller arguments do not match snapshot")

    controller.previous_actions = {
        slot: np.asarray(state["previous_actions"][slot], dtype=np.float32).copy()
        for slot in ACTION_SLOTS
    }
    controller.issued_actions = {
        slot: np.asarray(state["issued_actions"][slot], dtype=np.float32).copy()
        for slot in ACTION_SLOTS
    }
    controller.heading_origin_xy = (
        None
        if state["heading_origin_xy"] is None
        else tuple(float(value) for value in state["heading_origin_xy"])
    )
    for name in SCALAR_FIELDS:
        setattr(controller, name, deepcopy(state[name]))
    for name in MAPPING_FIELDS:
        setattr(controller, name, deepcopy(state[name]))
    for name in ARRAY_FIELDS:
        value = state[name]
        setattr(
            controller,
            name,
            None if value is None else np.asarray(value, dtype=np.float32).copy(),
        )


def audit_trace_controller_state(path: str | Path) -> dict[str, Any]:
    """Report whether every recorded row can support an honest suffix restore."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = payload.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("official trace is missing or empty")
    missing_rows: list[int] = []
    invalid_rows: dict[int, str] = {}
    for index, row in enumerate(trace):
        state = row.get("controller_state") if isinstance(row, Mapping) else None
        if state is None:
            missing_rows.append(index)
            continue
        try:
            validate_controller_state(state)
        except ValueError as error:
            invalid_rows[index] = str(error)
    return {
        "path": str(path),
        "row_count": len(trace),
        "valid_controller_state_rows": len(trace) - len(missing_rows) - len(invalid_rows),
        "missing_controller_state_rows": len(missing_rows),
        "invalid_controller_state_rows": len(invalid_rows),
        "first_missing_rows": missing_rows[:10],
        "first_invalid_rows": dict(list(invalid_rows.items())[:10]),
        "suffix_replay_ready": not missing_rows and not invalid_rows,
    }


__all__ = [
    "ACTION_SLOTS",
    "SCHEMA",
    "audit_trace_controller_state",
    "capture_controller_state",
    "restore_controller_state",
    "validate_controller_state",
]
