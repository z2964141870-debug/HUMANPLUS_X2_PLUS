"""Clone the dynamic state needed for parallel X2 response-domain rollouts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def clone_tensor_rows(
    tensor: torch.Tensor,
    source_id: int,
    target_ids: torch.Tensor,
) -> None:
    """Copy one batch row into selected rows without changing other rows."""

    if tensor.ndim < 1:
        raise ValueError("clone target must have a batch dimension")
    if not 0 <= source_id < tensor.shape[0]:
        raise IndexError("source_id is outside the batch")
    if target_ids.ndim != 1 or target_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_ids must be a one-dimensional integer tensor")
    if target_ids.numel() == 0:
        return
    tensor[target_ids] = tensor[source_id].clone()


def clone_delay_buffer_rows(
    delay_buffer: Any,
    source_id: int,
    target_ids: torch.Tensor,
) -> bool:
    """Clone per-environment state from an IsaacLab ``DelayBuffer``."""

    circular = delay_buffer._circular_buffer
    if circular._buffer is None:
        # X2 actuator groups whose configured maximum delay is zero bypass
        # DelayBuffer.compute(), so their circular storage is intentionally
        # never allocated.  Such a buffer has no rollout state to clone.  An
        # uninitialized buffer with non-zero history or lag is unexpected and
        # must still fail closed.
        if delay_buffer._history_length == 0 and not bool(delay_buffer._time_lags.any()):
            return False
        raise RuntimeError("cannot clone an active uninitialized delay buffer")
    clone_tensor_rows(delay_buffer._time_lags, source_id, target_ids)
    # CircularBuffer stores time first and batch second.
    circular._buffer[:, target_ids] = circular._buffer[:, source_id : source_id + 1]
    clone_tensor_rows(circular._num_pushes, source_id, target_ids)
    delay_buffer._min_time_lag = int(delay_buffer._time_lags.min().item())
    delay_buffer._max_time_lag = int(delay_buffer._time_lags.max().item())
    return True


def clone_response_actuator_rows(
    actuator: Any,
    source_id: int,
    target_ids: torch.Tensor,
) -> list[str]:
    """Clone every recognized per-environment response-actuator field."""

    cloned: list[str] = []
    for name in (
        "_position_alpha",
        "_filtered_joint_positions",
        "_position_filter_initialized",
        "_ideal_env_mask",
        "computed_effort",
        "applied_effort",
        "stiffness",
        "damping",
    ):
        value = getattr(actuator, name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > source_id:
            clone_tensor_rows(value, source_id, target_ids)
            cloned.append(name)
    for name in (
        "positions_delay_buffer",
        "velocities_delay_buffer",
        "efforts_delay_buffer",
    ):
        value = getattr(actuator, name, None)
        if value is not None and clone_delay_buffer_rows(value, source_id, target_ids):
            cloned.append(name)
    return cloned


def clone_robot_actuator_rows(
    robot: Any,
    source_id: int,
    target_ids: torch.Tensor,
) -> dict[str, list[str]]:
    """Clone dynamic response state for every actuator group on a robot."""

    result = {}
    for name, actuator in robot.actuators.items():
        fields = clone_response_actuator_rows(actuator, source_id, target_ids)
        if fields:
            result[name] = fields
    return result


def _clone_named_tensor_fields(
    owner: Any,
    field_names: Iterable[str],
    source_id: int,
    target_ids: torch.Tensor,
) -> list[str]:
    cloned = []
    for name in field_names:
        value = getattr(owner, name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > source_id:
            clone_tensor_rows(value, source_id, target_ids)
            cloned.append(name)
    return cloned


def clone_action_manager_rows(
    action_manager: Any,
    source_id: int,
    target_ids: torch.Tensor,
) -> dict[str, list[str]]:
    """Clone action history and term-local action state."""

    result = {
        "manager": _clone_named_tensor_fields(
            action_manager,
            ("_action", "_prev_action"),
            source_id,
            target_ids,
        )
    }
    for term_name, term in action_manager._terms.items():
        fields = _clone_named_tensor_fields(
            term,
            (
                "_raw_actions",
                "_processed_actions",
                "_combined_normalized_actions",
                "_normalized_template_bias",
                "_preclip_combined_actions",
            ),
            source_id,
            target_ids,
        )
        if fields:
            result[term_name] = fields
    return result


def max_row_deviation(tensor: torch.Tensor, source_id: int = 0) -> float:
    """Maximum absolute deviation from one reference row."""

    if tensor.ndim < 1 or tensor.shape[0] < 1:
        raise ValueError("tensor must contain at least one batch row")
    return float((tensor - tensor[source_id : source_id + 1]).abs().max())
