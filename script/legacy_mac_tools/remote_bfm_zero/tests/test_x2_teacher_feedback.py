from __future__ import annotations

import pytest
import torch

from humanoidverse.x2_teacher_feedback import (
    FeedbackCommandSpec,
    low_speed_candidates,
    right_turn_candidates,
    state_feedback_command,
)


def _call(desired, measured, *, elapsed=1.0, ramp=1.0, vx_gain=1.0, yaw_gain=1.0):
    desired = torch.tensor(desired, dtype=torch.float32)
    measured = torch.tensor(measured, dtype=torch.float32)
    n = desired.shape[0]
    vector = lambda value: torch.full((n,), value, dtype=torch.float32)
    return state_feedback_command(
        desired,
        measured,
        vector(elapsed),
        vector(ramp),
        vector(vx_gain),
        vector(yaw_gain),
        vector(0.65),
        vector(0.80),
    )


def test_feedback_grid_is_frozen_and_balanced():
    low = low_speed_candidates()
    right = right_turn_candidates()
    assert len(low) == len(right) == 16
    assert len({candidate.name for candidate in (*low, *right)}) == 32
    assert {candidate.role for candidate in low} == {"vx_0p20"}
    assert {candidate.role for candidate in right} == {"turn_right"}


def test_ramp_and_zero_error_contract():
    command = _call([[0.2, 0.0, -0.3]], [[0.1, 0.0, -0.15]], elapsed=0.5)
    torch.testing.assert_close(command, torch.tensor([[0.1, 0.0, -0.15]]))


def test_feedback_is_bounded_and_preserves_yaw_sign():
    command = _call(
        [[0.5, 0.0, -0.3], [0.5, 0.0, 0.3]],
        [[-2.0, 0.0, 2.0], [-2.0, 0.0, -2.0]],
        vx_gain=4.0,
        yaw_gain=4.0,
    )
    torch.testing.assert_close(command[:, 0], torch.tensor([0.65, 0.65]))
    torch.testing.assert_close(command[:, 2], torch.tensor([-0.8, 0.8]))


def test_invalid_spec_and_shapes_fail_closed():
    with pytest.raises(ValueError):
        FeedbackCommandSpec("bad", "role", 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        state_feedback_command(
            torch.zeros(2, 3),
            torch.zeros(2, 2),
            *(torch.ones(2) for _ in range(6)),
        )
