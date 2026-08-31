from types import SimpleNamespace

import torch

from humanoidverse.x2_response_clone import (
    clone_action_manager_rows,
    clone_delay_buffer_rows,
    clone_tensor_rows,
    max_row_deviation,
)


def test_clone_tensor_rows_preserves_unselected_rows() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expected_last = value[3].clone()
    clone_tensor_rows(value, 1, torch.tensor((0, 2)))
    torch.testing.assert_close(value[0], value[1])
    torch.testing.assert_close(value[2], value[1])
    torch.testing.assert_close(value[3], expected_last)


def test_clone_delay_buffer_rows_copies_history_lag_and_push_count() -> None:
    circular = SimpleNamespace(
        _buffer=torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
        _num_pushes=torch.tensor((2, 1, 2, 1)),
    )
    delay = SimpleNamespace(
        _circular_buffer=circular,
        _history_length=1,
        _time_lags=torch.tensor((1, 0, 1, 0), dtype=torch.int32),
        _min_time_lag=0,
        _max_time_lag=1,
    )
    assert clone_delay_buffer_rows(delay, 0, torch.tensor((1, 2, 3)))
    torch.testing.assert_close(circular._buffer[:, 1:], circular._buffer[:, :1].expand(-1, 3, -1))
    assert circular._num_pushes.tolist() == [2, 2, 2, 2]
    assert delay._time_lags.tolist() == [1, 1, 1, 1]
    assert delay._min_time_lag == delay._max_time_lag == 1


def test_clone_delay_buffer_rows_skips_inactive_zero_delay_buffer() -> None:
    delay = SimpleNamespace(
        _circular_buffer=SimpleNamespace(_buffer=None),
        _history_length=0,
        _time_lags=torch.zeros(4, dtype=torch.int32),
    )
    assert not clone_delay_buffer_rows(delay, 0, torch.tensor((1, 2, 3)))


def test_clone_action_manager_rows_copies_manager_and_term_state() -> None:
    term = SimpleNamespace(
        _raw_actions=torch.randn(4, 2),
        _processed_actions=torch.randn(4, 2),
    )
    manager = SimpleNamespace(
        _action=torch.randn(4, 2),
        _prev_action=torch.randn(4, 2),
        _terms={"joint_pos": term},
    )
    result = clone_action_manager_rows(manager, 0, torch.tensor((1, 2, 3)))
    assert result["manager"] == ["_action", "_prev_action"]
    assert result["joint_pos"] == ["_raw_actions", "_processed_actions"]
    assert max_row_deviation(manager._action) == 0.0
    assert max_row_deviation(term._processed_actions) == 0.0
