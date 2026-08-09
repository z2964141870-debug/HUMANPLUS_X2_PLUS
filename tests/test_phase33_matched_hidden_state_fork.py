import numpy as np

from official_x2.run_phase33_matched_hidden_state_fork import (
    contact_pairs_from_trace,
    quat_geodesic,
)


def test_contact_pairs_are_order_invariant_and_sorted():
    row = {"contacts": [{"geom1": 9, "geom2": 2}, {"geom1": 1, "geom2": 4}]}
    assert contact_pairs_from_trace(row) == [(1, 4), (2, 9)]


def test_quaternion_sign_is_equivalent():
    q = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert quat_geodesic(q, -q) == 0.0


def test_quaternion_metric_normalizes_inputs():
    assert quat_geodesic(np.asarray([0.999999999, 0.0, 0.0, 0.0]),
                         np.asarray([1.0, 0.0, 0.0, 0.0])) == 0.0
