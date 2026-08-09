import argparse

import numpy as np

from official_x2.audit_phase31_prepare_and_ros_timing import (
    DEFAULT_ACTIVATION,
    DEFAULT_ADAPTER,
    DEFAULT_EVENT20,
    DEFAULT_HISTORICAL,
    DEFAULT_RESET,
    _first_after,
    _latest_before,
    analyze_prepare,
    build_report,
    stats_ns,
)


def test_receipt_bracketing_is_causal():
    reference = np.asarray([10, 20, 30], dtype=np.int64)
    query = np.asarray([5, 10, 15, 35], dtype=np.int64)
    previous, has_previous = _latest_before(reference, query)
    following, has_following = _first_after(reference, query)
    assert previous.tolist() == [-1, 0, 0, 2]
    assert has_previous.tolist() == [False, True, True, True]
    assert following.tolist() == [0, 0, 1, 3]
    assert has_following.tolist() == [True, True, True, False]


def test_ns_statistics_are_reported_in_ms():
    result = stats_ns(np.asarray([1_000_000, 2_000_000, 3_000_000]))
    assert result["p50_ms"] == 2.0
    assert result["max_ms"] == 3.0


def test_prepare_rows_do_not_support_unique_reconstruction():
    import json

    result = analyze_prepare(json.loads(DEFAULT_HISTORICAL.read_text(encoding="utf-8")))
    assert result["rows"] == 10
    assert not result["uniquely_reconstructable"]
    assert all(count == 0 for count in result["field_presence_rows"].values())


def test_phase31_forbids_fabricated_offline_replay():
    args = argparse.Namespace(
        historical=DEFAULT_HISTORICAL,
        event20=DEFAULT_EVENT20,
        reset=DEFAULT_RESET,
        activation=DEFAULT_ACTIVATION,
        adapter=DEFAULT_ADAPTER,
    )
    report = build_report(args)
    decision = report["offline_replay_decision"]
    assert not decision["exact_stage250_prepare_replay_possible"]
    assert not decision["exact_command_application_timing_replay_possible"]
    assert not decision["new_physics_probe_executed"]
    assert decision["status"] == "BLOCKED_BY_UNRECORDED_PREPARE_AND_APPLICATION_TIMING"
