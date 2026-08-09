from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import retarget.run_x2_native_event_replay_phase15 as phase15


REPO = Path(__file__).resolve().parents[1]


def test_phase15_joint_reorder_is_name_based():
    source = ["b", "a", "c"]
    value = np.asarray([[1.0, 2.0, 3.0]])
    got = phase15._q_by_model_order(value, source, ("a", "b", "c"))
    assert got.tolist() == [[2.0, 1.0, 3.0]]


def test_phase15_reference_and_initial_context_are_active29():
    reference, events = phase15.build_reference(phase15.DEFAULT_SOURCE)
    with np.load(phase15.DEFAULT_SOURCE, allow_pickle=False) as archive:
        reference["source_start_elapsed_s"] = float(archive["time_s"][0])
    context = phase15.initial_event_context(reference, events)
    assert reference["q"].shape == (999, 31)
    assert reference["recorded_command_used_as_reference"] is False
    assert context["events_before_or_at_snapshot0"] == 60
    assert int(context["valid"].sum()) == 29
    names = events["source_names"].astype(str)
    assert not np.any(context["valid"][[name.startswith("head_") for name in names]])


def test_phase15_saved_report_stops_after_prescribed_failure():
    path = REPO / "reports/retarget/x2_native_event_replay_phase15.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["truth_boundary"]["command_is_control_input_only"] is True
    assert report["truth_boundary"]["active29_head_inactive"] is True
    assert report["truth_boundary"]["event_time_is_subscriber_receipt_not_publisher_time"] is True
    if report["gate"]["prescribed"]["pass"] is False:
        assert report["gate"]["free"] is None
        assert set(report["replay"]) == {phase15.phase12.MODE_PRESCRIBED}
