from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import retarget.audit_x2_native_control_contract_phase13 as phase13


REPO = Path(__file__).resolve().parents[1]


def test_field_metrics_rejects_missing_or_wrong_shape_arrays():
    class Archive(dict):
        @property
        def files(self):
            return list(self)

    complete = Archive({
        field: np.zeros((3, 2), dtype=bool if field == "command_valid" else np.float32)
        for field in phase13.COMMAND_FIELDS
    })
    result = phase13.field_metrics(complete, 3, 2)
    assert all(value["present"] and value["shape_exact"] and value["finite"] for value in result.values())
    complete["command_kd"] = np.zeros((2, 2), dtype=np.float32)
    assert phase13.field_metrics(complete, 3, 2)["command_kd"]["shape_exact"] is False


def test_phase13_stops_before_physics_on_incomplete_event_contract():
    report = json.loads(
        (REPO / "reports/retarget/x2_native_control_contract_phase13.json").read_text(
            encoding="utf-8"
        )
    )
    checks = report["contract_gate"]["checks"]
    assert checks["all_command_arrays_complete"] is True
    assert checks["all_joint_commands_valid"] is True
    assert checks["archive_joint_order_exact_recorder"] is True
    assert checks["values_are_sampled_from_official_command_topics"] is True
    assert checks["official_publisher_emits_control_fields"] is True
    assert checks["original_command_event_timestamp_preserved"] is False
    assert checks["original_command_sequence_preserved"] is False
    assert checks["four_group_atomic_cycle_preserved"] is False
    assert checks["source_policy_and_config_identity_hashes_preserved"] is False
    assert report["contract_gate"]["pass"] is False
    assert report["truth_boundary"]["physics_prescribed_free_ab_executed"] is False
    assert report["decision"]["status"] == "PHASE13_CONTROL_CONTRACT_INCOMPLETE_STOP"


def test_phase13_does_not_promote_command_to_reference():
    report = json.loads(
        (REPO / "reports/retarget/x2_native_control_contract_phase13.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["truth_boundary"]["command_is_control_input_candidate_not_reference"] is True
    event = report["audit"]["event_contract"]
    assert event["official_publisher_hz"] == 500
    assert event["recorder_snapshot_hz"] == 50
    assert event["latest_value_cache"] is True
    assert event["message_header_defines_stamp_and_sequence"] is True
    assert event["official_createCommand_populates_header"] is False
    assert event["recorder_saves_header_or_callback_receipt_time"] is False
