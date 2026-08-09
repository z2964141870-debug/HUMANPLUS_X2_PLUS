from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from official_x2.official_x2_rollout_recorder_v2 import (
    GROUPS,
    GROUP_BY_JOINT,
    JOINTS,
    ReplayBuffer,
    fake_header,
    fake_joint,
    make_dry_buffer,
    write_outputs,
    write_readiness_diagnostics,
    DEFAULT_CONTROL,
    DEFAULT_ONNX,
    DEFAULT_SCENE,
)


REPO = Path(__file__).resolve().parents[1]
RECORDER = REPO / "tools/official_x2/official_x2_rollout_recorder_v2.py"
OUTER = REPO / "tools/official_x2/run_official_native_capture_v2.sh"
INNER = REPO / "tools/official_x2/run_official_native_capture_v2_inner.sh"
PHASE17 = REPO / "tools/official_x2/run_official_native_reset_prefix_capture_phase17.sh"
PHASE20_OUTER = REPO / "tools/official_x2/run_official_native_activation_prefix_phase20.sh"
PHASE20_INNER = REPO / "tools/official_x2/run_official_native_activation_prefix_phase20_inner.sh"
PHASE21_OUTER = REPO / "tools/official_x2/run_official_native_activation_prefix_phase21.sh"


def message_for(group: str, *, sequence: int = 0):
    return SimpleNamespace(
        header=fake_header(sequence),
        joints=[fake_joint(name) for name in JOINTS if GROUP_BY_JOINT[name] == group],
    )


def test_event_stream_records_receipt_header_group_indices_and_full31_map():
    buffer = ReplayBuffer(start_monotonic_ns=100)
    buffer.record_command("leg", message_for("leg", sequence=7), 120)
    buffer.record_command("arm", message_for("arm"), 130)
    buffer.record_command("leg", message_for("leg", sequence=8), 140)
    arrays = buffer.arrays()
    assert arrays["command_event_global_index"].tolist() == [0, 1, 2]
    assert arrays["command_event_group"].tolist() == ["leg", "arm", "leg"]
    assert arrays["command_event_group_index"].tolist() == [0, 0, 1]
    assert arrays["command_event_receipt_monotonic_ns"].tolist() == [120, 130, 140]
    assert arrays["command_event_elapsed_ns"].tolist() == [20, 30, 40]
    assert arrays["command_event_header_available"].tolist() == [True, True, True]
    assert arrays["command_event_header_populated"].tolist() == [True, False, True]
    assert arrays["command_event_header_sequence"].tolist() == [7, 0, 8]
    assert arrays["command_event_q_rad"].shape == (3, 31)
    assert arrays["command_event_valid"].shape == (3, 31)
    assert int(arrays["command_event_changed_joint_mask"][0].sum()) == 12
    assert int(arrays["command_event_changed_joint_mask"][1].sum()) == 14
    # Event 1 includes the latest leg context plus the newly changed arm map.
    assert int(arrays["command_event_valid"][1].sum()) == 26


def test_mode_events_preserve_buttons_receipt_header_and_inferred_sequence():
    buffer = ReplayBuffer(start_monotonic_ns=100)
    joint = SimpleNamespace(header=fake_header(3), buttons=[0, 0, 1, 0], axes=[])
    rl = SimpleNamespace(header=fake_header(4), buttons=[0, 0, 0, 1], axes=[0.25])
    buffer.record_mode_event(joint, 120)
    buffer.record_mode_event(rl, 140)
    arrays = buffer.arrays()
    assert arrays["mode_event_global_index"].tolist() == [0, 1]
    assert arrays["mode_event_elapsed_ns"].tolist() == [20, 40]
    assert arrays["mode_event_buttons"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 1]]
    assert arrays["mode_event_axis_count"].tolist() == [0, 1]
    assert arrays["mode_event_inferred_mode"].tolist() == ["JOINT_DEFAULT", "RL_DEFAULT"]
    assert arrays["mode_event_header_sequence"].tolist() == [3, 4]


def test_dry_schema_preserves_legacy_snapshot_and_new_receipt_fields(tmp_path):
    buffer = make_dry_buffer()
    output = tmp_path / "dry.npz"
    manifest = write_outputs(
        buffer,
        output,
        ros_domain_id=214,
        control_mode="RL_DEFAULT",
        launch_identity="pytest-dry",
        onnx=DEFAULT_ONNX,
        control_config=DEFAULT_CONTROL,
        scene=DEFAULT_SCENE,
        recorder=RECORDER,
        dry_schema=True,
    )
    archive = np.load(output, allow_pickle=False)
    legacy = {
        "joint_names", "time_s", "joint_q_rad", "joint_dq_radps", "joint_effort",
        "command_q_rad", "command_dq_radps", "command_effort", "command_kp",
        "command_kd", "command_valid", "root_pos_w_m", "root_quat_xyzw",
        "root_lin_vel_w_mps", "root_ang_vel", "torso_imu_quat_xyzw",
        "torso_imu_ang_vel", "torso_imu_lin_acc", "source", "truth_label",
    }
    assert legacy.issubset(archive.files)
    assert archive["joint_q_rad"].shape == (1, 31)
    assert archive["command_event_q_rad"].shape == (4, 31)
    assert archive["snapshot_state_group_receipt_monotonic_ns"].shape == (1, 4)
    assert archive["snapshot_odom_receipt_monotonic_ns"].shape == (1,)
    assert archive["snapshot_imu_receipt_monotonic_ns"].shape == (1,)
    assert archive["subscription_ready_monotonic_ns"].shape == ()
    assert archive["command_event_group"].tolist() == list(GROUPS)
    assert manifest["legacy_v1_snapshot_keys_preserved"] is True
    assert manifest["command_events_by_group"] == {group: 1 for group in GROUPS}
    assert manifest["ros_domain_id"] == 214
    assert manifest["control_mode"] == "RL_DEFAULT"
    assert manifest["launch_identity"] == "pytest-dry"
    assert [row["inferred_mode"] for row in manifest["control_mode_sequence"]] == [
        "JOINT_DEFAULT", "RL_DEFAULT"
    ]
    assert manifest["mode_events"] == 2
    assert set(manifest["artifacts"]) == {
        "official_onnx", "official_control_config", "official_scene", "recorder"
    }
    saved = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert saved["npz"]["sha256"] == manifest["npz"]["sha256"]


def test_state_group_mismatch_is_rejected():
    buffer = ReplayBuffer(start_monotonic_ns=0)
    wrong = SimpleNamespace(joints=[fake_joint("left_hip_pitch_joint")])
    try:
        buffer.record_command("arm", wrong, 1)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("group mismatch was silently accepted")


def test_ready_contract_requires_all_31_state_plus_odom_and_imu():
    buffer = ReplayBuffer(start_monotonic_ns=0)
    for group in GROUPS:
        state = SimpleNamespace(
            joints=[fake_joint(name) for name in JOINTS if GROUP_BY_JOINT[name] == group]
        )
        buffer.record_state(group, state, 1)
    assert buffer.snapshot_ready() is False
    dry = make_dry_buffer()
    assert dry.snapshot_ready() is True


def test_readiness_diagnostics_distinguish_missing_joint_odom_and_imu(tmp_path):
    buffer = ReplayBuffer(start_monotonic_ns=100)
    for group in GROUPS:
        names = [name for name in JOINTS if GROUP_BY_JOINT[name] == group]
        if group == "head":
            names = names[:-1]
        buffer.record_state(group, SimpleNamespace(joints=[fake_joint(name) for name in names]), 120)
    report = buffer.readiness_diagnostics(130)
    assert report["state_joint_seen_count"] == 30
    assert report["state_joint_missing"] == ["head_pitch_joint"]
    assert report["odom_seen"] is False
    assert report["imu_seen"] is False
    assert report["snapshot_ready"] is False
    path = tmp_path / "ready.json"
    write_readiness_diagnostics(path, buffer, 130)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["state_by_group"]["head"]["missing"] == ["head_pitch_joint"]


def test_readiness_diagnostics_report_odom_only_and_imu_only_boundaries():
    dry = make_dry_buffer()
    dry.imu = None
    imu_missing = dry.readiness_diagnostics(1_030_000_000)
    assert imu_missing["odom_seen"] is True and imu_missing["imu_seen"] is False
    assert imu_missing["snapshot_ready"] is False
    dry.imu = SimpleNamespace()
    dry.odom = None
    odom_missing = dry.readiness_diagnostics(1_030_000_000)
    assert odom_missing["odom_seen"] is False and odom_missing["imu_seen"] is True
    assert odom_missing["snapshot_ready"] is False


def test_capture_launcher_isolated_and_uses_official_mode_transition():
    outer = OUTER.read_text(encoding="utf-8")
    inner = INNER.read_text(encoding="utf-8")
    assert '[[ "$ROS_DOMAIN_ID" != "232" ]]' in outer
    assert "--network host --ipc host" in outer
    assert "x2-aimdk-humble:1.0" in outer
    assert "ros2 run x2_rl_deploy_controller x2_rl_deploy_controller" in inner
    assert "printf '0\\n' | ./start_sim.sh -s" in inner
    assert "'{buttons: [0, 0, 1, 0]}'" in inner
    assert "'{buttons: [0, 0, 0, 1]}'" in inner
    assert "official_x2_rollout_recorder_v2.py" in inner
    assert "kill \"${RECORDER_PID:-}\" \"${CONTROLLER_PID:-}\" \"${SIM_PID:-}\"" in inner


def test_phase17_launcher_is_frozen_to_new_isolated_domain():
    text = PHASE17.read_text(encoding="utf-8")
    assert 'ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-216}"' in text
    assert '[[ "$ROS_DOMAIN_ID" == "216" ]]' in text
    assert '[[ "$ROS_DOMAIN_ID" != "232" ]]' in text
    assert "run_official_native_reset_prefix_capture_v2_inner.sh" in text


def test_phase20_activation_order_keeps_snapshot_ready_as_hard_gate():
    outer = PHASE20_OUTER.read_text(encoding="utf-8")
    inner = PHASE20_INNER.read_text(encoding="utf-8")
    assert 'ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-219}"' in outer
    assert '[[ "$ROS_DOMAIN_ID" == "219" ]]' in outer
    subscription = inner.index('test -s "$SUB_READY"')
    joint = inner.index("'{buttons: [0, 0, 1, 0]}'")
    snapshot = inner.index('test -s "$SNAP_READY"')
    prefix = inner.index("sleep 4", snapshot)
    rl = inner.index("'{buttons: [0, 0, 0, 1]}'")
    assert subscription < joint < snapshot < prefix < rl
    assert '--subscription-ready-file "$SUB_READY"' in inner
    assert '--ready-file "$SNAP_READY"' in inner
    assert "topic_ready" in inner  # topic existence is only preflight, not state-ready.


def test_phase21_reuses_phase20_protocol_on_only_domain220():
    text = PHASE21_OUTER.read_text(encoding="utf-8")
    assert 'ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-220}"' in text
    assert '[[ "$ROS_DOMAIN_ID" == "220" ]]' in text
    assert "run_official_native_activation_prefix_phase20_inner.sh" in text
    assert "RL_SECONDS=20" in text
