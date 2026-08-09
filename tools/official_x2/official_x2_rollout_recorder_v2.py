#!/usr/bin/env python3
"""Event-level replayable recorder for the official AimDK X2 ROS contract.

The legacy 50 Hz snapshot arrays remain present and retain their old names.
V2 additionally records every arm/leg/waist/head command callback, its
monotonic receipt time, group-local sequence, message header fields, changed
joint mask, and the complete latest 31-joint command map.  State/odom/IMU
snapshots carry callback receipt times.  ``--dry-schema`` works without ROS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)
GROUPS = ("leg", "waist", "arm", "head")
GROUP_BY_JOINT = {
    name: (
        "leg" if any(token in name for token in ("hip", "knee", "ankle"))
        else "waist" if name.startswith("waist_")
        else "head" if name.startswith("head_")
        else "arm"
    )
    for name in JOINTS
}
JOINT_INDEX = {name: index for index, name in enumerate(JOINTS)}
COMMAND_WIDTH = 5  # q, dq, effort, stiffness, damping
SCHEMA_VERSION = "x2_official_event_replay_v2"
SOURCE_LABEL = "AimDK X2 v1.0 official MuJoCo + shipped kuailechongbai.onnx"
TRUTH_LABEL = "official_simulation_truth_not_real_hardware_truth"

OFFICIAL_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy"
)
DEFAULT_ONNX = OFFICIAL_ROOT / "x2_rl_deploy_controller/config/rl_model/kuailechongbai.onnx"
DEFAULT_CONTROL = OFFICIAL_ROOT / "x2_rl_deploy_controller/config/motion_control.yaml"
DEFAULT_SCENE = (
    OFFICIAL_ROOT / "x2_rl_deploy_mujoco/configuration/robot/"
    "lx2501_3_t2d5/model_info/scene.xml"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def header_fields(msg: Any) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = int(getattr(stamp, "sec", 0)) if stamp is not None else 0
    nanosec = int(getattr(stamp, "nanosec", 0)) if stamp is not None else 0
    sequence = int(getattr(header, "sequence", 0)) if header is not None else 0
    available = header is not None and stamp is not None
    populated = bool(available and (sec != 0 or nanosec != 0 or sequence != 0))
    return {
        "available": available,
        "populated": populated,
        "stamp_sec": sec,
        "stamp_nanosec": nanosec,
        "sequence": sequence,
    }


def infer_mode(buttons: list[int] | tuple[int, ...] | np.ndarray) -> str:
    values = [int(value) for value in buttons]
    if len(values) > 0 and values[0] == 1:
        return "STOP_REQUEST"
    if len(values) > 1 and values[1] == 1:
        return "DAMPING_DEFAULT"
    if len(values) > 2 and values[2] == 1:
        return "JOINT_DEFAULT"
    if len(values) > 3 and values[3] == 1:
        return "RL_DEFAULT"
    return "UNKNOWN"


def _empty(width: int, dtype: Any) -> np.ndarray:
    return np.empty((0, width), dtype=dtype)


class ReplayBuffer:
    """Pure-python capture buffer, intentionally testable without ROS."""

    def __init__(self, start_monotonic_ns: int | None = None):
        self.start_monotonic_ns = int(time.monotonic_ns() if start_monotonic_ns is None else start_monotonic_ns)
        self.subscription_ready_monotonic_ns = -1
        self.state: dict[str, tuple[float, float, float]] = {}
        self.state_first_receipt_ns: dict[str, int] = {}
        self.state_group_receipt_ns = {group: -1 for group in GROUPS}
        self.command: dict[str, tuple[float, float, float, float, float]] = {}
        self.command_events: list[dict[str, Any]] = []
        self.mode_events: list[dict[str, Any]] = []
        self.group_event_count: defaultdict[str, int] = defaultdict(int)
        self.imu: Any | None = None
        self.imu_receipt_ns = -1
        self.imu_first_receipt_ns = -1
        self.odom: Any | None = None
        self.odom_receipt_ns = -1
        self.odom_first_receipt_ns = -1
        self.snapshots: list[dict[str, Any]] = []

    def record_state(self, group: str, msg: Any, receipt_monotonic_ns: int | None = None) -> None:
        if group not in GROUPS:
            raise ValueError(group)
        receipt = int(time.monotonic_ns() if receipt_monotonic_ns is None else receipt_monotonic_ns)
        for joint in msg.joints:
            name = str(joint.name)
            if name not in JOINT_INDEX:
                continue
            if GROUP_BY_JOINT[name] != group:
                raise ValueError(f"state joint/group mismatch: {name}/{group}")
            self.state[name] = (float(joint.position), float(joint.velocity), float(joint.effort))
            self.state_first_receipt_ns.setdefault(name, receipt)
        self.state_group_receipt_ns[group] = receipt

    def record_command(self, group: str, msg: Any, receipt_monotonic_ns: int | None = None) -> None:
        if group not in GROUPS:
            raise ValueError(group)
        receipt = int(time.monotonic_ns() if receipt_monotonic_ns is None else receipt_monotonic_ns)
        changed = np.zeros(len(JOINTS), dtype=bool)
        for joint in msg.joints:
            name = str(joint.name)
            if name not in JOINT_INDEX:
                continue
            if GROUP_BY_JOINT[name] != group:
                raise ValueError(f"command joint/group mismatch: {name}/{group}")
            self.command[name] = (
                float(joint.position), float(joint.velocity), float(joint.effort),
                float(joint.stiffness), float(joint.damping),
            )
            changed[JOINT_INDEX[name]] = True
        values = np.asarray([self.command.get(name, (np.nan,) * COMMAND_WIDTH) for name in JOINTS], np.float64)
        valid = np.asarray([name in self.command for name in JOINTS], bool)
        header = header_fields(msg)
        group_index = self.group_event_count[group]
        self.group_event_count[group] += 1
        self.command_events.append({
            "global_index": len(self.command_events),
            "group": group,
            "group_index": group_index,
            "receipt_monotonic_ns": receipt,
            "elapsed_ns": receipt - self.start_monotonic_ns,
            "header": header,
            "changed_mask": changed,
            "valid": valid,
            "values": values,
        })

    def record_imu(self, msg: Any, receipt_monotonic_ns: int | None = None) -> None:
        self.imu = msg
        self.imu_receipt_ns = int(time.monotonic_ns() if receipt_monotonic_ns is None else receipt_monotonic_ns)
        if self.imu_first_receipt_ns < 0:
            self.imu_first_receipt_ns = self.imu_receipt_ns

    def record_odom(self, msg: Any, receipt_monotonic_ns: int | None = None) -> None:
        self.odom = msg
        self.odom_receipt_ns = int(time.monotonic_ns() if receipt_monotonic_ns is None else receipt_monotonic_ns)
        if self.odom_first_receipt_ns < 0:
            self.odom_first_receipt_ns = self.odom_receipt_ns

    def record_mode_event(self, msg: Any, receipt_monotonic_ns: int | None = None) -> None:
        receipt = int(time.monotonic_ns() if receipt_monotonic_ns is None else receipt_monotonic_ns)
        buttons = [int(value) for value in getattr(msg, "buttons", [])]
        axes = [float(value) for value in getattr(msg, "axes", [])]
        self.mode_events.append({
            "global_index": len(self.mode_events),
            "receipt_monotonic_ns": receipt,
            "elapsed_ns": receipt - self.start_monotonic_ns,
            "header": header_fields(msg),
            "buttons": buttons,
            "axes": axes,
            "inferred_mode": infer_mode(buttons),
        })

    def snapshot_ready(self) -> bool:
        return self.imu is not None and self.odom is not None and all(name in self.state for name in JOINTS)

    def readiness_diagnostics(self, now_monotonic_ns: int | None = None) -> dict[str, Any]:
        now = int(time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns)
        seen = [name for name in JOINTS if name in self.state]
        missing = [name for name in JOINTS if name not in self.state]
        by_group = {
            group: {
                "seen": int(sum(name in self.state for name in JOINTS if GROUP_BY_JOINT[name] == group)),
                "expected": int(sum(GROUP_BY_JOINT[name] == group for name in JOINTS)),
                "missing": [name for name in JOINTS if GROUP_BY_JOINT[name] == group and name not in self.state],
            }
            for group in GROUPS
        }
        elapsed = lambda value: None if value < 0 else int(value - self.start_monotonic_ns)
        return {
            "schema_version": "x2_official_readiness_diagnostics_v1",
            "capture_start_monotonic_ns": self.start_monotonic_ns,
            "diagnostic_receipt_monotonic_ns": now,
            "diagnostic_elapsed_ns": now - self.start_monotonic_ns,
            "snapshot_ready": self.snapshot_ready(),
            "state_joint_seen_count": len(seen),
            "state_joint_expected_count": len(JOINTS),
            "state_joint_seen": seen,
            "state_joint_missing": missing,
            "state_by_group": by_group,
            "state_joint_first_elapsed_ns": {
                name: int(self.state_first_receipt_ns[name] - self.start_monotonic_ns)
                for name in seen
            },
            "odom_seen": self.odom is not None,
            "odom_first_elapsed_ns": elapsed(self.odom_first_receipt_ns),
            "imu_seen": self.imu is not None,
            "imu_first_elapsed_ns": elapsed(self.imu_first_receipt_ns),
        }

    def record_snapshot(self, receipt_monotonic_ns: int | None = None) -> bool:
        if not self.snapshot_ready():
            return False
        receipt = int(time.monotonic_ns() if receipt_monotonic_ns is None else receipt_monotonic_ns)
        state = np.asarray([self.state[name] for name in JOINTS], np.float64)
        command = np.asarray([self.command.get(name, (np.nan,) * COMMAND_WIDTH) for name in JOINTS], np.float64)
        command_valid = np.asarray([name in self.command for name in JOINTS], bool)
        op = self.odom.pose.pose
        ot = self.odom.twist.twist
        iq = self.imu.orientation
        iw = self.imu.angular_velocity
        ia = self.imu.linear_acceleration
        self.snapshots.append({
            "receipt_monotonic_ns": receipt,
            "elapsed_ns": receipt - self.start_monotonic_ns,
            "state": state,
            "state_group_receipt_ns": np.asarray([self.state_group_receipt_ns[group] for group in GROUPS], np.int64),
            "command": command,
            "command_valid": command_valid,
            "root_pos": np.asarray([op.position.x, op.position.y, op.position.z], np.float64),
            "root_quat": np.asarray([op.orientation.x, op.orientation.y, op.orientation.z, op.orientation.w], np.float64),
            "root_lin": np.asarray([ot.linear.x, ot.linear.y, ot.linear.z], np.float64),
            "root_ang": np.asarray([ot.angular.x, ot.angular.y, ot.angular.z], np.float64),
            "odom_receipt_ns": self.odom_receipt_ns,
            "imu_quat": np.asarray([iq.x, iq.y, iq.z, iq.w], np.float64),
            "imu_ang": np.asarray([iw.x, iw.y, iw.z], np.float64),
            "imu_acc": np.asarray([ia.x, ia.y, ia.z], np.float64),
            "imu_receipt_ns": self.imu_receipt_ns,
        })
        return True

    def arrays(self) -> dict[str, np.ndarray]:
        snapshots = self.snapshots
        events = self.command_events
        mode_events = self.mode_events
        if snapshots:
            state = np.stack([row["state"] for row in snapshots])
            command = np.stack([row["command"] for row in snapshots])
        else:
            state = np.empty((0, len(JOINTS), 3), np.float64)
            command = np.empty((0, len(JOINTS), COMMAND_WIDTH), np.float64)
        if events:
            event_values = np.stack([row["values"] for row in events])
            event_valid = np.stack([row["valid"] for row in events])
            event_changed = np.stack([row["changed_mask"] for row in events])
        else:
            event_values = np.empty((0, len(JOINTS), COMMAND_WIDTH), np.float64)
            event_valid = _empty(len(JOINTS), bool)
            event_changed = _empty(len(JOINTS), bool)
        button_width = max((len(row["buttons"]) for row in mode_events), default=0)
        axis_width = max((len(row["axes"]) for row in mode_events), default=0)
        mode_buttons = np.zeros((len(mode_events), button_width), dtype=np.int32)
        mode_axes = np.full((len(mode_events), axis_width), np.nan, dtype=np.float32)
        for index, row in enumerate(mode_events):
            mode_buttons[index, :len(row["buttons"])] = row["buttons"]
            mode_axes[index, :len(row["axes"])] = row["axes"]

        # Legacy v1 keys remain byte-compatible in shape/meaning (dtypes may be
        # normalized to float32 exactly as v1 did).
        result: dict[str, np.ndarray] = {
            "joint_names": np.asarray(JOINTS),
            "time_s": np.asarray([row["elapsed_ns"] * 1.0e-9 for row in snapshots], np.float64),
            "joint_q_rad": state[:, :, 0].astype(np.float32),
            "joint_dq_radps": state[:, :, 1].astype(np.float32),
            "joint_effort": state[:, :, 2].astype(np.float32),
            "command_q_rad": command[:, :, 0].astype(np.float32),
            "command_dq_radps": command[:, :, 1].astype(np.float32),
            "command_effort": command[:, :, 2].astype(np.float32),
            "command_kp": command[:, :, 3].astype(np.float32),
            "command_kd": command[:, :, 4].astype(np.float32),
            "command_valid": np.stack([row["command_valid"] for row in snapshots]) if snapshots else _empty(len(JOINTS), bool),
            "root_pos_w_m": np.stack([row["root_pos"] for row in snapshots]).astype(np.float32) if snapshots else _empty(3, np.float32),
            "root_quat_xyzw": np.stack([row["root_quat"] for row in snapshots]).astype(np.float32) if snapshots else _empty(4, np.float32),
            "root_lin_vel_w_mps": np.stack([row["root_lin"] for row in snapshots]).astype(np.float32) if snapshots else _empty(3, np.float32),
            "root_ang_vel": np.stack([row["root_ang"] for row in snapshots]).astype(np.float32) if snapshots else _empty(3, np.float32),
            "torso_imu_quat_xyzw": np.stack([row["imu_quat"] for row in snapshots]).astype(np.float32) if snapshots else _empty(4, np.float32),
            "torso_imu_ang_vel": np.stack([row["imu_ang"] for row in snapshots]).astype(np.float32) if snapshots else _empty(3, np.float32),
            "torso_imu_lin_acc": np.stack([row["imu_acc"] for row in snapshots]).astype(np.float32) if snapshots else _empty(3, np.float32),
            "source": np.asarray(SOURCE_LABEL),
            "truth_label": np.asarray(TRUTH_LABEL),
            # V2 snapshot provenance.
            "schema_version": np.asarray(SCHEMA_VERSION),
            "capture_start_monotonic_ns": np.asarray(self.start_monotonic_ns, np.int64),
            "subscription_ready_monotonic_ns": np.asarray(self.subscription_ready_monotonic_ns, np.int64),
            "snapshot_receipt_monotonic_ns": np.asarray([row["receipt_monotonic_ns"] for row in snapshots], np.int64),
            "snapshot_state_group_order": np.asarray(GROUPS),
            "snapshot_state_group_receipt_monotonic_ns": np.stack([row["state_group_receipt_ns"] for row in snapshots]) if snapshots else np.empty((0, len(GROUPS)), np.int64),
            "snapshot_odom_receipt_monotonic_ns": np.asarray([row["odom_receipt_ns"] for row in snapshots], np.int64),
            "snapshot_imu_receipt_monotonic_ns": np.asarray([row["imu_receipt_ns"] for row in snapshots], np.int64),
            # Event-level exact receipt order and complete latest 31-map.
            "command_event_global_index": np.asarray([row["global_index"] for row in events], np.int64),
            "command_event_group": np.asarray([row["group"] for row in events]),
            "command_event_group_index": np.asarray([row["group_index"] for row in events], np.int64),
            "command_event_receipt_monotonic_ns": np.asarray([row["receipt_monotonic_ns"] for row in events], np.int64),
            "command_event_elapsed_ns": np.asarray([row["elapsed_ns"] for row in events], np.int64),
            "command_event_header_available": np.asarray([row["header"]["available"] for row in events], bool),
            "command_event_header_populated": np.asarray([row["header"]["populated"] for row in events], bool),
            "command_event_header_stamp_sec": np.asarray([row["header"]["stamp_sec"] for row in events], np.int64),
            "command_event_header_stamp_nanosec": np.asarray([row["header"]["stamp_nanosec"] for row in events], np.int64),
            "command_event_header_sequence": np.asarray([row["header"]["sequence"] for row in events], np.int64),
            "command_event_changed_joint_mask": event_changed,
            "command_event_valid": event_valid,
            "command_event_q_rad": event_values[:, :, 0].astype(np.float32),
            "command_event_dq_radps": event_values[:, :, 1].astype(np.float32),
            "command_event_effort": event_values[:, :, 2].astype(np.float32),
            "command_event_kp": event_values[:, :, 3].astype(np.float32),
            "command_event_kd": event_values[:, :, 4].astype(np.float32),
            # Optional, backwards-compatible Joy/mode transition stream.
            "mode_event_global_index": np.asarray([row["global_index"] for row in mode_events], np.int64),
            "mode_event_receipt_monotonic_ns": np.asarray([row["receipt_monotonic_ns"] for row in mode_events], np.int64),
            "mode_event_elapsed_ns": np.asarray([row["elapsed_ns"] for row in mode_events], np.int64),
            "mode_event_header_available": np.asarray([row["header"]["available"] for row in mode_events], bool),
            "mode_event_header_populated": np.asarray([row["header"]["populated"] for row in mode_events], bool),
            "mode_event_header_stamp_sec": np.asarray([row["header"]["stamp_sec"] for row in mode_events], np.int64),
            "mode_event_header_stamp_nanosec": np.asarray([row["header"]["stamp_nanosec"] for row in mode_events], np.int64),
            "mode_event_header_sequence": np.asarray([row["header"]["sequence"] for row in mode_events], np.int64),
            "mode_event_button_count": np.asarray([len(row["buttons"]) for row in mode_events], np.int32),
            "mode_event_buttons": mode_buttons,
            "mode_event_axis_count": np.asarray([len(row["axes"]) for row in mode_events], np.int32),
            "mode_event_axes": mode_axes,
            "mode_event_inferred_mode": np.asarray([row["inferred_mode"] for row in mode_events]),
        }
        return result


def write_outputs(
    buffer: ReplayBuffer,
    output: Path,
    *,
    ros_domain_id: int,
    control_mode: str,
    launch_identity: str,
    onnx: Path,
    control_config: Path,
    scene: Path,
    recorder: Path,
    dry_schema: bool,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = buffer.arrays()
    np.savez_compressed(output, **arrays)
    digest = sha256(output)
    event_groups = arrays["command_event_group"]
    mode_sequence = [
        {
            "event_index": int(index),
            "elapsed_ns": int(elapsed),
            "inferred_mode": str(mode),
            "buttons": arrays["mode_event_buttons"][index, :int(arrays["mode_event_button_count"][index])].tolist(),
        }
        for index, (elapsed, mode) in enumerate(zip(
            arrays["mode_event_elapsed_ns"], arrays["mode_event_inferred_mode"]
        ))
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "legacy_v1_snapshot_keys_preserved": True,
        "source": SOURCE_LABEL,
        "truth_label": TRUTH_LABEL,
        "dry_schema": bool(dry_schema),
        "sample_rate_hz": 50,
        "snapshots": int(len(arrays["time_s"])),
        "command_events": int(len(event_groups)),
        "command_events_by_group": {
            group: int(np.count_nonzero(event_groups == group)) for group in GROUPS
        },
        "mode_events": int(len(mode_sequence)),
        "control_mode_sequence": mode_sequence,
        "joint_count": len(JOINTS),
        "ros_domain_id": int(ros_domain_id),
        "control_mode": str(control_mode),
        "launch_identity": str(launch_identity),
        "command_event_time_source": "callback time.monotonic_ns; authoritative when message header is default/unpopulated",
        "subscription_ready_elapsed_ns": (
            None if buffer.subscription_ready_monotonic_ns < 0
            else int(buffer.subscription_ready_monotonic_ns - buffer.start_monotonic_ns)
        ),
        "artifacts": {
            "official_onnx": artifact(onnx),
            "official_control_config": artifact(control_config),
            "official_scene": artifact(scene),
            "recorder": artifact(recorder),
        },
        "npz": {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": digest},
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_readiness_diagnostics(
    path: Path, buffer: ReplayBuffer, now_monotonic_ns: int | None = None
) -> dict[str, Any]:
    report = buffer.readiness_diagnostics(now_monotonic_ns)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return report


def fake_joint(name: str, offset: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, position=offset, velocity=0.1 + offset, effort=0.0,
        stiffness=40.0, damping=2.0,
    )


def fake_header(sequence: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), sequence=sequence)


def make_dry_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(start_monotonic_ns=1_000_000_000)
    buffer.record_mode_event(
        SimpleNamespace(header=fake_header(1), buttons=[0, 0, 1, 0], axes=[]),
        1_000_500_000,
    )
    for group_index, group in enumerate(GROUPS):
        names = [name for name in JOINTS if GROUP_BY_JOINT[name] == group]
        command = SimpleNamespace(
            header=fake_header(), joints=[fake_joint(name, group_index * 0.01) for name in names]
        )
        state = SimpleNamespace(joints=[fake_joint(name, group_index * 0.01) for name in names])
        buffer.record_command(group, command, 1_001_000_000 + group_index * 1000)
        buffer.record_state(group, state, 1_002_000_000 + group_index * 1000)
    vector = lambda **values: SimpleNamespace(x=values.get("x", 0.0), y=values.get("y", 0.0), z=values.get("z", 0.0))
    quat = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    odom = SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(position=vector(z=0.65), orientation=quat)),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=vector(), angular=vector())),
    )
    imu = SimpleNamespace(orientation=quat, angular_velocity=vector(), linear_acceleration=vector(z=9.81))
    buffer.record_odom(odom, 1_003_000_000)
    buffer.record_imu(imu, 1_003_100_000)
    assert buffer.record_snapshot(1_020_000_000)
    buffer.record_mode_event(
        SimpleNamespace(header=fake_header(2), buttons=[0, 0, 0, 1], axes=[]),
        1_021_000_000,
    )
    return buffer


def run_ros(args: argparse.Namespace) -> ReplayBuffer:
    try:
        import rclpy
        from aimdk_msgs.msg import JointCommandArray, JointStateArray
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu
        from sensor_msgs.msg import Joy
    except ImportError as exc:
        raise RuntimeError("ROS/AimDK imports unavailable; use --dry-schema or official container") from exc

    class RecorderV2(Node):
        def __init__(self) -> None:
            super().__init__("official_x2_rollout_recorder_v2")
            self.buffer = ReplayBuffer()
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
            for group in GROUPS:
                self.create_subscription(
                    JointStateArray, f"/aima/hal/joint/{group}/state",
                    lambda msg, value=group: self.buffer.record_state(value, msg, time.monotonic_ns()), qos,
                )
                self.create_subscription(
                    JointCommandArray, f"/aima/hal/joint/{group}/command",
                    lambda msg, value=group: self.buffer.record_command(value, msg, time.monotonic_ns()), qos,
                )
            self.create_subscription(
                Imu, "/aima/hal/imu/torso/state",
                lambda msg: self.buffer.record_imu(msg, time.monotonic_ns()), qos,
            )
            self.create_subscription(
                Odometry, "/aima/hal/odom/state",
                lambda msg: self.buffer.record_odom(msg, time.monotonic_ns()), qos,
            )
            self.create_subscription(
                Joy, "/joy",
                lambda msg: self.buffer.record_mode_event(msg, time.monotonic_ns()), qos,
            )
            self.create_timer(0.02, lambda: self.buffer.record_snapshot(time.monotonic_ns()))

    rclpy.init()
    node = RecorderV2()
    node.buffer.subscription_ready_monotonic_ns = time.monotonic_ns()
    if args.subscription_ready_file is not None:
        args.subscription_ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.subscription_ready_file.write_text(
            json.dumps({
                "pid": os.getpid(),
                "subscription_ready_monotonic_ns": node.buffer.subscription_ready_monotonic_ns,
                "subscription_ready_elapsed_ns": (
                    node.buffer.subscription_ready_monotonic_ns - node.buffer.start_monotonic_ns
                ),
                "truth_boundary": "subscriptions constructed; not state/snapshot ready",
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    deadline = time.monotonic() + args.duration
    ready_written = False
    next_diagnostic = 0.0
    last_diagnostic_signature: tuple[Any, ...] | None = None
    while (
        rclpy.ok()
        and time.monotonic() <= deadline
        and not (args.stop_file is not None and args.stop_file.exists())
    ):
        rclpy.spin_once(node, timeout_sec=0.01)
        now = time.monotonic()
        if args.readiness_diagnostics is not None and now >= next_diagnostic:
            diagnostic = write_readiness_diagnostics(args.readiness_diagnostics, node.buffer)
            signature = (
                diagnostic["state_joint_seen_count"], diagnostic["odom_seen"],
                diagnostic["imu_seen"], diagnostic["snapshot_ready"],
            )
            if signature != last_diagnostic_signature:
                print("READINESS " + json.dumps(diagnostic, ensure_ascii=False), flush=True)
                last_diagnostic_signature = signature
            next_diagnostic = now + 0.25
        # Phase17 handshake: subscriptions existing is insufficient.  Signal
        # only after all 31 state joints plus odom and IMU have actually arrived.
        if (
            args.ready_file is not None and not ready_written
            and node.buffer.snapshot_ready() and len(node.buffer.snapshots) > 0
        ):
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text(str(os.getpid()) + "\n", encoding="utf-8")
            ready_written = True
    if args.readiness_diagnostics is not None:
        write_readiness_diagnostics(args.readiness_diagnostics, node.buffer)
    result = node.buffer
    node.destroy_node()
    rclpy.shutdown()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ros-domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "0")))
    parser.add_argument("--control-mode", default="RL_DEFAULT")
    parser.add_argument("--launch-identity", required=True)
    parser.add_argument("--official-onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--official-control-config", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--official-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--dry-schema", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--subscription-ready-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--readiness-diagnostics", type=Path)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    env_domain = int(os.environ.get("ROS_DOMAIN_ID", str(args.ros_domain_id)))
    if not args.dry_schema and env_domain != args.ros_domain_id:
        parser.error("ROS_DOMAIN_ID environment and --ros-domain-id differ")
    buffer = make_dry_buffer() if args.dry_schema else run_ros(args)
    manifest = write_outputs(
        buffer, args.output, ros_domain_id=args.ros_domain_id,
        control_mode=args.control_mode, launch_identity=args.launch_identity,
        onnx=args.official_onnx, control_config=args.official_control_config,
        scene=args.official_scene, recorder=Path(__file__), dry_schema=args.dry_schema,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
