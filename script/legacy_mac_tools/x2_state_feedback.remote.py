#!/usr/bin/env python3
"""X2 real-state feedback bridge for the policy observation.

The policy process and the ROS 2 runtime are intentionally separated.  The
ROS 2 helper subscribes to the four AimDK joint-state topics plus the torso
IMU and forwards a small, versioned UDP packet to ``x2_v2_teleop.py``.

This file has two independent parts:

* ``pack_feedback`` / ``unpack_feedback`` / ``FeedbackReceiver`` are usable
  without ROS 2 and are covered by the offline tests.
* ``RosStateFeedbackNode`` is only imported when this file is run as the
  ROS 2 helper on the robot.

The helper does not publish commands and does not change the robot mode.  It
only reads state topics.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


MAGIC = b"X2FB"
VERSION = 1
FLAG_JOINTS_VALID = 1 << 0
FLAG_BASE_VALID = 1 << 1
FLAG_IMU_VALID = 1 << 2
REQUIRED_FLAGS = FLAG_JOINTS_VALID | FLAG_BASE_VALID | FLAG_IMU_VALID
N_QPOS = 38  # root position(3) + root quaternion wxyz(4) + 31 joints
N_QVEL = 37  # root linear(3) + root angular(3) + 31 joint velocities
DEFAULT_FEEDBACK_PORT = 50041
PACKET_HEADER = struct.Struct("<4sBBIQ")
PACKET_PAYLOAD = struct.Struct(f"<{N_QPOS + N_QVEL}f")
PACKET_SIZE = PACKET_HEADER.size + PACKET_PAYLOAD.size

# This is the MuJoCo/policy order.  The names are checked against every
# incoming AimDK group before a value is accepted; no silent permutation is
# allowed.
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint", "head_yaw_joint",
    "head_pitch_joint",
)

GROUPS = {
    "leg": ("/aima/hal/joint/leg/state", 0, 12),
    "waist": ("/aima/hal/joint/waist/state", 12, 3),
    "arm": ("/aima/hal/joint/arm/state", 15, 14),
    "head": ("/aima/hal/joint/head/state", 29, 2),
}
IMU_TOPIC = "/aima/hal/imu/torso/state"
ODOM_TOPIC = "/aima/hal/odom/state"


def _as_vector(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def pack_feedback(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    seq: int = 0,
    sent_monotonic_ns: Optional[int] = None,
    flags: int = 0,
) -> bytes:
    """Pack one X2 state snapshot into the private local UDP format."""

    qpos = _as_vector(qpos, N_QPOS, "qpos")
    qvel = _as_vector(qvel, N_QVEL, "qvel")
    if sent_monotonic_ns is None:
        sent_monotonic_ns = time.monotonic_ns()
    header = PACKET_HEADER.pack(
        MAGIC, VERSION, int(flags) & 0xFF, int(seq) & 0xFFFFFFFF,
        int(sent_monotonic_ns) & 0xFFFFFFFFFFFFFFFF,
    )
    return header + PACKET_PAYLOAD.pack(*(np.concatenate([qpos, qvel]).tolist()))


def unpack_feedback(packet: bytes) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Validate and decode a feedback packet.

    Returns ``(seq, sent_monotonic_ns, flags, qpos, qvel)``.  Extra bytes are
    rejected so a future format change cannot be accepted accidentally.
    """

    if len(packet) != PACKET_SIZE:
        raise ValueError(f"feedback packet has {len(packet)} bytes, expected {PACKET_SIZE}")
    magic, version, flags, seq, sent_ns = PACKET_HEADER.unpack_from(packet)
    if magic != MAGIC:
        raise ValueError(f"bad feedback magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported feedback version: {version}")
    values = np.asarray(PACKET_PAYLOAD.unpack_from(packet, PACKET_HEADER.size), dtype=np.float32)
    qpos = values[:N_QPOS].copy()
    qvel = values[N_QPOS:].copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("feedback packet contains non-finite values")
    quat_norm = float(np.linalg.norm(qpos[3:7]))
    if quat_norm < 1e-6:
        raise ValueError("feedback base quaternion is zero")
    qpos[3:7] /= quat_norm
    return int(seq), int(sent_ns), int(flags), qpos, qvel


@dataclass(frozen=True)
class FeedbackSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    seq: int
    flags: int
    sent_monotonic_ns: int
    received_monotonic: float

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.received_monotonic)

    @property
    def source_age_s(self) -> float:
        """Age measured from the producer timestamp on the same host."""
        return max(0.0, (time.monotonic_ns() - self.sent_monotonic_ns) / 1e9)


class FeedbackReceiver:
    """Non-blocking receiver used by the 50 Hz policy loop."""

    def __init__(self, bind_host: str = "127.0.0.1", port: int = DEFAULT_FEEDBACK_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, int(port)))
        self.sock.setblocking(False)
        self.latest: Optional[FeedbackSnapshot] = None
        self.bad_packets = 0
        self.replayed_packets = 0

    @staticmethod
    def _seq_is_newer(seq: int, previous: int) -> bool:
        """RFC-1982 style uint32 comparison, including wraparound."""
        delta = (int(seq) - int(previous)) & 0xFFFFFFFF
        return 0 < delta < 0x80000000

    def poll(self, max_packets: int = 64) -> int:
        received = 0
        for _ in range(max_packets):
            try:
                packet, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                seq, sent_ns, flags, qpos, qvel = unpack_feedback(packet)
            except ValueError:
                self.bad_packets += 1
                continue
            if self.latest is not None and not self._seq_is_newer(seq, self.latest.seq):
                self.replayed_packets += 1
                continue
            self.latest = FeedbackSnapshot(
                qpos=qpos,
                qvel=qvel,
                seq=seq,
                flags=flags,
                sent_monotonic_ns=sent_ns,
                received_monotonic=time.monotonic(),
            )
            received += 1
        return received

    def snapshot(self, max_age_s: float,
                 required_flags: int = REQUIRED_FLAGS) -> Optional[FeedbackSnapshot]:
        state = self.latest
        if state is None:
            return None
        if state.age_s > float(max_age_s) or state.source_age_s > float(max_age_s):
            return None
        if state.sent_monotonic_ns > time.monotonic_ns() + 50_000_000:
            return None
        if (state.flags & int(required_flags)) != int(required_flags):
            return None
        return state

    def close(self) -> None:
        self.sock.close()


class RosStateFeedbackNode:
    """Read-only ROS 2 -> UDP state bridge.

    ROS 2 imports are deliberately local so the packet codec remains usable
    on the conda environment that runs the ONNX policy.
    """

    def __init__(self, udp_host: str, udp_port: int, send_hz: float,
                 max_age_s: float, base_source: str, root_z: float):
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from aimdk_msgs.msg import JointStateArray
        from sensor_msgs.msg import Imu

        class _Node(Node):
            def __init__(self):
                super().__init__("x2_policy_state_feedback")
                self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._udp_target = (udp_host, int(udp_port))
                self._send_seq = 0
                self._send_hz = float(send_hz)
                self._max_age_s = float(max_age_s)
                self._base_source = str(base_source)
                self._root_z = float(root_z)
                self._positions = np.zeros(31, dtype=np.float32)
                self._velocities = np.zeros(31, dtype=np.float32)
                self._root_position = np.zeros(3, dtype=np.float32)
                self._root_quat_wxyz = np.asarray(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float32,
                )
                self._root_linear_velocity = np.zeros(3, dtype=np.float32)
                self._root_angular_velocity = np.zeros(3, dtype=np.float32)
                self._imu_quat_wxyz = np.asarray(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float32,
                )
                self._imu_angular_velocity = np.zeros(3, dtype=np.float32)
                self._seen = {name: False for name in GROUPS}
                self._last_seen = {name: 0.0 for name in GROUPS}
                self._imu_seen = False
                self._imu_last_seen = 0.0
                self._odom_seen = False
                self._odom_last_seen = 0.0
                self._bad_group_logged = set()
                self._last_wait_log = 0.0
                self._last_send_log = 0.0

                qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                    durability=DurabilityPolicy.VOLATILE,
                )
                self._subscriptions = []
                for group_name, (topic, start, length) in GROUPS.items():
                    callback = lambda msg, group=group_name: self._on_joint_group(group, msg)
                    self._subscriptions.append(
                        self.create_subscription(JointStateArray, topic, callback, qos)
                    )
                self._subscriptions.append(
                    self.create_subscription(Imu, IMU_TOPIC, self._on_imu, qos)
                )
                self._subscriptions.append(
                    self.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos)
                )
                self._timer = self.create_timer(1.0 / self._send_hz, self._publish_snapshot)
                self.get_logger().info(
                    f"read-only state bridge -> udp://{udp_host}:{udp_port}; "
                    f"topics={len(GROUPS)} joints + odom + torso IMU; "
                    f"base_source={self._base_source}"
                )

            def _on_joint_group(self, group_name, msg):
                _, start, length = GROUPS[group_name]
                joints = list(getattr(msg, "joints", []))
                expected = set(JOINT_NAMES[start:start + length])
                actual = {str(getattr(joint, "name", "")) for joint in joints}
                if len(joints) != length or actual != expected:
                    if group_name not in self._bad_group_logged:
                        self._bad_group_logged.add(group_name)
                        self.get_logger().error(
                            f"{group_name} state layout mismatch: "
                            f"got={[getattr(j, 'name', '') for j in joints]}, "
                            f"expected={list(JOINT_NAMES[start:start + length])}; "
                            "feedback is being withheld"
                        )
                    return
                by_name = {str(joint.name): joint for joint in joints}
                for offset, name in enumerate(JOINT_NAMES[start:start + length]):
                    joint = by_name[name]
                    self._positions[start + offset] = float(joint.position)
                    self._velocities[start + offset] = float(joint.velocity)
                now = time.monotonic()
                self._seen[group_name] = True
                self._last_seen[group_name] = now

            def _on_imu(self, msg):
                q = np.asarray([
                    float(msg.orientation.w), float(msg.orientation.x),
                    float(msg.orientation.y), float(msg.orientation.z),
                ], dtype=np.float32)
                norm = float(np.linalg.norm(q))
                if norm < 1e-6 or not np.all(np.isfinite(q)):
                    return
                angular_velocity = np.asarray([
                    float(msg.angular_velocity.x),
                    float(msg.angular_velocity.y),
                    float(msg.angular_velocity.z),
                ], dtype=np.float32)
                if not np.all(np.isfinite(angular_velocity)):
                    return
                self._imu_quat_wxyz[:] = q / norm
                self._imu_angular_velocity[:] = angular_velocity
                self._imu_seen = True
                self._imu_last_seen = time.monotonic()

            def _on_odom(self, msg):
                pose = msg.pose.pose
                twist = msg.twist.twist
                q = np.asarray([
                    float(pose.orientation.w), float(pose.orientation.x),
                    float(pose.orientation.y), float(pose.orientation.z),
                ], dtype=np.float32)
                values = np.asarray([
                    float(pose.position.x), float(pose.position.y),
                    float(pose.position.z),
                    float(twist.linear.x), float(twist.linear.y),
                    float(twist.linear.z), float(twist.angular.x),
                    float(twist.angular.y), float(twist.angular.z),
                ], dtype=np.float32)
                norm = float(np.linalg.norm(q))
                if norm < 1e-6 or not np.all(np.isfinite(q)) or \
                        not np.all(np.isfinite(values)):
                    return
                self._root_position[:] = values[0:3]
                self._root_quat_wxyz[:] = q / norm
                self._root_linear_velocity[:] = values[3:6]
                self._root_angular_velocity[:] = values[6:9]
                self._odom_seen = True
                self._odom_last_seen = time.monotonic()

            def _fresh(self):
                if not all(self._seen.values()) or not self._imu_seen:
                    return False
                now = time.monotonic()
                fresh = all(
                    now - stamp <= self._max_age_s
                    for stamp in self._last_seen.values()
                ) and now - self._imu_last_seen <= self._max_age_s
                if self._base_source == "odom":
                    fresh = fresh and self._odom_seen \
                        and now - self._odom_last_seen <= self._max_age_s
                return fresh

            def _publish_snapshot(self):
                now = time.monotonic()
                if not self._fresh():
                    if now - self._last_wait_log > 2.0:
                        missing = [name for name, seen in self._seen.items() if not seen]
                        if not self._imu_seen:
                            missing.append("torso_imu")
                        if self._base_source == "odom" and not self._odom_seen:
                            missing.append("odom")
                        self.get_logger().warning(
                            "withholding feedback; missing/stale=" + ",".join(missing)
                        )
                        self._last_wait_log = now
                    return
                qpos = np.zeros(N_QPOS, dtype=np.float32)
                if self._base_source == "odom":
                    qpos[:3] = self._root_position
                    qpos[3:7] = self._root_quat_wxyz
                else:
                    qpos[:3] = [0.0, 0.0, self._root_z]
                    qpos[3:7] = self._imu_quat_wxyz
                qpos[7:38] = self._positions
                qvel = np.zeros(N_QVEL, dtype=np.float32)
                if self._base_source == "odom":
                    qvel[0:3] = self._root_linear_velocity
                    qvel[3:6] = self._root_angular_velocity
                else:
                    qvel[3:6] = self._imu_angular_velocity
                qvel[6:37] = self._velocities
                packet = pack_feedback(
                    qpos, qvel, seq=self._send_seq, flags=REQUIRED_FLAGS,
                )
                self._udp.sendto(packet, self._udp_target)
                self._send_seq = (self._send_seq + 1) & 0xFFFFFFFF
                if now - self._last_send_log > 2.0:
                    self.get_logger().info(
                        f"feedback sent: seq={self._send_seq} "
                        f"joint_vel_max={float(np.max(np.abs(self._velocities))):.3f}"
                    )
                    self._last_send_log = now

            def close(self):
                self._udp.close()

        self.node = _Node()
        self._rclpy = rclpy


def _run_ros_helper(args) -> None:
    import rclpy

    rclpy.init()
    bridge = RosStateFeedbackNode(
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        send_hz=args.send_hz,
        max_age_s=args.max_age_s,
        base_source=args.base_source,
        root_z=args.root_z,
    )
    try:
        rclpy.spin(bridge.node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # rclpy may invalidate subscriptions while handling SIGINT.  Preserve
        # genuine runtime errors, but treat an already-shutdown context as a
        # normal exit.
        if rclpy.ok():
            raise
    finally:
        bridge.node.close()
        if rclpy.ok():
            bridge.node.destroy_node()
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only X2 ROS2 state -> policy UDP feedback")
    parser.add_argument("--udp-host", default="127.0.0.1",
                        help="policy host; same machine is safest")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_FEEDBACK_PORT)
    parser.add_argument("--send-hz", type=float, default=100.0)
    parser.add_argument("--base-source", choices=["imu", "odom"], default="imu",
                        help="imu works on current X2; odom requires an active publisher")
    parser.add_argument("--root-z", type=float, default=0.65,
                        help="placeholder root height used with --base-source imu")
    parser.add_argument("--max-age-s", type=float, default=0.15)
    args = parser.parse_args()
    _run_ros_helper(args)


if __name__ == "__main__":
    main()
