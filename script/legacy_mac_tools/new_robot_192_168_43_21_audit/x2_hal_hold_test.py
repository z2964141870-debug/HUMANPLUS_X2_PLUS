#!/usr/bin/env python3
"""One-shot, guarded X2 HAL current-pose hold test."""

import argparse
import math
import time
from collections import defaultdict

import rclpy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


AREAS = {
    "leg": "/aima/hal/joint/leg",
    "waist": "/aima/hal/joint/waist",
    "arm": "/aima/hal/joint/arm",
    "head": "/aima/hal/joint/head",
}

# Sonic controls the 29 body joints. The head is intentionally excluded because
# this robot currently reports persistent head motor error 1026.
CONTROL_AREAS = ("leg", "waist", "arm")

GAINS = {
    "left_hip_pitch_joint": (40.0, 4.0),
    "left_hip_roll_joint": (40.0, 4.0),
    "left_hip_yaw_joint": (30.0, 3.0),
    "left_knee_joint": (80.0, 8.0),
    "left_ankle_pitch_joint": (40.0, 4.0),
    "left_ankle_roll_joint": (20.0, 2.0),
    "right_hip_pitch_joint": (40.0, 4.0),
    "right_hip_roll_joint": (40.0, 4.0),
    "right_hip_yaw_joint": (30.0, 3.0),
    "right_knee_joint": (80.0, 8.0),
    "right_ankle_pitch_joint": (40.0, 4.0),
    "right_ankle_roll_joint": (20.0, 2.0),
    "waist_yaw_joint": (20.0, 4.0),
    "waist_pitch_joint": (20.0, 4.0),
    "waist_roll_joint": (20.0, 4.0),
    "left_shoulder_pitch_joint": (20.0, 2.0),
    "left_shoulder_roll_joint": (20.0, 2.0),
    "left_shoulder_yaw_joint": (20.0, 2.0),
    "left_elbow_joint": (20.0, 2.0),
    "left_wrist_yaw_joint": (20.0, 2.0),
    "left_wrist_pitch_joint": (20.0, 2.0),
    "left_wrist_roll_joint": (20.0, 2.0),
    "right_shoulder_pitch_joint": (20.0, 2.0),
    "right_shoulder_roll_joint": (20.0, 2.0),
    "right_shoulder_yaw_joint": (20.0, 2.0),
    "right_elbow_joint": (20.0, 2.0),
    "right_wrist_yaw_joint": (20.0, 2.0),
    "right_wrist_pitch_joint": (20.0, 2.0),
    "right_wrist_roll_joint": (20.0, 2.0),
    "head_yaw_joint": (20.0, 2.0),
    "head_pitch_joint": (20.0, 2.0),
}


class HoldTest(Node):
    def __init__(self):
        super().__init__("x2_hal_hold_test")
        self.states = {}
        self.state_times = {}
        self.subscriptions_by_area = []
        self.publishers_by_area = {}
        for area, base in AREAS.items():
            sub = self.create_subscription(
                JointStateArray,
                f"{base}/state",
                lambda msg, area=area: self._on_state(area, msg),
                qos_profile_sensor_data,
            )
            self.subscriptions_by_area.append(sub)

    def _on_state(self, area, msg):
        self.states[area] = list(msg.joints)
        self.state_times[area] = time.monotonic()

    def wait_for_states(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not set(CONTROL_AREAS).issubset(self.states):
            rclpy.spin_once(self, timeout_sec=0.05)
        return set(CONTROL_AREAS).issubset(self.states)

    def assert_command_topics_unowned(self):
        conflicts = defaultdict(list)
        for area, base in AREAS.items():
            for info in self.get_publishers_info_by_topic(f"{base}/command"):
                conflicts[area].append(info.node_name)
        if conflicts:
            detail = ", ".join(f"{area}={names}" for area, names in conflicts.items())
            raise RuntimeError(f"HAL command topic already has publisher(s): {detail}")

    def create_command_publishers(self):
        for area in CONTROL_AREAS:
            base = AREAS[area]
            self.publishers_by_area[area] = self.create_publisher(
                JointCommandArray, f"{base}/command", qos_profile_sensor_data
            )

    def make_command(self, area, targets, sequence):
        msg = JointCommandArray()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.meas_stamp = now
        msg.header.frame_id = "x2_hal_hold_test"
        msg.header.sequence = sequence
        for state in self.states[area]:
            if state.name not in targets or state.name not in GAINS:
                raise RuntimeError(f"Unexpected or unsupported joint: {state.name}")
            kp, kd = GAINS[state.name]
            cmd = JointCommand()
            cmd.name = state.name
            cmd.position = targets[state.name]
            cmd.velocity = 0.0
            cmd.effort = 0.0
            cmd.stiffness = kp
            cmd.damping = kd
            msg.joints.append(cmd)
        return msg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--max-drift", type=float, default=0.08)
    parser.add_argument("--state-timeout", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.confirm != "X2_SUSPENDED_HAL_HOLD":
        raise SystemExit("Confirmation token mismatch; refusing to publish")
    if not 0.5 <= args.duration <= 5.0:
        raise SystemExit("Duration must be between 0.5 and 5.0 seconds")

    rclpy.init()
    node = HoldTest()
    try:
        # Allow DDS discovery to settle before ownership and state checks.
        discovery_deadline = time.monotonic() + 2.0
        while time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.assert_command_topics_unowned()
        if not node.wait_for_states(3.0):
            raise RuntimeError(
                f"Missing state groups: {set(CONTROL_AREAS) - set(node.states)}"
            )

        expected_counts = {"leg": 12, "waist": 3, "arm": 14}
        actual_counts = {
            area: len(node.states[area]) for area in CONTROL_AREAS
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Joint count mismatch: expected={expected_counts}, actual={actual_counts}"
            )

        targets = {
            state.name: state.position
            for area in CONTROL_AREAS
            for state in node.states[area]
        }
        if len(targets) != 29 or any(not math.isfinite(x) for x in targets.values()):
            raise RuntimeError("Invalid current-pose snapshot")

        node.create_command_publishers()
        print(f"ARMED current-pose hold: joints={len(targets)} duration={args.duration:.1f}s")
        start = time.monotonic()
        next_tick = start
        sequence = 0
        max_drift_seen = 0.0
        while time.monotonic() - start < args.duration:
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            stale = [
                area
                for area in CONTROL_AREAS
                if now - node.state_times.get(area, 0.0) > args.state_timeout
            ]
            if stale:
                raise RuntimeError(f"State timeout: {stale}")

            max_drift_joint = ""
            for area in CONTROL_AREAS:
                for state in node.states[area]:
                    drift = abs(state.position - targets[state.name])
                    if drift > max_drift_seen:
                        max_drift_seen = drift
                        max_drift_joint = state.name
            if max_drift_seen > args.max_drift:
                raise RuntimeError(
                    f"Pose drift {max_drift_seen:.4f} rad at {max_drift_joint} "
                    f"exceeds {args.max_drift:.4f}"
                )

            for area in CONTROL_AREAS:
                node.publishers_by_area[area].publish(
                    node.make_command(area, targets, sequence)
                )
            sequence += 1
            next_tick += 0.002
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        elapsed = time.monotonic() - start
        print(
            f"PASS elapsed={elapsed:.3f}s cycles={sequence} "
            f"rate={sequence / elapsed:.1f}Hz max_drift={max_drift_seen:.5f}rad"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
