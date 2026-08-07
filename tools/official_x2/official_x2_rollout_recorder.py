#!/usr/bin/env python3
"""Record a synchronized 50 Hz rollout from the official X2 ROS contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from aimdk_msgs.msg import JointCommandArray, JointStateArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)


class Recorder(Node):
    def __init__(self, duration: float):
        super().__init__("official_x2_rollout_recorder")
        self.duration = duration
        self.state = {}
        self.command = {}
        self.imu = None
        self.odom = None
        self.start = None
        self.rows = []
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for area in ("leg", "waist", "arm", "head"):
            self.create_subscription(JointStateArray, f"/aima/hal/joint/{area}/state", self.state_cb, qos)
            self.create_subscription(JointCommandArray, f"/aima/hal/joint/{area}/command", self.command_cb, qos)
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.imu_cb, qos)
        self.create_subscription(Odometry, "/aima/hal/odom/state", self.odom_cb, qos)
        self.create_timer(0.02, self.sample)

    def state_cb(self, msg):
        for joint in msg.joints:
            self.state[joint.name] = (float(joint.position), float(joint.velocity), float(joint.effort))

    def command_cb(self, msg):
        for joint in msg.joints:
            self.command[joint.name] = (
                float(joint.position), float(joint.velocity), float(joint.effort),
                float(joint.stiffness), float(joint.damping),
            )

    def imu_cb(self, msg):
        self.imu = msg

    def odom_cb(self, msg):
        self.odom = msg

    def ready(self):
        return self.imu is not None and self.odom is not None and all(name in self.state for name in JOINTS)

    def sample(self):
        if not self.ready():
            return
        now = time.monotonic()
        if self.start is None:
            self.start = now
        elapsed = now - self.start
        if elapsed > self.duration:
            return
        q = np.asarray([self.state[name][0] for name in JOINTS], np.float32)
        dq = np.asarray([self.state[name][1] for name in JOINTS], np.float32)
        tau = np.asarray([self.state[name][2] for name in JOINTS], np.float32)
        command_valid = np.asarray([name in self.command for name in JOINTS], np.bool_)
        command = np.asarray([self.command.get(name, (np.nan,) * 5) for name in JOINTS], np.float32)
        op = self.odom.pose.pose
        ot = self.odom.twist.twist
        iq = self.imu.orientation
        iw = self.imu.angular_velocity
        ia = self.imu.linear_acceleration
        self.rows.append((
            elapsed, q, dq, tau, command, command_valid,
            np.asarray([op.position.x, op.position.y, op.position.z], np.float32),
            np.asarray([op.orientation.x, op.orientation.y, op.orientation.z, op.orientation.w], np.float32),
            np.asarray([ot.linear.x, ot.linear.y, ot.linear.z], np.float32),
            np.asarray([ot.angular.x, ot.angular.y, ot.angular.z], np.float32),
            np.asarray([iq.x, iq.y, iq.z, iq.w], np.float32),
            np.asarray([iw.x, iw.y, iw.z], np.float32),
            np.asarray([ia.x, ia.y, ia.z], np.float32),
        ))


def write_outputs(node: Recorder, output: Path):
    rows = node.rows
    if not rows:
        raise RuntimeError("no synchronized X2 samples were recorded")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joint_names=np.asarray(JOINTS),
        time_s=np.asarray([x[0] for x in rows], np.float64),
        joint_q_rad=np.stack([x[1] for x in rows]),
        joint_dq_radps=np.stack([x[2] for x in rows]),
        joint_effort=np.stack([x[3] for x in rows]),
        command_q_rad=np.stack([x[4][:, 0] for x in rows]),
        command_dq_radps=np.stack([x[4][:, 1] for x in rows]),
        command_effort=np.stack([x[4][:, 2] for x in rows]),
        command_kp=np.stack([x[4][:, 3] for x in rows]),
        command_kd=np.stack([x[4][:, 4] for x in rows]),
        command_valid=np.stack([x[5] for x in rows]),
        root_pos_w_m=np.stack([x[6] for x in rows]),
        root_quat_xyzw=np.stack([x[7] for x in rows]),
        root_lin_vel_w_mps=np.stack([x[8] for x in rows]),
        root_ang_vel=np.stack([x[9] for x in rows]),
        torso_imu_quat_xyzw=np.stack([x[10] for x in rows]),
        torso_imu_ang_vel=np.stack([x[11] for x in rows]),
        torso_imu_lin_acc=np.stack([x[12] for x in rows]),
        source=np.asarray("AimDK X2 v1.0 official MuJoCo + shipped kuailechongbai.onnx"),
        truth_label=np.asarray("official_simulation_truth_not_real_hardware_truth"),
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "source": "AimDK X2 v1.0 official MuJoCo + shipped kuailechongbai.onnx",
        "truth_label": "official_simulation_truth_not_real_hardware_truth",
        "sample_rate_hz": 50,
        "samples": len(rows),
        "duration_s": float(rows[-1][0]),
        "joint_count": len(JOINTS),
        "npz": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.duration)
    wall = time.monotonic()
    while rclpy.ok() and (node.start is None or time.monotonic() - node.start <= args.duration + 0.1):
        if node.start is None and time.monotonic() - wall > 30.0:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    write_outputs(node, Path(args.output))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
