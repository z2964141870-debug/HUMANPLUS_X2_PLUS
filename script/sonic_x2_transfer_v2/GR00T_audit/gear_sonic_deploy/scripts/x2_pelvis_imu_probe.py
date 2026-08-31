#!/usr/bin/env python3
"""Read-only X2 torso-IMU to pelvis reconstruction probe."""

import argparse
import math
import statistics
import time

import rclpy
from aimdk_msgs.msg import JointStateArray
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Imu


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qnorm(q):
    norm = math.sqrt(sum(x * x for x in q))
    return tuple(x / norm for x in q)


def waist_quat(yaw, pitch, roll):
    hy, hp, hr = 0.5 * yaw, 0.5 * pitch, 0.5 * roll
    qz = (math.cos(hy), 0.0, 0.0, math.sin(hy))
    qy = (math.cos(hp), 0.0, math.sin(hp), 0.0)
    qx = (math.cos(hr), math.sin(hr), 0.0, 0.0)
    return qnorm(qmul(qmul(qz, qy), qx))


def pelvis_quat(torso_q, waist):
    qpt = waist_quat(*waist)
    return qnorm(qmul(qnorm(torso_q), (qpt[0], -qpt[1], -qpt[2], -qpt[3])))


def roll_pitch_deg(q):
    w, x, y, z = qnorm(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


class Probe(Node):
    def __init__(self):
        super().__init__("x2_pelvis_imu_probe")
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.torso = None
        self.waist = None
        self.samples = []
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.on_imu, qos)
        self.create_subscription(
            JointStateArray, "/aima/hal/joint/waist/state", self.on_waist, qos
        )

    def on_imu(self, msg):
        self.torso = (
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        )
        self.capture()

    def on_waist(self, msg):
        by_name = {joint.name: joint.position for joint in msg.joints}
        names = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
        if all(name in by_name for name in names):
            self.waist = tuple(by_name[name] for name in names)
            self.capture()

    def capture(self):
        if self.torso is None or self.waist is None:
            return
        raw_roll, raw_pitch = roll_pitch_deg(self.torso)
        pelvis_roll, pelvis_pitch = roll_pitch_deg(
            pelvis_quat(self.torso, self.waist)
        )
        self.samples.append(
            (raw_roll, raw_pitch, pelvis_roll, pelvis_pitch, *self.waist)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    deadline = time.monotonic() + args.seconds
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            # State topics arrive at 500-1000 Hz. A 20 ms single-callback
            # poll can repeatedly service only the IMU and starve the waist
            # callback, producing a false "no synchronized samples" result.
            rclpy.spin_once(node, timeout_sec=0.001)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not node.samples:
        raise SystemExit("FAIL: no synchronized torso/waist samples")
    med = [statistics.median(column) for column in zip(*node.samples)]
    print(
        "PELVIS_IMU_PROBE "
        f"samples={len(node.samples)} "
        f"raw_roll/pitch={med[0]:+.2f}/{med[1]:+.2f}deg "
        f"pelvis_roll/pitch={med[2]:+.2f}/{med[3]:+.2f}deg "
        f"waist_y/p/r={math.degrees(med[4]):+.2f}/"
        f"{math.degrees(med[5]):+.2f}/{math.degrees(med[6]):+.2f}deg"
    )


if __name__ == "__main__":
    main()
