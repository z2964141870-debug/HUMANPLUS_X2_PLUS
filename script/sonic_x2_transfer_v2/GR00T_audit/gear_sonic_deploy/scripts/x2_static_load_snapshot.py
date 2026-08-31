#!/usr/bin/env python3
"""Capture a read-only X2 static-load snapshot from HAL and MC topics."""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict

import rclpy
from aimdk_msgs.msg import JointCommandArray, JointStateArray
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Imu


GROUPS = ("leg", "waist", "arm", "head")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
    }


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
    norm = math.sqrt(sum(value * value for value in q))
    return tuple(value / norm for value in q)


def waist_quat(yaw, pitch, roll):
    hy, hp, hr = 0.5 * yaw, 0.5 * pitch, 0.5 * roll
    qz = (math.cos(hy), 0.0, 0.0, math.sin(hy))
    qy = (math.cos(hp), 0.0, math.sin(hp), 0.0)
    qx = (math.cos(hr), math.sin(hr), 0.0, 0.0)
    return qnorm(qmul(qmul(qz, qy), qx))


def roll_pitch_deg(q):
    w, x, y, z = qnorm(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


class StaticLoadProbe(Node):
    def __init__(self):
        super().__init__("x2_static_load_snapshot")
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.started = time.monotonic()
        self.state_counts = defaultdict(int)
        self.command_counts = defaultdict(int)
        self.domain_states = defaultdict(list)
        self.state_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.command_values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.latest_waist = None
        self.imu_values = defaultdict(list)

        for group in GROUPS:
            self.create_subscription(
                JointStateArray,
                f"/aima/hal/joint/{group}/state",
                lambda message, group=group: self.on_state(group, message),
                qos,
            )
            self.create_subscription(
                JointCommandArray,
                f"/aima/hal/joint/{group}/command",
                lambda message, group=group: self.on_command(group, message),
                qos,
            )
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.on_imu, qos)

    def on_state(self, group, message):
        self.state_counts[group] += 1
        self.domain_states[group].append(int(message.state.value))
        waist = {}
        for joint in message.joints:
            values = self.state_values[group][joint.name]
            values["position"].append(float(joint.position))
            values["velocity"].append(float(joint.velocity))
            values["effort"].append(float(joint.effort))
            values["coil_temp"].append(float(joint.coil_temp))
            values["motor_temp"].append(float(joint.motor_temp))
            values["motor_voltage"].append(float(joint.motor_vol))
            waist[joint.name] = float(joint.position)
        if group == "waist":
            names = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
            if all(name in waist for name in names):
                self.latest_waist = tuple(waist[name] for name in names)

    def on_command(self, group, message):
        self.command_counts[group] += 1
        for joint in message.joints:
            values = self.command_values[group][joint.name]
            values["position"].append(float(joint.position))
            values["velocity"].append(float(joint.velocity))
            values["effort"].append(float(joint.effort))
            values["stiffness"].append(float(joint.stiffness))
            values["damping"].append(float(joint.damping))

    def on_imu(self, message):
        torso_q = (
            float(message.orientation.w),
            float(message.orientation.x),
            float(message.orientation.y),
            float(message.orientation.z),
        )
        raw_roll, raw_pitch = roll_pitch_deg(torso_q)
        self.imu_values["torso_roll_deg"].append(raw_roll)
        self.imu_values["torso_pitch_deg"].append(raw_pitch)
        if self.latest_waist is None:
            return
        waist_q = waist_quat(*self.latest_waist)
        pelvis_q = qnorm(
            qmul(torso_q, (waist_q[0], -waist_q[1], -waist_q[2], -waist_q[3]))
        )
        pelvis_roll, pelvis_pitch = roll_pitch_deg(pelvis_q)
        self.imu_values["pelvis_roll_deg"].append(pelvis_roll)
        self.imu_values["pelvis_pitch_deg"].append(pelvis_pitch)

    def command_graph(self):
        result = {}
        for group in GROUPS:
            topic = f"/aima/hal/joint/{group}/command"
            infos = self.get_publishers_info_by_topic(topic)
            result[group] = {
                "publisher_count": len(infos),
                "publishers": sorted(
                    {f"{info.node_namespace}/{info.node_name}" for info in infos}
                ),
            }
        return result

    def result(self, label, elapsed):
        states = {}
        commands = {}
        for group in GROUPS:
            states[group] = {
                "samples": self.state_counts[group],
                "rate_hz": self.state_counts[group] / elapsed,
                "domain_states": sorted(set(self.domain_states[group])),
                "joints": {
                    name: {field: summarize(values) for field, values in fields.items()}
                    for name, fields in sorted(self.state_values[group].items())
                },
            }
            commands[group] = {
                "samples": self.command_counts[group],
                "rate_hz": self.command_counts[group] / elapsed,
                "joints": {
                    name: {field: summarize(values) for field, values in fields.items()}
                    for name, fields in sorted(self.command_values[group].items())
                },
            }
        return {
            "label": label,
            "elapsed_s": elapsed,
            "read_only": True,
            "command_graph": self.command_graph(),
            "states": states,
            "commands": commands,
            "imu": {
                "samples": len(self.imu_values["torso_roll_deg"]),
                "values": {
                    name: summarize(values)
                    for name, values in sorted(self.imu_values.items())
                    if values
                },
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rclpy.init()
    node = StaticLoadProbe()
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.001)
        elapsed = time.monotonic() - started
        result = node.result(args.label, elapsed)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if any(result["states"][group]["samples"] == 0 for group in GROUPS):
        raise SystemExit("FAIL: one or more HAL state groups produced no samples")
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        "STATIC_LOAD_SNAPSHOT "
        f"label={args.label} elapsed={elapsed:.3f}s "
        f"state_rates="
        + ",".join(
            f"{group}:{result['states'][group]['rate_hz']:.1f}Hz"
            for group in GROUPS
        )
        + " publishers="
        + ",".join(
            f"{group}:{result['command_graph'][group]['publisher_count']}"
            for group in GROUPS
        )
    )


if __name__ == "__main__":
    main()
