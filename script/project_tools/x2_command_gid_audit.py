#!/usr/bin/env python3
"""Finite command subscriber: identify received writers by DDS publisher GID."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import time


def emit(event, **values):
    print(json.dumps({"event": event,
                      "utc": datetime.now(timezone.utc).isoformat(),
                      "mono_s": time.monotonic(), **values}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=40)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error("seconds must be in (0, 120]")
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointCommandArray

    rclpy.init(args=[])
    node = rclpy.create_node("x2_command_gid_readonly_audit")
    counts = {group: Counter() for group in ("leg", "waist", "arm", "head")}
    topics = {group: f"/aima/hal/joint/{group}/command" for group in counts}
    endpoints = {}
    known = {group: {} for group in counts}
    subs = []
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)

    def make_callback(group):
        def callback(message, info):
            gid = bytes(info.publisher_gid).hex()
            counts[group][gid] += 1
        return callback

    for group, topic in topics.items():
        subs.append(node.create_subscription(
            JointCommandArray, topic, make_callback(group), qos))
    emit("start", seconds=args.seconds, scope="subscription-only; no signals or HAL publication")
    started = time.monotonic()
    next_report = started
    try:
        while time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() >= next_report:
                for group, topic in topics.items():
                    current = {
                        bytes(e.endpoint_gid).hex(): e.node_name
                        for e in node.get_publishers_info_by_topic(topic)
                    }
                    known[group].update(current)
                    if current != endpoints.get(group):
                        emit("endpoints", group=group, publishers=current)
                        endpoints[group] = current
                emit("sample", received_by_gid={g: dict(c) for g, c in counts.items()})
                next_report = time.monotonic() + 1
        unexpected = {
            g: {gid: n for gid, n in counts[g].items()
                if known[g].get(gid) != "mc_ros2_node"}
            for g in counts
        }
        passed = all(counts.values()) and not any(unexpected.values())
        emit("end", official_only=passed, received_by_gid={g: dict(c) for g, c in counts.items()},
             known_publishers=known, unexpected=unexpected)
        return 0 if passed else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
