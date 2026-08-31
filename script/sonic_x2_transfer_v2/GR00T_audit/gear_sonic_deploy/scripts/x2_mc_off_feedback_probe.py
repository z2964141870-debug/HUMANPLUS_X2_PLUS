#!/usr/bin/env python3
"""Measure one X2 HAL state stream without creating any command publisher."""

import argparse
import math
import time

import rclpy
from aimdk_msgs.msg import JointStateArray
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles


class FeedbackProbe(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("x2_mc_off_feedback_probe")
        self.count = 0
        self.first_time = math.nan
        self.last_time = math.nan
        self.max_gap = 0.0
        self.subscription = self.create_subscription(
            JointStateArray,
            topic,
            self._on_state,
            QoSPresetProfiles.SENSOR_DATA.value,
        )

    def _on_state(self, _message: JointStateArray) -> None:
        now = time.monotonic()
        if self.count == 0:
            self.first_time = now
        else:
            self.max_gap = max(self.max_gap, now - self.last_time)
        self.last_time = now
        self.count += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic", default="/aima/hal/joint/leg/state"
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--stale-seconds", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = FeedbackProbe(args.topic)
    started = time.monotonic()
    next_report = started + 1.0
    stale_events = 0
    stale_latched = False
    try:
        while rclpy.ok() and time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            age = math.inf if node.count == 0 else now - node.last_time
            stale = age > args.stale_seconds
            if stale and not stale_latched:
                stale_events += 1
            stale_latched = stale
            if now >= next_report:
                print(
                    f"PROBE count={node.count} age={age:.3f}s "
                    f"max_gap={node.max_gap:.3f}s stale_events={stale_events}",
                    flush=True,
                )
                next_report += 1.0
    finally:
        elapsed = time.monotonic() - started
        rate = node.count / elapsed if elapsed > 0.0 else 0.0
        age = math.inf if node.count == 0 else time.monotonic() - node.last_time
        print(
            f"RESULT topic={args.topic} elapsed={elapsed:.3f}s "
            f"count={node.count} rate={rate:.1f}Hz age={age:.3f}s "
            f"max_gap={node.max_gap:.3f}s stale_events={stale_events}",
            flush=True,
        )
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
