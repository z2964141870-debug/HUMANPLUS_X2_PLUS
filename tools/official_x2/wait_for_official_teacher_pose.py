#!/usr/bin/env python3
"""Wait until official X2 joint state reaches a saved teacher handoff pose."""

from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from aimdk_msgs.msg import JointStateArray
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class PoseWaiter(Node):
    def __init__(self, archive: str, threshold: float, timeout: float) -> None:
        super().__init__("wait_for_official_teacher_pose")
        data = np.load(archive, allow_pickle=False)
        self.names = tuple(data["action_joint_order"].tolist())
        self.target = np.asarray(data["source_joint_q_rad"], dtype=np.float32)
        self.threshold = threshold
        self.deadline = time.monotonic() + timeout
        self.state: dict[str, float] = {}
        self.best_rmse = float("inf")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for area in ("leg", "waist", "arm", "head"):
            self.create_subscription(JointStateArray, f"/aima/hal/joint/{area}/state", self._callback, qos)

    def _callback(self, message: JointStateArray) -> None:
        for joint in message.joints:
            self.state[joint.name] = float(joint.position)

    def poll(self) -> bool:
        if all(name in self.state for name in self.names):
            current = np.asarray([self.state[name] for name in self.names], dtype=np.float32)
            rmse = float(np.sqrt(np.mean((current - self.target) ** 2)))
            self.best_rmse = min(self.best_rmse, rmse)
            if rmse <= self.threshold:
                print(f"matched rmse_rad={rmse:.6f}", flush=True)
                return True
        if time.monotonic() >= self.deadline:
            print(f"timeout best_rmse_rad={self.best_rmse:.6f}", flush=True)
            raise TimeoutError("official policy did not reach saved teacher pose")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--rmse-threshold", type=float, default=0.08)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    rclpy.init()
    node = PoseWaiter(args.archive, args.rmse_threshold, args.timeout)
    try:
        while rclpy.ok() and not node.poll():
            rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
