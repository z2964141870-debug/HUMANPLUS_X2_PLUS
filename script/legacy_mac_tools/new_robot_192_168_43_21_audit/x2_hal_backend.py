#!/usr/bin/env python3
"""Guarded 500 Hz HAL backend for X2 Sonic teleoperation.

The 50 Hz policy only updates a target.  A private worker interpolates and
publishes the 29 supported body joints at 500 Hz.  No head command is emitted
because this robot has a persistent head motor error 1026.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


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
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
)

GROUPS = {
    "leg": (0, 12),
    "waist": (12, 15),
    "arm": (15, 29),
}

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
    **{name: (20.0, 2.0) for name in JOINT_NAMES[15:29]},
}

# Gains captured from this robot's official MC JOINT_DEFAULT HAL command
# stream.  They are used only during the explicit Stand transition.  The
# transition starts at measured q, so selecting this profile does not create
# a position error at the handoff.
STAND_GAINS = dict(GAINS)
STAND_GAINS.update({
    "waist_yaw_joint": (150.0, 3.0),
    "waist_pitch_joint": (300.0, 3.0),
    "waist_roll_joint": (300.0, 3.0),
    "left_shoulder_pitch_joint": (30.0, 1.0),
    "left_shoulder_roll_joint": (20.0, 1.0),
    "left_shoulder_yaw_joint": (20.0, 1.0),
    "left_elbow_joint": (50.0, 1.0),
    "left_wrist_yaw_joint": (50.0, 1.0),
    "left_wrist_pitch_joint": (20.0, 1.0),
    "left_wrist_roll_joint": (20.0, 1.0),
    "right_shoulder_pitch_joint": (30.0, 1.0),
    "right_shoulder_roll_joint": (20.0, 1.0),
    "right_shoulder_yaw_joint": (20.0, 1.0),
    "right_elbow_joint": (50.0, 1.0),
    "right_wrist_yaw_joint": (50.0, 1.0),
    "right_wrist_pitch_joint": (20.0, 1.0),
    "right_wrist_roll_joint": (20.0, 1.0),
})
GAIN_PROFILES = {"hold": GAINS, "stand": STAND_GAINS}

# AimDK v0.9.0 documented limits in policy/MuJoCo order.  Head limits are
# retained for shape checks, although head commands are intentionally omitted.
LOWER_LIMITS = np.asarray([
    -2.704, -0.235, -1.684, 0.0, -0.803, -0.2625,
    -2.704, -2.906, -3.430, 0.0, -0.803, -0.2625,
    -3.430, -0.314, -0.488,
    -3.08, -0.061, -2.556, -2.3556, -2.556, -0.558, -1.571,
    -3.08, -2.993, -2.556, -2.3556, -2.556, -0.558, -0.724,
    -0.366, -0.3838,
], dtype=np.float64)
UPPER_LIMITS = np.asarray([
    2.556, 2.906, 3.430, 2.4073, 0.453, 0.2625,
    2.556, 0.235, 1.684, 2.4073, 0.453, 0.2625,
    2.382, 0.314, 0.488,
    2.04, 2.993, 2.556, 0.0, 2.556, 0.558, 0.724,
    2.04, 0.061, 2.556, 0.0, 2.556, 0.558, 1.571,
    0.366, 0.3838,
], dtype=np.float64)

# This robot's unloaded absolute-encoder state sits up to 0.077 rad outside
# the generic v0.9.0 example's soft limits at waist/shoulder roll.  Permit a
# small measured-pose holding margin so arming does not snap to the generic
# limit.  Sonic targets remain constrained by its stricter policy envelope.
MEASURED_HOLD_MARGIN_RAD = 0.10
HOLD_LOWER_LIMITS = LOWER_LIMITS - MEASURED_HOLD_MARGIN_RAD
HOLD_UPPER_LIMITS = UPPER_LIMITS + MEASURED_HOLD_MARGIN_RAD


class LinearTargetInterpolator:
    """Thread-safe caller-independent linear target interpolation state."""

    def __init__(self, transition_s: float = 0.02):
        self.transition_s = float(transition_s)
        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.started_at = 0.0

    def update(self, target: np.ndarray, now: float) -> None:
        target = np.asarray(target, dtype=np.float64).copy()
        current = target if self.goal is None else self.sample(now)
        self.start = current
        self.goal = target
        self.started_at = float(now)

    def sample(self, now: float) -> np.ndarray:
        if self.goal is None or self.start is None:
            raise RuntimeError("interpolator has no target")
        if self.transition_s <= 0.0:
            return self.goal.copy()
        alpha = np.clip((float(now) - self.started_at) / self.transition_s, 0.0, 1.0)
        return self.start + alpha * (self.goal - self.start)

    def clear(self) -> None:
        self.start = None
        self.goal = None
        self.started_at = 0.0


class HalTeleopBackend:
    """50 Hz target input -> guarded 500 Hz X2 HAL command publisher."""

    def __init__(self, *, dry_run: bool = True, enable_network: bool = False,
                 publish_hz: float = 500.0, policy_hz: float = 50.0,
                 watchdog_s: float = 0.10):
        if not dry_run and not enable_network:
            raise PermissionError("HAL publishing requires explicit real-control authorization")
        self.dry_run = bool(dry_run)
        self.publish_hz = float(publish_hz)
        self.watchdog_s = float(watchdog_s)
        self.last_qpos62 = np.zeros(62, dtype=np.float64)
        self.last_qpos62[2:4] = [0.65, 1.0]
        self._lock = threading.Lock()
        self._interpolator = LinearTargetInterpolator(1.0 / float(policy_hz))
        self._last_update = 0.0
        self._active = False
        self._gain_scale = 1.0
        self._gain_profile = "hold"
        self._fault: Optional[str] = None
        self._sequence = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._rclpy = None
        self._node = None
        self._publishers = {}
        self._JointCommand = None
        self._JointCommandArray = None
        self._last_dry_log = 0.0
        self._closed = False
        if not self.dry_run:
            self._initialize_ros()

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault

    def _initialize_ros(self) -> None:
        import rclpy
        from aimdk_msgs.msg import JointCommand, JointCommandArray
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        rclpy.init()
        self._rclpy = rclpy
        self._node = Node("x2_sonic_hal_backend")
        discovery_deadline = time.monotonic() + 2.0
        while time.monotonic() < discovery_deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        conflicts = {}
        for area in GROUPS:
            topic = f"/aima/hal/joint/{area}/command"
            owners = [info.node_name for info in self._node.get_publishers_info_by_topic(topic)]
            if owners:
                conflicts[area] = owners
        if conflicts:
            self._node.destroy_node()
            rclpy.shutdown()
            raise RuntimeError(f"HAL command topic already owned: {conflicts}")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._JointCommand = JointCommand
        self._JointCommandArray = JointCommandArray
        self._publishers = {
            area: self._node.create_publisher(
                JointCommandArray, f"/aima/hal/joint/{area}/command", qos
            )
            for area in GROUPS
        }
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, name="x2-hal-500hz", daemon=True)
        self._thread.start()

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None
            self._active = False
            self._interpolator.clear()

    def send(self, targets31: np.ndarray, root7=None, *, gain_scale: float = 1.0,
             gain_profile: str = "hold") -> bool:
        targets = np.asarray(targets31, dtype=np.float64).reshape(-1)
        if targets.shape != (31,) or not np.all(np.isfinite(targets)):
            raise ValueError(f"HAL target must be finite shape (31,), got {targets.shape}")
        gain_scale = float(gain_scale)
        if not np.isfinite(gain_scale) or not 0.0 <= gain_scale <= 1.0:
            raise ValueError("gain_scale must be finite and in [0, 1]")
        if gain_profile not in GAIN_PROFILES:
            raise ValueError(f"unknown gain profile: {gain_profile}")
        targets = np.clip(targets, HOLD_LOWER_LIMITS, HOLD_UPPER_LIMITS)
        self.last_qpos62[7:38] = targets
        now = time.monotonic()
        if self.dry_run:
            if now - self._last_dry_log >= 0.5:
                print(f"[hal-dry] body29={np.round(targets[:4], 3)} head=omitted")
                self._last_dry_log = now
            return True
        with self._lock:
            if self._fault is not None:
                raise RuntimeError(self._fault)
            self._interpolator.update(targets[:29], now)
            self._gain_scale = gain_scale
            self._gain_profile = gain_profile
            self._last_update = now
            self._active = True
        return True

    def stop(self) -> None:
        """Immediately stop publishing; HAL's own missing-command watchdog takes over."""
        with self._lock:
            self._active = False
            self._interpolator.clear()

    def _make_message(self, area: str, command29: np.ndarray, gain_scale: float,
                      gain_profile: str):
        start, stop = GROUPS[area]
        msg = self._JointCommandArray()
        now_msg = self._node.get_clock().now().to_msg()
        msg.header.stamp = now_msg
        msg.header.meas_stamp = now_msg
        msg.header.frame_id = "x2_sonic_hal_backend"
        msg.header.sequence = self._sequence
        for index in range(start, stop):
            name = JOINT_NAMES[index]
            kp, kd = GAIN_PROFILES[gain_profile][name]
            joint = self._JointCommand()
            joint.name = name
            joint.position = float(command29[index])
            joint.velocity = 0.0
            joint.effort = 0.0
            joint.stiffness = kp * gain_scale
            joint.damping = kd * gain_scale
            msg.joints.append(joint)
        return msg

    def _publish_loop(self) -> None:
        period = 1.0 / self.publish_hz
        next_tick = time.monotonic()
        while self._running:
            now = time.monotonic()
            command = None
            gain_scale = 1.0
            gain_profile = "hold"
            with self._lock:
                if self._active:
                    if now - self._last_update > self.watchdog_s:
                        self._active = False
                        self._fault = "hal_command_watchdog_timeout"
                        self._interpolator.clear()
                    else:
                        command = self._interpolator.sample(now)
                        gain_scale = self._gain_scale
                        gain_profile = self._gain_profile
            if command is not None:
                for area, publisher in self._publishers.items():
                    publisher.publish(self._make_message(
                        area, command, gain_scale, gain_profile
                    ))
                self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
