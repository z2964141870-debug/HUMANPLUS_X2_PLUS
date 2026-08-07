#!/usr/bin/env python3
"""Run the frozen Stage208 lower-body actor against AimDK X2 MuJoCo.

This is an evaluation adapter, not a training path.  It reconstructs the exact
93-D Stage208 actor observation and the 15-D gait-template residual contract,
then publishes X2 joint-position PD commands through the official ROS topics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


ISAAC_JOINTS = (
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

LOWER_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)

LEG_JOINTS = LOWER_JOINTS[:12]
WAIST_JOINTS = LOWER_JOINTS[12:]
ARM_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")

# PD gains emitted by the official kuailechongbai policy for the 15 lower/waist
# joints.  Keep this separate from Stage208's training-time 300/20 profile so
# the official-domain gain A/B is explicit and reproducible.
OFFICIAL_NATIVE_LOWER_PD = {
    "left_hip_pitch_joint": (120.0, 5.0),
    "right_hip_pitch_joint": (120.0, 5.0),
    "left_hip_roll_joint": (100.0, 4.0),
    "right_hip_roll_joint": (100.0, 4.0),
    "left_hip_yaw_joint": (100.0, 4.0),
    "right_hip_yaw_joint": (100.0, 4.0),
    "left_knee_joint": (150.0, 5.0),
    "right_knee_joint": (150.0, 5.0),
    "left_ankle_pitch_joint": (40.0, 2.0),
    "right_ankle_pitch_joint": (40.0, 2.0),
    "left_ankle_roll_joint": (40.0, 2.0),
    "right_ankle_roll_joint": (40.0, 2.0),
    "waist_yaw_joint": (40.1792, 2.5579),
    "waist_pitch_joint": (200.0, 2.0),
    "waist_roll_joint": (200.0, 2.0),
}

DEFAULT = {name: 0.0 for name in ISAAC_JOINTS}
for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
    DEFAULT[name] = -0.248
for name in ("left_knee_joint", "right_knee_joint"):
    DEFAULT[name] = 0.5303
for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
    DEFAULT[name] = -0.2823
for name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
    DEFAULT[name] = 0.4
for name in ("left_elbow_joint", "right_elbow_joint"):
    DEFAULT[name] = -1.2

LOWER_SCALE = np.asarray(
    [0.4, 0.4, 0.4, 0.4, 0.12, 0.08, 0.4, 0.4, 0.4, 0.4, 0.12, 0.08, 0.4, 0.16, 0.16],
    dtype=np.float32,
)


def gravity_body(imu: Imu) -> np.ndarray:
    q = imu.orientation
    return np.asarray(
        [
            2.0 * (-q.z * q.x + q.w * q.y),
            -2.0 * (q.z * q.y + q.w * q.x),
            1.0 - 2.0 * (q.w * q.w + q.z * q.z),
        ],
        dtype=np.float32,
    )


def tilt_from_quaternion(odom: Odometry) -> float:
    q = odom.pose.pose.orientation
    gravity = np.asarray(
        [
            2.0 * (-q.z * q.x + q.w * q.y),
            -2.0 * (q.z * q.y + q.w * q.x),
            1.0 - 2.0 * (q.w * q.w + q.z * q.z),
        ]
    )
    return float(math.acos(np.clip(-gravity[2], -1.0, 1.0)))


def yaw_from_quaternion(odom: Odometry) -> float:
    q = odom.pose.pose.orientation
    return float(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def world_vector_to_body(odom: Odometry, vector: np.ndarray) -> np.ndarray:
    """Rotate an odometry world-frame vector into the pelvis/body frame."""
    q = odom.pose.pose.orientation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    rotation_body_to_world = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation_body_to_world.T @ vector


class Stage208OfficialAdapter(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("stage208_official_mujoco_adapter")
        self.args = args
        self.session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        archive = np.load(args.template, allow_pickle=False)
        if tuple(archive["joint_names_15"].tolist()) != LOWER_JOINTS:
            raise RuntimeError("gait-template joint order does not match Stage208")
        if not np.allclose(archive["action_scale_rad"], LOWER_SCALE, atol=1e-6, rtol=0.0):
            raise RuntimeError("gait-template action scale does not match Stage208")
        self.template = archive["q_cycle_zero_mean_rad"].astype(np.float32)
        self.period = float(archive["period_s"])
        self.double_support_fraction = float(archive["double_support_fraction"])
        self.replay_targets: np.ndarray | None = None
        if args.control_mode == "isaac_target_replay":
            if not args.joint_target_trace:
                raise ValueError("--joint-target-trace is required for isaac_target_replay")
            replay = np.load(args.joint_target_trace, allow_pickle=False)
            if tuple(replay["action_joint_order"].tolist()) != LOWER_JOINTS:
                raise RuntimeError("Isaac replay target joint order does not match Stage208")
            self.replay_targets = replay["joint_target_rad"].astype(np.float32)
            if self.replay_targets.ndim != 2 or self.replay_targets.shape[1] != 15:
                raise RuntimeError(f"invalid replay target shape {self.replay_targets.shape}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.joints: dict[str, tuple[float, float]] = {}
        self.imu: Imu | None = None
        self.odom: Odometry | None = None
        for area in ("leg", "waist", "arm", "head"):
            self.create_subscription(
                JointStateArray,
                f"/aima/hal/joint/{area}/state",
                self._joint_callback,
                qos,
            )
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self._imu_callback, qos)
        self.create_subscription(Odometry, "/aima/hal/odom/state", self._odom_callback, qos)
        self.command_publishers = {
            area: self.create_publisher(JointCommandArray, f"/aima/hal/joint/{area}/command", qos)
            for area in ("leg", "waist", "arm", "head")
        }

        self.previous_action = np.zeros(15, dtype=np.float32)
        self.heading_target_rad: float | None = None
        self.ready_wall_time: float | None = None
        self.prepare_start_q: dict[str, float] = {}
        self.control_steps = 0
        self.trace: list[dict[str, float | list[float] | str]] = []
        self.finished = False
        self.timer = self.create_timer(0.02, self._control)

    def _joint_callback(self, msg: JointStateArray) -> None:
        for joint in msg.joints:
            self.joints[joint.name] = (float(joint.position), float(joint.velocity))

    def _imu_callback(self, msg: Imu) -> None:
        self.imu = msg

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom = msg

    def _state_ready(self) -> bool:
        return all(name in self.joints for name in ISAAC_JOINTS) and self.imu is not None and self.odom is not None

    def _pd(self, name: str) -> tuple[float, float]:
        if name in LOWER_JOINTS:
            if self.args.pd_profile == "official_native":
                return OFFICIAL_NATIVE_LOWER_PD[name]
            if self.args.pd_profile == "official_kp_only":
                return OFFICIAL_NATIVE_LOWER_PD[name][0], 20.0
            if self.args.pd_profile == "official_kd_only":
                return 300.0, OFFICIAL_NATIVE_LOWER_PD[name][1]
            group_match = (
                self.args.pd_profile == "official_kp_proximal"
                and any(token in name for token in ("hip", "knee"))
            ) or (
                self.args.pd_profile == "official_kp_ankle" and "ankle" in name
            ) or (
                self.args.pd_profile == "official_kp_waist" and "waist" in name
            )
            if group_match:
                return OFFICIAL_NATIVE_LOWER_PD[name][0], 20.0
            return 300.0, 20.0
        if "shoulder" in name or "elbow" in name:
            return 40.0, 5.0
        if "wrist" in name:
            return 30.0, 3.0
        return 50.0, 5.0

    def _publish_group(self, area: str, names: tuple[str, ...], targets: dict[str, float]) -> None:
        message = JointCommandArray()
        commands = []
        for name in names:
            kp, kd = self._pd(name)
            command = JointCommand()
            command.name = name
            command.position = float(targets[name])
            command.velocity = 0.0
            command.effort = 0.0
            command.stiffness = kp
            command.damping = kd
            commands.append(command)
        message.joints = commands
        self.command_publishers[area].publish(message)

    def _publish(self, targets: dict[str, float]) -> None:
        self._publish_group("leg", LEG_JOINTS, targets)
        self._publish_group("waist", WAIST_JOINTS, targets)
        self._publish_group("arm", ARM_JOINTS, targets)
        self._publish_group("head", HEAD_JOINTS, targets)

    def _phase_features(self, elapsed: float, moving: bool) -> np.ndarray:
        if not moving:
            return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        phase = (elapsed / self.period) % 1.0
        half_ds_width = self.double_support_fraction / 4.0
        right_swing = half_ds_width <= phase < 0.5 - half_ds_width
        left_swing = 0.5 + half_ds_width <= phase < 1.0 - half_ds_width
        return np.asarray(
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase), not left_swing, not right_swing],
            dtype=np.float32,
        )

    def _template_bias(self, elapsed: float) -> np.ndarray:
        phase = (elapsed / self.period) % 1.0
        bins = self.template.shape[0]
        phase_position = phase * bins - 0.5
        lower_unwrapped = math.floor(phase_position)
        blend = phase_position - lower_unwrapped
        lower = lower_unwrapped % bins
        upper = (lower + 1) % bins
        return (1.0 - blend) * self.template[lower] + blend * self.template[upper]

    def _policy_targets(self, phase_elapsed: float, command_vx: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        assert self.imu is not None and self.odom is not None
        twist = self.odom.twist.twist
        base_lin_vel_w = np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float32)
        base_lin_vel = world_vector_to_body(self.odom, base_lin_vel_w)
        omega = self.imu.angular_velocity
        base_ang_vel = np.asarray([omega.x, omega.y, omega.z], dtype=np.float32)
        current_yaw = yaw_from_quaternion(self.odom)
        if self.heading_target_rad is None:
            self.heading_target_rad = current_yaw
        heading_error = math.atan2(
            math.sin(self.heading_target_rad - current_yaw),
            math.cos(self.heading_target_rad - current_yaw),
        )
        command_wz = float(
            np.clip(
                self.args.heading_gain * heading_error,
                -self.args.heading_rate_limit,
                self.args.heading_rate_limit,
            )
        )
        if self.args.fixed_wz is not None and abs(command_vx) > 0.1:
            command_wz = self.args.fixed_wz
        command = np.asarray([command_vx, 0.0, command_wz], dtype=np.float32)
        joint_pos = np.asarray([self.joints[name][0] - DEFAULT[name] for name in ISAAC_JOINTS], dtype=np.float32)
        joint_vel = np.asarray([self.joints[name][1] for name in ISAAC_JOINTS], dtype=np.float32)
        moving = abs(command_vx) > 0.1
        phase = self._phase_features(phase_elapsed, moving=moving)
        obs = np.concatenate(
            (base_lin_vel, base_ang_vel, gravity_body(self.imu), command, joint_pos, joint_vel, self.previous_action, phase)
        ).astype(np.float32)
        if obs.shape != (93,):
            raise RuntimeError(f"invalid Stage208 observation shape {obs.shape}")
        raw_action = self.session.run(["actions"], {"obs": obs[None]})[0][0].astype(np.float32)
        residual = raw_action.copy()
        if self.args.control_mode == "template_only":
            residual.fill(0.0)
        template_bias = (
            0.15 * self._template_bias(phase_elapsed) / LOWER_SCALE
            if moving
            else np.zeros(15, dtype=np.float32)
        )
        if self.args.control_mode == "actor_only":
            template_bias.fill(0.0)
        combined_preclip = residual + template_bias
        if moving and self.args.action_bias_mode == "hip_yaw_feedback":
            feedback_bias = float(
                np.clip(
                    self.args.yaw_action_gain * heading_error,
                    -self.args.action_bias,
                    self.args.action_bias,
                )
            )
            combined_preclip[[2, 8]] += feedback_bias
        elif moving and self.args.action_bias != 0.0:
            if self.args.action_bias_mode == "hip_yaw_common":
                combined_preclip[[2, 8]] += self.args.action_bias
            elif self.args.action_bias_mode == "waist_yaw":
                combined_preclip[12] += self.args.action_bias
            elif self.args.action_bias_mode == "hip_waist_counter":
                combined_preclip[[2, 8]] += self.args.action_bias
                combined_preclip[12] -= self.args.action_bias
        combined = np.clip(combined_preclip, -1.0, 1.0)
        targets = dict(DEFAULT)
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] = DEFAULT[name] + float(combined[index] * LOWER_SCALE[index])
        # IsaacLab's last_action observation is the unbounded raw policy action;
        # clipping happens only after adding the normalized gait-template bias.
        self.previous_action = residual.copy()
        return targets, obs, combined

    def _replay_target(self, elapsed: float) -> dict[str, float]:
        assert self.replay_targets is not None
        index = min(int(elapsed / 0.02), self.replay_targets.shape[0] - 1)
        targets = dict(DEFAULT)
        for joint_index, name in enumerate(LOWER_JOINTS):
            targets[name] = float(self.replay_targets[index, joint_index])
        return targets

    def _record(self, stage: str, elapsed: float, obs: np.ndarray | None = None, action: np.ndarray | None = None) -> None:
        assert self.odom is not None
        pose = self.odom.pose.pose
        twist = self.odom.twist.twist
        self.trace.append(
            {
                "stage": stage,
                "elapsed_s": elapsed,
                "root_x_m": float(pose.position.x),
                "root_y_m": float(pose.position.y),
                "root_z_m": float(pose.position.z),
                "root_tilt_rad": tilt_from_quaternion(self.odom),
                "root_yaw_rad": yaw_from_quaternion(self.odom),
                "root_vx_w_mps": float(twist.linear.x),
                "root_vy_w_mps": float(twist.linear.y),
                "root_yaw_rate_radps": float(twist.angular.z),
                "obs": [] if obs is None else obs.tolist(),
                "action": [] if action is None else action.tolist(),
            }
        )

    def _control(self) -> None:
        if self.finished or not self._state_ready():
            return
        now = time.monotonic()
        if self.ready_wall_time is None:
            self.ready_wall_time = now
            self.prepare_start_q = {name: self.joints[name][0] for name in ISAAC_JOINTS}
            self.get_logger().info(
                f"state ready; begin {self.args.prepare_seconds:.2f} s matched-pose interpolation"
            )
        elapsed = now - self.ready_wall_time

        if elapsed < self.args.prepare_seconds:
            alpha = min(1.0, elapsed / self.args.prepare_seconds)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            targets = {
                name: self.prepare_start_q[name] + alpha * (DEFAULT[name] - self.prepare_start_q[name])
                for name in ISAAC_JOINTS
            }
            self._publish(targets)
            self._record("prepare", elapsed)
            return

        stand_elapsed = elapsed - self.args.prepare_seconds
        if stand_elapsed < self.args.stand_seconds:
            targets, obs, action = self._policy_targets(0.0, 0.0)
            self._publish(targets)
            self._record("stand", stand_elapsed, obs, action)
            return

        move_elapsed = stand_elapsed - self.args.stand_seconds
        if move_elapsed >= self.args.move_seconds:
            stop_elapsed = move_elapsed - self.args.move_seconds
            if stop_elapsed >= self.args.stop_seconds:
                self._finish()
                return
            targets, obs, action = self._policy_targets(self.args.move_seconds, 0.0)
            self._publish(targets)
            self._record("stop", stop_elapsed, obs, action)
            return
        if self.args.control_mode == "isaac_target_replay":
            targets = self._replay_target(move_elapsed)
            self._publish(targets)
            self._record("move", move_elapsed)
        else:
            targets, obs, action = self._policy_targets(move_elapsed, self.args.vx)
            self._publish(targets)
            self._record("move", move_elapsed, obs, action)
        self.control_steps += 1

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        move = [row for row in self.trace if row["stage"] == "move"]
        stop = [row for row in self.trace if row["stage"] == "stop"]
        root_z = np.asarray([row["root_z_m"] for row in move], dtype=np.float64)
        tilt = np.asarray([row["root_tilt_rad"] for row in move], dtype=np.float64)
        x = np.asarray([row["root_x_m"] for row in move], dtype=np.float64)
        y = np.asarray([row["root_y_m"] for row in move], dtype=np.float64)
        summary = {
            "domain": "aimdk_x2_v1_official_mujoco",
            "model": self.args.model,
            "template": self.args.template,
            "command_vx_mps": self.args.vx,
            "control_mode": self.args.control_mode,
            "pd_profile": self.args.pd_profile,
            "heading_gain": self.args.heading_gain,
            "heading_rate_limit_radps": self.args.heading_rate_limit,
            "fixed_wz_radps": self.args.fixed_wz,
            "action_bias_mode": self.args.action_bias_mode,
            "action_bias": self.args.action_bias,
            "yaw_action_gain": self.args.yaw_action_gain,
            "prepare_seconds": self.args.prepare_seconds,
            "stand_seconds": self.args.stand_seconds,
            "move_seconds": self.args.move_seconds,
            "stop_seconds": self.args.stop_seconds,
            "control_steps": len(move),
            "observation_contract": "Stage208 deterministic 93D",
            "action_contract": "15D lower/waist residual + template_scale=0.15",
            "root_z_min_m": float(root_z.min()) if root_z.size else None,
            "root_z_final_m": float(root_z[-1]) if root_z.size else None,
            "root_tilt_max_rad": float(tilt.max()) if tilt.size else None,
            "root_xy_displacement_m": float(math.hypot(x[-1] - x[0], y[-1] - y[0])) if x.size else None,
            "survived_height_gate": bool(root_z.size and root_z.min() >= 0.45),
            "notes": [
                "Official odometry linear velocity is rotated from world into the pelvis/body frame.",
                "This first adapter smoke does not yet claim an IsaacLab matched-domain comparison.",
            ],
        }
        if stop:
            stop_z = np.asarray([row["root_z_m"] for row in stop], dtype=np.float64)
            stop_tilt = np.asarray([row["root_tilt_rad"] for row in stop], dtype=np.float64)
            stop_x = np.asarray([row["root_x_m"] for row in stop], dtype=np.float64)
            stop_y = np.asarray([row["root_y_m"] for row in stop], dtype=np.float64)
            summary.update(
                {
                    "stop_root_z_min_m": float(stop_z.min()),
                    "stop_root_z_final_m": float(stop_z[-1]),
                    "stop_root_tilt_max_rad": float(stop_tilt.max()),
                    "stop_root_xy_drift_m": float(math.hypot(stop_x[-1] - stop_x[0], stop_y[-1] - stop_y[0])),
                    "survived_stop_height_gate": bool(stop_z.min() >= 0.45),
                }
            )
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": summary, "trace": self.trace}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vx", type=float, default=0.30)
    parser.add_argument(
        "--control-mode",
        choices=("full", "template_only", "actor_only", "isaac_target_replay"),
        default="full",
    )
    parser.add_argument("--joint-target-trace")
    parser.add_argument("--prepare-seconds", type=float, default=3.0)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument("--move-seconds", type=float, default=8.0)
    parser.add_argument("--stop-seconds", type=float, default=0.0)
    parser.add_argument("--heading-gain", type=float, default=0.0)
    parser.add_argument("--heading-rate-limit", type=float, default=0.5)
    parser.add_argument("--fixed-wz", type=float)
    parser.add_argument(
        "--action-bias-mode",
        choices=("none", "hip_yaw_common", "waist_yaw", "hip_waist_counter", "hip_yaw_feedback"),
        default="none",
    )
    parser.add_argument("--action-bias", type=float, default=0.0)
    parser.add_argument("--yaw-action-gain", type=float, default=0.0)
    parser.add_argument(
        "--pd-profile",
        choices=(
            "stage208",
            "official_native",
            "official_kp_only",
            "official_kd_only",
            "official_kp_proximal",
            "official_kp_ankle",
            "official_kp_waist",
        ),
        default="stage208",
        help="Lower/waist PD gains; all other control semantics remain fixed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Stage208OfficialAdapter(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
