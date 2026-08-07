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


def default_pose(profile: str) -> dict[str, float]:
    """Return the Stage208 or official v1.0 equilibrium pose.

    The official SDK uses a visibly deeper lower-body crouch than the
    IsaacLab substrate on which Stage208 was trained.  Keeping this as an
    explicit evaluation switch lets us test the physical equilibrium without
    silently changing the frozen actor or its observation contract.
    """
    pose = dict(DEFAULT)
    if profile == "stage208":
        return pose
    if profile != "official_v1":
        raise ValueError(f"unknown default-pose profile: {profile}")
    for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        pose[name] = -0.312
    for name in ("left_knee_joint", "right_knee_joint"):
        pose[name] = 0.669
    for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        pose[name] = -0.363
    # Match the official RL deploy default rather than its separate generic
    # stand configuration.  Uncontrolled joints are still held deterministically.
    for name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        pose[name] = 0.2
    for name in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
        pose[name] = 0.2 if name.startswith("left") else -0.2
    for name in ("left_elbow_joint", "right_elbow_joint"):
        pose[name] = -0.3
    return pose

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
        self.default = default_pose(args.default_pose_profile)
        self.session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        self.stationary_session = (
            ort.InferenceSession(args.stationary_model, providers=["CPUExecutionProvider"])
            if args.stationary_model
            else self.session
        )
        archive = np.load(args.template, allow_pickle=False)
        if tuple(archive["joint_names_15"].tolist()) != LOWER_JOINTS:
            raise RuntimeError("gait-template joint order does not match Stage208")
        if not np.allclose(archive["action_scale_rad"], LOWER_SCALE, atol=1e-6, rtol=0.0):
            raise RuntimeError("gait-template action scale does not match Stage208")
        self.template = archive["q_cycle_zero_mean_rad"].astype(np.float32)
        self.period = float(archive["period_s"])
        self.double_support_fraction = float(archive["double_support_fraction"])
        self.replay_targets: np.ndarray | None = None
        self.replay_joint_names: tuple[str, ...] = LOWER_JOINTS
        self.replay_pd: dict[str, tuple[float, float]] = {}
        if args.control_mode == "isaac_target_replay":
            if not args.joint_target_trace:
                raise ValueError("--joint-target-trace is required for isaac_target_replay")
            replay = np.load(args.joint_target_trace, allow_pickle=False)
            self.replay_joint_names = tuple(replay["action_joint_order"].tolist())
            if self.replay_joint_names not in (LOWER_JOINTS, ISAAC_JOINTS):
                raise RuntimeError("Isaac replay target joint order must be exact lower-15 or full-31 order")
            self.replay_targets = replay["joint_target_rad"].astype(np.float32)
            if self.replay_targets.ndim != 2 or self.replay_targets.shape[1] != len(self.replay_joint_names):
                raise RuntimeError(f"invalid replay target shape {self.replay_targets.shape}")
            if "joint_kp" in replay.files and "joint_kd" in replay.files:
                replay_kp = np.asarray(replay["joint_kp"], dtype=np.float32)
                replay_kd = np.asarray(replay["joint_kd"], dtype=np.float32)
                if replay_kp.shape != (len(self.replay_joint_names),) or replay_kd.shape != replay_kp.shape:
                    raise RuntimeError("replay PD arrays do not match replay joint order")
                self.replay_pd = {
                    name: (float(replay_kp[index]), float(replay_kd[index]))
                    for index, name in enumerate(self.replay_joint_names)
                }

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

        # Keep recurrent observation state separate across skills.  Both actors
        # are feed-forward, but last_action is part of the 93-D observation.
        # Sharing it across a handoff would inject the locomotion actor's final
        # action into a stand actor that was trained from a zero-action reset.
        self.previous_actions = {
            "main": np.zeros(15, dtype=np.float32),
            "stationary": np.zeros(15, dtype=np.float32),
        }
        self.last_move_targets: dict[str, float] | None = None
        self.stop_hold_targets: dict[str, float] | None = None
        self.stop_hold_latch_s: float | None = None
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
        if name in self.replay_pd:
            return self.replay_pd[name]
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

    def _policy_targets(
        self,
        phase_elapsed: float,
        command_vx: float,
        *,
        command_vy: float = 0.0,
        force_moving: bool = False,
        template_multiplier: float = 1.0,
        policy_slot: str = "main",
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
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
        command = np.asarray([command_vx, command_vy, command_wz], dtype=np.float32)
        joint_pos = np.asarray([self.joints[name][0] - self.default[name] for name in ISAAC_JOINTS], dtype=np.float32)
        joint_vel = np.asarray([self.joints[name][1] for name in ISAAC_JOINTS], dtype=np.float32)
        moving = force_moving or abs(command_vx) > 0.1
        phase = self._phase_features(phase_elapsed, moving=moving)
        previous_action = self.previous_actions[policy_slot]
        obs = np.concatenate(
            (base_lin_vel, base_ang_vel, gravity_body(self.imu), command, joint_pos, joint_vel, previous_action, phase)
        ).astype(np.float32)
        if obs.shape != (93,):
            raise RuntimeError(f"invalid Stage208 observation shape {obs.shape}")
        session = self.stationary_session if policy_slot == "stationary" else self.session
        raw_action = session.run(["actions"], {"obs": obs[None]})[0][0].astype(np.float32)
        if (
            policy_slot == "stationary"
            and self.stationary_session is not self.session
            and self.args.stationary_blend < 1.0
        ):
            main_obs = obs.copy()
            main_obs[74:89] = self.previous_actions["main"]
            main_action = self.session.run(["actions"], {"obs": main_obs[None]})[0][0].astype(np.float32)
            alpha = self.args.stationary_blend
            raw_action = (1.0 - alpha) * main_action + alpha * raw_action
        residual = raw_action.copy()
        if self.args.control_mode == "template_only":
            residual.fill(0.0)
        template_bias = (
            0.15 * template_multiplier * self._template_bias(phase_elapsed) / LOWER_SCALE
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
        targets = dict(self.default)
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] = self.default[name] + float(combined[index] * LOWER_SCALE[index])
        # IsaacLab's last_action observation is the unbounded raw policy action;
        # clipping happens only after adding the normalized gait-template bias.
        self.previous_actions[policy_slot] = residual.copy()
        if policy_slot == "stationary":
            # Both actors observe the action that was actually issued, not an
            # unexecuted branch-specific proposal.
            self.previous_actions["main"] = residual.copy()
        return targets, obs, combined

    def _brake_command(self) -> tuple[float, float, float]:
        """Return bounded body-frame velocity feedback for a stop transition.

        The official simulator's foot-contact publisher is an empty binary
        stub, so the deployable stop teacher cannot wait on that topic.  This
        controller instead uses odometry already present in the Stage208
        observation contract.  It asks the frozen velocity policy to oppose
        the measured horizontal velocity while keeping the gait phase alive.
        """
        assert self.odom is not None
        twist = self.odom.twist.twist
        body_velocity = world_vector_to_body(
            self.odom,
            np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float32),
        )
        command_xy = np.clip(
            -self.args.stop_brake_gain * body_velocity[:2],
            -self.args.stop_brake_limit,
            self.args.stop_brake_limit,
        )
        speed = float(np.linalg.norm(body_velocity[:2]))
        return float(command_xy[0]), float(command_xy[1]), speed

    def _replay_target(self, elapsed: float) -> dict[str, float]:
        assert self.replay_targets is not None
        raw_index = int(elapsed / 0.02)
        if self.args.replay_loop:
            index = raw_index % self.replay_targets.shape[0]
        else:
            index = min(raw_index, self.replay_targets.shape[0] - 1)
        targets = dict(self.default)
        for joint_index, name in enumerate(self.replay_joint_names):
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
                "root_vx_b_mps": None if obs is None else float(obs[0]),
                "root_vy_b_mps": None if obs is None else float(obs[1]),
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
                name: self.prepare_start_q[name] + alpha * (self.default[name] - self.prepare_start_q[name])
                for name in ISAAC_JOINTS
            }
            self._publish(targets)
            self._record("prepare", elapsed)
            return

        stand_elapsed = elapsed - self.args.prepare_seconds
        if stand_elapsed < self.args.stand_seconds:
            if self.args.stationary_controller == "default_pose":
                targets = dict(self.default)
                obs = action = None
                self.previous_actions["stationary"].fill(0.0)
            else:
                use_stationary = stand_elapsed >= self.args.stationary_warmup_seconds
                targets, obs, action = self._policy_targets(
                    0.0,
                    0.0,
                    policy_slot="stationary" if use_stationary else "main",
                )
            self._publish(targets)
            self._record("stand", stand_elapsed, obs, action)
            return

        move_elapsed = stand_elapsed - self.args.stand_seconds
        if move_elapsed >= self.args.move_seconds:
            stop_elapsed = move_elapsed - self.args.move_seconds
            if stop_elapsed >= self.args.stop_seconds:
                self._finish()
                return
            if self.args.stop_controller == "event_hold":
                if self.stop_hold_targets is None:
                    targets, obs, action = self._policy_targets(self.args.move_seconds, 0.0)
                    assert self.odom is not None
                    twist = self.odom.twist.twist
                    speed = math.hypot(twist.linear.x, twist.linear.y)
                    tilt = tilt_from_quaternion(self.odom)
                    if (
                        stop_elapsed >= self.args.event_hold_min_seconds
                        and speed <= self.args.event_hold_speed
                        and tilt <= self.args.event_hold_tilt
                    ):
                        self.stop_hold_targets = dict(targets)
                        self.stop_hold_latch_s = stop_elapsed
                else:
                    targets = dict(self.stop_hold_targets)
                    obs = action = None
            elif self.args.stop_controller in ("velocity_brake", "brake_then_policy"):
                command_vx, command_vy, measured_speed = self._brake_command()
                phase_elapsed = self.args.move_seconds + stop_elapsed
                phase = self._phase_features(phase_elapsed, moving=True)
                double_support = bool(phase[2] > 0.5 and phase[3] > 0.5)
                if (
                    self.args.stop_controller == "brake_then_policy"
                    and self.stop_hold_latch_s is None
                    and stop_elapsed >= self.args.event_hold_min_seconds
                    and measured_speed <= self.args.event_hold_speed
                    and double_support
                ):
                    self.stop_hold_latch_s = stop_elapsed
                if self.stop_hold_latch_s is not None:
                    targets, obs, action = self._policy_targets(0.0, 0.0)
                else:
                    template_multiplier = float(
                        np.clip(
                            measured_speed / max(self.args.stop_brake_template_speed, 1.0e-6),
                            self.args.stop_brake_template_floor,
                            1.0,
                        )
                    )
                    targets, obs, action = self._policy_targets(
                        phase_elapsed,
                        command_vx,
                        command_vy=command_vy,
                        force_moving=True,
                        template_multiplier=template_multiplier,
                    )
            elif self.args.stop_controller == "ramp_then_hold":
                ramp_seconds = 0.5 * self.args.stop_seconds
                if stop_elapsed < ramp_seconds:
                    ramp = max(0.0, 1.0 - stop_elapsed / ramp_seconds)
                    targets, obs, action = self._policy_targets(
                        self.args.move_seconds + stop_elapsed,
                        self.args.vx * ramp,
                        force_moving=True,
                        template_multiplier=ramp,
                    )
                    self.stop_hold_targets = dict(targets)
                else:
                    targets = dict(self.stop_hold_targets or self.last_move_targets or self.default)
                    obs = action = None
            elif self.args.stop_controller == "hold_last":
                targets = dict(self.last_move_targets or self.default)
                obs = action = None
            elif self.args.stop_controller == "default_pose":
                targets = dict(self.default)
                obs = action = None
                self.previous_actions["stationary"].fill(0.0)
            elif self.args.stop_controller == "ramp_policy":
                ramp = max(0.0, 1.0 - stop_elapsed / self.args.stop_seconds)
                targets, obs, action = self._policy_targets(
                    self.args.move_seconds + stop_elapsed,
                    self.args.vx * ramp,
                    force_moving=True,
                    template_multiplier=ramp,
                )
            else:
                targets, obs, action = self._policy_targets(
                    self.args.move_seconds,
                    0.0,
                    policy_slot="stationary",
                )
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
        self.last_move_targets = dict(targets)
        self.control_steps += 1

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        stand = [row for row in self.trace if row["stage"] == "stand"]
        move = [row for row in self.trace if row["stage"] == "move"]
        stop = [row for row in self.trace if row["stage"] == "stop"]
        root_z = np.asarray([row["root_z_m"] for row in move], dtype=np.float64)
        tilt = np.asarray([row["root_tilt_rad"] for row in move], dtype=np.float64)
        x = np.asarray([row["root_x_m"] for row in move], dtype=np.float64)
        y = np.asarray([row["root_y_m"] for row in move], dtype=np.float64)
        summary = {
            "domain": "aimdk_x2_v1_official_mujoco",
            "model": self.args.model,
            "stationary_model": self.args.stationary_model or self.args.model,
            "stationary_warmup_seconds": self.args.stationary_warmup_seconds,
            "stationary_blend": self.args.stationary_blend,
            "template": self.args.template,
            "command_vx_mps": self.args.vx,
            "control_mode": self.args.control_mode,
            "replay_loop": self.args.replay_loop,
            "pd_profile": self.args.pd_profile,
            "default_pose_profile": self.args.default_pose_profile,
            "heading_gain": self.args.heading_gain,
            "heading_rate_limit_radps": self.args.heading_rate_limit,
            "fixed_wz_radps": self.args.fixed_wz,
            "action_bias_mode": self.args.action_bias_mode,
            "action_bias": self.args.action_bias,
            "yaw_action_gain": self.args.yaw_action_gain,
            "stationary_controller": self.args.stationary_controller,
            "stop_controller": self.args.stop_controller,
            "stop_hold_latch_s": self.stop_hold_latch_s,
            "stop_brake_gain": self.args.stop_brake_gain,
            "stop_brake_limit_mps": self.args.stop_brake_limit,
            "stop_brake_template_speed_mps": self.args.stop_brake_template_speed,
            "stop_brake_template_floor": self.args.stop_brake_template_floor,
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
            "gate_thresholds": {
                "stand_root_z_min_m": 0.45,
                "stand_tilt_max_rad": 0.25,
                "stand_xy_drift_max_m": 0.10,
                "stand_tail_speed_max_mps": 0.03,
                "move_root_z_min_m": 0.45,
                "move_tilt_max_rad": 0.40,
                "move_forward_min_m": 0.50,
                "move_lateral_max_m": 0.30,
                "move_heading_max_rad": 0.30,
                "stop_root_z_min_m": 0.45,
                "stop_tilt_max_rad": 0.30,
                "stop_xy_drift_max_m": 0.15,
                "stop_tail_speed_max_mps": 0.03,
            },
        }
        if stand:
            stand_z = np.asarray([row["root_z_m"] for row in stand], dtype=np.float64)
            stand_tilt = np.asarray([row["root_tilt_rad"] for row in stand], dtype=np.float64)
            stand_x = np.asarray([row["root_x_m"] for row in stand], dtype=np.float64)
            stand_y = np.asarray([row["root_y_m"] for row in stand], dtype=np.float64)
            stand_speed = np.hypot(
                np.asarray([row["root_vx_w_mps"] for row in stand], dtype=np.float64),
                np.asarray([row["root_vy_w_mps"] for row in stand], dtype=np.float64),
            )
            tail_count = min(50, stand_speed.size)
            summary.update(
                {
                    "stand_root_z_min_m": float(stand_z.min()),
                    "stand_root_z_final_m": float(stand_z[-1]),
                    "stand_root_tilt_max_rad": float(stand_tilt.max()),
                    "stand_root_xy_drift_m": float(
                        math.hypot(stand_x[-1] - stand_x[0], stand_y[-1] - stand_y[0])
                    ),
                    "stand_last_1s_mean_speed_mps": float(stand_speed[-tail_count:].mean()),
                    "survived_stand_height_gate": bool(stand_z.min() >= 0.45),
                    "stand_gate_pass": bool(
                        stand_z.min() >= 0.45
                        and stand_tilt.max() <= 0.25
                        and math.hypot(stand_x[-1] - stand_x[0], stand_y[-1] - stand_y[0]) <= 0.10
                        and stand_speed[-tail_count:].mean() <= 0.03
                    ),
                }
            )
        if move:
            move_yaw = np.unwrap(np.asarray([row["root_yaw_rad"] for row in move], dtype=np.float64))
            start_yaw = float(move_yaw[0])
            dx = float(x[-1] - x[0])
            dy = float(y[-1] - y[0])
            forward = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
            lateral = -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
            heading_max = float(np.max(np.abs(move_yaw - move_yaw[0])))
            body_vx_values = [row["root_vx_b_mps"] for row in move if row["root_vx_b_mps"] is not None]
            body_vx = np.asarray(body_vx_values, dtype=np.float64)
            move_tilt_max = float(tilt.max())
            summary.update(
                {
                    "move_forward_displacement_m": forward,
                    "move_lateral_displacement_m": lateral,
                    "move_heading_max_deviation_rad": heading_max,
                    "move_body_vx_mean_mps": float(body_vx.mean()) if body_vx.size else None,
                    "move_body_vx_rmse_mps": (
                        float(np.sqrt(np.mean((body_vx - self.args.vx) ** 2))) if body_vx.size else None
                    ),
                    "move_gate_pass": bool(
                        root_z.min() >= 0.45
                        and move_tilt_max <= 0.40
                        and forward >= 0.50
                        and abs(lateral) <= 0.30
                        and heading_max <= 0.30
                    ),
                }
            )
        if stop:
            stop_z = np.asarray([row["root_z_m"] for row in stop], dtype=np.float64)
            stop_tilt = np.asarray([row["root_tilt_rad"] for row in stop], dtype=np.float64)
            stop_x = np.asarray([row["root_x_m"] for row in stop], dtype=np.float64)
            stop_y = np.asarray([row["root_y_m"] for row in stop], dtype=np.float64)
            stop_speed = np.hypot(
                np.asarray([row["root_vx_w_mps"] for row in stop], dtype=np.float64),
                np.asarray([row["root_vy_w_mps"] for row in stop], dtype=np.float64),
            )
            tail_count = min(50, stop_speed.size)
            settle_time = None
            for index in range(stop_speed.size):
                window = stop_speed[index : min(index + 50, stop_speed.size)]
                if window.size == 50 and np.all(window <= 0.03):
                    settle_time = index * 0.02
                    break
            stop_drift = float(math.hypot(stop_x[-1] - stop_x[0], stop_y[-1] - stop_y[0]))
            stop_tilt_max = float(stop_tilt.max())
            summary.update(
                {
                    "stop_root_z_min_m": float(stop_z.min()),
                    "stop_root_z_final_m": float(stop_z[-1]),
                    "stop_root_tilt_max_rad": stop_tilt_max,
                    "stop_root_xy_drift_m": stop_drift,
                    "stop_last_1s_mean_speed_mps": float(stop_speed[-tail_count:].mean()),
                    "stop_settle_time_s": settle_time,
                    "survived_stop_height_gate": bool(stop_z.min() >= 0.45),
                    "stop_gate_pass": bool(
                        stop_z.min() >= 0.45
                        and stop_tilt_max <= 0.30
                        and stop_drift <= 0.15
                        and stop_speed[-tail_count:].mean() <= 0.03
                    ),
                }
            )
        required_gates = []
        if stand:
            required_gates.append(bool(summary.get("stand_gate_pass")))
        if move:
            required_gates.append(bool(summary.get("move_gate_pass")))
        if stop:
            required_gates.append(bool(summary.get("stop_gate_pass")))
        summary["full_gate_pass"] = bool(required_gates and all(required_gates))
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": summary, "trace": self.trace}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--stationary-model",
        help="Optional ONNX actor used for zero-command stand/stop phases.",
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vx", type=float, default=0.30)
    parser.add_argument(
        "--control-mode",
        choices=("full", "template_only", "actor_only", "isaac_target_replay"),
        default="full",
    )
    parser.add_argument("--joint-target-trace")
    parser.add_argument(
        "--replay-loop",
        action="store_true",
        help="Loop an Isaac target replay instead of holding its final frame.",
    )
    parser.add_argument("--prepare-seconds", type=float, default=3.0)
    parser.add_argument(
        "--default-pose-profile",
        choices=("stage208", "official_v1"),
        default="stage208",
        help="Equilibrium pose used for joint_pos_rel and action targets.",
    )
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument(
        "--stationary-warmup-seconds",
        type=float,
        default=0.0,
        help="Use the main actor for this many seconds before handing stand to --stationary-model.",
    )
    parser.add_argument(
        "--stationary-blend",
        type=float,
        default=1.0,
        help="Blend fraction of the stationary actor after handoff (0=main actor, 1=stationary actor).",
    )
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
        "--stationary-controller",
        choices=("policy", "default_pose"),
        default="policy",
        help="Controller used while commanded velocity is zero during stand and stop phases.",
    )
    parser.add_argument(
        "--stop-controller",
        choices=(
            "policy",
            "default_pose",
            "ramp_policy",
            "hold_last",
            "ramp_then_hold",
            "event_hold",
            "velocity_brake",
            "brake_then_policy",
        ),
        default="policy",
        help="Controller used after the moving phase; ramp_policy preserves phase while reducing speed and template amplitude.",
    )
    parser.add_argument("--event-hold-min-seconds", type=float, default=0.5)
    parser.add_argument("--event-hold-speed", type=float, default=0.05)
    parser.add_argument("--event-hold-tilt", type=float, default=0.10)
    parser.add_argument("--stop-brake-gain", type=float, default=0.8)
    parser.add_argument("--stop-brake-limit", type=float, default=0.30)
    parser.add_argument("--stop-brake-template-speed", type=float, default=0.30)
    parser.add_argument("--stop-brake-template-floor", type=float, default=0.25)
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
    args = parser.parse_args()
    if not 0.0 <= args.stationary_blend <= 1.0:
        parser.error("--stationary-blend must be in [0, 1]")
    if args.stationary_warmup_seconds < 0.0:
        parser.error("--stationary-warmup-seconds must be non-negative")
    return args


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
