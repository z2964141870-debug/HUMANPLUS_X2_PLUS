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


def _mirror_joint_sign(name: str) -> float:
    return -1.0 if any(axis in name for axis in ("_roll_", "_yaw_")) else 1.0


def _mirror_joint_vector(values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Reflect a named joint vector across the robot sagittal plane."""
    index = {name: i for i, name in enumerate(names)}
    mirrored = np.empty_like(values)
    for output_index, name in enumerate(names):
        if name.startswith("left_"):
            source_name = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            source_name = "left_" + name[len("right_") :]
        else:
            source_name = name
        mirrored[output_index] = _mirror_joint_sign(name) * values[index[source_name]]
    return mirrored


def _sample_upper_track(q_rad: np.ndarray, fps: float, time_s: float, *, loop: bool) -> np.ndarray:
    """Linearly sample a portable frames-by-14 upper-body trajectory."""
    last = q_rad.shape[0] - 1
    frame = time_s * fps
    frame = frame % last if loop else float(np.clip(frame, 0.0, float(last)))
    lower = int(math.floor(frame))
    upper = min(lower + 1, last)
    blend = frame - lower
    return ((1.0 - blend) * q_rad[lower] + blend * q_rad[upper]).astype(np.float32)


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

        self.upper_q_rad: np.ndarray | None = None
        self.upper_fps: float | None = None
        self.upper_source: str | None = None
        self.upper_baseline: np.ndarray | None = None
        self.upper_previous_target = np.asarray(
            [self.default[name] for name in ARM_JOINTS], dtype=np.float32
        )
        self.upper_last_target = self.upper_previous_target.copy()
        self.upper_fallback_steps = 0
        self.upper_fallback_active = False
        self.upper_fallback_first_step: int | None = None
        if args.upper_motion:
            upper_archive = np.load(args.upper_motion, allow_pickle=False)
            upper_names = tuple(upper_archive["joint_names"].tolist())
            if upper_names != ARM_JOINTS:
                raise RuntimeError(
                    f"upper-motion joint order mismatch: {upper_names} != {ARM_JOINTS}"
                )
            self.upper_q_rad = np.asarray(upper_archive["q_rad"], dtype=np.float32)
            self.upper_fps = float(upper_archive["fps"])
            if self.upper_q_rad.ndim != 2 or self.upper_q_rad.shape[1] != len(ARM_JOINTS):
                raise RuntimeError(f"invalid upper-motion shape {self.upper_q_rad.shape}")
            if self.upper_q_rad.shape[0] < 2 or self.upper_fps <= 0.0:
                raise RuntimeError("upper-motion must contain at least two timed frames")
            duration_s = (self.upper_q_rad.shape[0] - 1) / self.upper_fps
            if not 0.0 <= args.upper_start_seconds <= duration_s:
                raise ValueError(
                    f"upper start {args.upper_start_seconds} outside [0,{duration_s}]"
                )
            self.upper_source = (
                str(upper_archive["source"].item())
                if "source" in upper_archive.files
                else str(args.upper_motion)
            )
            self.upper_baseline = _sample_upper_track(
                self.upper_q_rad,
                self.upper_fps,
                args.upper_start_seconds,
                loop=args.upper_loop,
            )

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=args.state_qos_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.joints: dict[str, tuple[float, float]] = {}
        self.joint_sample_times: dict[str, float] = {}
        self.joint_callback_wall_times: dict[str, float] = {}
        self.imu: Imu | None = None
        self.odom: Odometry | None = None
        self.imu_sample_time: float | None = None
        self.odom_sample_time: float | None = None
        self.imu_callback_wall_time: float | None = None
        self.odom_callback_wall_time: float | None = None
        for area in ("leg", "waist", "arm", "head"):
            self.create_subscription(
                JointStateArray,
                f"/aima/hal/joint/{area}/state",
                self._joint_callback,
                state_qos,
            )
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self._imu_callback, state_qos)
        self.create_subscription(Odometry, "/aima/hal/odom/state", self._odom_callback, state_qos)
        self.command_publishers = {
            area: self.create_publisher(
                JointCommandArray, f"/aima/hal/joint/{area}/command", command_qos
            )
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
        self.issued_actions = {
            "main": np.zeros(15, dtype=np.float32),
            "stationary": np.zeros(15, dtype=np.float32),
        }
        self.last_move_targets: dict[str, float] | None = None
        self.stop_hold_targets: dict[str, float] | None = None
        self.stop_hold_latch_s: float | None = None
        self.heading_target_rad: float | None = None
        self.heading_origin_xy: tuple[float, float] | None = None
        self.move_heading_initialized = False
        self.stop_policy_initialized = False
        self.lateral_recovery_state = "off"
        self.lateral_recovery_bias = np.zeros(2, dtype=np.float32)
        self.heading_recovery_active = False
        self.heading_action_recovery_active = False
        self.heading_action_recovery_steps = 0
        self.ready_wall_time: float | None = None
        self.prepare_start_q: dict[str, float] = {}
        self.sequence_step = 0
        self.control_steps = 0
        # Optional fixed-horizon state prediction for deployment stacks whose
        # joint/IMU/odometry sample reaches the 50 Hz actor slightly late.  The
        # raw history is kept in the actor's physical coordinates and updated
        # at most once per control tick, so multi-policy stop transitions do
        # not manufacture a zero-dt acceleration sample.
        self.previous_physical_observation: np.ndarray | None = None
        self.predicted_physical_observation: np.ndarray | None = None
        self.predicted_physical_step = -1
        self.current_control_wall_time: float | None = None
        self.previous_control_wall_time: float | None = None
        self.current_control_wall_dt: float | None = None
        self.trace: list[dict[str, float | list[float] | str]] = []
        self.finished = False
        self.timer = self.create_timer(0.02, self._control)

    def _joint_callback(self, msg: JointStateArray) -> None:
        sample_time = self._message_sample_time(msg)
        callback_time = time.monotonic()
        for joint in msg.joints:
            self.joints[joint.name] = (float(joint.position), float(joint.velocity))
            self.joint_sample_times[joint.name] = sample_time
            self.joint_callback_wall_times[joint.name] = callback_time

    def _imu_callback(self, msg: Imu) -> None:
        self.imu = msg
        self.imu_sample_time = self._message_sample_time(msg)
        self.imu_callback_wall_time = time.monotonic()

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom = msg
        self.odom_sample_time = self._message_sample_time(msg)
        self.odom_callback_wall_time = time.monotonic()

    @staticmethod
    def _message_sample_time(msg: object) -> float:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "meas_stamp", None)
        if stamp is None:
            stamp = getattr(header, "stamp", None)
        if stamp is None:
            return 0.0
        return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)

    def _state_ready(self) -> bool:
        return all(name in self.joints for name in ISAAC_JOINTS) and self.imu is not None and self.odom is not None

    def _pd(self, name: str) -> tuple[float, float]:
        if name in self.replay_pd:
            kp, kd = self.replay_pd[name]
        elif name in LOWER_JOINTS:
            if self.args.pd_profile == "official_native":
                kp, kd = OFFICIAL_NATIVE_LOWER_PD[name]
            elif self.args.pd_profile == "official_kp_only":
                kp, kd = OFFICIAL_NATIVE_LOWER_PD[name][0], 20.0
            elif self.args.pd_profile == "official_kd_only":
                kp, kd = 300.0, OFFICIAL_NATIVE_LOWER_PD[name][1]
            else:
                group_match = (
                    self.args.pd_profile == "official_kp_proximal"
                    and any(token in name for token in ("hip", "knee"))
                ) or (
                    self.args.pd_profile == "official_kp_ankle" and "ankle" in name
                ) or (
                    self.args.pd_profile == "official_kp_waist" and "waist" in name
                )
                kp, kd = (
                    (OFFICIAL_NATIVE_LOWER_PD[name][0], 20.0)
                    if group_match
                    else (300.0, 20.0)
                )
        elif "shoulder" in name or "elbow" in name:
            kp, kd = 40.0, 5.0
        elif "wrist" in name:
            kp, kd = 30.0, 3.0
        else:
            kp, kd = 50.0, 5.0
        if name in LOWER_JOINTS:
            kp *= self.args.pd_kp_multiplier
            kd *= self.args.pd_kd_multiplier
        return kp, kd

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

    def _apply_upper_motion(
        self,
        targets: dict[str, float],
        *,
        elapsed: float | None,
        return_to_default: bool,
    ) -> None:
        """Apply bounded upper intent without changing the lower actor path."""
        if self.upper_q_rad is None:
            return
        default = np.asarray([self.default[name] for name in ARM_JOINTS], dtype=np.float32)
        desired = default.copy()
        if elapsed is not None:
            assert self.upper_fps is not None and self.upper_baseline is not None
            sample = _sample_upper_track(
                self.upper_q_rad,
                self.upper_fps,
                self.args.upper_start_seconds + elapsed * self.args.upper_time_scale,
                loop=self.args.upper_loop,
            )
            desired += self.args.upper_scale * (sample - self.upper_baseline)
            desired = default + np.clip(
                desired - default,
                -self.args.upper_max_excursion_rad,
                self.args.upper_max_excursion_rad,
            )
            assert self.odom is not None
            healthy = (
                tilt_from_quaternion(self.odom) <= self.args.upper_fallback_tilt_rad
                and float(self.odom.pose.pose.position.z) >= self.args.upper_fallback_height_m
            )
            if (
                healthy
                and self.heading_target_rad is not None
                and (self.args.fixed_wz is None or abs(self.args.fixed_wz) <= 1.0e-6)
            ):
                heading_error = math.atan2(
                    math.sin(yaw_from_quaternion(self.odom) - self.heading_target_rad),
                    math.cos(yaw_from_quaternion(self.odom) - self.heading_target_rad),
                )
                healthy = abs(heading_error) <= self.args.upper_fallback_heading_rad
            if self.args.upper_fallback_latch and not healthy:
                self.upper_fallback_active = True
            fallback = self.upper_fallback_active or not healthy
            if fallback:
                desired = default
                self.upper_fallback_steps += 1
                if self.upper_fallback_first_step is None:
                    self.upper_fallback_first_step = self.sequence_step
        elif not return_to_default:
            # Prepare/stand owns its own exact target path.  Synchronize the
            # slew state without changing those targets.
            self.upper_previous_target = np.asarray(
                [targets[name] for name in ARM_JOINTS], dtype=np.float32
            )
            self.upper_last_target = self.upper_previous_target.copy()
            return
        if return_to_default and self.args.upper_stop_mode == "hold_last":
            desired = self.upper_previous_target.copy()
        max_step = self.args.upper_max_velocity_radps * 0.02
        bounded = self.upper_previous_target + np.clip(
            desired - self.upper_previous_target, -max_step, max_step
        )
        self.upper_previous_target = bounded.astype(np.float32, copy=False)
        self.upper_last_target = self.upper_previous_target.copy()
        for index, name in enumerate(ARM_JOINTS):
            targets[name] = float(self.upper_last_target[index])

    def _publish(
        self,
        targets: dict[str, float],
        *,
        upper_elapsed: float | None = None,
        upper_return: bool = False,
    ) -> None:
        self._apply_upper_motion(
            targets, elapsed=upper_elapsed, return_to_default=upper_return
        )
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

    def _predict_physical_observation(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extrapolate a delayed physical sample by a small fixed horizon.

        This mirrors the intervention that was first falsified in the direct
        vendor-MJCF loop.  Commands, gait phase and previous action are not
        predicted: only deploy-observed physical state is advanced.
        """
        physical = np.concatenate(
            (base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel)
        ).astype(np.float32)
        tau = self.args.state_prediction_seconds
        if tau <= 0.0:
            return base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel
        if self.predicted_physical_step == self.sequence_step:
            assert self.predicted_physical_observation is not None
            predicted = self.predicted_physical_observation.copy()
        else:
            acceleration = np.zeros_like(physical)
            if self.previous_physical_observation is not None:
                acceleration = (physical - self.previous_physical_observation) / 0.02
            predicted = physical.copy()
            delayed_joint_velocity = physical[40:71].copy()
            predicted[0:6] += tau * acceleration[0:6]
            predicted[40:71] += tau * acceleration[40:71]
            omega = predicted[3:6].copy()
            gravity = physical[6:9].copy() - tau * np.cross(omega, physical[6:9])
            norm = float(np.linalg.norm(gravity))
            if norm > 1.0e-8:
                predicted[6:9] = gravity / norm
            predicted[9:40] += (
                tau * delayed_joint_velocity + 0.5 * tau * tau * acceleration[40:71]
            )
            self.previous_physical_observation = physical.copy()
            self.predicted_physical_observation = predicted.copy()
            self.predicted_physical_step = self.sequence_step
        return (
            predicted[0:3], predicted[3:6], predicted[6:9],
            predicted[9:40], predicted[40:71],
        )

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
        if force_moving or abs(command_vx) > 0.1:
            phase_elapsed += self.args.phase_offset
        twist = self.odom.twist.twist
        base_lin_vel_w = np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float32)
        base_lin_vel = world_vector_to_body(self.odom, base_lin_vel_w)
        omega = self.imu.angular_velocity
        base_ang_vel = np.asarray([omega.x, omega.y, omega.z], dtype=np.float32)
        current_yaw = yaw_from_quaternion(self.odom)
        if self.heading_target_rad is None:
            self.heading_target_rad = current_yaw
            self.heading_origin_xy = (
                float(self.odom.pose.pose.position.x),
                float(self.odom.pose.pose.position.y),
            )
        path_heading_offset = 0.0
        if self.args.cross_track_heading_gain != 0.0 and self.heading_origin_xy is not None:
            position = self.odom.pose.pose.position
            dx = float(position.x) - self.heading_origin_xy[0]
            dy = float(position.y) - self.heading_origin_xy[1]
            cross_track = -math.sin(self.heading_target_rad) * dx + math.cos(self.heading_target_rad) * dy
            path_heading_offset = math.atan(-self.args.cross_track_heading_gain * cross_track)
            path_heading_offset = float(
                np.clip(
                    path_heading_offset,
                    -self.args.cross_track_heading_limit,
                    self.args.cross_track_heading_limit,
                )
            )
        effective_heading_target = self.heading_target_rad + path_heading_offset
        heading_error = math.atan2(
            math.sin(effective_heading_target - current_yaw),
            math.cos(effective_heading_target - current_yaw),
        )
        heading_control_error = heading_error
        if self.args.heading_recovery_enter_rad is not None:
            if (
                not self.heading_recovery_active
                and abs(heading_error) >= self.args.heading_recovery_enter_rad
            ):
                self.heading_recovery_active = True
            elif (
                self.heading_recovery_active
                and abs(heading_error) <= self.args.heading_recovery_exit_rad
            ):
                self.heading_recovery_active = False
            if not self.heading_recovery_active:
                heading_control_error = 0.0
        command_wz = float(
            np.clip(
                self.args.heading_gain * heading_control_error,
                -self.args.heading_rate_limit,
                self.args.heading_rate_limit,
            )
        )
        if self.args.fixed_wz is not None and abs(command_vx) > 0.1:
            command_wz = self.args.fixed_wz
        command = np.asarray([command_vx, command_vy, command_wz], dtype=np.float32)
        joint_pos = np.asarray([self.joints[name][0] - self.default[name] for name in ISAAC_JOINTS], dtype=np.float32)
        joint_vel = np.asarray([self.joints[name][1] for name in ISAAC_JOINTS], dtype=np.float32)
        projected_gravity = gravity_body(self.imu)
        (
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
        ) = self._predict_physical_observation(
            base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel
        )
        moving = force_moving or abs(command_vx) > 0.1
        phase = self._phase_features(phase_elapsed, moving=moving)
        previous_action = self.previous_actions[policy_slot]
        if self.args.mirror_policy:
            base_lin_vel = base_lin_vel * np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
            base_ang_vel = base_ang_vel * np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
            projected_gravity = projected_gravity * np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
            command = command * np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
            joint_pos = _mirror_joint_vector(joint_pos, ISAAC_JOINTS)
            joint_vel = _mirror_joint_vector(joint_vel, ISAAC_JOINTS)
            previous_action = _mirror_joint_vector(previous_action, LOWER_JOINTS)
        obs = np.concatenate(
            (base_lin_vel, base_ang_vel, projected_gravity, command, joint_pos, joint_vel, previous_action, phase)
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
            main_obs[74:89] = (
                _mirror_joint_vector(self.previous_actions["main"], LOWER_JOINTS)
                if self.args.mirror_policy
                else self.previous_actions["main"]
            )
            main_action = self.session.run(["actions"], {"obs": main_obs[None]})[0][0].astype(np.float32)
            alpha = self.args.stationary_blend
            raw_action = (1.0 - alpha) * main_action + alpha * raw_action
        # RslRlVecEnvWrapper clips the actor output before env.step().  The
        # gait-template action term therefore receives a clipped policy action,
        # and its raw_actions buffer (used by the next last_action observation)
        # is clipped as well.  Feeding the unbounded ONNX mean back here creates
        # a deploy-only positive feedback loop: values above one are re-observed
        # even though they never existed in the IsaacLab training contract.
        residual = np.clip(raw_action, -1.0, 1.0)
        if self.args.mirror_policy:
            residual = _mirror_joint_vector(residual, LOWER_JOINTS)
        if self.args.control_mode == "template_only":
            residual.fill(0.0)
        template_cycle = self._template_bias(phase_elapsed)
        if self.args.mirror_policy:
            template_cycle = _mirror_joint_vector(template_cycle, LOWER_JOINTS)
        template_bias = (
            0.15 * template_multiplier * template_cycle / LOWER_SCALE
            if moving
            else np.zeros(15, dtype=np.float32)
        )
        if self.args.control_mode == "actor_only":
            template_bias.fill(0.0)
        combined_preclip = residual + template_bias
        if moving and self.args.ankle_roll_common_bias != 0.0:
            combined_preclip[[5, 11]] += self.args.ankle_roll_common_bias * template_multiplier
        if moving and self.args.heading_action_recovery_enter_rad is not None:
            if (
                not self.heading_action_recovery_active
                and abs(heading_error) >= self.args.heading_action_recovery_enter_rad
            ):
                self.heading_action_recovery_active = True
            elif (
                self.heading_action_recovery_active
                and abs(heading_error) <= self.args.heading_action_recovery_exit_rad
            ):
                self.heading_action_recovery_active = False
            if self.heading_action_recovery_active:
                self.heading_action_recovery_steps += 1
                correction = float(
                    np.clip(
                        self.args.heading_action_recovery_gain * heading_error,
                        -self.args.heading_action_recovery_limit,
                        self.args.heading_action_recovery_limit,
                    )
                )
                if correction >= 0.0:
                    # Positive yaw authority is the verified common hip-yaw mode.
                    combined_preclip[[2, 8]] += correction
                else:
                    # Negative yaw authority uses the asymmetric pair validated
                    # by the left-turn contract; common negative yaw was unstable.
                    magnitude = -correction
                    combined_preclip[2] += magnitude
                    combined_preclip[8] -= magnitude
        if moving:
            combined_preclip[2] += self.args.left_hip_yaw_bias * template_multiplier
            combined_preclip[8] += self.args.right_hip_yaw_bias * template_multiplier
        if moving and self.args.action_bias_mode == "hip_yaw_feedback":
            feedback_bias = float(
                np.clip(
                    self.args.yaw_action_gain * heading_error,
                    -self.args.action_bias,
                    self.args.action_bias,
                )
            )
            combined_preclip[[2, 8]] += feedback_bias
        elif moving and self.args.action_bias_mode == "turn_progress_feedback":
            if self.args.fixed_wz is None:
                raise RuntimeError("turn_progress_feedback requires --fixed-wz")
            actual_yaw_progress = math.atan2(
                math.sin(current_yaw - self.heading_target_rad),
                math.cos(current_yaw - self.heading_target_rad),
            )
            desired_yaw_progress = self.args.fixed_wz * min(
                max(phase_elapsed, 0.0), self.args.move_seconds
            )
            progress_error = math.atan2(
                math.sin(desired_yaw_progress - actual_yaw_progress),
                math.cos(desired_yaw_progress - actual_yaw_progress),
            )
            correction = float(
                np.clip(
                    self.args.yaw_action_gain * progress_error,
                    -self.args.action_bias,
                    self.args.action_bias,
                )
            ) * template_multiplier
            if self.args.fixed_wz >= 0.0:
                # Positive yaw was empirically controllable through a bounded
                # common hip-yaw mode in the official X2 model.
                combined_preclip[[2, 8]] += correction
            else:
                # The mirrored left-turn policy uses the already-validated
                # common negative hip-yaw contract.  Applying the unmirrored
                # asymmetric negative-yaw mode here changes the sign again and
                # can make a requested left turn move right.
                if self.args.mirror_policy and correction <= 0.0:
                    combined_preclip[[2, 8]] += correction
                elif correction <= 0.0:
                    # For an unmirrored negative-yaw request, retain the
                    # asymmetric pair validated by the straight recovery path.
                    magnitude = -correction
                    combined_preclip[2] += magnitude
                    combined_preclip[8] -= magnitude
                else:
                    # Once the left turn overshoots, the verified positive
                    # common mode supplies a bounded counter-yaw correction.
                    combined_preclip[[2, 8]] += correction
        elif moving and self.args.action_bias_mode == "ankle_roll_lateral_feedback":
            assert self.heading_origin_xy is not None
            position = self.odom.pose.pose.position
            dx = float(position.x) - self.heading_origin_xy[0]
            dy = float(position.y) - self.heading_origin_xy[1]
            cross_track = -math.sin(self.heading_target_rad) * dx + math.cos(self.heading_target_rad) * dy
            feedback_bias = float(
                np.clip(
                    -self.args.lateral_position_gain * cross_track
                    - self.args.lateral_velocity_gain * float(base_lin_vel[1]),
                    -self.args.action_bias,
                    self.args.action_bias,
                )
            )
            combined_preclip[[5, 11]] += feedback_bias
        elif moving and self.args.action_bias_mode == "lateral_recovery_supervisor":
            assert self.heading_origin_xy is not None
            position = self.odom.pose.pose.position
            dx = float(position.x) - self.heading_origin_xy[0]
            dy = float(position.y) - self.heading_origin_xy[1]
            cross_track = -math.sin(self.heading_target_rad) * dx + math.cos(self.heading_target_rad) * dy
            if self.lateral_recovery_state == "off":
                if cross_track <= -self.args.recovery_enter_m:
                    self.lateral_recovery_state = "right"
                elif cross_track >= self.args.recovery_enter_m:
                    self.lateral_recovery_state = "left"
            elif self.lateral_recovery_state == "right" and cross_track >= -self.args.recovery_exit_m:
                self.lateral_recovery_state = "off"
            elif self.lateral_recovery_state == "left" and cross_track <= self.args.recovery_exit_m:
                self.lateral_recovery_state = "off"
            if self.lateral_recovery_state == "right":
                target_bias = np.asarray([self.args.action_bias, self.args.action_bias], dtype=np.float32)
            elif self.lateral_recovery_state == "left":
                target_bias = np.asarray([self.args.action_bias, -self.args.action_bias], dtype=np.float32)
            else:
                target_bias = np.zeros(2, dtype=np.float32)
            max_step = self.args.recovery_slew_rate_per_s * 0.02
            self.lateral_recovery_bias += np.clip(
                target_bias - self.lateral_recovery_bias,
                -max_step,
                max_step,
            )
            combined_preclip[[2, 8]] += self.lateral_recovery_bias
        elif moving and self.args.action_bias != 0.0:
            bias_ramp = (
                min(1.0, max(0.0, phase_elapsed) / self.args.action_bias_ramp_seconds)
                if self.args.action_bias_ramp_seconds > 0.0
                else 1.0
            )
            action_bias = self.args.action_bias * bias_ramp * template_multiplier
            if self.args.action_bias_mode == "hip_yaw_common":
                combined_preclip[[2, 8]] += action_bias
            elif self.args.action_bias_mode == "left_hip_yaw":
                combined_preclip[2] += action_bias
            elif self.args.action_bias_mode == "right_hip_yaw":
                combined_preclip[8] += action_bias
            elif self.args.action_bias_mode == "hip_yaw_left_stance":
                if phase[2] > 0.5 and phase[3] < 0.5:
                    combined_preclip[[2, 8]] += action_bias
            elif self.args.action_bias_mode == "hip_yaw_right_stance":
                if phase[3] > 0.5 and phase[2] < 0.5:
                    combined_preclip[[2, 8]] += action_bias
            elif self.args.action_bias_mode == "hip_yaw_pair_left_stance":
                if phase[2] > 0.5 and phase[3] < 0.5:
                    combined_preclip[2] += action_bias
                    combined_preclip[8] -= action_bias
            elif self.args.action_bias_mode == "hip_yaw_pair_right_stance":
                if phase[3] > 0.5 and phase[2] < 0.5:
                    combined_preclip[2] += action_bias
                    combined_preclip[8] -= action_bias
            elif self.args.action_bias_mode == "waist_yaw":
                combined_preclip[12] += action_bias
            elif self.args.action_bias_mode == "hip_waist_counter":
                combined_preclip[[2, 8]] += action_bias
                combined_preclip[12] -= action_bias
            elif self.args.action_bias_mode == "hip_roll_common":
                combined_preclip[[1, 7]] += action_bias
            elif self.args.action_bias_mode == "ankle_roll_common":
                combined_preclip[[5, 11]] += action_bias
            elif self.args.action_bias_mode == "ankle_roll_lstance_boost":
                phase_boost = (
                    self.args.phase_action_boost
                    if phase[2] > 0.5 and phase[3] < 0.5
                    else 0.0
                )
                combined_preclip[[5, 11]] += action_bias + phase_boost
            elif self.args.action_bias_mode == "left_ankle_roll":
                combined_preclip[5] += action_bias
            elif self.args.action_bias_mode == "right_ankle_roll":
                combined_preclip[11] += action_bias
            elif self.args.action_bias_mode == "hip_ankle_roll_counter":
                combined_preclip[[1, 7]] += action_bias
                combined_preclip[[5, 11]] -= action_bias
            elif self.args.action_bias_mode == "waist_roll":
                combined_preclip[14] += action_bias
        combined = np.clip(combined_preclip, -1.0, 1.0)
        if self.args.action_ema_alpha < 1.0:
            alpha = self.args.action_ema_alpha
            combined = alpha * combined + (1.0 - alpha) * self.issued_actions[policy_slot]
        combined[[13, 14]] *= self.args.waist_tilt_action_multiplier
        self.issued_actions[policy_slot] = combined.copy()
        targets = dict(self.default)
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] = self.default[name] + float(combined[index] * LOWER_SCALE[index])
        # Match RslRlVecEnvWrapper + GaitTemplateLowerBodyJointPositionAction:
        # next-step last_action is the wrapper-clipped policy output, while the
        # template is added afterward and clipped independently for execution.
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
        sample_times = [self.joint_sample_times[name] for name in ISAAC_JOINTS]
        sample_times.extend([self.imu_sample_time or 0.0, self.odom_sample_time or 0.0])
        callback_times = [self.joint_callback_wall_times[name] for name in ISAAC_JOINTS]
        callback_times.extend([
            self.imu_callback_wall_time or 0.0,
            self.odom_callback_wall_time or 0.0,
        ])
        valid_sample_times = [value for value in sample_times if value > 0.0]
        valid_callback_times = [value for value in callback_times if value > 0.0]
        source_meas_skew = (
            max(valid_sample_times) - min(valid_sample_times) if valid_sample_times else None
        )
        source_callback_age = (
            (self.current_control_wall_time or time.monotonic()) - min(valid_callback_times)
            if valid_callback_times
            else None
        )
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
                "control_wall_dt_s": self.current_control_wall_dt,
                "source_meas_skew_s": source_meas_skew,
                "source_callback_age_max_s": source_callback_age,
                "obs": [] if obs is None else obs.tolist(),
                "action": [] if action is None else action.tolist(),
                "upper_target_rad": self.upper_last_target.tolist(),
                "upper_actual_rad": [float(self.joints[name][0]) for name in ARM_JOINTS],
            }
        )

    def _control(self) -> None:
        if self.finished or not self._state_ready():
            return
        now = time.monotonic()
        self.current_control_wall_time = now
        self.current_control_wall_dt = (
            None if self.previous_control_wall_time is None else now - self.previous_control_wall_time
        )
        self.previous_control_wall_time = now
        if self.ready_wall_time is None:
            self.ready_wall_time = now
            self.prepare_start_q = {name: self.joints[name][0] for name in ISAAC_JOINTS}
            self.get_logger().info(
                f"state ready; begin {self.args.prepare_seconds:.2f} s matched-pose interpolation"
            )
        if self.args.clock_mode == "step":
            # Gate runs must not depend on host scheduling jitter.  Advance the
            # controller contract at the policy's exact 50 Hz cadence even if
            # a ROS wall timer callback arrives a little early or late.
            elapsed = self.sequence_step * 0.02
        else:
            elapsed = now - self.ready_wall_time
        self.sequence_step += 1

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
        if not self.move_heading_initialized:
            assert self.odom is not None
            # The first locomotion target must be filtered from the action
            # actually being executed by the stand actor, not from zeros.
            self.issued_actions["main"] = self.issued_actions["stationary"].copy()
            self.heading_target_rad = yaw_from_quaternion(self.odom)
            self.heading_origin_xy = (
                float(self.odom.pose.pose.position.x),
                float(self.odom.pose.pose.position.y),
            )
            self.move_heading_initialized = True
        if move_elapsed >= self.args.move_seconds:
            stop_elapsed = move_elapsed - self.args.move_seconds
            if stop_elapsed >= self.args.stop_seconds:
                self._finish()
                return
            if not self.stop_policy_initialized:
                # The stationary actor observes previous_action.  Carry the
                # actual last moving action across the controller handoff;
                # otherwise it receives the stale action from the stand phase.
                self.previous_actions["stationary"] = self.previous_actions["main"].copy()
                self.issued_actions["stationary"] = self.issued_actions["main"].copy()
                self.stop_policy_initialized = True
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
            elif self.args.stop_controller == "blend_to_policy":
                transition = max(self.args.stop_transition_seconds, 1.0e-6)
                if stop_elapsed < transition:
                    ramp = max(0.0, 1.0 - stop_elapsed / transition)
                    moving_targets, _, _ = self._policy_targets(
                        self.args.move_seconds + stop_elapsed,
                        self.args.vx * ramp,
                        force_moving=True,
                        template_multiplier=ramp,
                    )
                    stationary_targets, obs, action = self._policy_targets(
                        0.0,
                        0.0,
                        policy_slot="stationary",
                    )
                    targets = {
                        name: ramp * moving_targets[name] + (1.0 - ramp) * stationary_targets[name]
                        for name in ISAAC_JOINTS
                    }
                else:
                    targets, obs, action = self._policy_targets(
                        0.0,
                        0.0,
                        policy_slot="stationary",
                    )
            else:
                targets, obs, action = self._policy_targets(
                    self.args.move_seconds,
                    0.0,
                    policy_slot="stationary",
                )
            self._publish(targets, upper_return=True)
            self._record("stop", stop_elapsed, obs, action)
            return
        if self.args.control_mode == "isaac_target_replay":
            targets = self._replay_target(move_elapsed)
            self._publish(targets, upper_elapsed=move_elapsed)
            self._record("move", move_elapsed)
        else:
            policy_vx = math.copysign(
                max(abs(self.args.vx), self.args.policy_vx_floor), self.args.vx
            )
            targets, obs, action = self._policy_targets(
                move_elapsed,
                policy_vx,
                template_multiplier=self.args.move_template_multiplier,
            )
            self._publish(targets, upper_elapsed=move_elapsed)
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
            "state_prediction_seconds": self.args.state_prediction_seconds,
            "state_qos_depth": self.args.state_qos_depth,
            "template": self.args.template,
            "command_vx_mps": self.args.vx,
            "policy_vx_floor_mps": self.args.policy_vx_floor,
            "phase_offset_s": self.args.phase_offset,
            "control_mode": self.args.control_mode,
            "replay_loop": self.args.replay_loop,
            "pd_profile": self.args.pd_profile,
            "pd_kp_multiplier": self.args.pd_kp_multiplier,
            "pd_kd_multiplier": self.args.pd_kd_multiplier,
            "default_pose_profile": self.args.default_pose_profile,
            "heading_gain": self.args.heading_gain,
            "heading_recovery_enter_rad": self.args.heading_recovery_enter_rad,
            "heading_recovery_exit_rad": self.args.heading_recovery_exit_rad,
            "heading_action_recovery_enter_rad": self.args.heading_action_recovery_enter_rad,
            "heading_action_recovery_exit_rad": self.args.heading_action_recovery_exit_rad,
            "heading_action_recovery_gain": self.args.heading_action_recovery_gain,
            "heading_action_recovery_limit": self.args.heading_action_recovery_limit,
            "heading_action_recovery_steps": self.heading_action_recovery_steps,
            "cross_track_heading_gain": self.args.cross_track_heading_gain,
            "cross_track_heading_limit_rad": self.args.cross_track_heading_limit,
            "heading_rate_limit_radps": self.args.heading_rate_limit,
            "fixed_wz_radps": self.args.fixed_wz,
            "action_bias_mode": self.args.action_bias_mode,
            "action_bias": self.args.action_bias,
            "action_bias_ramp_seconds": self.args.action_bias_ramp_seconds,
            "ankle_roll_common_bias": self.args.ankle_roll_common_bias,
            "left_hip_yaw_bias": self.args.left_hip_yaw_bias,
            "right_hip_yaw_bias": self.args.right_hip_yaw_bias,
            "yaw_action_gain": self.args.yaw_action_gain,
            "lateral_position_gain": self.args.lateral_position_gain,
            "lateral_velocity_gain": self.args.lateral_velocity_gain,
            "recovery_enter_m": self.args.recovery_enter_m,
            "recovery_exit_m": self.args.recovery_exit_m,
            "recovery_slew_rate_per_s": self.args.recovery_slew_rate_per_s,
            "phase_action_boost": self.args.phase_action_boost,
            "action_ema_alpha": self.args.action_ema_alpha,
            "waist_tilt_action_multiplier": self.args.waist_tilt_action_multiplier,
            "upper_motion": self.args.upper_motion,
            "upper_motion_source": self.upper_source,
            "upper_scale": self.args.upper_scale,
            "upper_time_scale": self.args.upper_time_scale,
            "upper_max_excursion_rad": self.args.upper_max_excursion_rad,
            "upper_max_velocity_radps": self.args.upper_max_velocity_radps,
            "upper_fallback_tilt_rad": self.args.upper_fallback_tilt_rad,
            "upper_fallback_height_m": self.args.upper_fallback_height_m,
            "upper_fallback_heading_rad": self.args.upper_fallback_heading_rad,
            "upper_fallback_latch": self.args.upper_fallback_latch,
            "upper_fallback_steps": self.upper_fallback_steps,
            "upper_fallback_first_step": self.upper_fallback_first_step,
            "upper_stop_mode": self.args.upper_stop_mode,
            "stationary_controller": self.args.stationary_controller,
            "stop_controller": self.args.stop_controller,
            "stop_transition_seconds": self.args.stop_transition_seconds,
            "stop_hold_latch_s": self.stop_hold_latch_s,
            "stop_brake_gain": self.args.stop_brake_gain,
            "stop_brake_limit_mps": self.args.stop_brake_limit,
            "stop_brake_template_speed_mps": self.args.stop_brake_template_speed,
            "stop_brake_template_floor": self.args.stop_brake_template_floor,
            "prepare_seconds": self.args.prepare_seconds,
            "stand_seconds": self.args.stand_seconds,
            "move_seconds": self.args.move_seconds,
            "move_template_multiplier": self.args.move_template_multiplier,
            "stop_seconds": self.args.stop_seconds,
            "control_steps": len(move),
            "clock_mode": self.args.clock_mode,
            "mirror_policy": self.args.mirror_policy,
            "observation_contract": "Stage208 deterministic 93D",
            "action_contract": (
                "RSL actor clip [-1,1] -> 15D lower/waist residual + "
                "template_scale=0.15 -> execution clip [-1,1]"
            ),
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
                "startup_window_s": 1.0,
                "startup_root_z_min_m": 0.60,
                "startup_tilt_max_rad": 0.30,
                "startup_forward_min_m": 0.10,
                "startup_backward_excursion_max_m": 0.03,
                "move_forward_min_m": 0.50,
                "move_lateral_max_m": 0.30,
                "move_heading_max_rad": 0.30,
                "turn_yaw_progress_ratio_min": 0.50,
                "turn_yaw_progress_ratio_max": 1.50,
                "turn_body_vx_mean_min_mps": 0.15,
                "stop_root_z_min_m": 0.45,
                "stop_tilt_max_rad": 0.30,
                "stop_xy_drift_max_m": 0.15,
                "stop_tail_speed_max_mps": 0.03,
            },
        }
        timing_fields = (
            "control_wall_dt_s",
            "source_meas_skew_s",
            "source_callback_age_max_s",
        )
        for field in timing_fields:
            values = np.asarray(
                [float(row[field]) for row in self.trace if row.get(field) is not None],
                dtype=np.float64,
            )
            if values.size:
                summary[f"{field}_p50"] = float(np.quantile(values, 0.50))
                summary[f"{field}_p95"] = float(np.quantile(values, 0.95))
                summary[f"{field}_max"] = float(values.max())
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
            upper_target = np.asarray([row["upper_target_rad"] for row in move], dtype=np.float64)
            upper_actual = np.asarray([row["upper_actual_rad"] for row in move], dtype=np.float64)
            upper_default = np.asarray([self.default[name] for name in ARM_JOINTS], dtype=np.float64)
            upper_error = upper_actual - upper_target
            upper_target_speed = (
                np.diff(upper_target, axis=0) / 0.02
                if upper_target.shape[0] > 1
                else np.zeros((0, len(ARM_JOINTS)), dtype=np.float64)
            )
            summary.update(
                {
                    "upper_target_excursion_abs_max_rad": float(
                        np.max(np.abs(upper_target - upper_default))
                    ),
                    "upper_target_speed_abs_max_radps": (
                        float(np.max(np.abs(upper_target_speed)))
                        if upper_target_speed.size
                        else 0.0
                    ),
                    "upper_tracking_rmse_rad": float(np.sqrt(np.mean(upper_error**2))),
                    "upper_tracking_abs_p95_rad": float(np.quantile(np.abs(upper_error), 0.95)),
                }
            )
            move_yaw = np.unwrap(np.asarray([row["root_yaw_rad"] for row in move], dtype=np.float64))
            start_yaw = float(move_yaw[0])
            dx = float(x[-1] - x[0])
            dy = float(y[-1] - y[0])
            forward = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
            lateral = -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
            startup = [row for row in move if float(row["elapsed_s"]) <= 1.0001]
            startup_forward = np.asarray(
                [
                    math.cos(start_yaw) * (float(row["root_x_m"]) - float(x[0]))
                    + math.sin(start_yaw) * (float(row["root_y_m"]) - float(y[0]))
                    for row in startup
                ],
                dtype=np.float64,
            )
            startup_z = np.asarray(
                [float(row["root_z_m"]) for row in startup], dtype=np.float64
            )
            startup_tilt = np.asarray(
                [float(row["root_tilt_rad"]) for row in startup], dtype=np.float64
            )
            startup_backward_excursion = (
                float(max(0.0, -startup_forward.min())) if startup_forward.size else None
            )
            startup_gate_pass = bool(
                startup_forward.size
                and startup_z.min() >= 0.60
                and startup_tilt.max() <= 0.30
                and startup_forward[-1] >= 0.10
                and startup_backward_excursion is not None
                and startup_backward_excursion <= 0.03
            )
            heading_max = float(np.max(np.abs(move_yaw - move_yaw[0])))
            body_vx_values = [row["root_vx_b_mps"] for row in move if row["root_vx_b_mps"] is not None]
            body_vx = np.asarray(body_vx_values, dtype=np.float64)
            yaw_rate_values = [row["root_yaw_rate_radps"] for row in move]
            body_yaw_rate = np.asarray(yaw_rate_values, dtype=np.float64)
            move_tilt_max = float(tilt.max())
            yaw_progress = float(move_yaw[-1] - move_yaw[0])
            expected_yaw_progress = (
                float(self.args.fixed_wz * self.args.move_seconds)
                if self.args.fixed_wz is not None
                else 0.0
            )
            turn_progress_ratio = (
                yaw_progress / expected_yaw_progress
                if abs(expected_yaw_progress) > 1.0e-6
                else None
            )
            turn_mode = self.args.fixed_wz is not None and abs(self.args.fixed_wz) > 1.0e-6
            straight_gate_pass = bool(
                root_z.min() >= 0.45
                and move_tilt_max <= 0.40
                and forward >= 0.50
                and abs(lateral) <= 0.30
                and heading_max <= 0.30
            )
            turn_gate_pass = bool(
                root_z.min() >= 0.45
                and move_tilt_max <= 0.40
                and body_vx.size
                and body_vx.mean() >= 0.15
                and turn_progress_ratio is not None
                and turn_progress_ratio >= 0.50
                and turn_progress_ratio <= 1.50
            )
            summary.update(
                {
                    "move_forward_displacement_m": forward,
                    "move_lateral_displacement_m": lateral,
                    "startup_window_s": 1.0,
                    "startup_root_z_min_m": float(startup_z.min()) if startup_z.size else None,
                    "startup_tilt_max_rad": (
                        float(startup_tilt.max()) if startup_tilt.size else None
                    ),
                    "startup_forward_displacement_m": (
                        float(startup_forward[-1]) if startup_forward.size else None
                    ),
                    "startup_backward_excursion_m": startup_backward_excursion,
                    "startup_gate_pass": startup_gate_pass,
                    "move_heading_max_deviation_rad": heading_max,
                    "move_yaw_progress_rad": yaw_progress,
                    "expected_yaw_progress_rad": expected_yaw_progress,
                    "turn_yaw_progress_ratio": turn_progress_ratio,
                    "move_yaw_rate_mean_radps": (
                        float(body_yaw_rate.mean()) if body_yaw_rate.size else None
                    ),
                    "move_body_vx_mean_mps": float(body_vx.mean()) if body_vx.size else None,
                    "move_body_vx_rmse_mps": (
                        float(np.sqrt(np.mean((body_vx - self.args.vx) ** 2))) if body_vx.size else None
                    ),
                    "move_gate_kind": "turn" if turn_mode else "straight",
                    "move_gate_pass": turn_gate_pass if turn_mode else straight_gate_pass,
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
            required_gates.append(bool(summary.get("startup_gate_pass")))
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
        "--policy-vx-floor",
        type=float,
        default=0.0,
        help="Map a nonzero external speed request to at least this actor-command magnitude.",
    )
    parser.add_argument(
        "--phase-offset",
        type=float,
        default=0.0,
        help="Gait-clock offset in seconds; used to align locomotion onset with the standing state.",
    )
    parser.add_argument(
        "--clock-mode",
        choices=("wall", "step"),
        default="wall",
        help="Use exact 50 Hz step time for reproducible gates or legacy wall time.",
    )
    parser.add_argument(
        "--mirror-policy",
        action="store_true",
        help="Mirror observations, gait phase, template, and actor action across the sagittal plane.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("full", "template_only", "actor_only", "isaac_target_replay"),
        default="full",
    )
    parser.add_argument("--joint-target-trace")
    parser.add_argument(
        "--upper-motion",
        help="Portable NPZ containing a 14-joint X2 upper-body reference.",
    )
    parser.add_argument("--upper-scale", type=float, default=0.25)
    parser.add_argument("--upper-start-seconds", type=float, default=0.0)
    parser.add_argument("--upper-time-scale", type=float, default=0.5)
    parser.add_argument("--upper-max-excursion-rad", type=float, default=0.12)
    parser.add_argument("--upper-max-velocity-radps", type=float, default=0.20)
    parser.add_argument("--upper-fallback-tilt-rad", type=float, default=0.35)
    parser.add_argument("--upper-fallback-height-m", type=float, default=0.58)
    parser.add_argument("--upper-fallback-heading-rad", type=float, default=float("inf"))
    parser.add_argument("--upper-fallback-latch", action="store_true")
    parser.add_argument("--upper-loop", action="store_true")
    parser.add_argument(
        "--upper-stop-mode",
        choices=("return", "hold_last"),
        default="return",
        help="Return arms to default at locomotion stop, or preserve the last teleoperation pose.",
    )
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
    parser.add_argument(
        "--move-template-multiplier",
        type=float,
        default=1.0,
        help="Scale the gait template and coupled bounded gait biases during steady movement.",
    )
    parser.add_argument("--stop-seconds", type=float, default=0.0)
    parser.add_argument(
        "--state-qos-depth",
        type=int,
        default=10,
        help="KEEP_LAST depth for high-rate state topics; 1 consumes only the newest sample.",
    )
    parser.add_argument(
        "--state-prediction-seconds",
        type=float,
        default=0.0,
        help=(
            "Fixed physical-state extrapolation horizon for delayed deploy observations; "
            "0 preserves the baseline observation contract."
        ),
    )
    parser.add_argument("--heading-gain", type=float, default=0.0)
    parser.add_argument(
        "--heading-recovery-enter-rad",
        type=float,
        help="Enable heading feedback only after absolute yaw error reaches this threshold.",
    )
    parser.add_argument("--heading-recovery-exit-rad", type=float, default=0.08)
    parser.add_argument("--heading-action-recovery-enter-rad", type=float)
    parser.add_argument("--heading-action-recovery-exit-rad", type=float, default=0.10)
    parser.add_argument("--heading-action-recovery-gain", type=float, default=1.0)
    parser.add_argument("--heading-action-recovery-limit", type=float, default=0.25)
    parser.add_argument("--cross-track-heading-gain", type=float, default=0.0)
    parser.add_argument("--cross-track-heading-limit", type=float, default=0.30)
    parser.add_argument("--heading-rate-limit", type=float, default=0.5)
    parser.add_argument("--fixed-wz", type=float)
    parser.add_argument(
        "--action-bias-mode",
        choices=(
            "none",
            "hip_yaw_common",
            "left_hip_yaw",
            "right_hip_yaw",
            "hip_yaw_left_stance",
            "hip_yaw_right_stance",
            "hip_yaw_pair_left_stance",
            "hip_yaw_pair_right_stance",
            "waist_yaw",
            "hip_waist_counter",
            "hip_yaw_feedback",
            "turn_progress_feedback",
            "hip_roll_common",
            "ankle_roll_common",
            "ankle_roll_lstance_boost",
            "left_ankle_roll",
            "right_ankle_roll",
            "hip_ankle_roll_counter",
            "waist_roll",
            "ankle_roll_lateral_feedback",
            "lateral_recovery_supervisor",
        ),
        default="none",
    )
    parser.add_argument("--action-bias", type=float, default=0.0)
    parser.add_argument("--action-bias-ramp-seconds", type=float, default=0.0)
    parser.add_argument(
        "--ankle-roll-common-bias",
        type=float,
        default=0.0,
        help="Independent bounded common ankle-roll bias, allowing a turn residual to be tested separately.",
    )
    parser.add_argument("--left-hip-yaw-bias", type=float, default=0.0)
    parser.add_argument("--right-hip-yaw-bias", type=float, default=0.0)
    parser.add_argument("--yaw-action-gain", type=float, default=0.0)
    parser.add_argument("--lateral-position-gain", type=float, default=0.8)
    parser.add_argument("--lateral-velocity-gain", type=float, default=0.2)
    parser.add_argument("--recovery-enter-m", type=float, default=0.12)
    parser.add_argument("--recovery-exit-m", type=float, default=0.04)
    parser.add_argument("--recovery-slew-rate-per-s", type=float, default=1.0)
    parser.add_argument("--phase-action-boost", type=float, default=0.0)
    parser.add_argument(
        "--action-ema-alpha",
        type=float,
        default=1.0,
        help="EMA coefficient for the issued normalized 15-D action; 1 disables filtering.",
    )
    parser.add_argument(
        "--waist-tilt-action-multiplier",
        type=float,
        default=1.0,
        help="Scale only waist pitch/roll normalized targets after clipping; waist yaw and legs are unchanged.",
    )
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
            "blend_to_policy",
            "hold_last",
            "ramp_then_hold",
            "event_hold",
            "velocity_brake",
            "brake_then_policy",
        ),
        default="policy",
        help="Controller used after the moving phase; ramp_policy preserves phase while reducing speed and template amplitude.",
    )
    parser.add_argument(
        "--stop-transition-seconds",
        type=float,
        default=1.0,
        help="Duration of the moving-to-stationary target blend used by blend_to_policy.",
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
    parser.add_argument(
        "--pd-kp-multiplier",
        type=float,
        default=1.0,
        help="Multiply every emitted lower/waist Kp after selecting the named PD profile.",
    )
    parser.add_argument(
        "--pd-kd-multiplier",
        type=float,
        default=1.0,
        help="Multiply every emitted lower/waist Kd after selecting the named PD profile.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.stationary_blend <= 1.0:
        parser.error("--stationary-blend must be in [0, 1]")
    if args.stationary_warmup_seconds < 0.0:
        parser.error("--stationary-warmup-seconds must be non-negative")
    if args.action_bias_ramp_seconds < 0.0:
        parser.error("--action-bias-ramp-seconds must be non-negative")
    if not 0.0 <= args.state_prediction_seconds <= 0.02:
        parser.error("--state-prediction-seconds must be in [0, 0.02]")
    if args.state_qos_depth < 1:
        parser.error("--state-qos-depth must be positive")
    if not 0.0 <= args.move_template_multiplier <= 1.5:
        parser.error("--move-template-multiplier must be in [0, 1.5]")
    if not 0.0 <= args.policy_vx_floor <= 0.60:
        parser.error("--policy-vx-floor must be in [0, 0.60]")
    if args.heading_recovery_enter_rad is not None:
        if not 0.0 <= args.heading_recovery_exit_rad < args.heading_recovery_enter_rad:
            parser.error("heading recovery requires 0 <= exit < enter")
    if args.heading_action_recovery_enter_rad is not None:
        if not 0.0 <= args.heading_action_recovery_exit_rad < args.heading_action_recovery_enter_rad:
            parser.error("heading action recovery requires 0 <= exit < enter")
        if args.heading_action_recovery_gain <= 0.0:
            parser.error("--heading-action-recovery-gain must be positive")
        if not 0.0 < args.heading_action_recovery_limit <= 0.5:
            parser.error("--heading-action-recovery-limit must be in (0, 0.5]")
    if not 0.0 <= args.recovery_exit_m < args.recovery_enter_m:
        parser.error("recovery thresholds require 0 <= exit < enter")
    if args.recovery_slew_rate_per_s <= 0.0:
        parser.error("--recovery-slew-rate-per-s must be positive")
    if not 0.0 < args.action_ema_alpha <= 1.0:
        parser.error("--action-ema-alpha must be in (0, 1]")
    if not 0.0 <= args.waist_tilt_action_multiplier <= 1.0:
        parser.error("--waist-tilt-action-multiplier must be in [0, 1]")
    if args.upper_scale < 0.0:
        parser.error("--upper-scale must be non-negative")
    if args.upper_start_seconds < 0.0:
        parser.error("--upper-start-seconds must be non-negative")
    if args.upper_time_scale <= 0.0:
        parser.error("--upper-time-scale must be positive")
    if args.upper_max_excursion_rad <= 0.0 or args.upper_max_velocity_radps <= 0.0:
        parser.error("upper excursion and velocity limits must be positive")
    if args.upper_fallback_heading_rad <= 0.0:
        parser.error("--upper-fallback-heading-rad must be positive")
    if not 0.5 <= args.pd_kp_multiplier <= 1.5:
        parser.error("--pd-kp-multiplier must be in [0.5, 1.5]")
    if not 0.5 <= args.pd_kd_multiplier <= 1.5:
        parser.error("--pd-kd-multiplier must be in [0.5, 1.5]")
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
