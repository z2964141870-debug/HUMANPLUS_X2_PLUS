#!/usr/bin/env python3
"""Guarded X2 v1 current-pose hold test using Develop_MC."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
import socket
import sys
import threading
import time

import numpy as np

from protocol import seq_is_newer, unpack_command
from x2_hal_guard import ARM_SLICE, JOINT_NAMES, LOWER_LIMITS, UPPER_LIMITS
from x2_pd_contract import load_verified_sonic_pd_contract
from x2_v1_state_probe import transition_complete


CONFIRMATION = "X2_SUSPENDED_V1_HOLD"
LIVE_DEVELOP_MC_ENABLED = False
LIVE_MIRROR_ONLY_ENABLED = (
    os.environ.get("X2_ENABLE_V1_MIRROR_PROBE") == "SUSPENDED_OPERATOR_READY"
)
LIVE_TORQUE_STEP_ENABLED = (
    os.environ.get("X2_ENABLE_V1_TORQUE_STEP") == "SUSPENDED_OPERATOR_READY"
)
GET_STATE_SERVICE = "/aimdk_5Fmsgs/srv/GetSystemState"
MIGRATE_SERVICE = "/aimdk_5Fmsgs/srv/MigrateSystemState"
GROUPS = {
    "leg": ("/aima/hal/joint/leg", 0, 12),
    "waist": ("/aima/hal/joint/waist", 12, 3),
    "arm": ("/aima/hal/joint/arm", 15, 14),
    "head": ("/aima/hal/joint/head", 29, 2),
}

MAX_HOLD_DRIFT_RAD = 0.03
OFFICIAL_COMMAND_MAX_AGE_S = 0.10
MAX_SUSPENDED_TARGET_ERROR_RAD = 0.30
MAX_OFFICIAL_TARGET_CHANGE_RAD = 0.01
RT_RELAY_PREFIX = "/_x2_sonic_rt"
RT_RELAY_COMMAND_HZ = 100.0
RT_RELAY_MAX_P95_MS = 3.0
RT_RELAY_MAX_PERIOD_MS = 8.0
VIBRATION_WINDOW_S = 0.35
VIBRATION_MIN_WINDOW_S = 0.25
VIBRATION_MIN_SAMPLES = 20
VIBRATION_MIN_POSITION_SPAN_RAD = 0.008
VIBRATION_MIN_VELOCITY_RMS_RAD_S = 0.15
VIBRATION_MIN_ACCELERATION_RMS_RAD_S2 = 6.0
VIBRATION_MIN_REVERSAL_HZ = 6.0
VIBRATION_VELOCITY_DEADBAND_RAD_S = 0.05
SONIC_GAIN_RAMP_S = 2.0


@dataclass
class HandoffLifecycle:
    """Fail-closed ownership record used by cleanup and fault paths."""

    migration_requested: bool = False
    develop_confirmed: bool = False
    standing_verified: bool = False
    estop_confirmed: bool = False

    def may_release_mirror(self) -> bool:
        return (
            not self.migration_requested
            or self.standing_verified
            or self.estop_confirmed
        )

    def recovery_required(self) -> bool:
        return self.migration_requested and not self.may_release_mirror()


class VibrationMonitor:
    """Detect sustained, reversing joint motion during a nominal hold."""

    def __init__(self) -> None:
        self._samples: dict[str, deque] = {}
        self._peak = {
            "position_span_rad": 0.0,
            "velocity_rms_rad_s": 0.0,
            "acceleration_rms_rad_s2": 0.0,
            "reversal_hz": 0.0,
            "joint": "none",
        }

    def reset(self) -> None:
        self._samples.clear()
        for key in self._peak:
            self._peak[key] = "none" if key == "joint" else 0.0

    def summary(self) -> dict:
        return dict(self._peak)

    def update(
        self,
        group: str,
        names: list[str],
        position: np.ndarray,
        velocity: np.ndarray,
        now_s: float,
    ) -> str | None:
        samples = self._samples.setdefault(group, deque())
        samples.append((float(now_s), position.copy(), velocity.copy()))
        cutoff = now_s - VIBRATION_WINDOW_S
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if len(samples) < VIBRATION_MIN_SAMPLES:
            return None
        duration = samples[-1][0] - samples[0][0]
        if duration < VIBRATION_MIN_WINDOW_S:
            return None

        times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        q = np.stack([sample[1] for sample in samples])
        dq = np.stack([sample[2] for sample in samples])
        dt = np.diff(times)
        valid_dt = dt > 1e-4
        if np.count_nonzero(valid_dt) < VIBRATION_MIN_SAMPLES - 1:
            return None

        position_span = np.ptp(q, axis=0)
        velocity_rms = np.sqrt(np.mean(np.square(dq), axis=0))
        acceleration = np.diff(dq, axis=0)[valid_dt] / dt[valid_dt, None]
        acceleration_rms = np.sqrt(np.mean(np.square(acceleration), axis=0))

        signs = np.sign(dq)
        signs[np.abs(dq) < VIBRATION_VELOCITY_DEADBAND_RAD_S] = 0.0
        reversals = np.zeros(dq.shape[1], dtype=np.int64)
        previous = np.zeros(dq.shape[1], dtype=np.float64)
        for row in signs:
            active = row != 0.0
            reversals += active & (previous != 0.0) & (row != previous)
            previous[active] = row[active]
        reversal_hz = reversals / duration

        worst_velocity = int(np.argmax(velocity_rms))
        if velocity_rms[worst_velocity] > self._peak["velocity_rms_rad_s"]:
            self._peak = {
                "position_span_rad": float(position_span[worst_velocity]),
                "velocity_rms_rad_s": float(velocity_rms[worst_velocity]),
                "acceleration_rms_rad_s2": float(
                    acceleration_rms[worst_velocity]
                ),
                "reversal_hz": float(reversal_hz[worst_velocity]),
                "joint": names[worst_velocity],
            }

        tripped = (
            (position_span >= VIBRATION_MIN_POSITION_SPAN_RAD)
            & (velocity_rms >= VIBRATION_MIN_VELOCITY_RMS_RAD_S)
            & (acceleration_rms >= VIBRATION_MIN_ACCELERATION_RMS_RAD_S2)
            & (reversal_hz >= VIBRATION_MIN_REVERSAL_HZ)
        )
        if not np.any(tripped):
            return None
        candidates = np.flatnonzero(tripped)
        worst = int(candidates[np.argmax(velocity_rms[candidates])])
        return (
            f"vibration {names[worst]} span={position_span[worst]:.4f}rad "
            f"velocity_rms={velocity_rms[worst]:.3f}rad/s "
            f"acceleration_rms={acceleration_rms[worst]:.1f}rad/s^2 "
            f"reversals={reversal_hz[worst]:.1f}Hz"
        )


def relay_status_problem(status: dict, require_active: bool = True) -> str | None:
    if not status:
        return "C++ 500Hz relay status has not been received"
    if bool(status.get("fault", False)):
        return "C++ 500Hz relay fault is latched"
    if not bool(status.get("ready", False)):
        return "C++ 500Hz relay has not received all four command groups"
    if require_active and not bool(status.get("active", False)):
        return "C++ 500Hz relay is not actively publishing"
    try:
        p95_ms = float(status["period_p95_ms"])
        max_ms = float(status["period_max_ms"])
    except (KeyError, TypeError, ValueError):
        return "C++ 500Hz relay timing fields are invalid"
    if not np.isfinite(p95_ms) or not np.isfinite(max_ms):
        return "C++ 500Hz relay timing fields are non-finite"
    if p95_ms > RT_RELAY_MAX_P95_MS:
        return f"C++ 500Hz relay p95 period is too high: {p95_ms:.3f}ms"
    if max_ms > RT_RELAY_MAX_PERIOD_MS:
        return f"C++ 500Hz relay maximum period is too high: {max_ms:.3f}ms"
    return None


def takeover_target_match_required(
    mirror_only: bool,
    allow_suspended_target_offset: bool,
) -> bool:
    return not (mirror_only or allow_suspended_target_offset)


def is_official_command_sequence(sequence: int) -> bool:
    return int(sequence) > 0


def official_profile_problem(
    profile: dict[str, tuple[float, float, float, float, float]]
) -> str | None:
    missing = sorted(set(JOINT_NAMES) - set(profile))
    extra = sorted(set(profile) - set(JOINT_NAMES))
    if missing or extra:
        return f"official command profile mismatch: missing={missing} extra={extra}"
    for name in JOINT_NAMES:
        position, velocity, effort, stiffness, damping = profile[name]
        values = (position, velocity, effort, stiffness, damping)
        if not all(np.isfinite(value) for value in values):
            return f"official command field is non-finite: {name}"
        if abs(velocity) > 0.10:
            return f"official command is not stationary: {name} velocity={velocity}"
        if stiffness <= 0.0 or damping < 0.0:
            return (
                "official command is not a powered position hold: "
                f"{name} stiffness={stiffness} damping={damping}; "
                "enter official Standing first"
            )
    return None


def official_target_problem(
    profile: dict[str, tuple[float, float, float, float, float]],
    measured: np.ndarray,
) -> str | None:
    target = np.asarray([profile[name][0] for name in JOINT_NAMES], dtype=np.float64)
    error = np.abs(target[:29] - np.asarray(measured, dtype=np.float64)[:29])
    worst = int(np.argmax(error))
    if error[worst] > 0.15:
        return (
            f"official target differs from measured pose: {JOINT_NAMES[worst]} "
            f"target={target[worst]:.3f}rad measured={measured[worst]:.3f}rad "
            f"error={error[worst]:.3f}rad"
        )
    return None


def suspended_target_problem(
    profile: dict[str, tuple[float, float, float, float, float]],
    measured: np.ndarray,
) -> str | None:
    target = np.asarray([profile[name][0] for name in JOINT_NAMES], dtype=np.float64)
    error = np.abs(target[:29] - np.asarray(measured, dtype=np.float64)[:29])
    worst = int(np.argmax(error))
    if error[worst] > MAX_SUSPENDED_TARGET_ERROR_RAD:
        return (
            f"suspended target offset is too large: {JOINT_NAMES[worst]} "
            f"target={target[worst]:.3f}rad measured={measured[worst]:.3f}rad "
            f"error={error[worst]:.3f}rad"
        )
    return None


def torque_equivalent_profile(
    profile: dict[str, tuple[float, float, float, float, float]],
    measured_position: np.ndarray,
    measured_velocity: np.ndarray,
    contract,
    converted_joints: tuple[str, ...],
) -> tuple[dict[str, tuple[float, float, float, float, float]], dict]:
    problem = official_profile_problem(profile)
    if problem:
        raise ValueError(problem)
    q = np.asarray(measured_position, dtype=np.float64).reshape(-1)
    dq = np.asarray(measured_velocity, dtype=np.float64).reshape(-1)
    if q.shape != (31,) or dq.shape != (31,) or not (
        np.all(np.isfinite(q)) and np.all(np.isfinite(dq))
    ):
        raise ValueError("measured position/velocity must be finite shape (31,)")
    if not converted_joints or len(set(converted_joints)) != len(converted_joints):
        raise ValueError("converted_joints must be nonempty and unique")

    candidate = dict(profile)
    details = []
    for name in converted_joints:
        if name not in JOINT_NAMES:
            raise ValueError(f"unknown converted joint: {name}")
        index = JOINT_NAMES.index(name)
        target, velocity, effort, stiffness, damping = profile[name]
        official_torque = (
            effort
            + stiffness * (target - q[index])
            + damping * (velocity - dq[index])
        )
        candidate_velocity = 0.0
        candidate_effort = 0.0
        candidate_stiffness = float(contract.stiffness[index])
        candidate_damping = float(contract.damping[index])
        candidate_target = q[index] + (
            official_torque
            - candidate_effort
            - candidate_damping * (candidate_velocity - dq[index])
        ) / candidate_stiffness
        if not contract.lower_limit[index] <= candidate_target <= contract.upper_limit[index]:
            raise ValueError(
                f"torque-equivalent target outside model limit: {name}="
                f"{candidate_target:.3f}rad"
            )
        candidate_torque = (
            candidate_effort
            + candidate_stiffness * (candidate_target - q[index])
            + candidate_damping * (candidate_velocity - dq[index])
        )
        residual = abs(candidate_torque - official_torque)
        if residual > 1e-9:
            raise ValueError(f"torque conversion residual is too high: {name}={residual}")
        candidate[name] = (
            float(candidate_target),
            candidate_velocity,
            candidate_effort,
            candidate_stiffness,
            candidate_damping,
        )
        details.append({
            "joint": name,
            "official_torque_nm": float(official_torque),
            "candidate_target_rad": float(candidate_target),
            "candidate_offset_rad": float(candidate_target - q[index]),
            "torque_residual_nm": float(residual),
        })
    return candidate, {
        "contract_fingerprint": contract.fingerprint,
        "converted": details,
        "max_torque_residual_nm": max(
            item["torque_residual_nm"] for item in details
        ),
    }


def blended_torque_equivalent_profile(
    profile: dict[str, tuple[float, float, float, float, float]],
    measured_position: np.ndarray,
    measured_velocity: np.ndarray,
    contract,
    alpha: float,
) -> tuple[dict[str, tuple[float, float, float, float, float]], dict]:
    """Blend official gains to Sonic while preserving instantaneous torque."""
    problem = official_profile_problem(profile)
    if problem:
        raise ValueError(problem)
    q = np.asarray(measured_position, dtype=np.float64).reshape(-1)
    dq = np.asarray(measured_velocity, dtype=np.float64).reshape(-1)
    if q.shape != (31,) or dq.shape != (31,) or not (
        np.all(np.isfinite(q)) and np.all(np.isfinite(dq))
    ):
        raise ValueError("measured position/velocity must be finite shape (31,)")
    blend = float(alpha)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("gain blend alpha must be finite in [0, 1]")

    candidate = {}
    max_residual = 0.0
    max_offset = 0.0
    max_offset_joint = ""
    for index, name in enumerate(JOINT_NAMES):
        target, velocity, effort, official_kp, official_kd = profile[name]
        torque = (
            effort
            + official_kp * (target - q[index])
            + official_kd * (velocity - dq[index])
        )
        kp = (1.0 - blend) * official_kp + blend * float(
            contract.stiffness[index]
        )
        kd = (1.0 - blend) * official_kd + blend * float(
            contract.damping[index]
        )
        candidate_velocity = 0.0
        candidate_effort = 0.0
        candidate_target = q[index] + (
            torque - candidate_effort - kd * (candidate_velocity - dq[index])
        ) / kp
        if not contract.lower_limit[index] <= candidate_target <= contract.upper_limit[index]:
            raise ValueError(
                f"blended torque-equivalent target outside model limit: "
                f"{name}={candidate_target:.3f}rad alpha={blend:.3f}"
            )
        candidate_torque = (
            candidate_effort
            + kp * (candidate_target - q[index])
            + kd * (candidate_velocity - dq[index])
        )
        residual = abs(candidate_torque - torque)
        max_residual = max(max_residual, residual)
        offset = abs(candidate_target - q[index])
        if offset > max_offset:
            max_offset = offset
            max_offset_joint = name
        candidate[name] = (
            float(candidate_target),
            candidate_velocity,
            candidate_effort,
            float(kp),
            float(kd),
        )
    return candidate, {
        "alpha": blend,
        "max_torque_residual_nm": float(max_residual),
        "max_offset_rad": float(max_offset),
        "max_offset_joint": max_offset_joint,
    }


def official_profile_change_problem(
    before: dict[str, tuple[float, float, float, float, float]],
    after: dict[str, tuple[float, float, float, float, float]],
) -> str | None:
    if set(before) != set(JOINT_NAMES) or set(after) != set(JOINT_NAMES):
        return "official command profile is incomplete during stability check"
    for name in JOINT_NAMES:
        position_change = abs(after[name][0] - before[name][0])
        if position_change > MAX_OFFICIAL_TARGET_CHANGE_RAD:
            return (
                f"official target is changing: {name} "
                f"delta={position_change:.3f}rad"
            )
        for field, index in (("stiffness", 3), ("damping", 4)):
            baseline = max(abs(before[name][index]), 1.0)
            relative_change = abs(after[name][index] - before[name][index]) / baseline
            if relative_change > 0.05:
                return f"official {field} is changing: {name} ratio={relative_change:.3f}"
    return None


def official_profile_fresh_problem(
    profile: dict[str, tuple[float, float, float, float, float]],
    group_times: dict[str, float],
    after_s: float,
    now_s: float,
) -> str | None:
    problem = official_profile_problem(profile)
    if problem:
        return problem
    missing = sorted(set(GROUPS) - set(group_times))
    if missing:
        return f"official command groups missing timestamps: {missing}"
    for group in GROUPS:
        received_s = group_times[group]
        if received_s <= after_s:
            return f"official {group} command predates handback"
        age = now_s - received_s
        if age > OFFICIAL_COMMAND_MAX_AGE_S:
            return f"official {group} command is stale: age={age:.3f}s"
    return None


def official_handoff_problem(
    state: str,
    status: int,
    in_ready: int,
    profile: dict[str, tuple[float, float, float, float, float]],
    group_times: dict[str, float],
    after_s: float,
    now_s: float,
) -> str | None:
    if state.lower() != "business" or status != in_ready:
        return f"system is not stable Business: state={state} status={status}"
    return official_profile_fresh_problem(profile, group_times, after_s, now_s)


def hold_drift_problem(
    q: np.ndarray, target: np.ndarray, arms_active: bool = False
) -> str | None:
    checked = 15 if arms_active else 29
    error = np.abs(np.asarray(q, dtype=np.float64)[:checked] - target[:checked])
    worst = int(np.argmax(error))
    if error[worst] > MAX_HOLD_DRIFT_RAD:
        return f"hold drift {JOINT_NAMES[worst]}={error[worst]:.3f}rad"
    return None


def anchor_problem(q: np.ndarray, before: np.ndarray | None = None) -> str | None:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (31,) or not np.all(np.isfinite(q)):
        return "joint position snapshot is invalid"
    outside = (q < LOWER_LIMITS - 0.10) | (q > UPPER_LIMITS + 0.10)
    if np.any(outside[:29]):
        names = [JOINT_NAMES[i] for i in np.flatnonzero(outside[:29])]
        return f"body joints outside limits: {names}"
    if before is not None:
        delta = np.abs(q[:29] - np.asarray(before, dtype=np.float64)[:29])
        worst = int(np.argmax(delta))
        if delta[worst] > 0.05:
            return (
                f"pose changed during migration: {JOINT_NAMES[worst]} "
                f"delta={delta[worst]:.3f}rad"
            )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X2 v1 suspended current-pose hold probe")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--sonic-seconds", type=float, default=0.0)
    parser.add_argument("--command-bind", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=50051)
    parser.add_argument("--command-source-host", default="127.0.0.1")
    parser.add_argument("--step-joint", choices=JOINT_NAMES[15:29])
    parser.add_argument("--step-rad", type=float, default=0.03)
    parser.add_argument("--mirror-only", action="store_true")
    parser.add_argument("--allow-suspended-target-offset", action="store_true")
    parser.add_argument("--torque-equivalent-step", action="store_true")
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        parser.error(f"HAL hold requires --confirm {CONFIRMATION}")
    if not 1.0 <= args.hold_seconds <= 10.0:
        parser.error("--hold-seconds must be in [1, 10]")
    if not 0.0 <= args.sonic_seconds <= 10.0:
        parser.error("--sonic-seconds must be in [0, 10]")
    if not 0.0 < args.step_rad <= 0.05:
        parser.error("--step-rad must be in (0, 0.05]")
    if args.step_joint and args.sonic_seconds > 0.0:
        parser.error("--step-joint and --sonic-seconds are mutually exclusive")
    if args.mirror_only and (args.step_joint or args.sonic_seconds > 0.0):
        parser.error("--mirror-only cannot be combined with motion")
    if args.mirror_only and args.allow_suspended_target_offset:
        parser.error("mirror-only does not use takeover target-offset authorization")
    if args.torque_equivalent_step and args.step_joint != "left_shoulder_pitch_joint":
        parser.error("torque-equivalent first motion is limited to left_shoulder_pitch_joint")
    if args.torque_equivalent_step and args.allow_suspended_target_offset:
        parser.error("torque-equivalent step does not use the legacy target-offset exception")
    return args


def main() -> int:
    args = parse_args()
    if args.mirror_only:
        live_enabled = LIVE_MIRROR_ONLY_ENABLED
    elif args.torque_equivalent_step:
        live_enabled = LIVE_TORQUE_STEP_ENABLED
    else:
        live_enabled = LIVE_DEVELOP_MC_ENABLED
    if not live_enabled:
        print(
            "[v1-hold] BLOCKED: candidate hardware probe is not enabled; "
            "no ROS node, state migration, or HAL publisher was started",
            file=sys.stderr,
            flush=True,
        )
        return 1
    import rclpy
    import aimdk_msgs.msg as aimdk_msg
    import aimdk_msgs.srv as aimdk_srv
    from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, String

    rclpy.init()
    node = Node("x2_v1_hold_probe")
    service_node = Node("x2_v1_hold_probe_services")
    subscriber_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
    publisher_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )
    official_command_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    lock = threading.Lock()
    positions = np.zeros(31, dtype=np.float64)
    velocities = np.zeros(31, dtype=np.float64)
    seen = {group: False for group in GROUPS}
    state_times = {group: 0.0 for group in GROUPS}
    official_profile = {}
    official_command_seen = {group: False for group in GROUPS}
    official_command_times = {group: 0.0 for group in GROUPS}
    official_sequences = {group: 0 for group in GROUPS}
    command_lock = threading.RLock()
    captured_profile = {}
    command_anchor: np.ndarray | None = None
    feedback_anchor: np.ndarray | None = None
    publishers = {}
    publishing = False
    control_running = True
    relay = {
        "status": {},
        "status_s": 0.0,
        "input_cycles": 0,
        "enabled": False,
    }
    command_socket = None
    network = {
        "target": None,
        "received_s": 0.0,
        "sequence": None,
        "accepted": 0,
        "bad": 0,
        "wrong_source": 0,
    }
    motion = {
        "active": False,
        "fault": None,
        "output": None,
        "last_s": None,
        "desired": None,
        "command_peak": 0.0,
        "measured_peak": 0.0,
    }
    vibration_monitor = VibrationMonitor()
    if args.sonic_seconds > 0.0:
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        command_socket.bind((args.command_bind, args.command_port))
        command_socket.setblocking(False)
        expected_command_source = socket.gethostbyname(args.command_source_host)
    else:
        expected_command_source = None

    def on_state(group: str, msg) -> None:
        _, start, length = GROUPS[group]
        expected = JOINT_NAMES[start:start + length]
        by_name = {str(joint.name): joint for joint in msg.joints}
        if set(by_name) != set(expected):
            with lock:
                seen[group] = False
            return
        q = np.asarray([float(by_name[name].position) for name in expected])
        dq = np.asarray([float(by_name[name].velocity) for name in expected])
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            return
        now_s = time.monotonic()
        with lock:
            positions[start:start + length] = q
            velocities[start:start + length] = dq
            seen[group] = True
            state_times[group] = now_s
        if relay["enabled"] and motion["fault"] is None:
            problem = vibration_monitor.update(group, expected, q, dq, now_s)
            if problem is not None:
                motion["active"] = False
                motion["fault"] = problem

    subscriptions = [
        node.create_subscription(
            JointStateArray,
            f"{base}/state",
            lambda msg, name=group: on_state(name, msg),
            subscriber_qos,
        )
        for group, (base, _, _) in GROUPS.items()
    ]

    def on_official_command(group: str, msg) -> None:
        sequence = int(msg.header.sequence)
        if not is_official_command_sequence(sequence):
            return
        _, start, length = GROUPS[group]
        expected = JOINT_NAMES[start:start + length]
        by_name = {str(joint.name): joint for joint in msg.joints}
        if set(by_name) != set(expected):
            official_command_seen[group] = False
            return
        for name in expected:
            official_profile[name] = (
                float(by_name[name].position),
                float(by_name[name].velocity),
                float(by_name[name].effort),
                float(by_name[name].stiffness),
                float(by_name[name].damping),
            )
        official_command_seen[group] = True
        official_command_times[group] = time.monotonic()
        official_sequences[group] = sequence

    command_subscriptions = [
        node.create_subscription(
            JointCommandArray,
            f"{base}/command",
            lambda msg, name=group: on_official_command(name, msg),
            official_command_qos,
        )
        for group, (base, _, _) in GROUPS.items()
    ]

    def on_relay_status(msg) -> None:
        try:
            status = json.loads(str(msg.data))
        except (TypeError, ValueError):
            return
        if not isinstance(status, dict):
            return
        relay["status"] = status
        relay["status_s"] = time.monotonic()
        if relay["enabled"] and motion["fault"] is None:
            problem = relay_status_problem(status, require_active=True)
            if problem is not None:
                motion["fault"] = problem

    relay_enable_publisher = node.create_publisher(
        Bool, f"{RT_RELAY_PREFIX}/enable", publisher_qos
    )
    relay_status_subscription = node.create_subscription(
        String,
        f"{RT_RELAY_PREFIX}/status",
        on_relay_status,
        publisher_qos,
    )
    del subscriptions, command_subscriptions, relay_status_subscription

    def snapshot(max_age_s: float = 0.08):
        now = time.monotonic()
        with lock:
            if not all(seen.values()):
                return None
            age = max(now - state_times[group] for group in GROUPS)
            if age > max_age_s:
                return None
            return positions.copy(), velocities.copy(), age

    def spin_future(future, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            # Service responses have a dedicated wait set so high-rate HAL
            # state, command, and timer callbacks cannot starve them. Keep the
            # control node progressing as well so mirror publication continues
            # throughout a migration request.
            rclpy.spin_once(service_node, timeout_sec=0.01)
            rclpy.spin_once(node, timeout_sec=0.0)
        if not future.done():
            future.cancel()
            raise TimeoutError("ROS service response timeout")
        result = future.result()
        if result is None:
            raise RuntimeError("ROS service returned no result")
        return result

    def stamp(request) -> None:
        now = node.get_clock().now().nanoseconds
        request.header.header.stamp.sec = now // 1_000_000_000
        request.header.header.stamp.nanosec = now % 1_000_000_000

    get_client = service_node.create_client(
        aimdk_srv.GetSystemState, GET_STATE_SERVICE
    )
    migrate_client = service_node.create_client(
        aimdk_srv.MigrateSystemState, MIGRATE_SERVICE
    )

    def get_state() -> tuple[str, int]:
        last_error = None
        for attempt in range(1, 4):
            request = aimdk_srv.GetSystemState.Request()
            stamp(request)
            try:
                response = spin_future(get_client.call_async(request), 5.0)
            except TimeoutError as exc:
                last_error = exc
                print(
                    f"[v1-hold] GetSystemState retry {attempt}/3 after timeout",
                    flush=True,
                )
                continue
            if response.header.status.value != aimdk_msg.CommonState.SUCCESS:
                raise RuntimeError(
                    f"GetSystemState failed: {response.header.message}"
                )
            return str(response.cur_state), int(response.curr_status.value)
        raise TimeoutError(f"GetSystemState failed after 3 attempts: {last_error}")

    def migrate(target: str) -> None:
        request = aimdk_srv.MigrateSystemState.Request()
        stamp(request)
        request.state = target
        response = spin_future(migrate_client.call_async(request), 10.0)
        if response.header.status.value != aimdk_msg.CommonState.SUCCESS:
            raise RuntimeError(f"migration request failed: {response.header.message}")
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            if target.lower() == "develop_mc" and motion["fault"] is not None:
                raise RuntimeError(
                    f"takeover aborted by motion guard: {motion['fault']}"
                )
            state, status = get_state()
            print(f"[v1-hold] waiting target={target} state={state} status={status}", flush=True)
            if transition_complete(target, state, status, aimdk_msg.SystemStatus.IN_READY):
                print(f"[v1-hold] reached {target} via state={state} status={status}", flush=True)
                return
            end = time.monotonic() + 0.05
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.002)
        raise TimeoutError(f"migration to {target} did not complete")

    def wait_fresh(timeout_s: float):
        deadline = time.monotonic() + timeout_s
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.002)
            current = snapshot()
            if current is None or np.max(np.abs(current[1][:29])) > 0.10:
                stable_since = None
                continue
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.20:
                return current
        raise TimeoutError("fresh, stable HAL feedback did not recover")

    def wait_official_profile(
        timeout_s: float,
        after_s: float = -1.0,
    ) -> dict[str, tuple[float, float, float, float, float]]:
        deadline = time.monotonic() + timeout_s
        last_problem = "official command groups have not all been received"
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.002)
            if all(official_command_seen.values()):
                profile = dict(official_profile)
                problem = official_profile_fresh_problem(
                    profile,
                    dict(official_command_times),
                    after_s,
                    time.monotonic(),
                )
                if problem is None:
                    return profile
                last_problem = problem
        raise TimeoutError(
            f"complete fresh official Standing profile was not received: {last_problem}"
        )

    def wait_stable_official_profile(
        timeout_s: float,
        stable_s: float = 0.5,
    ) -> dict[str, tuple[float, float, float, float, float]]:
        deadline = time.monotonic() + timeout_s
        baseline = None
        stable_since = None
        last_problem = "official command profile has not stabilized"
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.002)
            if not all(official_command_seen.values()):
                baseline = None
                stable_since = None
                continue
            current = dict(official_profile)
            freshness = official_profile_fresh_problem(
                current,
                dict(official_command_times),
                -1.0,
                time.monotonic(),
            )
            if freshness:
                last_problem = freshness
                baseline = None
                stable_since = None
                continue
            if baseline is None:
                baseline = current
                stable_since = time.monotonic()
                continue
            change = official_profile_change_problem(baseline, current)
            if change:
                last_problem = change
                baseline = current
                stable_since = time.monotonic()
                continue
            if time.monotonic() - stable_since >= stable_s:
                return current
        raise TimeoutError(f"official Standing command is not stable: {last_problem}")

    def make_message(group: str, target: np.ndarray):
        _, start, length = GROUPS[group]
        msg = JointCommandArray()
        msg.header.sequence = 0
        for index in range(start, start + length):
            joint = JointCommand()
            joint.name = JOINT_NAMES[index]
            joint.position = float(target[index])
            joint.velocity = 0.0
            joint.effort = 0.0
            _, velocity, effort, kp, kd = captured_profile[joint.name]
            joint.velocity = velocity
            joint.effort = effort
            joint.stiffness = kp
            joint.damping = kd
            msg.joints.append(joint)
        return msg

    def control_tick() -> None:
        if command_socket is not None:
            for _ in range(32):
                try:
                    raw, sender = command_socket.recvfrom(4096)
                except BlockingIOError:
                    break
                if sender[0] != expected_command_source:
                    network["wrong_source"] += 1
                    continue
                try:
                    packet_sequence, _, target = unpack_command(raw)
                    previous_sequence = network["sequence"]
                    if previous_sequence is not None and not seq_is_newer(
                        packet_sequence, previous_sequence
                    ):
                        raise ValueError("replayed or out-of-order target")
                except ValueError:
                    network["bad"] += 1
                    continue
                network["target"] = target
                network["received_s"] = time.monotonic()
                network["sequence"] = packet_sequence
                network["accepted"] += 1
        with command_lock:
            if publishing is False or command_anchor is None or feedback_anchor is None:
                return
            now = time.monotonic()
            output = motion["output"]
            if output is None:
                output = command_anchor.copy()
            if motion["active"]:
                if args.step_joint:
                    desired = motion["desired"]
                elif network["target"] is None or now - network["received_s"] > 0.12:
                    desired = None
                    motion["active"] = False
                    motion["fault"] = "Sonic target timeout"
                else:
                    desired = command_anchor.copy()
                    desired[ARM_SLICE] = network["target"][ARM_SLICE]
                    lower = np.maximum(LOWER_LIMITS, command_anchor - 0.05)
                    upper = np.minimum(UPPER_LIMITS, command_anchor + 0.05)
                    desired = np.clip(desired, lower, upper)
                if desired is not None:
                    elapsed = 0.0 if motion["last_s"] is None else now - motion["last_s"]
                    max_step = 0.08 * max(0.0, elapsed)
                    output = output + np.clip(desired - output, -max_step, max_step)
                    current = snapshot()
                    if current is not None:
                        motion["command_peak"] = max(
                            motion["command_peak"],
                            float(
                                np.max(
                                    np.abs(
                                        output[ARM_SLICE] - command_anchor[ARM_SLICE]
                                    )
                                )
                            ),
                        )
                        motion["measured_peak"] = max(
                            motion["measured_peak"],
                            float(
                                np.max(
                                    np.abs(
                                        current[0][ARM_SLICE] - feedback_anchor[ARM_SLICE]
                                    )
                                )
                            ),
                        )
                        commanded_delta = output[:29] - command_anchor[:29]
                        measured_delta = current[0][:29] - feedback_anchor[:29]
                        error = np.abs(commanded_delta - measured_delta)
                        worst = int(np.argmax(error))
                        if error[worst] > 0.12:
                            motion["active"] = False
                            motion["fault"] = (
                                f"tracking error {JOINT_NAMES[worst]}={error[worst]:.3f}rad"
                            )
            motion["output"] = output
            motion["last_s"] = now
            current = snapshot()
            if current is not None and motion["fault"] is None:
                motion["fault"] = hold_drift_problem(
                    current[0], feedback_anchor, arms_active=bool(motion["active"])
                )
                if motion["fault"] is not None:
                    motion["active"] = False
            messages = {
                group: make_message(group, output) for group in publishers
            }
            relay["input_cycles"] += 1
        for group, message in messages.items():
            publishers[group].publish(message)

    def control_loop() -> None:
        period_s = 1.0 / RT_RELAY_COMMAND_HZ
        deadline = time.monotonic()
        while control_running and rclpy.ok():
            deadline += period_s
            try:
                control_tick()
            except Exception as exc:  # Keep recovery in the main thread.
                motion["fault"] = f"candidate command loop failed: {exc}"
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -period_s:
                deadline = time.monotonic()

    control_thread = threading.Thread(
        target=control_loop,
        name="x2_candidate_command_loop",
        daemon=False,
    )
    control_thread.start()
    lifecycle = HandoffLifecycle()

    def create_hold_publishers() -> None:
        nonlocal publishing
        relay["input_cycles"] = 0
        for group in GROUPS:
            publishers[group] = node.create_publisher(
                JointCommandArray,
                f"{RT_RELAY_PREFIX}/{group}/command",
                publisher_qos,
            )
        publishing = True

    def send_relay_enable(enabled: bool) -> None:
        if enabled:
            vibration_monitor.reset()
        msg = Bool()
        msg.data = enabled
        relay["enabled"] = enabled
        for _ in range(3):
            relay_enable_publisher.publish(msg)
            time.sleep(0.01)

    def wait_relay_ready(timeout_s: float, require_active: bool = True) -> dict:
        deadline = time.monotonic() + timeout_s
        last_problem = "C++ 500Hz relay status has not been received"
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.002)
            status = dict(relay["status"])
            if time.monotonic() - relay["status_s"] > 1.5:
                last_problem = "C++ 500Hz relay status is stale"
                continue
            problem = relay_status_problem(status, require_active=require_active)
            if problem is None:
                return status
            last_problem = problem
        raise TimeoutError(last_problem)

    def destroy_hold_publishers() -> None:
        nonlocal publishing
        send_relay_enable(False)
        publishing = False
        for publisher in list(publishers.values()):
            node.destroy_publisher(publisher)
        publishers.clear()

    def apply_mirror_profile(
        profile: dict[str, tuple[float, float, float, float, float]],
        measured: np.ndarray,
        require_target_match: bool = True,
    ) -> None:
        nonlocal captured_profile, command_anchor, feedback_anchor
        problem = official_profile_problem(profile)
        if problem is None and require_target_match:
            problem = official_target_problem(profile, measured)
        if problem:
            raise RuntimeError(problem)
        with command_lock:
            captured_profile = dict(profile)
            command_anchor = np.asarray(
                [captured_profile[name][0] for name in JOINT_NAMES], dtype=np.float64
            )
            feedback_anchor = np.asarray(measured, dtype=np.float64).copy()
            motion["output"] = command_anchor.copy()
            motion["desired"] = command_anchor.copy()
            motion["active"] = False
            motion["last_s"] = time.monotonic()

    def apply_gain_ramp_profile(
        profile: dict[str, tuple[float, float, float, float, float]],
    ) -> None:
        nonlocal captured_profile, command_anchor
        anchor = np.asarray(
            [profile[name][0] for name in JOINT_NAMES], dtype=np.float64
        )
        with command_lock:
            captured_profile = dict(profile)
            command_anchor = anchor
            motion["output"] = anchor.copy()
            motion["desired"] = anchor.copy()
            motion["active"] = False
            motion["last_s"] = time.monotonic()

    def safe_handback_to_standing() -> None:
        motion["active"] = False
        motion["desired"] = command_anchor.copy() if command_anchor is not None else None
        marker = time.monotonic()
        state, _ = get_state()
        if state.lower() != "business":
            print(
                "[v1-hold] HANDOFF: requesting official Ready while mirror hold stays active",
                flush=True,
            )
            migrate("Ready")
        lifecycle.develop_confirmed = False
        print(
            "[v1-hold] HANDOFF WAIT: use the remote/app to enter official Standing now; "
            "mirror hold remains active",
            flush=True,
        )
        stable_since = None
        last_poll = 0.0
        last_report = 0.0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            if now - last_poll < 0.5:
                continue
            last_poll = now
            try:
                state, status = get_state()
            except (RuntimeError, TimeoutError):
                stable_since = None
                continue
            profile = dict(official_profile)
            problem = official_handoff_problem(
                state,
                status,
                aimdk_msg.SystemStatus.IN_READY,
                profile,
                dict(official_command_times),
                marker,
                time.monotonic(),
            )
            current = snapshot(0.20)
            if problem is not None or current is None:
                stable_since = None
                if now - last_report >= 2.0:
                    last_report = now
                    print(
                        f"[v1-hold] HANDOFF WAIT: state={state} status={status} "
                        f"official={problem or 'waiting for fresh joint feedback'}",
                        flush=True,
                    )
                continue
            apply_mirror_profile(
                profile,
                current[0],
                require_target_match=False,
            )
            if stable_since is None:
                stable_since = time.monotonic()
                continue
            if time.monotonic() - stable_since >= 1.0:
                break
        if not rclpy.ok():
            raise RuntimeError("ROS stopped before official Standing handback completed")
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.002)
        lifecycle.standing_verified = True
        destroy_hold_publishers()
        print(
            "[v1-hold] HANDOFF COMPLETE: fresh powered official Standing commands "
            "confirmed; mirror publisher released",
            flush=True,
        )

    try:
        if not (get_client.wait_for_service(10.0) and migrate_client.wait_for_service(10.0)):
            raise RuntimeError("X2 v1 system-state services are unavailable")
        state, status = get_state()
        print(f"[v1-hold] initial state={state} status={status}", flush=True)
        if state.lower() != "business":
            raise RuntimeError("official Standing/Business is required before takeover")
        print("[v1-hold] Robot must be suspended with physical E-stop ready.", flush=True)
        confirmation = "MIRROR" if args.mirror_only else (
            "STEP" if args.step_joint else (
                "SONIC" if args.sonic_seconds > 0.0 else "HOLD"
            )
        )
        if args.mirror_only:
            print(
                "[v1-hold] Type MIRROR then Enter for a no-migration official-command "
                "overlap test:",
                flush=True,
            )
        elif args.step_joint:
            print(
                f"[v1-hold] Type STEP then Enter for {args.step_joint} "
                f"+{args.step_rad:.3f}rad and return:",
                flush=True,
            )
        elif args.sonic_seconds > 0.0:
            print(
                "[v1-hold] Type SONIC then Enter for a guarded 5-second arms-only trial:",
                flush=True,
            )
        else:
            print(
                "[v1-hold] Type HOLD then Enter to publish a 3-second current-pose hold:",
                flush=True,
            )
        if input().strip() != confirmation:
            print("[v1-hold] cancelled; no state or HAL command change", flush=True)
            return 2
        before, before_velocity, _ = wait_fresh(5.0)
        problem = anchor_problem(before)
        if problem:
            raise RuntimeError(problem)
        official_capture = wait_stable_official_profile(5.0)
        problem = official_target_problem(official_capture, before)
        if args.torque_equivalent_step:
            contract = load_verified_sonic_pd_contract()
            _, conversion = torque_equivalent_profile(
                official_capture,
                before,
                before_velocity,
                contract,
                tuple(JOINT_NAMES),
            )
            converted_by_name = {
                item["joint"]: item for item in conversion["converted"]
            }
            converted = converted_by_name[args.step_joint]
            max_offset = max(
                conversion["converted"],
                key=lambda item: abs(item["candidate_offset_rad"]),
            )
            print(
                "[v1-hold] TORQUE-EQUIVALENT FULL BODY: "
                f"joints={len(conversion['converted'])} "
                f"max_offset={max_offset['joint']}="
                f"{max_offset['candidate_offset_rad']:.4f}rad "
                f"max_residual={conversion['max_torque_residual_nm']:.3e}Nm "
                f"contract={conversion['contract_fingerprint']}",
                flush=True,
            )
            print(
                "[v1-hold] TORQUE-EQUIVALENT LEFT SHOULDER STEP ANCHOR: "
                f"official_tau={converted['official_torque_nm']:.4f}Nm "
                f"candidate={converted['candidate_target_rad']:.4f}rad "
                f"offset={converted['candidate_offset_rad']:.4f}rad "
                f"residual={converted['torque_residual_nm']:.3e}Nm",
                flush=True,
            )
        elif problem and not args.mirror_only:
            if not args.allow_suspended_target_offset:
                raise RuntimeError(
                    f"{problem}; suspended takeover requires explicit target-offset "
                    "authorization"
                )
            suspended_problem = suspended_target_problem(official_capture, before)
            if suspended_problem:
                raise RuntimeError(suspended_problem)
            print(
                f"[v1-hold] SUSPENDED NOTE: {problem}; bounded offset accepted after "
                "mirror-only hardware validation",
                flush=True,
            )
        if problem and not args.torque_equivalent_step:
            print(
                f"[v1-hold] MIRROR NOTE: {problem}; exact official command will be "
                "duplicated without state migration",
                flush=True,
            )
        print("[v1-hold] official Standing gains captured for all 31 joints", flush=True)
        apply_mirror_profile(
            official_capture,
            before,
            require_target_match=(
                False if args.torque_equivalent_step
                else takeover_target_match_required(
                    args.mirror_only,
                    args.allow_suspended_target_offset,
                )
            ),
        )
        create_hold_publishers()
        prewarm_deadline = time.monotonic() + 0.5
        while time.monotonic() < prewarm_deadline:
            rclpy.spin_once(node, timeout_sec=0.002)
        if motion["fault"] is not None:
            raise RuntimeError(f"pre-migration mirror hold failed: {motion['fault']}")
        if relay["input_cycles"] < 20:
            raise RuntimeError(
                "candidate command loop did not sustain its 100Hz relay input"
            )
        send_relay_enable(True)
        relay_status = wait_relay_ready(3.0, require_active=True)
        print(
            "[v1-hold] PRE-MIGRATION MIRROR READY: C++ relay active at 500Hz | "
            f"period median/p95/max={relay_status['period_median_ms']:.3f}/"
            f"{relay_status['period_p95_ms']:.3f}/"
            f"{relay_status['period_max_ms']:.3f}ms",
            flush=True,
        )
        if args.mirror_only:
            marker = time.monotonic()
            wait_official_profile(2.0, marker)
            state, status = get_state()
            if state.lower() != "business" or status != aimdk_msg.SystemStatus.IN_READY:
                raise RuntimeError(
                    f"official state changed during mirror-only test: {state}/{status}"
                )
            destroy_hold_publishers()
            print(
                "[v1-hold] PASS: mirror-only overlap completed; no state migration requested",
                flush=True,
            )
            return 0
        lifecycle.migration_requested = True
        migrate("Develop_MC")
        lifecycle.develop_confirmed = True
        after, _, _ = wait_fresh(8.0)
        problem = anchor_problem(after, before)
        if problem:
            raise RuntimeError(problem)
        if motion["fault"] is not None:
            raise RuntimeError(f"migration mirror hold failed: {motion['fault']}")
        feedback_anchor = after
        if args.torque_equivalent_step:
            vibration_monitor.reset()
            print(
                "[v1-hold] EXACT OFFICIAL HOLD: Develop_MC confirmed; "
                "keeping the captured Standing PD unchanged for 1.0s",
                flush=True,
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and motion["fault"] is None:
                rclpy.spin_once(node, timeout_sec=0.002)
            if motion["fault"] is not None:
                raise RuntimeError(f"exact-official hold failed: {motion['fault']}")
            baseline_vibration = vibration_monitor.summary()
            print(
                "[v1-hold] EXACT OFFICIAL VIBRATION: "
                f"joint={baseline_vibration['joint']} "
                f"span={baseline_vibration['position_span_rad']:.4f}rad "
                f"velocity_rms={baseline_vibration['velocity_rms_rad_s']:.3f}rad/s "
                f"acceleration_rms={baseline_vibration['acceleration_rms_rad_s2']:.1f}rad/s^2 "
                f"reversals={baseline_vibration['reversal_hz']:.1f}Hz",
                flush=True,
            )
            vibration_monitor.reset()
            print(
                f"[v1-hold] GAIN RAMP ACTIVE: official Standing -> Sonic over "
                f"{SONIC_GAIN_RAMP_S:.1f}s with instantaneous torque preservation",
                flush=True,
            )
            ramp_start = time.monotonic()
            next_update = ramp_start
            ramp_report = None
            while True:
                now = time.monotonic()
                if motion["fault"] is not None:
                    raise RuntimeError(f"gain ramp failed: {motion['fault']}")
                current = snapshot(0.08)
                if current is None:
                    raise RuntimeError("gain ramp failed: fresh HAL feedback unavailable")
                alpha = min(1.0, (now - ramp_start) / SONIC_GAIN_RAMP_S)
                ramp_profile, ramp_report = blended_torque_equivalent_profile(
                    official_capture,
                    current[0],
                    current[1],
                    contract,
                    alpha,
                )
                apply_gain_ramp_profile(ramp_profile)
                if alpha >= 1.0:
                    break
                next_update += 1.0 / RT_RELAY_COMMAND_HZ
                while time.monotonic() < next_update:
                    rclpy.spin_once(node, timeout_sec=0.002)
            if motion["fault"] is not None:
                raise RuntimeError(f"gain ramp failed: {motion['fault']}")
            try:
                wait_relay_ready(1.0, require_active=True)
            except TimeoutError as exc:
                raise RuntimeError(f"gain ramp failed: {exc}") from exc
            ramp_vibration = vibration_monitor.summary()
            print(
                "[v1-hold] GAIN RAMP COMPLETE: "
                f"alpha={ramp_report['alpha']:.3f} "
                f"max_offset={ramp_report['max_offset_joint']}="
                f"{ramp_report['max_offset_rad']:.4f}rad "
                f"max_residual={ramp_report['max_torque_residual_nm']:.3e}Nm | "
                f"vibration joint={ramp_vibration['joint']} "
                f"span={ramp_vibration['position_span_rad']:.4f}rad "
                f"velocity_rms={ramp_vibration['velocity_rms_rad_s']:.3f}rad/s "
                f"acceleration_rms={ramp_vibration['acceleration_rms_rad_s2']:.1f}rad/s^2 "
                f"reversals={ramp_vibration['reversal_hz']:.1f}Hz",
                flush=True,
            )
        for group, (base, _, _) in GROUPS.items():
            existing_publishers = [
                info.node_name for info in node.get_publishers_info_by_topic(f"{base}/command")
            ]
            existing_subscribers = [
                info.node_name for info in node.get_subscriptions_info_by_topic(f"{base}/command")
            ]
            print(
                f"[v1-hold] command graph {group}: pubs={existing_publishers} "
                f"subs={existing_subscribers}",
                flush=True,
            )
        print(
            "[v1-hold] HOLD ACTIVE: "
            f"gains={'Sonic torque-equivalent' if args.torque_equivalent_step else 'official Standing'}; "
            "pose delta pre/post="
            f"{np.max(np.abs(after[:29] - before[:29])):.4f}rad",
            flush=True,
        )
        warmup_seconds = 1.0 if (args.sonic_seconds > 0.0 or args.step_joint) else args.hold_seconds
        deadline = time.monotonic() + warmup_seconds
        while time.monotonic() < deadline and motion["fault"] is None:
            rclpy.spin_once(node, timeout_sec=0.002)
        if motion["fault"] is not None:
            raise RuntimeError(f"hold preflight failed: {motion['fault']}")
        if args.step_joint:
            step_index = JOINT_NAMES.index(args.step_joint)
            desired = command_anchor.copy()
            desired[step_index] = np.clip(
                command_anchor[step_index] + args.step_rad,
                LOWER_LIMITS[step_index],
                UPPER_LIMITS[step_index],
            )
            if args.torque_equivalent_step and not np.isclose(
                desired[step_index] - command_anchor[step_index],
                args.step_rad,
                atol=1e-9,
            ):
                raise RuntimeError("left-shoulder step would be clipped by a joint limit")
            motion["desired"] = desired
            motion["active"] = True
            motion["last_s"] = time.monotonic()
            print(
                f"[v1-hold] STEP ACTIVE: {args.step_joint} "
                f"delta={desired[step_index] - command_anchor[step_index]:.3f}rad",
                flush=True,
            )
            if motion["fault"] is not None:
                raise RuntimeError(f"joint step failed: {motion['fault']}")
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and motion["fault"] is None:
                rclpy.spin_once(node, timeout_sec=0.002)
            outward = snapshot(0.20)
            outward_delta = (
                float(outward[0][step_index] - feedback_anchor[step_index])
                if outward is not None else float("nan")
            )
            motion["desired"] = command_anchor.copy()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and motion["fault"] is None:
                rclpy.spin_once(node, timeout_sec=0.002)
            motion["active"] = False
            returned = snapshot(0.20)
            return_delta = (
                float(returned[0][step_index] - feedback_anchor[step_index])
                if returned is not None else float("nan")
            )
            print(
                f"[v1-hold] STEP RESULT: measured_out={outward_delta:.4f}rad "
                f"measured_return={return_delta:.4f}rad "
                f"command_peak={motion['command_peak']:.4f}rad "
                f"measured_peak={motion['measured_peak']:.4f}rad "
                f"fault={motion['fault'] or 'none'}",
                flush=True,
            )
            if motion["fault"] is not None:
                raise RuntimeError(f"joint step failed: {motion['fault']}")
        elif args.sonic_seconds > 0.0:
            target_age = time.monotonic() - network["received_s"]
            if network["accepted"] < 10 or target_age > 0.12:
                raise RuntimeError(
                    f"Sonic target stream is not ready: accepted={network['accepted']} "
                    f"age={target_age:.3f}s bad={network['bad']} "
                    f"wrong_source={network['wrong_source']}"
                )
            motion["active"] = True
            motion["last_s"] = time.monotonic()
            print(
                f"[v1-hold] SONIC ACTIVE: arms only for {args.sonic_seconds:.1f}s; "
                "offset<=0.05rad rate<=0.08rad/s",
                flush=True,
            )
            deadline = time.monotonic() + args.sonic_seconds
            while time.monotonic() < deadline and motion["fault"] is None:
                rclpy.spin_once(node, timeout_sec=0.002)
            motion["active"] = False
            print(
                f"[v1-hold] SONIC STOP: accepted={network['accepted']} "
                f"bad={network['bad']} wrong_source={network['wrong_source']} "
                f"command_peak={motion['command_peak']:.4f}rad "
                f"measured_peak={motion['measured_peak']:.4f}rad "
                f"fault={motion['fault'] or 'none'}",
                flush=True,
            )
            if motion["fault"] is not None:
                raise RuntimeError(f"Sonic trial failed: {motion['fault']}")
            end_hold = time.monotonic() + 0.5
            while time.monotonic() < end_hold:
                rclpy.spin_once(node, timeout_sec=0.002)
        safe_handback_to_standing()
        result = "joint step trial" if args.step_joint else (
            "Sonic arms trial" if args.sonic_seconds > 0.0 else "fixed-pose hold"
        )
        print(
            f"[v1-hold] PASS: {result} completed; official Standing handback verified",
            flush=True,
        )
        return 0
    except (EOFError, KeyboardInterrupt):
        print("[v1-hold] interrupted", file=sys.stderr, flush=True)
        if lifecycle.recovery_required() and publishing and rclpy.ok():
            safe_handback_to_standing()
        return 130
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[v1-hold] ERROR: {exc}", file=sys.stderr, flush=True)
        if lifecycle.recovery_required() and publishing and rclpy.ok():
            safe_handback_to_standing()
        return 1
    finally:
        if publishing and lifecycle.recovery_required() and rclpy.ok():
            print(
                "[v1-hold] CRITICAL: automatic handback failed; mirror hold will "
                "continue. Enter official Standing or engage physical E-stop.",
                file=sys.stderr,
                flush=True,
            )
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.01)
        if publishing and lifecycle.may_release_mirror():
            destroy_hold_publishers()
        elif publishing:
            print(
                "[v1-hold] CRITICAL: ROS stopped before Standing handback; "
                "mirror ownership was not explicitly released. Engage physical E-stop.",
                file=sys.stderr,
                flush=True,
            )
        if lifecycle.may_release_mirror():
            control_running = False
            control_thread.join(timeout=1.0)
            node.destroy_node()
            service_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if command_socket is not None:
            command_socket.close()


if __name__ == "__main__":
    raise SystemExit(main())
