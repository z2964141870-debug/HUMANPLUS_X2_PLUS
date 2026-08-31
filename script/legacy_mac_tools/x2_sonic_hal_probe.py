#!/usr/bin/env python3
"""Short, guarded Sonic -> HAL probe for a physically suspended X2."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.expanduser("~/x2_v2_teleop")
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)

from eval_official_sonic_x2 import (  # noqa: E402
    DEFAULT_ANGLES_MJ, JOINT_NAMES, MJ_TO_IL, SonicPolicy,
)
from x2_control_safety import ClosedLoopSafetyFilter, SafetyConfig  # noqa: E402
from x2_hal_backend import HalTeleopBackend  # noqa: E402
from x2_state_feedback import FeedbackReceiver  # noqa: E402


class StandReference:
    fps = 50.0
    frames = 600
    name = "guarded_stand_probe"
    joint_pos = np.tile(DEFAULT_ANGLES_MJ, (frames, 1))
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frames, 1))
    root_pos = np.tile(np.asarray([0.0, 0.0, 0.65]), (frames, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--feedback-port", type=int, default=50041)
    parser.add_argument("--model", default=os.path.join(
        ROOT, "models/x2_sonic_frozen_g1core_lora_v2.onnx"
    ))
    args = parser.parse_args()
    if args.confirm != "X2_SUSPENDED_SONIC_HAL_PROBE":
        raise SystemExit("confirmation mismatch; refusing HAL publication")
    if not 0.1 <= args.duration <= 0.3:
        raise SystemExit("duration must be in [0.1, 0.3] seconds")

    policy = SonicPolicy(args.model, require_cuda=False)
    feedback = FeedbackReceiver("127.0.0.1", args.feedback_port)
    backend = None
    initial = None
    last_measured = None
    max_actual_displacement = 0.0
    max_tracking_error = 0.0
    cycles = 0
    safety = ClosedLoopSafetyFilter(SafetyConfig(
        feedback_timeout_s=0.15,
        max_policy_offset_rad=0.20,
        max_tracking_error_rad=0.08,
        max_target_rate_rad_s=0.20,
        control_hz=50.0,
    ))
    try:
        deadline = time.monotonic() + 2.0
        state = None
        while time.monotonic() < deadline:
            feedback.poll(max_packets=4096)
            state = feedback.snapshot(0.15)
            if state is not None:
                break
            time.sleep(0.01)
        if state is None:
            raise RuntimeError("fresh real-state feedback unavailable")
        initial = state.qpos[7:38].astype(np.float64).copy()
        last_measured = initial.copy()
        safety.reset(initial)
        backend = HalTeleopBackend(
            dry_run=False, enable_network=True,
            publish_hz=500.0, policy_hz=50.0, watchdog_s=0.10,
        )
        print(
            f"SONIC_HAL_PROBE_ARMED duration={args.duration:.3f}s "
            "rate_limit=0.20rad/s head=omitted"
        )
        start = time.monotonic()
        next_tick = start
        while time.monotonic() - start < args.duration:
            # DDS discovery during backend initialization can leave more than
            # the receiver's default 64 UDP packets queued.  Drain through to
            # the newest producer timestamp before applying the freshness gate.
            feedback.poll(max_packets=4096)
            state = feedback.snapshot(0.15)
            if state is None:
                raise RuntimeError("feedback_missing_or_stale")
            measured = state.qpos[7:38].astype(np.float64)
            actual_displacement = float(np.max(np.abs(measured[:29] - initial[:29])))
            max_actual_displacement = max(max_actual_displacement, actual_displacement)
            if actual_displacement > 0.10:
                raise RuntimeError(
                    f"actual_displacement_limit: {actual_displacement:.5f}rad"
                )

            _, action = policy.infer(
                StandReference(), cycles / StandReference.fps,
                state.qpos, state.qvel,
            )
            for name in (
                "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
                "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
            ):
                action[MJ_TO_IL[JOINT_NAMES.index(name)]] = 0.0
            targets = policy.action_to_targets(
                action, StandReference(), 0, wrist_ref=False,
            )
            targets[29:31] = measured[29:31]
            decision = safety.filter(targets, state, enforce_tracking=True)
            if not decision.accepted:
                raise RuntimeError(decision.reason)
            tracking_error = float(np.max(np.abs(decision.targets[:29] - measured[:29])))
            max_tracking_error = max(max_tracking_error, tracking_error)
            backend.send(decision.targets)
            last_measured = measured
            cycles += 1
            next_tick += 0.02
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        print(
            f"SONIC_HAL_PROBE_PASS cycles={cycles} "
            f"max_actual_displacement={max_actual_displacement:.5f}rad "
            f"max_tracking_error={max_tracking_error:.5f}rad"
        )
    finally:
        if backend is not None:
            backend.stop()
            backend.close()
        feedback.close()
        if initial is not None and last_measured is not None:
            print("SONIC_HAL_PROBE_STOPPED")


if __name__ == "__main__":
    main()
