#!/usr/bin/env python3
"""Offline tests for the fail-closed live-reference proxy."""

from __future__ import annotations

import math
import unittest

import numpy as np

import gate_live_reference as gate
import publish_v51_reference as v51


def pose_frame(
    index: int,
    now_s: float,
    *,
    joint_pos: np.ndarray | None = None,
    joint_vel: np.ndarray | None = None,
    yaw_deg: float = 0.0,
) -> gate.PoseFrame:
    return gate.PoseFrame(
        joint_pos=(v51.DEFAULT_ANGLES.copy() if joint_pos is None else joint_pos.copy()),
        joint_vel=(np.zeros(v51.NUM_DOFS) if joint_vel is None else joint_vel.copy()),
        root_quat_xyzw=gate.yaw_quat_xyzw(math.radians(yaw_deg)),
        frame_index=index,
        recv_s=now_s,
    )


def robot_frame(tick: int, now_s: float, yaw_deg: float = 30.0) -> gate.RobotFrame:
    xyzw = gate.yaw_quat_xyzw(math.radians(yaw_deg))
    return gate.RobotFrame(
        base_quat_wxyz=xyzw[[3, 0, 1, 2]],
        control_tick=tick,
        ros_timestamp_s=now_s + 1000.0,
        dry_run=False,
        recv_s=now_s,
    )


class GateStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = gate.GateConfig(
            source_stale_s=0.5,
            robot_stale_s=0.5,
            still_seconds=0.20,
            still_frames=3,
            blend_seconds=1.0,
        )

    def warm_to_ready(
        self,
        machine: gate.LiveReferenceGate,
        *,
        source_yaw_deg: float = -80.0,
        joint_pos: np.ndarray | None = None,
    ) -> float:
        now = 0.0
        for index in range(5):
            now = index * 0.1
            machine.on_robot(robot_frame(index + 1, now), now)
            machine.on_source(
                pose_frame(index, now, joint_pos=joint_pos, yaw_deg=source_yaw_deg), now
            )
        self.assertEqual(machine.state, gate.GateState.STANDSTILL_READY)
        return now

    def test_tpose_is_rejected_even_when_stationary(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        tpose = v51.DEFAULT_ANGLES.copy()
        tpose[16] += 1.0
        tpose[23] -= 1.0
        for index in range(8):
            now = index * 0.1
            machine.on_robot(robot_frame(index + 1, now), now)
            self.assertIsNone(machine.on_source(pose_frame(index, now, joint_pos=tpose), now))
        self.assertEqual(machine.state, gate.GateState.WARMUP)
        self.assertIn("arm offset", machine.reason)

    def test_motion_resets_continuous_stillness(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        moving = np.zeros(v51.NUM_DOFS)
        moving[0] = 0.4
        for index in range(3):
            now = index * 0.1
            machine.on_robot(robot_frame(index + 1, now), now)
            machine.on_source(pose_frame(index, now, joint_vel=moving), now)
        self.assertEqual(machine.state, gate.GateState.WARMUP)
        for index in range(3, 7):
            now = index * 0.1
            machine.on_robot(robot_frame(index + 1, now), now)
            machine.on_source(pose_frame(index, now), now)
        self.assertEqual(machine.state, gate.GateState.STANDSTILL_READY)

    def test_ready_publishes_exact_anchored_standstill(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        now = self.warm_to_ready(machine)
        output = machine.reference_at(now)
        np.testing.assert_allclose(output.joint_pos, v51.DEFAULT_ANGLES, atol=0.0)
        np.testing.assert_allclose(output.joint_vel, 0.0, atol=0.0)
        self.assertAlmostEqual(math.degrees(gate.yaw_from_xyzw(output.root_quat_xyzw)), 30.0)

    def test_yaw_rebase_and_blend_endpoints(self) -> None:
        live_pos = v51.DEFAULT_ANGLES.copy()
        live_pos[15] += 0.20
        live_vel = np.zeros(v51.NUM_DOFS)
        live_vel[15] = 0.05
        machine = gate.LiveReferenceGate(self.config)
        now = self.warm_to_ready(machine, source_yaw_deg=-80.0, joint_pos=live_pos)
        machine.source = pose_frame(99, now, joint_pos=live_pos, joint_vel=live_vel, yaw_deg=-80.0)

        accepted, _ = machine.arm(now)
        self.assertTrue(accepted)
        start = machine.reference_at(now)
        np.testing.assert_allclose(start.joint_pos, v51.DEFAULT_ANGLES, atol=1e-12)
        np.testing.assert_allclose(start.joint_vel, 0.0, atol=1e-12)
        self.assertAlmostEqual(math.degrees(gate.yaw_from_xyzw(start.root_quat_xyzw)), 30.0)

        end = machine.reference_at(now + self.config.blend_seconds)
        np.testing.assert_allclose(end.joint_pos, live_pos, atol=1e-12)
        np.testing.assert_allclose(end.joint_vel, live_vel, atol=1e-12)
        self.assertAlmostEqual(math.degrees(gate.yaw_from_xyzw(end.root_quat_xyzw)), 30.0)
        self.assertEqual(machine.state, gate.GateState.LIVE)

    def test_stale_stream_is_terminal_lockout(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        now = self.warm_to_ready(machine)
        machine.tick(now + self.config.source_stale_s + 0.01)
        self.assertEqual(machine.state, gate.GateState.LOCKOUT)
        self.assertIn("stale", machine.reason)

        later = now + 1.0
        machine.on_robot(robot_frame(100, later), later)
        output = machine.on_source(pose_frame(100, later), later)
        self.assertIsNone(output)
        self.assertEqual(machine.state, gate.GateState.LOCKOUT)
        accepted, _ = machine.arm(later)
        self.assertFalse(accepted)

    def test_source_event_cannot_emit_after_robot_is_stale(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        now = self.warm_to_ready(machine)
        later = now + self.config.robot_stale_s + 0.01
        output = machine.on_source(pose_frame(100, later), later)
        self.assertIsNone(output)
        self.assertEqual(machine.state, gate.GateState.LOCKOUT)
        self.assertIn("x2_debug stale", machine.reason)

    def test_static_debug_may_hold_control_tick_if_timestamp_advances(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        first = robot_frame(0, 0.0)
        second = robot_frame(0, 0.01)
        machine.on_robot(first, 0.0)
        machine.on_robot(second, 0.01)
        self.assertNotEqual(machine.state, gate.GateState.LOCKOUT)

    def test_motion_after_ready_locks_out_stationary_probe(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        now = self.warm_to_ready(machine)
        moving = np.zeros(v51.NUM_DOFS)
        moving[18] = 0.5
        later = now + 0.05
        machine.on_robot(robot_frame(20, later), later)
        output = machine.on_source(pose_frame(20, later, joint_vel=moving), later)
        self.assertIsNone(output)
        self.assertEqual(machine.state, gate.GateState.LOCKOUT)
        self.assertIn("stationary reference gate", machine.reason)

    def test_dry_run_debug_cannot_ready_powered_default(self) -> None:
        machine = gate.LiveReferenceGate(self.config)
        for index in range(5):
            now = index * 0.1
            robot = robot_frame(index + 1, now)
            machine.on_robot(
                gate.RobotFrame(
                    robot.base_quat_wxyz,
                    robot.control_tick,
                    robot.ros_timestamp_s,
                    True,
                    now,
                ),
                now,
            )
            machine.on_source(pose_frame(index, now), now)
        self.assertEqual(machine.state, gate.GateState.WARMUP)
        self.assertIn("dry_run=1", machine.reason)


class WireTest(unittest.TestCase):
    @staticmethod
    def debug_message(tick: int = 4, ros_timestamp_s: float = 123.5) -> bytes:
        fields = [
            ("control_tick", np.array([tick], dtype=np.int64)),
            ("ros_timestamp", np.array([ros_timestamp_s], dtype=np.float64)),
            ("policy_time", np.array([0.0], dtype=np.float64)),
            ("base_quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
            ("base_ang_vel", np.zeros(3, dtype=np.float64)),
            ("body_q", np.zeros(v51.NUM_DOFS, dtype=np.float64)),
            ("body_dq", np.zeros(v51.NUM_DOFS, dtype=np.float64)),
            ("last_action", np.zeros(v51.NUM_DOFS, dtype=np.float64)),
            ("left_hand_q", np.zeros(10, dtype=np.float64)),
            ("right_hand_q", np.zeros(10, dtype=np.float64)),
            ("hand_frame_idx", np.array([-1], dtype=np.int64)),
            ("ramp_alpha", np.array([0.0], dtype=np.float64)),
            ("tilt_trip", np.array([0], dtype=np.uint8)),
            ("dry_run", np.array([0], dtype=np.uint8)),
        ]
        dtype_names = {
            np.dtype(np.float64): "f64",
            np.dtype(np.int64): "i64",
            np.dtype(np.uint8): "u8",
        }
        descriptors = []
        payload = []
        for name, value in fields:
            descriptors.append({
                "name": name,
                "dtype": dtype_names[value.dtype],
                "shape": list(value.shape),
            })
            payload.append(np.ascontiguousarray(value).tobytes())
        return b"x2_debug" + v51.build_header(descriptors, version=5) + b"".join(payload)

    def test_robot_debug_decode(self) -> None:
        decoded = gate.decode_robot_frame(self.debug_message(), "x2_debug", 8.0)
        self.assertEqual(decoded.control_tick, 4)
        self.assertEqual(decoded.ros_timestamp_s, 123.5)
        self.assertFalse(decoded.dry_run)
        np.testing.assert_allclose(decoded.base_quat_wxyz, [1.0, 0.0, 0.0, 0.0])

    def test_pose_decode_and_guarded_repack(self) -> None:
        stream = v51.V51ReferenceStream(velocity_mode="timestamp")
        stream.push(v51.DEFAULT_ANGLES, gate.yaw_quat_xyzw(math.radians(-20.0)), timestamp_s=1.0)
        raw = v51.pack_message(stream.build_fields(7), topic="pose", version=5)
        decoded = gate.decode_pose_frame(raw, "pose", 2.0)
        self.assertEqual(decoded.frame_index, 7)
        output = gate.OutputReference(
            decoded.joint_pos,
            decoded.joint_vel,
            decoded.root_quat_xyzw,
            11,
            1.0,
        )
        roundtrip = gate.decode_pose_frame(gate.build_output_message(output, "pose"), "pose", 3.0)
        self.assertEqual(roundtrip.frame_index, 11)
        np.testing.assert_allclose(roundtrip.joint_pos, decoded.joint_pos, atol=2e-6)

    def test_future_window_mismatch_is_rejected(self) -> None:
        stream = v51.V51ReferenceStream()
        stream.push(v51.DEFAULT_ANGLES, v51.IDENTITY_QUAT_XYZW)
        fields = stream.build_fields(0)
        modified = []
        for name, value in fields:
            value = value.copy()
            if name == "joint_pos_mj_future":
                value[3, 0] += 0.1
            modified.append((name, value))
        raw = v51.pack_message(modified, topic="pose", version=5)
        with self.assertRaisesRegex(gate.WireError, "live-edge clamp"):
            gate.decode_pose_frame(raw, "pose", 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
