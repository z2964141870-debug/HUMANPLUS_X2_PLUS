#!/usr/bin/env python3
"""Pure-Python tests for the HMCP -> X2 -> ZMQ v5.1 adapter."""

from __future__ import annotations

import json
import socket
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import publish_v51_reference as pub
import summarize_live_dryrun as summary
import verify_v51_bytes as verify
from hmcp_protocol_compat.deploy.onboard_deploy_wo_GMR import protocol


def neutral_qpos() -> np.ndarray:
    qpos = np.zeros(pub.G1_QPOS_N, dtype=np.float32)
    qpos[:3] = [0.1, -0.2, 0.8]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:] = pub.G1_DEFAULT_ANGLES.astype(np.float32)
    return qpos


def hmcp_bytes(
    sequence: int,
    qpos: np.ndarray,
    flags: int = 0,
    send_timestamp_s: float = 0.0,
) -> bytes:
    extras = []
    if flags & pub.HMCP_FLAG_HAS_HAND:
        extras.extend([0.0] * 4)
    if flags & pub.HMCP_FLAG_HAS_BRAINCO:
        extras.extend([0.0] * 24)
    header = pub.HMCP_HEADER.pack(
        pub.HMCP_MAGIC,
        pub.HMCP_VERSION,
        flags,
        sequence & 0xFFFFFFFF,
        struct.pack("<d", send_timestamp_s),
        pub.G1_QPOS_N,
    )
    payload = np.concatenate([
        np.asarray(qpos, dtype="<f4"),
        np.asarray(extras, dtype="<f4"),
    ])
    return header + payload.tobytes()


class HmcpAdapterTest(unittest.TestCase):
    def test_compat_encoder_round_trips_through_strict_parser(self) -> None:
        qpos = neutral_qpos()
        raw = protocol.encode_frame(seq=0xFFFFFFFF, send_ts=123.5, qpos=qpos)
        packet = pub.parse_hmcp(raw)
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.sequence, 0xFFFFFFFF)
        self.assertEqual(packet.flags, 0)
        self.assertEqual(packet.send_timestamp_s, 123.5)
        np.testing.assert_array_equal(packet.qpos36, qpos)

    def test_strict_parser_accepts_known_optional_tails(self) -> None:
        qpos = neutral_qpos()
        for flags in (
            0,
            pub.HMCP_FLAG_HAS_HAND,
            pub.HMCP_FLAG_HAS_BRAINCO,
            pub.HMCP_KNOWN_FLAGS,
        ):
            packet = pub.parse_hmcp(hmcp_bytes(17, qpos, flags))
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.sequence, 17)
            self.assertEqual(packet.flags, flags)
            self.assertIsNone(packet.send_timestamp_s)
            np.testing.assert_allclose(packet.qpos36, qpos, atol=0.0)

    def test_strict_parser_rejects_malformed_packets(self) -> None:
        good = bytearray(hmcp_bytes(3, neutral_qpos()))
        cases = []
        bad_magic = good.copy()
        bad_magic[:4] = b"NOPE"
        cases.append(bytes(bad_magic))
        bad_version = good.copy()
        bad_version[4] = 2
        cases.append(bytes(bad_version))
        bad_flags = good.copy()
        bad_flags[5] = 0x80
        cases.append(bytes(bad_flags))
        bad_count = good.copy()
        bad_count[18:20] = (35).to_bytes(2, "little")
        cases.append(bytes(bad_count))
        cases.append(bytes(good[:-1]))
        nonfinite = neutral_qpos()
        nonfinite[7] = np.nan
        cases.append(hmcp_bytes(3, nonfinite))
        for raw in cases:
            self.assertIsNone(pub.parse_hmcp(raw))

    def test_named_joint_mapping_pose_scale_and_quaternion_order(self) -> None:
        qpos = neutral_qpos().astype(np.float64)
        qpos[3:7] = [2**-0.5, 0.0, 0.0, 2**-0.5]
        qpos[7 + pub.G1_JOINT_NAMES.index("waist_roll_joint")] += 0.2
        qpos[7 + pub.G1_JOINT_NAMES.index("left_wrist_roll_joint")] += 0.3

        joints, quat_xyzw, root_pos = pub.hmcp_to_x2_reference(qpos, 0.7)
        expected = pub.DEFAULT_ANGLES.copy()
        expected[pub.MUJOCO_JOINT_NAMES.index("waist_roll_joint")] += 0.14
        expected[pub.MUJOCO_JOINT_NAMES.index("left_wrist_roll_joint")] += 0.21
        np.testing.assert_allclose(joints, expected, atol=1e-12)
        np.testing.assert_allclose(
            quat_xyzw, [0.0, 0.0, 2**-0.5, 2**-0.5], atol=1e-12
        )
        np.testing.assert_allclose(root_pos, [0.1, -0.2, 0.8], atol=1e-6)
        self.assertEqual(joints[pub.MUJOCO_JOINT_NAMES.index("head_yaw_joint")], 0.0)
        self.assertEqual(joints[pub.MUJOCO_JOINT_NAMES.index("head_pitch_joint")], 0.0)

    def test_neutral_g1_maps_to_x2_training_default(self) -> None:
        joints, quat, _ = pub.hmcp_to_x2_reference(neutral_qpos(), 0.7)
        np.testing.assert_allclose(joints, pub.DEFAULT_ANGLES, atol=2e-8)
        np.testing.assert_allclose(quat, pub.IDENTITY_QUAT_XYZW, atol=0.0)

    def test_sequence_wraparound(self) -> None:
        self.assertTrue(pub.is_newer_sequence(1, None))
        self.assertTrue(pub.is_newer_sequence(11, 10))
        self.assertFalse(pub.is_newer_sequence(10, 10))
        self.assertFalse(pub.is_newer_sequence(9, 10))
        self.assertTrue(pub.is_newer_sequence(0, 0xFFFFFFFF))

    def test_hmcp_frame_is_accepted_by_strict_v51_decoder(self) -> None:
        q0 = neutral_qpos().astype(np.float64)
        q1 = q0.copy()
        g1_index = pub.G1_JOINT_NAMES.index("left_shoulder_pitch_joint")
        q1[7 + g1_index] += 0.1
        stream = pub.V51ReferenceStream()
        for frame_index, qpos in enumerate((q0, q1)):
            joints, quat, _ = pub.hmcp_to_x2_reference(qpos, 0.7)
            stream.push(joints, quat)
            verdict = verify.evaluate_frame(
                *verify.decode_packed(pub.pack_message(stream.build_fields(frame_index))),
                strict=True,
            )
            self.assertTrue(verdict.accepted, verdict.reject_reason)
        expected_velocity = 0.1 * 0.7 * pub.VELOCITY_DIFF_HZ
        mj_index = pub.MUJOCO_JOINT_NAMES.index("left_shoulder_pitch_joint")
        assert verdict.joint_vel is not None
        self.assertAlmostEqual(verdict.joint_vel[mj_index], expected_velocity, places=5)
        assert verdict.window_vel is not None
        for slot_velocity in verdict.window_vel:
            self.assertAlmostEqual(slot_velocity[mj_index], expected_velocity, places=5)

    def test_timestamp_velocity_matches_fixed_mode_at_20ms(self) -> None:
        p0 = pub.DEFAULT_ANGLES.copy()
        p1 = p0.copy()
        p1[3] += 0.04
        fixed = pub.V51ReferenceStream(velocity_mode="fixed")
        timestamp = pub.V51ReferenceStream(velocity_mode="timestamp")
        for stream in (fixed, timestamp):
            stream.push(p0, pub.IDENTITY_QUAT_XYZW, timestamp_s=100.0)
            stream.push(p1, pub.IDENTITY_QUAT_XYZW, timestamp_s=100.02)
        fixed_velocity = dict(fixed.build_fields(1))["joint_vel_mj"]
        timestamp_velocity = dict(timestamp.build_fields(1))["joint_vel_mj"]
        np.testing.assert_allclose(timestamp_velocity, fixed_velocity, atol=1e-5)

    def test_timestamp_velocity_uses_variable_dt(self) -> None:
        p0 = pub.DEFAULT_ANGLES.copy()
        p1 = p0.copy()
        p1[4] += 0.08
        stream = pub.V51ReferenceStream(velocity_mode="timestamp")
        stream.push(p0, pub.IDENTITY_QUAT_XYZW, timestamp_s=200.0)
        stream.push(p1, pub.IDENTITY_QUAT_XYZW, timestamp_s=200.04)
        velocity = dict(stream.build_fields(1))["joint_vel_mj"]
        self.assertAlmostEqual(float(velocity[4]), 2.0, places=4)

    def test_timestamp_velocity_resets_after_bad_or_long_gaps(self) -> None:
        stream = pub.V51ReferenceStream(velocity_mode="timestamp")
        positions = [pub.DEFAULT_ANGLES.copy() for _ in range(5)]
        for index, position in enumerate(positions):
            position[5] += 0.01 * index

        stream.push(positions[0], pub.IDENTITY_QUAT_XYZW, timestamp_s=10.0)
        stream.push(positions[1], pub.IDENTITY_QUAT_XYZW, timestamp_s=10.04)
        self.assertGreater(float(dict(stream.build_fields(1))["joint_vel_mj"][5]), 0.0)

        stream.push(positions[2], pub.IDENTITY_QUAT_XYZW, timestamp_s=10.40)
        np.testing.assert_array_equal(
            dict(stream.build_fields(2))["joint_vel_mj"],
            np.zeros(pub.NUM_DOFS, dtype=np.float32),
        )
        stream.push(positions[3], pub.IDENTITY_QUAT_XYZW, timestamp_s=10.44)
        self.assertAlmostEqual(
            float(dict(stream.build_fields(3))["joint_vel_mj"][5]), 0.25, places=4
        )

        stream.push(positions[4], pub.IDENTITY_QUAT_XYZW, timestamp_s=10.43)
        np.testing.assert_array_equal(
            dict(stream.build_fields(4))["joint_vel_mj"],
            np.zeros(pub.NUM_DOFS, dtype=np.float32),
        )
        self.assertIn("derivative_resets=2", stream.velocity_status())

    def test_capture_and_shadow_jsonl_replay_skip_duplicate_sequences(self) -> None:
        qpos = neutral_qpos()
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture.jsonl"
            records = [
                {"type": "metadata", "schema": pub.HMCP_CAPTURE_SCHEMA},
                {
                    "schema": pub.HMCP_CAPTURE_SCHEMA,
                    "sequence": 7,
                    "received_monotonic_ns": 1_000_000_000,
                    "qpos36": qpos.tolist(),
                },
                {
                    "schema": pub.HMCP_CAPTURE_SCHEMA,
                    "sequence": 7,
                    "received_monotonic_ns": 1_010_000_000,
                    "qpos36": qpos.tolist(),
                },
                {
                    "schema": pub.HMCP_CAPTURE_SCHEMA,
                    "sequence": 8,
                    "velocity_timestamp_s": 1.04,
                    "qpos36": qpos.tolist(),
                },
            ]
            capture.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            capture_replay = list(
                pub.jsonl_source(str(capture), 0.7, include_timestamp=True)
            )
            self.assertEqual(len(capture_replay), 2)
            self.assertEqual(capture_replay[0][2], 1.0)
            self.assertEqual(capture_replay[1][2], 1.04)

            shadow = Path(tmp) / "shadow.jsonl"
            shadow_records = [
                {"type": "metadata", "version": 1},
                {
                    "type": "sample",
                    "garment_sequence": 10,
                    "reference_position": pub.DEFAULT_ANGLES.tolist(),
                    "reference_root_quaternion": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "type": "sample",
                    "garment_sequence": 10,
                    "reference_position": pub.DEFAULT_ANGLES.tolist(),
                    "reference_root_quaternion": [1.0, 0.0, 0.0, 0.0],
                },
            ]
            shadow.write_text(
                "".join(json.dumps(record) + "\n" for record in shadow_records),
                encoding="utf-8",
            )
            replay = list(pub.jsonl_source(str(shadow), 0.7, include_timestamp=True))
            self.assertEqual(len(replay), 1)
            np.testing.assert_allclose(replay[0][1], pub.IDENTITY_QUAT_XYZW)
            self.assertIsNone(replay[0][2])

    def test_udp_source_is_event_driven_and_captures_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "live.jsonl"
            source = pub.HmcpUdpSource("127.0.0.1", 0, 0.7, 0.05, str(capture))
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                endpoint = source.sock.getsockname()
                raw = hmcp_bytes(42, neutral_qpos(), send_timestamp_s=123.5)
                sender.sendto(raw, endpoint)
                self.assertIsNotNone(source.poll())
                sender.sendto(raw, endpoint)
                self.assertIsNone(source.poll())
                self.assertEqual(source.received, 2)
                self.assertEqual(source.accepted, 1)
                self.assertEqual(source.duplicate_or_old, 1)
                self.assertEqual(source.hmcp_send_timestamps, 1)
                self.assertEqual(source.receive_timestamp_fallbacks, 0)
            finally:
                sender.close()
                source.close()
            records = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["sequence"], 42)
            self.assertEqual(records[1]["send_timestamp_s"], 123.5)
            self.assertEqual(records[1]["velocity_timestamp_s"], 123.5)
            self.assertEqual(records[1]["velocity_timestamp_source"], "hmcp-send")

    def test_live_summary_reports_timestamp_and_action_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "capture.jsonl"
            rows = [{"type": "metadata", "schema": pub.HMCP_CAPTURE_SCHEMA}]
            for sequence, timestamp_s in enumerate((100.0, 100.04, 100.08)):
                qpos = neutral_qpos()
                qpos[7] += 0.04 * sequence
                rows.append({
                    "schema": pub.HMCP_CAPTURE_SCHEMA,
                    "sequence": sequence,
                    "send_timestamp_s": timestamp_s,
                    "received_monotonic_ns": int((10.0 + 0.04 * sequence) * 1e9),
                    "velocity_timestamp_s": timestamp_s,
                    "velocity_timestamp_source": "hmcp-send",
                    "qpos36": qpos.tolist(),
                })
            capture.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            deploy_log = Path(tmp) / "deploy.log"
            deploy_log.write_text(
                "CONTROL tick=50 policy_t=1.0s act_clip_ticks=3 max_pre_clip=20.5\n"
                "CONTROL tick=100 policy_t=2.0s act_clip_ticks=9 max_pre_clip=21.7\n",
                encoding="utf-8",
            )

            capture_summary = summary.summarize_capture(capture)
            policy_summary = summary.summarize_deploy_log(deploy_log)
            self.assertEqual(capture_summary["frames"], 3)
            self.assertEqual(capture_summary["derivative_resets"], 0)
            self.assertAlmostEqual(float(capture_summary["fixed_ratio_median"]), 2.0)
            self.assertEqual(capture_summary["timestamp_sources"]["hmcp-send"], 3)
            self.assertEqual(policy_summary["control_ticks"], 100)
            self.assertEqual(policy_summary["clipped_ticks"], 9)
            self.assertEqual(policy_summary["max_pre_clip"], 21.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
