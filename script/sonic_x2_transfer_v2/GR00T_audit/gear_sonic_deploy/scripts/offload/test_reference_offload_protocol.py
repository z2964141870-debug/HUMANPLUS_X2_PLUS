from __future__ import annotations

import socket
import threading
import time
import unittest

import numpy as np

from reference_offload_protocol import (
    ACK,
    FRAME,
    RESET,
    RESULT,
    ProtocolError,
    encode_message,
    pack_frame,
    pack_result,
    recv_message,
    send_message,
    unpack_frame,
    unpack_result,
)
from reference_offload_client import OneFrameReferencePipeline, RemoteReferenceSession


class ReferenceOffloadProtocolTest(unittest.TestCase):
    def test_message_and_frame_round_trip(self):
        pose = np.arange(72, dtype=np.float32).reshape(24, 3) / 100.0
        velocity = -pose
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        left.sendall(
            encode_message(
                FRAME,
                epoch=7,
                sequence=12,
                source_monotonic_ns=34,
                payload=pack_frame(pose, velocity),
            )
        )
        message = recv_message(right)
        actual_pose, actual_velocity = unpack_frame(message.payload)
        self.assertEqual(message.epoch, 7)
        self.assertEqual(message.sequence, 12)
        np.testing.assert_array_equal(actual_pose, pose)
        np.testing.assert_array_equal(actual_velocity, velocity)

    def test_result_round_trip(self):
        expected = np.arange(36, dtype=np.float64) / 10.0
        actual, processing_ns = unpack_result(pack_result(expected, 123456))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(processing_ns, 123456)

    def test_crc_mismatch_is_rejected(self):
        encoded = bytearray(
            encode_message(FRAME, epoch=1, sequence=0, payload=bytes(72 * 2 * 4))
        )
        encoded[-1] ^= 0xFF
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        left.sendall(encoded)
        with self.assertRaises(ProtocolError):
            recv_message(right)

    def test_startup_timeout_only_applies_to_configured_frames(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        server_errors = []

        def serve():
            try:
                conn, _ = listener.accept()
                with conn:
                    reset = recv_message(conn)
                    self.assertEqual(reset.message_type, RESET)
                    send_message(
                        conn,
                        ACK,
                        epoch=reset.epoch,
                        sequence=reset.sequence,
                        source_monotonic_ns=reset.source_monotonic_ns,
                    )
                    for delay in (0.08, 0.08, 0.08):
                        frame = recv_message(conn)
                        self.assertEqual(frame.message_type, FRAME)
                        time.sleep(delay)
                        try:
                            send_message(
                                conn,
                                RESULT,
                                epoch=frame.epoch,
                                sequence=frame.sequence,
                                source_monotonic_ns=frame.source_monotonic_ns,
                                payload=pack_result(np.zeros(36), int(delay * 1e9)),
                            )
                        except OSError:
                            break
            except BaseException as error:
                server_errors.append(error)

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        session = RemoteReferenceSession(
            "127.0.0.1",
            port,
            frame_timeout_s=0.02,
            reset_timeout_s=0.5,
            epoch=11,
            startup_frame_timeout_s=0.25,
            startup_frame_count=2,
        )
        self.addCleanup(session.close)
        pose = np.zeros((24, 3), dtype=np.float32)
        velocity = np.zeros((24, 3), dtype=np.float32)
        qpos, _, _ = session.process(pose, velocity)
        np.testing.assert_array_equal(qpos, np.zeros(36))
        qpos, _, _ = session.process(pose, velocity)
        np.testing.assert_array_equal(qpos, np.zeros(36))
        with self.assertRaises(socket.timeout):
            session.process(pose, velocity)
        self.assertIsNone(session.socket)
        server.join(timeout=0.5)
        self.assertFalse(server.is_alive())
        if server_errors:
            raise server_errors[0]


class FakeReferenceSession:
    def __init__(
        self, *, fail_value: int | None = None, block_value: int | None = None
    ):
        self.fail_value = fail_value
        self.block_value = block_value
        self.values: list[int] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def process(self, pose, velocity):
        value = int(np.asarray(pose)[0, 0])
        self.values.append(value)
        self.started.set()
        if value == self.block_value:
            self.release.wait(timeout=1.0)
        if value == self.fail_value:
            raise TimeoutError(f"synthetic timeout for {value}")
        return np.full(36, value, dtype=np.float64), 1.0, 2.0

    def close(self):
        self.closed = True


class OneFrameReferencePipelineTest(unittest.TestCase):
    @staticmethod
    def frame(value: int):
        pose = np.full((24, 3), value, dtype=np.float32)
        velocity = np.full((24, 3), -value, dtype=np.float32)
        return pose, velocity

    def test_one_frame_delay_preserves_order(self):
        session = FakeReferenceSession()
        pipeline = OneFrameReferencePipeline(session)
        self.addCleanup(pipeline.close)

        self.assertIsNone(pipeline.exchange(*self.frame(0)))
        first = pipeline.exchange(*self.frame(1))
        second = pipeline.flush()

        np.testing.assert_array_equal(first[0], np.zeros(36))
        np.testing.assert_array_equal(second[0], np.ones(36))
        self.assertEqual(session.values, [0, 1])
        self.assertEqual(
            pipeline.snapshot(),
            {"submitted": 2, "completed": 2, "discarded": 0, "pending": False},
        )

    def test_hold_discards_old_result_before_resume(self):
        session = FakeReferenceSession()
        pipeline = OneFrameReferencePipeline(session)
        self.addCleanup(pipeline.close)

        self.assertIsNone(pipeline.exchange(*self.frame(0)))
        self.assertTrue(session.started.wait(timeout=0.5))
        while pipeline.snapshot()["completed"] == 0:
            time.sleep(0.001)
        pipeline.hold()

        self.assertIsNone(pipeline.exchange(*self.frame(1)))
        resumed = pipeline.flush()
        np.testing.assert_array_equal(resumed[0], np.ones(36))
        self.assertEqual(pipeline.snapshot()["discarded"], 1)

    def test_hold_marks_an_inflight_result_for_discard(self):
        session = FakeReferenceSession(block_value=0)
        pipeline = OneFrameReferencePipeline(session)
        self.addCleanup(pipeline.close)

        self.assertIsNone(pipeline.exchange(*self.frame(0)))
        self.assertTrue(session.started.wait(timeout=0.5))
        pipeline.hold()
        session.release.set()

        self.assertIsNone(pipeline.exchange(*self.frame(1)))
        resumed = pipeline.flush()
        np.testing.assert_array_equal(resumed[0], np.ones(36))
        self.assertEqual(pipeline.snapshot()["discarded"], 1)

    def test_failure_is_raised_before_another_frame_is_submitted(self):
        session = FakeReferenceSession(fail_value=1)
        pipeline = OneFrameReferencePipeline(session)
        self.addCleanup(pipeline.close)

        self.assertIsNone(pipeline.exchange(*self.frame(0)))
        first = pipeline.exchange(*self.frame(1))
        np.testing.assert_array_equal(first[0], np.zeros(36))
        with self.assertRaises(TimeoutError):
            pipeline.exchange(*self.frame(2))
        self.assertEqual(session.values, [0, 1])
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
