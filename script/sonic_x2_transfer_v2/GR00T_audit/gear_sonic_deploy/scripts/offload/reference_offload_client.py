"""Guarded synchronous and one-frame-pipelined 3588S reference clients."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from reference_offload_protocol import (
    ACK,
    ERROR,
    FRAME,
    RESET,
    RESULT,
    ProtocolError,
    pack_frame,
    recv_message,
    send_message,
    unpack_result,
)


class RemoteReferenceSession:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        frame_timeout_s: float,
        reset_timeout_s: float,
        epoch: int,
        startup_frame_timeout_s: float | None = None,
        startup_frame_count: int = 1,
    ):
        self.host = host
        self.port = port
        self.frame_timeout_s = frame_timeout_s
        self.startup_frame_timeout_s = (
            frame_timeout_s
            if startup_frame_timeout_s is None
            else startup_frame_timeout_s
        )
        if startup_frame_count < 0:
            raise ValueError("startup_frame_count must be non-negative")
        self.startup_frame_count = startup_frame_count
        self.reset_timeout_s = reset_timeout_s
        self.epoch = epoch
        self.sequence = 0
        self.socket: socket.socket | None = None
        self.connect_and_reset()

    def connect_and_reset(self) -> None:
        self.close()
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.reset_timeout_s
        )
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.reset_timeout_s)
        self.socket = sock
        source_ns = time.monotonic_ns()
        try:
            send_message(
                sock,
                RESET,
                epoch=self.epoch,
                sequence=0,
                source_monotonic_ns=source_ns,
            )
            reply = recv_message(sock)
            self._validate_reply(reply, ACK, 0, source_ns)
        except Exception:
            self.close()
            raise
        sock.settimeout(self.frame_timeout_s)
        self.sequence = 0

    def process(
        self, pose: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        if self.socket is None:
            raise RuntimeError("remote reference session is closed")
        sequence = self.sequence
        source_ns = time.monotonic_ns()
        start_ns = time.perf_counter_ns()
        startup_frame = sequence < self.startup_frame_count
        if startup_frame:
            self.socket.settimeout(self.startup_frame_timeout_s)
        try:
            send_message(
                self.socket,
                FRAME,
                epoch=self.epoch,
                sequence=sequence,
                source_monotonic_ns=source_ns,
                payload=pack_frame(pose, velocity),
            )
            reply = recv_message(self.socket)
            end_ns = time.perf_counter_ns()
            self._validate_reply(reply, RESULT, sequence, source_ns)
            qpos, processing_ns = unpack_result(reply.payload)
        except Exception:
            self.close()
            raise
        if startup_frame:
            self.socket.settimeout(self.frame_timeout_s)
        self.sequence += 1
        return qpos, processing_ns / 1_000_000.0, (end_ns - start_ns) / 1_000_000.0

    def _validate_reply(
        self, reply, expected_type: int, sequence: int, source_ns: int
    ) -> None:
        if reply.message_type == ERROR:
            raise ProtocolError(reply.payload.decode("utf-8", errors="replace"))
        if reply.message_type != expected_type:
            raise ProtocolError(
                f"unexpected reply type: {reply.message_type}, expected {expected_type}"
            )
        if reply.epoch != self.epoch or reply.sequence != sequence:
            raise ProtocolError(
                f"reply identity mismatch: epoch={reply.epoch} seq={reply.sequence}"
            )
        if reply.source_monotonic_ns != source_ns:
            raise ProtocolError("server did not echo the source timestamp")

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class OneFrameReferencePipeline:
    """Overlap local frame preparation with one ordered remote request.

    The remote reference generator is stateful, so this class never has more
    than one request in flight and never reorders requests. ``exchange``
    returns the preceding frame's result and submits the current frame. The
    first call only primes the pipeline and returns ``None``.
    """

    def __init__(self, session: RemoteReferenceSession):
        self.session = session
        self._condition = threading.Condition()
        self._request: tuple[np.ndarray, np.ndarray] | None = None
        self._result: tuple[np.ndarray, float, float] | None = None
        self._error: Exception | None = None
        self._pending = False
        self._discard_pending = False
        self._closing = False
        self._closed = False
        self.submitted_count = 0
        self.completed_count = 0
        self.discarded_count = 0
        self._thread = threading.Thread(
            target=self._worker,
            name="x2-reference-offload",
            daemon=True,
        )
        self._thread.start()

    def exchange(
        self, pose: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, float, float] | None:
        """Return the prior result, then submit the current frame."""

        pose_copy = np.asarray(pose, dtype=np.float32).reshape(24, 3).copy()
        velocity_copy = (
            np.asarray(velocity, dtype=np.float32).reshape(24, 3).copy()
        )
        prior_result = None
        error = None
        with self._condition:
            self._ensure_open_locked()
            while self._pending and self._result is None and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
            elif self._pending:
                prior_result = self._result
                self._result = None
                self._pending = False
                if self._discard_pending:
                    prior_result = None
                    self.discarded_count += 1
                self._discard_pending = False

            if error is None:
                self._request = (pose_copy, velocity_copy)
                self._pending = True
                self.submitted_count += 1
                self._condition.notify_all()

        if error is not None:
            self.close()
            raise error
        return prior_result

    def hold(self) -> None:
        """Stop old in-flight output from being emitted after an input pause."""

        error = None
        with self._condition:
            if self._closed:
                return
            if self._error is not None:
                error = self._error
            elif self._pending:
                if self._request is not None:
                    # The worker has not consumed this frame, so cancel it.
                    self._request = None
                    self._pending = False
                    self._discard_pending = False
                    self.discarded_count += 1
                elif self._result is not None:
                    self._result = None
                    self._pending = False
                    self._discard_pending = False
                    self.discarded_count += 1
                else:
                    self._discard_pending = True
            else:
                self._discard_pending = False
        if error is not None:
            self.close()
            raise error

    def flush(self, *, discard: bool = False):
        """Wait for and consume the final in-flight result without submitting."""

        result = None
        error = None
        with self._condition:
            self._ensure_open_locked()
            while self._pending and self._result is None and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
            elif self._pending:
                result = self._result
                self._result = None
                self._pending = False
                should_discard = discard or self._discard_pending
                self._discard_pending = False
                if should_discard:
                    result = None
                    self.discarded_count += 1
        if error is not None:
            self.close()
            raise error
        return result

    def snapshot(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "submitted": self.submitted_count,
                "completed": self.completed_count,
                "discarded": self.discarded_count,
                "pending": self._pending,
            }

    def _worker(self) -> None:
        while True:
            with self._condition:
                while self._request is None and not self._closing:
                    self._condition.wait()
                if self._closing:
                    return
                pose, velocity = self._request
                self._request = None

            try:
                result = self.session.process(pose, velocity)
            except Exception as error:
                with self._condition:
                    if not self._closing:
                        self._error = error
                    self._condition.notify_all()
                return

            with self._condition:
                if self._closing:
                    return
                self._result = result
                self.completed_count += 1
                self._condition.notify_all()

    def _ensure_open_locked(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("reference pipeline is closed")

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._request = None
            self._condition.notify_all()
        self.session.close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        with self._condition:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
