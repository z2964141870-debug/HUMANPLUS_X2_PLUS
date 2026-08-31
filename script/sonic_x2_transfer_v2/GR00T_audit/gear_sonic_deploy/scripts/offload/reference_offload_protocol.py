"""Versioned binary protocol for SoC1 -> reference-generator offload."""

from __future__ import annotations

import socket
import struct
import zlib
from dataclasses import dataclass

import numpy as np


MAGIC = b"X2RF"
VERSION = 1
RESET = 1
FRAME = 2
ACK = 3
RESULT = 4
ERROR = 5

HEADER = struct.Struct("!4sBBHIQQII")
PROCESSING_NS = struct.Struct("<Q")
FRAME_FLOAT_COUNT = 24 * 3 * 2
FRAME_PAYLOAD_BYTES = FRAME_FLOAT_COUNT * 4
QPOS_COUNT = 36
RESULT_PAYLOAD_BYTES = PROCESSING_NS.size + QPOS_COUNT * 8
MAX_PAYLOAD_BYTES = 64 * 1024


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Message:
    message_type: int
    flags: int
    epoch: int
    sequence: int
    source_monotonic_ns: int
    payload: bytes


def recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed while receiving a framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_message(
    message_type: int,
    *,
    epoch: int,
    sequence: int,
    source_monotonic_ns: int = 0,
    payload: bytes = b"",
    flags: int = 0,
) -> bytes:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload too large: {len(payload)} bytes")
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        VERSION,
        message_type,
        flags,
        epoch,
        sequence,
        source_monotonic_ns,
        len(payload),
        checksum,
    )
    return header + payload


def send_message(sock: socket.socket, message_type: int, **kwargs) -> None:
    sock.sendall(encode_message(message_type, **kwargs))


def recv_message(sock: socket.socket) -> Message:
    header = recv_exact(sock, HEADER.size)
    (
        magic,
        version,
        message_type,
        flags,
        epoch,
        sequence,
        source_monotonic_ns,
        payload_size,
        expected_checksum,
    ) = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload too large: {payload_size} bytes")
    payload = recv_exact(sock, payload_size)
    actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_checksum != expected_checksum:
        raise ProtocolError(
            f"payload CRC mismatch: expected={expected_checksum:#x} actual={actual_checksum:#x}"
        )
    return Message(
        message_type=message_type,
        flags=flags,
        epoch=epoch,
        sequence=sequence,
        source_monotonic_ns=source_monotonic_ns,
        payload=payload,
    )


def pack_frame(pose: np.ndarray, velocity: np.ndarray) -> bytes:
    pose_array = np.asarray(pose, dtype="<f4").reshape(24, 3)
    velocity_array = np.asarray(velocity, dtype="<f4").reshape(24, 3)
    payload = np.concatenate((pose_array.reshape(-1), velocity_array.reshape(-1)))
    return np.ascontiguousarray(payload, dtype="<f4").tobytes()


def unpack_frame(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) != FRAME_PAYLOAD_BYTES:
        raise ProtocolError(
            f"frame payload has {len(payload)} bytes, expected {FRAME_PAYLOAD_BYTES}"
        )
    values = np.frombuffer(payload, dtype="<f4")
    pose = values[: 24 * 3].reshape(24, 3).copy()
    velocity = values[24 * 3 :].reshape(24, 3).copy()
    if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
        raise ProtocolError("frame payload contains non-finite values")
    return pose, velocity


def pack_result(qpos: np.ndarray, processing_ns: int) -> bytes:
    qpos_array = np.asarray(qpos, dtype="<f8").reshape(-1)
    if qpos_array.size != QPOS_COUNT:
        raise ProtocolError(f"result has {qpos_array.size} qpos values, expected {QPOS_COUNT}")
    if not np.isfinite(qpos_array).all():
        raise ProtocolError("result qpos contains non-finite values")
    return PROCESSING_NS.pack(processing_ns) + qpos_array.tobytes()


def unpack_result(payload: bytes) -> tuple[np.ndarray, int]:
    if len(payload) != RESULT_PAYLOAD_BYTES:
        raise ProtocolError(
            f"result payload has {len(payload)} bytes, expected {RESULT_PAYLOAD_BYTES}"
        )
    (processing_ns,) = PROCESSING_NS.unpack_from(payload)
    qpos = np.frombuffer(payload, dtype="<f8", offset=PROCESSING_NS.size).copy()
    if qpos.size != QPOS_COUNT or not np.isfinite(qpos).all():
        raise ProtocolError("invalid qpos result payload")
    return qpos, processing_ns
