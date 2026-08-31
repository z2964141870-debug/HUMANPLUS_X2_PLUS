#!/usr/bin/env python3
"""Replay a deterministic sequence through the 3588S reference RPC server."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from benchmark_reference_pipeline import make_sequence
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


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "max_ms": float(np.max(values)),
    }


def require_reply(message, expected_type: int, epoch: int, sequence: int) -> None:
    if message.message_type == ERROR:
        raise ProtocolError(message.payload.decode("utf-8", errors="replace"))
    if message.message_type != expected_type:
        raise ProtocolError(
            f"unexpected reply type: {message.message_type}, expected {expected_type}"
        )
    if message.epoch != epoch or message.sequence != sequence:
        raise ProtocolError(
            f"reply identity mismatch: epoch={message.epoch} seq={message.sequence}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.0.1.42")
    parser.add_argument("--port", type=int, default=51237)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--reset-timeout", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--expected-npz", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference_rpc_output.npz"))
    parser.add_argument("--qpos-atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= args.warmup:
        raise ValueError("--frames must be greater than --warmup")

    poses, velocities = make_sequence(args.frames)
    qpos = np.empty((args.frames, 36), dtype=np.float64)
    rtt_ms = np.empty(args.frames, dtype=np.float64)
    remote_ms = np.empty(args.frames, dtype=np.float64)
    overhead_ms = np.empty(args.frames, dtype=np.float64)

    sock = socket.create_connection((args.host, args.port), timeout=args.reset_timeout)
    sock.settimeout(args.reset_timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with sock:
        reset_timestamp = time.monotonic_ns()
        send_message(
            sock,
            RESET,
            epoch=args.epoch,
            sequence=0,
            source_monotonic_ns=reset_timestamp,
        )
        require_reply(recv_message(sock), ACK, args.epoch, 0)
        sock.settimeout(args.timeout)

        wall_start_ns = time.perf_counter_ns()
        for sequence, (pose, velocity) in enumerate(zip(poses, velocities)):
            source_ns = time.monotonic_ns()
            request_start_ns = time.perf_counter_ns()
            send_message(
                sock,
                FRAME,
                epoch=args.epoch,
                sequence=sequence,
                source_monotonic_ns=source_ns,
                payload=pack_frame(pose, velocity),
            )
            reply = recv_message(sock)
            request_end_ns = time.perf_counter_ns()
            require_reply(reply, RESULT, args.epoch, sequence)
            if reply.source_monotonic_ns != source_ns:
                raise ProtocolError("server did not echo the source timestamp")
            qpos[sequence], processing_ns = unpack_result(reply.payload)
            rtt_ms[sequence] = (request_end_ns - request_start_ns) / 1_000_000.0
            remote_ms[sequence] = processing_ns / 1_000_000.0
            overhead_ms[sequence] = rtt_ms[sequence] - remote_ms[sequence]
        wall_elapsed_s = (time.perf_counter_ns() - wall_start_ns) / 1_000_000_000.0

    measured = slice(args.warmup, None)
    max_qpos_error = None
    passed = True
    if args.expected_npz is not None:
        with np.load(args.expected_npz, allow_pickle=False) as expected:
            expected_qpos = expected["qpos"]
        if expected_qpos.shape != qpos.shape:
            raise ValueError(f"qpos shape mismatch: {expected_qpos.shape} != {qpos.shape}")
        max_qpos_error = float(np.max(np.abs(expected_qpos - qpos)))
        passed = max_qpos_error <= args.qpos_atol

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        qpos=qpos,
        rtt_ms=rtt_ms,
        remote_ms=remote_ms,
        overhead_ms=overhead_ms,
    )
    print(
        json.dumps(
            {
                "frames": args.frames,
                "warmup": args.warmup,
                "wall_rate_hz": args.frames / wall_elapsed_s,
                "rtt": stats(rtt_ms[measured]),
                "remote_processing": stats(remote_ms[measured]),
                "rpc_overhead": stats(overhead_ms[measured]),
                "max_qpos_error_rad": max_qpos_error,
                "qpos_atol_rad": args.qpos_atol,
                "pass": passed,
                "npz": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
