#!/usr/bin/env python3
"""Isolated 3588S reference-generation server for dry benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from articulate.math import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.SmpleXConverter import SMPLXConverter
from mos1.physics_models import KinematicModel
from native_gmr import NativePreprocessedGMR
from reference_offload_protocol import (
    ACK,
    ERROR,
    FRAME,
    RESET,
    RESULT,
    ProtocolError,
    pack_result,
    recv_message,
    send_message,
    unpack_frame,
)


class NativeGMR(NativePreprocessedGMR, GeneralMotionRetargeting):
    pass


class ReferenceWorker:
    def __init__(self, max_iter: int, damping: float):
        self.physics = KinematicModel(fps=30, bio_axis=True)
        self.converter = SMPLXConverter()
        self.retarget = NativeGMR(
            src_human="smplx",
            tgt_robot="unitree_g1",
            actual_human_height=1.8,
            damping=damping,
            verbose=False,
        )
        self.retarget.max_iter = max_iter

    def process(
        self, pose: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        start_ns = time.perf_counter_ns()
        with torch.inference_mode():
            rotation = axis_angle_to_rotation_matrix(torch.from_numpy(pose))
            self.physics.update_state(
                pose=rotation,
                vel=torch.from_numpy(velocity),
                stationary=None,
            )
            optimized_rotation, translation = self.physics.get_state()
            optimized_axis_angle = rotation_matrix_to_axis_angle(optimized_rotation[0])
            physics_ns = time.perf_counter_ns()
            human_data = self.converter.convert_axis_angle_to_human_data_fast(
                optimized_axis_angle.detach().cpu().numpy(),
                translation.detach().cpu().numpy(),
                device="cpu",
            )
            smpl_ns = time.perf_counter_ns()
            qpos = self.retarget.retarget(human_data)
            gmr_ns = time.perf_counter_ns()
        return qpos, {
            "physics_ms": (physics_ns - start_ns) / 1_000_000.0,
            "fast_smpl_ms": (smpl_ns - physics_ns) / 1_000_000.0,
            "native_gmr_ms": (gmr_ns - smpl_ns) / 1_000_000.0,
            "total_ms": (gmr_ns - start_ns) / 1_000_000.0,
        }


def timing_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(np.max(array)),
    }


def serve_connection(conn: socket.socket, args: argparse.Namespace) -> dict[str, object]:
    conn.settimeout(args.socket_timeout)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    worker: ReferenceWorker | None = None
    active_epoch: int | None = None
    expected_sequence = 0
    processing_ms: list[float] = []
    physics_ms: list[float] = []
    fast_smpl_ms: list[float] = []
    native_gmr_ms: list[float] = []
    captured_pose: list[np.ndarray] = []
    captured_velocity: list[np.ndarray] = []
    captured_qpos: list[np.ndarray] = []

    try:
        while True:
            try:
                message = recv_message(conn)
            except EOFError:
                break

            if message.message_type == RESET:
                if message.payload:
                    raise ProtocolError("RESET payload must be empty")
                worker = ReferenceWorker(args.max_iter, args.damping)
                active_epoch = message.epoch
                expected_sequence = 0
                send_message(
                    conn,
                    ACK,
                    epoch=message.epoch,
                    sequence=message.sequence,
                    source_monotonic_ns=message.source_monotonic_ns,
                )
                continue

            if message.message_type != FRAME:
                raise ProtocolError(f"unexpected message type: {message.message_type}")
            if worker is None or active_epoch is None:
                raise ProtocolError("FRAME received before RESET")
            if message.epoch != active_epoch:
                raise ProtocolError(
                    f"epoch mismatch: received={message.epoch} active={active_epoch}"
                )
            if message.sequence != expected_sequence:
                raise ProtocolError(
                    f"sequence mismatch: received={message.sequence} expected={expected_sequence}"
                )

            pose, velocity = unpack_frame(message.payload)
            qpos, stage = worker.process(pose, velocity)
            elapsed_ns = int(round(stage["total_ms"] * 1_000_000.0))
            processing_ms.append(stage["total_ms"])
            physics_ms.append(stage["physics_ms"])
            fast_smpl_ms.append(stage["fast_smpl_ms"])
            native_gmr_ms.append(stage["native_gmr_ms"])
            if args.capture_npz is not None:
                captured_pose.append(pose.copy())
                captured_velocity.append(velocity.copy())
                captured_qpos.append(np.asarray(qpos).copy())
            if stage["total_ms"] >= args.slow_frame_ms:
                print(
                    json.dumps(
                        {
                            "event": "slow_frame",
                            "sequence": message.sequence,
                            **stage,
                            "pose_abs_max": float(np.max(np.abs(pose))),
                            "velocity_abs_max": float(np.max(np.abs(velocity))),
                            "qpos_abs_max": float(np.max(np.abs(qpos))),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if (
                args.disconnect_after_frames > 0
                and expected_sequence + 1 >= args.disconnect_after_frames
            ):
                print(
                    json.dumps(
                        {
                            "event": "intentional_disconnect",
                            "processed_frames": expected_sequence + 1,
                            "sequence_without_result": message.sequence,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
            send_message(
                conn,
                RESULT,
                epoch=active_epoch,
                sequence=message.sequence,
                source_monotonic_ns=message.source_monotonic_ns,
                payload=pack_result(qpos, elapsed_ns),
            )
            expected_sequence += 1
    finally:
        if args.capture_npz is not None and captured_pose:
            capture_path = args.capture_npz.expanduser().resolve()
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                capture_path,
                pose=np.stack(captured_pose),
                velocity=np.stack(captured_velocity),
                qpos=np.stack(captured_qpos),
                physics_ms=np.asarray(physics_ms),
                fast_smpl_ms=np.asarray(fast_smpl_ms),
                native_gmr_ms=np.asarray(native_gmr_ms),
                total_ms=np.asarray(processing_ms),
            )
            print(
                json.dumps(
                    {
                        "event": "capture_saved",
                        "frames": len(captured_pose),
                        "path": str(capture_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    processing = timing_summary(processing_ms)
    return {
        "frames": len(processing_ms),
        "epoch": active_epoch,
        "processing": processing,
        "processing_median_ms": None if processing is None else processing["median_ms"],
        "processing_p95_ms": None if processing is None else processing["p95_ms"],
        "processing_max_ms": None if processing is None else processing["max_ms"],
        "physics": timing_summary(physics_ms),
        "fast_smpl": timing_summary(fast_smpl_ms),
        "native_gmr": timing_summary(native_gmr_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.0.1.42")
    parser.add_argument("--port", type=int, default=51237)
    parser.add_argument("--socket-timeout", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--slow-frame-ms", type=float, default=80.0)
    parser.add_argument("--capture-npz", type=Path)
    parser.add_argument(
        "--disconnect-after-frames",
        type=int,
        default=0,
        help="diagnostic only: close before replying to this processed frame",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.threads <= 0
        or args.max_iter < 0
        or args.slow_frame_ms <= 0.0
        or args.disconnect_after_frames < 0
    ):
        raise ValueError("threads and slow-frame threshold must be positive")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(1)
    print(json.dumps({"event": "listening", "host": args.host, "port": args.port}))

    try:
        while True:
            conn, address = listener.accept()
            print(json.dumps({"event": "connected", "peer": address[0]}))
            try:
                with conn:
                    summary = serve_connection(conn, args)
            except (OSError, ProtocolError, RuntimeError, ValueError) as error:
                try:
                    send_message(
                        conn,
                        ERROR,
                        epoch=0,
                        sequence=0,
                        payload=str(error).encode("utf-8")[:1024],
                    )
                except OSError:
                    pass
                summary = {"error": f"{type(error).__name__}: {error}"}
            print(json.dumps({"event": "disconnected", **summary}, sort_keys=True))
            if args.once:
                break
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
