#!/usr/bin/env python3
"""Replay verified SmartWear V2 raw frames through LFP and the reference RPC.

This benchmark has no BLE, ROS, Sonic, MC, or HAL publisher. It uses the A3
raw-dataset validator and frozen clothing frontend, then sends only LFP
pose/velocity frames to the isolated 3588S reference server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np

from reference_offload_client import OneFrameReferencePipeline, RemoteReferenceSession
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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-metadata", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--a3-tools-root", required=True, type=Path)
    parser.add_argument("--vendor-root", required=True, type=Path)
    parser.add_argument("--provider", choices=("cpu", "auto"), default="auto")
    parser.add_argument("--host", default="10.0.1.42")
    parser.add_argument("--port", type=int, default=51237)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--reset-timeout", type=float, default=10.0)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--calibration-samples", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("raw_reference_rpc.npz"))
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="overlap each local LFP frame with the preceding ordered RPC",
    )
    parser.add_argument(
        "--expected-qpos-npz",
        type=Path,
        help="optional synchronous output whose qpos must match exactly",
    )
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("--frames must be greater than --warmup")
    if not 30 <= args.calibration_samples <= 300:
        parser.error("--calibration-samples must be between 30 and 300")
    return args


def main() -> int:
    args = parse_args()
    tools_root = args.a3_tools_root.expanduser().resolve()
    sys.path.insert(0, str(tools_root))
    import a3_clothing_frontend
    import a3_smartwear_v2_raw_replay

    records, metadata, raw_digest = a3_smartwear_v2_raw_replay.load_dataset(
        args.input,
        args.input_metadata,
        args.expected_input_sha256,
        np,
    )
    t_pose = [record for record in records if record["phase"] == "t_pose_calibration"]
    tracking = [record for record in records if record["phase"] != "t_pose_calibration"]
    if len(t_pose) < args.calibration_samples:
        raise RuntimeError("raw dataset has too few T-Pose frames")
    if len(tracking) < args.frames:
        raise RuntimeError("raw dataset has too few tracking frames")
    selected_records = tracking[: args.frames]

    frontend = a3_clothing_frontend.ClothingPoseFrontend(
        args.vendor_root, provider=args.provider
    )
    frontend.calibrate(
        [record["_jacket"] for record in t_pose[-args.calibration_samples :]],
        [record["_pants"] for record in t_pose[-args.calibration_samples :]],
    )

    preprocess_ms = np.empty(args.frames, dtype=np.float64)
    lfp_ms = np.empty(args.frames, dtype=np.float64)
    remote_ms = np.empty(args.frames, dtype=np.float64)
    rtt_ms = np.empty(args.frames, dtype=np.float64)
    overhead_ms = np.empty(args.frames, dtype=np.float64)
    qpos = np.empty((args.frames, 36), dtype=np.float64)
    phases: list[str] = []

    sock = None
    remote_session = None
    pipeline = None
    try:
        if args.pipeline:
            remote_session = RemoteReferenceSession(
                args.host,
                args.port,
                frame_timeout_s=args.timeout,
                reset_timeout_s=args.reset_timeout,
                epoch=args.epoch,
                startup_frame_timeout_s=args.timeout,
                startup_frame_count=0,
            )
            pipeline = OneFrameReferencePipeline(remote_session)
        else:
            sock = socket.create_connection(
                (args.host, args.port), timeout=args.reset_timeout
            )
            sock.settimeout(args.reset_timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
        for sequence, record in enumerate(selected_records):
            preprocess_start_ns = time.perf_counter_ns()
            selected = a3_clothing_frontend.select_model_imus(
                record["_jacket"],
                record["_pants"],
                frontend.up_order,
                frontend.down_order,
            )
            calibrated = frontend.server.calibrate(frontend._sensor_vector(selected))
            frontend.server.operator(calibrated)
            data_feed = frontend.server.to_predict_data()
            preprocess_end_ns = time.perf_counter_ns()

            result = frontend.pose_session.run(output_names=None, input_feed=data_feed)
            lfp_end_ns = time.perf_counter_ns()
            if len(result) != 9:
                raise RuntimeError(f"LFP returned {len(result)} outputs, expected 9")
            (
                pose,
                _joint,
                velocity,
                frontend.server.h_1,
                frontend.server.c_1,
                frontend.server.h_2,
                frontend.server.c_2,
                frontend.server.h_3,
                frontend.server.c_3,
            ) = result
            pose = np.asarray(pose, dtype=np.float32).reshape(24, 3)
            velocity = np.asarray(velocity, dtype=np.float32).reshape(24, 3)
            if not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                raise RuntimeError(f"non-finite LFP output at frame {sequence}")

            preprocess_ms[sequence] = (
                preprocess_end_ns - preprocess_start_ns
            ) / 1_000_000.0
            lfp_ms[sequence] = (lfp_end_ns - preprocess_end_ns) / 1_000_000.0
            phases.append(str(record["phase"]))

            if pipeline is not None:
                reference_result = pipeline.exchange(pose, velocity)
                if reference_result is not None:
                    output_index = sequence - 1
                    (
                        qpos[output_index],
                        remote_ms[output_index],
                        rtt_ms[output_index],
                    ) = reference_result
                    overhead_ms[output_index] = (
                        rtt_ms[output_index] - remote_ms[output_index]
                    )
            else:
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
                rtt_ms[sequence] = (
                    request_end_ns - request_start_ns
                ) / 1_000_000.0
                remote_ms[sequence] = processing_ns / 1_000_000.0
                overhead_ms[sequence] = rtt_ms[sequence] - remote_ms[sequence]

        if pipeline is not None:
            final_result = pipeline.flush()
            if final_result is None:
                raise RuntimeError("reference pipeline ended without its final result")
            qpos[-1], remote_ms[-1], rtt_ms[-1] = final_result
            overhead_ms[-1] = rtt_ms[-1] - remote_ms[-1]
        wall_elapsed_s = (
            time.perf_counter_ns() - wall_start_ns
        ) / 1_000_000_000.0
    finally:
        if pipeline is not None:
            pipeline.close()
        elif remote_session is not None:
            remote_session.close()
        if sock is not None:
            sock.close()
        frontend.close()

    if not np.isfinite(qpos).all():
        raise RuntimeError("qpos output contains NaN or Inf")
    max_qpos_error = None
    if args.expected_qpos_npz is not None:
        with np.load(args.expected_qpos_npz, allow_pickle=False) as expected:
            expected_qpos = expected["qpos"]
        if expected_qpos.shape != qpos.shape:
            raise ValueError(
                f"expected qpos shape {expected_qpos.shape}, actual {qpos.shape}"
            )
        max_qpos_error = float(np.max(np.abs(expected_qpos - qpos)))
        if max_qpos_error != 0.0:
            raise RuntimeError(
                f"pipelined qpos differs from expected by {max_qpos_error} rad"
            )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos,
        preprocess_ms=preprocess_ms,
        lfp_ms=lfp_ms,
        remote_ms=remote_ms,
        rtt_ms=rtt_ms,
        overhead_ms=overhead_ms,
        phase=np.asarray(phases),
    )

    measured = slice(args.warmup, None)
    phase_stats = {}
    phase_array = np.asarray(phases)
    for phase in sorted(set(phases)):
        values = remote_ms[(phase_array == phase) & (np.arange(args.frames) >= args.warmup)]
        if values.size:
            phase_stats[phase] = stats(values)
    summary = {
        "dataset_sha256": raw_digest,
        "recorded_rate_hz": metadata.get("rates_hz", {}).get("recorded_pairs"),
        "frames": args.frames,
        "warmup": args.warmup,
        "provider": frontend.provenance["selected_providers"],
        "pipeline": args.pipeline,
        "wall_rate_hz": args.frames / wall_elapsed_s,
        "preprocess": stats(preprocess_ms[measured]),
        "lfp": stats(lfp_ms[measured]),
        "remote_processing": stats(remote_ms[measured]),
        "rtt": stats(rtt_ms[measured]),
        "rpc_overhead": stats(overhead_ms[measured]),
        "remote_first_20_ms": np.round(remote_ms[:20], 3).tolist(),
        "remote_by_phase": phase_stats,
        "qpos_finite": True,
        "max_qpos_error_rad": max_qpos_error,
        "publishing": "disabled_by_design",
        "output": str(output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
