#!/usr/bin/env python3
"""Deterministic benchmark for the garment physics postprocessor.

This imports the same KinematicModel used by DataProcessServer_FullBody, but
does not start BLE, ONNX, networking, Sonic, MC, or HAL.  Run the same sequence
on two hosts and compare the resulting NPZ files before considering offload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from articulate.math import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from mos1.physics_models import KinematicModel


def make_sequence(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    joint_phase = np.arange(24, dtype=np.float32) * np.float32(0.11)
    poses = np.empty((frame_count, 24, 3), dtype=np.float32)
    velocities = np.empty_like(poses)

    for frame in range(frame_count):
        phase = np.float32(frame * 0.04)
        pose = np.zeros((24, 3), dtype=np.float32)
        pose[:, 0] = np.float32(0.025) * np.cos(phase + joint_phase)
        pose[:, 1] = np.float32(0.035) * np.sin(phase + joint_phase)
        pose[16, 2] = np.float32(0.25) * np.sin(phase)
        pose[17, 2] = -np.float32(0.25) * np.sin(phase)

        velocity = np.zeros((24, 3), dtype=np.float32)
        velocity[:, 0] = np.float32(0.01) * np.cos(phase + joint_phase)
        velocity[:, 2] = np.float32(0.02) * np.sin(phase + joint_phase)
        poses[frame] = pose
        velocities[frame] = velocity

    return poses, velocities


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def run_benchmark(args: argparse.Namespace) -> int:
    if args.frames <= args.warmup:
        raise ValueError("--frames must be greater than --warmup")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    input_pose, input_velocity = make_sequence(args.frames)
    model = KinematicModel(fps=30, bio_axis=True)
    output_pose = np.empty((args.frames, 24, 3), dtype=np.float32)
    output_translation = np.empty((args.frames, 3), dtype=np.float32)
    timings_ms = np.empty(args.frames, dtype=np.float64)

    with torch.inference_mode():
        for index, (pose, velocity) in enumerate(zip(input_pose, input_velocity)):
            start_ns = time.perf_counter_ns()
            rotation = axis_angle_to_rotation_matrix(torch.from_numpy(pose))
            model.update_state(
                pose=rotation,
                vel=torch.from_numpy(velocity),
                stationary=None,
            )
            optimized_rotation, translation = model.get_state()
            optimized_axis_angle = rotation_matrix_to_axis_angle(optimized_rotation[0])
            output_pose[index] = optimized_axis_angle.detach().cpu().numpy()
            output_translation[index] = translation.detach().cpu().numpy()
            timings_ms[index] = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    if not np.isfinite(output_pose).all() or not np.isfinite(output_translation).all():
        raise RuntimeError("physics stage produced non-finite output")

    measured = timings_ms[args.warmup :]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        input_pose=input_pose,
        input_velocity=input_velocity,
        output_pose=output_pose,
        output_translation=output_translation,
        timings_ms=timings_ms,
    )

    summary = {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "frames": args.frames,
        "warmup": args.warmup,
        "median_ms": float(np.median(measured)),
        "p95_ms": percentile(measured, 95),
        "mean_ms": float(np.mean(measured)),
        "max_ms": float(np.max(measured)),
        "output_sha256": array_digest(output_pose, output_translation),
        "npz": str(output_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def compare_outputs(args: argparse.Namespace) -> int:
    expected_path, actual_path = (Path(value).expanduser() for value in args.compare)
    with np.load(expected_path, allow_pickle=False) as expected, np.load(
        actual_path, allow_pickle=False
    ) as actual:
        for key in ("input_pose", "input_velocity", "output_pose", "output_translation"):
            if key not in expected or key not in actual:
                raise KeyError(f"missing array {key!r}")
            if expected[key].shape != actual[key].shape:
                raise ValueError(
                    f"shape mismatch for {key}: {expected[key].shape} != {actual[key].shape}"
                )

        input_pose_error = float(np.max(np.abs(expected["input_pose"] - actual["input_pose"])))
        input_velocity_error = float(
            np.max(np.abs(expected["input_velocity"] - actual["input_velocity"]))
        )
        pose_error = float(np.max(np.abs(expected["output_pose"] - actual["output_pose"])))
        translation_error = float(
            np.max(np.abs(expected["output_translation"] - actual["output_translation"]))
        )

    passed = (
        input_pose_error == 0.0
        and input_velocity_error == 0.0
        and pose_error <= args.pose_atol
        and translation_error <= args.translation_atol
    )
    print(
        json.dumps(
            {
                "input_pose_max_abs": input_pose_error,
                "input_velocity_max_abs": input_velocity_error,
                "output_pose_max_abs_rad": pose_error,
                "output_translation_max_abs_m": translation_error,
                "pose_atol_rad": args.pose_atol,
                "translation_atol_m": args.translation_atol,
                "pass": passed,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="physics_stage_output.npz")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("EXPECTED_NPZ", "ACTUAL_NPZ"),
        help="compare two prior outputs instead of running the benchmark",
    )
    parser.add_argument("--pose-atol", type=float, default=5e-5)
    parser.add_argument("--translation-atol", type=float, default=5e-5)
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0 or args.threads <= 0:
        parser.error("frames and threads must be positive; warmup must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_outputs(args)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
