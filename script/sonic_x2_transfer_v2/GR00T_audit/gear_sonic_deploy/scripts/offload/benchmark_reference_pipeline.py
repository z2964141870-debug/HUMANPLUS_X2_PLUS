#!/usr/bin/env python3
"""Benchmark the offloadable garment reference-generation stages together.

The measured path is the same order used by garment_udp_v2_reconnect_safe.py:
physics postprocess -> Fast-SMPL -> native-preprocessed GMR.  It intentionally
does not include BLE, TIC/LFP, networking, Sonic, MC, or HAL.
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
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.SmpleXConverter import SMPLXConverter
from mos1.physics_models import KinematicModel
from native_gmr import NativePreprocessedGMR


class NativeGMR(NativePreprocessedGMR, GeneralMotionRetargeting):
    pass


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


def digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def timing_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "max_ms": float(np.max(values)),
    }


def make_retarget(max_iter: int, damping: float) -> NativeGMR:
    retarget = NativeGMR(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=1.8,
        damping=damping,
        verbose=False,
    )
    retarget.max_iter = max_iter
    return retarget


def run_benchmark(args: argparse.Namespace) -> int:
    if args.frames <= args.warmup:
        raise ValueError("--frames must be greater than --warmup")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    input_pose, input_velocity = make_sequence(args.frames)
    physics = KinematicModel(fps=30, bio_axis=True)
    converter = SMPLXConverter()
    retarget = make_retarget(args.max_iter, args.damping)

    physics_pose = np.empty((args.frames, 24, 3), dtype=np.float32)
    physics_translation = np.empty((args.frames, 3), dtype=np.float32)
    qpos = np.empty((args.frames, retarget.configuration.data.qpos.size), dtype=np.float64)
    physics_ms = np.empty(args.frames, dtype=np.float64)
    smpl_ms = np.empty(args.frames, dtype=np.float64)
    gmr_ms = np.empty(args.frames, dtype=np.float64)
    total_ms = np.empty(args.frames, dtype=np.float64)

    with torch.inference_mode():
        for index, (pose, velocity) in enumerate(zip(input_pose, input_velocity)):
            total_start_ns = time.perf_counter_ns()
            rotation = axis_angle_to_rotation_matrix(torch.from_numpy(pose))
            physics.update_state(
                pose=rotation,
                vel=torch.from_numpy(velocity),
                stationary=None,
            )
            optimized_rotation, translation = physics.get_state()
            optimized_axis_angle = rotation_matrix_to_axis_angle(optimized_rotation[0])
            physics_pose[index] = optimized_axis_angle.detach().cpu().numpy()
            physics_translation[index] = translation.detach().cpu().numpy()
            physics_end_ns = time.perf_counter_ns()

            human_data = converter.convert_axis_angle_to_human_data_fast(
                physics_pose[index], physics_translation[index], device="cpu"
            )
            smpl_end_ns = time.perf_counter_ns()
            qpos[index] = retarget.retarget(human_data)
            gmr_end_ns = time.perf_counter_ns()

            physics_ms[index] = (physics_end_ns - total_start_ns) / 1_000_000.0
            smpl_ms[index] = (smpl_end_ns - physics_end_ns) / 1_000_000.0
            gmr_ms[index] = (gmr_end_ns - smpl_end_ns) / 1_000_000.0
            total_ms[index] = (gmr_end_ns - total_start_ns) / 1_000_000.0

    for name, values in (
        ("physics_pose", physics_pose),
        ("physics_translation", physics_translation),
        ("qpos", qpos),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} contains non-finite values")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        input_pose=input_pose,
        input_velocity=input_velocity,
        physics_pose=physics_pose,
        physics_translation=physics_translation,
        qpos=qpos,
        physics_ms=physics_ms,
        smpl_ms=smpl_ms,
        gmr_ms=gmr_ms,
        total_ms=total_ms,
    )

    measured = slice(args.warmup, None)
    summary = {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "frames": args.frames,
        "warmup": args.warmup,
        "max_iter": args.max_iter,
        "physics": timing_summary(physics_ms[measured]),
        "fast_smpl": timing_summary(smpl_ms[measured]),
        "native_gmr": timing_summary(gmr_ms[measured]),
        "total": timing_summary(total_ms[measured]),
        "output_sha256": digest_arrays(physics_pose, physics_translation, qpos),
        "npz": str(output_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def compare_outputs(args: argparse.Namespace) -> int:
    expected_path, actual_path = (Path(value).expanduser() for value in args.compare)
    errors: dict[str, float] = {}
    with np.load(expected_path, allow_pickle=False) as expected, np.load(
        actual_path, allow_pickle=False
    ) as actual:
        for key in (
            "input_pose",
            "input_velocity",
            "physics_pose",
            "physics_translation",
            "qpos",
        ):
            if key not in expected or key not in actual:
                raise KeyError(f"missing array {key!r}")
            if expected[key].shape != actual[key].shape:
                raise ValueError(
                    f"shape mismatch for {key}: {expected[key].shape} != {actual[key].shape}"
                )
            errors[key] = float(np.max(np.abs(expected[key] - actual[key])))

    passed = (
        errors["input_pose"] == 0.0
        and errors["input_velocity"] == 0.0
        and errors["physics_pose"] <= args.pose_atol
        and errors["physics_translation"] <= args.translation_atol
        and errors["qpos"] <= args.qpos_atol
    )
    print(
        json.dumps(
            {
                "max_abs": errors,
                "pose_atol_rad": args.pose_atol,
                "translation_atol_m": args.translation_atol,
                "qpos_atol_rad": args.qpos_atol,
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
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--output", default="reference_pipeline_output.npz")
    parser.add_argument("--compare", nargs=2, metavar=("EXPECTED_NPZ", "ACTUAL_NPZ"))
    parser.add_argument("--pose-atol", type=float, default=5e-5)
    parser.add_argument("--translation-atol", type=float, default=5e-5)
    parser.add_argument("--qpos-atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0 or args.threads <= 0 or args.max_iter < 0:
        parser.error("invalid frame, warmup, thread, or iteration count")
    return args


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_outputs(args)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
