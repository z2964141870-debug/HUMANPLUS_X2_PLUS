from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from general_motion_retargeting.SmpleXConverter import (
    SMPLXConverter as DonorConverter,
)


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_smpl_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SMPLXConverter


def make_sequence(count: int):
    sequence = []
    joint_phase = np.arange(24, dtype=np.float32) * 0.13
    for frame in range(count):
        phase = frame * 0.04
        axis_angles = np.zeros((24, 3), dtype=np.float32)
        axis_angles[:, 0] = 0.03 * np.cos(phase + joint_phase)
        axis_angles[:, 1] = 0.04 * np.sin(phase + joint_phase)
        axis_angles[16, 2] = 0.35 * np.sin(phase)
        axis_angles[17, 2] = -0.35 * np.sin(phase)
        axis_angles[18, 1] = -0.5 * np.sin(phase)
        axis_angles[19, 1] = 0.5 * np.sin(phase)
        root_translation = np.array(
            [0.02 * np.sin(phase), 0.0, 0.8 + 0.01 * np.cos(phase)],
            dtype=np.float32,
        )
        sequence.append((axis_angles, root_translation))
    return sequence


def donor_frame(converter, axis_angles, root_translation):
    bundle = converter.convert_axis_angle_to_smpl_bundle_fast(
        axis_angles, root_translation
    )
    return bundle["human_data"]


def candidate_frame(converter, axis_angles, root_translation, device):
    return converter.convert_axis_angle_to_human_data_fast(
        axis_angles, root_translation, device=device
    )


def run(name, converter, sequence, warmup, frame_fn):
    outputs = []
    timings = []
    for index, (axis_angles, root_translation) in enumerate(sequence):
        start = time.perf_counter()
        outputs.append(frame_fn(converter, axis_angles, root_translation))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index >= warmup:
            timings.append(elapsed_ms)
    values = np.asarray(timings)
    print(
        f"{name}: median={np.median(values):.3f}ms "
        f"p95={np.percentile(values, 95):.3f}ms "
        f"mean={np.mean(values):.3f}ms max={np.max(values):.3f}ms"
    )
    return outputs


def compare(name, expected, actual):
    max_position_error = 0.0
    max_quaternion_error = 0.0
    for expected_frame, actual_frame in zip(expected, actual):
        if expected_frame.keys() != actual_frame.keys():
            raise RuntimeError("converter body-name sets differ")
        for body_name in expected_frame:
            expected_position, expected_quaternion = expected_frame[body_name]
            actual_position, actual_quaternion = actual_frame[body_name]
            max_position_error = max(
                max_position_error,
                float(np.max(np.abs(expected_position - actual_position))),
            )
            direct = np.max(np.abs(expected_quaternion - actual_quaternion))
            negated = np.max(np.abs(expected_quaternion + actual_quaternion))
            max_quaternion_error = max(
                max_quaternion_error, float(min(direct, negated))
            )
    print(
        f"{name}: max_position_error={max_position_error:.3e} "
        f"max_quaternion_error={max_quaternion_error:.3e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=220)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("--frames must be greater than --warmup")

    candidate_class = load_candidate(args.candidate)
    sequence = make_sequence(args.frames)

    donor = run(
        "donor_auto",
        DonorConverter(),
        sequence,
        args.warmup,
        donor_frame,
    )
    candidate_cpu = run(
        "candidate_cpu",
        candidate_class(),
        sequence,
        args.warmup,
        lambda converter, angles, translation: candidate_frame(
            converter, angles, translation, "cpu"
        ),
    )
    compare("candidate_cpu_vs_donor", donor, candidate_cpu)

    try:
        candidate_cuda = run(
            "candidate_cuda",
            candidate_class(),
            sequence,
            args.warmup,
            lambda converter, angles, translation: candidate_frame(
                converter, angles, translation, "cuda"
            ),
        )
    except (AssertionError, RuntimeError) as error:
        print(f"candidate_cuda: unavailable ({error})")
    else:
        compare("candidate_cuda_vs_donor", donor, candidate_cuda)


if __name__ == "__main__":
    main()
