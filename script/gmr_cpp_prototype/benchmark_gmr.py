from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.SmpleXConverter import SMPLXConverter
from native_gmr import LimitsFixedGMR, NativePreprocessedGMR


class FixedGMR(LimitsFixedGMR, GeneralMotionRetargeting):
    pass


class NativeGMR(NativePreprocessedGMR, GeneralMotionRetargeting):
    pass


def load_converter(path: Path | None):
    if path is None:
        return SMPLXConverter
    spec = importlib.util.spec_from_file_location("candidate_smpl_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SMPLXConverter


def make_retarget(cls, max_iter, damping):
    retarget = cls(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=1.8,
        damping=damping,
        verbose=False,
    )
    retarget.max_iter = max_iter
    return retarget


def make_sequence(converter, count):
    sequence = []
    joint_phase = np.arange(24, dtype=np.float32) * 0.1
    for frame in range(count):
        phase = frame * 0.04
        axis_angles = np.zeros((24, 3), dtype=np.float32)
        axis_angles[:, 1] = 0.04 * np.sin(phase + joint_phase)
        axis_angles[16, 2] = 0.35 * np.sin(phase)
        axis_angles[17, 2] = -0.35 * np.sin(phase)
        root_translation = np.array(
            [0.02 * np.sin(phase), 0.0, 0.8 + 0.01 * np.cos(phase)], dtype=np.float32
        )
        sequence.append(
            converter.convert_axis_angle_to_human_data_fast(
                axis_angles, root_translation, device="cpu"
            )
        )
    return sequence


def run(retarget, sequence, warmup):
    output = []
    elapsed_ms = []
    for index, human_data in enumerate(sequence):
        start = time.perf_counter()
        output.append(retarget.retarget(human_data))
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= warmup:
            elapsed_ms.append(elapsed)
    return np.asarray(output), np.asarray(elapsed_ms)


def timing(name, elapsed_ms):
    print(
        f"{name}: median={np.median(elapsed_ms):.3f}ms "
        f"p95={np.percentile(elapsed_ms, 95):.3f}ms mean={np.mean(elapsed_ms):.3f}ms"
    )


def delta(name, expected, actual):
    absolute = np.abs(expected - actual)
    print(
        f"{name}: median={np.median(absolute):.9f}rad "
        f"p95={np.percentile(absolute, 95):.9f}rad max={np.max(absolute):.9f}rad"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=220)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--converter", type=Path)
    args = parser.parse_args()

    converter = load_converter(args.converter)()
    sequence = make_sequence(converter, args.frames)
    baseline = make_retarget(GeneralMotionRetargeting, args.max_iter, args.damping)
    fixed = make_retarget(FixedGMR, args.max_iter, args.damping)
    native = make_retarget(NativeGMR, args.max_iter, args.damping)

    baseline_qpos, baseline_ms = run(baseline, sequence, args.warmup)
    fixed_qpos, fixed_ms = run(fixed, sequence, args.warmup)
    native_qpos, native_ms = run(native, sequence, args.warmup)

    timing("baseline", baseline_ms)
    timing("limits_fixed", fixed_ms)
    timing("native_preprocess", native_ms)
    delta("limits_fixed_qpos_error", baseline_qpos, fixed_qpos)
    delta("native_qpos_error", baseline_qpos, native_qpos)


if __name__ == "__main__":
    main()
