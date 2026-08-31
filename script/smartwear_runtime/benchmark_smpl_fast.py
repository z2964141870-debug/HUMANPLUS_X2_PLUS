import argparse
import time

import numpy as np
import smplx
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.SmpleXConverter import SMPLXConverter


def make_sequence(count):
    sequence = []
    for frame in range(count):
        phase = frame / max(count - 1, 1) * 2.0 * np.pi
        axis_angles = np.zeros((24, 3), dtype=np.float32)
        axis_angles[3, 1] = 0.08 * np.sin(phase)
        axis_angles[16, 2] = 0.35 * np.sin(phase)
        axis_angles[17, 2] = -0.35 * np.sin(phase)
        axis_angles[18, 1] = -0.5 * np.sin(phase)
        axis_angles[19, 1] = 0.5 * np.sin(phase)
        root_translation = np.array(
            [0.04 * np.sin(phase), 0.0, 0.02 * np.cos(phase)], dtype=np.float32
        )
        sequence.append((axis_angles, root_translation))
    return sequence


def make_retarget(max_iter):
    retarget = GMR(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=1.8,
        damping=1.0,
        verbose=False,
    )
    retarget.max_iter = max_iter
    return retarget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--smplx-folder", required=True)
    args = parser.parse_args()

    converter = SMPLXConverter()
    body_model = smplx.create(
        args.smplx_folder, "smplx", gender="neutral", use_pca=False
    ).to("cuda")
    sequence = make_sequence(args.count)

    # Warm both GPU paths before measuring.
    converter.convert_axis_angle_to_human_data_fast(*sequence[0])
    converter.convert_axis_angle_to_smplx_bundle(*sequence[0], body_model)
    torch.cuda.synchronize()

    retarget_fast = make_retarget(args.max_iter)
    retarget_full = make_retarget(args.max_iter)
    q_fast = []
    q_full = []
    fast_ms = []
    full_ms = []
    for axis_angles, root_translation in sequence:
        start = time.perf_counter()
        human_fast = converter.convert_axis_angle_to_human_data_fast(
            axis_angles, root_translation
        )
        fast_ms.append((time.perf_counter() - start) * 1000.0)
        q_fast.append(retarget_fast.retarget(human_fast).copy())

        start = time.perf_counter()
        human_full = converter.convert_axis_angle_to_smplx_bundle(
            axis_angles, root_translation, body_model
        )["human_data"]
        full_ms.append((time.perf_counter() - start) * 1000.0)
        q_full.append(retarget_full.retarget(human_full).copy())

    q_fast = np.asarray(q_fast)
    q_full = np.asarray(q_full)
    delta = np.abs(q_fast - q_full)
    print(
        "fast_ms median={:.2f} p95={:.2f} full_ms median={:.2f} p95={:.2f}".format(
            np.median(fast_ms),
            np.percentile(fast_ms, 95),
            np.median(full_ms),
            np.percentile(full_ms, 95),
        )
    )
    print(
        "qpos_delta median={:.5f} p95={:.5f} max={:.5f}".format(
            np.median(delta), np.percentile(delta, 95), np.max(delta)
        )
    )
    print("fast_finite={} full_finite={}".format(np.isfinite(q_fast).all(), np.isfinite(q_full).all()))


if __name__ == "__main__":
    main()
