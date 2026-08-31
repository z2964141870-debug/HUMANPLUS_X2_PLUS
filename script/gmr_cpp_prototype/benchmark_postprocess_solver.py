from __future__ import annotations

import argparse
import time

import numpy as np

from Socket.UDP import DataProcessServer_FullBody, demo_mode
import config
from mos1 import physics_models


def make_server():
    return DataProcessServer_FullBody(
        rotation_type="AXIS_ANGLE",
        part="body",
        config=[config.device_config.jacket_6IMU, config.device_config.pants_5IMU],
        mode=demo_mode.FULL,
        track_trans=True,
        calibration_session=None,
        run_unity_package=False,
        physics_optim=True,
        cali_pose="T",
        beta=None,
    )


def make_sequence(count):
    sequence = []
    joint_phase = np.arange(24, dtype=np.float32) * 0.11
    hidden = np.zeros((2, 1, 256), dtype=np.float32)
    for frame in range(count):
        phase = frame * 0.04
        pose = np.zeros((24, 3), dtype=np.float32)
        pose[:, 0] = 0.025 * np.cos(phase + joint_phase)
        pose[:, 1] = 0.035 * np.sin(phase + joint_phase)
        pose[16, 2] = 0.25 * np.sin(phase)
        pose[17, 2] = -0.25 * np.sin(phase)
        velocity = np.zeros((24, 3), dtype=np.float32)
        velocity[:, 0] = 0.01 * np.cos(phase + joint_phase)
        velocity[:, 2] = 0.02 * np.sin(phase + joint_phase)
        joint = np.zeros((24, 3), dtype=np.float32)
        sequence.append(
            (
                pose.reshape(1, -1),
                joint.reshape(1, -1),
                velocity.reshape(1, -1),
                *(hidden.copy() for _ in range(6)),
            )
        )
    return sequence


def direct_solver(P, q, initvals=None):
    del initvals
    return np.linalg.solve(P, -q)


def run(name, solver, sequence, warmup):
    physics_models._solve_qp_with_available_solver = solver
    server = make_server()
    outputs = []
    timings = []
    for index, inputs in enumerate(sequence):
        start = time.perf_counter()
        result = server.predict_result(inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        outputs.append(
            (
                np.asarray(result["axis_angles"], dtype=np.float64),
                np.asarray(result["root_translation"], dtype=np.float64),
            )
        )
        if index >= warmup:
            timings.append(elapsed_ms)
    values = np.asarray(timings)
    print(
        f"{name}: median={np.median(values):.3f}ms "
        f"p95={np.percentile(values, 95):.3f}ms "
        f"mean={np.mean(values):.3f}ms max={np.max(values):.3f}ms"
    )
    return outputs


def compare(expected, actual):
    pose_error = 0.0
    translation_error = 0.0
    for (expected_pose, expected_translation), (actual_pose, actual_translation) in zip(
        expected, actual
    ):
        pose_error = max(
            pose_error, float(np.max(np.abs(expected_pose - actual_pose)))
        )
        translation_error = max(
            translation_error,
            float(np.max(np.abs(expected_translation - actual_translation))),
        )
    print(
        f"direct_vs_quadprog: max_pose_error={pose_error:.3e}rad "
        f"max_translation_error={translation_error:.3e}m"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=220)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("--frames must be greater than --warmup")

    sequence = make_sequence(args.frames)
    quadprog_outputs = run(
        "quadprog",
        physics_models._solve_qp_with_available_solver,
        sequence,
        args.warmup,
    )
    direct_outputs = run("numpy_solve", direct_solver, sequence, args.warmup)
    compare(quadprog_outputs, direct_outputs)


if __name__ == "__main__":
    main()
