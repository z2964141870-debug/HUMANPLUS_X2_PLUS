import argparse
import time

import numpy as np

from benchmark_smpl_fast import make_retarget, make_sequence
from general_motion_retargeting.SmpleXConverter import SMPLXConverter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    converter = SMPLXConverter()
    retarget_one = make_retarget(1)
    retarget_two = make_retarget(2)
    q_one = []
    q_two = []
    convert_ms = []
    one_ms = []
    two_ms = []

    sequence = make_sequence(args.count)
    converter.convert_axis_angle_to_human_data_fast(*sequence[0])
    for axis_angles, root_translation in sequence:
        start = time.perf_counter()
        human_data = converter.convert_axis_angle_to_human_data_fast(
            axis_angles, root_translation
        )
        convert_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        q_one.append(retarget_one.retarget(human_data).copy())
        one_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        q_two.append(retarget_two.retarget(human_data).copy())
        two_ms.append((time.perf_counter() - start) * 1000.0)

    delta = np.abs(np.asarray(q_one) - np.asarray(q_two))
    print(
        "convert_ms median={:.2f} p95={:.2f} gmr1_ms median={:.2f} p95={:.2f} "
        "gmr2_ms median={:.2f} p95={:.2f}".format(
            np.median(convert_ms),
            np.percentile(convert_ms, 95),
            np.median(one_ms),
            np.percentile(one_ms, 95),
            np.median(two_ms),
            np.percentile(two_ms, 95),
        )
    )
    print(
        "iter_delta median={:.6f} p95={:.6f} max={:.6f}".format(
            np.median(delta), np.percentile(delta, 95), np.max(delta)
        )
    )


if __name__ == "__main__":
    main()
