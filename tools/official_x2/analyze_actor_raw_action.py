#!/usr/bin/env python3
"""Replay saved actor observations to separate raw-policy and issued-action saturation."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_pitch", "waist_roll",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rollout-glob", required=True)
    parser.add_argument("--stage", default="move")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = [Path(item) for item in sorted(glob.glob(args.rollout_glob))]
    if not paths:
        parser.error(f"no rollout matched {args.rollout_glob!r}")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    raw_groups: list[np.ndarray] = []
    issued_groups: list[np.ndarray] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [row for row in payload["trace"] if row["stage"] == args.stage]
        observations = np.asarray([row["obs"] for row in rows], dtype=np.float32)
        raw_groups.append(session.run(["actions"], {"obs": observations})[0])
        issued_groups.append(np.asarray([row["action"] for row in rows], dtype=np.float32))
    raw = np.concatenate(raw_groups)
    issued = np.concatenate(issued_groups)

    def metrics(values: np.ndarray) -> dict[str, float]:
        return {
            "abs_mean": float(np.mean(np.abs(values))),
            "abs_p95": float(np.quantile(np.abs(values), 0.95)),
            "abs_max": float(np.max(np.abs(values))),
            "fraction_abs_ge_0p999": float(np.mean(np.abs(values) >= 0.999)),
            "fraction_abs_ge_1": float(np.mean(np.abs(values) >= 1.0)),
        }

    result = {
        "model": str(args.model.resolve()),
        "rollouts": [str(path.resolve()) for path in paths],
        "stage": args.stage,
        "samples": int(raw.shape[0]),
        "raw_policy": metrics(raw),
        "issued_action": metrics(issued),
        "per_joint": {
            name: {
                "raw_policy": metrics(raw[:, index]),
                "issued_action": metrics(issued[:, index]),
            }
            for index, name in enumerate(JOINT_NAMES)
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
