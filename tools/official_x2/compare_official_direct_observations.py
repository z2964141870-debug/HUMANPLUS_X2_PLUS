#!/usr/bin/env python3
"""Compare aligned observation/action groups in official and direct rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "base_lin_vel": (0, 3), "base_ang_vel": (3, 6),
    "projected_gravity": (6, 9), "command": (9, 12),
    "joint_pos": (12, 43), "joint_vel": (43, 74),
    "previous_action": (74, 89), "phase": (89, 93),
}


def metrics(a: list[dict], b: list[dict], indices: list[int]) -> dict[str, float]:
    result = {}
    for name, (start, end) in GROUPS.items():
        delta = np.concatenate([
            np.asarray(a[i]["obs"])[start:end] - np.asarray(b[i]["obs"])[start:end]
            for i in indices
        ])
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(delta**2)))
    action_delta = np.concatenate([
        np.asarray(a[i]["action"]) - np.asarray(b[i]["action"]) for i in indices
    ])
    result["action_rmse"] = float(np.sqrt(np.mean(action_delta**2)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = json.loads(args.official.read_text(encoding="utf-8"))["trace"]
    b = json.loads(args.direct.read_text(encoding="utf-8"))["trace"]
    if len(a) != len(b):
        raise RuntimeError(f"trace length mismatch: {len(a)} vs {len(b)}")
    stand = [i for i in range(len(a)) if a[i]["stage"] == b[i]["stage"] == "stand"]
    move_early = [
        i for i in range(len(a))
        if a[i]["stage"] == b[i]["stage"] == "move" and a[i]["elapsed_s"] <= 0.5
    ]
    result = {
        "official": str(args.official.resolve()), "direct": str(args.direct.resolve()),
        "stand_rows": len(stand), "stand": metrics(a, b, stand),
        "move_first_0p5s_rows": len(move_early),
        "move_first_0p5s": metrics(a, b, move_early),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
