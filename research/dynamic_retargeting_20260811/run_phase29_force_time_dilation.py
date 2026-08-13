#!/usr/bin/env python3
"""Repeat Phase28 with the sole preregistered variable dt 0.02 -> 0.04 s."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p28", HERE / "run_phase28_left_half_centroidal_force.py")
p28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p28)
p28.DT = 0.04


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    with tempfile.TemporaryDirectory(prefix="phase29-") as directory:
        inner = Path(directory) / "inner.json"
        old_argv = sys.argv
        try:
            sys.argv = [
                str(HERE / "run_phase28_left_half_centroidal_force.py"),
                "--scene", str(args.scene), "--boundary", str(args.boundary),
                "--path", str(args.path), "--output", str(inner),
            ]
            p28.main()
        finally:
            sys.argv = old_argv
        candidate = json.loads(inner.read_text())
    evaluated = candidate["execution"]["evaluated_frames"]
    passed = candidate["summary"]["jointly_feasible_frames"] == evaluated
    result = {
        "stage": "Phase29 centroidal-force time-dilation A/B",
        "intervention": {"time_scale": 2.0, "dt_s": 0.04, "only_variable": "time"},
        "control": {
            "dt_s": 0.02,
            "summary": control["summary"],
            "evaluated_frames": control["execution"]["evaluated_frames"],
        },
        "candidate": candidate,
        "decision": {
            "time_dilation_force_preflight_pass": passed,
            "physics_unlocked": False, "training_unlocked": False,
            "next": "review before resampling/physics" if passed else "stop time-dilation route; require force-aware trajectory generation",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "control": result["control"],
        "candidate": {"execution": candidate["execution"], "summary": candidate["summary"]},
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
