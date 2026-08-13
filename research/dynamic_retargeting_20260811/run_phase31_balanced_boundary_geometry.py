#!/usr/bin/env python3
"""Run frozen Phase26 geometry from the Phase43 balanced native boundary."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p26", HERE / "run_phase26_centered_load_transfer_skeleton.py")
p26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p26)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="phase31-") as directory:
        inner = Path(directory) / "inner.json"
        old_argv = sys.argv
        try:
            sys.argv = [
                str(HERE / "run_phase26_centered_load_transfer_skeleton.py"),
                "--scene", str(args.scene), "--boundary", str(args.boundary),
                "--output", str(inner),
            ]
            p26.main()
        finally:
            sys.argv = old_argv
        result = json.loads(inner.read_text())
    complete = len(result["steps"]) == 9 and all(step["success"] for step in result["steps"])
    result["stage"] = "Phase31 balanced-native centered load-transfer skeleton"
    result["provenance"] = {"base_implementation": "Phase26", "only_variable": "Phase43 balanced native boundary"}
    result["decision"] = {
        "balanced_boundary_geometry_complete": complete,
        "dynamics_truth": False, "physics_unlocked": False, "training_unlocked": False,
        "next": "build continuous/force preflight from this boundary" if complete else "stop without alternate configuration",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"execution": result["execution"], "steps": result["steps"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
