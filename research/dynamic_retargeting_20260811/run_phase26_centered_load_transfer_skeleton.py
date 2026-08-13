#!/usr/bin/env python3
"""Phase25 with the single preregistered native-center unloading states inserted."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p25", HERE / "run_phase25_load_transfer_skeleton.py")
p25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p25)

p25.STEPS = (
    ("DS_CENTER", ("left", "right"), (), "center"),
    ("DS_LOAD_LEFT", ("left", "right"), (), "left"),
    ("L_SUPPORT_R_SWING", ("left",), ("right",), "left"),
    ("DS_R_TOUCHDOWN_LEFT_LOADED", ("left", "right"), (), "left"),
    ("DS_UNLOAD_LEFT_TO_CENTER", ("left", "right"), (), "center"),
    ("DS_LOAD_RIGHT", ("left", "right"), (), "right"),
    ("R_SUPPORT_L_SWING", ("right",), ("left",), "right"),
    ("DS_L_TOUCHDOWN_RIGHT_LOADED", ("left", "right"), (), "right"),
    ("DS_UNLOAD_RIGHT_TO_CENTER", ("left", "right"), (), "center"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="phase26-") as directory:
        inner = Path(directory) / "inner.json"
        old_argv = sys.argv
        try:
            sys.argv = [
                str(HERE / "run_phase25_load_transfer_skeleton.py"),
                "--scene", str(args.scene), "--boundary", str(args.boundary),
                "--output", str(inner),
            ]
            p25.main()
        finally:
            sys.argv = old_argv
        result = json.loads(inner.read_text())
    complete = len(result["steps"]) == len(p25.STEPS) and all(step["success"] for step in result["steps"])
    result["stage"] = "Phase26 centered quasi-static load-transfer skeleton"
    result["provenance"] = {
        "base_implementation": "Phase25",
        "only_change": "insert native-center unload states; all numerical contracts unchanged",
    }
    result["decision"] = {
        "centered_load_transfer_skeleton_complete": complete,
        "dynamics_truth": False,
        "physics_unlocked": False,
        "training_unlocked": False,
        "next": "time-parameterize COM transfer before force feasibility" if complete else "stop without alternate configuration",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"execution": result["execution"], "steps": result["steps"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
