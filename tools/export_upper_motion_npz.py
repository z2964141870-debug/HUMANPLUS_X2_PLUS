#!/usr/bin/env python3
"""Export a trusted X2 motion-cache upper-body track to a portable NPZ.

The official AimDK container intentionally does not carry joblib.  This tool
performs the one-time conversion outside the container and stores only the
14 arm joints required by the deploy-time upper-body contract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cwi_x2.upper_motion_contract import load_upper_motion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    clip = load_upper_motion(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        q_rad=clip.q_rad,
        fps=np.asarray(clip.fps, dtype=np.float64),
        joint_names=np.asarray(clip.joint_names),
        source=np.asarray(str(clip.path)),
        key=np.asarray(clip.key),
    )
    print(f"{output}: {clip.q_rad.shape}, {clip.fps:.3f} Hz, {clip.duration_s:.3f} s")


if __name__ == "__main__":
    main()
