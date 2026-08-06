#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcpeft_motion_composition import compose_gait_with_upper  # noqa: E402


def load_single(path: Path) -> tuple[str, dict]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"expected one motion per split file: {path}")
    return next(iter(payload.items()))


def joint_ranges(path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        name = joint.get("name")
        raw_range = joint.get("range")
        if name and raw_range:
            low, high = (float(value) for value in raw_range.split())
            result[name] = (low, high)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait-dir", required=True, type=Path)
    parser.add_argument("--upper-dir", required=True, type=Path)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.60)
    args = parser.parse_args()

    gait_files = sorted(path for path in args.gait_dir.glob("*.pkl") if path.name != "metadata.pkl")
    upper_files = sorted(path for path in args.upper_dir.glob("*.pkl") if path.name != "metadata.pkl")
    if not gait_files or not upper_files:
        raise SystemExit("gait and upper directories must both contain split motion files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranges = joint_ranges(args.mjcf)
    metadata: dict[str, dict] = {}
    for gait_file in gait_files:
        gait_name, gait = load_single(gait_file)
        for upper_file in upper_files:
            upper_name, upper = load_single(upper_file)
            motion_name = f"compose__{gait_name}__upper__{upper_name}"
            composed, report = compose_gait_with_upper(
                gait,
                upper,
                alpha=args.alpha,
                joint_ranges=ranges,
            )
            composed["composition_base_motion"] = gait_name
            composed["composition_upper_motion"] = upper_name
            composed["composition_alpha"] = args.alpha
            joblib.dump({motion_name: composed}, args.output_dir / f"{motion_name}.pkl", compress=3)
            metadata[motion_name] = {
                "fps": float(composed["fps"]),
                "length": len(composed["dof"]),
                "duration_s": len(composed["dof"]) / float(composed["fps"]),
                "base_motion": gait_name,
                "upper_motion": upper_name,
                **report,
            }
    joblib.dump(metadata, args.output_dir / "metadata.pkl", compress=3)
    (args.output_dir / "composition_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "conditional upper-body-during-gait evaluation only",
                "training_authorized": False,
                "gait_dir": str(args.gait_dir),
                "upper_dir": str(args.upper_dir),
                "mjcf": str(args.mjcf),
                "alpha": args.alpha,
                "motions": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"built={len(metadata)} output={args.output_dir}")


if __name__ == "__main__":
    main()
