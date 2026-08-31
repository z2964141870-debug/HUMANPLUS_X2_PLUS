#!/usr/bin/env python3
"""Convert PHUMA-X2 joblib motions to the public SONIC web-bank JSON contract.

This is an offline data adapter.  It does not run a policy, start MuJoCo, or
touch a robot.  The output intentionally contains the pre-stand-blend motion;
the public evaluator applies its one-second stand blend at runtime.  Keeping
that boundary explicit prevents accidental double blending when comparing a
source ``.pkl`` with its canonical ``.json`` representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Iterable

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_official_sonic_x2 as sonic  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(motion: sonic.Motion) -> dict[str, object]:
    return {
        "frames": int(motion.frames),
        "fps": float(motion.fps),
        "duration_s": float((motion.frames - 1) / motion.fps),
        "root_xy_start_m": motion.root_pos[0, :2].astype(float).tolist(),
        "root_z_min_m": float(np.min(motion.root_pos[:, 2])),
        "root_z_max_m": float(np.max(motion.root_pos[:, 2])),
        "joint_abs_p99_rad": float(np.quantile(np.abs(motion.joint_pos), 0.99)),
        "joint_abs_max_rad": float(np.max(np.abs(motion.joint_pos))),
        "finite": bool(
            np.all(np.isfinite(motion.joint_pos))
            and np.all(np.isfinite(motion.root_pos))
            and np.all(np.isfinite(motion.root_quat))
        ),
    }


def _resample(motion: sonic.Motion, target_fps: float) -> sonic.Motion:
    if target_fps <= 0:
        raise ValueError(f"target fps must be positive, got {target_fps}")
    if abs(float(motion.fps) - target_fps) < 1e-9:
        return motion

    duration = (motion.frames - 1) / float(motion.fps)
    count = max(2, int(round(duration * target_fps)) + 1)
    target_times = np.arange(count, dtype=np.float64) / target_fps
    source_position = target_times * float(motion.fps)
    lo = np.floor(source_position).astype(np.int64)
    hi = np.minimum(lo + 1, motion.frames - 1)
    alpha = (source_position - lo).astype(np.float64)

    joint_pos = (1.0 - alpha[:, None, None]) * motion.joint_pos[lo]
    joint_pos += alpha[:, None, None] * motion.joint_pos[hi]
    root_pos = (1.0 - alpha[:, None]) * motion.root_pos[lo]
    root_pos += alpha[:, None] * motion.root_pos[hi]
    root_quat = np.asarray(
        [sonic.slerp(motion.root_quat[a], motion.root_quat[b], float(t))
         for a, b, t in zip(lo, hi, alpha)],
        dtype=np.float64,
    )
    return sonic.Motion(
        motion.name,
        float(target_fps),
        joint_pos.astype(np.float64),
        root_pos.astype(np.float64),
        root_quat,
    )


def canonicalize(
    source: pathlib.Path,
    output: pathlib.Path,
    target_fps: float,
    normalize_heading: bool,
    overwrite: bool,
) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source_motion = sonic._load_joblib_motion(source)
    if not source_motion.frames:
        raise ValueError(f"empty motion: {source}")
    if not np.all(np.isfinite(source_motion.joint_pos)):
        raise ValueError(f"non-finite joint position in {source}")
    if normalize_heading:
        source_motion = sonic.normalize_heading(source_motion)
    canonical_motion = _resample(source_motion, target_fps)
    if not _stats(canonical_motion)["finite"]:
        raise ValueError(f"non-finite canonical motion: {source}")

    payload = {
        "name": canonical_motion.name,
        "display": canonical_motion.name,
        "fps": float(canonical_motion.fps),
        "frames": int(canonical_motion.frames),
        "root_pos": canonical_motion.root_pos.tolist(),
        "root_quat": canonical_motion.root_quat.tolist(),
        "joint_pos": canonical_motion.joint_pos.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "transform": {
            "normalize_heading": bool(normalize_heading),
            "target_fps": float(target_fps),
            "stand_blend_included": False,
            "joint_order": "MuJoCo JOINT_NAMES from eval_official_sonic_x2.py",
        },
        "source_stats": _stats(sonic._load_joblib_motion(source)),
        "canonical_stats": _stats(canonical_motion),
    }


def _iter_inputs(args: argparse.Namespace) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = [pathlib.Path(p) for p in args.input]
    if args.selection_report:
        report = json.loads(args.selection_report.read_text(encoding="utf-8"))
        paths.extend(pathlib.Path(p) for p in report.get("selected_files", []))
    if args.input_dir:
        iterator = args.input_dir.rglob("*.pkl") if args.recursive else args.input_dir.glob("*.pkl")
        paths.extend(iterator)
    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    unique.sort(key=lambda p: str(p))
    if args.limit is not None:
        unique = unique[: args.limit]
    if not unique:
        raise ValueError("no input motions selected")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--input-dir", type=pathlib.Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--selection-report", type=pathlib.Path,
                        help="Reuse selected_files from an evaluator report.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--normalize-heading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parent-prefix", action=argparse.BooleanOptionalAction, default=True,
                        help="Prefix output names with the source subset directory (default: on).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = _iter_inputs(args)
    records = []
    for index, source in enumerate(inputs, 1):
        prefix = f"{source.parent.name}__" if args.parent_prefix else ""
        output = args.output_dir / f"{prefix}{source.stem}.json"
        print(f"[{index}/{len(inputs)}] {source.name} -> {output.name}", flush=True)
        records.append(canonicalize(
            source=source,
            output=output,
            target_fps=args.target_fps,
            normalize_heading=args.normalize_heading,
            overwrite=args.overwrite,
        ))

    index_entries = [
        {
            "id": pathlib.Path(record["output"]).stem,
            "display": pathlib.Path(record["output"]).stem,
            "frames": int(record["canonical_stats"]["frames"]),
            "fps": float(record["canonical_stats"]["fps"]),
            "group": "PHUMA canonical",
        }
        for record in records
    ]
    index_path = args.output_dir / "index.json"
    if index_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing index: {index_path}")
    index_path.write_text(json.dumps(index_entries, ensure_ascii=False, indent=2) + "\n")

    manifest = {
        "schema": "x2_sonic_canonical_motion_manifest/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapter": str(pathlib.Path(__file__).resolve()),
        "adapter_source_sha256": sha256(pathlib.Path(__file__).resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "index": str(index_path.resolve()),
        "index_sha256": sha256(index_path),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"motions": len(records), "manifest": str(args.manifest.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
