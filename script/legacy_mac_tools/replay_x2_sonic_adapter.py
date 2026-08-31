#!/usr/bin/env python3
"""Replay a canonical X2-Sonic input stream through the offline adapter.

The file format is intentionally independent of BLE, ROS, or a particular
garment firmware. A later garment parser only needs to emit one JSON object
per frame with timestamp_s, a 31-element canonical joint_pos vector, and a
scalar-first root_quat. This keeps the current research boundary testable
without touching a robot or guessing a proprietary packet layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile
import time
from typing import Any

import numpy as np

try:
    from x2_sonic_input_adapter import X2SonicAdapterConfig, X2SonicInputAdapter
except ImportError:  # pragma: no cover - supports direct package execution
    from .x2_sonic_input_adapter import X2SonicAdapterConfig, X2SonicInputAdapter


SCHEMA = "x2_sonic_garment_replay_jsonl_v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value: Any, size: int, field: str, line_no: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_no}: {field} is not numeric") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"line {line_no}: {field} must be finite shape ({size},), got {array.shape}")
    return array


def _parse_record(raw: Any, line_no: int, default_fps: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_no}: expected a JSON object")
    try:
        timestamp_s = float(raw["timestamp_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line_no}: timestamp_s is required and must be numeric") from exc
    if not math.isfinite(timestamp_s):
        raise ValueError(f"line {line_no}: timestamp_s must be finite")
    joint_pos = _finite_vector(raw.get("joint_pos"), 31, "joint_pos", line_no)
    root_quat = _finite_vector(raw.get("root_quat"), 4, "root_quat", line_no)
    if float(np.linalg.norm(root_quat)) <= 1e-12:
        raise ValueError(f"line {line_no}: root_quat has zero norm")
    try:
        fps = float(raw.get("fps", default_fps))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_no}: fps must be numeric") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"line {line_no}: fps must be positive and finite")
    return timestamp_s, joint_pos, root_quat, fps


def _atomic_write_text(path: pathlib.Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def replay_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    manifest_path: pathlib.Path,
    config: X2SonicAdapterConfig,
    default_fps: float = 50.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay JSONL and write adapted JSONL plus an auditable manifest."""

    if default_fps <= 0.0 or not math.isfinite(default_fps):
        raise ValueError("default_fps must be positive and finite")
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    manifest_path = pathlib.Path(manifest_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite")

    adapter = X2SonicInputAdapter(config)
    output_lines: list[str] = []
    timestamps: list[float] = []
    raw_tilts: list[float] = []
    adapted_tilts: list[float] = []
    rate_limited = 0
    previous_timestamp: float | None = None

    with input_path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            timestamp_s, joint_pos, root_quat, fps = _parse_record(raw, line_no, default_fps)
            if previous_timestamp is not None and timestamp_s < previous_timestamp - 1e-12:
                raise ValueError(
                    f"line {line_no}: timestamp_s is not monotonic "
                    f"({timestamp_s} < {previous_timestamp})"
                )
            previous_timestamp = timestamp_s
            adapted_joints, adapted_quat, telemetry = adapter.process(joint_pos, root_quat, fps)
            output_lines.append(json.dumps({
                "input_line": line_no,
                "timestamp_s": timestamp_s,
                "fps": fps,
                "joint_pos": adapted_joints.tolist(),
                "root_quat": adapted_quat.tolist(),
                "telemetry": telemetry,
            }, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            timestamps.append(timestamp_s)
            raw_tilts.append(float(telemetry["raw_root_tilt_rad"]))
            adapted_tilts.append(float(telemetry["adapted_root_tilt_rad"]))
            rate_limited += int(bool(telemetry["joint_rate_limited"]))

    if not output_lines:
        raise ValueError(f"input contains no non-empty JSONL records: {input_path}")

    output_text = "".join(output_lines)
    _atomic_write_text(output_path, output_text, overwrite=overwrite)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline_only": True,
        "hardware_touched": False,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": sha256(input_path),
            "fields": {
                "timestamp_s": "monotonic seconds",
                "joint_pos": "31 canonical X2 joint positions, radians",
                "root_quat": "4-element scalar-first quaternion [w,x,y,z]",
                "fps": "optional positive per-frame rate; defaults to CLI value",
            },
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "records": len(output_lines),
            "first_timestamp_s": timestamps[0],
            "last_timestamp_s": timestamps[-1],
        },
        "adapter": {
            "root_tilt_scale": float(config.root_tilt_scale),
            "pose_scale": float(config.pose_scale),
            "joint_speed_limit_radps": (
                None if config.joint_speed_limit_radps is None
                else float(config.joint_speed_limit_radps)
            ),
        },
        "telemetry_summary": {
            "raw_root_tilt_p95_rad": float(np.quantile(raw_tilts, 0.95)),
            "adapted_root_tilt_p95_rad": float(np.quantile(adapted_tilts, 0.95)),
            "rate_limited_records": rate_limited,
            "finite": True,
        },
        "garment_mapping": {
            "status": "pending",
            "note": "Map the supplied garment log fields to this canonical schema before replay; no BLE packet assumption is made here.",
        },
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        overwrite=overwrite,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True, help="canonical JSONL input")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="adapted JSONL output")
    parser.add_argument("--manifest", type=pathlib.Path, required=True, help="manifest output")
    parser.add_argument("--default-fps", type=float, default=50.0)
    parser.add_argument("--root-tilt-scale", type=float, default=0.0)
    parser.add_argument("--pose-scale", type=float, default=0.5)
    parser.add_argument("--joint-speed-limit-radps", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = X2SonicAdapterConfig(
        root_tilt_scale=args.root_tilt_scale,
        pose_scale=args.pose_scale,
        joint_speed_limit_radps=args.joint_speed_limit_radps,
    )
    manifest = replay_file(
        args.input,
        args.output,
        args.manifest,
        config=config,
        default_fps=args.default_fps,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "records": manifest["output"]["records"],
        "output": manifest["output"]["path"],
        "manifest": str(args.manifest.resolve()),
        "telemetry_summary": manifest["telemetry_summary"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
