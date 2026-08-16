#!/usr/bin/env python3
"""Apply a phase-aware root-tilt allowance to X2-Sonic motion JSON files.

This is an offline input-distribution probe, not a policy or robot controller.
The source motion is expected to already contain any joint-pose adaptation (for
example ``pose_scale=0.5``).  Root roll/pitch is retained up to ``max-scale``
when the requested root height is supported by a nominal standing phase.  The
allowance is smoothly reduced while the reference root is low or descending.
The transform is deliberately explicit and logs the per-frame scale so that
retention and stability can be evaluated together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("root quaternion must be finite and non-zero")
    return q / norm


def yaw_quat(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def slerp(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    a, b = normalize(a), normalize(b)
    dot = float(np.dot(a, b))
    sign = -1.0 if dot < 0.0 else 1.0
    dot = abs(dot)
    if dot > 1.0 - 1.0e-8:
        return normalize((1.0 - amount) * a + sign * amount * b)
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    denom = math.sin(theta)
    return normalize(
        math.sin((1.0 - amount) * theta) / denom * a
        + sign * math.sin(amount * theta) / denom * b
    )


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def allowance(
    root_pos: np.ndarray,
    fps: float,
    max_scale: float,
    height_low: float,
    height_high: float,
    descent_low: float,
    descent_high: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"expected root_pos shape (T,3), got {root_pos.shape}")
    if fps <= 0.0 or not (0.0 <= max_scale <= 1.0):
        raise ValueError("fps must be positive and max_scale must be in [0,1]")
    if not height_low < height_high or not descent_low < descent_high:
        raise ValueError("gate bounds must be strictly increasing")
    vz = np.gradient(root_pos[:, 2], 1.0 / fps)
    height_gate = smoothstep((root_pos[:, 2] - height_low) / (height_high - height_low))
    descent_gate = smoothstep((vz - descent_low) / (descent_high - descent_low))
    scales = max_scale * height_gate * descent_gate
    stats = {
        "frames": int(len(scales)),
        "scale_mean": float(np.mean(scales)),
        "scale_p05": float(np.quantile(scales, 0.05)),
        "scale_p50": float(np.quantile(scales, 0.50)),
        "scale_p95": float(np.quantile(scales, 0.95)),
        "scale_max": float(np.max(scales)),
        "fraction_zero": float(np.mean(scales <= 1.0e-8)),
        "fraction_below_max": float(np.mean(scales < max_scale - 1.0e-8)),
        "root_z_min_m": float(np.min(root_pos[:, 2])),
        "root_vz_min_mps": float(np.min(vz)),
        "root_vz_max_mps": float(np.max(vz)),
    }
    return scales, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-scale", type=float, default=0.1)
    parser.add_argument("--height-low", type=float, default=0.50)
    parser.add_argument("--height-high", type=float, default=0.58)
    parser.add_argument("--descent-low", type=float, default=-0.05)
    parser.add_argument("--descent-high", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to write non-empty output dir: {args.output_dir}")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_record in source_manifest["records"]:
        source = Path(source_record["output"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
        root_quat = np.asarray(payload["root_quat"], dtype=np.float64)
        fps = float(payload["fps"])
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 31:
            raise ValueError(f"{source}: expected joint_pos (T,31), got {joint_pos.shape}")
        if root_pos.shape != (len(joint_pos), 3) or root_quat.shape != (len(joint_pos), 4):
            raise ValueError(f"{source}: root arrays do not match joint_pos length")
        scales, scale_stats = allowance(
            root_pos, fps, args.max_scale, args.height_low, args.height_high,
            args.descent_low, args.descent_high,
        )
        adapted_quat = np.asarray(
            [slerp(yaw_quat(q), q, float(scale)) for q, scale in zip(root_quat, scales)],
            dtype=np.float64,
        )
        out = args.output_dir / source.name
        if out.exists() and not args.overwrite:
            raise FileExistsError(out)
        output_payload = dict(payload)
        output_payload["root_quat"] = adapted_quat.tolist()
        output_payload["display"] = f"{payload.get('display', source.stem)}__phase_root_tilt"
        out.write_text(json.dumps(output_payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        records.append({
            "source": source_record["source"],
            "source_sha256": source_record["source_sha256"],
            "input": str(source.resolve()),
            "input_sha256": sha256(source),
            "output": str(out.resolve()),
            "output_sha256": sha256(out),
            "transform": {
                "name": "phase_aware_root_tilt",
                "max_scale": float(args.max_scale),
                "height_low_m": float(args.height_low),
                "height_high_m": float(args.height_high),
                "descent_low_mps": float(args.descent_low),
                "descent_high_mps": float(args.descent_high),
                "root_position_changed": False,
                "root_orientation_changed": True,
            },
            "scale_stats": scale_stats,
        })

    index = [
        {"id": Path(row["output"]).stem, "display": Path(row["output"]).stem,
         "frames": row["scale_stats"]["frames"], "fps": 50.0, "group": "PHUMA phase-aware adapted"}
        for row in records
    ]
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "x2_sonic_phase_aware_root_tilt_manifest_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest": str(args.source_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "index": str(index_path.resolve()),
        "index_sha256": sha256(index_path),
        "parameters": {
            "max_scale": float(args.max_scale),
            "height_low_m": float(args.height_low),
            "height_high_m": float(args.height_high),
            "descent_low_mps": float(args.descent_low),
            "descent_high_mps": float(args.descent_high),
            "definition": "reference root height/descent gates; no robot state or contact force is used",
        },
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"motions": len(records), "manifest": str(args.manifest.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
