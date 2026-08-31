#!/usr/bin/env python3
"""Apply a single, auditable motion-distribution intervention for X2-Sonic.

The first intervention is a causal per-joint velocity limiter.  It changes only
the commanded joint-position trajectory; root translation and root orientation
are copied bit-for-bit.  This tests the hypothesis suggested by the stratified
benchmark: PHUMA joint velocities outside the policy's stable support are a
dominant failure source.  Each output is a new artifact; inputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np


DEFAULT_ANGLES_MJ = np.asarray(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
     0.0, 0.0], dtype=np.float64)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def limit_joint_speed(joint_pos: np.ndarray, fps: float, max_speed: float) -> np.ndarray:
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 31:
        raise ValueError(f"expected (T,31) joint_pos, got {joint_pos.shape}")
    if fps <= 0 or max_speed <= 0:
        raise ValueError("fps and max_speed must be positive")
    out = np.empty_like(joint_pos, dtype=np.float64)
    out[0] = joint_pos[0]
    max_step = float(max_speed) / float(fps)
    for i in range(1, len(joint_pos)):
        delta = joint_pos[i] - out[i - 1]
        out[i] = out[i - 1] + np.clip(delta, -max_step, max_step)
    return out


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def yaw_quat(q):
    q = quat_normalize(q)
    yaw = np.arctan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def quat_slerp(a, b, t):
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))
    sign = -1.0 if dot < 0 else 1.0
    dot = abs(dot)
    if dot > 1.0 - 1e-8:
        return quat_normalize((1.0 - t) * a + sign * t * b)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return quat_normalize(
        np.sin((1.0 - t) * theta) / np.sin(theta) * a
        + sign * np.sin(t * theta) / np.sin(theta) * b
    )


def scale_root_tilt(root_quat: np.ndarray, scale: float) -> np.ndarray:
    if not 0.0 <= scale <= 1.0:
        raise ValueError("root tilt scale must be in [0,1]")
    return np.asarray([quat_slerp(yaw_quat(q), q, scale) for q in root_quat], dtype=np.float64)


def adapt_motion(joint_pos, root_pos, fps, mode, parameter):
    if mode == "joint_velocity_limit":
        return limit_joint_speed(joint_pos, fps, parameter), root_pos.copy()
    if mode == "pose_scale":
        if not 0.0 < parameter <= 1.0:
            raise ValueError("pose scale must be in (0,1]")
        return DEFAULT_ANGLES_MJ + parameter * (joint_pos - DEFAULT_ANGLES_MJ), root_pos.copy()
    if mode == "root_xy_scale":
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("root XY scale must be in [0,1]")
        out = root_pos.copy()
        out[:, :2] = root_pos[0, :2] + parameter * (root_pos[:, :2] - root_pos[0, :2])
        return joint_pos.copy(), out
    if mode == "root_tilt_scale":
        raise RuntimeError("root_tilt_scale is applied to root_quat, not root_pos")
    raise ValueError(f"unknown adapter mode: {mode}")


def stats(jp: np.ndarray, fps: float) -> dict[str, float]:
    speed = np.abs(np.diff(jp, axis=0)) * fps if len(jp) > 1 else np.zeros_like(jp)
    return {
        "frames": int(len(jp)),
        "joint_speed_p95_radps": float(np.quantile(speed, 0.95)),
        "joint_speed_max_radps": float(np.max(speed)),
        "joint_abs_angle_p99_rad": float(np.quantile(np.abs(jp), 0.99)),
        "joint_abs_angle_max_rad": float(np.max(np.abs(jp))),
        "finite": bool(np.all(np.isfinite(jp))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=pathlib.Path, required=True)
    ap.add_argument("--source-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--output-dir", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--mode", choices=["joint_velocity_limit", "pose_scale", "root_xy_scale", "root_tilt_scale"], default="joint_velocity_limit")
    ap.add_argument("--parameter", type=float)
    ap.add_argument("--joint-speed-limit", type=float, help="Backward-compatible alias for --mode joint_velocity_limit --parameter")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.parameter is None:
        args.parameter = args.joint_speed_limit
    if args.parameter is None:
        raise ValueError("provide --parameter (or --joint-speed-limit)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to write non-empty output dir: {args.output_dir}")

    source_manifest = json.loads(args.source_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_record in source_manifest["records"]:
        inp = pathlib.Path(source_record["output"])
        out = args.output_dir / inp.name
        payload = json.loads(inp.read_text())
        fps = float(payload["fps"])
        jp = np.asarray(payload["joint_pos"], dtype=np.float64)
        rp = np.asarray(payload["root_pos"], dtype=np.float64)
        rq = np.asarray(payload["root_quat"], dtype=np.float64)
        before = stats(jp, fps)
        if args.mode == "root_tilt_scale":
            adapted, adapted_root = jp.copy(), rp.copy()
            adapted_quat = scale_root_tilt(rq, args.parameter)
        else:
            adapted, adapted_root = adapt_motion(jp, rp, fps, args.mode, args.parameter)
            adapted_quat = rq.copy()
        after = stats(adapted, fps)
        if not after["finite"] or not np.all(np.isfinite(adapted_root)):
            raise RuntimeError(f"non-finite or malformed adapted motion: {inp}")
        out_payload = dict(payload)
        out_payload["joint_pos"] = adapted.tolist()
        out_payload["root_pos"] = adapted_root.tolist()
        out_payload["root_quat"] = adapted_quat.tolist()
        out_payload["display"] = f"{payload.get('display', inp.stem)}__{args.mode}_{args.parameter:g}"
        if out.exists() and not args.overwrite:
            raise FileExistsError(out)
        out.write_text(json.dumps(out_payload, ensure_ascii=False, allow_nan=False) + "\n")
        records.append({
            "source": source_record["source"],
            "source_sha256": source_record["source_sha256"],
            "input": str(inp.resolve()),
            "input_sha256": sha256(inp),
            "output": str(out.resolve()),
            "output_sha256": sha256(out),
            "transform": {
                "name": args.mode,
                "mode": args.mode,
                "parameter": float(args.parameter),
                "root_position_changed": args.mode == "root_xy_scale",
                "root_orientation_changed": args.mode == "root_tilt_scale",
            },
            "before_stats": before,
            "after_stats": after,
        })

    index = [
        {"id": pathlib.Path(r["output"]).stem, "display": pathlib.Path(r["output"]).stem,
         "frames": r["after_stats"]["frames"], "fps": 50.0, "group": "PHUMA adapted"}
        for r in records
    ]
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema": "x2_sonic_motion_distribution_adapter_manifest/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapter": args.mode,
        "parameter": float(args.parameter),
        "source_manifest": str(args.source_manifest.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "index": str(index_path.resolve()),
        "index_sha256": sha256(index_path),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"motions": len(records), "mode": args.mode, "parameter": args.parameter,
                      "manifest": str(args.manifest.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
