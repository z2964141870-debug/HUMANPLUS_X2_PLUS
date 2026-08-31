#!/usr/bin/env python3
"""Convert an X2 motion-lib PKL into the compact X2M2 deploy format."""

import argparse
import hashlib
import struct
from pathlib import Path

import joblib
import numpy as np


MAGIC = 0x58324D32
NUM_DOFS = 31


def _select_clip(payload, requested):
    if not isinstance(payload, dict) or not payload:
        raise ValueError("motion PKL must contain a non-empty clip mapping")
    if requested:
        if requested not in payload:
            raise ValueError(
                f"clip {requested!r} not found; available: {', '.join(payload)}"
            )
        return requested, payload[requested]
    if len(payload) != 1:
        raise ValueError(
            "motion PKL contains multiple clips; select one with --clip: "
            + ", ".join(payload)
        )
    name = next(iter(payload))
    return name, payload[name]


def export_motion(source, destination, clip_name=None):
    payload = joblib.load(source)
    selected_name, clip = _select_clip(payload, clip_name)
    dof = np.asarray(clip["dof"], dtype=np.float64)
    root_rot = np.asarray(clip["root_rot"], dtype=np.float64)
    fps = float(clip["fps"])

    if dof.ndim != 2 or dof.shape[1] != NUM_DOFS:
        raise ValueError(f"dof must have shape (frames, {NUM_DOFS}), got {dof.shape}")
    if root_rot.shape != (dof.shape[0], 4):
        raise ValueError(
            f"root_rot must have shape ({dof.shape[0]}, 4), got {root_rot.shape}"
        )
    if dof.shape[0] == 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid frame count/fps: frames={dof.shape[0]} fps={fps}")
    if not np.isfinite(dof).all() or not np.isfinite(root_rot).all():
        raise ValueError("motion contains NaN or infinity")

    quat_norm = np.linalg.norm(root_rot, axis=1)
    if np.max(np.abs(quat_norm - 1.0)) > 1e-3:
        raise ValueError(
            "root_rot is not normalized xyzw quaternion data: "
            f"max norm error={np.max(np.abs(quat_norm - 1.0)):.3g}"
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        stream.write(struct.pack("<IIId", MAGIC, dof.shape[0], NUM_DOFS, fps))
        for joint_pos, quat_xyzw in zip(dof, root_rot):
            stream.write(struct.pack("<31d4d", *joint_pos, *quat_xyzw))

    expected_size = struct.calcsize("<IIId") + dof.shape[0] * struct.calcsize("<31d4d")
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"short X2M2 write: expected {expected_size}, got {actual_size}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return selected_name, dof.shape[0], fps, actual_size, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, help="Input motion-lib PKL")
    parser.add_argument("--out", dest="destination", required=True, help="Output .x2m2")
    parser.add_argument("--clip", default=None, help="Clip key for a multi-clip PKL")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if Path(args.source).suffix.lower() != ".pkl":
        parser.error("this exporter currently accepts motion-lib .pkl files")
    name, frames, fps, size, digest = export_motion(
        args.source, args.destination, args.clip
    )
    if not args.quiet:
        print(
            f"X2M2 clip={name} frames={frames} fps={fps:g} bytes={size} "
            f"sha256={digest} -> {args.destination}"
        )


if __name__ == "__main__":
    main()
