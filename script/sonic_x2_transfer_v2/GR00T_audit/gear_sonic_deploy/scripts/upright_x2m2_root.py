#!/usr/bin/env python3
"""Copy an X2M2 motion while removing root roll/pitch from every frame."""

import argparse
import hashlib
import math
import struct
from pathlib import Path


HEADER = struct.Struct("<IIId")
MAGIC = 0x58324D32
NUM_DOFS = 31
FRAME = struct.Struct("<31d4d")


def upright_root(source, destination):
    raw = Path(source).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("X2M2 header is truncated")
    magic, frames, dofs, fps = HEADER.unpack_from(raw)
    if magic != MAGIC or dofs != NUM_DOFS or frames == 0 or fps <= 0.0:
        raise ValueError(
            f"invalid X2M2 header: magic={magic:#x} frames={frames} "
            f"dofs={dofs} fps={fps}"
        )
    expected = HEADER.size + frames * FRAME.size
    if len(raw) != expected:
        raise ValueError(f"invalid X2M2 size: expected {expected}, got {len(raw)}")

    output = bytearray(raw[: HEADER.size])
    for index in range(frames):
        values = FRAME.unpack_from(raw, HEADER.size + index * FRAME.size)
        joints = values[:NUM_DOFS]
        x, y, z, w = values[NUM_DOFS:]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        output.extend(FRAME.pack(*joints, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output)
    return frames, fps, hashlib.sha256(output).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    frames, fps, digest = upright_root(args.source, args.destination)
    print(
        f"upright X2M2 frames={frames} fps={fps:g} sha256={digest} "
        f"-> {args.destination}"
    )


if __name__ == "__main__":
    main()
