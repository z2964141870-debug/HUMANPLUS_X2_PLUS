"""Dependency-light HMCP v1 encoder compatible with HumanPlus.

Only the encoder surface imported by ``garment_udp_v2_reconnect_safe.py`` is
provided.  The strict target receiver independently validates every field.
"""

from __future__ import annotations

import struct
from typing import Sequence

import numpy as np


DEFAULT_PORT = 51234
G1_DOF_FULL = 29

_MAGIC = b"HMCP"
_VERSION = 1
_QPOS_COUNT = 36
_HEADER = struct.Struct("<4sBBI8sH")


def encode_frame(*, seq: int, send_ts: float, qpos: Sequence[float]) -> bytes:
    """Encode one exact 36-float HMCP v1 body-reference datagram."""
    values = np.asarray(qpos, dtype="<f4").reshape(-1)
    if values.shape != (_QPOS_COUNT,):
        raise ValueError(f"HMCP qpos must have shape (36,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("HMCP qpos contains non-finite values")
    timestamp = struct.pack("<d", float(send_ts))
    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        0,
        int(seq) & 0xFFFFFFFF,
        timestamp,
        _QPOS_COUNT,
    )
    return header + values.tobytes(order="C")
