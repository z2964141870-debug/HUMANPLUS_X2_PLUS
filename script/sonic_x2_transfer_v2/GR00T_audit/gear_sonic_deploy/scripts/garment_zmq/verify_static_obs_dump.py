#!/usr/bin/env python3
"""Verify a C++ X2OBSV01 dump made from the static v5.1 reference source.

The companion publisher must run with ``--source synthetic --amplitude 0``.
This makes every accepted reference frame deterministic: trained default joint
angles, zero reference velocity, and an identity XYZW root quaternion.

The verifier independently rebuilds the 680-D tokenizer and first-tick 990-D
proprioception vector from the robot state embedded in the dump.  When an ONNX
model is supplied, it also reruns the 1670-D observation through Python ONNX
Runtime and compares the raw 31-D action with the C++ result.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import publish_v51_reference as wire


MAGIC = b"X2OBSV01"
TOK_DIM = 680
PROP_DIM = 990
ACTION_DIM = wire.NUM_DOFS
HISTORY_LEN = 10

# policy_parameters.hpp::isaaclab_to_mujoco
IL_TO_MJ = np.asarray([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10,
    30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)


@dataclass(frozen=True)
class ObsDump:
    policy_time: float
    tokenizer: np.ndarray
    proprioception: np.ndarray
    action_il: np.ndarray
    joint_pos_mj: np.ndarray
    joint_vel_mj: np.ndarray
    base_quat_wxyz: np.ndarray
    base_ang_vel: np.ndarray


def read_dump(path: Path) -> ObsDump:
    raw = path.read_bytes()
    if len(raw) < 28 or raw[:8] != MAGIC:
        raise ValueError(f"{path}: not an X2OBSV01 dump")
    tok_dim, prop_dim, action_dim = struct.unpack_from("<III", raw, 8)
    if (tok_dim, prop_dim, action_dim) != (TOK_DIM, PROP_DIM, ACTION_DIM):
        raise ValueError(
            f"{path}: dimensions {(tok_dim, prop_dim, action_dim)} != "
            f"{(TOK_DIM, PROP_DIM, ACTION_DIM)}"
        )

    offset = 20
    policy_time = struct.unpack_from("<d", raw, offset)[0]
    offset += 8

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        dt = np.dtype(dtype)
        end = offset + dt.itemsize * count
        if end > len(raw):
            raise ValueError(f"{path}: truncated at byte {offset}")
        out = np.frombuffer(raw, dtype=dt, count=count, offset=offset).copy()
        offset = end
        return out

    result = ObsDump(
        policy_time=policy_time,
        tokenizer=take("<f4", TOK_DIM),
        proprioception=take("<f4", PROP_DIM),
        action_il=take("<f8", ACTION_DIM),
        joint_pos_mj=take("<f8", ACTION_DIM),
        joint_vel_mj=take("<f8", ACTION_DIM),
        base_quat_wxyz=take("<f8", 4),
        base_ang_vel=take("<f8", 3),
    )
    if offset != len(raw):
        raise ValueError(f"{path}: {len(raw) - offset} unexpected trailing bytes")
    for name, value in result.__dict__.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"{path}: non-finite values in {name}")
    return result


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def rot6d_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.asarray([
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy),
    ], dtype=np.float64)
    return matrix[[0, 1, 3, 4, 6, 7]]


def rotate_inverse_wxyz(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    conjugate = np.asarray([w, -x, -y, -z], dtype=np.float64)
    cw, cx, cy, cz = conjugate
    vx, vy, vz = vector
    tx = 2.0 * (cy * vz - cz * vy)
    ty = 2.0 * (cz * vx - cx * vz)
    tz = 2.0 * (cx * vy - cy * vx)
    return np.asarray([
        vx + cw * tx + (cy * tz - cz * ty),
        vy + cw * ty + (cz * tx - cx * tz),
        vz + cw * tz + (cx * ty - cy * tx),
    ], dtype=np.float64)


def expected_static_tokenizer(base_quat_wxyz: np.ndarray) -> np.ndarray:
    current_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
    current_inverse = current_xyzw * np.asarray([-1.0, -1.0, -1.0, 1.0])
    relative = quat_mul_xyzw(current_inverse, wire.IDENTITY_QUAT_XYZW)
    orientation = rot6d_from_quat_xyzw(relative).astype(np.float32)

    position_il = wire.DEFAULT_ANGLES[IL_TO_MJ].astype(np.float32)
    velocity_il = np.zeros(ACTION_DIM, dtype=np.float32)
    rows = []
    for row_index in range(10):
        values = position_il if row_index < 5 else velocity_il
        rows.append(np.concatenate([values, values, orientation]))
    result = np.concatenate(rows).astype(np.float32)
    assert result.shape == (TOK_DIM,)
    return result


def expected_first_proprioception(dump: ObsDump) -> np.ndarray:
    position_relative_il = (
        dump.joint_pos_mj[IL_TO_MJ] - wire.DEFAULT_ANGLES[IL_TO_MJ]
    ).astype(np.float32)
    velocity_il = dump.joint_vel_mj[IL_TO_MJ].astype(np.float32)
    action_il = np.zeros(ACTION_DIM, dtype=np.float32)
    gravity = rotate_inverse_wxyz(
        dump.base_quat_wxyz, np.asarray([0.0, 0.0, -1.0])
    ).astype(np.float32)
    terms = (
        np.tile(dump.base_ang_vel.astype(np.float32), HISTORY_LEN),
        np.tile(position_relative_il, HISTORY_LEN),
        np.tile(velocity_il, HISTORY_LEN),
        np.tile(action_il, HISTORY_LEN),
        np.tile(gravity, HISTORY_LEN),
    )
    result = np.concatenate(terms).astype(np.float32)
    assert result.shape == (PROP_DIM,)
    return result


def diff(name: str, actual: np.ndarray, expected: np.ndarray) -> float:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    worst = int(np.argmax(delta))
    maximum = float(delta[worst])
    rms = float(np.sqrt(np.mean(delta * delta)))
    print(
        f"{name}: max_abs={maximum:.9g} rms={rms:.9g} index={worst} "
        f"actual={actual[worst]:.9g} expected={expected[worst]:.9g}"
    )
    return maximum


def infer_python(model: Path, tokenizer: np.ndarray, proprioception: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required with --model; use the project's .venv"
        ) from exc
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    observation = np.concatenate([tokenizer, proprioception]).astype(np.float32)
    result = session.run([output_meta.name], {input_meta.name: observation[None, :]})[0]
    return np.asarray(result[0], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("obs_dump", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--obs-tolerance", type=float, default=5e-6)
    parser.add_argument("--action-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    dump = read_dump(args.obs_dump)
    expected_tokenizer = expected_static_tokenizer(dump.base_quat_wxyz)
    expected_proprioception = expected_first_proprioception(dump)
    tokenizer_error = diff("tokenizer", dump.tokenizer, expected_tokenizer)
    proprioception_error = diff(
        "proprioception", dump.proprioception, expected_proprioception
    )

    action_error = 0.0
    if args.model is not None:
        expected_action = infer_python(
            args.model, expected_tokenizer, expected_proprioception
        )
        action_error = diff("action", dump.action_il, expected_action)

    passed = (
        tokenizer_error <= args.obs_tolerance
        and proprioception_error <= args.obs_tolerance
        and (args.model is None or action_error <= args.action_tolerance)
    )
    print(
        f"policy_time={dump.policy_time:.6f}s file_bytes={args.obs_dump.stat().st_size} "
        f"result={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
