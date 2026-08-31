#!/usr/bin/env python3
"""Offline reproduction of the public SONIC-X2 Web demo.

This is intentionally a small, dependency-light Python port of the public
``sonic-web-demo-x2/js/policy.js`` and its policy loop in ``js/demo.js``.  It
is an evaluation harness, not a robot driver: it only loads an ONNX model and
steps MuJoCo in memory.

The important parts are kept literal rather than guessed:

* 1670-D observation = 680-D tokenizer + 990-D term-major history;
* 31-D IsaacLab action -> MuJoCo action permutation;
* raw training PD (tuning is opt-in), actuator permutation and 4 x 0.005 s
  MuJoCo substeps per 50 Hz policy tick;
* heading normalization, one-second stand blend and full reset on loop wrap.

The input motion files are the joblib-compressed PHUMA-X2 ``.pkl`` files that
are already present on the 3090 workstation.  Results are written to the
large-data mount, never to the Git work tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import joblib
import mujoco
import numpy as np
import onnxruntime as ort


# Public source snapshot used for this port.
POLICY_SOURCE = (
    "https://raw.githubusercontent.com/meetsitaram/sonic-web-demo-x2/"
    "refs/heads/master/js/policy.js"
)
DEMO_SOURCE = (
    "https://raw.githubusercontent.com/meetsitaram/sonic-web-demo-x2/"
    "refs/heads/master/js/demo.js"
)
WEB_REPO = "https://github.com/meetsitaram/sonic-web-demo-x2"

N_JOINTS = 31
HISTORY_LEN = 10
NUM_FUTURE_FRAMES = 10
DT_FUTURE_REF = 0.1
TOK_DIM = 680
PROP_DIM = 990
OBS_DIM = 1670
ACTION_CLIP = 20.0
DEFAULT_ROOT_HEIGHT = 0.60


def preload_onnx_cuda_runtime() -> tuple[bool, str | None]:
    """Expose pip-installed CUDA/cuDNN libraries to ONNX Runtime.

    Recent ``onnxruntime-gpu`` wheels may advertise CUDAExecutionProvider
    even when the dynamic loader cannot yet see the NVIDIA wheels installed
    below ``site-packages/nvidia``.  ``preload_dlls`` loads those libraries
    without modifying the system CUDA installation or global environment.
    """

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False, "CUDAExecutionProvider is not advertised by onnxruntime"
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return False, "onnxruntime.preload_dlls is unavailable"
    try:
        preload()
    except Exception as exc:  # pragma: no cover - depends on host runtime
        return False, f"{type(exc).__name__}: {exc}"
    return True, None

IL_TO_MJ = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10, 30, 16, 23,
     5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
MJ_TO_IL = np.asarray(
    [0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12, 17, 21,
     23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16],
    dtype=np.int64,
)
JOINT_TO_ACTUATOR = np.asarray(
    list(range(15)) + list(range(17, 31)) + [15, 16], dtype=np.int64
)

KP_MJ = np.asarray(
    [99.0984, 99.0984, 99.0984, 99.0984, 21.3759, 21.3759,
     99.0984, 99.0984, 99.0984, 99.0984, 21.3759, 21.3759,
     40.1792, 14.2506, 14.2506,
     14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,
     14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,
     16.7783, 16.7783], dtype=np.float32)
KD_MJ = np.asarray(
    [6.3088, 6.3088, 6.3088, 6.3088, 0.9072, 0.9072,
     6.3088, 6.3088, 6.3088, 6.3088, 0.9072, 0.9072,
     2.5579, 0.9072, 0.9072,
     0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
     0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
     1.0681, 1.0681], dtype=np.float32)
ACTION_SCALE_MJ = np.asarray(
    [0.302729, 0.302729, 0.302729, 0.302729, 0.631551, 0.421034,
     0.302729, 0.302729, 0.302729, 0.302729, 0.631551, 0.421034,
     0.746654, 0.631551, 0.631551,
     0.631551, 0.631551, 0.421034, 0.421034, 0.421034, 0.089401, 0.089401,
     0.631551, 0.631551, 0.421034, 0.421034, 0.421034, 0.089401, 0.089401,
     0.038740, 0.008940], dtype=np.float32)
DEFAULT_ANGLES_MJ = np.asarray(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,
     0.0, 0.0], dtype=np.float32)
WRIST_MJ_IDX = np.asarray([19, 20, 21, 26, 27, 28], dtype=np.int64)

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint", "head_yaw_joint",
    "head_pitch_joint",
]


def quat_normalize(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.asarray([1.0, 0.0, 0.0, 0.0])


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate a vector by scalar-first quaternion q."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.asarray([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dtype=np.float64)


def quat_rotate_inv(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    return quat_rotate([q[0], -q[1], -q[2], -q[3]], v)


def quat_to_rotmat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    x2, y2, z2 = 2 * x, 2 * y, 2 * z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return np.asarray([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ], dtype=np.float64)


@dataclass
class Motion:
    name: str
    fps: float
    joint_pos: np.ndarray       # (T, 31), MuJoCo order
    root_pos: np.ndarray        # (T, 3)
    root_quat: np.ndarray       # (T, 4), scalar first

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])


def _load_joblib_motion(path: pathlib.Path) -> Motion:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        jp = np.asarray(raw["joint_pos"], dtype=np.float64)
        rp = np.asarray(raw["root_pos"], dtype=np.float64)
        rq = np.asarray([quat_normalize(q) for q in raw["root_quat"]], dtype=np.float64)
        if jp.ndim != 2 or jp.shape[1] != N_JOINTS or rp.shape != (jp.shape[0], 3):
            raise ValueError(f"{path}: unexpected official motion shapes {jp.shape} {rp.shape}")
        return Motion(str(raw.get("name", path.stem)), float(raw["fps"]), jp, rp, rq)
    outer = joblib.load(path)
    if isinstance(outer, dict) and "dof" in outer:
        inner = outer
        name = path.stem
    elif isinstance(outer, dict) and outer:
        name, inner = next(iter(outer.items()))
    else:
        raise ValueError(f"unsupported motion object in {path}: {type(outer)}")

    required = {"dof", "root_rot", "root_trans_offset", "fps", "joint_names_mujoco"}
    missing = required.difference(inner)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    names = list(inner["joint_names_mujoco"])
    if names != JOINT_NAMES:
        raise ValueError(f"{path}: PHUMA joint order does not match X2 XML")
    jp = np.asarray(inner["dof"], dtype=np.float64)
    rp = np.asarray(inner["root_trans_offset"], dtype=np.float64)
    rq = np.asarray(inner["root_rot"], dtype=np.float64)
    if jp.ndim != 2 or jp.shape[1] != N_JOINTS or rp.shape != (jp.shape[0], 3):
        raise ValueError(f"{path}: unexpected motion shapes {jp.shape} {rp.shape}")
    rq = np.asarray([quat_normalize(q) for q in rq], dtype=np.float64)
    return Motion(str(name), float(inner["fps"]), jp, rp, rq)


def normalize_heading(m: Motion) -> Motion:
    """Exact normalizeHeading() from the public demo."""
    q0 = m.root_quat[0]
    yaw = math.atan2(
        2 * (q0[0] * q0[3] + q0[1] * q0[2]),
        1 - 2 * (q0[2] * q0[2] + q0[3] * q0[3]),
    )
    c, s = math.cos(-yaw), math.sin(-yaw)
    rz = np.asarray([math.cos(-yaw / 2), 0.0, 0.0, math.sin(-yaw / 2)])
    x0, y0 = m.root_pos[0, 0], m.root_pos[0, 1]
    rp = m.root_pos.copy()
    p = m.root_pos - np.asarray([x0, y0, 0.0])
    rp[:, 0] = c * p[:, 0] - s * p[:, 1]
    rp[:, 1] = s * p[:, 0] + c * p[:, 1]
    rq = np.asarray([quat_normalize(quat_mul(rz, q)) for q in m.root_quat])
    return Motion(m.name, m.fps, m.joint_pos.copy(), rp, rq)


def slerp(a: Sequence[float], b: Sequence[float], t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    sign = -1.0 if d < 0 else 1.0
    d = abs(d)
    if d > 1 - 1e-8:
        return quat_normalize((1 - t) * a + sign * t * b)
    th = math.acos(max(-1.0, min(1.0, d)))
    return quat_normalize(
        math.sin((1 - t) * th) / math.sin(th) * a
        + sign * math.sin(t * th) / math.sin(th) * b
    )


def prepend_stand_blend(m: Motion, blend_sec: float = 1.0) -> Motion:
    """Exact prependStandBlend() from policy.js (50 Hz stand→clip blend)."""
    n = int(round(blend_sec * m.fps))
    jp0, rp0, rq0 = m.joint_pos[0], m.root_pos[0], m.root_quat[0]
    yaw = math.atan2(
        2 * (rq0[0] * rq0[3] + rq0[1] * rq0[2]),
        1 - 2 * (rq0[2] * rq0[2] + rq0[3] * rq0[3]),
    )
    stand_q = np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    jp_new, rp_new, rq_new = [], [], []
    for k in range(n):
        t = k / n
        e = t * t * (3 - 2 * t)
        jp_new.append(DEFAULT_ANGLES_MJ * (1 - e) + jp0 * e)
        rp_new.append([
            rp0[0] * e,
            rp0[1] * e,
            DEFAULT_ROOT_HEIGHT * (1 - e) + rp0[2] * e,
        ])
        rq_new.append(slerp(stand_q, rq0, e))
    return Motion(
        m.name,
        m.fps,
        np.concatenate([np.asarray(jp_new), m.joint_pos], axis=0),
        np.concatenate([np.asarray(rp_new), m.root_pos], axis=0),
        np.concatenate([np.asarray(rq_new), m.root_quat], axis=0),
    )


class SonicPolicy:
    def __init__(self, model_path: pathlib.Path, require_cuda: bool = True,
                 future_mode: str = "oracle"):
        available = ort.get_available_providers()
        # Disable TF32 so CUDA stays numerically aligned with the CPU/WASM
        # reference path.  On RTX 3090 the default TF32 path differed by up to
        # 7.5e-3 on a deterministic X2-Sonic provider-parity probe, whereas
        # FP32 CUDA reduced the maximum difference below 3e-6.
        providers = [
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
        if require_cuda and "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
        self.cuda_runtime_preloaded, self.cuda_runtime_preload_error = (
            preload_onnx_cuda_runtime()
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        # ONNX may preserve a symbolic batch dimension ("batch") even though
        # the public JS always feeds [1, 1670].
        if len(self.input.shape) != 2 or self.input.shape[1] != OBS_DIM:
            raise RuntimeError(f"unexpected ONNX input {self.input.name} {self.input.shape}")
        if len(self.output.shape) != 2 or self.output.shape[1] != N_JOINTS:
            raise RuntimeError(f"unexpected ONNX output {self.output.name} {self.output.shape}")
        if "CUDAExecutionProvider" not in self.session.get_providers() and require_cuda:
            raise RuntimeError(
                "session did not use CUDA: "
                f"{self.session.get_providers()}; "
                f"preload_error={self.cuda_runtime_preload_error}"
            )
        self.future_mode = future_mode
        self.reset()

    def reset(self) -> None:
        self.h_ang = []
        self.h_jp = []
        self.h_jv = []
        self.h_action = []
        self.h_gravity = []
        self.last_action_mj = np.zeros(N_JOINTS, dtype=np.float32)

    def _append_history(self, gravity, angvel, jp_rel_il, jv_il, action_il):
        bufs = [self.h_ang, self.h_jp, self.h_jv, self.h_action, self.h_gravity]
        vals = [angvel, jp_rel_il, jv_il, action_il, gravity]
        if not self.h_ang:
            for buf, val in zip(bufs, vals):
                buf.extend([np.asarray(val, dtype=np.float32).copy() for _ in range(HISTORY_LEN)])
        else:
            for buf, val in zip(bufs, vals):
                buf.pop(0)
                buf.append(np.asarray(val, dtype=np.float32).copy())

    def _tokenizer(self, motion: Motion, motion_time: float, base_q: Sequence[float]) -> np.ndarray:
        n, fps, dt = motion.frames, motion.fps, 1.0 / motion.fps
        jpos_flat = np.zeros(NUM_FUTURE_FRAMES * N_JOINTS, dtype=np.float32)
        jvel_flat = np.zeros_like(jpos_flat)
        ori = np.zeros(NUM_FUTURE_FRAMES * 6, dtype=np.float32)
        lookahead_s = {
            "oracle": None,
            "hold": 0.0,
            "lookahead_0p1": 0.1,
            "lookahead_0p2": 0.2,
            "lookahead_0p3": 0.3,
        }[self.future_mode]
        for f in range(NUM_FUTURE_FRAMES):
            requested_time = motion_time + f * DT_FUTURE_REF
            if lookahead_s is None:
                future_time = requested_time
                held_beyond_available_horizon = False
            else:
                available_until = motion_time + lookahead_s
                future_time = min(requested_time, available_until)
                held_beyond_available_horizon = (
                    self.future_mode == "hold"
                    or requested_time > available_until + 1e-9
                )
            fi = min(int(math.floor(future_time / dt)), n - 1)
            prev_fi = max(0, fi - 1)
            jp, jpp = motion.joint_pos[fi], motion.joint_pos[prev_fi]
            for il, mj in enumerate(IL_TO_MJ):
                jpos_flat[f * N_JOINTS + il] = jp[mj]
                jvel_flat[f * N_JOINTS + il] = (
                    0.0 if held_beyond_available_horizon
                    else (jp[mj] - jpp[mj]) * fps
                )
            rel = quat_mul(
                [base_q[0], -base_q[1], -base_q[2], -base_q[3]],
                motion.root_quat[fi],
            )
            r = quat_to_rotmat(rel)
            o = f * 6
            ori[o:o + 6] = [r[0, 0], r[0, 1], r[1, 0], r[1, 1], r[2, 0], r[2, 1]]
        out = np.zeros(TOK_DIM, dtype=np.float32)
        # The public JS builds command_flat=[all jp|all jv], then slices
        # 62-wide rows.  This deliberately does not use [jp_k|jv_k].
        command_flat = np.concatenate([jpos_flat, jvel_flat])
        for k in range(NUM_FUTURE_FRAMES):
            out[68 * k:68 * k + 62] = command_flat[62 * k:62 * k + 62]
            out[68 * k + 62:68 * k + 68] = ori[6 * k:6 * k + 6]
        return out

    def _proprio(self) -> np.ndarray:
        out = np.zeros(PROP_DIM, dtype=np.float32)
        p = 0
        for buf in [self.h_ang, self.h_jp, self.h_jv, self.h_action, self.h_gravity]:
            for frame in buf:
                out[p:p + len(frame)] = frame
                p += len(frame)
        if p != PROP_DIM:
            raise AssertionError(f"proprio length {p} != {PROP_DIM}")
        return out

    def infer(self, motion: Motion, motion_time: float, qpos: np.ndarray, qvel: np.ndarray):
        q = qpos[3:7]
        gravity = quat_rotate_inv(q, [0.0, 0.0, -1.0])
        angvel = qvel[3:6]
        jp_rel = np.empty(N_JOINTS, dtype=np.float32)
        jv = np.empty(N_JOINTS, dtype=np.float32)
        act = np.empty(N_JOINTS, dtype=np.float32)
        for il, mj in enumerate(IL_TO_MJ):
            jp_rel[il] = qpos[7 + mj] - DEFAULT_ANGLES_MJ[mj]
            jv[il] = qvel[6 + mj]
            act[il] = self.last_action_mj[mj]
        self._append_history(gravity, angvel, jp_rel, jv, act)
        obs = np.concatenate([self._tokenizer(motion, motion_time, q), self._proprio()])
        if obs.shape != (OBS_DIM,) or not np.all(np.isfinite(obs)):
            raise RuntimeError("non-finite or malformed observation")
        result = self.session.run([self.output.name], {self.input.name: obs[None, :]})[0]
        action = np.asarray(result[0], dtype=np.float32)
        if action.shape != (N_JOINTS,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"non-finite or malformed action: {action.shape}")
        return obs, action

    def action_to_targets(self, action: np.ndarray, motion: Motion, frame: int, wrist_ref: bool):
        targets = np.empty(N_JOINTS, dtype=np.float32)
        for mj in range(N_JOINTS):
            a = float(np.clip(action[MJ_TO_IL[mj]], -ACTION_CLIP, ACTION_CLIP))
            self.last_action_mj[mj] = a
            targets[mj] = a * ACTION_SCALE_MJ[mj] + DEFAULT_ANGLES_MJ[mj]
        if wrist_ref:
            targets[WRIST_MJ_IDX] = motion.joint_pos[frame, WRIST_MJ_IDX]
        return targets

    @staticmethod
    def targets_to_ctrl(targets, qpos, qvel, ctrl, tuning: bool = False):
        # The public default is tuning OFF.  This harness intentionally keeps
        # the training-equivalent KP/KD; the optional tuning stack is not used
        # for the first proof of weight usability.
        if tuning:
            raise NotImplementedError("public ?tuning=on stack is intentionally not enabled in v1")
        for mj in range(N_JOINTS):
            torque = KP_MJ[mj] * (targets[mj] - qpos[7 + mj]) - KD_MJ[mj] * qvel[6 + mj]
            ctrl[JOINT_TO_ACTUATOR[mj]] = torque


def _assert_scene(model: mujoco.MjModel) -> dict:
    if model.nq < 38 or model.nv < 37 or model.nu < 31:
        raise RuntimeError(f"scene too small for X2: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    qpos_adr, qvel_adr, actuator_ids, joint_ids = [], [], [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + name)
        if jid < 0 or aid < 0:
            raise RuntimeError(f"missing X2 joint/actuator {name}")
        joint_ids.append(jid)
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        qvel_adr.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(aid)
    if qpos_adr != list(range(7, 38)) or qvel_adr != list(range(6, 37)):
        raise RuntimeError(f"unexpected X2 qpos/qvel addresses: {qpos_adr} {qvel_adr}")
    if actuator_ids != JOINT_TO_ACTUATOR.tolist():
        raise RuntimeError(
            "actuator order differs from public JOINT_TO_ACTUATOR: "
            f"{actuator_ids} != {JOINT_TO_ACTUATOR.tolist()}"
        )
    ranges = np.asarray([model.jnt_range[jid] for jid in joint_ids], dtype=np.float64)
    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "timestep": float(model.opt.timestep), "joint_ids": joint_ids,
        "qpos_adr": qpos_adr, "qvel_adr": qvel_adr,
        "actuator_ids": actuator_ids, "joint_ranges": ranges.tolist(),
    }


def _write_spawn(model, data, motion: Motion, frame: int = 0):
    fi = min(frame, motion.frames - 1)
    fi1 = min(fi + 1, motion.frames - 1)
    fps = motion.fps
    data.qpos[0:2] = 0.0
    data.qpos[2] = motion.root_pos[fi, 2]
    data.qpos[3:7] = motion.root_quat[fi]
    data.qpos[7:38] = motion.joint_pos[fi]
    data.qvel[:] = 0.0
    if fi1 > fi:
        data.qvel[0:3] = (motion.root_pos[fi1] - motion.root_pos[fi]) * fps
        dq = quat_mul(
            [motion.root_quat[fi, 0], -motion.root_quat[fi, 1],
             -motion.root_quat[fi, 2], -motion.root_quat[fi, 3]],
            motion.root_quat[fi1],
        )
        sign = 1.0 if dq[0] >= 0 else -1.0
        data.qvel[3:6] = 2.0 * dq[1:4] * fps * sign
        data.qvel[6:37] = (motion.joint_pos[fi1] - motion.joint_pos[fi]) * fps
    mujoco.mj_forward(model, data)


def _motion_files(root: pathlib.Path, sets: Sequence[str], clips_per_set: int,
                  max_clips: int, substrings: Sequence[str] = ()):
    out = []
    for name in sets:
        # The public Web demo stores its already-retargeted motion bank as
        # JSON directly below assets/motions; use index order for a stable
        # official smoke matrix.  PHUMA remains the joblib directory path.
        if name in {"official", "official_web", "web"} or (root / "index.json").is_file():
            index_path = root / "index.json"
            if index_path.is_file():
                entries = json.loads(index_path.read_text())
                files = [root / (str(e["id"]) + ".json") for e in entries]
                files = [x for x in files if x.is_file()]
            else:
                files = sorted(x for x in root.glob("*.json") if x.name != "index.json")
        else:
            p = root / name
            files = sorted(p.glob("*.pkl"))
        if substrings:
            files = [x for x in files if any(s.lower() in x.stem.lower() for s in substrings)]
        out.extend(files[:clips_per_set])
    return out[:max_clips]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def evaluate_clip(model, scene_info, policy, path, seconds, wrist_ref, seed):
    motion = prepend_stand_blend(normalize_heading(_load_joblib_motion(path)))
    data = mujoco.MjData(model)
    policy.reset()
    mujoco.mj_resetData(model, data)
    _write_spawn(model, data, motion, 0)
    ctrl = np.zeros(model.nu, dtype=np.float64)
    steps = int(round(seconds * 50))
    max_drift = max_root_z = 0.0
    min_root_z = float(data.qpos[2])
    max_tilt = max_joint_violation = 0.0
    joint_tracking_abs_sum = 0.0
    joint_tracking_samples = 0
    max_joint_tracking_error = 0.0
    max_action = 0.0
    max_obs = 0.0
    first_obs_sha = first_action_sha = None
    fall = None
    loop_resets = 0
    t0 = time.perf_counter()
    for tick in range(steps):
        if tick > 0 and tick % motion.frames == 0:
            # Public demo resets on loop wrap; it never teleports a live state
            # to frame zero.
            loop_resets += 1
            policy.reset()
            mujoco.mj_resetData(model, data)
            _write_spawn(model, data, motion, 0)
        frame = tick % motion.frames
        motion_time = frame / motion.fps
        obs, action = policy.infer(motion, motion_time, data.qpos, data.qvel)
        if first_obs_sha is None:
            first_obs_sha = hashlib.sha256(obs.tobytes()).hexdigest()
            first_action_sha = hashlib.sha256(action.tobytes()).hexdigest()
        max_action = max(max_action, float(np.max(np.abs(action))))
        max_obs = max(max_obs, float(np.max(np.abs(obs))))
        targets = policy.action_to_targets(action, motion, frame, wrist_ref)
        for _ in range(4):
            ctrl[:] = 0.0
            policy.targets_to_ctrl(targets, data.qpos, data.qvel, ctrl)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        root_z = float(data.qpos[2])
        tilt_vec = quat_rotate_inv(data.qpos[3:7], [0.0, 0.0, -1.0])
        tilt = math.acos(float(np.clip(-tilt_vec[2], -1.0, 1.0)))
        qvals = np.asarray(data.qpos[7:38], dtype=np.float64)
        joint_tracking_error = np.abs(
            qvals - np.asarray(motion.joint_pos[frame], dtype=np.float64)
        )
        joint_tracking_abs_sum += float(np.sum(joint_tracking_error))
        joint_tracking_samples += int(joint_tracking_error.size)
        max_joint_tracking_error = max(
            max_joint_tracking_error,
            float(np.max(joint_tracking_error)),
        )
        ranges = np.asarray(scene_info["joint_ranges"], dtype=np.float64)
        violation = float(np.max(np.maximum(ranges[:, 0] - qvals, qvals - ranges[:, 1])))
        violation = max(0.0, violation)
        max_drift = max(max_drift, float(np.hypot(data.qpos[0], data.qpos[1])))
        max_root_z = max(max_root_z, root_z)
        min_root_z = min(min_root_z, root_z)
        max_tilt = max(max_tilt, tilt)
        max_joint_violation = max(max_joint_violation, violation)
        if not np.all(np.isfinite(data.qpos[:38])) or not np.all(np.isfinite(data.qvel[:37])):
            fall = {"tick": tick, "time": tick / 50.0, "reason": "nonfinite_state"}
            break
        if fall is None and (root_z < 0.35 or tilt > 1.10):
            fall = {"tick": tick, "time": tick / 50.0, "reason": "root_height_or_tilt"}
            break
    elapsed = time.perf_counter() - t0
    return {
        "motion": motion.name,
        "source_file": str(path),
        "source_frames": int(motion.frames - round(motion.fps)),
        "blended_frames": int(motion.frames),
        "fps": float(motion.fps),
        "requested_seconds": seconds,
        "simulated_seconds": (steps if fall is None else fall["tick"] + 1) / 50.0,
        "steps": steps if fall is None else fall["tick"] + 1,
        "loop_resets": loop_resets,
        "fall": fall,
        "max_root_xy_drift_m": max_drift,
        "min_root_z_m": min_root_z,
        "max_root_z_m": max_root_z,
        "max_tilt_rad": max_tilt,
        "max_joint_limit_violation_rad": max_joint_violation,
        "mean_joint_tracking_mae_rad": (
            joint_tracking_abs_sum / max(joint_tracking_samples, 1)
        ),
        "max_joint_tracking_error_rad": max_joint_tracking_error,
        "max_abs_raw_action": max_action,
        "max_abs_observation": max_obs,
        "first_obs_sha256": first_obs_sha,
        "first_action_sha256": first_action_sha,
        "wall_seconds": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--scene", type=pathlib.Path, required=True)
    ap.add_argument("--motion-root", type=pathlib.Path, required=True)
    ap.add_argument("--sets", default="phuma_x2_strict89_foundation_50fps,phuma_x2_medium169_50fps,phuma_x2_broad227_50fps,phuma_x2_hybrid1200_clean291_50fps")
    ap.add_argument("--clips-per-set", type=int, default=2)
    ap.add_argument("--max-clips", type=int, default=8)
    ap.add_argument("--motion-substrings", default="",
                    help="comma-separated filename substrings; useful for official named clips")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument(
        "--future-mode",
        choices=["oracle", "hold", "lookahead_0p1", "lookahead_0p2", "lookahead_0p3"],
        default="oracle",
        help=(
            "Tokenizer reference availability: oracle uses the full 0.9 s future; "
            "hold repeats the current pose with zero velocity; lookahead_* exposes "
            "only that much true future and holds the remainder."
        ),
    )
    ap.add_argument("--wrist-ref", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    np.random.seed(args.seed)

    model_hash = _sha256(args.model)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    scene_info = _assert_scene(model)
    policy = SonicPolicy(
        args.model,
        require_cuda=args.require_cuda,
        future_mode=args.future_mode,
    )
    substrings = [x.strip() for x in args.motion_substrings.split(",") if x.strip()]
    files = _motion_files(args.motion_root, args.sets.split(","), args.clips_per_set,
                          args.max_clips, substrings)
    if not files:
        raise RuntimeError("no PHUMA files selected")
    print(json.dumps({
        "model": str(args.model), "model_sha256": model_hash,
        "scene": str(args.scene), "scene_info": scene_info,
        "providers": policy.session.get_providers(),
        "onnx_input": {"name": policy.input.name, "shape": policy.input.shape},
        "onnx_output": {"name": policy.output.name, "shape": policy.output.shape},
        "files": [str(x) for x in files], "wrist_ref": args.wrist_ref,
        "future_mode": args.future_mode,
    }, ensure_ascii=False), flush=True)

    results = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        results.append(evaluate_clip(model, scene_info, policy, path, args.seconds, args.wrist_ref, args.seed + i))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    passed = [r for r in results if r["fall"] is None]
    report = {
        "status": "passed" if len(passed) == len(results) else "failed_closed_loop_safety",
        "definition": "ONNX loads with 1670->31 contract and completes requested MuJoCo loops without root fall/nonfinite state",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "policy_source": POLICY_SOURCE,
        "demo_source": DEMO_SOURCE,
        "web_repo": WEB_REPO,
        "model": str(args.model), "model_sha256": model_hash,
        "scene": str(args.scene), "scene_info": scene_info,
        "onnx": {
            "input_name": policy.input.name, "input_shape": list(policy.input.shape),
            "output_name": policy.output.name, "output_shape": list(policy.output.shape),
            "providers": policy.session.get_providers(),
            "cuda_provider_options": policy.session.get_provider_options().get(
                "CUDAExecutionProvider", {}
            ),
            "cuda_runtime_preloaded": policy.cuda_runtime_preloaded,
            "cuda_runtime_preload_error": policy.cuda_runtime_preload_error,
        },
        "contract": {
            "obs_dim": OBS_DIM, "tokenizer_dim": TOK_DIM, "proprio_dim": PROP_DIM,
            "action_dim": N_JOINTS, "history_len": HISTORY_LEN,
            "policy_hz": 50, "substeps_per_tick": 4,
            "wrist_reference_fix": bool(args.wrist_ref), "tuning": False,
            "future_mode": args.future_mode,
            "joint_order": JOINT_NAMES, "il_to_mj": IL_TO_MJ.tolist(),
            "mj_to_il": MJ_TO_IL.tolist(), "joint_to_actuator": JOINT_TO_ACTUATOR.tolist(),
        },
        "selected_files": [str(x) for x in files],
        "results": results,
        "summary": {
            "clips": len(results), "passed": len(passed), "failed": len(results) - len(passed),
            "min_survival_seconds": min(r["simulated_seconds"] for r in results),
            "max_drift_m": max(r["max_root_xy_drift_m"] for r in results),
            "max_tilt_rad": max(r["max_tilt_rad"] for r in results),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"REPORT={args.output}", flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
