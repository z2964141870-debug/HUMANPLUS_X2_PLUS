#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort


NUM_DOFS = 31
HISTORY = 10
IL_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_pitch",
    "left_hip_yaw", "right_hip_yaw", "waist_roll",
    "left_knee", "right_knee", "head_yaw",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch", "head_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow", "left_wrist_yaw",
    "right_wrist_yaw", "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_roll", "right_wrist_roll",
]


def read_real_blob(path: Path):
    data = path.read_bytes()
    offset = 0
    magic = data[offset:offset + 8]
    offset += 8
    if magic != b"X2OBSV01":
        raise ValueError(f"bad magic: {magic!r}")
    tok_dim, prop_dim, action_dim = struct.unpack_from("<III", data, offset)
    offset += 12
    policy_time = struct.unpack_from("<d", data, offset)[0]
    offset += 8

    def take(dtype, count):
        nonlocal offset
        values = np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy()
        offset += values.nbytes
        return values

    payload = {
        "policy_time": policy_time,
        "tokenizer_obs": take("<f4", tok_dim),
        "proprioception": take("<f4", prop_dim),
        "action_il": take("<f8", action_dim),
        "joint_pos_mj": take("<f8", NUM_DOFS),
        "joint_vel_mj": take("<f8", NUM_DOFS),
        "base_quat_wxyz": take("<f8", 4),
        "base_ang_vel": take("<f8", 3),
    }
    if offset != len(data):
        raise ValueError(f"unexpected trailing bytes: {len(data) - offset}")
    return payload


def stats(label, real, sim):
    delta = np.asarray(real) - np.asarray(sim)
    print(
        f"{label:18s} rmse={np.sqrt(np.mean(delta * delta)):.6f} "
        f"max={np.max(np.abs(delta)):.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()

    real = read_real_blob(args.real)
    sim_json = json.loads(args.sim.read_text(encoding="utf-8"))
    sim = {key: np.asarray(value) if isinstance(value, list) else value
           for key, value in sim_json.items()}

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def infer(tok, prop):
        obs = np.concatenate((tok, prop)).astype(np.float32)[None, :]
        return np.asarray(session.run(None, {input_name: obs})[0]).reshape(-1)

    real_tok = real["tokenizer_obs"]
    real_prop = real["proprioception"]
    sim_tok = sim["tokenizer_obs"].astype(np.float32)
    sim_prop = sim["proprioception"].astype(np.float32)
    outputs = {
        "real_tok + real_prop": infer(real_tok, real_prop),
        "sim_tok  + real_prop": infer(sim_tok, real_prop),
        "real_tok + sim_prop": infer(real_tok, sim_prop),
        "sim_tok  + sim_prop": infer(sim_tok, sim_prop),
    }

    print("Input parity")
    stats("tokenizer", real_tok, sim_tok)
    stats("proprioception", real_prop, sim_prop)
    stats("base_quat", real["base_quat_wxyz"], sim["base_quat_wxyz"])
    stats("base_ang_vel", real["base_ang_vel"], sim["base_ang_vel"])
    print("\nHybrid inference (left/right ankle pitch)")
    for label, output in outputs.items():
        print(f"{label:24s} L={output[14]:+.4f} R={output[15]:+.4f}")
    stats("blob action check", outputs["real_tok + real_prop"], real["action_il"])

    term_layout = [
        ("ang_vel", 0, 3 * HISTORY),
        ("joint_pos", 3 * HISTORY, NUM_DOFS * HISTORY),
        ("joint_vel", 3 * HISTORY + NUM_DOFS * HISTORY, NUM_DOFS * HISTORY),
        ("last_action", 3 * HISTORY + 2 * NUM_DOFS * HISTORY,
         NUM_DOFS * HISTORY),
        ("gravity", 3 * HISTORY + 3 * NUM_DOFS * HISTORY, 3 * HISTORY),
    ]
    print("\nReplace one real proprioception term with simulation")
    for name, start, width in term_layout:
        hybrid = real_prop.copy()
        hybrid[start:start + width] = sim_prop[start:start + width]
        output = infer(real_tok, hybrid)
        print(f"{name:14s} L={output[14]:+.4f} R={output[15]:+.4f}")

    jpos_start = 3 * HISTORY
    jvel_start = jpos_start + NUM_DOFS * HISTORY
    sim_output = outputs["sim_tok  + sim_prop"]
    ranking = []
    for dof, name in enumerate(IL_NAMES):
        hybrid = real_prop.copy()
        for frame in range(HISTORY):
            jp = jpos_start + frame * NUM_DOFS + dof
            jv = jvel_start + frame * NUM_DOFS + dof
            hybrid[jp] = sim_prop[jp]
            hybrid[jv] = sim_prop[jv]
        output = infer(real_tok, hybrid)
        score = abs(output[15] - sim_output[15])
        ranking.append((score, name, output[14], output[15]))

    print("\nBest single-joint state replacements for right ankle action")
    for score, name, left, right in sorted(ranking)[:12]:
        print(f"{name:28s} L={left:+.4f} R={right:+.4f} residual={score:.4f}")


if __name__ == "__main__":
    main()
