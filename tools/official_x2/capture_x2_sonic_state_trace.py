#!/usr/bin/env python3
"""Capture an official MuJoCo qpos/qvel trace for offline parity checks.

The trace is a data artifact, not a robot log. It records the state presented
to the official SONIC policy before each 50 Hz inference tick, allowing two
different motion streams to be compared under the same physically evolved
state sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import mujoco
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_official_sonic_x2 as sonic  # noqa: E402


SCHEMA = "x2_sonic_mujoco_state_trace_v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--scene", type=pathlib.Path, required=True)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrist-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")

    source = sonic._load_joblib_motion(args.motion)
    motion = sonic.prepend_stand_blend(sonic.normalize_heading(source))
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    scene_info = sonic._assert_scene(model)
    policy = sonic.SonicPolicy(args.model, require_cuda=args.require_cuda)
    data = mujoco.MjData(model)
    ctrl = np.zeros(model.nu, dtype=np.float64)
    mujoco.mj_resetData(model, data)
    sonic._write_spawn(model, data, motion, 0)

    records = []
    fall = None
    for tick in range(args.ticks):
        if tick > 0 and tick % motion.frames == 0:
            policy.reset()
            mujoco.mj_resetData(model, data)
            sonic._write_spawn(model, data, motion, 0)
        frame = tick % motion.frames
        records.append({
            "tick": tick,
            "motion_frame": frame,
            "time_s": tick / 50.0,
            "qpos": data.qpos.astype(np.float64).tolist(),
            "qvel": data.qvel.astype(np.float64).tolist(),
        })
        obs, action = policy.infer(motion, frame / motion.fps, data.qpos, data.qvel)
        targets = policy.action_to_targets(action, motion, frame, args.wrist_ref)
        for _ in range(4):
            ctrl[:] = 0.0
            policy.targets_to_ctrl(targets, data.qpos, data.qvel, ctrl)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        tilt_vec = sonic.quat_rotate_inv(data.qpos[3:7], [0.0, 0.0, -1.0])
        tilt = float(np.arccos(np.clip(-tilt_vec[2], -1.0, 1.0)))
        if not np.all(np.isfinite(data.qpos[:38])) or not np.all(np.isfinite(data.qvel[:37])):
            fall = {"tick": tick, "reason": "nonfinite_state"}
            break
        if data.qpos[2] < 0.35 or tilt > 1.10:
            fall = {"tick": tick, "reason": "root_height_or_tilt", "tilt_rad": tilt}
            break

    report = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline_only": True,
        "hardware_touched": False,
        "motion": {"path": str(args.motion.resolve()), "sha256": sha256(args.motion), "frames": motion.frames, "fps": motion.fps},
        "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model), "providers": policy.session.get_providers()},
        "scene": {"path": str(args.scene.resolve()), "sha256": sha256(args.scene), "info": scene_info},
        "requested_ticks": args.ticks,
        "captured_ticks": len(records),
        "fall": fall,
        "records": records,
        "interpretation": "Shared MuJoCo state trace for offline observation/action parity; not a hardware trace.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "captured_ticks": len(records),
        "fall": fall,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0 if len(records) == args.ticks and fall is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
