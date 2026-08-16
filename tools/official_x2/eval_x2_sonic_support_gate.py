#!/usr/bin/env python3
"""Offline support-recovery gate probe for the official X2-Sonic loop.

This is an evaluation/safety experiment, not a robot driver.  It reuses the
public-contract evaluator and blends only the lower-body target toward the
official standing pose when the simulated robot is both descending quickly
and has lost a foot support side, or when its root height is already below a
configured threshold.  The default gate deliberately does *not* treat a
single-support phase by itself as a fault.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import time

import mujoco
import numpy as np

import eval_official_sonic_x2 as official


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def x2_foot_sides(model: mujoco.MjModel) -> tuple[int, dict[int, str]]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise RuntimeError("official scene has no floor geom")
    sides: dict[int, str] = {}
    for gid in range(model.ngeom):
        if int(model.geom_group[gid]) != 3:
            continue
        body_name = model.body(int(model.geom_bodyid[gid])).name or ""
        if body_name == "left_ankle_roll_link":
            sides[gid] = "left"
        elif body_name == "right_ankle_roll_link":
            sides[gid] = "right"
    if not sides or set(sides.values()) != {"left", "right"}:
        raise RuntimeError("could not identify both X2 foot geom groups")
    return floor_id, sides


def contact_sides(data: mujoco.MjData, floor_id: int,
                  foot_sides: dict[int, str]) -> int:
    present: set[str] = set()
    for i in range(data.ncon):
        contact = data.contact[i]
        first, second = int(contact.geom1), int(contact.geom2)
        if first == floor_id:
            foot = second
        elif second == floor_id:
            foot = first
        else:
            continue
        side = foot_sides.get(foot)
        if side is not None:
            present.add(side)
    return len(present)


def evaluate_clip(model: mujoco.MjModel, policy_path: pathlib.Path,
                  motion_path: pathlib.Path, seconds: float,
                  wrist_reference: bool, gain: float,
                  height_threshold: float,
                  vertical_velocity_threshold: float,
                  min_contact_sides: int,
                  lower_body_joints: int) -> dict:
    motion = official.prepend_stand_blend(
        official.normalize_heading(official._load_joblib_motion(motion_path))
    )
    policy = official.SonicPolicy(policy_path, require_cuda=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    official._write_spawn(model, data, motion, 0)
    floor_id, foot_sides = x2_foot_sides(model)
    ctrl = np.zeros(model.nu, dtype=np.float64)
    steps = int(round(seconds * 50.0))
    loop_resets = 0
    gate_ticks = 0
    fall = None
    min_root_z = float(data.qpos[2])
    max_root_drift = 0.0
    max_tilt = 0.0
    started = time.perf_counter()

    for tick in range(steps):
        if tick > 0 and tick % motion.frames == 0:
            loop_resets += 1
            policy.reset()
            mujoco.mj_resetData(model, data)
            official._write_spawn(model, data, motion, 0)

        frame = tick % motion.frames
        _, action = policy.infer(
            motion, frame / motion.fps, data.qpos, data.qvel
        )
        targets = policy.action_to_targets(
            action, motion, frame, wrist_reference
        )
        support_sides = contact_sides(data, floor_id, foot_sides)
        fast_descent = float(data.qvel[2]) < vertical_velocity_threshold
        low_root = float(data.qpos[2]) < height_threshold
        risk = low_root or (fast_descent and support_sides < min_contact_sides)
        if risk:
            gate_ticks += 1
            targets[:lower_body_joints] = (
                (1.0 - gain) * targets[:lower_body_joints]
                + gain * official.DEFAULT_ANGLES_MJ[:lower_body_joints]
            )

        for _ in range(4):
            ctrl.fill(0.0)
            policy.targets_to_ctrl(targets, data.qpos, data.qvel, ctrl)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

        root_z = float(data.qpos[2])
        gravity = official.quat_rotate_inv(data.qpos[3:7], [0.0, 0.0, -1.0])
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        min_root_z = min(min_root_z, root_z)
        max_root_drift = max(
            max_root_drift, float(np.hypot(data.qpos[0], data.qpos[1]))
        )
        max_tilt = max(max_tilt, tilt)
        if not np.all(np.isfinite(data.qpos[:38])) or not np.all(
                np.isfinite(data.qvel[:37])):
            fall = {"tick": tick, "time": tick / 50.0,
                    "reason": "nonfinite_state"}
            break
        if root_z < 0.35 or tilt > 1.10:
            fall = {"tick": tick, "time": tick / 50.0,
                    "reason": "root_height_or_tilt"}
            break

    simulated_steps = steps if fall is None else fall["tick"] + 1
    return {
        "motion": motion.name,
        "source_file": str(motion_path),
        "requested_seconds": seconds,
        "simulated_seconds": simulated_steps / 50.0,
        "steps": simulated_steps,
        "loop_resets": loop_resets,
        "fall": fall,
        "gate_ticks": gate_ticks,
        "gate_fraction": gate_ticks / max(1, simulated_steps),
        "min_root_z_m": min_root_z,
        "max_root_xy_drift_m": max_root_drift,
        "max_tilt_rad": max_tilt,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--scene", type=pathlib.Path, required=True)
    ap.add_argument("--motion-root", type=pathlib.Path, required=True)
    ap.add_argument("--sets", default="official")
    ap.add_argument("--clips-per-set", type=int, default=96)
    ap.add_argument("--max-clips", type=int, default=96)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--wrist-reference", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--height-threshold", type=float, default=0.48)
    ap.add_argument("--vertical-velocity-threshold", type=float, default=-0.45)
    ap.add_argument("--min-contact-sides", type=int, default=2)
    ap.add_argument("--lower-body-joints", type=int, default=15)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if not 0.0 <= args.gain <= 1.0:
        raise ValueError("gain must be in [0,1]")
    if args.lower_body_joints <= 0 or args.lower_body_joints > 31:
        raise ValueError("lower-body joint count must be in [1,31]")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    official._assert_scene(model)
    files = official._motion_files(
        args.motion_root, args.sets.split(","), args.clips_per_set,
        args.max_clips
    )
    if not files:
        raise RuntimeError("no motion files selected")
    results = []
    for index, path in enumerate(files, 1):
        results.append(evaluate_clip(
            model, args.model, path, args.seconds, args.wrist_reference,
            args.gain, args.height_threshold,
            args.vertical_velocity_threshold, args.min_contact_sides,
            args.lower_body_joints,
        ))
        if index % 16 == 0:
            print(f"progress {index}/{len(files)}", flush=True)
    passed = [item for item in results if item["fall"] is None]
    report = {
        "schema": "x2_sonic_support_recovery_gate_probe_v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "project_root": "/home/yu/projects/BFM-Zero",
        "code": str(pathlib.Path(__file__).resolve()),
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "scene": {"path": str(args.scene), "sha256": sha256(args.scene)},
        "motion_root": str(args.motion_root),
        "definition": (
            "Offline support-recovery gate after official ONNX target generation; "
            "lower-body targets blend toward DEFAULT_ANGLES_MJ only when root "
            "height is low or fast descent coincides with lost support; no hardware."
        ),
        "parameters": {
            "gain": args.gain,
            "height_threshold_m": args.height_threshold,
            "vertical_velocity_threshold_mps": args.vertical_velocity_threshold,
            "min_contact_sides": args.min_contact_sides,
            "lower_body_joint_count": args.lower_body_joints,
            "policy_hz": 50,
            "substeps": 4,
        },
        "results": results,
        "summary": {
            "clips": len(results),
            "passed": len(passed),
            "failed": len(results) - len(passed),
            "min_survival_seconds": min(item["simulated_seconds"] for item in results),
            "max_drift_m": max(item["max_root_xy_drift_m"] for item in results),
            "max_tilt_rad": max(item["max_tilt_rad"] for item in results),
            "gate_ticks_total": sum(item["gate_ticks"] for item in results),
            "clips_gate_used": sum(item["gate_ticks"] > 0 for item in results),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"REPORT={args.output}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0 if len(passed) == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
