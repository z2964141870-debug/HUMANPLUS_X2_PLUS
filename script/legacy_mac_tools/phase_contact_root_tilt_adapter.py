#!/usr/bin/env python3
"""Add a kinematic foot-support proxy to the phase-aware root-tilt adapter.

This is an offline motion-distribution probe.  It evaluates the candidate
phase-aware root tilt in the official X2 MuJoCo model and suppresses the tilt
only on frames where the candidate loses all nominal foot support while the
upright/yaw-only reference still has support.  It does not read policy state,
contact forces, or a real robot; the foot-height/speed test is explicitly a
kinematic proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from phase_aware_root_tilt_adapter import allowance, sha256, slerp, yaw_quat


def foot_geom_ids(model: mujoco.MjModel) -> list[int]:
    result = []
    for side in ("left", "right"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        if body < 0:
            raise RuntimeError(f"missing X2 foot body: {side}")
        candidates = [
            index for index in range(model.ngeom)
            if int(model.geom_bodyid[index]) == body
            and int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        ]
        if not candidates:
            raise RuntimeError(f"missing X2 foot mesh: {side}")
        result.append(candidates[0])
    return result


def foot_positions(model: mujoco.MjModel, data: mujoco.MjData,
                   joint_pos: np.ndarray, root_pos: np.ndarray,
                   root_quat: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    positions = np.empty((len(joint_pos), 2, 3), dtype=np.float64)
    for index in range(len(joint_pos)):
        data.qpos[:] = 0.0
        data.qpos[2] = root_pos[index, 2]
        data.qpos[3:7] = root_quat[index]
        data.qpos[7:38] = joint_pos[index]
        mujoco.mj_forward(model, data)
        positions[index] = data.geom_xpos[geom_ids]
    return positions


def support_mask(positions: np.ndarray, fps: float,
                 height_m: float, speed_mps: float) -> np.ndarray:
    if len(positions) <= 1:
        speeds = np.zeros((len(positions), 2), dtype=np.float64)
    else:
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=2) * fps
        speeds = np.vstack([speeds, speeds[-1:]])
    near_floor = positions[:, :, 2] <= height_m
    stationary = speeds <= speed_mps
    return (near_floor & stationary).any(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-scale", type=float, default=0.1)
    parser.add_argument("--height-low", type=float, default=0.50)
    parser.add_argument("--height-high", type=float, default=0.58)
    parser.add_argument("--descent-low", type=float, default=-0.05)
    parser.add_argument("--descent-high", type=float, default=0.0)
    parser.add_argument("--contact-height", type=float, default=0.08)
    parser.add_argument("--contact-speed", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to write non-empty output dir: {args.output_dir}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    geom_ids = foot_geom_ids(model)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_record in source_manifest["records"]:
        source = Path(source_record["output"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        joints = np.asarray(payload["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
        root_quat = np.asarray(payload["root_quat"], dtype=np.float64)
        fps = float(payload["fps"])
        phase_scales, phase_stats = allowance(
            root_pos, fps, args.max_scale, args.height_low, args.height_high,
            args.descent_low, args.descent_high,
        )
        upright_quat = np.asarray([yaw_quat(q) for q in root_quat], dtype=np.float64)
        candidate_quat = np.asarray([
            slerp(yaw_quat(q), q, float(scale))
            for q, scale in zip(root_quat, phase_scales)
        ], dtype=np.float64)
        upright_foot = foot_positions(model, data, joints, root_pos, upright_quat, geom_ids)
        candidate_foot = foot_positions(model, data, joints, root_pos, candidate_quat, geom_ids)
        upright_support = support_mask(upright_foot, fps, args.contact_height, args.contact_speed)
        candidate_support = support_mask(candidate_foot, fps, args.contact_height, args.contact_speed)
        suppressed = (~candidate_support) & upright_support
        final_scales = phase_scales.copy()
        final_scales[suppressed] = 0.0
        final_quat = np.asarray([
            slerp(yaw_quat(q), q, float(scale))
            for q, scale in zip(root_quat, final_scales)
        ], dtype=np.float64)
        out = args.output_dir / source.name
        if out.exists() and not args.overwrite:
            raise FileExistsError(out)
        output_payload = dict(payload)
        output_payload["root_quat"] = final_quat.tolist()
        output_payload["display"] = f"{payload.get('display', source.stem)}__phase_contact_root_tilt"
        out.write_text(json.dumps(output_payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        records.append({
            "source": source_record["source"],
            "source_sha256": source_record["source_sha256"],
            "input": str(source.resolve()),
            "input_sha256": sha256(source),
            "output": str(out.resolve()),
            "output_sha256": sha256(out),
            "transform": {
                "name": "phase_contact_aware_root_tilt",
                "max_scale": float(args.max_scale),
                "height_low_m": float(args.height_low),
                "height_high_m": float(args.height_high),
                "descent_low_mps": float(args.descent_low),
                "descent_high_mps": float(args.descent_high),
                "contact_height_m": float(args.contact_height),
                "contact_speed_mps": float(args.contact_speed),
                "definition": "suppress candidate tilt only when candidate loses nominal foot support and upright reference retains it",
            },
            "phase_scale_stats": phase_stats,
            "contact_proxy_stats": {
                "upright_gap_fraction": float(np.mean(~upright_support)),
                "candidate_gap_fraction": float(np.mean(~candidate_support)),
                "suppressed_fraction": float(np.mean(suppressed)),
                "suppressed_frames": int(np.sum(suppressed)),
            },
        })

    index = [
        {"id": Path(record["output"]).stem, "display": Path(record["output"]).stem,
         "frames": record["phase_scale_stats"]["frames"], "fps": 50.0,
         "group": "PHUMA phase/contact-aware adapted"}
        for record in records
    ]
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "x2_sonic_phase_contact_root_tilt_manifest_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scene": str(args.scene.resolve()),
        "scene_sha256": sha256(args.scene),
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
            "contact_height_m": float(args.contact_height),
            "contact_speed_mps": float(args.contact_speed),
            "hardware_contact_force": False,
        },
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"motions": len(records), "manifest": str(args.manifest.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
