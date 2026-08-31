#!/usr/bin/env python3
"""Inventory and stratify the PHUMA-X2 source motion bank.

The script only reads source ``.pkl`` files.  It computes distribution features
needed to choose an informative closed-loop benchmark instead of relying on
lexicographic clips or a small hand-picked sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

import mujoco
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_official_sonic_x2 as sonic  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def category(path: pathlib.Path) -> str:
    return path.stem.split("__", 1)[0].lower()


def bin_name(value: float, thresholds: tuple[float, float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "mid"
    return "high"


def _contact_metrics(motion: sonic.Motion, model: mujoco.MjModel,
                     data: mujoco.MjData, foot_geom_ids: tuple[int, int]) -> dict[str, float]:
    """Approximate foot support feasibility in the official X2 scene.

    The public evaluator spawns with root XY set to zero, so this diagnostic does
    the same.  ``geom_xpos`` is the center of each ankle/foot mesh; a center
    height <= 8 cm is treated as near-floor, and <= 0.25 m/s as a stationary
    contact candidate.  These are diagnostic thresholds, not a contact solver.
    """
    positions = np.empty((motion.frames, 2, 3), dtype=np.float64)
    for i, (jp, rp, rq) in enumerate(zip(motion.joint_pos, motion.root_pos, motion.root_quat)):
        data.qpos[:] = 0.0
        data.qpos[2] = rp[2]
        data.qpos[3:7] = rq
        data.qpos[7:38] = jp
        mujoco.mj_forward(model, data)
        positions[i, 0] = data.geom_xpos[foot_geom_ids[0]]
        positions[i, 1] = data.geom_xpos[foot_geom_ids[1]]
    heights = positions[:, :, 2]
    if motion.frames > 1:
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=2) * float(motion.fps)
        speed = np.vstack([speed, speed[-1:]])
    else:
        speed = np.zeros((1, 2), dtype=np.float64)
    near_floor = heights <= 0.08
    stationary_contact = near_floor & (speed <= 0.25)
    return {
        "foot_center_z_min_m": float(np.min(heights)),
        "foot_center_z_p95_m": float(np.quantile(heights, 0.95)),
        "foot_near_floor_fraction": float(np.mean(near_floor.any(axis=1))),
        "foot_neither_near_floor_fraction": float(np.mean(~near_floor.any(axis=1))),
        "foot_stationary_contact_fraction": float(np.mean(stationary_contact.any(axis=1))),
        "foot_contact_gap_fraction": float(np.mean(~stationary_contact.any(axis=1))),
    }


def inventory_one(path: pathlib.Path, ranges: np.ndarray | None,
                  model: mujoco.MjModel, data: mujoco.MjData,
                  foot_geom_ids: tuple[int, int]) -> dict[str, object]:
    motion = sonic._load_joblib_motion(path)
    fps = float(motion.fps)
    root_delta = np.diff(motion.root_pos, axis=0) * fps
    joint_delta = np.diff(motion.joint_pos, axis=0) * fps
    root_speed = np.linalg.norm(root_delta[:, :2], axis=1) if len(root_delta) else np.zeros(1)
    root_z_speed = np.abs(root_delta[:, 2]) if len(root_delta) else np.zeros(1)
    joint_speed = np.linalg.norm(joint_delta, axis=1) if len(joint_delta) else np.zeros(1)
    root_path = float(np.sum(np.linalg.norm(np.diff(motion.root_pos[:, :2], axis=0), axis=1))) if motion.frames > 1 else 0.0
    root_xy_disp = float(np.linalg.norm(motion.root_pos[-1, :2] - motion.root_pos[0, :2]))
    gravity_body = np.asarray([
        sonic.quat_rotate_inv(q, [0.0, 0.0, -1.0]) for q in motion.root_quat
    ], dtype=np.float64)
    tilt = np.arccos(np.clip(-gravity_body[:, 2], -1.0, 1.0))
    if len(motion.root_quat) > 1:
        quat_dot = np.abs(np.sum(motion.root_quat[1:] * motion.root_quat[:-1], axis=1))
        root_ang_speed = 2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0)) * fps
    else:
        root_ang_speed = np.zeros(1)
    limit_violation = 0.0
    if ranges is not None:
        low = ranges[:, 0][None, :]
        high = ranges[:, 1][None, :]
        limit_violation = float(np.max(np.maximum(low - motion.joint_pos, motion.joint_pos - high)))
        limit_violation = max(0.0, limit_violation)
    finite = bool(
        np.all(np.isfinite(motion.joint_pos))
        and np.all(np.isfinite(motion.root_pos))
        and np.all(np.isfinite(motion.root_quat))
    )
    root_p95 = float(np.quantile(root_speed, 0.95))
    joint_p95 = float(np.quantile(joint_speed, 0.95))
    joint_p99_angle = float(np.quantile(np.abs(motion.joint_pos), 0.99))
    record = {
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "subset": path.parent.name,
        "category": category(path),
        "frames": int(motion.frames),
        "fps": fps,
        "duration_s": float((motion.frames - 1) / fps),
        "root_xy_displacement_m": root_xy_disp,
        "root_xy_path_m": root_path,
        "root_speed_p95_mps": root_p95,
        "root_speed_max_mps": float(np.max(root_speed)),
        "root_z_speed_p95_mps": float(np.quantile(root_z_speed, 0.95)),
        "joint_speed_p95_radps": joint_p95,
        "joint_speed_max_radps": float(np.max(joint_speed)),
        "joint_abs_angle_p99_rad": joint_p99_angle,
        "joint_abs_angle_max_rad": float(np.max(np.abs(motion.joint_pos))),
        "root_z_range_m": float(np.ptp(motion.root_pos[:, 2])),
        "root_tilt_p95_rad": float(np.quantile(tilt, 0.95)),
        "root_tilt_max_rad": float(np.max(tilt)),
        "root_angular_speed_p95_radps": float(np.quantile(root_ang_speed, 0.95)),
        "root_angular_speed_max_radps": float(np.max(root_ang_speed)),
        "joint_limit_violation_rad": limit_violation,
        "finite": finite,
    }
    record.update(_contact_metrics(motion, model, data, foot_geom_ids))
    record["root_speed_bin"] = bin_name(root_p95, (0.15, 0.50))
    record["joint_speed_bin"] = bin_name(joint_p95, (2.0, 6.0))
    record["pose_amplitude_bin"] = bin_name(joint_p99_angle, (0.40, 0.90))
    record["stratum"] = "|".join([
        str(record["category"]),
        f"root_{record['root_speed_bin']}",
        f"joint_{record['joint_speed_bin']}",
        f"pose_{record['pose_amplitude_bin']}",
    ])
    return record


def select(records: list[dict[str, object]], per_set: int, per_stratum: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["finite"]:
            grouped[f"{record['subset']}|{record['stratum']}"].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda r: (float(r["joint_limit_violation_rad"]), str(r["source"])))
    chosen: list[dict[str, object]] = []
    per_subset: Counter[str] = Counter()
    for key in sorted(grouped):
        subset = key.split("|", 1)[0]
        for row in grouped[key][:per_stratum]:
            if per_subset[subset] >= per_set:
                break
            chosen.append(row)
            per_subset[subset] += 1
    # Fill any underrepresented subset with deterministic farthest-duration rows.
    for subset in sorted({str(r["subset"]) for r in records}):
        if per_subset[subset] >= per_set:
            continue
        candidates = [r for r in records if r["subset"] == subset and r["finite"] and r not in chosen]
        candidates.sort(key=lambda r: (-float(r["joint_speed_p95_radps"]), str(r["source"])))
        for row in candidates[: per_set - per_subset[subset]]:
            chosen.append(row)
            per_subset[subset] += 1
    return sorted(chosen, key=lambda r: str(r["source"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-root", type=pathlib.Path, required=True)
    parser.add_argument("--sets", required=True,
                        help="Comma-separated PHUMA subset directory names.")
    parser.add_argument("--scene", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--selection-output", type=pathlib.Path, required=True)
    parser.add_argument("--per-set", type=int, default=24)
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    scene_info = sonic._assert_scene(model)
    ranges = np.asarray(scene_info["joint_ranges"], dtype=np.float64)
    foot_geom_ids = []
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"missing X2 foot body: {body_name}")
        mesh_ids = [
            i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) == body_id
            and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        ]
        if not mesh_ids:
            raise RuntimeError(f"missing mesh foot geom: {body_name}")
        foot_geom_ids.append(mesh_ids[0])
    data = mujoco.MjData(model)
    paths = []
    for name in args.sets.split(","):
        subset = args.motion_root / name
        paths.extend(sorted(p for p in subset.glob("*.pkl") if p.name != "metadata.pkl"))
    if not paths:
        raise RuntimeError("no PHUMA pkl files selected")
    records = []
    for index, path in enumerate(paths, 1):
        if index == 1 or index % 50 == 0 or index == len(paths):
            print(f"inventory {index}/{len(paths)}", flush=True)
        records.append(inventory_one(path, ranges, model, data, tuple(foot_geom_ids)))
    selected = select(records, args.per_set, args.per_stratum)
    subsets = Counter(str(r["subset"]) for r in records)
    selected_subsets = Counter(str(r["subset"]) for r in selected)
    strata = Counter(str(r["stratum"]) for r in records)
    selected_strata = Counter(str(r["stratum"]) for r in selected)
    report = {
        "schema": "x2_sonic_motion_inventory/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motion_root": str(args.motion_root.resolve()),
        "scene": str(args.scene.resolve()),
        "scene_info": {
            "joint_ranges": scene_info["joint_ranges"],
            "contact_diagnostic": {
                "foot_geom_center_height_threshold_m": 0.08,
                "foot_speed_threshold_mps": 0.25,
                "foot_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
        },
        "records": records,
        "selection": selected,
        "summary": {
            "records": len(records),
            "subsets": dict(subsets),
            "selected_records": len(selected),
            "selected_subsets": dict(selected_subsets),
            "strata": dict(strata),
            "selected_strata": dict(selected_strata),
        },
    }
    selection = {
        "schema": "x2_sonic_stratified_selection/v1",
        "generated_utc": report["generated_utc"],
        "source_inventory": str(args.output.resolve()),
        "selected_files": [str(r["source"]) for r in selected],
        "selected_records": selected,
        "summary": report["summary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.selection_output.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
