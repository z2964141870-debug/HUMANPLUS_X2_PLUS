#!/usr/bin/env python3
"""Quasi-static support-margin audit for the frozen Phase30 X2 lunge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import mujoco
import numpy as np
from scipy.spatial import ConvexHull


REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "tools")]
import retarget.run_x2_forefoot_official_physics_screen as physics  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_mask(length: int, phases: list[list[int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in phases:
        mask[int(start):int(end)] = True
    return mask


def expanded_hull_margin(point: np.ndarray, centers: np.ndarray, radius: float) -> float:
    hull = ConvexHull(centers)
    equations = hull.equations  # ax + by + c <= 0 inside
    normal = np.linalg.norm(equations[:, :2], axis=1)
    raw = -(equations[:, :2] @ point + equations[:, 2]) / normal
    return float(np.min(raw) + radius)


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def run(scene: Path, motion: Path, motion_id: str) -> dict:
    entry = joblib.load(motion)[motion_id]
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    q = np.asarray(entry["dof"], dtype=np.float64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    names = list(entry["joint_names_mujoco"])
    qadr = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names])
    phases = entry["phase30_time_dilation"]["resampled_source_contact_phases"]
    intent = {side: phase_mask(len(q), phases[side]) for side in ("left", "right")}
    floor, helper = physics.foot_geom_contract(model)
    feet = {
        side: sorted(geom for geom in geoms if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE) and int(model.geom_contype[geom]) != 0)
        for side, geoms in helper.items()
    }
    if any(len(value) != 12 for value in feet.values()):
        raise ValueError("active12 contract failed")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    rows = []
    for frame in range(len(q)):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root[frame]
        data.qpos[3:7] = quat[frame][[3, 0, 1, 2]]
        data.qpos[qadr] = q[frame]
        mujoco.mj_forward(model, data)
        com = data.subtree_com[pelvis].copy()
        active_sides = [side for side in ("left", "right") if intent[side][frame]]
        centers = np.concatenate([data.geom_xpos[feet[side], :2] for side in active_sides], axis=0) if active_sides else np.empty((0, 2))
        radius = float(model.geom_size[feet[active_sides[0]][0], 0]) if active_sides else 0.0
        margin = None if len(active_sides) == 0 else expanded_hull_margin(com[:2], centers, radius)
        foot_centroids = {side: np.mean(data.geom_xpos[feet[side], :2], axis=0) for side in feet}
        rows.append({
            "frame": frame, "time_s": frame / float(entry["fps"]), "intent": active_sides,
            "com_xy_m": com[:2].tolist(), "support_margin_m": margin,
            "foot_centroid_xy_m": {side: value.tolist() for side, value in foot_centroids.items()},
            "foot_separation_m": float(np.linalg.norm(foot_centroids["left"] - foot_centroids["right"])),
        })

    single = [row for row in rows if len(row["intent"]) == 1]
    double = [row for row in rows if len(row["intent"]) == 2]
    flight = [row for row in rows if len(row["intent"]) == 0]
    outside_gap = [max(0.0, -float(row["support_margin_m"])) for row in single]
    per_side = {}
    for side in ("left", "right"):
        selected = [row for row in single if row["intent"] == [side]]
        gaps = [max(0.0, -float(row["support_margin_m"])) for row in selected]
        per_side[side] = {
            "frame_count": len(selected),
            "outside_count": sum(value > 0 for value in gaps),
            "outside_fraction": None if not gaps else float(np.mean(np.asarray(gaps) > 0)),
            "outside_gap_p50_m": percentile(gaps, 50),
            "outside_gap_p95_m": percentile(gaps, 95),
            "outside_gap_max_m": max(gaps, default=None),
        }
    result = {
        "stage": "dynamic retargeting Phase13 quasi-static support-margin audit",
        "execution": {"mj_forward_calls": len(rows), "mj_step_calls": 0, "optimizer_steps": 0, "gpu": False},
        "assets": {"scene": str(scene), "scene_sha256": sha256(scene), "motion": str(motion), "motion_sha256": sha256(motion), "motion_id": motion_id},
        "truth_boundary": {
            "contact": "frozen Phase30 source-height intent, not realized collision/GRF/COP",
            "margin": "COM projection versus convex hull of intended active12 sole centers expanded by 5mm sphere radius",
            "interpretation": "quasi-static geometric diagnostic only; outside margin is not by itself a dynamic infeasibility proof",
        },
        "counts": {"frames": len(rows), "single_support": len(single), "double_support": len(double), "flight": len(flight)},
        "single_support": {
            "outside_count": sum(value > 0 for value in outside_gap),
            "outside_fraction": float(np.mean(np.asarray(outside_gap) > 0)) if outside_gap else None,
            "outside_gap_p50_m": percentile(outside_gap, 50),
            "outside_gap_p95_m": percentile(outside_gap, 95),
            "outside_gap_max_m": max(outside_gap, default=None),
            "per_side": per_side,
        },
        "foot_separation": {
            "p50_m": percentile([row["foot_separation_m"] for row in rows], 50),
            "p95_m": percentile([row["foot_separation_m"] for row in rows], 95),
            "max_m": max(row["foot_separation_m"] for row in rows),
        },
        "rows": rows,
    }
    result["decision"] = {
        "original_contact_schedule_is_quasi_statically_supported": bool(result["single_support"]["outside_fraction"] <= 0.05),
        "physics_or_teacher_unlocked": False,
        "next": "if outside support is systematic, re-optimize foot placement/contact schedule jointly before another dynamic bridge",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--motion-id", default="PHUMA-LUNGE-R-001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.scene, args.motion, args.motion_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "single_support": result["single_support"], "foot_separation": result["foot_separation"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
