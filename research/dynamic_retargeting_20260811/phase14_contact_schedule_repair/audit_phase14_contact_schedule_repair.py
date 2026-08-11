#!/usr/bin/env python3
"""Repair only contact labels using X2 quasi-static support margins."""

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


SUPPORTS = (("left",), ("right",), ("left", "right"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask(length: int, phases: list[list[int]]) -> np.ndarray:
    out = np.zeros(length, dtype=bool)
    for start, end in phases:
        out[int(start):int(end)] = True
    return out


def margin(point: np.ndarray, centers: np.ndarray, radius: float) -> float:
    equations = ConvexHull(centers).equations
    distances = -(equations[:, :2] @ point + equations[:, 2]) / np.linalg.norm(equations[:, :2], axis=1)
    return float(np.min(distances) + radius)


def transitions(labels: list[list[str]]) -> int:
    return sum(left != right for left, right in zip(labels[:-1], labels[1:]))


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
    intent = {side: mask(len(q), phases[side]) for side in ("left", "right")}
    floor, helper = physics.foot_geom_contract(model)
    feet = {
        side: sorted(geom for geom in geoms if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE) and int(model.geom_contype[geom]) != 0)
        for side, geoms in helper.items()
    }
    if any(len(value) != 12 for value in feet.values()):
        raise ValueError("active12 contract failed")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    radius = float(model.geom_size[feet["left"][0], 0])
    rows, original_labels, repaired_labels = [], [], []
    for frame in range(len(q)):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root[frame]
        data.qpos[3:7] = quat[frame][[3, 0, 1, 2]]
        data.qpos[qadr] = q[frame]
        mujoco.mj_forward(model, data)
        com = data.subtree_com[pelvis, :2].copy()
        support_margin = {}
        for support in SUPPORTS:
            centers = np.concatenate([data.geom_xpos[feet[side], :2] for side in support], axis=0)
            support_margin["+".join(support)] = margin(com, centers, radius)
        original = [side for side in ("left", "right") if intent[side][frame]]
        original_key = "+".join(original)
        feasible = [list(support) for support in SUPPORTS if support_margin["+".join(support)] >= 0.0]
        if original and support_margin.get(original_key, -np.inf) >= 0.0:
            repaired = original
        elif feasible:
            original_set = set(original)
            repaired = min(feasible, key=lambda value: (len(set(value) ^ original_set), len(value), value))
        else:
            repaired = []
        min_clearance = {
            side: float(min(data.geom_xpos[geom, 2] - model.geom_size[geom, 0] - data.geom_xpos[floor, 2] for geom in feet[side]))
            for side in feet
        }
        original_labels.append(original)
        repaired_labels.append(repaired)
        rows.append({
            "frame": frame, "time_s": frame / float(entry["fps"]), "original": original,
            "repaired": repaired, "support_margin_m": support_margin,
            "sole_min_clearance_m": min_clearance,
        })

    changed = [left != right for left, right in zip(original_labels, repaired_labels)]
    unresolved = [not value for value in repaired_labels]
    repaired_counts = {label: sum(value == label.split("+") for value in repaired_labels) for label in ("left", "right", "left+right")}
    ds_margins = [row["support_margin_m"]["left+right"] for row in rows]
    result = {
        "stage": "dynamic retargeting Phase14 contact-schedule-only repair",
        "execution": {"mj_forward_calls": len(rows), "mj_step_calls": 0, "optimizer_steps": 0, "gpu": False},
        "assets": {"scene": str(scene), "scene_sha256": sha256(scene), "motion": str(motion), "motion_sha256": sha256(motion), "motion_id": motion_id},
        "contract": {
            "q_root_foot_placement_modified": False,
            "candidate_labels": ["left", "right", "left+right"],
            "selection": "preserve original if quasi-static margin>=0; else minimum symmetric-difference feasible label",
            "support_geometry": "official active12 sphere-center convex hull expanded by 5mm radius",
        },
        "summary": {
            "frames": len(rows), "changed_frames": sum(changed), "changed_fraction": float(np.mean(changed)),
            "unresolved_frames": sum(unresolved), "repaired_counts": repaired_counts,
            "original_transitions": transitions(original_labels), "repaired_transitions": transitions(repaired_labels),
            "double_support_margin_min_m": min(ds_margins),
            "double_support_margin_p05_m": float(np.percentile(ds_margins, 5)),
        },
        "rows": rows,
    }
    result["decision"] = {
        "label_only_quasi_static_repair_exists": bool(not any(unresolved)),
        "label_only_repair_is_small": bool(np.mean(changed) <= 0.20),
        "physics_or_teacher_unlocked": False,
        "next": "a label-only all/mostly-DS result is diagnostic, not a teacher; jointly change foot placement/root while preserving motion semantics",
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
    print(json.dumps({"summary": result["summary"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
