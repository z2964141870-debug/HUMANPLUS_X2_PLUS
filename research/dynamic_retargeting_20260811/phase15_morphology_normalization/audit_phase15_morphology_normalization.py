#!/usr/bin/env python3
"""Deterministic X2 morphology-normalization counterfactual for Phase30 lunge."""

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
from official_x2.replay_official_trace_direct_mujoco import official_start_pose  # noqa: E402


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


def moving_average9(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, (4, 4), mode="edge")
    return np.convolve(padded, np.ones(9) / 9.0, mode="valid")


def run(scene: Path, motion: Path, motion_id: str) -> dict:
    entry = joblib.load(motion)[motion_id]
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    q = np.asarray(entry["dof"], dtype=np.float64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    names = list(entry["joint_names_mujoco"])
    name_to_col = {name: index for index, name in enumerate(names)}
    qadr = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names])
    jids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    phases = entry["phase30_time_dilation"]["resampled_source_contact_phases"]
    intent = {side: mask(len(q), phases[side]) for side in ("left", "right")}
    floor, helper = physics.foot_geom_contract(model)
    feet = {
        side: sorted(geom for geom in geoms if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE) and int(model.geom_contype[geom]) != 0)
        for side, geoms in helper.items()
    }
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    upper_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("torso_link", "head_pitch_link", "left_wrist_roll_link", "right_wrist_roll_link")]
    radius = float(model.geom_size[feet["left"][0], 0])

    def fk(q_row: np.ndarray, root_row: np.ndarray, quat_row: np.ndarray) -> dict:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root_row
        data.qpos[3:7] = quat_row[[3, 0, 1, 2]]
        data.qpos[qadr] = q_row
        mujoco.mj_forward(model, data)
        centers = {side: data.geom_xpos[geoms].copy() for side, geoms in feet.items()}
        return {
            "centers": centers, "centroids": {side: np.mean(value, axis=0) for side, value in centers.items()},
            "com": data.subtree_com[pelvis].copy(), "body": data.xpos.copy(),
            "clearance": {side: np.min(value[:, 2] - radius - data.geom_xpos[floor, 2]) for side, value in centers.items()},
        }

    neutral_q = np.zeros(len(names), dtype=np.float64)
    start = official_start_pose()
    for index, name in enumerate(names):
        neutral_q[index] = start.get(name, 0.0)
    neutral_fk = fk(neutral_q, np.asarray([0.0, 0.0, 0.68]), np.asarray([0.0, 0.0, 0.0, 1.0]))
    neutral_separation = float(np.linalg.norm(neutral_fk["centroids"]["left"][:2] - neutral_fk["centroids"]["right"][:2]))

    original_fk = [fk(q[i], root[i], quat[i]) for i in range(len(q))]
    original_separation = np.asarray([np.linalg.norm(row["centroids"]["left"][:2] - row["centroids"]["right"][:2]) for row in original_fk])
    scale = neutral_separation / float(np.median(original_separation))
    candidate_q = q.copy()
    for name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        column = name_to_col[name]
        candidate_q[:, column] = neutral_q[column] + scale * (q[:, column] - neutral_q[column])

    raw_correction = []
    for frame in range(len(q)):
        row = fk(candidate_q[frame], root[frame], quat[frame])
        raw_correction.append(0.00505 - min(row["clearance"].values()))
    root_candidate = root.copy()
    root_candidate[:, 2] += moving_average9(np.asarray(raw_correction))
    candidate_fk = [fk(candidate_q[i], root_candidate[i], quat[i]) for i in range(len(q))]

    single_margin, ds_margin, labels = [], [], []
    body_error, upper_error = [], []
    for frame, row in enumerate(candidate_fk):
        active = [side for side in ("left", "right") if intent[side][frame]]
        labels.append(active)
        ds_centers = np.concatenate([row["centers"][side][:, :2] for side in ("left", "right")], axis=0)
        ds_margin.append(margin(row["com"][:2], ds_centers, radius))
        if len(active) == 1:
            single_margin.append(margin(row["com"][:2], row["centers"][active[0]][:, :2], radius))
        body_error.extend(np.linalg.norm(row["body"] - original_fk[frame]["body"], axis=1).tolist())
        upper_error.extend(np.linalg.norm(row["body"][upper_ids] - original_fk[frame]["body"][upper_ids], axis=1).tolist())

    separation = np.asarray([np.linalg.norm(row["centroids"]["left"][:2] - row["centroids"]["right"][:2]) for row in candidate_fk])
    ranges = model.jnt_range[jids]
    violation = np.maximum(ranges[:, 0] - candidate_q, 0.0) + np.maximum(candidate_q - ranges[:, 1], 0.0)
    qstep = np.max(np.abs(np.diff(candidate_q, axis=0)), axis=1)
    result = {
        "stage": "dynamic retargeting Phase15 deterministic morphology normalization",
        "execution": {"mj_forward_calls": 1 + 3 * len(q), "mj_step_calls": 0, "optimizer_steps": 0, "parameter_search": False, "gpu": False},
        "assets": {"scene": str(scene), "scene_sha256": sha256(scene), "motion": str(motion), "motion_sha256": sha256(motion), "motion_id": motion_id},
        "intervention": {
            "only_joint_values_changed": ["left_hip_roll_joint", "right_hip_roll_joint"],
            "hip_roll_scale_formula": "official neutral foot separation / Phase30 median foot separation",
            "neutral_foot_separation_m": neutral_separation,
            "original_median_foot_separation_m": float(np.median(original_separation)),
            "hip_roll_scale": scale,
            "root_z": "Phase7-style MA9 correction to 5.05mm minimum active12 clearance",
            "root_xy_orientation_other_joints_unchanged": True,
        },
        "geometry": {
            "candidate_foot_separation_p50_m": float(np.percentile(separation, 50)),
            "candidate_foot_separation_p95_m": float(np.percentile(separation, 95)),
            "candidate_foot_separation_max_m": float(np.max(separation)),
            "single_support_frame_count": len(single_margin),
            "single_support_outside_count": int(np.sum(np.asarray(single_margin) < 0.0)),
            "single_support_outside_fraction": float(np.mean(np.asarray(single_margin) < 0.0)),
            "single_support_outside_gap_p50_m": float(np.percentile(np.maximum(-np.asarray(single_margin), 0.0), 50)),
            "single_support_outside_gap_p95_m": float(np.percentile(np.maximum(-np.asarray(single_margin), 0.0), 95)),
            "double_support_outside_count": int(np.sum(np.asarray(ds_margin) < 0.0)),
            "double_support_margin_min_m": float(np.min(ds_margin)),
        },
        "preservation": {
            "body_position_error_p95_m": float(np.percentile(body_error, 95)),
            "body_position_error_max_m": float(np.max(body_error)),
            "upper_four_position_error_max_m": float(np.max(upper_error)),
            "joint_limit_violation_count": int(np.sum(violation > 1e-9)),
            "joint_limit_overshoot_max_rad": float(np.max(violation)),
            "joint_step_p95_rad": float(np.percentile(qstep, 95)),
            "root_z_correction_max_abs_m": float(np.max(np.abs(root_candidate[:, 2] - root[:, 2]))),
        },
    }
    result["decision"] = {
        "deterministic_hip_roll_normalization_fixes_single_support": bool(result["geometry"]["single_support_outside_fraction"] <= 0.05),
        "deterministic_hip_roll_normalization_fixes_double_support": bool(result["geometry"]["double_support_outside_count"] == 0),
        "physics_or_teacher_unlocked": False,
        "next": "if still systematic, a two-joint morphology scale is insufficient; use joint foot-placement/root/contact optimization",
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
    print(json.dumps({"intervention": result["intervention"], "geometry": result["geometry"], "preservation": result["preservation"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
