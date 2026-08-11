#!/usr/bin/env python3
"""Local single-support reachability with the swing foot unconstrained."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "tools")]
import retarget.run_x2_forefoot_official_physics_screen as physics  # noqa: E402


LOWER15 = [
    *[f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
        "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"
    )],
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
]
MAX_DELTA_RAD = 0.35


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_run(values: list[str | None]) -> tuple[int, str | None, int | None]:
    best_length, best_value, best_start = 0, None, None
    start = 0
    while start < len(values):
        value = values[start]
        end = start + 1
        while end < len(values) and values[end] == value:
            end += 1
        if value is not None and end - start > best_length:
            best_length, best_value, best_start = end - start, value, start
        start = end
    return best_length, best_value, best_start


def run(scene: Path, motion: Path, phase15: Path, motion_id: str) -> dict:
    entry = joblib.load(motion)[motion_id]
    phase15_result = json.loads(phase15.read_text(encoding="utf-8"))
    scale = float(phase15_result["intervention"]["hip_roll_scale"])
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    q = np.asarray(entry["dof"], dtype=np.float64).copy()
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    names = list(entry["joint_names_mujoco"])
    name_to_col = {name: index for index, name in enumerate(names)}
    for name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        q[:, name_to_col[name]] *= scale
    qadr = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names])
    lower_jids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in LOWER15])
    lower_dofs = model.jnt_dofadr[lower_jids]
    lower_qadr = model.jnt_qposadr[lower_jids]
    lower_cols = np.asarray([name_to_col[name] for name in LOWER15])
    limits = model.jnt_range[lower_jids]
    _floor, helper = physics.foot_geom_contract(model)
    feet = {
        side: sorted(geom for geom in geoms if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE) and int(model.geom_contype[geom]) != 0)
        for side, geoms in helper.items()
    }
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rows = []
    selected: list[str | None] = []
    for frame in range(len(q)):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root[frame]
        data.qpos[3:7] = quat[frame][[3, 0, 1, 2]]
        data.qpos[qadr] = q[frame]
        mujoco.mj_forward(model, data)
        com = data.subtree_com[pelvis].copy()
        com_jac_full = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(model, data, com_jac_full, pelvis)
        per_side = {}
        for side, geoms in feet.items():
            jacobians = []
            for geom in geoms:
                jacp = np.zeros((3, model.nv), dtype=np.float64)
                jacr = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacGeom(model, data, jacp, jacr, geom)
                jacobians.append(jacp[:, lower_dofs])
            foot_jac = np.mean(jacobians, axis=0)
            foot = np.mean(data.geom_xpos[geoms], axis=0)
            task = np.vstack((foot_jac, com_jac_full[:2, lower_dofs]))
            desired = np.r_[np.zeros(3), foot[:2] - com[:2]]
            delta, *_ = np.linalg.lstsq(task, desired, rcond=None)
            residual = task @ delta - desired
            projected = q[frame, lower_cols] + delta
            overshoot = np.maximum(limits[:, 0] - projected, 0.0) + np.maximum(projected - limits[:, 1], 0.0)
            feasible = bool(np.max(np.abs(delta)) <= MAX_DELTA_RAD and np.max(overshoot) <= 1e-9 and np.max(np.abs(residual)) <= 1e-6)
            per_side[side] = {
                "delta_rms_rad": float(np.sqrt(np.mean(delta**2))),
                "delta_max_abs_rad": float(np.max(np.abs(delta))),
                "task_residual_max_abs_m": float(np.max(np.abs(residual))),
                "joint_limit_violation_count": int(np.sum(overshoot > 1e-9)),
                "joint_limit_overshoot_max_rad": float(np.max(overshoot)),
                "feasible": feasible,
            }
        feasible_sides = [side for side in ("left", "right") if per_side[side]["feasible"]]
        choice = min(feasible_sides, key=lambda side: per_side[side]["delta_max_abs_rad"]) if feasible_sides else None
        selected.append(choice)
        rows.append({"frame": frame, "time_s": frame / float(entry["fps"]), "per_side": per_side, "selected": choice})
    longest, longest_side, longest_start = longest_run(selected)
    side_distribution = {}
    for side in ("left", "right"):
        maximum = np.asarray([row["per_side"][side]["delta_max_abs_rad"] for row in rows])
        rms = np.asarray([row["per_side"][side]["delta_rms_rad"] for row in rows])
        violations = np.asarray([row["per_side"][side]["joint_limit_violation_count"] for row in rows])
        side_distribution[side] = {
            "delta_max_abs_min_rad": float(np.min(maximum)),
            "delta_max_abs_p50_rad": float(np.percentile(maximum, 50)),
            "delta_max_abs_p95_rad": float(np.percentile(maximum, 95)),
            "delta_rms_p50_rad": float(np.percentile(rms, 50)),
            "joint_limit_violation_frame_fraction": float(np.mean(violations > 0)),
            "best_frame": int(np.argmin(maximum)),
        }
    result = {
        "stage": "dynamic retargeting Phase16 local support reachability",
        "execution": {"mj_forward_calls": len(rows), "mj_step_calls": 0, "optimizer_steps": 0, "gpu": False},
        "assets": {"scene": str(scene), "scene_sha256": sha256(scene), "motion": str(motion), "motion_sha256": sha256(motion), "phase15": str(phase15), "phase15_sha256": sha256(phase15)},
        "contract": {
            "initialization": "Phase15 deterministic hip-roll normalization",
            "task_per_side": "candidate support-foot centroid xyz fixed + COM xy moved to that foot; swing foot unconstrained",
            "lower15_delta_max_abs_rad": MAX_DELTA_RAD,
            "joint_limits": True,
            "linearized_task_residual_max_m": 1e-6,
        },
        "summary": {
            "frames": len(rows),
            "left_feasible_frames": sum(row["per_side"]["left"]["feasible"] for row in rows),
            "right_feasible_frames": sum(row["per_side"]["right"]["feasible"] for row in rows),
            "either_feasible_frames": sum(row["selected"] is not None for row in rows),
            "longest_selected_support_run_frames": longest,
            "longest_selected_support_run_side": longest_side,
            "longest_selected_support_run_start": longest_start,
            "longest_selected_support_run_seconds": longest / float(entry["fps"]),
            "selected_support_transitions": sum(a != b for a, b in zip(selected[:-1], selected[1:])),
            "per_side_distribution": side_distribution,
        },
        "rows": rows,
    }
    result["decision"] = {
        "has_at_least_100ms_local_single_support_window": bool(longest / float(entry["fps"]) >= 0.10),
        "full_sequence_locally_covered": bool(all(value is not None for value in selected)),
        "physics_or_teacher_unlocked": False,
        "next": "local linear reachability is only an initialization certificate; any feasible window still needs temporally smooth joint/foot/root optimization and raw physics",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--phase15", type=Path, required=True)
    parser.add_argument("--motion-id", default="PHUMA-LUNGE-R-001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.scene, args.motion, args.phase15, args.motion_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
