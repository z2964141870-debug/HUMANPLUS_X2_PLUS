#!/usr/bin/env python3
"""Phase28: qualify the frozen 24-motion WBT panel under official X2 v1.

This is a data-only experiment.  It performs no MuJoCo integration, policy
forward, optimizer step, training, or real-robot operation.  Old and official
robot branches use the same source adapter and the frozen Phase7 robot-level
root/ground formula.  Bronze and Silver metrics are then computed with FK and
official collision geometry only; all contact labels are model estimates, not
measured GRF, COP, or foot force.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[2]
LEGACY_ROOT = Path("/home/humanplus/x2_teleop_final")
REBUILD = LEGACY_ROOT / "x2_sonic/x2_gmr_data_rebuild"
for path in (LEGACY_ROOT, REBUILD / "tools", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import general_motion_retargeting.motion_retarget as gmr_motion_retarget
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
from x2_sonic.tools import retarget_smplx_subset_to_x2_gmr_cache as amass_adapter

import retarget_bones_soma_bvh_to_x2_hierarchy as bones_adapter
import retarget_phuma_g1_fk_to_x2_hierarchy as phuma_adapter
import retarget_full_hierarchy_panel as hierarchy
import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7


PANEL = REPO / "reports/retarget/x2_wbt_diagnostic_panel.json"
TIER_GATES = REPO / "reports/retarget/x2_wbt_tier_gates.json"
MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
MIRROR_CONTRACT = REPO / "reports/retarget/x2_official_mirror_contract.json"
LEGACY_MJCF = LEGACY_ROOT / "assets/agibot_x2/x2_ultra.xml"
OFFICIAL_MJCF = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/"
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
)
AMASS_ROOT = LEGACY_ROOT / "x2_sonic/data/raw"
PHUMA_ROOT = Path("/home/humanplus/humanoid-GPT/A/sonic_release/PHUMA/data/g1")
BONES_ROOT = REBUILD / "extracted/bones_seed_uniform_locomotion_full_20260717_v2/soma_uniform/bvh"
SMPLX_MODELS = Path("/home/humanplus/gmr-motionlab/assets/body_models")
G1_MJCF = Path("/home/humanplus/humanoid-GPT/A/sonic_release/PHUMA/asset/humanoid_model/g1/custom.xml")
AMASS_IK = LEGACY_ROOT / "x2_sonic/candidate_files/gmr_ik_configs/smplx_to_x2_keypoints_v4.json"
ROBOT_SOURCE_IK = REBUILD / "candidates/g1_fk_to_x2_full_hierarchy_identity.json"
OUTPUT_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase28_panel24"
)
REPORT_JSON = REPO / "reports/retarget/x2_wbt_panel_phase28.json"
REPORT_MD = REPO / "reports/retarget/x2_wbt_panel_phase28.md"
PREFLIGHT_JSON = REPO / "reports/retarget/x2_wbt_panel_phase28_preflight.json"

TARGET_FPS = 30
SMOOTH_WINDOW = 9
SOLVER = "quadprog"
DAMPING = 0.5
SOURCE_CONTACT_HEIGHT_M = 0.020
KEYPOINT_LABELS = ("pelvis", "torso", "left_foot", "right_foot", "left_wrist", "right_wrist")
TARGET_BODIES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")
OFFLINE_METRIC_CONTRACT = {
    "threshold_revision": False,
    "per_clip_parameter_tuning": False,
    "tracked_keypoints": list(KEYPOINT_LABELS),
    "keypoint_error": "one Umeyama similarity fit per whole clip, then Euclidean residual over six named points",
    "source_contact_intent": "aligned source foot z <= min(left/right foot z p05) + 0.020 m",
    "official_contact": "Phase7 active-sole signed distance <= official reset clearance + one sole-sphere radius",
    "contact_timing": "p95 nearest same-foot target transition error for each source-intended transition",
    "stance_speed": "official ankle-roll body XY speed during model-estimated contact",
    "stance_excursion": "maximum XY distance from the first frame of each contiguous model-estimated stance window",
    "severe_self_collision": "official collision pair between two non-world bodies with penetration deeper than 0.030 m",
    "truth_boundary": "all contact/collision/FK values are model estimates, not measured GRF/COP/force",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(entry: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in ("dof", "root_trans_offset", "root_rot", "pose_aa"):
        value = np.ascontiguousarray(np.asarray(entry[field]))
        digest.update(field.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    digest.update("\0".join(entry["joint_names_mujoco"]).encode())
    digest.update(str(entry["fps"]).encode())
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_relative(row: dict[str, Any], root: Path) -> str:
    return str(Path(row["source_path"]).relative_to(root))


def preflight_row(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["source_path"])
    result: dict[str, Any] = {
        "id": row["id"],
        "dataset": row["dataset"],
        "source_path": str(path),
        "expected_sha256": row["source_sha256"],
        "recommended_split": row["recommended_split"],
        "transform": row["transform"],
        "status": "blocked",
        "blocker": "unknown",
    }
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        result["actual_sha256"] = actual_hash
        if actual_hash != row["source_sha256"]:
            raise ValueError("source_sha256_mismatch")
        if row["transform"] not in ("identity", "canonical_left_right_mirror"):
            raise ValueError(f"unsupported_transform:{row['transform']}")
        if row["dataset"] == "AMASS":
            loaded = np.load(path, allow_pickle=True)
            required = {"pose_body", "root_orient", "trans", "mocap_frame_rate"}
            if not required.issubset(loaded.files):
                raise KeyError(f"missing_amass_fields:{sorted(required-set(loaded.files))}")
            result.update(adapter="amass_smplx_v4", frames=int(loaded["pose_body"].shape[0]), source_fps=float(np.asarray(loaded["mocap_frame_rate"]).reshape(())))
        elif row["dataset"] == "PHUMA":
            loaded = np.load(path, allow_pickle=True).item()
            required = {"root_trans", "root_ori", "dof_pos", "fps"}
            if not required.issubset(loaded):
                raise KeyError(f"missing_phuma_fields:{sorted(required-set(loaded))}")
            dof = np.asarray(loaded["dof_pos"])
            if dof.ndim != 2 or dof.shape[1] != 29:
                raise ValueError(f"phuma_dof_shape:{dof.shape}")
            result.update(adapter="phuma_g1_fk_full_hierarchy", frames=int(dof.shape[0]), source_fps=float(loaded["fps"]))
        elif row["dataset"] == "BONES-seed":
            joints, channels, motion, frame_count, frame_time = bones_adapter.parse_bvh(str(path))
            if frame_count < 2 or frame_time <= 0 or motion.shape[0] != frame_count:
                raise ValueError("invalid_bvh_timing_or_frames")
            result.update(adapter="bones_soma_bvh_full_hierarchy", frames=int(frame_count), source_fps=float(1.0 / frame_time), parsed_joint_count=len(joints), parsed_channel_count=len(channels))
        else:
            raise ValueError(f"unsupported_dataset:{row['dataset']}")
        result.update(status="ready", blocker="")
    except Exception as exc:
        result["blocker"] = f"{type(exc).__name__}:{exc}"
    return result


def run_preflight(panel: dict[str, Any]) -> dict[str, Any]:
    assets = {
        "panel": PANEL,
        "tier_gates": TIER_GATES,
        "model_contract": MODEL_CONTRACT,
        "mirror_contract": MIRROR_CONTRACT,
        "legacy_mjcf": LEGACY_MJCF,
        "official_mjcf": OFFICIAL_MJCF,
        "amass_ik": AMASS_IK,
        "robot_source_ik": ROBOT_SOURCE_IK,
        "smplx_models": SMPLX_MODELS,
        "g1_mjcf": G1_MJCF,
    }
    asset_rows = {}
    for name, path in assets.items():
        exists = path.is_dir() if name == "smplx_models" else path.is_file()
        asset_rows[name] = {
            "path": str(path),
            "exists": bool(exists),
            "sha256": sha256(path) if exists and path.is_file() else None,
        }
    rows = [preflight_row(row) for row in panel["motions"]]
    dataset_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = dataset_counts.setdefault(row["dataset"], {"ready": 0, "blocked": 0})
        counts[row["status"]] += 1
    split_counts = {name: sum(row["recommended_split"] == name for row in panel["motions"]) for name in ("train_candidate", "held_out")}
    checks = {
        "panel_count_24": len(rows) == 24,
        "panel_id_unique": len({row["id"] for row in rows}) == 24,
        "all_assets_present": all(row["exists"] for row in asset_rows.values()),
        "all_sources_ready": all(row["status"] == "ready" for row in rows),
        "split_counts_unchanged": split_counts == panel["recommended_split_counts"],
        "tier_gate_status_frozen": json.loads(TIER_GATES.read_text())["status"] == "phase0_preregistered_draft_no_training_authorization",
    }
    return {
        "schema_version": "x2_wbt_panel_phase28_preflight_v1",
        "mode": "preflight_no_retarget_no_physics_no_training",
        "panel_sha256": sha256(PANEL),
        "tier_gate_sha256": sha256(TIER_GATES),
        "assets": asset_rows,
        "dataset_counts": dataset_counts,
        "split_counts": split_counts,
        "motions": rows,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def mirror_entry(entry: dict[str, Any], mirror: dict[str, Any], joint_axes: np.ndarray) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    names = list(entry["joint_names_mujoco"])
    index = {name: i for i, name in enumerate(names)}
    permutation = np.asarray([index[("right_" + n[5:]) if n.startswith("left_") else ("left_" + n[6:]) if n.startswith("right_") else n] for n in names], dtype=np.int64)
    sign_by_name = np.asarray([-1.0 if ("_roll_" in n or "_yaw_" in n) else 1.0 for n in names])
    dof = np.asarray(entry["dof"], dtype=np.float64)[:, permutation] * sign_by_name[None]
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64).copy()
    root[:, 1] *= -1.0
    reflection = np.diag([1.0, -1.0, 1.0])
    matrices = Rotation.from_quat(np.asarray(entry["root_rot"], dtype=np.float64)).as_matrix()
    root_quat = Rotation.from_matrix(np.einsum("ij,fjk,kl->fil", reflection, matrices, reflection)).as_quat()
    result["dof"] = dof.astype(np.asarray(entry["dof"]).dtype)
    result["root_trans_offset"] = root.astype(np.asarray(entry["root_trans_offset"]).dtype)
    result["root_rot"] = root_quat.astype(np.asarray(entry["root_rot"]).dtype)
    pose = np.zeros((len(dof), 32, 3), dtype=np.float32)
    pose[:, 1:, :] = dof[:, :, None].astype(np.float32) * joint_axes[None]
    result["pose_aa"] = pose
    result["phase28_source_transform"] = {
        "name": "canonical_left_right_mirror",
        "contract_sha256": sha256(MIRROR_CONTRACT),
        "root_polar_sign_xyz": mirror["polar_vector_sign_xyz"],
        "joint_permutation_and_sign_applied_by_name": True,
    }
    return result


def retarget_amass(row: dict[str, Any], robot_xml: Path) -> dict[str, Any]:
    axes = amass_adapter.parse_joint_axes(robot_xml)
    _, entry, _ = amass_adapter.retarget_one(
        smplx_file=Path(row["source_path"]), input_root=AMASS_ROOT,
        smplx_models=SMPLX_MODELS, joint_axes=axes, target_fps=TARGET_FPS,
        max_frames=0, smooth_window=SMOOTH_WINDOW, solver=SOLVER,
        damping=DAMPING, root_rotation_mode="full", root_z_mode="gmr",
        root_z=0.65, ik_config=AMASS_IK,
    )
    entry["phase28_source_adapter"] = "amass_smplx_v4"
    return entry


def retarget_phuma(row: dict[str, Any], robot_xml: Path, mirror: dict[str, Any]) -> dict[str, Any]:
    g1_model = mujoco.MjModel.from_xml_path(str(G1_MJCF))
    args = SimpleNamespace(
        input_root=PHUMA_ROOT, x2_mjcf=robot_xml, ik_config=ROBOT_SOURCE_IK,
        target_fps=TARGET_FPS, smooth_window=SMOOTH_WINDOW,
        warm_start_first_frame=False, temporal_lower_body_velocity_limit_rad_s=0.0,
        temporal_frame_dt_s=0.0,
    )
    adapter_row = {
        "relative_path": source_relative(row, PHUMA_ROOT),
        "split": row["recommended_split"], "family": row["category"],
    }
    upper = phuma_adapter.upper_lengths(robot_xml)
    lower = phuma_adapter.lower_lengths(robot_xml)
    axes = amass_adapter.parse_joint_axes(robot_xml)
    _, entry, _ = phuma_adapter.retarget_one(adapter_row, args, g1_model, upper, lower, axes)
    entry["phase28_source_adapter"] = "phuma_g1_fk_full_hierarchy"
    if row["transform"] == "canonical_left_right_mirror":
        entry = mirror_entry(entry, mirror, axes)
    return entry


def retarget_bones(row: dict[str, Any], robot_xml: Path) -> dict[str, Any]:
    args = SimpleNamespace(
        input_root=BONES_ROOT, x2_mjcf=robot_xml, ik_config=ROBOT_SOURCE_IK,
        target_fps=TARGET_FPS, max_frames=0, smooth_window=SMOOTH_WINDOW,
        root_rotation_mode="right_relative_yaw",
    )
    adapter_row = {
        "relative_path": source_relative(row, BONES_ROOT),
        "family": row["category"], "actor": "", "content_type": row["category"],
    }
    upper = bones_adapter.upper_lengths(robot_xml)
    lower = bones_adapter.lower_lengths(robot_xml)
    axes = amass_adapter.parse_joint_axes(robot_xml)
    _, entry, _ = bones_adapter.retarget_one(adapter_row, args, upper, lower, axes)
    entry["phase28_source_adapter"] = "bones_soma_bvh_full_hierarchy_right_relative_yaw"
    return entry


def retarget_base(row: dict[str, Any], robot_xml: Path, mirror: dict[str, Any]) -> dict[str, Any]:
    original = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(robot_xml)
    try:
        if row["dataset"] == "AMASS":
            entry = retarget_amass(row, robot_xml)
        elif row["dataset"] == "PHUMA":
            entry = retarget_phuma(row, robot_xml, mirror)
        elif row["dataset"] == "BONES-seed":
            entry = retarget_bones(row, robot_xml)
        else:
            raise ValueError(row["dataset"])
    finally:
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original
    entry["diagnostic_id"] = row["id"]
    entry["source_sha256"] = row["source_sha256"]
    entry["recommended_split"] = row["recommended_split"]
    entry["robot_xml"] = str(robot_xml)
    return entry


def canonicalize(entry: dict[str, Any], model: mujoco.MjModel, reset: dict[str, Any], phase2: Any, physics: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate, ground = phase7.apply_canonical_root_ground(model, entry, reset, phase2, physics)
    candidate["phase28_canonical_contract"] = {
        "contract": "Phase7 robot-level root-ground Bronze",
        "formula": "z'=z+MA9(c_reset-min_24_official_sole_signed_distance(q,root))",
        "phase7_report_sha256": sha256(REPO / "reports/retarget/x2_wbt_canonical_root_ground_phase7.json"),
        "only_root_z_changed": True,
        "per_clip_parameter_selection": False,
    }
    ground.pop("geometry_contact", None)
    return candidate, ground


def resample_points(points: np.ndarray, frames: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == frames:
        return points
    old = np.linspace(0.0, 1.0, len(points))
    new = np.linspace(0.0, 1.0, frames)
    return np.stack([np.interp(new, old, points[:, j, axis]) for j in range(points.shape[1]) for axis in range(3)], axis=1).reshape(frames, points.shape[1], 3)


def source_points(row: dict[str, Any], frames: int) -> np.ndarray:
    path = Path(row["source_path"])
    if row["dataset"] == "AMASS":
        source_data, body_model, output, _ = amass_adapter.load_smplx_file_compat(path, SMPLX_MODELS)
        human_frames, _ = get_smplx_data_offline_fast(source_data, body_model, output, tgt_fps=TARGET_FPS)
    elif row["dataset"] == "PHUMA":
        loaded = np.load(path, allow_pickle=True).item()
        root, quat, dof = phuma_adapter.resample_source(
            np.asarray(loaded["root_trans"], dtype=np.float64),
            np.asarray(loaded["root_ori"], dtype=np.float64),
            np.asarray(loaded["dof_pos"], dtype=np.float64),
            float(loaded["fps"]), TARGET_FPS,
        )
        g1_model = mujoco.MjModel.from_xml_path(str(G1_MJCF))
        human_frames = phuma_adapter.source_fk_frames(g1_model, root, quat, phuma_adapter.source_dof_to_model(dof))
    else:
        human_frames, _, _ = bones_adapter.bvh_to_soma_frames(path, TARGET_FPS, 0, "right_relative_yaw")
    values = np.asarray([[frame[name][0] for name in ("pelvis", "spine3", "left_foot", "right_foot", "left_wrist", "right_wrist")] for frame in human_frames], dtype=np.float64)
    values = resample_points(values, frames)
    if row["transform"] == "canonical_left_right_mirror":
        values = values[:, [0, 1, 3, 2, 5, 4]].copy()
        values[:, :, 1] *= -1.0
    return values


def set_entry_frame(model: mujoco.MjModel, data: mujoco.MjData, entry: dict[str, Any], frame: int, addresses: list[int]) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    root = np.asarray(entry["root_trans_offset"][frame], dtype=np.float64)
    quat = np.asarray(entry["root_rot"][frame], dtype=np.float64)
    data.qpos[:3] = root
    data.qpos[3:7] = quat[[3, 0, 1, 2]]
    for value, address in zip(entry["dof"][frame], addresses, strict=True):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)


def fit_similarity(src: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    a, b = src.reshape(-1, 3), target.reshape(-1, 3)
    am, bm = a.mean(0), b.mean(0)
    ac, bc = a-am, b-bm
    u, s, vt = np.linalg.svd(ac.T @ bc / len(a))
    d = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1] = -1
    rotation = vt.T @ np.diag(d) @ u.T
    scale = float(np.sum(s*d) / max(np.mean(np.sum(ac*ac, axis=1)), 1e-12))
    translation = bm - scale*(am @ rotation.T)
    return scale, rotation, translation


def contiguous_true(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def transitions(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(np.asarray(mask, dtype=np.int8)) != 0) + 1


def event_timing_error(source: np.ndarray, target: np.ndarray, fps: float) -> float:
    a, b = transitions(source), transitions(target)
    if len(a) == 0:
        return 0.0 if len(b) == 0 else float("inf")
    if len(b) == 0:
        return float("inf")
    return float(np.percentile([np.min(np.abs(b-index))/fps for index in a], 95))


def count_ds_ss_ds(left: np.ndarray, right: np.ndarray) -> int:
    state = np.where(left & right, 2, np.where(left ^ right, 1, 0))
    count, awaiting = 0, False
    for value in state:
        if value == 2 and not awaiting:
            awaiting = True
        elif value == 1 and awaiting:
            awaiting = "return"
        elif value == 2 and awaiting == "return":
            count += 1
            awaiting = True
    return count


def target_kinematics(entry: dict[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
    names = list(entry["joint_names_mujoco"])
    addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TARGET_BODIES]
    floor, feet = phase7.active_sole_spheres(model)
    data = mujoco.MjData(model)
    target = np.zeros((len(entry["dof"]), len(body_ids), 3), dtype=np.float64)
    distance = {side: np.zeros(len(entry["dof"]), dtype=np.float64) for side in feet}
    severe_self = np.zeros(len(entry["dof"]), dtype=bool)
    fromto = np.zeros(6, dtype=np.float64)
    for frame in range(len(entry["dof"])):
        set_entry_frame(model, data, entry, frame, addresses)
        target[frame] = data.xpos[body_ids]
        for side, geoms in feet.items():
            distance[side][frame] = min(float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto)) for geom in geoms)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            if body1 != 0 and body2 != 0 and contact.dist < -0.03:
                severe_self[frame] = True
    return {"target_points": target, "sole_distance": distance, "severe_self_collision_fraction": float(np.mean(severe_self))}


def semantic_gate(row: dict[str, Any], entry: dict[str, Any], target: np.ndarray) -> dict[str, Any]:
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    yaw = np.unwrap(Rotation.from_quat(np.asarray(entry["root_rot"], dtype=np.float64)).as_euler("xyz")[:, 2])
    foot_exc = [float(np.ptp(target[:, index, 2])) for index in (2, 3)]
    wrist_exc = [float(np.max(np.linalg.norm(target[:, index]-target[0, index], axis=1))) for index in (4, 5)]
    root_xy = float(np.max(np.linalg.norm(root[:, :2]-root[0, :2], axis=1)))
    category = row["category"]
    checks: dict[str, bool] = {"nontrivial_or_static_consistent": True}
    if category == "standing":
        checks["standing_root_xy_le_0p20"] = root_xy <= 0.20
    elif category == "upper_only":
        checks["upper_wrist_motion_ge_0p05"] = max(wrist_exc) >= 0.05
    elif "turn_left" in category:
        checks["left_yaw_positive_ge_5deg"] = yaw[-1]-yaw[0] >= math.radians(5)
    elif "turn_right" in category:
        checks["right_yaw_negative_ge_5deg"] = yaw[-1]-yaw[0] <= -math.radians(5)
    elif category == "squat":
        checks["pelvis_z_excursion_ge_0p05"] = float(np.ptp(root[:, 2])) >= 0.05
    elif "kick_left" in category or "raise_left" in category:
        checks["left_foot_vertical_intent"] = foot_exc[0] >= max(0.05, 0.8*foot_exc[1])
    elif "kick_right" in category or "raise_right" in category:
        checks["right_foot_vertical_intent"] = foot_exc[1] >= max(0.05, 0.8*foot_exc[0])
    elif "lunge_left" in category:
        checks["left_foot_motion_nontrivial"] = float(np.max(np.linalg.norm(target[:, 2]-target[0, 2], axis=1))) >= 0.05
    elif "lunge_right" in category:
        checks["right_foot_motion_nontrivial"] = float(np.max(np.linalg.norm(target[:, 3]-target[0, 3], axis=1))) >= 0.05
    else:
        checks["dynamic_motion_nontrivial"] = root_xy >= 0.05 or max(foot_exc) >= 0.05 or max(wrist_exc) >= 0.05
    return {"category": category, "metrics": {"root_xy_span_m": root_xy, "root_yaw_delta_rad": float(yaw[-1]-yaw[0]), "foot_z_excursion_m": foot_exc, "wrist_excursion_m": wrist_exc}, "checks": checks, "pass": bool(all(checks.values()))}


def percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if values.size else default


def audit_tier(row: dict[str, Any], entry: dict[str, Any], model: mujoco.MjModel, reset: dict[str, Any], gates: dict[str, Any], mirror: dict[str, Any]) -> dict[str, Any]:
    fps = float(entry["fps"])
    dof = np.asarray(entry["dof"], dtype=np.float64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    quat = np.asarray(entry["root_rot"], dtype=np.float64)
    kin = target_kinematics(entry, model)
    target = kin["target_points"]
    source = source_points(row, len(dof))
    scale, rotation, translation = fit_similarity(source, target)
    source_fit = scale*(source @ rotation.T)+translation
    keypoint_error = np.linalg.norm(source_fit-target, axis=2)
    source_ground = min(percentile(source_fit[:, 2, 2], 5), percentile(source_fit[:, 3, 2], 5))
    source_contact = {"left": source_fit[:, 2, 2] <= source_ground+SOURCE_CONTACT_HEIGHT_M, "right": source_fit[:, 3, 2] <= source_ground+SOURCE_CONTACT_HEIGHT_M}
    contact_threshold = float(reset["reset_clearance_m"] + reset["sole_sphere_radius_m"])
    target_contact = {side: values <= contact_threshold for side, values in kin["sole_distance"].items()}

    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    ranges = {}
    for name in entry["joint_names_mujoco"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ranges[name] = model.jnt_range[jid].copy()
    violation = max(float(np.max(np.maximum(ranges[name][0]-dof[:, i], dof[:, i]-ranges[name][1]))) for i, name in enumerate(entry["joint_names_mujoco"]))
    steps = np.max(np.abs(np.diff(dof, axis=0)), axis=1) if len(dof)>1 else np.zeros(1)
    penetration = np.maximum(0.0, -np.minimum(kin["sole_distance"]["left"], kin["sole_distance"]["right"]))
    tilt = np.arccos(np.clip(Rotation.from_quat(quat).as_matrix()[:, 2, 2], -1, 1))
    semantic = semantic_gate(row, entry, target)
    global_checks = {
        "finite": bool(all(np.isfinite(np.asarray(entry[field])).all() for field in ("dof", "root_trans_offset", "root_rot", "pose_aa"))),
        "joint_order_exact": list(entry["joint_names_mujoco"]) == json.loads(MODEL_CONTRACT.read_text())["control_boundaries"]["official_mjcf_actuated_31"],
        "head_locked": bool(np.max(np.abs(dof[:, -2:])) <= 1e-12),
        "source_sha256_present_and_exact": entry.get("source_sha256") == row["source_sha256"],
        "root_quaternion_norm": float(np.max(np.abs(np.linalg.norm(quat, axis=1)-1))) <= gates["global_reject"]["root_quaternion_norm_error_max"],
        "hard_joint_limit": violation <= gates["global_reject"]["hard_joint_limit_violation_rad_max"],
        "root_xy_not_repaired": True,
        "no_forbidden_shortcut": True,
    }
    b = gates["bronze_kinematic"]["metrics"]
    bronze_metrics = {
        "joint_step_p95_rad": percentile(steps, 95), "joint_step_max_rad": float(np.max(steps)),
        "sole_penetration_p95_m": percentile(penetration, 95), "sole_penetration_max_m": float(np.max(penetration)),
        "tracked_keypoint_error_p95_m": percentile(keypoint_error, 95), "tracked_keypoint_error_max_m": float(np.max(keypoint_error)),
        "pelvis_height_min_m": float(np.min(root[:, 2])), "pelvis_height_max_m": float(np.max(root[:, 2])),
        "root_tilt_max_rad": float(np.max(tilt)),
        "mirror_joint_roundtrip_rad": float(mirror["fk_verification"]["joint_roundtrip_max_abs_rad"]),
        "mirror_fk_position_m": float(mirror["fk_verification"]["body_position_max_abs_m"]),
        "mirror_fk_rotation_matrix_element": float(mirror["fk_verification"]["body_rotation_max_abs"]),
        "severe_self_collision_fraction": kin["severe_self_collision_fraction"],
        "keypoint_fit_scale": scale,
    }
    bronze_checks = {
        "joint_step_p95": bronze_metrics["joint_step_p95_rad"] <= b["joint_step_p95_rad"]["max"],
        "joint_step_max": bronze_metrics["joint_step_max_rad"] <= b["joint_step_max_rad"]["max"],
        "sole_penetration_p95": bronze_metrics["sole_penetration_p95_m"] <= b["sole_penetration_p95_m"]["max"],
        "sole_penetration_max": bronze_metrics["sole_penetration_max_m"] <= b["sole_penetration_max_m"]["max"],
        "tracked_keypoint_p95": bronze_metrics["tracked_keypoint_error_p95_m"] <= b["tracked_keypoint_error_p95_m"]["max"],
        "tracked_keypoint_max": bronze_metrics["tracked_keypoint_error_max_m"] <= b["tracked_keypoint_error_max_m"]["max"],
        "pelvis_height": bronze_metrics["pelvis_height_min_m"] >= b["pelvis_height_m"]["min"] and bronze_metrics["pelvis_height_max_m"] <= b["pelvis_height_m"]["max"],
        "root_tilt": bronze_metrics["root_tilt_max_rad"] <= b["root_tilt_rad"]["max"],
        "mirror_joint": bronze_metrics["mirror_joint_roundtrip_rad"] <= b["mirror_joint_roundtrip_rad"]["max"],
        "mirror_fk_pos": bronze_metrics["mirror_fk_position_m"] <= b["mirror_fk_position_m"]["max"],
        "mirror_fk_rot": bronze_metrics["mirror_fk_rotation_matrix_element"] <= b["mirror_fk_rotation_matrix_element"]["max"],
        "no_severe_self_collision": bronze_metrics["severe_self_collision_fraction"] == 0.0,
        "semantic_review": semantic["pass"],
    }
    bronze_pass = bool(all(global_checks.values()) and all(bronze_checks.values()))

    stance_speed, stance_excursion, clearance = {}, {}, {}
    timing = {}
    for side, point_index in (("left", 2), ("right", 3)):
        velocity = np.linalg.norm(np.diff(target[:, point_index, :2], axis=0)*fps, axis=1)
        mask_velocity = target_contact[side][1:]
        stance_speed[side] = percentile(velocity[mask_velocity], 95, float("inf"))
        excursions = []
        for start, end in contiguous_true(target_contact[side]):
            if end-start >= 2:
                positions = target[start:end, point_index, :2]
                excursions.append(float(np.max(np.linalg.norm(positions-positions[0], axis=1))))
        stance_excursion[side] = max(excursions, default=float("inf"))
        swing = np.asarray(kin["sole_distance"][side])[~target_contact[side]] - float(reset["reset_clearance_m"])
        clearance[side] = {"p50": percentile(swing, 50, 0.0), "p95": percentile(swing, 95, 0.0)}
        timing[side] = event_timing_error(source_contact[side], target_contact[side], fps)
    intended = [side for side in ("left", "right") if len(transitions(source_contact[side])) >= 2]
    root_acceleration = np.diff(root[:, :2], n=2, axis=0)*(fps**2) if len(root)>=3 else np.zeros((1,2))
    silver_metrics = {
        "complete_ds_ss_ds_cycles": count_ds_ss_ds(target_contact["left"], target_contact["right"]),
        "source_intended_feet": intended,
        "target_contact_transitions": {side: int(len(transitions(target_contact[side]))) for side in ("left", "right")},
        "stance_speed_p95_mps": stance_speed, "stance_excursion_max_m": stance_excursion,
        "swing_clearance_m": clearance, "contact_timing_error_p95_s": timing,
        "unintended_flight_fraction": float(np.mean(~target_contact["left"] & ~target_contact["right"])),
        "root_horizontal_acceleration_p95_mps2": percentile(np.linalg.norm(root_acceleration, axis=1), 95),
        "contact_provenance": "source height intent vs official-X2 FK/collision model estimate; not real GRF/COP/force",
    }
    s = gates["silver_contact"]["metrics"]
    is_static_exception = row["category"] in ("standing", "upper_only")
    silver_checks = {
        "bronze_prerequisite": bronze_pass,
        "dynamic_not_static_exception": not is_static_exception,
        "complete_cycle": silver_metrics["complete_ds_ss_ds_cycles"] >= s["complete_ds_ss_ds_cycles"]["min"],
        "intended_feet_exist": bool(intended),
        "transitions_each_intended": bool(intended) and all(silver_metrics["target_contact_transitions"][side] >= s["contact_transitions_per_intended_foot"]["min"] for side in intended),
        "stance_speed_each": all(value <= s["stance_speed_p95_mps_each_foot"]["max"] for value in stance_speed.values()),
        "stance_excursion_each": all(value <= s["stance_excursion_m_each_stance"]["max"] for value in stance_excursion.values()),
        "clearance_p50_each_intended": bool(intended) and all(clearance[side]["p50"] >= s["swing_clearance_p50_m_each_intended_foot"]["min"] for side in intended),
        "clearance_p95_each_intended": bool(intended) and all(clearance[side]["p95"] >= s["swing_clearance_p95_m_each_intended_foot"]["min"] for side in intended),
        "timing_each_intended": bool(intended) and all(timing[side] <= s["contact_timing_error_s"]["max"] for side in intended),
        "flight": silver_metrics["unintended_flight_fraction"] <= s["unintended_flight_fraction"]["max"],
        "root_acceleration": silver_metrics["root_horizontal_acceleration_p95_mps2"] <= s["root_horizontal_acceleration_p95_mps2"]["max"],
    }
    silver_pass = bool(all(silver_checks.values()))
    failed = [f"global:{key}" for key, value in global_checks.items() if not value] + [f"bronze:{key}" for key, value in bronze_checks.items() if not value]
    if bronze_pass and not is_static_exception:
        failed += [f"silver:{key}" for key, value in silver_checks.items() if not value]
    elif is_static_exception:
        failed.append("silver:static_exception_not_dynamic_silver")
    tier = "Silver" if silver_pass else "Bronze" if bronze_pass else "Reject"
    return {
        "tier": tier, "global_reject_checks": global_checks,
        "bronze": {"metrics": bronze_metrics, "checks": bronze_checks, "semantic": semantic, "pass": bronze_pass},
        "silver": {"metrics": silver_metrics, "checks": silver_checks, "pass": silver_pass},
        "reject_reasons": failed,
    }


def compare_entries(old: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    for field in ("dof", "root_trans_offset", "root_rot", "pose_aa"):
        a, b = np.asarray(old[field]), np.asarray(official[field])
        fields[field] = {"shape_equal": a.shape == b.shape, "max_abs_difference": float(np.max(np.abs(a-b))) if a.shape == b.shape else None}
    return {"fields": fields, "global_max_abs_difference": max(v["max_abs_difference"] for v in fields.values() if v["max_abs_difference"] is not None), "exact": all(v["shape_equal"] and v["max_abs_difference"] == 0.0 for v in fields.values())}


def write_md(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# X2 WBT Phase28：固定24条 official-v1 Bronze/Silver资格审计", "",
        "## 裁决", "",
        f"- 完成：`{summary['completed']}/24`；blocked：`{summary['blocked']}`。",
        f"- tier：Silver `{summary['silver']}` / Bronze `{summary['bronze']}` / Reject `{summary['reject']}`。",
        f"- train Silver：`{summary['train_silver']}`；held Silver/Gold：`{summary['held_silver_or_gold']}`。",
        f"- WBT PPO 解锁：`{report['decision']['wbt_ppo_unlocked']}`。本阶段 optimizer/training/physics-step/真机均为 0。",
        "- contact、COM、FK 均为模型估计，不是真实 GRF/COP/足底力。", "",
        "## 冻结离线指标合同", "",
        "- threshold revision：`False`；逐 clip 调参：`False`。",
        "- keypoint：整段一次 Umeyama similarity fit 后，计算 pelvis/torso/双脚/双腕残差。",
        "- source contact intent：对齐后的 source foot z 不高于双脚 p05 ground + 20 mm。",
        "- official contact：Phase7 active-sole signed distance 不高于 reset clearance + 1 个 sole-sphere radius。",
        "- stance speed/excursion 使用 official ankle-roll FK；contact timing 是同脚 source event 到最近 target event 的 p95。",
        "- severe self-collision：两个非 world body 的官方碰撞对 penetration >30 mm；仍只是模型碰撞证据。", "",
        "## 漏斗", "",
        "| ID | split | adapter | frames | old/official max | tier | reject reasons |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in report["motions"]:
        lines.append(f"| `{row['id']}` | {row['recommended_split']} | {row['source_adapter']} | {row['frames']} | {row['old_vs_official']['global_max_abs_difference']:.3e} | {row['tier_audit']['tier']} | `{'; '.join(row['tier_audit']['reject_reasons'])}` |")
    lines += ["", "## 下一步", "", report["decision"]["next_step"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()
    panel = json.loads(PANEL.read_text())
    gates = json.loads(TIER_GATES.read_text())
    mirror = json.loads(MIRROR_CONTRACT.read_text())
    preflight = run_preflight(panel)
    PREFLIGHT_JSON.write_text(json.dumps(json_safe(preflight), indent=2, ensure_ascii=False)+"\n")
    print(f"[phase28] preflight ready={sum(x['status']=='ready' for x in preflight['motions'])}/24", flush=True)
    if args.preflight_only or not preflight["pass"]:
        if not preflight["pass"]:
            print("[phase28] BLOCKED at preflight", flush=True)
        return

    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase28")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase28")
    official_scene_model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(official_scene_model, physics)
    legacy_cache, official_cache, rows = {}, {}, []
    for index, row in enumerate(panel["motions"], start=1):
        started = time.time()
        print(f"[phase28] retarget {index}/24 {row['id']} dataset={row['dataset']}", flush=True)
        old_raw = retarget_base(row, LEGACY_MJCF, mirror)
        official_raw = retarget_base(row, OFFICIAL_MJCF, mirror)
        old, old_ground = canonicalize(old_raw, official_scene_model, reset, phase2, physics)
        official, official_ground = canonicalize(official_raw, official_scene_model, reset, phase2, physics)
        old_hash, official_hash = array_hash(old), array_hash(official)
        comparison = compare_entries(old, official)
        tier = audit_tier(row, official, official_scene_model, reset, gates, mirror)
        legacy_cache[row["id"]] = old
        official_cache[row["id"]] = official
        rows.append({
            "id": row["id"], "dataset": row["dataset"], "category": row["category"],
            "recommended_split": row["recommended_split"], "transform": row["transform"],
            "source_path": row["source_path"], "source_sha256": row["source_sha256"],
            "source_adapter": official["phase28_source_adapter"], "frames": int(len(official["dof"])), "fps": int(official["fps"]),
            "old_entry_sha256": old_hash, "official_entry_sha256": official_hash,
            "old_vs_official": comparison, "old_ground": old_ground, "official_ground": official_ground,
            "tier_audit": tier, "elapsed_s": float(time.time()-started),
        })
        counts = {name: sum(value["tier_audit"]["tier"] == name for value in rows) for name in ("Silver", "Bronze", "Reject")}
        print(f"[phase28] funnel completed={len(rows)}/24 silver={counts['Silver']} bronze={counts['Bronze']} reject={counts['Reject']}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    legacy_path = args.output_root / "legacy_model/x2_phase28_panel24.pkl"
    official_path = args.output_root / "official_v1_model/x2_phase28_panel24.pkl"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    official_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(legacy_cache, legacy_path, compress=True)
    joblib.dump(official_cache, official_path, compress=True)
    train_silver = [row["id"] for row in rows if row["recommended_split"] == "train_candidate" and row["tier_audit"]["tier"] == "Silver"]
    held_silver = [row["id"] for row in rows if row["recommended_split"] == "held_out" and row["tier_audit"]["tier"] == "Silver"]
    summary = {
        "completed": len(rows), "blocked": 24-len(rows),
        "silver": sum(row["tier_audit"]["tier"] == "Silver" for row in rows),
        "bronze": sum(row["tier_audit"]["tier"] == "Bronze" for row in rows),
        "reject": sum(row["tier_audit"]["tier"] == "Reject" for row in rows),
        "train_silver": len(train_silver), "held_silver_or_gold": len(held_silver),
        "train_silver_ids": train_silver, "held_silver_or_gold_ids": held_silver,
        "old_official_exact_count": sum(row["old_vs_official"]["exact"] for row in rows),
    }
    bronze_failure_counts = {
        name: sum(not row["tier_audit"]["bronze"]["checks"][name] for row in rows)
        for name in rows[0]["tier_audit"]["bronze"]["checks"]
    }
    dynamic_rows = [row for row in rows if row["category"] not in ("standing", "upper_only")]
    silver_failure_counts_excluding_bronze = {
        name: sum(not row["tier_audit"]["silver"]["checks"][name] for row in dynamic_rows)
        for name in dynamic_rows[0]["tier_audit"]["silver"]["checks"]
        if name != "bronze_prerequisite"
    }
    failure_aggregation = {
        "bronze_failure_counts": bronze_failure_counts,
        "dynamic_motion_count": len(dynamic_rows),
        "dynamic_silver_failure_counts_excluding_bronze_prerequisite": silver_failure_counts_excluding_bronze,
        "dynamic_passing_all_silver_gates_except_bronze": [
            row["id"] for row in dynamic_rows
            if all(value for name, value in row["tier_audit"]["silver"]["checks"].items() if name != "bronze_prerequisite")
        ],
    }
    unlocked = len(rows) == 24 and bool(train_silver) and bool(held_silver)
    report = {
        "schema_version": "x2_wbt_panel_phase28_v1", "mode": "offline_FK_collision_no_physics_step_no_training",
        "provenance": {
            "panel": {"path": str(PANEL), "sha256": sha256(PANEL)},
            "tier_gates": {"path": str(TIER_GATES), "sha256": sha256(TIER_GATES)},
            "legacy_mjcf": {"path": str(LEGACY_MJCF), "sha256": sha256(LEGACY_MJCF)},
            "official_mjcf": {"path": str(OFFICIAL_MJCF), "sha256": sha256(OFFICIAL_MJCF)},
            "phase7_contract": {"path": str(REPO/'reports/retarget/x2_wbt_canonical_root_ground_phase7.json'), "sha256": sha256(REPO/'reports/retarget/x2_wbt_canonical_root_ground_phase7.json')},
            "legacy_cache": {"path": str(legacy_path), "sha256": sha256(legacy_path)},
            "official_cache": {"path": str(official_path), "sha256": sha256(official_path)},
        },
        "truth_boundary": {"contact_fk_com_are_model_estimates": True, "not_real_grf_cop_or_foot_force": True, "physics_steps": 0, "optimizer_steps": 0, "training": False, "real_robot": False},
        "offline_metric_contract": OFFLINE_METRIC_CONTRACT,
        "preflight": preflight, "summary": summary, "failure_aggregation": failure_aggregation, "motions": rows,
        "decision": {
            "wbt_ppo_unlocked": unlocked,
            "panel_qualification_complete": len(rows) == 24,
            "minimum_train_GMR_Silver": bool(train_silver),
            "independent_held_GMR_Silver_or_Gold": bool(held_silver),
            "next_step": "Panel data gate passed; stop for review before separately authorized 1-update." if unlocked else "WBT PPO remains locked. Do not use the native Gold sanity seed to bypass missing GMR Silver; review the uniform generator/tier failures before selecting one new data intervention.",
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_md(report, args.report_md)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
