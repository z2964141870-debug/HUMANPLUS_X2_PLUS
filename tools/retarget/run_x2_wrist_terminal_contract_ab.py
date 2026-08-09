#!/usr/bin/env python3
"""A/B the upper-hierarchy wrist terminal against the official X2 contract.

The legacy positive upper-hierarchy branch targets ``wrist_yaw_link`` while
the frozen AimDK-v1 WBT contract observes ``wrist_roll_link``.  This script
changes that semantic terminal as one coupled contract variable (target body
and X2 forearm length), reruns the same three AMASS clips, and measures both
canonical wrist error and lower-body collateral change.  It never trains and
never edits the legacy GMR package or source motions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[2]
LEGACY = Path("/home/humanplus/x2_teleop_final")
GMR_TOOLS = LEGACY / "x2_sonic/x2_gmr_data_rebuild/tools"
for root in (LEGACY, GMR_TOOLS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import general_motion_retargeting.motion_retarget as gmr_motion_retarget
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
from retarget_upper_hierarchy_panel import UpperHierarchyGMR, retarget_one
from x2_sonic.tools.retarget_smplx_subset_to_x2_gmr_cache import (
    X2_MUJOCO_ORDER_31,
    load_smplx_file_compat,
    parse_joint_axes,
)


OFFICIAL = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/"
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
)
PANEL = REPO / "reports/retarget/x2_wbt_diagnostic_panel.json"
BASE_CONFIG = LEGACY / "x2_sonic/x2_gmr_data_rebuild/candidates/smplx_to_x2_v4_upper_hierarchy.json"
BODY_MODELS = Path("/home/humanplus/gmr-motionlab/assets/body_models")
INPUT_ROOT = LEGACY / "x2_sonic/data/raw"
OUTPUT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_wrist_terminal_smoke3"
)
REPORT = REPO / "reports/retarget/x2_wrist_terminal_contract_ab.json"
SMOKE_IDS = ("AMASS-UPPER-001", "AMASS-WALK-001", "AMASS-SQUAT-001")
SIDES = ("left", "right")
LOWER_WAIST = {
    f"{side}_{joint}_joint"
    for side in SIDES
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
} | {"waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"missing official X2 body: {name}")
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def segment_lengths(mjcf: Path, terminal: str) -> dict[str, dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis = body_position(model, data, "pelvis")
    result: dict[str, dict[str, float]] = {}
    for side in SIDES:
        shoulder = body_position(model, data, f"{side}_shoulder_roll_link")
        elbow = body_position(model, data, f"{side}_elbow_link")
        wrist = body_position(model, data, f"{side}_{terminal}_link")
        result[side] = {
            "pelvis_to_shoulder": float(np.linalg.norm(shoulder - pelvis)),
            "shoulder_to_elbow": float(np.linalg.norm(elbow - shoulder)),
            "elbow_to_wrist": float(np.linalg.norm(wrist - elbow)),
        }
    return result


def make_config(base: Path, terminal: str, output: Path) -> Path:
    config = copy.deepcopy(json.loads(base.read_text(encoding="utf-8")))
    if terminal == "wrist_roll":
        for side in SIDES:
            old = f"{side}_wrist_yaw_link"
            new = f"{side}_wrist_roll_link"
            config["ik_match_table1"][new] = config["ik_match_table1"].pop(old)
    elif terminal != "wrist_yaw":
        raise ValueError(terminal)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def retarget_variant(
    name: str,
    rows: list[dict[str, Any]],
    config: Path,
    lengths: dict[str, dict[str, float]],
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    directory = output_root / name
    cache = directory / f"x2_{name}_smoke3.pkl"
    if cache.exists():
        return joblib.load(cache), cache
    original_xml = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(OFFICIAL)
    axes = parse_joint_axes(OFFICIAL)
    entries: dict[str, Any] = {}
    try:
        for row in rows:
            source = Path(row["source_path"])
            key, entry, _ = retarget_one(
                row={
                    "relative_path": str(source.relative_to(INPUT_ROOT)),
                    "segment_start_s": 0.0,
                    "segment_duration_s": 0.0,
                },
                input_root=INPUT_ROOT,
                smplx_models=BODY_MODELS,
                mjcf=OFFICIAL,
                ik_config=config,
                target_fps=30,
                smooth_window=9,
                lengths=lengths,
                joint_axes=axes,
            )
            entry["diagnostic_id"] = row["id"]
            entry["terminal_contract"] = name
            entries[key] = entry
    finally:
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original_xml
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(entries, cache, compress=True)
    return entries, cache


def canonical_targets(
    source: Path,
    canonical_config: Path,
    canonical_lengths: dict[str, dict[str, float]],
    frame_count: int,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    source_data, body_model, output, human_height = load_smplx_file_compat(
        source, BODY_MODELS, segment_start_s=0.0, segment_duration_s=0.0
    )
    frames, _ = get_smplx_data_offline_fast(source_data, body_model, output, tgt_fps=30)
    original_config = gmr_motion_retarget.IK_CONFIG_DICT["smplx"]["agibot_x2"]
    original_xml = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.IK_CONFIG_DICT["smplx"]["agibot_x2"] = canonical_config
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(OFFICIAL)
    try:
        retargeter = UpperHierarchyGMR(
            src_human="smplx",
            tgt_robot="agibot_x2",
            actual_human_height=float(human_height),
            solver="quadprog",
            damping=0.5,
            verbose=False,
            use_velocity_limit=False,
            upper_segment_lengths=canonical_lengths,
        )
    finally:
        gmr_motion_retarget.IK_CONFIG_DICT["smplx"]["agibot_x2"] = original_config
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original_xml
    result = []
    for frame in frames[:frame_count]:
        retargeter.update_targets(frame, offset_to_ground=True)
        result.append({
            name: (np.asarray(value[0]).copy(), np.asarray(value[1]).copy())
            for name, value in retargeter.scaled_human_data.items()
        })
    return result


def set_frame(model: mujoco.MjModel, data: mujoco.MjData, entry: dict[str, Any], index: int) -> None:
    data.qpos[:3] = np.asarray(entry["root_trans_offset"][index], dtype=np.float64)
    quat = np.asarray(entry["root_rot"][index], dtype=np.float64)
    data.qpos[3:7] = (quat[3], quat[0], quat[1], quat[2])
    data.qpos[7:38] = np.asarray(entry["dof"][index], dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def analyze_variant(
    entries: dict[str, Any],
    canonical_config: Path,
    canonical_lengths: dict[str, dict[str, float]],
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL))
    data = mujoco.MjData(model)
    motion_rows = []
    all_position: dict[str, list[float]] = {side: [] for side in SIDES}
    all_rotation: dict[str, list[float]] = {side: [] for side in SIDES}
    all_steps: list[float] = []
    for key, entry in entries.items():
        dof = np.asarray(entry["dof"], dtype=np.float64)
        targets = canonical_targets(
            Path(entry["source_smplx_file"]), canonical_config, canonical_lengths, len(dof)
        )
        frame_count = min(len(dof), len(targets))
        xy_origin = targets[0]["pelvis"][0][:2].copy()
        by_side = {side: {"position": [], "rotation": []} for side in SIDES}
        for index in range(frame_count):
            set_frame(model, data, entry, index)
            for side in SIDES:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link"
                )
                target_pos, target_quat = targets[index][f"{side}_wrist"]
                target_pos = target_pos.copy()
                target_pos[:2] -= xy_origin
                position_error = float(np.linalg.norm(data.xpos[body_id] - target_pos))
                actual = Rotation.from_quat(data.xquat[body_id], scalar_first=True)
                target = Rotation.from_quat(target_quat, scalar_first=True)
                rotation_error = float((target.inv() * actual).magnitude())
                by_side[side]["position"].append(position_error)
                by_side[side]["rotation"].append(rotation_error)
                all_position[side].append(position_error)
                all_rotation[side].append(rotation_error)
        steps = np.max(np.abs(np.diff(dof[:frame_count], axis=0)), axis=1)
        all_steps.extend(steps.tolist())
        motion_rows.append({
            "key": key,
            "diagnostic_id": entry["diagnostic_id"],
            "frames": frame_count,
            "joint_step_abs_rad": stats(steps),
            "wrist_roll_position_error_m": {
                side: stats(by_side[side]["position"]) for side in SIDES
            },
            "wrist_roll_rotation_error_rad": {
                side: stats(by_side[side]["rotation"]) for side in SIDES
            },
        })
    return {
        "motions": motion_rows,
        "aggregate": {
            "joint_step_abs_rad": stats(all_steps),
            "wrist_roll_position_error_m": {side: stats(all_position[side]) for side in SIDES},
            "wrist_roll_rotation_error_rad": {side: stats(all_rotation[side]) for side in SIDES},
        },
    }


def compare_lower_body(yaw: dict[str, Any], roll: dict[str, Any]) -> dict[str, float]:
    indices = [i for i, name in enumerate(X2_MUJOCO_ORDER_31) if name in LOWER_WAIST]
    differences = []
    for key in sorted(yaw):
        a = np.asarray(yaw[key]["dof"], dtype=np.float64)
        b = np.asarray(roll[key]["dof"], dtype=np.float64)
        n = min(len(a), len(b))
        differences.append((a[:n, indices] - b[:n, indices]).reshape(-1))
    delta = np.concatenate(differences)
    return {
        "count": int(delta.size),
        "rms_rad": float(np.sqrt(np.mean(np.square(delta)))),
        "p95_abs_rad": float(np.percentile(np.abs(delta), 95)),
        "max_abs_rad": float(np.max(np.abs(delta))),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    yaw = report["variants"]["wrist_yaw_terminal"]["aggregate"]
    roll = report["variants"]["wrist_roll_terminal"]["aggregate"]
    lines = [
        "# X2 Official Wrist Terminal Contract A/B",
        "",
        "## 裁决",
        "",
        f"- candidate gate: **{'PASS' if report['gate']['pass'] else 'FAIL'}**。",
        f"- canonical wrist-roll position p95（左右最差）：`{report['gate']['baseline_worst_wrist_p95_m']:.4f} → {report['gate']['candidate_worst_wrist_p95_m']:.4f} m`。",
        f"- joint-step p95：`{yaw['joint_step_abs_rad']['p95']:.4f} → {roll['joint_step_abs_rad']['p95']:.4f} rad/frame`。",
        f"- 腿腰 collateral RMS：`{report['lower_waist_collateral']['rms_rad']:.5f} rad`。",
        "- 这里只验证运动学契约；没有训练，也没有把结果称为动态 Silver。",
        "",
        "## 单变量",
        "",
        "旧正收益 upper-hierarchy 的终端由 `wrist_yaw_link` 改为官方 WBT 观测契约的 `wrist_roll_link`；与该终端对应的 X2 elbow→wrist 几何长度同步更新。源动作、IK 权重、root、fps、平滑和求解器不变。",
        "",
        "| variant | L wrist p95 m | R wrist p95 m | joint-step p95/max rad |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in report["variants"].items():
        a = result["aggregate"]
        lines.append(
            f"| `{name}` | {a['wrist_roll_position_error_m']['left']['p95']:.5f} | "
            f"{a['wrist_roll_position_error_m']['right']['p95']:.5f} | "
            f"{a['joint_step_abs_rad']['p95']:.5f}/{a['joint_step_abs_rad']['max']:.5f} |"
        )
    lines += [
        "",
        "## 门禁",
        "",
        f"- wrist p95 至少改善 5%：`{report['gate']['wrist_p95_improves_5pct']}`",
        f"- joint-step p95 不劣化超过 10%：`{report['gate']['joint_step_p95_nonregression']}`",
        f"- 腿腰差异 RMS ≤ 0.03 rad：`{report['gate']['lower_waist_rms_bounded']}`",
        "",
        "大 cache 保存在 Git 外；本报告只记录方法、哈希和结果。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in panel["motions"]}
    rows = [by_id[motion_id] for motion_id in SMOKE_IDS]
    args.output_root.mkdir(parents=True, exist_ok=True)

    yaw_config = make_config(BASE_CONFIG, "wrist_yaw", args.output_root / "wrist_yaw_terminal.json")
    roll_config = make_config(BASE_CONFIG, "wrist_roll", args.output_root / "wrist_roll_terminal.json")
    yaw_lengths = segment_lengths(OFFICIAL, "wrist_yaw")
    roll_lengths = segment_lengths(OFFICIAL, "wrist_roll")
    yaw_entries, yaw_cache = retarget_variant(
        "wrist_yaw_terminal", rows, yaw_config, yaw_lengths, args.output_root
    )
    roll_entries, roll_cache = retarget_variant(
        "wrist_roll_terminal", rows, roll_config, roll_lengths, args.output_root
    )
    variants = {
        "wrist_yaw_terminal": analyze_variant(yaw_entries, roll_config, roll_lengths),
        "wrist_roll_terminal": analyze_variant(roll_entries, roll_config, roll_lengths),
    }
    collateral = compare_lower_body(yaw_entries, roll_entries)
    yaw_p95 = max(
        variants["wrist_yaw_terminal"]["aggregate"]["wrist_roll_position_error_m"][side]["p95"]
        for side in SIDES
    )
    roll_p95 = max(
        variants["wrist_roll_terminal"]["aggregate"]["wrist_roll_position_error_m"][side]["p95"]
        for side in SIDES
    )
    yaw_step = variants["wrist_yaw_terminal"]["aggregate"]["joint_step_abs_rad"]["p95"]
    roll_step = variants["wrist_roll_terminal"]["aggregate"]["joint_step_abs_rad"]["p95"]
    gate = {
        "baseline_worst_wrist_p95_m": yaw_p95,
        "candidate_worst_wrist_p95_m": roll_p95,
        "wrist_p95_improves_5pct": roll_p95 <= 0.95 * yaw_p95,
        "joint_step_p95_nonregression": roll_step <= 1.10 * yaw_step,
        "lower_waist_rms_bounded": collateral["rms_rad"] <= 0.03,
    }
    gate["pass"] = bool(
        gate["wrist_p95_improves_5pct"]
        and gate["joint_step_p95_nonregression"]
        and gate["lower_waist_rms_bounded"]
    )
    report = {
        "schema_version": 1,
        "status": "completed_no_training",
        "smoke_ids": list(SMOKE_IDS),
        "fixed_contract": {
            "official_mjcf": str(OFFICIAL),
            "fps": 30,
            "root_rotation": "full",
            "root_z": "gmr",
            "smooth_window": 9,
            "solver": "quadprog",
            "damping": 0.5,
        },
        "assets": {
            "official_mjcf_sha256": sha256(OFFICIAL),
            "base_upper_hierarchy_config_sha256": sha256(BASE_CONFIG),
            "yaw_config_sha256": sha256(yaw_config),
            "roll_config_sha256": sha256(roll_config),
            "yaw_cache": str(yaw_cache),
            "roll_cache": str(roll_cache),
        },
        "segment_lengths_m": {"wrist_yaw": yaw_lengths, "wrist_roll": roll_lengths},
        "variants": variants,
        "lower_waist_collateral": collateral,
        "gate": gate,
        "training": False,
        "postprocess": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.report.with_suffix(".md"))
    print(json.dumps({"report": str(args.report), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
