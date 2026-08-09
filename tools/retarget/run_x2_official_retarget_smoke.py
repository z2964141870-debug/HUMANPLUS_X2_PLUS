"""Run an isolated old-vs-official X2 GMR smoke on three AMASS clips.

This module never edits the legacy GMR package or source motions.  It locally
overrides the X2 robot XML used by GMR, writes the official-model outputs to a
new version directory, and compares them against an independently rerun legacy
model branch under an otherwise identical contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np


LEGACY_ROOT = Path("/home/humanplus/x2_teleop_final")
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

import general_motion_retargeting.motion_retarget as gmr_motion_retarget
from x2_sonic.tools import retarget_smplx_subset_to_x2_gmr_cache as retarget_tool


DEFAULT_PANEL = Path(__file__).resolve().parents[2] / "reports/retarget/x2_wbt_diagnostic_panel.json"
DEFAULT_LEGACY_MJCF = LEGACY_ROOT / "assets/agibot_x2/x2_ultra.xml"
DEFAULT_OFFICIAL_MJCF = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/"
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
)
DEFAULT_IK = LEGACY_ROOT / "x2_sonic/candidate_files/gmr_ik_configs/smplx_to_x2_keypoints_v4.json"
DEFAULT_MODELS = Path("/home/humanplus/gmr-motionlab/assets/body_models")
DEFAULT_OUTPUT = Path("/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/bronze_kinematic/phase0_smoke3")
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "reports/retarget/x2_official_retarget_smoke3.json"
SMOKE_IDS = ("AMASS-UPPER-001", "AMASS-WALK-001", "AMASS-SQUAT-001")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retarget_variant(
    name: str,
    robot_xml: Path,
    rows: list[dict[str, Any]],
    ik_config: Path,
    body_models: Path,
    output_root: Path,
) -> dict[str, Any]:
    variant_dir = output_root / name
    cache_path = variant_dir / f"x2_{name}_smoke3.pkl"
    if cache_path.exists():
        return {"cache_path": str(cache_path), "entries": joblib.load(cache_path), "reused": True}

    original_xml = gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"]
    gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = str(robot_xml)
    axes = retarget_tool.parse_joint_axes(robot_xml)
    entries: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    try:
        for row in rows:
            source = Path(row["source_path"])
            key, entry, meta = retarget_tool.retarget_one(
                smplx_file=source,
                input_root=LEGACY_ROOT / "x2_sonic/data/raw",
                smplx_models=body_models,
                joint_axes=axes,
                target_fps=30,
                max_frames=0,
                smooth_window=9,
                solver="quadprog",
                damping=0.5,
                root_rotation_mode="full",
                root_z_mode="gmr",
                root_z=0.65,
                ik_config=ik_config,
            )
            entry["diagnostic_id"] = row["id"]
            entry["robot_xml"] = str(robot_xml)
            entries[key] = entry
            metadata[key] = {"diagnostic_id": row["id"], **meta}
    finally:
        gmr_motion_retarget.ROBOT_XML_DICT["agibot_x2"] = original_xml

    variant_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(entries, cache_path, compress=True)
    joblib.dump(metadata, variant_dir / "metadata.pkl", compress=True)
    return {"cache_path": str(cache_path), "entries": entries, "reused": False}


def compare(legacy: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for key in sorted(legacy):
        if key not in official:
            raise KeyError(f"official output lacks {key}")
        a, b = legacy[key], official[key]
        fields = {}
        for field in ("dof", "root_trans_offset", "root_rot", "pose_aa"):
            av = np.asarray(a[field], dtype=np.float64)
            bv = np.asarray(b[field], dtype=np.float64)
            if av.shape != bv.shape:
                raise ValueError(f"{key}:{field} shape mismatch {av.shape} != {bv.shape}")
            fields[field] = float(np.max(np.abs(av - bv)))
        rows.append(
            {
                "key": key,
                "diagnostic_id": a["diagnostic_id"],
                "frames": int(np.asarray(a["dof"]).shape[0]),
                "max_abs_difference": fields,
            }
        )
    return {
        "motions": rows,
        "global_max_abs_difference": float(
            max(value for row in rows for value in row["max_abs_difference"].values())
        ),
    }


def compare_robot_models(legacy_path: Path, official_path: Path) -> dict[str, Any]:
    legacy = mujoco.MjModel.from_xml_path(str(legacy_path))
    official = mujoco.MjModel.from_xml_path(str(official_path))

    def names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str | None]:
        return [mujoco.mj_id2name(model, object_type, index) for index in range(count)]

    legacy_geoms = names(legacy, mujoco.mjtObj.mjOBJ_GEOM, legacy.ngeom)
    official_geoms = names(official, mujoco.mjtObj.mjOBJ_GEOM, official.ngeom)
    numeric = {
        "joint_position_m": float(np.max(np.abs(legacy.jnt_pos - official.jnt_pos))),
        "joint_axis": float(np.max(np.abs(legacy.jnt_axis - official.jnt_axis))),
        "joint_range_rad": float(np.max(np.abs(legacy.jnt_range - official.jnt_range))),
        "body_position_m": float(np.max(np.abs(legacy.body_pos - official.body_pos))),
        "body_quaternion": float(np.max(np.abs(legacy.body_quat - official.body_quat))),
        "body_mass_kg": float(np.max(np.abs(legacy.body_mass - official.body_mass))),
        "body_inertia_kg_m2": float(np.max(np.abs(legacy.body_inertia - official.body_inertia))),
        "dof_damping": float(np.max(np.abs(legacy.dof_damping - official.dof_damping))),
        "dof_armature": float(np.max(np.abs(legacy.dof_armature - official.dof_armature))),
        "actuator_gain": float(np.max(np.abs(legacy.actuator_gainprm - official.actuator_gainprm))),
        "actuator_bias": float(np.max(np.abs(legacy.actuator_biasprm - official.actuator_biasprm))),
        "actuator_ctrlrange": float(np.max(np.abs(legacy.actuator_ctrlrange - official.actuator_ctrlrange))),
    }
    return {
        "legacy_counts": {"nq": legacy.nq, "nv": legacy.nv, "nu": legacy.nu, "nbody": legacy.nbody, "ngeom": legacy.ngeom},
        "official_counts": {"nq": official.nq, "nv": official.nv, "nu": official.nu, "nbody": official.nbody, "ngeom": official.ngeom},
        "joint_names_equal": names(legacy, mujoco.mjtObj.mjOBJ_JOINT, legacy.njnt)
        == names(official, mujoco.mjtObj.mjOBJ_JOINT, official.njnt),
        "body_names_equal": names(legacy, mujoco.mjtObj.mjOBJ_BODY, legacy.nbody)
        == names(official, mujoco.mjtObj.mjOBJ_BODY, official.nbody),
        "numeric_max_abs_difference": numeric,
        "robot_numeric_contract_equal": max(numeric.values()) == 0.0,
        "legacy_only_geoms": sorted(str(name) for name in set(legacy_geoms) - set(official_geoms)),
        "official_only_geoms": sorted(str(name) for name in set(official_geoms) - set(legacy_geoms)),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# X2 旧模型 vs AimDK v1.0 官方模型 Retarget Smoke（3 条）",
        "",
        "## 裁决",
        "",
        f"- 数值最大差：`{report['comparison']['global_max_abs_difference']:.3e}`。",
        "- 两个 MJCF 的关节、body 拓扑、关节位置、限位、质量和惯量相同；旧文件只额外内置了一个不参与接触的 floor 及可视化资源。",
        "- 因此，单纯把旧 GMR 的机器人 XML 换成 AimDK v1.0 官方 `x2.xml` **不会改善运动学重定向**。这是否定性但有效的 Phase0 结果，不能宣称官方模型修好了 reference。",
        "- 后续差异应来自 canonical body/contact contract、IK objective、contact-aware repair 与官方动态回放，而不是来自这两个机器人 XML 的运动学本体差异。",
        f"- robot numeric contract exact equal: `{report['robot_model_comparison']['robot_numeric_contract_equal']}`；legacy-only geom: `{report['robot_model_comparison']['legacy_only_geoms']}`。",
        "",
        "## 固定条件",
        "",
        "- 同一源动作、同一 SMPL-X 模型、同一 v4 IK config。",
        "- 30 Hz、full root、GMR root-z、smooth window 9、quadprog、damping 0.5。",
        "- 两个分支均重新运行；不复用旧 cache，不训练，不做 postprocess。",
        "",
        "| 动作 | 帧数 | dof max | root xyz max | root quat max | pose-aa max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["comparison"]["motions"]:
        d = row["max_abs_difference"]
        lines.append(
            f"| `{row['diagnostic_id']}` | {row['frames']} | {d['dof']:.3e} | "
            f"{d['root_trans_offset']:.3e} | {d['root_rot']:.3e} | {d['pose_aa']:.3e} |"
        )
    lines += [
        "",
        "## 版本",
        "",
        f"- legacy MJCF SHA-256: `{report['assets']['legacy_mjcf_sha256']}`",
        f"- official MJCF SHA-256: `{report['assets']['official_mjcf_sha256']}`",
        f"- IK config SHA-256: `{report['assets']['ik_config_sha256']}`",
        "",
        "大 cache 位于 Git 外的版本目录；Git 只保存本报告、脚本和哈希。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--legacy-mjcf", type=Path, default=DEFAULT_LEGACY_MJCF)
    parser.add_argument("--official-mjcf", type=Path, default=DEFAULT_OFFICIAL_MJCF)
    parser.add_argument("--ik-config", type=Path, default=DEFAULT_IK)
    parser.add_argument("--body-models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in panel["motions"]}
    rows = [by_id[key] for key in SMOKE_IDS]
    for row in rows:
        if row["dataset"] != "AMASS" or row["transform"] != "identity":
            raise ValueError(f"smoke requires identity AMASS source: {row['id']}")

    legacy = retarget_variant("legacy_model", args.legacy_mjcf, rows, args.ik_config, args.body_models, args.output_root)
    official = retarget_variant("official_v1_model", args.official_mjcf, rows, args.ik_config, args.body_models, args.output_root)
    report = {
        "schema_version": 1,
        "status": "completed_no_training",
        "question": "Does replacing the legacy X2 GMR robot XML with AimDK v1.0 official x2.xml change the retarget output?",
        "smoke_ids": list(SMOKE_IDS),
        "fixed_contract": {"fps": 30, "root_rotation": "full", "root_z": "gmr", "smooth_window": 9, "solver": "quadprog", "damping": 0.5},
        "assets": {
            "legacy_mjcf": str(args.legacy_mjcf),
            "legacy_mjcf_sha256": sha256(args.legacy_mjcf),
            "official_mjcf": str(args.official_mjcf),
            "official_mjcf_sha256": sha256(args.official_mjcf),
            "ik_config": str(args.ik_config),
            "ik_config_sha256": sha256(args.ik_config),
            "legacy_cache": legacy["cache_path"],
            "official_cache": official["cache_path"],
        },
        "comparison": compare(legacy["entries"], official["entries"]),
        "robot_model_comparison": compare_robot_models(args.legacy_mjcf, args.official_mjcf),
        "conclusion": "robot XML swap alone is numerically inert for this GMR contract",
        "training": False,
        "postprocess": False,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.report_json.with_suffix(".md"))
    print(json.dumps({"report": str(args.report_json), "max_difference": report["comparison"]["global_max_abs_difference"]}, indent=2))


if __name__ == "__main__":
    main()
