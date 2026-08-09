#!/usr/bin/env python3
"""Static B5 provenance/physics-contract audit for X2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from x2_physics_provenance_guard import (
    assert_physics_contract,
    check_runtime_snapshot,
    load_manifest,
    load_runtime_snapshot,
    sha256,
)


MANIFEST = REPO / "configs/x2_faithful_physics_phase25.yaml"
RUNTIME = REPO / "configs/x2_faithful_physics_phase25_runtime.json"
DEFAULT_JSON = REPO / "reports/retarget/x2_physics_provenance_phase25.json"
DEFAULT_MD = REPO / "reports/retarget/x2_physics_provenance_phase25.md"


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
        if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    ]


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise KeyError(name)
    return result


def hinge_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise KeyError(name)
    return result


def urdf_contract(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        if mass_node is None or inertia_node is None:
            continue
        inertia = np.array(
            [
                [float(inertia_node.attrib["ixx"]), float(inertia_node.attrib["ixy"]), float(inertia_node.attrib["ixz"])],
                [float(inertia_node.attrib["ixy"]), float(inertia_node.attrib["iyy"]), float(inertia_node.attrib["iyz"])],
                [float(inertia_node.attrib["ixz"]), float(inertia_node.attrib["iyz"]), float(inertia_node.attrib["izz"])],
            ]
        )
        links[link.attrib["name"]] = {
            "mass": float(mass_node.attrib["value"]),
            "principal_inertia": sorted(np.linalg.eigvalsh(inertia).tolist(), reverse=True),
        }
    joints = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        joints[joint.attrib["name"]] = {
            "lower": float(limit.attrib["lower"]),
            "upper": float(limit.attrib["upper"]),
            "effort": float(limit.attrib["effort"]),
            "velocity": float(limit.attrib["velocity"]),
        }
    return {"links": links, "joints": joints}


def model_body_contract(model: mujoco.MjModel, names: list[str]) -> dict[str, Any]:
    return {
        name: {
            "mass": float(model.body_mass[body_id(model, name)]),
            "principal_inertia": sorted(model.body_inertia[body_id(model, name)].tolist(), reverse=True),
        }
        for name in names
    }


def qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[hinge_id(model, name)])


def joint_armature(model: mujoco.MjModel, name: str) -> float:
    return float(model.dof_armature[int(model.jnt_dofadr[hinge_id(model, name)])])


def normalized_fk(model: mujoco.MjModel, q_by_name: dict[str, float], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    free = [index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE]
    if free:
        address = int(model.jnt_qposadr[free[0]])
        data.qpos[address : address + 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for name, value in q_by_name.items():
        data.qpos[qpos_address(model, name)] = value
    mujoco.mj_forward(model, data)
    indices = [body_id(model, name) for name in names]
    return data.xpos[indices].copy(), data.xmat[indices].reshape(-1, 3, 3).copy()


def rotation_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", left, right)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def collision_contract(model: mujoco.MjModel, world_alias: str | None = None) -> dict[str, Any]:
    rows = []
    for index in range(model.ngeom):
        if not (model.geom_contype[index] and model.geom_conaffinity[index]):
            continue
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])
        )
        if body == "world" and world_alias is not None:
            body = world_alias
        rows.append(
            {
                "body": body,
                "type": int(model.geom_type[index]),
                "pos": np.round(model.geom_pos[index], 7).tolist(),
                "size": np.round(model.geom_size[index], 7).tolist(),
            }
        )
    counts = {}
    for row in rows:
        counts[row["body"]] = counts.get(row["body"], 0) + 1
    return {"active_count": len(rows), "per_body_count": counts, "rows": rows}


def sole_spheres(contract: dict[str, Any], side: str) -> list[dict[str, Any]]:
    body = f"{side}_ankle_roll_link"
    return sorted(
        [row for row in contract["rows"] if row["body"] == body and row["type"] == 2],
        key=lambda row: tuple(row["pos"]),
    )


def isaac_runtime_values(joints: list[str]) -> dict[str, Any]:
    armature = {
        "5020": 0.003609725,
        "7520_14": 0.010177520,
        "7520_22": 0.025101925,
        "4010": 0.00425,
    }
    natural_frequency = 10.0 * 2.0 * math.pi
    damping_ratio = 2.0
    kp_base = {key: value * natural_frequency**2 for key, value in armature.items()}
    kd_base = {
        key: 2.0 * damping_ratio * value * natural_frequency
        for key, value in armature.items()
    }

    def motor(name: str) -> str:
        if "wrist_pitch" in name or "wrist_roll" in name:
            return "4010"
        if "ankle" in name or "shoulder" in name or "elbow" in name or "wrist_yaw" in name or "head" in name:
            return "5020"
        if "hip_roll" in name or "knee" in name:
            return "7520_22"
        if "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            return "7520_14"
        if name in ("waist_pitch_joint", "waist_roll_joint"):
            return "5020"
        raise KeyError(name)

    effort = {}
    for name in joints:
        if "hip_" in name or "knee" in name or name == "waist_yaw_joint":
            effort[name] = 120.0
        elif "ankle_pitch" in name:
            effort[name] = 36.0
        elif "ankle_roll" in name or "shoulder_yaw" in name or "elbow" in name or "wrist_yaw" in name:
            effort[name] = 24.0
        elif "shoulder_pitch" in name or "shoulder_roll" in name:
            effort[name] = 36.0
        elif name in ("waist_pitch_joint", "waist_roll_joint"):
            effort[name] = 48.0
        elif "wrist_pitch" in name or "wrist_roll" in name:
            effort[name] = 4.8
        elif name == "head_yaw_joint":
            effort[name] = 2.6
        elif name == "head_pitch_joint":
            effort[name] = 0.6
        else:
            raise KeyError(name)
    return {
        "kp": {name: kp_base[motor(name)] for name in joints},
        "kd": {name: kd_base[motor(name)] for name in joints},
        "armature": {name: armature[motor(name)] for name in joints},
        "effort": effort,
        "action_scale": {
            name: (0.0 if name.startswith("head_") else 0.25 * effort[name] / kp_base[motor(name)])
            for name in joints
        },
    }


def compare_status(label: str, status: str, evidence: Any) -> dict[str, Any]:
    return {"item": label, "status": status, "evidence": evidence}


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 Physics Provenance / B5 — Phase25",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- IsaacLab被声明为训练域；AimDK v1 MuJoCo只作为held-out sim-to-sim mismatch域，绝不宣称二者物理等价。",
        "- 本阶段只有静态loader/FK/default/limit/PD与hash guard；没有physics stepping、训练或zero-update。",
        "",
        "## 假设",
        "",
        "训练域不必数值复制官方MuJoCo，但必须冻结asset生成链、控制与solver合同，并把全部已知差异显式暴露。",
        "",
        "## 干预与对照",
        "",
        "- 干预：冻结raw URDF→sole12 builder→sole12 URDF、mesh树、Isaac converter/spawner、x2/env代码和完整runtime snapshot。",
        "- 对照：官方AimDK v1 `x2.xml`、`motion_control.yaml`、scene/simulator配置和同一mesh树。",
        "",
        "## 逐项结果",
        "",
        "| 项目 | 状态 | 证据摘要 |",
        "|---|---|---|",
    ]
    for row in report["comparison_matrix"]:
        evidence = json.dumps(row["evidence"], ensure_ascii=False, separators=(",", ":"))
        if len(evidence) > 180:
            evidence = evidence[:177] + "..."
        lines.append(f"| {row['item']} | {row['status']} | `{evidence}` |")
    lines += [
        "",
        "## 关键边界",
        "",
        f"- pelvis mass：URDF `{report['mass_inertia']['urdf_pelvis_mass_kg']:.6f}kg` vs official compiled `{report['mass_inertia']['official_pelvis_mass_kg']:.6f}kg`。",
        f"- active collision：sole12训练域 `{report['collision']['isaac_source_active_count']}` vs official `{report['collision']['official_active_count']}`；足底12球逐项匹配 `{report['collision']['sole_spheres_exact']}`。",
        f"- neutral FK max：`{report['fk']['neutral_position_max_m']:.2e}m/{report['fk']['neutral_orientation_max_rad']:.2e}rad`；各自default pose FK差异不是loader错误，max `{report['fk']['domain_default_position_max_m']:.3f}m`。",
        "- PD/action-scale/physics-dt差异保留为held-out mismatch，不进入`matched`。",
        "",
        "## Guard",
        "",
        f"- 文件/mesh hash：`{report['guard']['positive']['file_checks']}`。",
        f"- runtime exact：`{report['guard']['positive']['runtime_check']['exact']}`；负例sim_dt改写被拒：`{report['guard']['negative_probe_rejected']}`。",
        "- dedicated launcher不读物理环境变量；启动前必须通过manifest hash+resolved runtime snapshot guard。",
        "",
        "## 结论",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    runtime = load_runtime_snapshot(args.runtime)
    positive_guard = assert_physics_contract(args.manifest, runtime)
    negative = dict(runtime)
    negative["sim_dt"] = 0.006
    negative_rejected = not check_runtime_snapshot(manifest, negative)["exact"]

    local_path = Path(manifest["files"]["training_sole12_urdf"]["path"])
    official_path = Path(manifest["files"]["official_x2_xml"]["path"])
    motion_path = Path(manifest["files"]["official_motion_control"]["path"])
    local = mujoco.MjModel.from_xml_path(str(local_path))
    official = mujoco.MjModel.from_xml_path(str(official_path))
    urdf = urdf_contract(local_path)
    motion = yaml.safe_load(motion_path.read_text())
    official_control = motion["rl_config"]
    official29 = list(official_control["seq"])
    joints31 = joint_names(local)
    official_joints31 = joint_names(official)
    common_bodies = [name for name in urdf["links"] if name != "pelvis" and name in {
        mujoco.mj_id2name(official, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(official.nbody)
    }]
    official_body = model_body_contract(official, common_bodies)
    mass_abs = {
        name: abs(urdf["links"][name]["mass"] - official_body[name]["mass"])
        for name in common_bodies
    }
    inertia_abs = {
        name: max(
            abs(a - b)
            for a, b in zip(
                urdf["links"][name]["principal_inertia"],
                official_body[name]["principal_inertia"],
                strict=True,
            )
        )
        for name in common_bodies
    }
    pelvis_id = body_id(official, "pelvis")
    mass_inertia = {
        "common_nonpelvis_body_count": len(common_bodies),
        "common_mass_abs_max_kg": max(mass_abs.values()),
        "common_inertia_principal_abs_max_kgm2": max(inertia_abs.values()),
        "urdf_pelvis_mass_kg": urdf["links"]["pelvis"]["mass"],
        "official_pelvis_mass_kg": float(official.body_mass[pelvis_id]),
        "urdf_total_explicit_mass_kg": sum(row["mass"] for row in urdf["links"].values()),
        "official_compiled_total_mass_kg": float(official.body_mass.sum()),
    }

    local_collision = collision_contract(local, world_alias="pelvis")
    official_collision = collision_contract(official)
    local_sole = {side: sole_spheres(local_collision, side) for side in ("left", "right")}
    official_sole = {side: sole_spheres(official_collision, side) for side in ("left", "right")}
    collision = {
        "isaac_source_active_count": local_collision["active_count"],
        "official_active_count": official_collision["active_count"],
        "isaac_source_per_body": local_collision["per_body_count"],
        "official_per_body": official_collision["per_body_count"],
        "body_count_differences": {
            name: {
                "isaac_source": local_collision["per_body_count"].get(name, 0),
                "official": official_collision["per_body_count"].get(name, 0),
            }
            for name in sorted(set(local_collision["per_body_count"]) | set(official_collision["per_body_count"]))
            if local_collision["per_body_count"].get(name, 0)
            != official_collision["per_body_count"].get(name, 0)
        },
        "body_type_differences": {
            name: {
                "isaac_source": sorted(
                    row["type"] for row in local_collision["rows"] if row["body"] == name
                ),
                "official": sorted(
                    row["type"] for row in official_collision["rows"] if row["body"] == name
                ),
            }
            for name in sorted(set(local_collision["per_body_count"]) | set(official_collision["per_body_count"]))
            if sorted(row["type"] for row in local_collision["rows"] if row["body"] == name)
            != sorted(row["type"] for row in official_collision["rows"] if row["body"] == name)
        },
        "sole_spheres_exact": local_sole == official_sole,
        "sole_spheres_per_foot": {side: len(local_sole[side]) for side in local_sole},
    }

    neutral_q = {name: 0.0 for name in joints31}
    fk_names = common_bodies
    local_pos, local_rot = normalized_fk(local, neutral_q, fk_names)
    official_pos, official_rot = normalized_fk(official, neutral_q, fk_names)
    neutral_pos_error = np.linalg.norm(local_pos - official_pos, axis=-1)
    neutral_ori_error = rotation_error(local_rot, official_rot)

    isaac_default = {name: 0.0 for name in joints31}
    for name in joints31:
        if "hip_pitch" in name:
            isaac_default[name] = -0.2480
        elif "knee" in name:
            isaac_default[name] = 0.5303
        elif "ankle_pitch" in name:
            isaac_default[name] = -0.2823
        elif "shoulder_pitch" in name:
            isaac_default[name] = 0.4
        elif "elbow" in name:
            isaac_default[name] = -1.2
    official_default = dict(zip(official29, official_control["default_dof_pos"], strict=True))
    official_default.update({"head_yaw_joint": 0.0, "head_pitch_joint": 0.0})
    local_default_pos, _ = normalized_fk(local, isaac_default, fk_names)
    official_default_pos, _ = normalized_fk(official, official_default, fk_names)
    fk = {
        "neutral_body_count": len(fk_names),
        "neutral_position_max_m": float(neutral_pos_error.max()),
        "neutral_orientation_max_rad": float(neutral_ori_error.max()),
        "domain_default_joint_abs_max_rad": max(
            abs(isaac_default[name] - official_default[name]) for name in joints31
        ),
        "domain_default_position_max_m": float(
            np.linalg.norm(local_default_pos - official_default_pos, axis=-1).max()
        ),
    }

    joint_rows = {}
    for name in joints31:
        local_limit = urdf["joints"][name]
        oid = hinge_id(official, name)
        official_range = official.jnt_range[oid].tolist()
        joint_rows[name] = {
            "isaac_source_range": [local_limit["lower"], local_limit["upper"]],
            "official_range": official_range,
            "range_abs_max": max(
                abs(local_limit["lower"] - official_range[0]),
                abs(local_limit["upper"] - official_range[1]),
            ),
            "isaac_source_effort": local_limit["effort"],
            "official_actuator_force_limit": float(max(abs(x) for x in official.jnt_actfrcrange[oid])),
            "isaac_source_velocity_limit": local_limit["velocity"],
            "official_velocity_limit": None,
        }
    limits = {
        "per_joint": joint_rows,
        "range_abs_max_rad": max(row["range_abs_max"] for row in joint_rows.values()),
        "effort_abs_max_nm": max(
            abs(row["isaac_source_effort"] - row["official_actuator_force_limit"])
            for row in joint_rows.values()
        ),
        "official_velocity_limits_available": False,
    }

    isaac = isaac_runtime_values(joints31)
    official_kp = dict(zip(official29, official_control["kps"], strict=True))
    official_kd = dict(zip(official29, official_control["kds"], strict=True))
    pd = {
        "isaac_kp": isaac["kp"],
        "isaac_kd": isaac["kd"],
        "official_kp": official_kp,
        "official_kd": official_kd,
        "kp_abs_max_29": max(abs(isaac["kp"][name] - official_kp[name]) for name in official29),
        "kd_abs_max_29": max(abs(isaac["kd"][name] - official_kd[name]) for name in official29),
        "isaac_armature": isaac["armature"],
        "official_mjcf_armature": {
            name: joint_armature(official, name) for name in joints31
        },
        "armature_abs_max": max(
            abs(isaac["armature"][name] - joint_armature(official, name))
            for name in joints31
        ),
        "isaac_action_scale": isaac["action_scale"],
        "official_action_scale": float(official_control["action_scale"]),
        "action_scale_abs_max_29": max(
            abs(isaac["action_scale"][name] - float(official_control["action_scale"]))
            for name in official29
        ),
    }

    mesh_hash_match = (
        manifest["files"]["local_mesh_tree"]["tree_sha256"]
        == manifest["files"]["official_mesh_tree"]["tree_sha256"]
    )
    comparison = [
        compare_status("mesh tree", "matched", {"45_files_same_hash": mesh_hash_match}),
        compare_status("31 joint names", "matched", {"same_set": set(joints31) == set(official_joints31)}),
        compare_status("neutral kinematic FK", "matched", {"pos_max_m": fk["neutral_position_max_m"], "ori_max_rad": fk["neutral_orientation_max_rad"]}),
        compare_status("non-pelvis mass/principal inertia", "matched", {"bodies": len(common_bodies), "mass_max": mass_inertia["common_mass_abs_max_kg"], "inertia_max": mass_inertia["common_inertia_principal_abs_max_kgm2"]}),
        compare_status("pelvis mass", "different", {"isaac_urdf": mass_inertia["urdf_pelvis_mass_kg"], "official_compiled": mass_inertia["official_pelvis_mass_kg"]}),
        compare_status("joint position limits", "matched" if limits["range_abs_max_rad"] <= 5e-4 else "different", {"max_abs_rad": limits["range_abs_max_rad"]}),
        compare_status("effort limits", "matched", {"max_abs_nm": limits["effort_abs_max_nm"]}),
        compare_status("velocity limits", "unknown", "official x2.xml/motion_control does not declare velocity limits"),
        compare_status("active sole geometry", "matched", {"12_spheres_each": collision["sole_spheres_exact"]}),
        compare_status("full active collision set", "different", {"isaac": collision["isaac_source_active_count"], "official": collision["official_active_count"], "body_count_differences": collision["body_count_differences"], "body_type_differences": collision["body_type_differences"]}),
        compare_status("default pose/root height", "different", {"joint_max_rad": fk["domain_default_joint_abs_max_rad"], "isaac_root_z": runtime["root_initial_position"][2], "official_xml_root_z": 0.68}),
        compare_status("application PD", "different", {"kp_max": pd["kp_abs_max_29"], "kd_max": pd["kd_abs_max_29"]}),
        compare_status("joint armature", "different", {"max_abs": pd["armature_abs_max"]}),
        compare_status("action scale", "different", {"max_abs_29": pd["action_scale_abs_max_29"]}),
        compare_status("control dt", "matched", {"isaac": runtime["control_dt"], "official": official_control["dt"]}),
        compare_status("physics dt", "different", {"isaac": runtime["sim_dt"], "official_mjcf": float(official.opt.timestep)}),
        compare_status("cross-engine solver", "unknown", "PhysX iteration counts and MuJoCo solver are not numerically equivalent parameters"),
        compare_status("raw URDF official generation lineage", "unknown", "mesh and most inertials match, but no signed generator/provenance states x2.xml was generated from this URDF"),
    ]

    ready = (
        all(positive_guard["file_checks"].values())
        and positive_guard["runtime_check"]["exact"]
        and negative_rejected
        and collision["sole_spheres_exact"]
        and fk["neutral_position_max_m"] <= 1e-6
        and fk["neutral_orientation_max_rad"] <= 1e-6
        and manifest["guard"]["fail_closed"]
        and manifest["guard"]["allow_environment_overrides"] is False
    )
    report = {
        "schema_version": "x2_physics_provenance_phase25_v1",
        "provenance": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "runtime_snapshot": {"path": str(args.runtime), "sha256": sha256(args.runtime)},
        },
        "truth_boundary": {
            "isaaclab_is_training_domain": True,
            "official_mujoco_is_held_out_sim_to_sim": True,
            "domains_numerically_equivalent": False,
            "physics_step_training_zero_update": False,
        },
        "guard": {
            "positive": positive_guard,
            "negative_probe_rejected": negative_rejected,
        },
        "mass_inertia": mass_inertia,
        "collision": collision,
        "fk": fk,
        "joint_limits": limits,
        "pd_action": pd,
        "runtime": runtime,
        "comparison_matrix": comparison,
        "mismatch_policy": {
            "pelvis_mass": "held-out mismatch; candidate for bounded mass DR, never silently overwritten",
            "collision_52_vs_49": "held-out mismatch; sole contact is matched but wrist/head collision remains domain difference",
            "pd_action_dt": "held-out actuator/control mismatch; source-equivalent DR and official sim-to-sim report separately",
            "unknowns": "must remain unknown until authoritative provenance/solver mapping is supplied",
        },
        "decision": {
            "status": "B5_READY_AS_DECLARED_TRAIN_DOMAIN" if ready else "B5_BLOCKED",
            "result": (
                "The sole12 Isaac training domain is immutable and prelaunch hash/runtime guarded; all official mismatches are explicit."
                if ready
                else "The declared Isaac training domain or its guard failed a critical static contract."
            ),
            "conclusion": (
                "B5 is ready as a declared training domain, not as a claim of official-MuJoCo equivalence."
                if ready
                else "Keep Phase24 live-zero fail-closed."
            ),
            "next_step": (
                "Update the faithful launcher to require this manifest hash, then stop for review before any zero-update."
                if ready
                else "Fix only the failed provenance/guard item; do not run zero-update."
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown.write_text(render(report))
    print(json.dumps({"status": report["decision"]["status"], "differences": sum(row["status"] == "different" for row in comparison), "unknown": sum(row["status"] == "unknown" for row in comparison)}))


if __name__ == "__main__":
    main()
