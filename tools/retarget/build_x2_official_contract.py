#!/usr/bin/env python3
"""Build and verify the canonical AimDK v1.0 X2 model contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml


LOWER_15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


def joint_mirror_sign(name: str) -> int:
    return -1 if any(axis in name for axis in ("_roll_", "_yaw_")) else 1


def mirror_joint_values(values: np.ndarray, names: list[str]) -> np.ndarray:
    values = np.asarray(values)
    index = {name: i for i, name in enumerate(names)}
    return np.asarray(
        [joint_mirror_sign(name) * values[index[mirror_name(name)]] for name in names],
        dtype=values.dtype,
    )


def actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]


def body_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]


def contact_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id and int(model.geom_contype[geom_id]) != 0
    ]


def contact_mirror_permutation(model: mujoco.MjModel) -> list[int]:
    left = contact_geom_ids(model, "left_ankle_roll_link")
    right = contact_geom_ids(model, "right_ankle_roll_link")
    if len(left) != len(right) or not left:
        raise RuntimeError(f"invalid bilateral foot contacts: {len(left)} != {len(right)}")
    right_positions = model.geom_pos[right]
    permutation = []
    for geom_id in left:
        reflected = model.geom_pos[geom_id] * np.asarray([1.0, -1.0, 1.0])
        distances = np.linalg.norm(right_positions - reflected, axis=1)
        match = int(np.argmin(distances))
        if distances[match] > 1.0e-9:
            raise RuntimeError(f"foot contact has no exact mirror: geom {geom_id}")
        permutation.append(match)
    if sorted(permutation) != list(range(len(right))):
        raise RuntimeError("foot contact mirror is not a permutation")
    return permutation


def verify_mirror_fk(model: mujoco.MjModel, *, seeds: int = 8) -> dict[str, float]:
    names = actuated_joint_names(model)
    name_to_joint = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
    }
    bilateral_bodies = [name for name in body_names(model) if name.startswith("left_")]
    left_contacts = contact_geom_ids(model, "left_ankle_roll_link")
    right_contacts = contact_geom_ids(model, "right_ankle_roll_link")
    contact_perm = contact_mirror_permutation(model)
    reflection = np.diag([1.0, -1.0, 1.0])
    rng = np.random.default_rng(20260809)
    max_position_error = 0.0
    max_rotation_error = 0.0
    max_contact_error = 0.0
    max_roundtrip_error = 0.0
    for _ in range(seeds):
        values = []
        for name in names:
            joint_id = name_to_joint[name]
            low, high = model.jnt_range[joint_id]
            center = 0.5 * (low + high)
            half = min(0.35, 0.2 * (high - low))
            values.append(rng.uniform(center - half, center + half))
        values = np.asarray(values)
        mirrored = mirror_joint_values(values, names)
        roundtrip = mirror_joint_values(mirrored, names)
        max_roundtrip_error = max(max_roundtrip_error, float(np.max(np.abs(roundtrip - values))))

        data = mujoco.MjData(model)
        data.qpos[0:3] = 0.0
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, value in zip(names, values, strict=True):
            data.qpos[model.jnt_qposadr[name_to_joint[name]]] = value
        mujoco.mj_forward(model, data)
        left_positions = {}
        left_rotations = {}
        for left_name in bilateral_bodies:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
            left_positions[left_name] = data.xpos[body_id].copy()
            left_rotations[left_name] = data.xmat[body_id].reshape(3, 3).copy()
        left_contact_positions = data.geom_xpos[left_contacts].copy()

        mirrored_data = mujoco.MjData(model)
        mirrored_data.qpos[0:3] = 0.0
        mirrored_data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, value in zip(names, mirrored, strict=True):
            mirrored_data.qpos[model.jnt_qposadr[name_to_joint[name]]] = value
        mujoco.mj_forward(model, mirrored_data)
        for left_name in bilateral_bodies:
            right_name = mirror_name(left_name)
            right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)
            position_error = reflection @ left_positions[left_name] - mirrored_data.xpos[right_id]
            rotation_error = (
                reflection @ left_rotations[left_name] @ reflection
                - mirrored_data.xmat[right_id].reshape(3, 3)
            )
            max_position_error = max(max_position_error, float(np.max(np.abs(position_error))))
            max_rotation_error = max(max_rotation_error, float(np.max(np.abs(rotation_error))))
        reflected_contacts = left_contact_positions * np.asarray([1.0, -1.0, 1.0])
        mirrored_right = mirrored_data.geom_xpos[right_contacts][contact_perm]
        max_contact_error = max(
            max_contact_error, float(np.max(np.abs(reflected_contacts - mirrored_right)))
        )
    report = {
        "seeds": seeds,
        "joint_roundtrip_max_abs_rad": max_roundtrip_error,
        "body_position_max_abs_m": max_position_error,
        "body_rotation_max_abs": max_rotation_error,
        "foot_contact_position_max_abs_m": max_contact_error,
        "gates": {
            "joint_roundtrip_max_abs_rad": 1.0e-12,
            "body_position_max_abs_m": 3.0e-3,
            "body_rotation_max_abs": 7.0e-3,
            "foot_contact_position_max_abs_m": 1.0e-8,
        },
        "interpretation": (
            "joint/contact semantics are exact; non-zero body residual is the measured "
            "left/right CAD asymmetry in the official MJCF"
        ),
    }
    for key, threshold in report["gates"].items():
        if report[key] > threshold:
            raise AssertionError(f"official X2 mirror FK verification failed: {report}")
    return report


def flatten(nested: list[list[object]]) -> list[object]:
    return [item for group in nested for item in group]


def build_contract(scene_xml: Path, model_yaml: Path, control_yaml: Path) -> tuple[dict, dict]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model_config = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
    control_config = yaml.safe_load(control_yaml.read_text(encoding="utf-8"))
    joints_31 = actuated_joint_names(model)
    model_joint_names = flatten(model_config["actual"]["active_joint_name"])
    model_nominal = flatten(model_config["logical"]["nominal_configuration"])
    model_default = dict(zip(model_joint_names, model_nominal, strict=True))
    rl = control_config["rl_config"]
    rl_names = list(rl["seq"])
    if rl_names != list(rl["action_seq"]):
        raise RuntimeError("official observation and action joint sequences differ")
    rl_default = dict(zip(rl_names, rl["default_dof_pos"], strict=True))
    rl_kp = dict(zip(rl_names, rl["kps"], strict=True))
    rl_kd = dict(zip(rl_names, rl["kds"], strict=True))
    if set(joints_31) != set(model_joint_names):
        raise RuntimeError("MJCF and official model YAML joint sets differ")
    if set(joints_31) - set(rl_names) != {"head_yaw_joint", "head_pitch_joint"}:
        raise RuntimeError("official 29-DOF controller boundary is not head-only")

    joints = []
    for name in joints_31:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        body_id = int(model.jnt_bodyid[joint_id])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        parent = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id])
        )
        joints.append({
            "name": name,
            "mjcf_index_31": joints_31.index(name),
            "official_rl_index_29": rl_names.index(name) if name in rl_names else None,
            "speed_backend_index_15": list(LOWER_15).index(name) if name in LOWER_15 else None,
            "body": body,
            "parent_body": parent,
            "axis_xyz": model.jnt_axis[joint_id].tolist(),
            "range_rad": model.jnt_range[joint_id].tolist(),
            "model_nominal_rad": float(model_default[name]),
            "official_rl_default_rad": float(rl_default[name]) if name in rl_default else None,
            "official_rl_kp": float(rl_kp[name]) if name in rl_kp else None,
            "official_rl_kd": float(rl_kd[name]) if name in rl_kd else None,
        })

    bodies = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        parent_id = int(model.body_parentid[body_id])
        bodies.append({
            "name": name,
            "parent": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
            "local_position_m": model.body_pos[body_id].tolist(),
            "mirror_body": mirror_name(name),
        })

    contact_perm = contact_mirror_permutation(model)
    foot_contacts = {}
    for side in ("left", "right"):
        body = f"{side}_ankle_roll_link"
        ids = contact_geom_ids(model, body)
        foot_contacts[side] = [
            {
                "local_index": index,
                "mjcf_geom_id": geom_id,
                "body": body,
                "type": "sphere",
                "local_position_m": model.geom_pos[geom_id].tolist(),
                "radius_m": float(model.geom_size[geom_id, 0]),
                "friction": model.geom_friction[geom_id].tolist(),
            }
            for index, geom_id in enumerate(ids)
        ]

    hashes = {
        "scene_xml": {"path": str(scene_xml), "sha256": sha256(scene_xml)},
        "x2_xml": {
            "path": str(scene_xml.parent / "x2.xml"),
            "sha256": sha256(scene_xml.parent / "x2.xml"),
        },
        "model_yaml": {"path": str(model_yaml), "sha256": sha256(model_yaml)},
        "control_yaml": {"path": str(control_yaml), "sha256": sha256(control_yaml)},
    }
    fk_report = verify_mirror_fk(model)
    contract = {
        "contract_version": "x2_official_v1_phase0_20260809",
        "provenance": hashes,
        "model_counts": {
            "nq": model.nq, "nv": model.nv, "actuators": model.nu,
            "bodies_excluding_world": model.nbody - 1, "hinge_joints": len(joints_31),
        },
        "control_boundaries": {
            "official_mjcf_actuated_31": joints_31,
            "official_rl_sample_29": rl_names,
            "head_locked_2": ["head_yaw_joint", "head_pitch_joint"],
            "speed_backend_lower_15": list(LOWER_15),
            "wbt_target_29": rl_names,
        },
        "coordinate_system": {
            "world": "right-handed, +Z up",
            "root_body": "pelvis",
            "root_joint": "floating_base_joint",
            "tracking_frames": {
                "pelvis": "pelvis", "torso": "torso_link",
                "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
                "left_hand": "left_wrist_roll_link", "right_hand": "right_wrist_roll_link",
            },
            "warning": "official MJCF defines no dedicated palm/sole tracking site; these links are explicit proxies",
        },
        "control": {
            "frequency_hz": 50.0,
            "dt_s": float(rl["dt"]),
            "official_rl_action_scale": float(rl["action_scale"]),
            "official_rl_obs_dim": int(rl["num_obs"]),
            "official_rl_frame_stack": int(rl["infer"]["frame_stack"]),
            "official_rl_num_actions": int(rl["num_actions"]),
            "note": "WBT/Any2Any action scale must be pre-registered separately; do not inherit 0.25 silently",
        },
        "joints": joints,
        "bodies": bodies,
        "foot_contact_geometry": foot_contacts,
    }
    mirror = {
        "contract_version": contract["contract_version"],
        "polar_vector_sign_xyz": [1, -1, 1],
        "axial_vector_sign_xyz": [-1, 1, -1],
        "joint_names_31": joints_31,
        "joint_permutation_31": [joints_31.index(mirror_name(name)) for name in joints_31],
        "joint_sign_31": [joint_mirror_sign(name) for name in joints_31],
        "official_rl_names_29": rl_names,
        "official_rl_permutation_29": [rl_names.index(mirror_name(name)) for name in rl_names],
        "official_rl_sign_29": [joint_mirror_sign(name) for name in rl_names],
        "speed_backend_names_15": list(LOWER_15),
        "speed_backend_permutation_15": [list(LOWER_15).index(mirror_name(name)) for name in LOWER_15],
        "speed_backend_sign_15": [joint_mirror_sign(name) for name in LOWER_15],
        "body_map": {name: mirror_name(name) for name in body_names(model)},
        "left_to_right_foot_contact_permutation": contact_perm,
        "fk_verification": fk_report,
    }
    return contract, mirror


def write_markdown(path: Path, contract: dict, mirror: dict) -> None:
    lines = [
        "# X2 AimDK v1.0 官方模型 Canonical Contract",
        "",
        "日期：2026-08-09",
        "",
        "## 裁决",
        "",
        "- 官方 MJCF 是 31 个执行关节；官方示例 RL policy 是 29 维并仅排除两个头关节；当前速度后端是腿腰 15 维。三者不得混写。",
        "- WBT 第一阶段采用 29DoF body contract，头 yaw/pitch 锁定 model nominal；不涉及灵巧手。",
        "- 官方足底接触由每脚 12 个半径 5 mm 的球点构成，foot tracking proxy 为 ankle-roll link；不得继续沿用旧推测 offset。",
        "- 官方 MJCF 没有专用 palm/sole tracking site，左右 wrist-roll 与 ankle-roll link 只能作为显式 proxy，后续若获得官方 tracking frame 必须升版 contract。",
        "",
        "## 控制边界",
        "",
        f"- MJCF：`{len(contract['control_boundaries']['official_mjcf_actuated_31'])}` DoF。",
        f"- 官方 RL 示例：`{len(contract['control_boundaries']['official_rl_sample_29'])}` DoF，50 Hz，action scale 0.25。",
        f"- 速度保底后端：`{len(contract['control_boundaries']['speed_backend_lower_15'])}` DoF。",
        "- Any2Any/WBT：目标 29DoF，但 action scale 与 observation contract 必须由忠实基线单独预注册，不能静默照搬官方舞蹈 ONNX。",
        "",
        "## 坐标与接触",
        "",
        "- 世界系：右手系，+Z 向上；floating root 位于 pelvis。",
        "- pelvis/torso：`pelvis` / `torso_link`。",
        "- feet：`left_ankle_roll_link` / `right_ankle_roll_link`。",
        "- hands：`left_wrist_roll_link` / `right_wrist_roll_link`。",
        f"- 足底镜像点 permutation：`{mirror['left_to_right_foot_contact_permutation']}`。",
        "",
        "## 镜像/FK 门禁",
        "",
        f"- 关节 round-trip 最大误差：`{mirror['fk_verification']['joint_roundtrip_max_abs_rad']:.3e}` rad。",
        f"- body position 最大误差：`{mirror['fk_verification']['body_position_max_abs_m']:.3e}` m。",
        f"- body rotation 最大误差：`{mirror['fk_verification']['body_rotation_max_abs']:.3e}`。",
        f"- 足底接触点最大误差：`{mirror['fk_verification']['foot_contact_position_max_abs_m']:.3e}` m。",
        "- body 残差不是映射错误：官方左右 CAD 存在毫米级非对称；预注册容差为位置 3 mm、旋转矩阵元素 0.007。",
        "",
        "## 版本与限制",
        "",
        f"- `x2.xml` SHA-256：`{contract['provenance']['x2_xml']['sha256']}`。",
        f"- `default.yaml` SHA-256：`{contract['provenance']['model_yaml']['sha256']}`。",
        f"- `motion_control.yaml` SHA-256：`{contract['provenance']['control_yaml']['sha256']}`。",
        "- 官方 README 明确该 RL 示例只在仿真验证，真机仍需重新参数适配；本 contract 不授权真机发送命令。",
        "- model nominal 与官方 RL 示例 deeper-crouch default 同时保留，任何实验必须声明使用哪一套，禁止混用。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--model-yaml", type=Path, required=True)
    parser.add_argument("--control-yaml", type=Path, required=True)
    parser.add_argument("--map-json", type=Path, required=True)
    parser.add_argument("--mirror-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    contract, mirror = build_contract(args.scene_xml, args.model_yaml, args.control_yaml)
    for path, payload in ((args.map_json, contract), (args.mirror_json, mirror)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown, contract, mirror)
    print(json.dumps({"map": str(args.map_json), "mirror": str(args.mirror_json), "fk": mirror["fk_verification"]}, indent=2))


if __name__ == "__main__":
    main()
