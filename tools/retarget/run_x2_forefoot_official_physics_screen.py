#!/usr/bin/env python3
"""Matched-control AimDK v1.0 MuJoCo screen for two X2 retarget caches.

This tool deliberately reports two different questions:

* ``prescribed_root_trackability`` overwrites the free root with the reference
  at every physics step.  It isolates joint/torque trackability, but is not a
  balance test and must never be used as a Silver-data gate by itself.
* ``free_root_balance`` leaves the base completely free.  It is an open-loop
  position-PD feasibility screen, not a learned-policy evaluation.

The official AimDK scene, actuator limits, per-joint RL gains, 1 kHz physics
step, and 50 Hz command period are shared by both retarget variants.  No
training or checkpoint loading occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


REPO = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy")
DEFAULT_SCENE = OFFICIAL_ROOT / (
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"
)
DEFAULT_CONTROL = OFFICIAL_ROOT / "x2_rl_deploy_controller/config/motion_control.yaml"
CACHE_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase1_forefoot_smoothing_ab"
)
DEFAULT_CURRENT = CACHE_ROOT / "current_v4_exact30/x2_current_v4_exact30.pkl"
DEFAULT_FOREFOOT = CACHE_ROOT / "smooth9/x2_toe_forefoot_smooth9.pkl"
DEFAULT_OFFLINE_REPORT = REPO / "reports/retarget/x2_forefoot_smoothing_ab.json"
DEFAULT_JSON = REPO / "reports/retarget/x2_forefoot_official_physics_screen.json"
DEFAULT_MD = REPO / "reports/retarget/x2_forefoot_official_physics_screen.md"

DEFAULT_ROLES = ("walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand")
MODES = ("prescribed_root_trackability", "free_root_balance")
FALL_ROOT_Z_M = 0.42
FALL_TILT_RAD = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    if value is None:
        return f"unnamed_{index}"
    return value


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, q))


def mean_or_none(values: Iterable[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return None if array.size == 0 else float(np.mean(array))


def root_tilt(q_wxyz: np.ndarray) -> float:
    rotation = Rotation.from_quat(q_wxyz, scalar_first=True)
    body_up = rotation.apply(np.array([0.0, 0.0, 1.0]))
    return float(np.arccos(np.clip(body_up[2], -1.0, 1.0)))


def quat_distance(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    r1 = Rotation.from_quat(q1_wxyz, scalar_first=True)
    r2 = Rotation.from_quat(q2_wxyz, scalar_first=True)
    return float((r1.inv() * r2).magnitude())


@dataclass(frozen=True)
class OfficialControlContract:
    control_dt: float
    joint_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    qpos_addresses: np.ndarray
    qvel_addresses: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_low: np.ndarray
    torque_high: np.ndarray
    head_gain_source: str


def build_control_contract(
    model: mujoco.MjModel, control_yaml: Path
) -> OfficialControlContract:
    """Map official RL gains to the official model's actuator order."""
    raw = yaml.safe_load(control_yaml.read_text(encoding="utf-8"))
    rl = raw["rl_config"]
    seq = [str(name) for name in rl["seq"]]
    kps = np.asarray(rl["kps"], dtype=np.float64)
    kds = np.asarray(rl["kds"], dtype=np.float64)
    if len(seq) != 29 or kps.shape != (29,) or kds.shape != (29,):
        raise ValueError("official RL gain contract is not the expected 29-DOF body contract")
    gains = {name: (float(kp), float(kd)) for name, kp, kd in zip(seq, kps, kds)}
    default_kp = float(raw["default_kp"])
    default_kd = float(raw["default_kd"])

    actuator_joint_names: list[str] = []
    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    kp_act: list[float] = []
    kd_act: list[float] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_joint_names.append(name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
        gain = gains.get(name, (default_kp, default_kd))
        kp_act.append(gain[0])
        kd_act.append(gain[1])

    if set(seq) != set(actuator_joint_names) - {"head_yaw_joint", "head_pitch_joint"}:
        raise ValueError("official YAML and official MuJoCo body-joint sets disagree")
    joint_names = tuple(
        object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    )
    return OfficialControlContract(
        control_dt=float(rl["dt"]),
        joint_names=joint_names,
        actuator_joint_names=tuple(actuator_joint_names),
        qpos_addresses=np.asarray(qpos_addresses, dtype=np.int64),
        qvel_addresses=np.asarray(qvel_addresses, dtype=np.int64),
        kp=np.asarray(kp_act, dtype=np.float64),
        kd=np.asarray(kd_act, dtype=np.float64),
        torque_low=np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64),
        torque_high=np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64),
        head_gain_source=f"motion_control.yaml default_kp/default_kd={default_kp:g}/{default_kd:g}",
    )


def validate_motion_pair(current: dict[str, Any], forefoot: dict[str, Any]) -> dict[str, Any]:
    if set(current) != set(forefoot):
        raise ValueError("variant caches do not contain identical motion keys")
    rows = []
    for key in sorted(current):
        left, right = current[key], forefoot[key]
        if left.get("panel_role") != right.get("panel_role"):
            raise ValueError(f"panel role mismatch: {key}")
        if int(left["fps"]) != 30 or int(right["fps"]) != 30:
            raise ValueError(f"non-exact30 input: {key}")
        if np.asarray(left["dof"]).shape != np.asarray(right["dof"]).shape:
            raise ValueError(f"frame/DOF mismatch: {key}")
        if list(left["joint_names_mujoco"]) != list(right["joint_names_mujoco"]):
            raise ValueError(f"joint-order mismatch: {key}")
        rows.append({
            "key": key,
            "panel_role": left.get("panel_role"),
            "frames": int(len(left["dof"])),
            "fps": 30,
        })
    return {"matched": True, "motions": rows}


def interpolate_motion(entry: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fps = float(entry["fps"])
    dof = np.asarray(entry["dof"], dtype=np.float64)
    root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    root_xyzw = np.asarray(entry["root_rot"], dtype=np.float64)
    source_times = np.arange(len(dof), dtype=np.float64) / fps
    clipped = np.minimum(np.asarray(times, dtype=np.float64), source_times[-1])
    dof_i = np.column_stack([
        np.interp(clipped, source_times, dof[:, index]) for index in range(dof.shape[1])
    ])
    pos_i = np.column_stack([
        np.interp(clipped, source_times, root_pos[:, index]) for index in range(3)
    ])
    quat_i = Slerp(source_times, Rotation.from_quat(root_xyzw))(clipped).as_quat()
    return pos_i, quat_i[:, [3, 0, 1, 2]], dof_i


def reference_qpos(entry: dict[str, Any], times: np.ndarray) -> np.ndarray:
    pos, quat_wxyz, dof = interpolate_motion(entry, times)
    return np.concatenate([pos, quat_wxyz, dof], axis=1)


def joint_groups(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        "legs": np.asarray([i for i, name in enumerate(names) if any(
            token in name for token in ("hip", "knee", "ankle")
        )], dtype=np.int64),
        "waist": np.asarray([i for i, name in enumerate(names) if name.startswith("waist_")], dtype=np.int64),
        "upper": np.asarray([i for i, name in enumerate(names) if any(
            token in name for token in ("shoulder", "elbow", "wrist")
        )], dtype=np.int64),
        "head": np.asarray([i for i, name in enumerate(names) if name.startswith("head_")], dtype=np.int64),
        "all": np.arange(len(names), dtype=np.int64),
    }


def target_limit_report(model: mujoco.MjModel, entry: dict[str, Any]) -> dict[str, Any]:
    dof = np.asarray(entry["dof"], dtype=np.float64)
    names = list(entry["joint_names_mujoco"])
    violating = 0
    total = 0
    max_excess = 0.0
    per_joint: dict[str, int] = {}
    tolerance = 1.0e-6
    for index, name in enumerate(names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            continue
        low, high = model.jnt_range[joint_id]
        below = np.maximum(low - dof[:, index], 0.0)
        above = np.maximum(dof[:, index] - high, 0.0)
        count = int(np.count_nonzero((below + above) > tolerance))
        if count:
            per_joint[name] = count
        violating += count
        total += len(dof)
        max_excess = max(max_excess, float(np.max(below + above)))
    return {
        "violating_samples": violating,
        "limited_joint_samples": total,
        "fraction": float(violating / total) if total else 0.0,
        "max_excess_rad": max_excess,
        "count_tolerance_rad": tolerance,
        "per_joint": per_joint,
    }


def foot_geom_contract(model: mujoco.MjModel) -> tuple[int, dict[str, set[int]]]:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise ValueError("official scene has no floor geom")
    result: dict[str, set[int]] = {"left": set(), "right": set()}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name == "left_ankle_roll_link":
            result["left"].add(geom_id)
        elif body_name == "right_ankle_roll_link":
            result["right"].add(geom_id)
    if not result["left"] or not result["right"]:
        raise ValueError("official foot collision geoms were not found")
    return floor, result


def simulate_case(
    scene: Path,
    control_yaml: Path,
    entry: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(mode)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    contract = build_control_contract(model, control_yaml)
    entry_names = tuple(str(name) for name in entry["joint_names_mujoco"])
    if entry_names != contract.joint_names:
        raise ValueError("cache joint order differs from official MuJoCo joint order")
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError(f"unexpected official physics timestep: {model.opt.timestep}")
    steps_per_control = int(round(contract.control_dt / model.opt.timestep))
    if not np.isclose(steps_per_control * model.opt.timestep, contract.control_dt):
        raise ValueError("control period is not an integer number of physics steps")

    duration = (len(entry["dof"]) - 1) / float(entry["fps"])
    n_steps = int(round(duration / model.opt.timestep))
    sim_times = np.minimum(np.arange(n_steps + 1) * model.opt.timestep, duration)
    qpos_ref = reference_qpos(entry, sim_times)
    control_times = np.minimum(
        np.floor(sim_times / contract.control_dt + 1.0e-10) * contract.control_dt,
        duration,
    )
    _, _, dof_control = interpolate_motion(entry, control_times)

    # Exact reference free-joint velocities for the prescribed-root diagnostic.
    qvel_ref = None
    if mode == "prescribed_root_trackability":
        qvel_ref = np.zeros((n_steps + 1, model.nv), dtype=np.float64)
        for index in range(n_steps):
            mujoco.mj_differentiatePos(
                model, qvel_ref[index], model.opt.timestep, qpos_ref[index], qpos_ref[index + 1]
            )
        if n_steps:
            qvel_ref[-1] = qvel_ref[-2]

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    floor_geom, foot_geoms = foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    previous_geom_pos = data.geom_xpos.copy()
    groups = joint_groups(contract.joint_names)
    entry_index = {name: index for index, name in enumerate(contract.joint_names)}
    actuator_target_indices = np.asarray(
        [entry_index[name] for name in contract.actuator_joint_names], dtype=np.int64
    )

    errors: list[np.ndarray] = []
    root_xy_errors: list[float] = []
    root_orientation_errors: list[float] = []
    root_z_values: list[float] = []
    tilt_values: list[float] = []
    slip_speeds: dict[str, list[float]] = {"left": [], "right": []}
    normal_forces: dict[str, list[float]] = {"left": [], "right": []}
    contact_steps = {"left": 0, "right": 0}
    saturation_count = np.zeros(model.nu, dtype=np.int64)
    effort_count = 0
    fall_time: float | None = None
    achieved_steps = n_steps

    for step in range(n_steps):
        if mode == "prescribed_root_trackability":
            data.qpos[:7] = qpos_ref[step, :7]
            data.qvel[:6] = qvel_ref[step, :6]  # type: ignore[index]

        target_act = dof_control[step, actuator_target_indices]
        actual_q = data.qpos[contract.qpos_addresses]
        actual_dq = data.qvel[contract.qvel_addresses]
        raw_torque = contract.kp * (target_act - actual_q) - contract.kd * actual_dq
        saturation_count += ((raw_torque < contract.torque_low) | (raw_torque > contract.torque_high))
        effort_count += model.nu
        data.ctrl[:] = np.clip(raw_torque, contract.torque_low, contract.torque_high)
        mujoco.mj_step(model, data)

        if mode == "prescribed_root_trackability":
            data.qpos[:7] = qpos_ref[step + 1, :7]
            data.qvel[:6] = qvel_ref[step + 1, :6]  # type: ignore[index]
            mujoco.mj_forward(model, data)

        contacts_now = {"left": set(), "right": set()}
        force_buffer = np.zeros(6, dtype=np.float64)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            other = geom2 if geom1 == floor_geom else geom1 if geom2 == floor_geom else -1
            side = geom_side.get(other)
            if side is None:
                continue
            contacts_now[side].add(other)
            mujoco.mj_contactForce(model, data, contact_id, force_buffer)
            normal_forces[side].append(max(0.0, float(force_buffer[0])))
        for side in ("left", "right"):
            if contacts_now[side]:
                contact_steps[side] += 1
                for geom_id in contacts_now[side]:
                    delta = data.geom_xpos[geom_id, :2] - previous_geom_pos[geom_id, :2]
                    slip_speeds[side].append(float(np.linalg.norm(delta) / model.opt.timestep))
        previous_geom_pos[:] = data.geom_xpos

        if (step + 1) % steps_per_control == 0 or step + 1 == n_steps:
            current_by_name = {
                name: float(data.qpos[address])
                for name, address in zip(contract.actuator_joint_names, contract.qpos_addresses)
            }
            current_joint_order = np.asarray([current_by_name[name] for name in contract.joint_names])
            errors.append(current_joint_order - qpos_ref[step + 1, 7:])
            root_xy_errors.append(float(np.linalg.norm(data.qpos[:2] - qpos_ref[step + 1, :2])))
            root_orientation_errors.append(quat_distance(data.qpos[3:7], qpos_ref[step + 1, 3:7]))
            root_z_values.append(float(data.qpos[2]))
            tilt_values.append(root_tilt(data.qpos[3:7]))

        if mode == "free_root_balance":
            tilt = root_tilt(data.qpos[3:7])
            if float(data.qpos[2]) < FALL_ROOT_Z_M or tilt > FALL_TILT_RAD:
                fall_time = (step + 1) * model.opt.timestep
                achieved_steps = step + 1
                break

    error_array = np.asarray(errors, dtype=np.float64)
    group_metrics = {}
    for group, indices in groups.items():
        group_error = error_array[:, indices] if error_array.size else np.empty((0, len(indices)))
        group_metrics[group] = {
            "rmse_rad": float(np.sqrt(np.mean(np.square(group_error)))) if group_error.size else None,
            "abs_p95_rad": percentile(np.abs(group_error).ravel(), 95),
            "abs_max_rad": float(np.max(np.abs(group_error))) if group_error.size else None,
        }
    simulated_duration = min(achieved_steps * model.opt.timestep, duration)
    return {
        "mode": mode,
        "interpretation": (
            "joint/torque trackability only; root is externally prescribed and balance is not measured"
            if mode == "prescribed_root_trackability"
            else "free-root open-loop PD feasibility; not a learned-policy evaluation"
        ),
        "reference_duration_s": duration,
        "simulated_duration_s": simulated_duration,
        "duration_fraction": float(simulated_duration / duration) if duration else 1.0,
        "fell": fall_time is not None,
        "fall_time_s": fall_time,
        "fall_thresholds": {"root_z_m": FALL_ROOT_Z_M, "root_tilt_rad": FALL_TILT_RAD},
        "joint_tracking": group_metrics,
        "root_xy_error_rmse_m": (
            float(np.sqrt(np.mean(np.square(root_xy_errors)))) if root_xy_errors else None
        ),
        "root_xy_error_final_m": root_xy_errors[-1] if root_xy_errors else None,
        "root_orientation_error_p95_rad": percentile(root_orientation_errors, 95),
        "root_z_min_m": min(root_z_values) if root_z_values else float(data.qpos[2]),
        "root_tilt_max_rad": max(tilt_values) if tilt_values else root_tilt(data.qpos[3:7]),
        "torque_saturation_fraction": float(np.sum(saturation_count) / effort_count) if effort_count else 0.0,
        "torque_saturation_fraction_by_joint": {
            name: float(count / max(1, achieved_steps))
            for name, count in zip(contract.actuator_joint_names, saturation_count)
        },
        "contact_step_fraction": {
            side: float(count / max(1, achieved_steps)) for side, count in contact_steps.items()
        },
        "contact_slip_speed_p95_mps": {
            side: percentile(values, 95) for side, values in slip_speeds.items()
        },
        "normal_force_p50_n": {
            side: percentile(values, 50) for side, values in normal_forces.items()
        },
        "target_joint_limits": target_limit_report(model, entry),
    }


def compare_results(rows: list[dict[str, Any]], offline_report: Path) -> dict[str, Any]:
    by_variant_mode: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_variant_mode.setdefault((row["variant"], row["mode"]), []).append(row)

    aggregates: dict[str, Any] = {}
    for variant in ("current_v4_exact30", "toe_to_forefoot_exact30_smooth9"):
        aggregates[variant] = {}
        for mode in MODES:
            subset = by_variant_mode[(variant, mode)]
            aggregates[variant][mode] = {
                "motion_count": len(subset),
                "full_duration_count": int(sum(row["duration_fraction"] >= 0.999 for row in subset)),
                "mean_duration_fraction": float(np.mean([row["duration_fraction"] for row in subset])),
                "mean_leg_rmse_rad": mean_or_none(
                    row["joint_tracking"]["legs"]["rmse_rad"] for row in subset
                ),
                "mean_all_joint_rmse_rad": mean_or_none(
                    row["joint_tracking"]["all"]["rmse_rad"] for row in subset
                ),
                "mean_torque_saturation_fraction": float(np.mean([
                    row["torque_saturation_fraction"] for row in subset
                ])),
                "mean_root_xy_error_rmse_m": mean_or_none(
                    row["root_xy_error_rmse_m"] for row in subset
                ),
                "mean_contact_slip_p95_mps": mean_or_none(
                    value
                    for row in subset
                    for value in row["contact_slip_speed_p95_mps"].values()
                ),
            }

    prior = None
    if offline_report.exists():
        raw = json.loads(offline_report.read_text(encoding="utf-8"))
        # Keep the old report as immutable evidence rather than reproducing its
        # calculations or silently changing its gate here.
        prior = {
            "path": str(offline_report),
            "sha256": sha256(offline_report),
            "status": raw.get("status"),
            "smooth9_gate": raw.get("gates", {}).get("9"),
            "first_passing_window": raw.get("first_passing_window"),
            "training": raw.get("training"),
        }

    current = aggregates["current_v4_exact30"]
    toe = aggregates["toe_to_forefoot_exact30_smooth9"]
    track_not_worse = (
        toe["prescribed_root_trackability"]["mean_leg_rmse_rad"]
        <= 1.10 * current["prescribed_root_trackability"]["mean_leg_rmse_rad"]
        and toe["prescribed_root_trackability"]["mean_torque_saturation_fraction"]
        <= current["prescribed_root_trackability"]["mean_torque_saturation_fraction"] + 0.02
    )
    balance_not_worse = (
        toe["free_root_balance"]["mean_duration_fraction"]
        >= current["free_root_balance"]["mean_duration_fraction"] - 0.02
    )
    prior_pass = bool(prior and prior.get("smooth9_gate", {}).get("winner", False))
    promotable = bool(track_not_worse and balance_not_worse and prior_pass)
    return {
        "aggregates": aggregates,
        "prior_offline_continuity_evidence": prior,
        "screen_checks": {
            "prescribed_root_trackability_not_worse": track_not_worse,
            "free_root_balance_not_worse": balance_not_worse,
            "prior_offline_continuity_gate_pass": prior_pass,
        },
        "toe_forefoot_promotable": promotable,
        "decision": (
            "PROMOTE_TO_NEXT_GATE" if promotable else "DO_NOT_PROMOTE_FROM_THIS_SCREEN"
        ),
        "interpretation": (
            "toe-to-forefoot improves prescribed-root joint trackability but degrades "
            "free-root survival/slip; the local geometric benefit does not close root-contact balance"
        ),
        "next_step": (
            "retain current-v4 for training; do not train smooth9; any new repair must jointly "
            "address root/COM, contact timing, and foot targets rather than foot-only smoothing"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    def fmt(value: float | None, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# X2 Forefoot AimDK v1.0 Official-Physics Screen",
        "",
        f"- 结论：**{report['comparison']['decision']}**。",
        "- 本实验不训练，不载入 checkpoint；只比较同一批 reference 在官方 MuJoCo/PD 下的物理响应。",
        "- `prescribed_root_trackability` 的 root 每个 1 ms 被外部写回，仅回答关节/力矩可跟踪性，**不能证明平衡或可部署**。",
        "- `free_root_balance` 不给 root 外力，仅回答开环 PD 下的初步动力学兼容性；它仍不等价于闭环策略成功。",
        "",
        "## 假设 / 干预 / 对照",
        "",
        "- 假设：toe→forefoot 的语义修正若有真实物理价值，应在不恶化固定-root关节可跟踪性的同时，提高或至少不降低 free-root 生存。",
        "- 干预：只替换 `current_v4_exact30` 为 `toe_to_forefoot_exact30_smooth9` reference。",
        "- 对照：官方 scene/model、1 kHz physics、50 Hz command、官方 29DOF RL Kp/Kd、官方 actuator torque limits 完全相同；头部按同一官方默认 20/1 锁定。",
        "- Panel：walk straight、left/right turn、stand→walk、walk→stand。",
        "",
        "## 聚合结果",
        "",
        "| variant | mode | full | duration | leg RMSE(rad) | sat. | root XY RMSE(m) | slip p95(m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    agg = report["comparison"]["aggregates"]
    for variant, modes in agg.items():
        for mode, values in modes.items():
            lines.append(
                f"| {variant} | {mode} | {values['full_duration_count']}/{values['motion_count']} | "
                f"{values['mean_duration_fraction']:.3f} | {values['mean_leg_rmse_rad']:.4f} | "
                f"{values['mean_torque_saturation_fraction']:.4f} | "
                f"{fmt(values['mean_root_xy_error_rmse_m'])} | "
                f"{fmt(values['mean_contact_slip_p95_mps'])} |"
            )
    checks = report["comparison"]["screen_checks"]
    lines += [
        "",
        "## 门禁判断",
        "",
        f"- prescribed-root trackability 不劣化：`{checks['prescribed_root_trackability_not_worse']}`。",
        f"- free-root balance 不劣化：`{checks['free_root_balance_not_worse']}`。",
        f"- 既有离线连续性门通过：`{checks['prior_offline_continuity_gate_pass']}`。",
        f"- toe-forefoot 可晋升：`{report['comparison']['toe_forefoot_promotable']}`。",
        "",
        "即使 fixed-root 指标较好，也不得将其写成‘动作动力学可执行’；free-root 的失败也只说明开环 PD reference 不自稳，不能单独否定闭环 RL policy。",
        "",
        "## 结果 / 结论 / 下一步",
        "",
        "- 结果：toe→forefoot 在 prescribed-root 下更容易被官方 PD 跟踪，但在 5/5 free-root 动作中都更早失稳，且接触滑移更大。",
        "- 结论：它改善了局部几何/关节目标，却没有改善 root-contact 闭环；不能用 fixed-root 的好看数字覆盖 free-root 退化。",
        "- 下一步：训练数据继续保留 current-v4；不训练 smooth9。若再修 reference，应联合优化 root/COM、接触时序和足部目标，而不是继续做 foot-only offset/smoothing。",
        "",
        "## 逐动作结果",
        "",
        "| role | variant | mode | duration | fall(s) | leg RMSE | sat. | root XY RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        fall = "-" if row["fall_time_s"] is None else f"{row['fall_time_s']:.3f}"
        lines.append(
            f"| {row['panel_role']} | {row['variant']} | {row['mode']} | "
            f"{row['duration_fraction']:.3f} | {fall} | "
            f"{row['joint_tracking']['legs']['rmse_rad']:.4f} | "
            f"{row['torque_saturation_fraction']:.4f} | {row['root_xy_error_rmse_m']:.4f} |"
        )
    lines += [
        "",
        "## 资产契约",
        "",
        f"- scene: `{report['provenance']['scene']['path']}` (`{report['provenance']['scene']['sha256']}`)",
        f"- x2.xml: `{report['provenance']['model']['path']}` (`{report['provenance']['model']['sha256']}`)",
        f"- motion_control.yaml: `{report['provenance']['control']['path']}` (`{report['provenance']['control']['sha256']}`)",
        f"- physics/control period: `{report['contract']['physics_dt_s']}` / `{report['contract']['control_dt_s']}` s",
        f"- head contract: `{report['contract']['head_gain_source']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--control-yaml", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--forefoot", type=Path, default=DEFAULT_FOREFOOT)
    parser.add_argument("--offline-report", type=Path, default=DEFAULT_OFFLINE_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()

    current = joblib.load(args.current)
    forefoot = joblib.load(args.forefoot)
    pair_contract = validate_motion_pair(current, forefoot)
    role_to_key = {entry["panel_role"]: key for key, entry in current.items()}
    missing = [role for role in args.roles if role not in role_to_key]
    if missing:
        raise ValueError(f"roles missing from panel: {missing}")

    probe_model = mujoco.MjModel.from_xml_path(str(args.scene))
    control_contract = build_control_contract(probe_model, args.control_yaml)
    variants = {
        "current_v4_exact30": current,
        "toe_to_forefoot_exact30_smooth9": forefoot,
    }
    results = []
    for role in args.roles:
        key = role_to_key[role]
        for variant, motions in variants.items():
            for mode in args.modes:
                print(f"[screen] {role} / {variant} / {mode}", flush=True)
                row = simulate_case(args.scene, args.control_yaml, motions[key], mode)
                row.update({"panel_role": role, "motion_key": key, "variant": variant})
                results.append(row)

    model_xml = args.scene.parent / "x2.xml"
    report = {
        "schema_version": "x2_forefoot_official_physics_screen_v1",
        "provenance": {
            "scene": {"path": str(args.scene), "sha256": sha256(args.scene)},
            "model": {"path": str(model_xml), "sha256": sha256(model_xml)},
            "control": {"path": str(args.control_yaml), "sha256": sha256(args.control_yaml)},
            "current_cache": {"path": str(args.current), "sha256": sha256(args.current)},
            "forefoot_cache": {"path": str(args.forefoot), "sha256": sha256(args.forefoot)},
        },
        "contract": {
            "physics_dt_s": float(probe_model.opt.timestep),
            "control_dt_s": control_contract.control_dt,
            "control_semantics": "position PD, qd_target=0, torque clipped to official actuator ctrlrange",
            "body_gains": "official rl_config seq/kps/kds (29 DOF)",
            "head_gain_source": control_contract.head_gain_source,
            "free_root_external_wrench": False,
            "prescribed_root_external_write": "qpos/qvel root written every physics step",
        },
        "pair_contract": pair_contract,
        "results": results,
        "comparison": compare_results(results, args.offline_report),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
