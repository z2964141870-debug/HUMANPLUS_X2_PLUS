#!/usr/bin/env python3
"""Phase15 one-shot subscriber-receipt replay of the Phase14 official capture.

The scored reference is always the recorded 50 Hz actual state.  Command
events are only control inputs.  They are consumed in saved global callback
receipt order; no publisher time, gain, delay, filtering, or time shift is
inferred or scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO = Path(__file__).resolve().parents[2]
for value in (REPO / "tools", REPO / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import retarget.run_x2_forefoot_official_physics_screen as physics
import retarget.run_x2_native_gold_trackability_phase12 as phase12


DEFAULT_SOURCE = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_event_v2_20260809/official_native_event_v2_20s.npz"
)
DEFAULT_MANIFEST = DEFAULT_SOURCE.with_suffix(".manifest.json")
DEFAULT_PHASE12 = REPO / "reports/retarget/x2_native_gold_trackability_phase12.json"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_event_replay_phase15.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_event_replay_phase15.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _q_by_model_order(
    values: np.ndarray, source_names: list[str], model_names: tuple[str, ...]
) -> np.ndarray:
    index = {name: i for i, name in enumerate(source_names)}
    if set(index) != set(model_names):
        raise ValueError("Phase14 snapshot and official model joint sets differ")
    return np.asarray(values)[..., [index[name] for name in model_names]]


def build_reference(
    source: Path, start_snapshot_index: int = 0
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(source, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    data = mujoco.MjData(model)
    contract = physics.build_control_contract(model, physics.DEFAULT_CONTROL)
    names = [str(value) for value in raw["joint_names"].tolist()]
    if not 0 <= start_snapshot_index < len(raw["time_s"]):
        raise ValueError("invalid start snapshot index")
    rows = slice(start_snapshot_index, None)
    q = _q_by_model_order(raw["joint_q_rad"][rows], names, contract.joint_names).astype(np.float64)
    dq = _q_by_model_order(raw["joint_dq_radps"][rows], names, contract.joint_names).astype(np.float64)
    root_pos = raw["root_pos_w_m"][rows].astype(np.float64)
    root_quat = raw["root_quat_xyzw"][rows].astype(np.float64)
    root_lin = raw["root_lin_vel_w_mps"][rows].astype(np.float64)
    root_ang = raw["root_ang_vel"][rows].astype(np.float64)
    snapshot_receipt = raw["snapshot_receipt_monotonic_ns"][rows].astype(np.int64)
    times = (snapshot_receipt - snapshot_receipt[0]).astype(np.float64) * 1.0e-9
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("snapshot receipt times are not strictly monotonic")
    if q.shape != (len(times), 31):
        raise ValueError("Phase14 actual-q reference is not Nx31")

    body_ids = np.arange(1, model.nbody, dtype=np.int64)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in body_ids]
    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    entry_index = {name: i for i, name in enumerate(contract.joint_names)}
    actuator_indices = np.asarray([entry_index[name] for name in contract.actuator_joint_names])
    body_pos: list[np.ndarray] = []
    body_quat: list[np.ndarray] = []
    contact = {"left": [], "right": []}
    for frame in range(len(times)):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame, [3, 0, 1, 2]]
        data.qpos[contract.qpos_addresses] = q[frame, actuator_indices]
        data.qvel[:3] = root_lin[frame]
        data.qvel[3:6] = root_ang[frame]
        data.qvel[contract.qvel_addresses] = dq[frame, actuator_indices]
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[body_ids].copy())
        body_quat.append(data.xquat[body_ids][:, [1, 2, 3, 0]].copy())
        current = phase12.contact_now(model, data, floor, geom_side)
        for side in ("left", "right"):
            contact[side].append(bool(current[side]))

    reference = {
        "key": "official_native_event_v2_20s",
        "fps": 50.0,
        "times": times,
        "snapshot_receipt_ns": snapshot_receipt,
        "joint_names": list(contract.joint_names),
        "body_names": body_names,
        "q": q,
        "dq": dq,
        "root_pos": root_pos,
        "root_quat_xyzw": root_quat,
        "root_lin_vel": root_lin,
        "root_ang_vel": root_ang,
        "body_pos": np.asarray(body_pos),
        "body_quat_xyzw": np.asarray(body_quat),
        "contact": {side: np.asarray(values, dtype=bool) for side, values in contact.items()},
        "target_source": "Phase14 50Hz recorded actual q/root; contact is reconstructed official-model collision geometry",
        "recorded_command_used_as_reference": False,
        "source_start_elapsed_s": float(raw["time_s"][start_snapshot_index]),
        "source_start_snapshot_index": int(start_snapshot_index),
    }
    events = {
        "elapsed_ns": raw["command_event_elapsed_ns"].astype(np.int64),
        "global_index": raw["command_event_global_index"].astype(np.int64),
        "group": raw["command_event_group"].astype(str),
        "changed": raw["command_event_changed_joint_mask"].astype(bool),
        "valid": raw["command_event_valid"].astype(bool),
        "q": raw["command_event_q_rad"].astype(np.float64),
        "kp": raw["command_event_kp"].astype(np.float64),
        "kd": raw["command_event_kd"].astype(np.float64),
        "source_names": np.asarray(names),
    }
    return reference, events


def interpolate_root(reference: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, ...]:
    source_t = reference["times"]
    clipped = np.minimum(np.asarray(times, dtype=np.float64), source_t[-1])
    pos = np.column_stack([
        np.interp(clipped, source_t, reference["root_pos"][:, axis]) for axis in range(3)
    ])
    quat = Slerp(source_t, Rotation.from_quat(reference["root_quat_xyzw"]))(clipped).as_quat()
    lin = np.column_stack([
        np.interp(clipped, source_t, reference["root_lin_vel"][:, axis]) for axis in range(3)
    ])
    ang = np.column_stack([
        np.interp(clipped, source_t, reference["root_ang_vel"][:, axis]) for axis in range(3)
    ])
    return pos, quat, lin, ang


def initial_event_context(reference: dict[str, Any], events: dict[str, np.ndarray]) -> dict[str, Any]:
    # Event elapsed is relative to capture start; recover snapshot elapsed from
    # the source NPZ time_s, which uses that same capture origin.
    source_start_elapsed_ns = int(round(float(reference.get("source_start_elapsed_s", 0.0)) * 1.0e9))
    prior = np.flatnonzero(events["elapsed_ns"] <= source_start_elapsed_ns)
    if prior.size == 0:
        raise ValueError("no command context exists before first snapshot")
    index = int(prior[-1])
    return {
        "event_index": index,
        "events_before_or_at_snapshot0": int(prior.size),
        "q": events["q"][index].copy(),
        "kp": events["kp"][index].copy(),
        "kd": events["kd"][index].copy(),
        "valid": events["valid"][index].copy(),
        "next_event_index": index + 1,
        "source_start_elapsed_ns": source_start_elapsed_ns,
    }


def _record_state(
    data: mujoco.MjData,
    contract: Any,
    body_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_name = {
        name: float(data.qpos[address])
        for name, address in zip(contract.actuator_joint_names, contract.qpos_addresses)
    }
    q = np.asarray([by_name[name] for name in contract.joint_names])
    return (
        q,
        data.xpos[body_ids].copy(),
        data.xquat[body_ids][:, [1, 2, 3, 0]].copy(),
        data.qpos[:3].copy(),
        data.qpos[3:7][[1, 2, 3, 0]].copy(),
    )


def simulate_event_replay(
    reference: dict[str, Any], events: dict[str, np.ndarray], mode: str,
    *, expected_active_joint_count: int = 29, allow_head_active: bool = False,
) -> dict[str, Any]:
    if mode not in (phase12.MODE_PRESCRIBED, phase12.MODE_FREE):
        raise ValueError(mode)
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    data = mujoco.MjData(model)
    contract = physics.build_control_contract(model, physics.DEFAULT_CONTROL)
    if tuple(reference["joint_names"]) != contract.joint_names:
        raise ValueError("reference order differs from official model order")
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError("official physics timestep changed")

    source_names = events["source_names"].astype(str).tolist()
    event_index = {name: i for i, name in enumerate(source_names)}
    actuator_event_indices = np.asarray([event_index[name] for name in contract.actuator_joint_names])
    entry_index = {name: i for i, name in enumerate(contract.joint_names)}
    actuator_reference_indices = np.asarray([entry_index[name] for name in contract.actuator_joint_names])
    body_ids = np.arange(1, model.nbody, dtype=np.int64)

    # Shared capture elapsed clock: snapshot time_s[0] is injected by build caller.
    context = initial_event_context(reference, events)
    target = context["q"].copy()
    kp = context["kp"].copy()
    kd = context["kd"].copy()
    active = context["valid"].copy()
    if int(active.sum()) != expected_active_joint_count:
        raise ValueError("snapshot0 command context active count differs from frozen source contract")
    head_mask = np.asarray([name.startswith("head_") for name in source_names])
    if np.any(active & head_mask) and not allow_head_active:
        raise ValueError("Phase14 head unexpectedly active")

    duration = float(reference["times"][-1])
    n_steps = int(math.ceil(duration / model.opt.timestep))
    physics_times = np.arange(n_steps + 1, dtype=np.float64) * model.opt.timestep
    root_pos_i, root_quat_i, root_lin_i, root_ang_i = interpolate_root(reference, physics_times)
    root_wxyz_i = root_quat_i[:, [3, 0, 1, 2]]

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = reference["root_pos"][0]
    data.qpos[3:7] = reference["root_quat_xyzw"][0, [3, 0, 1, 2]]
    data.qpos[contract.qpos_addresses] = reference["q"][0, actuator_reference_indices]
    data.qvel[:3] = reference["root_lin_vel"][0]
    data.qvel[3:6] = reference["root_ang_vel"][0]
    data.qvel[contract.qvel_addresses] = reference["dq"][0, actuator_reference_indices]
    mujoco.mj_forward(model, data)

    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    previous_geom_pos = data.geom_xpos.copy()
    first = _record_state(data, contract, body_ids)
    q_samples, body_pos_samples, body_quat_samples, root_samples, root_quat_samples = (
        [first[0]], [first[1]], [first[2]], [first[3]], [first[4]]
    )
    realized = {"left": [], "right": []}
    interval_contact = {"left": False, "right": False}
    slip: list[float] = []
    saturation = np.zeros(model.nu, dtype=np.int64)
    active_effort_count = 0
    torque_fraction: list[float] = []
    next_snapshot = 1
    event_cursor = int(context["next_event_index"])
    events_consumed = 0
    update_counts = {"leg": 0, "waist": 0, "arm": 0, "head": 0}
    fell_at: float | None = None
    nonfinite_at: float | None = None
    achieved_steps = n_steps
    source_start_ns = int(context["source_start_elapsed_ns"])

    for step in range(n_steps):
        time_s = step * model.opt.timestep
        absolute_elapsed_ns = source_start_ns + int(round(time_s * 1.0e9))
        while event_cursor < len(events["elapsed_ns"]) and int(events["elapsed_ns"][event_cursor]) <= absolute_elapsed_ns:
            update = events["changed"][event_cursor] & events["valid"][event_cursor]
            target[update] = events["q"][event_cursor, update]
            kp[update] = events["kp"][event_cursor, update]
            kd[update] = events["kd"][event_cursor, update]
            active |= update
            update_counts[str(events["group"][event_cursor])] += 1
            event_cursor += 1
            events_consumed += 1

        if mode == phase12.MODE_PRESCRIBED:
            data.qpos[:3] = root_pos_i[step]
            data.qpos[3:7] = root_wxyz_i[step]
            data.qvel[:3] = root_lin_i[step]
            data.qvel[3:6] = root_ang_i[step]

        active_act = active[actuator_event_indices]
        target_act = target[actuator_event_indices]
        kp_act = kp[actuator_event_indices]
        kd_act = kd[actuator_event_indices]
        actual_q = data.qpos[contract.qpos_addresses]
        actual_dq = data.qvel[contract.qvel_addresses]
        raw = np.zeros(model.nu, dtype=np.float64)
        raw[active_act] = (
            kp_act[active_act] * (target_act[active_act] - actual_q[active_act])
            - kd_act[active_act] * actual_dq[active_act]
        )
        saturated = active_act & ((raw < contract.torque_low) | (raw > contract.torque_high))
        saturation += saturated
        active_effort_count += int(active_act.sum())
        denom = np.maximum(np.maximum(np.abs(contract.torque_low), np.abs(contract.torque_high)), 1.0e-9)
        torque_fraction.extend((np.abs(raw[active_act]) / denom[active_act]).tolist())
        data.ctrl[:] = np.clip(raw, contract.torque_low, contract.torque_high)
        mujoco.mj_step(model, data)

        if mode == phase12.MODE_PRESCRIBED:
            data.qpos[:3] = root_pos_i[step + 1]
            data.qpos[3:7] = root_wxyz_i[step + 1]
            data.qvel[:3] = root_lin_i[step + 1]
            data.qvel[3:6] = root_ang_i[step + 1]
            mujoco.mj_forward(model, data)

        contacts = phase12.contact_now(model, data, floor, geom_side)
        for side in ("left", "right"):
            interval_contact[side] = interval_contact[side] or bool(contacts[side])
            for geom_id in contacts[side]:
                delta = data.geom_xpos[geom_id, :2] - previous_geom_pos[geom_id, :2]
                slip.append(float(np.linalg.norm(delta) / model.opt.timestep))
        previous_geom_pos[:] = data.geom_xpos

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite_at = (step + 1) * model.opt.timestep
            achieved_steps = step + 1
            break
        if mode == phase12.MODE_FREE:
            tilt = physics.root_tilt(data.qpos[3:7])
            if float(data.qpos[2]) < phase12.FALL_ROOT_Z_M or tilt > phase12.FALL_TILT_RAD:
                fell_at = (step + 1) * model.opt.timestep
                achieved_steps = step + 1
                break

        reached = (step + 1) * model.opt.timestep
        while next_snapshot < len(reference["times"]) and reference["times"][next_snapshot] <= reached + 1.0e-12:
            row = _record_state(data, contract, body_ids)
            q_samples.append(row[0])
            body_pos_samples.append(row[1])
            body_quat_samples.append(row[2])
            root_samples.append(row[3])
            root_quat_samples.append(row[4])
            for side in ("left", "right"):
                realized[side].append(interval_contact[side])
                interval_contact[side] = False
            next_snapshot += 1

    sample_count = len(q_samples)
    q_actual = np.asarray(q_samples)
    body_pos = np.asarray(body_pos_samples)
    body_quat = np.asarray(body_quat_samples)
    replay_root = np.asarray(root_samples)
    replay_root_quat = np.asarray(root_quat_samples)
    ref_q = reference["q"][:sample_count]
    ref_body_pos = reference["body_pos"][:sample_count]
    ref_body_quat = reference["body_quat_xyzw"][:sample_count]
    ref_root = reference["root_pos"][:sample_count]
    ref_root_quat = reference["root_quat_xyzw"][:sample_count]
    q_error = q_actual - ref_q
    body_world_error = np.linalg.norm(body_pos - ref_body_pos, axis=-1)
    body_relative_error = np.linalg.norm(
        (body_pos - replay_root[:, None]) - (ref_body_pos - ref_root[:, None]), axis=-1
    )
    body_ori_error = phase12.quat_distance_xyzw(body_quat, ref_body_quat)
    root_pos_error = np.linalg.norm(replay_root - ref_root, axis=-1)
    root_ori_error = phase12.quat_distance_xyzw(replay_root_quat, ref_root_quat)
    actual_step = np.abs(np.diff(q_actual, axis=0))
    dt_samples = np.diff(reference["times"][:sample_count])
    actual_velocity = actual_step / np.maximum(dt_samples[:, None], 1.0e-12)
    realized_contact = {side: np.asarray(values, dtype=bool) for side, values in realized.items()}
    ref_contact = {
        side: reference["contact"][side][1:1 + len(realized_contact[side])]
        for side in ("left", "right")
    }
    agreement = {
        side: float(np.mean(realized_contact[side] == ref_contact[side])) if len(realized_contact[side]) else 0.0
        for side in ("left", "right")
    }
    agreement["mean"] = float(np.mean([agreement["left"], agreement["right"]]))
    achieved = min(achieved_steps * model.opt.timestep, duration)

    head_indices = np.asarray([i for i, name in enumerate(reference["joint_names"]) if name.startswith("head_")])
    body_indices = np.asarray([i for i in range(31) if i not in set(head_indices.tolist())])
    return {
        "mode": mode,
        "interpretation": (
            "subscriber-receipt recorded-control replay with externally prescribed root; trackability only"
            if mode == phase12.MODE_PRESCRIBED else
            "subscriber-receipt recorded-control free-root replay; not a learned-policy evaluation"
        ),
        "reference_duration_s": duration,
        "simulated_duration_s": achieved,
        "duration_fraction": float(achieved / duration),
        "fell": fell_at is not None,
        "fall_time_s": fell_at,
        "nonfinite_time_s": nonfinite_at,
        "q_tracking": {
            "rmse_rad": float(np.sqrt(np.mean(q_error ** 2))),
            "active29_rmse_rad": float(np.sqrt(np.mean(q_error[:, body_indices] ** 2))),
            "head2_rmse_rad": float(np.sqrt(np.mean(q_error[:, head_indices] ** 2))),
            "abs_p95_rad": float(np.percentile(np.abs(q_error), 95)),
            "abs_max_rad": float(np.max(np.abs(q_error))),
        },
        "body_tracking": {
            "world_position_p95_max_m": [float(np.percentile(body_world_error, 95)), float(np.max(body_world_error))],
            "root_relative_position_p95_max_m": [float(np.percentile(body_relative_error, 95)), float(np.max(body_relative_error))],
            "orientation_p95_max_rad": [float(np.percentile(body_ori_error, 95)), float(np.max(body_ori_error))],
        },
        "root_tracking": {
            "position_rmse_final_m": [float(np.sqrt(np.mean(root_pos_error ** 2))), float(root_pos_error[-1])],
            "xy_rmse_final_m": [float(np.sqrt(np.mean(np.sum((replay_root[:, :2] - ref_root[:, :2]) ** 2, axis=-1)))), float(np.linalg.norm(replay_root[-1, :2] - ref_root[-1, :2]))],
            "orientation_p95_max_rad": [float(np.percentile(root_ori_error, 95)), float(np.max(root_ori_error))],
            "z_min_m": float(np.min(replay_root[:, 2])),
            "tilt_max_rad": float(max(physics.root_tilt(q[[3, 0, 1, 2]]) for q in replay_root_quat)),
        },
        "contact": {
            "reference_phase": phase12.phase_stats(ref_contact),
            "realized_phase": phase12.phase_stats(realized_contact),
            "agreement": agreement,
            "f1": {side: phase12.f1_binary(ref_contact[side], realized_contact[side]) for side in ("left", "right")},
            "truth_boundary": "reference=model collision reconstructed from recorded actual state; realized=official MuJoCo collision; neither is hardware GRF/COP/wrench",
        },
        "slip_p95_max_mps": [phase12.percentile(np.asarray(slip), 95), float(max(slip)) if slip else None],
        "action_target": phase12.source_kinematics(reference, model),
        "torque": {
            "saturation_fraction": float(np.sum(saturation) / max(1, active_effort_count)),
            "raw_abs_over_limit_p95_max": [float(np.percentile(torque_fraction, 95)), float(np.max(torque_fraction))],
            "saturation_by_joint": {name: float(count / max(1, achieved_steps)) for name, count in zip(contract.actuator_joint_names, saturation)},
        },
        "replay_joint_motion": {
            "step_abs_p95_max_rad": [float(np.percentile(actual_step, 95)), float(np.max(actual_step))],
            "velocity_abs_p95_max_radps": [float(np.percentile(actual_velocity, 95)), float(np.max(actual_velocity))],
        },
        "event_replay": {
            "initial_context_event_index": int(context["event_index"]),
            "events_before_or_at_snapshot0": int(context["events_before_or_at_snapshot0"]),
            "events_consumed_after_snapshot0": int(events_consumed),
            "next_unconsumed_event_index": int(event_cursor),
            "group_updates_consumed": update_counts,
            "head_active": bool(np.any(active & head_mask)),
            "active_joint_count": int(active.sum()),
            "time_source": "subscriber callback monotonic receipt elapsed; publisher timestamp unavailable",
        },
        "initialization": {
            "q": "Phase14 first 50Hz snapshot recorded actual q",
            "dq": "Phase14 first 50Hz snapshot recorded actual dq",
            "root_pose": "Phase14 first 50Hz snapshot recorded root position/quaternion",
            "root_velocity": "Phase14 first 50Hz snapshot recorded root lin/ang velocity",
            "command_context": "last full valid command context at or before snapshot0; then changed+valid updates in global receipt order",
            "head": (
                "source active context replayed exactly"
                if np.any(active & head_mask) else
                "source inactive; zero actuator torque, no fabricated head target/gain"
            ),
            "solver_warmstart": "unavailable; reset default/zero",
            "contact_constraint_warmstart": "unavailable; reconstructed by mj_forward",
            "controller_hidden_state": "not replayed; recorded command events are the control input",
        },
    }


def render(report: dict[str, Any]) -> str:
    prescribed = report["replay"][phase12.MODE_PRESCRIBED]
    free = report["replay"].get(phase12.MODE_FREE)
    historical = report["historical_control_phase12"]
    lines = [
        "# X2 Native Subscriber-Receipt Event Replay Phase15",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- reference/评分始终是 Phase14 50Hz recorded actual q/root/model-contact；recorded command 只作 control input。",
        "- contact 是模型重建/官方MuJoCo碰撞，不是实机 GRF、COP、wrench；source trace稳定不冒充replay稳定。",
        "",
        "## 假设",
        "",
        "若 Phase12 失败主要来自 actual-q target + synthetic fixed-PD 控制合同不匹配，那么按 Phase14 subscriber receipt 时序回放 source q/Kp/Kd 应显著改善 prescribed-root trackability。",
        "",
        "## 干预 / 对照",
        "",
        "- 干预：逐1ms处理所有到期event，严格使用保存的global receipt order；changed+valid mask只更新active29 q/Kp/Kd，head保持source inactive。",
        f"- 历史对照 Phase12（不同8s源clip，仅作方法对照，不能当matched trajectory因果A/B）：q RMSE `{historical['q_rmse_rad']:.4f}`，body rel-pos p95 `{historical['body_relative_position_p95_m']:.4f}m`，contact `{historical['contact_agreement']:.3f}`，slip p95 `{historical['slip_p95_mps']:.3f}m/s`。",
        "- 初始化：首snapshot q/dq/root pose/root velocity；首snapshot前最后有效command context。缺失publisher timestamp、solver/contact warmstart。",
        "- 先 prescribed-root；仅过完全相同 Phase12 gate 才允许唯一一次 free-root。无时移/增益/滤波扫描。",
        "",
        "## 结果",
        "",
        "| mode | survival | q RMSE(all31/active29/head2) | body rel-pos p95 | contact | SS(ref/real) | slip p95 | torque sat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in [prescribed] + ([free] if free is not None else []):
        lines.append(
            f"| {result['mode']} | {result['simulated_duration_s']:.3f}/{result['reference_duration_s']:.3f}s | "
            f"{result['q_tracking']['rmse_rad']:.4f}/{result['q_tracking']['active29_rmse_rad']:.4f}/{result['q_tracking']['head2_rmse_rad']:.4f} | "
            f"{result['body_tracking']['root_relative_position_p95_max_m'][0]:.4f}m | {result['contact']['agreement']['mean']:.3f} | "
            f"{result['contact']['reference_phase']['single_support_ratio']:.3f}/{result['contact']['realized_phase']['single_support_ratio']:.3f} | "
            f"{result['slip_p95_max_mps'][0]:.3f}m/s | {result['torque']['saturation_fraction']:.4f} |"
        )
    lines += [
        "",
        f"- event context：snapshot0前 `{prescribed['event_replay']['events_before_or_at_snapshot0']}` events；之后消费 `{prescribed['event_replay']['events_consumed_after_snapshot0']}`；active joints `{prescribed['event_replay']['active_joint_count']}`，head active `{prescribed['event_replay']['head_active']}`。",
        f"- prescribed gate：`{report['gate']['prescribed']['pass']}`，失败项 `{report['gate']['prescribed']['failed']}`。",
        f"- free-root executed：`{free is not None}`" + (f"；gate `{report['gate']['free']['pass']}`。" if free is not None else "；prescribed未过门，按预注册规则停止。"),
        "",
        "## 误差归因边界",
        "",
        "- recorded q/Kp/Kd 与subscriber receipt时序已用；若prescribed仍失败，不能再归因于synthetic fixed-PD本身。",
        "- 仍无法排除：publisher真实时间与callback receipt抖动、首snapshot之前的solver/contact warmstart、原仿真进程内部状态，以及1ms离散replay对异步event的量化。",
        "- 本结果不评价Any2Any、训练policy或真机部署。",
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase12", type=Path, default=DEFAULT_PHASE12)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.source) != manifest["npz"]["sha256"]:
        raise ValueError("Phase14 NPZ hash differs from manifest")
    reference, events = build_reference(args.source)
    # Preserve the capture-origin elapsed time needed to align event and snapshot clocks.
    with np.load(args.source, allow_pickle=False) as archive:
        reference["source_start_elapsed_s"] = float(archive["time_s"][0])
    source_metrics = phase12.source_kinematics(
        reference, mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    )
    prescribed = simulate_event_replay(reference, events, phase12.MODE_PRESCRIBED)
    prescribed_gate = phase12.prescribed_gate(prescribed, source_metrics)
    replay = {phase12.MODE_PRESCRIBED: prescribed}
    free_gate = None
    if prescribed_gate["pass"]:
        free = simulate_event_replay(reference, events, phase12.MODE_FREE)
        replay[phase12.MODE_FREE] = free
        free_gate = phase12.free_gate(free)

    if not prescribed_gate["pass"]:
        status = "PHASE15_RECORDED_CONTROL_PRESCRIBED_REJECTED"
        result = "recorded event q/Kp/Kd + subscriber receipt timing仍未通过prescribed trackability门；按规则未运行free-root。"
        conclusion = "Phase12失败不能只归因于synthetic fixed-PD；剩余差异位于receipt-vs-publisher timing、初始化/warmstart或未保存仿真内部状态。"
        next_step = "停止Phase15，不扫时移/增益/滤波；先由主线裁决是否值得采集publisher-stamped或sim-state checkpoint。"
    elif free_gate is not None and free_gate["pass"]:
        status = "PHASE15_RECORDED_CONTROL_PRESCRIBED_AND_FREE_PASSED"
        result = "recorded-control event replay同时通过prescribed和一次free-root门。"
        conclusion = "Phase14 control event contract可复现该source闭环片段；仍不证明Any2Any或跨具身训练成功。"
        next_step = "冻结为official dynamic control sanity；本阶段不训练。"
    else:
        status = "PHASE15_RECORDED_CONTROL_PRESCRIBED_PASSED_FREE_REJECTED"
        result = "recorded-control通过prescribed，但一次free-root失败。"
        conclusion = "event control contract可跟踪；缺失warmstart/内部状态使free-root闭环未复现，不能据此否定Any2Any。"
        next_step = "保留prescribed sanity证据，停止本阶段，不调参。"

    prior = json.loads(args.phase12.read_text(encoding="utf-8"))
    old = prior["replay"][phase12.MODE_PRESCRIBED]
    report = {
        "schema_version": "x2_native_event_replay_phase15_v1",
        "provenance": {
            "phase14_source": {"path": str(args.source), "sha256": sha256(args.source)},
            "phase14_manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "phase12_control": {"path": str(args.phase12), "sha256": sha256(args.phase12)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": sha256(physics.DEFAULT_CONTROL)},
        },
        "truth_boundary": {
            "target_is_50hz_recorded_actual_q_root_not_command": True,
            "command_is_control_input_only": True,
            "event_time_is_subscriber_receipt_not_publisher_time": True,
            "active29_head_inactive": True,
            "source_trace_stability_is_not_replay_stability": True,
            "prescribed_root_is_not_balance": True,
            "free_failure_does_not_reject_any2any": True,
            "contact_is_model_not_hardware_grf_cop_wrench": True,
            "training_base_real_robot": False,
        },
        "hypothesis": "Recorded event q/Kp/Kd at Phase14 subscriber receipt timing should repair a Phase12 synthetic-control mismatch if that mismatch was the dominant prescribed-root error source.",
        "intervention": "One subscriber-receipt event replay; changed+valid active29 updates, source-inactive head, prescribed first and free only conditionally.",
        "control": "Historical Phase12 actual-q + synthetic fixed-PD. It uses a different captured clip and is therefore a method control, not a matched-trajectory causal A/B.",
        "pre_registered_gates": phase12.GATES,
        "source_reference": source_metrics,
        "initialization": prescribed["initialization"],
        "historical_control_phase12": {
            "different_source_clip": True,
            "q_rmse_rad": old["q_tracking"]["rmse_rad"],
            "body_relative_position_p95_m": old["body_tracking"]["root_relative_position_p95_max_m"][0],
            "body_orientation_p95_rad": old["body_tracking"]["orientation_p95_max_rad"][0],
            "contact_agreement": old["contact"]["agreement"]["mean"],
            "slip_p95_mps": old["slip_p95_max_mps"][0],
            "torque_saturation": old["torque"]["saturation_fraction"],
        },
        "replay": replay,
        "gate": {"prescribed": prescribed_gate, "free": free_gate},
        "decision": {"status": status, "result": result, "conclusion": conclusion, "next_step": next_step},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
