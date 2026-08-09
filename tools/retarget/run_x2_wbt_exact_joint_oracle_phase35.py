#!/usr/bin/env python3
"""Phase35 sole free-root exact-joint kinematic feasibility oracle.

At every official 50 Hz control tick the frozen Phase30 WBT29 q/dq are written
exactly into the official AimDK v1 MuJoCo state.  Head joints stay at model
nominal, actuator torque is zero, and only the free root/contact dynamics are
integrated.  Direct state overwrite breaks physical momentum continuity; this
is an attribution oracle, never an executable controller or a Gold gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_gold_qualification_phase33 as phase33
import retarget.run_x2_forefoot_official_physics_screen as physics


PHASE33_JSON = REPO / "reports/retarget/x2_wbt_gold_qualification_phase33.json"
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_exact_joint_oracle_phase35.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_exact_joint_oracle_phase35.md"
OUTPUT_PREFLIGHT = REPO / "reports/retarget/x2_wbt_exact_joint_oracle_phase35_preflight.json"
OUTPUT_TRACE = REPO / "artifacts/official_x2/x2_wbt_phase35_exact_joint_oracle_trace.npz"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return None if values.size == 0 else float(np.percentile(values, q))


def foot_contact_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor: int,
    geom_side: dict[int, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    collision = np.zeros(2, dtype=bool)
    linear_impulse = np.zeros((2, 3), dtype=np.float64)
    torque_impulse = np.zeros((2, 3), dtype=np.float64)
    contact_count = np.zeros(2, dtype=np.int32)
    side_index = {"left": 0, "right": 1}
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 == floor and geom2 in geom_side:
            side = geom_side[geom2]
        elif geom2 == floor and geom1 in geom_side:
            side = geom_side[geom1]
        else:
            continue
        index = side_index[side]
        collision[index] = True
        contact_count[index] += 1
        wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        # Wrench is expressed in the contact frame.  Integrating over one
        # physics step gives an impulse diagnostic, not hardware force truth.
        linear_impulse[index] += wrench[:3] * model.opt.timestep
        torque_impulse[index] += wrench[3:] * model.opt.timestep
    return collision, linear_impulse, torque_impulse, contact_count


def run_oracle(reference: dict[str, Any], trace_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    data = mujoco.MjData(model)
    control = physics.build_control_contract(model, physics.DEFAULT_CONTROL)
    if tuple(reference["joint_names"]) != control.joint_names:
        raise ValueError("reference joint order differs from official contract")
    if not np.isclose(model.opt.timestep, 0.001) or not np.isclose(control.control_dt, 0.02):
        raise ValueError("official 1kHz/50Hz contract drift")
    steps_per_control = int(round(control.control_dt / model.opt.timestep))
    frames = len(reference["q"])
    n_steps = (frames - 1) * steps_per_control
    entry_index = {name: index for index, name in enumerate(reference["joint_names"])}
    actuator_reference_index = np.asarray([entry_index[name] for name in control.actuator_joint_names])
    body_actuator_index = np.asarray([
        index for index, name in enumerate(control.actuator_joint_names) if not name.startswith("head_")
    ])
    head_actuator_index = np.asarray([
        index for index, name in enumerate(control.actuator_joint_names) if name.startswith("head_")
    ])
    if len(body_actuator_index) != 29 or len(head_actuator_index) != 2:
        raise ValueError("official WBT29/head2 split changed")
    body_reference_index = actuator_reference_index[body_actuator_index]
    body_qpos = control.qpos_addresses[body_actuator_index]
    body_qvel = control.qvel_addresses[body_actuator_index]
    head_qpos = control.qpos_addresses[head_actuator_index]
    head_qvel = control.qvel_addresses[head_actuator_index]
    head_nominal = model.qpos0[head_qpos].copy()

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = reference["root_pos"][0]
    data.qpos[3:7] = reference["root_quat_xyzw"][0][[3, 0, 1, 2]]
    data.qvel[:3] = reference["root_lin_vel"][0]
    data.qvel[3:6] = reference["root_ang_vel"][0]
    data.qpos[body_qpos] = reference["q"][0, body_reference_index]
    data.qvel[body_qvel] = reference["dq"][0, body_reference_index]
    data.qpos[head_qpos] = head_nominal
    data.qvel[head_qvel] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    trace: dict[str, list[np.ndarray | float | int | bool]] = {
        "time_s": [], "root_position_m": [], "root_quaternion_wxyz": [],
        "root_linear_velocity_mps": [], "root_angular_velocity_radps": [], "root_tilt_rad": [],
        "official_sole_collision_lr": [], "contact_linear_impulse_lr_Ns": [],
        "contact_torque_impulse_lr_Nms": [], "contact_count_lr": [],
        "q_error_wbt29_rad": [], "qfrc_constraint_wbt29": [], "qfrc_actuator_wbt29": [],
        "control_frame": [], "post_overwrite": [],
    }
    overwrite_delta: list[np.ndarray] = []
    interval_collision = np.zeros(2, dtype=bool)
    realized_control: list[np.ndarray] = []
    terminal: dict[str, Any] | None = None

    def append_trace(time_s: float, frame: int, post_overwrite: bool, target: np.ndarray) -> None:
        collision, linear_impulse, torque_impulse, count = foot_contact_wrench(
            model, data, floor, geom_side
        )
        interval_collision[:] |= collision
        trace["time_s"].append(float(time_s))
        trace["root_position_m"].append(data.qpos[:3].copy())
        trace["root_quaternion_wxyz"].append(data.qpos[3:7].copy())
        trace["root_linear_velocity_mps"].append(data.qvel[:3].copy())
        trace["root_angular_velocity_radps"].append(data.qvel[3:6].copy())
        trace["root_tilt_rad"].append(float(physics.root_tilt(data.qpos[3:7])))
        trace["official_sole_collision_lr"].append(collision)
        trace["contact_linear_impulse_lr_Ns"].append(linear_impulse)
        trace["contact_torque_impulse_lr_Nms"].append(torque_impulse)
        trace["contact_count_lr"].append(count)
        trace["q_error_wbt29_rad"].append(data.qpos[body_qpos].copy() - target)
        trace["qfrc_constraint_wbt29"].append(data.qfrc_constraint[body_qvel].copy())
        trace["qfrc_actuator_wbt29"].append(data.qfrc_actuator[body_qvel].copy())
        trace["control_frame"].append(int(frame))
        trace["post_overwrite"].append(bool(post_overwrite))

    append_trace(0.0, 0, True, reference["q"][0, body_reference_index])
    stop = False
    completed_physics_steps = 0
    for frame in range(frames - 1):
        target = reference["q"][frame, body_reference_index]
        target_dq = reference["dq"][frame, body_reference_index]
        if frame > 0:
            overwrite_delta.append(data.qpos[body_qpos].copy() - target)
            data.qpos[body_qpos] = target
            data.qvel[body_qvel] = target_dq
            data.qpos[head_qpos] = head_nominal
            data.qvel[head_qvel] = 0.0
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            append_trace(frame * control.control_dt, frame, True, target)
        for substep in range(steps_per_control):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            completed_physics_steps += 1
            time_s = completed_physics_steps * model.opt.timestep
            append_trace(time_s, frame, False, target)
            root_z = float(data.qpos[2])
            tilt = float(physics.root_tilt(data.qpos[3:7]))
            z_trigger = root_z < phase33.phase12.FALL_ROOT_Z_M
            tilt_trigger = tilt > phase33.phase12.FALL_TILT_RAD
            if z_trigger or tilt_trigger:
                terminal = {
                    "time_s": time_s,
                    "trigger": "both" if z_trigger and tilt_trigger else ("root_z" if z_trigger else "tilt"),
                    "root_z_m": root_z,
                    "root_tilt_rad": tilt,
                    "control_frame": frame,
                    "substep": substep + 1,
                    "q_error_rmse_max_rad": [
                        float(np.sqrt(np.mean((data.qpos[body_qpos] - target) ** 2))),
                        float(np.max(np.abs(data.qpos[body_qpos] - target))),
                    ],
                    "official_sole_collision_lr": np.asarray(trace["official_sole_collision_lr"][-1]).tolist(),
                }
                stop = True
                break
        realized_control.append(interval_collision.copy())
        interval_collision[:] = False
        if stop:
            break

    arrays = {key: np.asarray(value) for key, value in trace.items()}
    realized = np.asarray(realized_control, dtype=bool)
    intervals = len(realized)
    ref_contact = np.column_stack([
        np.asarray(reference["contact"]["left"], dtype=bool)[1:1 + intervals],
        np.asarray(reference["contact"]["right"], dtype=bool)[1:1 + intervals],
    ])
    agreement_lr = np.mean(realized == ref_contact, axis=0) if intervals else np.zeros(2)
    q_error = arrays["q_error_wbt29_rad"]
    overwrite = np.asarray(overwrite_delta) if overwrite_delta else np.zeros((0, 29))
    contact_impulse = arrays["contact_linear_impulse_lr_Ns"]
    torque_impulse = arrays["contact_torque_impulse_lr_Nms"]
    duration = (frames - 1) / float(reference["fps"])
    achieved = completed_physics_steps * model.opt.timestep
    full_duration = achieved >= 0.999 * duration
    contact_agreement = float(np.mean(agreement_lr))
    pd_baseline = json.loads(PHASE33_JSON.read_text())["paired_replay"]["phase30_candidate_free"]
    if full_duration and contact_agreement >= phase33.phase12.GATES["free_contact_agreement_min"]:
        attribution = "PD_OR_CLOSED_LOOP_PRIMARY"
    elif achieved <= max(1.0, 2.0 * float(pd_baseline["simulated_duration_s"])):
        attribution = "REFERENCE_GEOMETRY_CONTACT_PRIMARY"
    else:
        attribution = "MIXED_OR_UNRESOLVED"
    result = {
        "mode": "free_root_exact_joint_control_tick_oracle",
        "interpretation": "kinematic feasibility oracle; direct q/dq overwrite breaks joint momentum and is not an executable controller",
        "reference_duration_s": duration,
        "simulated_duration_s": achieved,
        "duration_fraction": float(achieved / duration),
        "terminal": terminal,
        "q_error_all_physics_frames_rmse_p95_max_rad": [
            float(np.sqrt(np.mean(q_error ** 2))),
            float(np.percentile(np.abs(q_error), 95)),
            float(np.max(np.abs(q_error))),
        ],
        "control_tick_overwrite_prejump_abs_p95_max_rad": [
            percentile(np.abs(overwrite), 95),
            float(np.max(np.abs(overwrite))) if overwrite.size else None,
        ],
        "root": {
            "z_min_m": float(np.min(arrays["root_position_m"][:, 2])),
            "tilt_p95_max_rad": [float(np.percentile(arrays["root_tilt_rad"], 95)), float(np.max(arrays["root_tilt_rad"]))],
            "final_position_m": arrays["root_position_m"][-1].tolist(),
            "final_linear_velocity_mps": arrays["root_linear_velocity_mps"][-1].tolist(),
            "final_angular_velocity_radps": arrays["root_angular_velocity_radps"][-1].tolist(),
        },
        "contact": {
            "reference_ratio_lr": np.mean(ref_contact, axis=0).tolist() if intervals else [0.0, 0.0],
            "realized_ratio_lr": np.mean(realized, axis=0).tolist() if intervals else [0.0, 0.0],
            "agreement_lr_mean": [float(agreement_lr[0]), float(agreement_lr[1]), contact_agreement],
            "realized_double_single_flight_ratio": [
                float(np.mean(realized[:, 0] & realized[:, 1])) if intervals else 0.0,
                float(np.mean(realized[:, 0] ^ realized[:, 1])) if intervals else 0.0,
                float(np.mean(~realized[:, 0] & ~realized[:, 1])) if intervals else 0.0,
            ],
            "linear_impulse_sum_abs_lr_Ns": np.sum(np.abs(contact_impulse), axis=(0, 2)).tolist(),
            "torque_impulse_sum_abs_lr_Nms": np.sum(np.abs(torque_impulse), axis=(0, 2)).tolist(),
            "truth_boundary": "official MuJoCo sole collision/wrench estimate; not hardware GRF/COP/wrench",
        },
        "generalized_force": {
            "actuator_wbt29_abs_max": float(np.max(np.abs(arrays["qfrc_actuator_wbt29"]))),
            "constraint_wbt29_abs_p95_max": [
                float(np.percentile(np.abs(arrays["qfrc_constraint_wbt29"]), 95)),
                float(np.max(np.abs(arrays["qfrc_constraint_wbt29"]))),
            ],
        },
        "pre_registered_attribution": {
            "oracle_full_and_contact_agreement_ge_0p55": "PD_OR_CLOSED_LOOP_PRIMARY",
            "oracle_quick_fall_le_max_1s_or_2x_pd": "REFERENCE_GEOMETRY_CONTACT_PRIMARY",
            "otherwise": "MIXED_OR_UNRESOLVED",
            "decision": attribution,
        },
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        **arrays,
        realized_contact_control_lr=realized,
        reference_contact_control_lr=ref_contact,
        body_joint_names=np.asarray([control.actuator_joint_names[i] for i in body_actuator_index]),
        head_joint_names=np.asarray([control.actuator_joint_names[i] for i in head_actuator_index]),
        head_nominal_rad=head_nominal,
    )
    return result, arrays


def render(report: dict[str, Any]) -> str:
    oracle = report["oracle"]
    baseline = report["paired_control_existing_phase33_free_pd"]
    terminal = oracle["terminal"]
    return f"""# X2 WBT Phase35：free-root exact-joint kinematic oracle

## 裁决

- **{report['decision']['status']}**
- exact-joint oracle：{oracle['simulated_duration_s']:.3f}/{oracle['reference_duration_s']:.3f}s；Phase33 free-PD：{baseline['simulated_duration_s']:.3f}/{baseline['reference_duration_s']:.3f}s。
- oracle attribution：`{oracle['pre_registered_attribution']['decision']}`。

## 假设 / 干预 / 对照

- 假设：若精确施加WBT29 q/dq后仍保持flight/快速倒，reference geometry/contact是主因；若稳定且接触显著改善，裸PD/闭环是主因。
- 干预：同一冻结Phase33 50Hz candidate与root初态；每20ms精确写WBT29 q/dq，head=model nominal，root/contact/gravity/solver自由；actuator torque恒零。
- 对照：只读取Phase33既有free-PD结果，不重跑。
- direct q/dq overwrite会破坏动量连续，故这是kinematic feasibility oracle，不是可执行controller或Gold晋升。

## 结果

- terminal：`{terminal}`。
- q error RMSE/p95/max：`{oracle['q_error_all_physics_frames_rmse_p95_max_rad']}` rad；overwrite前jump p95/max：`{oracle['control_tick_overwrite_prejump_abs_p95_max_rad']}` rad。
- root z min：{oracle['root']['z_min_m']:.4f}m；tilt p95/max：`{oracle['root']['tilt_p95_max_rad']}` rad。
- reference contact L/R：`{oracle['contact']['reference_ratio_lr']}`；realized：`{oracle['contact']['realized_ratio_lr']}`；agreement L/R/mean：`{oracle['contact']['agreement_lr_mean']}`。
- realized DS/SS/flight：`{oracle['contact']['realized_double_single_flight_ratio']}`。
- contact linear impulse L/R：`{oracle['contact']['linear_impulse_sum_abs_lr_Ns']}` Ns；torque impulse：`{oracle['contact']['torque_impulse_sum_abs_lr_Nms']}` Nms。
- actuator generalized force max：{oracle['generalized_force']['actuator_wbt29_abs_max']:.6g}；constraint p95/max：`{oracle['generalized_force']['constraint_wbt29_abs_p95_max']}`。

## 结论

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    phase33_report = json.loads(PHASE33_JSON.read_text())
    entry = joblib.load(phase33.PHASE30_CACHE)[phase33.MOTION_ID]
    reference = phase33.reference_from_entry(entry, "phase35_frozen_candidate")
    preflight = {
        "schema_version": "x2_wbt_exact_joint_oracle_phase35_preflight_v1",
        "motion_id": phase33.MOTION_ID,
        "reference_cache_sha256": phase33.phase12.sha256(phase33.PHASE30_CACHE),
        "phase33_report_sha256": phase33.phase12.sha256(PHASE33_JSON),
        "official_scene_sha256": phase33.phase12.sha256(physics.DEFAULT_SCENE),
        "official_control_sha256": phase33.phase12.sha256(physics.DEFAULT_CONTROL),
        "phase33_scene_control_hash_match": (
            phase33_report["preflight"]["official_scene"]["sha256"] == phase33.phase12.sha256(physics.DEFAULT_SCENE)
            and phase33_report["preflight"]["official_control"]["sha256"] == phase33.phase12.sha256(physics.DEFAULT_CONTROL)
        ),
        "reference_frames_fps": [len(reference["q"]), float(reference["fps"])],
        "head_reference_locked_zero": bool(np.max(np.abs(reference["q"][:, [reference["joint_names"].index("head_yaw_joint"), reference["joint_names"].index("head_pitch_joint")]])) <= 1e-12),
        "existing_free_pd_not_repeated": True,
        "new_physics_mode_count": 1,
        "training_ppo_optimizer_cem_pd_scan_warmup_reference_change": False,
        "attribution_thresholds_frozen_before_run": {
            "PD_OR_CLOSED_LOOP_PRIMARY": "full duration and contact agreement >=0.55",
            "REFERENCE_GEOMETRY_CONTACT_PRIMARY": "fall <= max(1.0s, 2x existing PD survival)",
            "MIXED_OR_UNRESOLVED": "otherwise",
        },
    }
    preflight["pass"] = bool(
        preflight["phase33_scene_control_hash_match"]
        and preflight["reference_frames_fps"] == [291, 50.0]
        and preflight["head_reference_locked_zero"]
        and phase33_report["gate"]["candidate_prescribed"]["pass"]
        and not phase33_report["gate"]["candidate_free"]["pass"]
    )
    OUTPUT_PREFLIGHT.write_text(json.dumps(json_safe(preflight), indent=2, ensure_ascii=False) + "\n")
    print(f"[phase35] preflight pass={preflight['pass']}", flush=True)
    if args.preflight_only or not preflight["pass"]:
        return
    oracle, _ = run_oracle(reference, OUTPUT_TRACE)
    baseline = phase33_report["paired_replay"]["phase30_candidate_free"]
    decision = oracle["pre_registered_attribution"]["decision"]
    if decision == "REFERENCE_GEOMETRY_CONTACT_PRIMARY":
        conclusion = "即使消除29DOF关节跟踪误差，free root仍快速失败；当前主因收敛到reference geometry/contact feasibility，而不是裸PD跟踪误差。"
        next_step = "停止oracle，不调PD/加warmup；回到contact-feasible reference生成合同。"
    elif decision == "PD_OR_CLOSED_LOOP_PRIMARY":
        conclusion = "精确关节oracle稳定且接触达门，说明reference kinematic support可行，Phase33失败主要来自裸PD/缺闭环。"
        next_step = "停止oracle；由主线决定闭环policy sanity，不把kinematic overwrite作为控制器。"
    else:
        conclusion = "exact-joint显著改变存活但未完整稳定/接触过门；reference与裸PD闭环均有贡献，不能单因归责。"
        next_step = "停止；只依据保存trace设计下一个单变量，不扫描。"
    report = {
        "schema_version": "x2_wbt_exact_joint_oracle_phase35_v1",
        "preflight": preflight,
        "truth_boundary": {
            "kinematic_oracle_not_executable_controller": True,
            "direct_qdq_overwrite_breaks_momentum": True,
            "contact_is_official_sim_model_not_hardware_truth": True,
            "phase33_pd_baseline_not_repeated": True,
            "root_reference_pd_warmup_contact_labels_unchanged": True,
            "training_ppo_optimizer_cem_real_robot_base_port51822": False,
        },
        "paired_control_existing_phase33_free_pd": {
            "simulated_duration_s": baseline["simulated_duration_s"],
            "reference_duration_s": baseline["reference_duration_s"],
            "contact": baseline["contact"],
            "root_tracking": baseline["root_tracking"],
            "q_tracking": baseline["q_tracking"],
            "torque": baseline["torque"],
        },
        "oracle": oracle,
        "trace": {"path": str(OUTPUT_TRACE), "sha256": phase33.phase12.sha256(OUTPUT_TRACE)},
        "decision": {"status": f"PHASE35_{decision}", "conclusion": conclusion, "next_step": next_step},
    }
    OUTPUT_JSON.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
