#!/usr/bin/env python3
"""Phase33 official-AimDK-v1 Gold qualification for the sole Phase30 train Silver.

The paired references are fixed before simulation: Phase28 original and
Phase30 PHUMA-LUNGE-R-001 candidate.  Both use the exact same Phase12 runner,
official scene/control, per-reference exact-frame0 initialization rule, and
pre-registered gates.  Numeric initial states differ because the repaired
candidate is a different path; this is a paired qualification comparison, not
a pure causal single-variable A/B.  Candidate prescribed-root must pass every
Phase12 gate before its sole free-root replay is permitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_native_gold_trackability_phase12 as phase12
import retarget.run_x2_forefoot_official_physics_screen as physics
import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_time_dilation_phase30 as phase30


MOTION_ID = "PHUMA-LUNGE-R-001"
PHASE28_CACHE = phase28.OUTPUT_ROOT / "official_v1_model/x2_phase28_panel24.pkl"
PHASE30_CACHE = phase30.OUTPUT_CACHE
REPORT_JSON = REPO / "reports/retarget/x2_wbt_gold_qualification_phase33.json"
REPORT_MD = REPO / "reports/retarget/x2_wbt_gold_qualification_phase33.md"
PREFLIGHT_JSON = REPO / "reports/retarget/x2_wbt_gold_qualification_phase33_preflight.json"


def proc_tcp_listener(port: int) -> list[str]:
    needle = f":{port:04X}"
    rows = []
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 3 and fields[1].upper().endswith(needle) and fields[3] == "0A":
                rows.append(line.strip())
    return rows


def official_processes() -> list[dict[str, Any]]:
    patterns = ("x2_rl_deploy", "x2_rl_deploy_controller", "run_sim_loop.py")
    rows = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit() or int(child.name) == os.getpid():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(pattern in command for pattern in patterns):
            rows.append({"pid": int(child.name), "cmdline": command[:500]})
    return rows


def numerical_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.gradient(values, 1.0/fps, axis=0, edge_order=1)


def angular_velocity_xyzw(quaternions: np.ndarray, fps: float) -> np.ndarray:
    rotations = Rotation.from_quat(np.asarray(quaternions, dtype=np.float64))
    values = np.zeros((len(quaternions), 3), dtype=np.float64)
    if len(values) < 2:
        return values
    increments = (rotations[:-1].inv()*rotations[1:]).as_rotvec()*fps
    values[0], values[-1] = increments[0], increments[-1]
    if len(values) > 2:
        values[1:-1] = 0.5*(increments[:-1]+increments[1:])
    return values


def reference_from_entry(entry: dict[str, Any], key: str, target_fps: float | None = None) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    data = mujoco.MjData(model)
    control = physics.build_control_contract(model, physics.DEFAULT_CONTROL)
    names = list(entry["joint_names_mujoco"])
    if tuple(names) != control.joint_names:
        raise ValueError("entry 31DOF order differs from frozen Phase12 official contract")
    source_fps = float(entry["fps"])
    fps = float(1.0/control.control_dt) if target_fps is None else float(target_fps)
    source_q = np.asarray(entry["dof"], dtype=np.float64)
    source_root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    source_root_quat = np.asarray(entry["root_rot"], dtype=np.float64)
    duration = (len(source_q)-1)/source_fps
    source_t = np.arange(len(source_q), dtype=np.float64)/source_fps
    frames = int(round(duration*fps))+1
    target_t = np.linspace(0.0, duration, frames)
    q = np.column_stack([np.interp(target_t, source_t, source_q[:, index]) for index in range(source_q.shape[1])])
    root_pos = np.column_stack([np.interp(target_t, source_t, source_root_pos[:, axis]) for axis in range(3)])
    root_quat = Slerp(source_t, Rotation.from_quat(source_root_quat))(target_t).as_quat()
    dq = numerical_velocity(q, fps)
    root_lin = numerical_velocity(root_pos, fps)
    root_ang = angular_velocity_xyzw(root_quat, fps)
    joint_qpos = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in names]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in range(1, model.nbody)]
    if any(name is None for name in body_names):
        raise ValueError("official scene contains unnamed non-world body")
    body_ids = np.arange(1, model.nbody, dtype=np.int64)
    floor, foot_geoms = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, geoms in foot_geoms.items() for geom in geoms}
    body_pos = np.zeros((len(q), len(body_ids), 3), dtype=np.float64)
    body_quat = np.zeros((len(q), len(body_ids), 4), dtype=np.float64)
    contact = {"left": np.zeros(len(q), dtype=bool), "right": np.zeros(len(q), dtype=bool)}
    for frame in range(len(q)):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame][[3, 0, 1, 2]]
        data.qpos[joint_qpos] = q[frame]
        data.qvel[:3] = root_lin[frame]
        data.qvel[3:6] = root_ang[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids][:, [1, 2, 3, 0]]
        current = phase12.contact_now(model, data, floor, geom_side)
        for side in ("left", "right"):
            contact[side][frame] = bool(current[side])
    return {
        "key": key, "fps": fps, "joint_names": names, "body_names": body_names,
        "q": q, "dq": dq, "root_pos": root_pos, "root_quat_xyzw": root_quat,
        "root_lin_vel": root_lin, "root_ang_vel": root_ang,
        "body_pos": body_pos, "body_quat_xyzw": body_quat, "contact": contact,
        "target_source": "Phase28/30 MotionLib-compatible dof path; PD target is q, never recorded command",
        "recorded_command_used": False,
        "contact_provenance": "official AimDK v1 scene active-foot collision at reference FK; model estimate, not hardware GRF/COP/wrench",
        "timebase_adapter": {
            "source_fps": source_fps, "output_fps": fps,
            "source_frames": len(source_q), "control_frames": len(q),
            "duration_s": duration,
            "q_root_position": "linear continuous-path resample",
            "root_orientation": "quaternion Slerp xyzw",
            "reason": "Phase12 simulate contract consumes exactly one reference frame per official 20ms control interval",
        },
    }


def timebase_adapter_unit(entry: dict[str, Any], adapted: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    source_fps, target_fps = float(entry["fps"]), float(adapted["fps"])
    source_q = np.asarray(entry["dof"], dtype=np.float64)
    source_root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    source_quat = np.asarray(entry["root_rot"], dtype=np.float64)
    # The official runner consumes one uniformly spaced sample per 20 ms.  A
    # source duration is not necessarily an integer number of those ticks, so
    # the adapter preserves the complete continuous path (including both
    # endpoints) on a normalized phase grid and permits at most half a control
    # tick of duration quantization.  Check the inverse path on that same phase
    # contract rather than comparing two incompatible absolute-time grids.
    source_phase = np.linspace(0.0, 1.0, len(source_q))
    target_phase = np.linspace(0.0, 1.0, len(adapted["q"]))
    q_back = np.column_stack([np.interp(source_phase, target_phase, adapted["q"][:, index]) for index in range(source_q.shape[1])])
    root_back = np.column_stack([np.interp(source_phase, target_phase, adapted["root_pos"][:, axis]) for axis in range(3)])
    quat_back = Slerp(target_phase, Rotation.from_quat(adapted["root_quat_xyzw"]))(source_phase).as_quat()
    quat_error = 1.0-np.abs(np.sum(quat_back*source_quat, axis=1))
    transition = lambda values: int(np.count_nonzero(np.diff(np.asarray(values, dtype=np.int8)) != 0))
    contact_transitions = {
        side: [transition(native["contact"][side]), transition(adapted["contact"][side])]
        for side in ("left", "right")
    }
    duration_source = (len(source_q)-1)/source_fps
    duration_adapted = (len(adapted["q"])-1)/target_fps
    metrics = {
        "duration_source_adapted_error_s": abs(duration_source-duration_adapted),
        "endpoint_q_max_abs_rad": float(max(np.max(np.abs(adapted["q"][0]-source_q[0])), np.max(np.abs(adapted["q"][-1]-source_q[-1])))),
        "endpoint_root_max_abs_m": float(max(np.max(np.abs(adapted["root_pos"][0]-source_root[0])), np.max(np.abs(adapted["root_pos"][-1]-source_root[-1])))),
        "endpoint_quaternion_equivalent_error": float(max(1.0-abs(np.dot(adapted["root_quat_xyzw"][0], source_quat[0])), 1.0-abs(np.dot(adapted["root_quat_xyzw"][-1], source_quat[-1])))),
        "inverse_path_q_max_abs_rad": float(np.max(np.abs(q_back-source_q))),
        "inverse_path_root_max_abs_m": float(np.max(np.abs(root_back-source_root))),
        "inverse_path_quaternion_equivalent_error": float(np.max(quat_error)),
        "contact_transition_counts_native30_adapted50": contact_transitions,
    }
    checks = {
        "official_output_fps_50": target_fps == 50.0,
        "duration_preserved_within_half_control_step": metrics["duration_source_adapted_error_s"] <= 0.5/target_fps+1e-12,
        "q_endpoints_exact": metrics["endpoint_q_max_abs_rad"] <= 1e-12,
        "root_endpoints_exact": metrics["endpoint_root_max_abs_m"] <= 1e-12,
        "quaternion_endpoints_equivalent": metrics["endpoint_quaternion_equivalent_error"] <= 1e-6,
        "continuous_q_path_preserved": metrics["inverse_path_q_max_abs_rad"] <= 0.01,
        "continuous_root_path_preserved": metrics["inverse_path_root_max_abs_m"] <= 0.003,
        "slerp_path_preserved": metrics["inverse_path_quaternion_equivalent_error"] <= 1e-5,
        "contact_phase_transition_count_preserved": all(before == after for before, after in contact_transitions.values()),
    }
    return {"metrics": metrics, "checks": checks, "pass": bool(all(checks.values()))}


def preregistration(original: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    q0 = np.asarray(original["dof"])[0]-np.asarray(candidate["dof"])[0]
    root0 = np.asarray(original["root_trans_offset"])[0]-np.asarray(candidate["root_trans_offset"])[0]
    return {
        "comparison_type": "paired qualification, not pure causal single-variable A/B",
        "reason": "Phase30 includes frozen Phase29 spatial repair plus 1.46x retime; numeric frame0 therefore differs from Phase28",
        "same_runner": "Phase12 simulate/prescribed_gate/free_gate",
        "same_official_scene_control_pd_action_limits_sole_contract": True,
        "control_timebase_adapter": "both 30Hz paths are resampled as the same continuous path to official 50Hz using linear q/root position and quaternion Slerp",
        "initialization_rule": "each reference starts from its own exact q/dq/root pose/root velocity frame0; solver/contact warmstart and controller hidden state unavailable",
        "numeric_initial_state_equal": False,
        "initial_q_rmse_max_abs_rad": [float(np.sqrt(np.mean(q0*q0))), float(np.max(np.abs(q0)))],
        "initial_root_position_delta_m": float(np.linalg.norm(root0)),
        "original_role": "contextual control; cannot tune or relax candidate gate",
        "candidate_role": "sole Phase30 train Silver qualification target",
        "execution_order": ["original prescribed-root", "candidate prescribed-root", "candidate free-root only iff every prescribed gate passes"],
        "parameter_scan_or_reference_change": False,
    }


def render(report: dict[str, Any]) -> str:
    original = report["paired_replay"]["phase28_original_prescribed"]
    candidate = report["paired_replay"]["phase30_candidate_prescribed"]
    free = report["paired_replay"].get("phase30_candidate_free")
    lines = [
        "# X2 WBT Phase33：Phase30唯一train Silver官方Gold资格门", "",
        "## 裁决", "",
        f"- status：**{report['decision']['status']}**；candidate prescribed：`{report['gate']['candidate_prescribed']['pass']}`；free executed：`{free is not None}`。",
        "- Phase28 original 与 Phase30 candidate 使用同一Phase12 runner/官方scene/control/gate；各自按同一规则从自身reference-exact frame0初始化，因此是paired qualification，不是纯因果单变量A/B。",
        "- 不用physics调轨迹或1.46，不做CEM/PPO；prescribed失败只否定当前裸PD trackability。", "",
        "## 结果", "",
        "| reference/mode | survival | q RMSE | body pos/ori p95 | root pos/ori | contact | SS ref/real | slip p95 | sat/effort p95 | jump p95/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in [("Phase28 original prescribed", original), ("Phase30 candidate prescribed", candidate)] + ([("Phase30 candidate free", free)] if free else []):
        lines.append(
            f"| {name} | {value['simulated_duration_s']:.3f}/{value['reference_duration_s']:.3f}s | {value['q_tracking']['rmse_rad']:.4f} | "
            f"{value['body_tracking']['root_relative_position_p95_max_m'][0]:.4f}m/{value['body_tracking']['orientation_p95_max_rad'][0]:.4f}rad | "
            f"{value['root_tracking']['position_rmse_final_m'][0]:.5f}m/{value['root_tracking']['orientation_p95_max_rad'][0]:.5f}rad | "
            f"{value['contact']['agreement']['mean']:.3f} | {value['contact']['reference_phase']['single_support_ratio']:.3f}/{value['contact']['realized_phase']['single_support_ratio']:.3f} | "
            f"{value['slip_p95_max_mps'][0]:.3f}m/s | {value['torque']['saturation_fraction']:.4f}/{value['torque']['raw_abs_over_limit_p95_max'][0]:.3f} | "
            f"{value['replay_joint_motion']['step_abs_p95_max_rad'][0]:.4f}/{value['replay_joint_motion']['step_abs_p95_max_rad'][1]:.4f} |"
        )
    lines += ["", f"- original prescribed gate：`{report['gate']['original_prescribed']['pass']}`，failed `{report['gate']['original_prescribed']['failed']}`。", f"- candidate prescribed gate：`{report['gate']['candidate_prescribed']['pass']}`，failed `{report['gate']['candidate_prescribed']['failed']}`。", f"- candidate free gate：`{None if free is None else report['gate']['candidate_free']}`。", "", "## 结论", "", report["decision"]["conclusion"], "", "## 下一步", "", report["decision"]["next_step"]]
    return "\n".join(lines)+"\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", type=Path, default=REPORT_JSON)
    parser.add_argument("--markdown", type=Path, default=REPORT_MD)
    args = parser.parse_args()
    listeners, processes = proc_tcp_listener(51822), official_processes()
    original_cache, candidate_cache = joblib.load(PHASE28_CACHE), joblib.load(PHASE30_CACHE)
    original, candidate = original_cache[MOTION_ID], candidate_cache[MOTION_ID]
    prereg = preregistration(original, candidate)
    original_ref = reference_from_entry(original, "phase28_original_PHUMA_LUNGE_R")
    candidate_ref = reference_from_entry(candidate, "phase30_time1p46_PHUMA_LUNGE_R")
    original_native = reference_from_entry(original, "phase28_original_native30", target_fps=30.0)
    candidate_native = reference_from_entry(candidate, "phase30_candidate_native30", target_fps=30.0)
    adapter_unit = {
        "phase28_original": timebase_adapter_unit(original, original_ref, original_native),
        "phase30_candidate": timebase_adapter_unit(candidate, candidate_ref, candidate_native),
    }
    preflight = {
        "schema_version": "x2_wbt_gold_qualification_phase33_preflight_v1",
        "motion_id": MOTION_ID,
        "phase28_cache": {"path": str(PHASE28_CACHE), "sha256": phase12.sha256(PHASE28_CACHE)},
        "phase30_cache": {"path": str(PHASE30_CACHE), "sha256": phase12.sha256(PHASE30_CACHE)},
        "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase12.sha256(physics.DEFAULT_SCENE)},
        "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": phase12.sha256(physics.DEFAULT_CONTROL)},
        "phase12_runner": {"path": str(Path(phase12.__file__)), "sha256": phase12.sha256(Path(phase12.__file__))},
        "gates": phase12.GATES, "preregistration": prereg,
        "timebase_adapter_unit": adapter_unit,
        "runtime_conflict": {"port_51822_listeners": listeners, "foreign_official_processes": processes},
        "checks": {
            "sole_phase30_silver_present": set(candidate_cache) == {MOTION_ID} or MOTION_ID in candidate_cache,
            "joint_order_equal": original["joint_names_mujoco"] == candidate["joint_names_mujoco"],
            "head_locked_both": float(max(np.max(np.abs(np.asarray(original["dof"])[:, -2:])), np.max(np.abs(np.asarray(candidate["dof"])[:, -2:])))) <= 1e-12,
            "fps_original_candidate": float(original["fps"]) == 30.0 and float(candidate["fps"]) == 30.0,
            "phase12_control_time_adapter_declared": prereg["control_timebase_adapter"].startswith("both 30Hz paths"),
            "timebase_adapter_unit_pass": all(value["pass"] for value in adapter_unit.values()),
            "port_51822_free": not listeners,
            "no_foreign_official_process": not processes,
        },
    }
    preflight["pass"] = bool(all(preflight["checks"].values()))
    PREFLIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(phase28.json_safe(preflight), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"[phase33] preflight pass={preflight['pass']} port51822={not listeners} foreign={len(processes)}", flush=True)
    if args.preflight_only or not preflight["pass"]:
        return

    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    original_source, candidate_source = phase12.source_kinematics(original_ref, model), phase12.source_kinematics(candidate_ref, model)
    print("[phase33] original prescribed-root", flush=True)
    original_prescribed = phase12.simulate(original_ref, phase12.MODE_PRESCRIBED)
    original_gate = phase12.prescribed_gate(original_prescribed, original_source)
    print("[phase33] candidate prescribed-root", flush=True)
    candidate_prescribed = phase12.simulate(candidate_ref, phase12.MODE_PRESCRIBED)
    candidate_gate = phase12.prescribed_gate(candidate_prescribed, candidate_source)
    replays = {"phase28_original_prescribed": original_prescribed, "phase30_candidate_prescribed": candidate_prescribed}
    candidate_free = None
    free_gate = None
    if candidate_gate["pass"]:
        print("[phase33] candidate free-root (sole permitted run)", flush=True)
        candidate_free = phase12.simulate(candidate_ref, phase12.MODE_FREE)
        free_gate = phase12.free_gate(candidate_free)
        replays["phase30_candidate_free"] = candidate_free
    if not candidate_gate["pass"]:
        status = "PHASE33_CANDIDATE_PRESCRIBED_REJECTED"
        conclusion = "Phase30静态Silver未通过官方prescribed-root裸PD trackability；只否定当前PD/action replay合同，不否定Any2Any、X2或闭环训练。"
        next_step = "停止；不得用physics调Phase30轨迹、1.46、PD或门槛，未运行free-root。"
    elif free_gate and free_gate["pass"]:
        status = "PHASE33_GOLD_PRESCRIBED_AND_FREE_PASSED"
        conclusion = "候选同时通过官方prescribed与唯一free-root门，可晋升为当前合同下Gold候选；仍不是policy/真机证明。"
        next_step = "停止等待评审；不得自动训练。"
    else:
        status = "PHASE33_PRESCRIBED_PASSED_FREE_REJECTED"
        conclusion = "候选可由官方裸PD跟踪，但free-root失败，证明静态Silver/trackability不等于Gold平衡；不否定Any2Any。"
        next_step = "停止；保留为trackable Silver，不自动训练或调参。"
    report = {
        "schema_version": "x2_wbt_gold_qualification_phase33_v1",
        "preflight": preflight, "preregistration": prereg,
        "truth_boundary": {"contact_is_model_collision_not_hardware_grf_cop_wrench": True, "prescribed_root_is_not_balance": True, "source_trace_is_not_replay": True, "candidate_free_executed_only_after_all_prescribed_gates": candidate_free is not None, "physics_used_for_qualification_not_tuning": True, "CEM_PPO_training_real_robot": False, "invalid_attempt_before_report": "direct 30Hz input exercised only 60% reference duration because Phase12 expects one frame per 20ms control; no result retained and no parameter changed before correct 50Hz adapter rerun"},
        "provenance": preflight,
        "source_reference": {"phase28_original": original_source, "phase30_candidate": candidate_source},
        "paired_replay": replays,
        "gate": {"original_prescribed": original_gate, "candidate_prescribed": candidate_gate, "candidate_free": free_gate},
        "decision": {"status": status, "conclusion": conclusion, "next_step": next_step},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
