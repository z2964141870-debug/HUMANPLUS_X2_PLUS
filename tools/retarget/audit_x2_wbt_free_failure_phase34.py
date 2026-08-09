#!/usr/bin/env python3
"""Phase34 read-only attribution of the Phase33 free-root failure.

This tool never calls ``mj_step`` and never runs an optimizer.  It reads the
frozen Phase30 source and Phase33 aggregate replay evidence, reconstructing
only the already-declared 50 Hz reference with static ``mj_forward`` FK/contact
queries.  Missing per-step replay evidence is reported as missing rather than
regenerated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_gold_qualification_phase33 as phase33
import retarget.run_x2_forefoot_official_physics_screen as physics


PHASE33_JSON = REPO / "reports/retarget/x2_wbt_gold_qualification_phase33.json"
OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_free_failure_attribution_phase34.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_free_failure_attribution_phase34.md"


def contiguous_true_windows(values: np.ndarray, fps: float) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    return [
        {
            "start_frame": int(start),
            "end_frame_inclusive": int(end),
            "start_s": float(start / fps),
            "end_s_inclusive": float(end / fps),
        }
        for start, end in zip(starts, ends)
    ]


def source_audit(reference: dict[str, Any], failure_s: float) -> dict[str, Any]:
    fps = float(reference["fps"])
    # Phase33 stopped at 0.575 s, between 50 Hz samples.  Include every source
    # frame at or before the last completed 0.56 s control sample.
    count = min(len(reference["q"]), int(np.floor(failure_s * fps)) + 1)
    q = np.asarray(reference["q"], dtype=np.float64)
    dq = np.asarray(reference["dq"], dtype=np.float64)
    root = np.asarray(reference["root_pos"], dtype=np.float64)
    quat = np.asarray(reference["root_quat_xyzw"], dtype=np.float64)
    q_step = np.abs(np.diff(q[:count], axis=0))
    root_step = np.linalg.norm(np.diff(root[:count], axis=0), axis=1)
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_quat(quat[:count])
    root_orientation_step = np.linalg.norm(
        (rotations[:-1].inv() * rotations[1:]).as_rotvec(), axis=1
    )
    contacts = {side: np.asarray(reference["contact"][side], dtype=bool) for side in ("left", "right")}
    left, right = contacts["left"], contacts["right"]
    return {
        "reference_frames_fps_duration_s": [len(q), fps, float((len(q) - 1) / fps)],
        "frame0": {
            "q_rad": q[0].tolist(),
            "dq_radps": dq[0].tolist(),
            "dq_l2_maxabs_p95_radps": [
                float(np.linalg.norm(dq[0])),
                float(np.max(np.abs(dq[0]))),
                float(np.percentile(np.abs(dq[0]), 95)),
            ],
            "root_position_m": root[0].tolist(),
            "root_quaternion_xyzw": quat[0].tolist(),
            "root_linear_velocity_mps": np.asarray(reference["root_lin_vel"])[0].tolist(),
            "root_linear_speed_mps": float(np.linalg.norm(reference["root_lin_vel"][0])),
            "root_angular_velocity_radps": np.asarray(reference["root_ang_vel"])[0].tolist(),
            "root_angular_speed_radps": float(np.linalg.norm(reference["root_ang_vel"][0])),
            "root_tilt_rad": float(physics.root_tilt(quat[0][[3, 0, 1, 2]])),
            "model_active_sole_contact": {side: bool(values[0]) for side, values in contacts.items()},
        },
        "full_contact_schedule": {
            "left_ratio": float(np.mean(left)),
            "right_ratio": float(np.mean(right)),
            "double_support_ratio": float(np.mean(left & right)),
            "single_support_ratio": float(np.mean(left ^ right)),
            "flight_ratio": float(np.mean(~left & ~right)),
            "left_windows": contiguous_true_windows(left, fps),
            "right_windows": contiguous_true_windows(right, fps),
            "boundary": "static official-model active-sole collision intent; not hardware GRF/COP/wrench",
        },
        "source_first_survival_window_0_to_last_completed_control_sample": {
            "end_s": float((count - 1) / fps),
            "frames": count,
            "q_step_p95_max_rad": [float(np.percentile(q_step, 95)), float(np.max(q_step))],
            "dq_abs_p95_max_radps": [float(np.percentile(np.abs(dq[:count]), 95)), float(np.max(np.abs(dq[:count])))],
            "root_position_step_p95_max_m": [float(np.percentile(root_step, 95)), float(np.max(root_step))],
            "root_orientation_step_p95_max_rad": [
                float(np.percentile(root_orientation_step, 95)),
                float(np.max(root_orientation_step)),
            ],
            "root_z_min_max_m": [float(np.min(root[:count, 2])), float(np.max(root[:count, 2]))],
            "contact": {
                side: {
                    "ratio": float(np.mean(values[:count])),
                    "transition_count": int(np.count_nonzero(np.diff(values[:count].astype(np.int8)))),
                }
                for side, values in contacts.items()
            },
        },
    }


def selected_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "simulated_reference_duration_s": [result["simulated_duration_s"], result["reference_duration_s"]],
        "fell_fall_time_nonfinite": [result["fell"], result["fall_time_s"], result["nonfinite_time_s"]],
        "q_tracking": result["q_tracking"],
        "body_tracking": result["body_tracking"],
        "root_tracking": result["root_tracking"],
        "contact": result["contact"],
        "slip_p95_max_mps": result["slip_p95_max_mps"],
        "torque": result["torque"],
        "replay_joint_motion": result["replay_joint_motion"],
        "initialization": result["initialization"],
    }


def render(report: dict[str, Any]) -> str:
    source = report["source_reference"]
    frame0 = source["frame0"]
    free = report["existing_replay_evidence"]["free_root_0_to_0p575_aggregate"]
    prescribed = report["existing_replay_evidence"]["prescribed_full_5p8s_aggregate_not_same_window"]
    contact = source["full_contact_schedule"]
    early = source["source_first_survival_window_0_to_last_completed_control_sample"]
    lines = [
        "# X2 WBT Phase34：Phase33 free-root失败只读归因",
        "",
        "## 裁决",
        "",
        "- **最强证据指向reference接触可行性缺口，裸PD缺闭环是并行成立但尚不能定主次；没有证据证明数值reset错位是主因。**",
        "- 本阶段未运行physics/optimizer；只读取Phase33 aggregate，并对冻结source做静态FK/连续性审计。",
        "- Phase33没有保存逐帧trace，故不能伪造0–0.575s曲线或prescribed同窗指标。",
        "",
        "## reset mismatch",
        "",
        f"- runner代码在两栏都显式把frame0 q/dq/root pose/root velocity写入sim；frame0 root速度为 {frame0['root_linear_speed_mps']:.4f}m/s、角速度 {frame0['root_angular_speed_radps']:.4f}rad/s，dq max {frame0['dq_l2_maxabs_p95_radps'][1]:.4f}rad/s。它不是静止reset，但不是数值漏写。",
        f"- frame0静态active-sole contact为 L={frame0['model_active_sole_contact']['left']} / R={frame0['model_active_sole_contact']['right']}；reference与sim初始化使用同一scene/q/root，因此未发现几何frame0错位。",
        "- solver/contact warmstart与原controller hidden state确实缺失，所以reset动态上下文仍是可疑项；现有aggregate不能量化它的影响。",
        "",
        "## reference contact / pose feasibility",
        "",
        f"- 全5.8s reference：flight={contact['flight_ratio']:.3f}、SS={contact['single_support_ratio']:.3f}、DS={contact['double_support_ratio']:.3f}；右脚active sole从未接触，左脚仅 {contact['left_ratio']:.3f}。这是动态lunge的严重接触可行性警报。",
        f"- 0–{early['end_s']:.2f}s source连续：q-step p95/max={early['q_step_p95_max_rad'][0]:.4f}/{early['q_step_p95_max_rad'][1]:.4f}rad，root-step max={early['root_position_step_p95_max_m'][1]:.4f}m，未见开局离散跳变。",
        f"- 同一free存活窗内，reference contact L/R={free['contact']['reference_phase']['left_ratio']:.3f}/{free['contact']['reference_phase']['right_ratio']:.3f}，realized L/R={free['contact']['realized_phase']['left_ratio']:.3f}/{free['contact']['realized_phase']['right_ratio']:.3f}，agreement={free['contact']['agreement']['mean']:.3f}。仿真迅速落到近双支撑，而reference几乎全flight。",
        "- contact均为模型碰撞标签，不是实机GRF/COP/wrench；但已足以否定‘reference提供了清晰可执行支撑时序’。",
        "",
        "## 裸PD闭环能力",
        "",
        f"- prescribed完整跟踪通过：q RMSE={prescribed['q_tracking']['rmse_rad']:.4f}rad、body pos/ori p95={prescribed['body_tracking']['root_relative_position_p95_max_m'][0]:.4f}m/{prescribed['body_tracking']['orientation_p95_max_rad'][0]:.4f}rad。",
        f"- free于 {free['fell_fall_time_nonfinite'][1]:.3f}s触发fall；最后已保存控制帧的root pos RMSE/final={free['root_tracking']['position_rmse_final_m'][0]:.4f}/{free['root_tracking']['position_rmse_final_m'][1]:.4f}m、orientation-error p95/max={free['root_tracking']['orientation_p95_max_rad'][0]:.4f}/{free['root_tracking']['orientation_p95_max_rad'][1]:.4f}rad、sampled tilt max={free['root_tracking']['tilt_max_rad']:.4f}rad、z min={free['root_tracking']['z_min_m']:.4f}m，torque sat={free['torque']['saturation_fraction']:.4f}。",
        "- 因termination发生在控制采样之间且terminal state未保存，现有证据不能区分最终是root-z还是tilt先越门；也没有prescribed 0–0.575s同窗trace。",
        "- prescribed靠外力逐物理步固定root，free只剩无状态位置PD；因此它证明PD可跟关节，却不证明能生成载荷转移/平衡闭环。",
        "",
        "## 三类归因",
        "",
        "| 假设 | 当前证据 | 裁决 |",
        "|---|---|---|",
        "| 数值reset mismatch | q/dq/root pose/velocity显式一致；frame0几何contact同合同；但无solver/contact warmstart和controller hidden state | **未证实为主因，保留次要不确定性** |",
        "| reference接触可行性 | 97.25% flight、右脚0接触；free迅速近双支撑，agreement 0.071 | **强支持存在系统性缺口** |",
        "| 裸PD缺闭环 | prescribed全程通过而free 0.575s失败；无策略/历史/载荷反馈 | **强支持，但与reference缺口尚纠缠** |",
        "",
        "## 唯一下一实验",
        "",
        report["single_next_falsifiable_experiment"],
        "",
        "该oracle只用于归因，不是可部署控制器、不是Gold晋升、不是训练。不得同时调PD、加warmup或修reference。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    phase33_report = json.loads(PHASE33_JSON.read_text(encoding="utf-8"))
    entry = joblib.load(phase33.PHASE30_CACHE)[phase33.MOTION_ID]
    reference = phase33.reference_from_entry(entry, "phase34_static_audit")
    free = phase33_report["paired_replay"]["phase30_candidate_free"]
    prescribed = phase33_report["paired_replay"]["phase30_candidate_prescribed"]
    source = source_audit(reference, float(free["fall_time_s"]))
    report = {
        "schema_version": "x2_wbt_free_failure_attribution_phase34_v1",
        "scope": {
            "new_physics_steps": 0,
            "optimizer_or_training": False,
            "source_mutation": False,
            "static_fk_forward_only": True,
            "phase33_per_frame_replay_trace_available": False,
        },
        "provenance": {
            "phase33_report": {"path": str(PHASE33_JSON), "sha256": phase33.phase12.sha256(PHASE33_JSON)},
            "phase30_cache": {"path": str(phase33.PHASE30_CACHE), "sha256": phase33.phase12.sha256(phase33.PHASE30_CACHE)},
            "phase12_runner": {"path": str(Path(phase33.phase12.__file__)), "sha256": phase33.phase12.sha256(Path(phase33.phase12.__file__))},
        },
        "source_reference": source,
        "existing_replay_evidence": {
            "free_root_0_to_0p575_aggregate": selected_metrics(free),
            "prescribed_full_5p8s_aggregate_not_same_window": selected_metrics(prescribed),
            "missing": [
                "free per-physics-step/per-control-step q/root/tilt/contact/torque/action trace",
                "terminal qpos/qvel/contact and exact z-vs-tilt termination trigger",
                "prescribed replay metrics restricted to the same 0-0.575s window",
            ],
        },
        "attribution": {
            "numeric_reset_mismatch": "not supported as primary; exact values assigned, but solver/contact warmstart and original controller hidden state are missing",
            "reference_contact_feasibility": "strongly supported deficiency: 97.25% flight, no right active-sole contact, free replay quickly realizes near-double-support",
            "bare_pd_closed_loop": "strongly supported deficiency: prescribed trackability passes but stateless free-root PD falls; causal share remains confounded with contact-infeasible reference",
            "primary_current_conclusion": "reference contact feasibility and bare-PD closed-loop insufficiency are entangled; available evidence does not justify blaming a numeric reset mismatch",
        },
        "single_next_falsifiable_experiment": (
            "只新增一次 **free-root exact-joint kinematic oracle**：同一Phase30 candidate、同一frame0 q/dq/root、同一official scene/50Hz path，"
            "逐物理步精确施加reference q/dq但保持root完全自由；与已存在的Phase33 free裸PD结果比较，不改reference、PD、root、"
            "不加warmup。若oracle仍在相近时刻倒下且contact仍与reference冲突，优先证伪‘只是PD跟踪误差’并锁定reference/contact/reset"
            "可行性；若oracle显著存活且contact对齐，则说明裸PD/闭环不足是主导。"
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(phase33.phase28.json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["attribution"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
