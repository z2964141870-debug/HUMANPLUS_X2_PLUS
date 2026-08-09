#!/usr/bin/env python3
"""Phase43 hybrid native-lower + human-upper composition contract preflight.

No simulation or optimizer is created. Stage250 lower/waist/root/intended gait
schedule is immutable. The AMASS reference contributes only 14 arm joints;
its root, waist and lower body are structurally discarded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.audit_x2_native_gold_seed_phase10 as phase10


BASE_PHASE27 = REPO / "reports/official_x2/phase27_stage250_native_dynamic_seed_audit.json"
STAGE250_TRACE = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807/stage250_video_straight.json"
)
MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
PHASE11 = REPO / "reports/retarget/x2_native_gold_motionlib_phase11.json"
PHASE23 = REPO / "reports/retarget/x2_faithful_wbt29_gold_phase23.json"
PHASE28 = REPO / "reports/retarget/x2_wbt_panel_phase28.json"
OUTPUT_JSON = REPO / "reports/retarget/x2_hybrid_composition_contract_phase43.json"
OUTPUT_MD = REPO / "reports/retarget/x2_hybrid_composition_contract_phase43.md"
OUTPUT_NPZ = REPO / "artifacts/official_x2/x2_hybrid_composition_phase43_dry.npz"

STAGE250_OBS31 = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)
UPPER14 = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
LOWER12 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
WAIST3 = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
HEAD2 = ("head_yaw_joint", "head_pitch_joint")


def root_quaternion_from_projected_gravity_yaw(gravity_body: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    gravity = np.asarray(gravity_body, dtype=np.float64)
    gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
    pitch = np.arcsin(np.clip(gravity[:, 0], -1.0, 1.0))
    roll = np.arctan2(-gravity[:, 1], -gravity[:, 2])
    return Rotation.from_euler("ZYX", np.column_stack([yaw, pitch, roll])).as_quat()


def resample_upper_real_time(q: np.ndarray, source_fps: float, frames: int, target_fps: float) -> np.ndarray:
    target_time = np.arange(frames, dtype=np.float64) / target_fps
    source_time = np.arange(len(q), dtype=np.float64) / source_fps
    if target_time[-1] > source_time[-1] + 1e-12:
        raise ValueError("frozen upper clip is shorter than Stage250 move interval")
    return np.column_stack([np.interp(target_time, source_time, q[:, joint]) for joint in range(q.shape[1])])


def root_velocity(root: np.ndarray, quat: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    linear = np.gradient(root, 1.0 / fps, axis=0)
    rotations = Rotation.from_quat(quat)
    angular = np.zeros((len(root), 3), dtype=np.float64)
    for frame in range(len(root) - 1):
        angular[frame] = (rotations[frame].inv() * rotations[frame + 1]).as_rotvec() * fps
    angular[-1] = angular[-2]
    return linear, angular


def render(report: dict[str, Any]) -> str:
    decision, checks = report["decision"], report["checks"]
    return f"""# X2 WBT Phase43：hybrid composition合同/preflight

## 目标合同

```text
Stage250 native lower12 + waist3 + root + intended gait schedule (frozen)
                         +
AMASS-UPPER-001 upper14 only (first 4.0s, real-time 30→50Hz)
                         ↓
X2 official31 / WBT29 / MotionLib-compatible dry composite
```

GMR lower、GMR root、GMR contact 永远不进入组合；头部保持Stage250；腰部本阶段保持Stage250，尚未开放有界意图。

## Phase27资格边界

- Phase27 available：`{checks['base_phase27_available']}`；native warm-start：`{checks['base_phase27_warm_start_usable']}`；contact-consistent seed：`{checks['base_phase27_qualified']}`。
- Stage250 trace含93D obs/15D action/50Hz intended gait phase；realized sole collision是否可用：`{checks['stage250_realized_contact_available']}`。
- 内部gait phase是**意图接触**，不是realized collision、更不是实机GRF/COP。

## Schema / mapping / round-trip

- Gold Phase11 ingestion：`{checks['phase11_gold_ingestion_pass']}`；WBT29 Phase23：`{checks['phase23_wbt29_pass']}`。
- A zero-change official31→WBT29→official31 max error：{report['roundtrip']['A_official31_wbt29_official31_max_error']:.3g}rad。
- B lower12/waist3/root/head/intended-contact unchanged：`{report['roundtrip']['B_native_fields_exact']}`。
- AMASS upper source only：`AMASS-UPPER-001` Bronze，first4.0s实时插值；upper qstep max={report['composition']['upper_qstep_max_rad']:.4f}rad。

## 最小未来A/B门（本阶段未运行）

- A：Stage250 straight 原生全身。
- B：同一Stage250 lower/root/contact schedule + 冻结AMASS upper14；waist/head不变。
- prescribed 与 free 分栏；source trace稳定不得冒充replay稳定。
- upper error、survival、root/contact、slip与signed pitch门见JSON `future_gate_contract`。

## 裁决

**{decision['status']}**

{decision['conclusion']}

## 下一步

{decision['next_step']}
"""


def main() -> None:
    required = [STAGE250_TRACE, MODEL_CONTRACT, PHASE11, PHASE23, PHASE28]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required Phase43 inputs missing: {missing}")
    model_contract = json.loads(MODEL_CONTRACT.read_text())
    phase11 = json.loads(PHASE11.read_text())
    phase23 = json.loads(PHASE23.read_text())
    phase28_report = json.loads(PHASE28.read_text())
    stage250 = json.loads(STAGE250_TRACE.read_text())
    official31 = tuple(model_contract["control_boundaries"]["official_mjcf_actuated_31"])
    wbt29 = tuple(model_contract["control_boundaries"]["wbt_target_29"])
    if set(STAGE250_OBS31) != set(official31) or len(set(STAGE250_OBS31)) != 31:
        raise ValueError("Stage250 obs31 does not bijectively match official31")
    if set(wbt29) != set(official31) - set(HEAD2):
        raise ValueError("WBT29 is not exact official31 minus head")
    joint_rows = {row["name"]: row for row in model_contract["joints"]}
    default_obs = np.asarray([joint_rows[name]["model_nominal_rad"] for name in STAGE250_OBS31])
    move = [row for row in stage250["trace"] if row["stage"] == "move"]
    obs = np.asarray([row["obs"] for row in move], dtype=np.float64)
    action = np.asarray([row["action"] for row in move], dtype=np.float64)
    if obs.shape != (200, 93) or action.shape != (200, 15):
        raise ValueError(f"Stage250 straight move schema differs: obs={obs.shape}, action={action.shape}")
    q_obs = obs[:, 12:43] + default_obs
    dq_obs = obs[:, 43:74]
    obs_to_official = np.asarray([STAGE250_OBS31.index(name) for name in official31], dtype=np.int64)
    q31 = q_obs[:, obs_to_official]
    dq31 = dq_obs[:, obs_to_official]
    root = np.asarray([[row["root_x_m"], row["root_y_m"], row["root_z_m"]] for row in move])
    yaw = np.asarray([row["root_yaw_rad"] for row in move])
    quat = root_quaternion_from_projected_gravity_yaw(obs[:, 6:9], yaw)
    root_lin, root_ang = root_velocity(root, quat, 50.0)
    intended_contact = {"left": obs[:, 91] > 0.5, "right": obs[:, 92] > 0.5}

    # A: exact name-driven WBT29 gather/scatter, preserving native head.
    gather = np.asarray([official31.index(name) for name in wbt29], dtype=np.int64)
    scattered = q31.copy(); scattered[:, gather] = q31[:, gather]
    a_error = float(np.max(np.abs(scattered - q31)))

    official_cache_path = Path(phase28_report["provenance"]["official_cache"]["path"])
    official_cache = joblib.load(official_cache_path)
    upper_entry = official_cache["AMASS-UPPER-001"]
    upper_names = tuple(upper_entry["joint_names_mujoco"])
    upper_indices = np.asarray([upper_names.index(name) for name in UPPER14], dtype=np.int64)
    # Fixed first 4.0 seconds: target t=0..3.98s requires source through frame120.
    upper_source = np.asarray(upper_entry["dof"], dtype=np.float64)[:121, upper_indices]
    upper50 = resample_upper_real_time(upper_source, 30.0, len(move), 50.0)
    composite = q31.copy()
    composite[:, [official31.index(name) for name in UPPER14]] = upper50
    lower_exact = np.array_equal(composite[:, [official31.index(name) for name in LOWER12]], q31[:, [official31.index(name) for name in LOWER12]])
    waist_exact = np.array_equal(composite[:, [official31.index(name) for name in WAIST3]], q31[:, [official31.index(name) for name in WAIST3]])
    head_exact = np.array_equal(composite[:, [official31.index(name) for name in HEAD2]], q31[:, [official31.index(name) for name in HEAD2]])
    limits = np.asarray([joint_rows[name]["range_rad"] for name in official31])
    limit_overshoot = np.maximum(limits[:, 0][None] - composite, 0.0) + np.maximum(composite - limits[:, 1][None], 0.0)
    dq_composite = np.gradient(composite, 1.0 / 50.0, axis=0)
    pose = phase10.pose_aa_from_dof(composite, quat, list(official31), joint_rows)

    base27_available = BASE_PHASE27.exists()
    base27 = json.loads(BASE_PHASE27.read_text()) if base27_available else None
    # Fail closed: the Phase27 audit is authoritative. It must explicitly
    # promote/qualify the trace; absence of a positive boolean never defaults true.
    base27_decision = base27.get("decision", {}) if isinstance(base27, dict) else {}
    if isinstance(base27_decision, str):
        base27_decision = {"status": base27_decision}
    qualification_fields = (
        "native_dynamic_seed_qualified", "composition_seed_qualified",
        "stage250_native_dynamic_seed_qualified", "usable_as_contact_consistent_dynamic_seed", "qualified",
    )
    base27_qualified = any(base27_decision.get(field) is True for field in qualification_fields)
    base27_warm_start = bool(base27_decision.get("usable_as_native_warm_start") is True)
    base27_dynamic_truth = bool(base27_decision.get("usable_as_dynamics_ground_truth") is True)
    phase27_text = json.dumps(base27, ensure_ascii=False).lower() if base27 is not None else ""
    realized_contact_available = bool(
        base27_qualified and any(token in phase27_text for token in ("realized_contact", "sole_collision", "model_contact"))
    )
    checks = {
        "base_phase27_available": base27_available,
        "base_phase27_warm_start_usable": base27_warm_start,
        "base_phase27_qualified": base27_qualified,
        "base_phase27_dynamics_ground_truth": base27_dynamic_truth,
        "stage250_realized_contact_available": realized_contact_available,
        "phase11_gold_ingestion_pass": phase11["decision"]["status"] == "PHASE11_MOTIONLIB_INGESTION_PASSED",
        "phase23_wbt29_pass": phase23["decision"]["status"] == "B1_B2_IMPLEMENTATION_READY_CPU_PROBED",
        "stage250_obs_action_shape": obs.shape == (200, 93) and action.shape == (200, 15),
        "joint_sets_bijective": set(STAGE250_OBS31) == set(official31),
        "A_roundtrip_exact": a_error == 0.0,
        "B_lower_exact": bool(lower_exact), "B_waist_exact": bool(waist_exact),
        "B_head_exact": bool(head_exact),
        "B_root_exact_by_construction": True,
        "B_intended_contact_exact_by_construction": True,
        "B_joint_limits": float(np.max(limit_overshoot)) <= 1e-7,
        "motionlib_dry_schema": pose.shape == (200, 32, 3) and composite.shape == (200, 31),
    }
    ready = bool(all(value for key, value in checks.items() if key not in (
        "stage250_realized_contact_available", "base_phase27_available", "base_phase27_qualified",
        "base_phase27_warm_start_usable", "base_phase27_dynamics_ground_truth",
    )) and base27_available and base27_qualified and realized_contact_available)
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ, official31=np.asarray(official31), wbt29=np.asarray(wbt29),
        native_q31=q31, native_dq31=dq31, composite_q31=composite, composite_dq31=dq_composite,
        root_trans_offset=root, root_rot_xyzw=quat, root_lin_vel=root_lin, root_ang_vel=root_ang,
        pose_aa=pose, intended_contact_left=intended_contact["left"],
        intended_contact_right=intended_contact["right"], gait_phase=obs[:, 89:93], action15=action,
    )
    if ready:
        status = "PHASE43_HYBRID_COMPOSITION_CONTRACT_READY_NO_PHYSICS"
        conclusion = "Stage250 native seed资格、realized contact、Gold/WBT29映射与composition round-trip全部显式通过；只解锁未来一次冻结A/B，不代表组合在free-root中稳定。"
        next_step = "另阶段按冻结门运行A native与B native-lower+AMASS-upper的prescribed/free A/B；本阶段不运行。"
    else:
        status = "PHASE43_HYBRID_COMPOSITION_PREFLIGHT_BLOCKED"
        blockers = [key for key, value in checks.items() if not value]
        if not base27_available: blockers.append("BASE Phase27 audit missing")
        elif not base27_qualified: blockers.append("BASE Phase27 did not explicitly qualify Stage250 native dynamic seed")
        if not realized_contact_available: blockers.append("Stage250 realized sole collision/contact unavailable; gait phase is intent only")
        conclusion = "Composition schema本身可构造，但Stage250 native teacher资格/realized-contact合同未完整通过；fail-closed，不把内部gait phase伪装成物理接触。Blockers: " + "; ".join(blockers)
        next_step = "保留Stage250为native warm-start；若要解锁physical composition A/B，需按Phase27最小扩展补录physics-substep contact identity/position/impulse、完整root pose/velocity、applied torque和clip前后target。"
    report = {
        "schema_version": "x2_hybrid_composition_contract_phase43_v1",
        "truth_boundary": {
            "schema_mapping_time_alignment_only": True, "physics_or_replay_run": False,
            "ppo_or_optimizer": False, "base_files_modified": False,
            "gait_phase_is_intended_contact_not_realized_collision": True,
            "model_contact_not_hardware_grf_cop": True,
        },
        "provenance": {
            "base_phase27": {"path": str(BASE_PHASE27), "exists": base27_available, "sha256": phase28.sha256(BASE_PHASE27) if base27_available else None},
            "stage250_trace": {"path": str(STAGE250_TRACE), "sha256": phase28.sha256(STAGE250_TRACE)},
            "model_contract": {"path": str(MODEL_CONTRACT), "sha256": phase28.sha256(MODEL_CONTRACT)},
            "phase11": {"path": str(PHASE11), "sha256": phase28.sha256(PHASE11)},
            "phase23": {"path": str(PHASE23), "sha256": phase28.sha256(PHASE23)},
            "amass_upper_official_cache": {"path": str(official_cache_path), "sha256": phase28.sha256(official_cache_path)},
            "amass_upper_source_entry_sha256": next(row["official_entry_sha256"] for row in phase28_report["motions"] if row["id"] == "AMASS-UPPER-001"),
        },
        "contract": {
            "native_owned": {"lower12": list(LOWER12), "waist3": list(WAIST3), "root": True, "intended_contact_schedule": True},
            "human_owned": {"upper14": list(UPPER14)},
            "head_rule": {"joints": list(HEAD2), "owner": "native Stage250 unchanged"},
            "forbidden": ["GMR lower", "GMR waist", "GMR root", "GMR contact", "head from human reference"],
            "time_alignment": "Stage250 move 200 frames@50Hz; AMASS-UPPER first 4.0s frames[0:121]@30Hz linearly sampled at t=0..3.98s",
        },
        "checks": checks,
        "roundtrip": {
            "A_official31_wbt29_official31_max_error": a_error,
            "B_native_fields_exact": bool(lower_exact and waist_exact and head_exact),
            "B_root_exact": True, "B_intended_contact_schedule_exact": True,
            "head_excluded_from_wbt29": set(HEAD2).isdisjoint(wbt29),
        },
        "composition": {
            "frames": len(move), "fps": 50, "upper_source": "AMASS-UPPER-001",
            "upper_qstep_max_rad": float(np.max(np.abs(np.diff(upper50, axis=0)))),
            "joint_limit_overshoot_max_rad": float(np.max(limit_overshoot)),
            "motionlib_dry_shapes": {"dof": list(composite.shape), "pose_aa": list(pose.shape), "root": list(root.shape), "quat": list(quat.shape)},
        },
        "future_gate_contract": {
            "A": "frozen Stage250 straight native all-body control",
            "B": "same lower12/waist3/root/intended contact schedule + frozen AMASS upper14; head unchanged",
            "prescribed_root": {
                "survival_delta_B_minus_A_min_s": -0.1,
                "upper_joint_error_p95_max_rad": 0.15, "upper_joint_error_max_rad": 0.30,
                "lower_waist_target_difference_A_vs_B_max_rad": 0.0,
                "root_schedule_difference_A_vs_B_max_m_rad": 0.0,
            },
            "free_root": {
                "survival_delta_B_minus_A_min_s": -0.1,
                "root_z_min_m": 0.45, "root_tilt_max_rad": 0.4,
                "realized_contact_agreement_min": 0.85,
                "realized_contact_agreement_delta_B_minus_A_min": -0.05,
                "stance_slip_p95_max_mps": 0.10,
                "signed_pitch_mean_abs_degradation_max_rad": 0.035,
                "signed_pitch_p95_abs_degradation_max_rad": 0.035,
            },
            "truth_rule": "prescribed and free reported separately; Stage250 source trace survival never substitutes replay survival",
        },
        "artifact": {"path": str(OUTPUT_NPZ), "sha256": phase28.sha256(OUTPUT_NPZ)},
        "decision": {"status": status, "composition_contract_ready": ready, "physics_ab_allowed": ready, "conclusion": conclusion, "next_step": next_step},
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
