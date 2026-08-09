#!/usr/bin/env python3
"""Audit the sole Phase50 B episode against immutable BASE Phase34 A evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "reports/official_x2/phase50_hybrid_upper_closed_prereg.json"
PREREG_AMENDMENT = REPO / "reports/official_x2/phase50_hybrid_upper_closed_prereg_amendment.json"
PHASE34_MANIFEST = REPO / "reports/official_x2/phase34_closed_full_trace_manifest.json"
ADAPTER = REPO / "tools/official_x2/stage208_official_mujoco_adapter.py"
UPPER = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
OUTPUT_JSON = REPO / "reports/retarget/x2_hybrid_upper_closed_phase50.json"
OUTPUT_MD = REPO / "reports/retarget/x2_hybrid_upper_closed_phase50.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_upper_only_contract(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_apply_upper_motion"
    )
    referenced = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    forbidden = sorted(referenced & {"LEG_JOINTS", "LOWER_JOINTS", "WAIST_JOINTS", "HEAD_JOINTS"})
    target_writes = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "targets"):
                target_writes.append(ast.unparse(target.slice))
    return {
        "function": "_apply_upper_motion",
        "forbidden_joint_group_references": forbidden,
        "target_write_subscripts": target_writes,
        "arm_loop_present": "for index, name in enumerate(ARM_JOINTS)" in ast.unparse(function),
        "pass": not forbidden and target_writes == ["name"]
    }


def move_rows(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in rollout["trace"] if row["stage"] == "move"]
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 move rows, got {len(rows)}")
    return rows


def render(report: dict[str, Any]) -> str:
    c = report["comparison"]
    g = report["gates"]
    return f"""# WBT Phase50：closed AimDK 上肢合成单次 A/B

## 假设

在 Stage219/Stage250 的腿、腰、root、PD、命令和 stand→move→stop 合同完全冻结时，只在 move 窗口叠加有界 `AMASS-UPPER-001` upper14，可以改善对该上肢意图的跟踪，而不破坏原生速度后端。

## 干预与对照

- A：复用 BASE Phase34 的 immutable closed AimDK episode，没有重跑。
- B：唯一一条新 episode；只有 upper14 target path 开启。新 physics episode 数：`{report['execution']['new_B_episodes']}`，重试：`0`。
- domain233 在 simulator 启动前被 CycloneDDS 拒绝，physics episode=`0`；按 amendment 只改为 domain229 后完成唯一实际 B。
- Phase32 `LD_PRELOAD` observer 保存了 B 的完整 1 kHz physics；接触是官方 MuJoCo 模型真值，不是实机 GRF/COP。

## 静态合同

- Phase34 adapter hash exact：`{report['static']['adapter_hash_exact']}`。
- model/PD/command/stand-move-stop 等冻结配置逐项 exact：`{report['static']['frozen_summary_config_exact']}`。
- `_apply_upper_motion` 不引用腿/腰/root/head group，直接写 mask 仅 ARM14：`{report['static']['upper_only_ast']['pass']}`。
- bounded target excursion/speed：{c['B_upper_target_excursion_max_rad']:.6f} rad / {c['B_upper_target_speed_max_radps']:.6f} rad/s。

## 结果

| 指标 | A | B | 门 | 通过 |
|---|---:|---:|---:|:---:|
| AMASS upper counterfactual RMSE | {c['A_counterfactual_upper_rmse_rad']:.5f} | {c['B_upper_rmse_rad']:.5f} | B < A | {g['upper_rmse_improved']} |
| AMASS upper counterfactual p95 | {c['A_counterfactual_upper_abs_p95_rad']:.5f} | {c['B_upper_abs_p95_rad']:.5f} | B < A | {g['upper_p95_improved']} |
| move forward (m) | {c['A_move_forward_m']:.4f} | {c['B_move_forward_m']:.4f} | drop≤0.15 | {g['move_forward']} |
| abs lateral (m) | {abs(c['A_move_lateral_m']):.4f} | {abs(c['B_move_lateral_m']):.4f} | increase≤0.05 | {g['lateral']} |
| heading max (rad) | {c['A_heading_max_rad']:.4f} | {c['B_heading_max_rad']:.4f} | increase≤0.035 | {g['heading']} |
| stop drift (m) | {c['A_stop_drift_m']:.4f} | {c['B_stop_drift_m']:.4f} | increase≤0.05 | {g['stop_drift']} |
| stop settle (s) | {c['A_stop_settle_s']:.3f} | {c['B_stop_settle_s']} | increase≤0.50 | {g['stop_settle']} |
| move signed pitch mean (rad) | {c['A_move_pitch_mean_rad']:.4f} | {c['B_move_pitch_mean_rad']:.4f} | abs increase≤0.035 | {g['signed_pitch']} |

- B full/start/move/stop gates：`{g['absolute_functional']}`。
- 左/右 realized-contact agreement gate：`{g['contact_agreement']}`；stance-slip p95 relative gate：`{g['stance_slip']}`。
- upper fallback steps：`{c['B_upper_fallback_steps']}`。

## 结论

**{report['decision']['status']}**

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-rollout", type=Path, required=True)
    parser.add_argument("--b-audit", type=Path, required=True)
    args = parser.parse_args()

    prereg = json.loads(PREREG.read_text())
    phase34 = json.loads(PHASE34_MANIFEST.read_text())
    a_rollout_path = Path(phase34["artifacts"]["rollout"])
    a_audit_path = Path(phase34["artifacts"]["full_offline_audit"])
    a = json.loads(a_rollout_path.read_text())
    b = json.loads(args.b_rollout.read_text())
    aa = json.loads(a_audit_path.read_text())
    ba = json.loads(args.b_audit.read_text())
    a_rows, b_rows = move_rows(a), move_rows(b)

    target = np.asarray([row["upper_target_rad"] for row in b_rows], dtype=np.float64)
    a_actual = np.asarray([row["upper_actual_rad"] for row in a_rows], dtype=np.float64)
    b_actual = np.asarray([row["upper_actual_rad"] for row in b_rows], dtype=np.float64)
    a_error = a_actual - target
    b_error = b_actual - target
    a_summary, b_summary = a["summary"], b["summary"]
    limits = prereg["physical_gates"]["relative_to_A"]

    def contact_metric(audit: dict[str, Any], side: str, field: str) -> Any:
        return audit["analysis"]["contact"][side][field]

    comparison = {
        "A_counterfactual_upper_rmse_rad": float(np.sqrt(np.mean(a_error ** 2))),
        "B_upper_rmse_rad": float(np.sqrt(np.mean(b_error ** 2))),
        "A_counterfactual_upper_abs_p95_rad": float(np.quantile(np.abs(a_error), .95)),
        "B_upper_abs_p95_rad": float(np.quantile(np.abs(b_error), .95)),
        "B_upper_target_excursion_max_rad": float(b_summary["upper_target_excursion_abs_max_rad"]),
        "B_upper_target_speed_max_radps": float(b_summary["upper_target_speed_abs_max_radps"]),
        "B_upper_fallback_steps": int(b_summary["upper_fallback_steps"]),
        "A_root_z_min_m": float(a_summary["root_z_min_m"]), "B_root_z_min_m": float(b_summary["root_z_min_m"]),
        "A_root_tilt_max_rad": float(a_summary["root_tilt_max_rad"]), "B_root_tilt_max_rad": float(b_summary["root_tilt_max_rad"]),
        "A_move_forward_m": float(a_summary["move_forward_displacement_m"]), "B_move_forward_m": float(b_summary["move_forward_displacement_m"]),
        "A_move_lateral_m": float(a_summary["move_lateral_displacement_m"]), "B_move_lateral_m": float(b_summary["move_lateral_displacement_m"]),
        "A_heading_max_rad": float(a_summary["move_heading_max_deviation_rad"]), "B_heading_max_rad": float(b_summary["move_heading_max_deviation_rad"]),
        "A_stop_drift_m": float(a_summary["stop_root_xy_drift_m"]), "B_stop_drift_m": float(b_summary["stop_root_xy_drift_m"]),
        "A_stop_settle_s": float(a_summary["stop_settle_time_s"]), "B_stop_settle_s": b_summary["stop_settle_time_s"],
        "A_move_pitch_mean_rad": float(a_summary["move_root_pitch_mean_rad"]), "B_move_pitch_mean_rad": float(b_summary["move_root_pitch_mean_rad"]),
        "contact": {},
    }
    for side in ("left", "right"):
        comparison["contact"][side] = {
            "A_agreement": float(contact_metric(aa, side, "generator_agreement_move_50hz")),
            "B_agreement": float(contact_metric(ba, side, "generator_agreement_move_50hz")),
            "A_stance_slip_p95_mps": float(contact_metric(aa, side, "realized_stance_contact_point_slip_mps")["p95"]),
            "B_stance_slip_p95_mps": float(contact_metric(ba, side, "realized_stance_contact_point_slip_mps")["p95"]),
        }

    gates = {
        "absolute_functional": all(bool(b_summary[key]) for key in ("startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")),
        "upper_rmse_improved": comparison["B_upper_rmse_rad"] < comparison["A_counterfactual_upper_rmse_rad"],
        "upper_p95_improved": comparison["B_upper_abs_p95_rad"] < comparison["A_counterfactual_upper_abs_p95_rad"],
        "upper_nontrivial_and_bounded": 0.01 <= comparison["B_upper_target_excursion_max_rad"] <= 0.1200001 and comparison["B_upper_target_speed_max_radps"] <= 0.200001,
        "root_z": comparison["B_root_z_min_m"] >= comparison["A_root_z_min_m"] - limits["root_z_min_drop_max_m"],
        "root_tilt": comparison["B_root_tilt_max_rad"] <= comparison["A_root_tilt_max_rad"] + limits["root_tilt_max_increase_rad"],
        "move_forward": comparison["B_move_forward_m"] >= comparison["A_move_forward_m"] - limits["move_forward_drop_max_m"],
        "lateral": abs(comparison["B_move_lateral_m"]) <= abs(comparison["A_move_lateral_m"]) + limits["move_abs_lateral_increase_max_m"],
        "heading": comparison["B_heading_max_rad"] <= comparison["A_heading_max_rad"] + limits["move_heading_max_increase_rad"],
        "stop_drift": comparison["B_stop_drift_m"] <= comparison["A_stop_drift_m"] + limits["stop_xy_drift_increase_max_m"],
        "stop_settle": comparison["B_stop_settle_s"] is not None and comparison["B_stop_settle_s"] <= comparison["A_stop_settle_s"] + limits["stop_settle_time_increase_max_s"],
        "signed_pitch": abs(comparison["B_move_pitch_mean_rad"]) <= abs(comparison["A_move_pitch_mean_rad"]) + limits["move_signed_pitch_abs_increase_max_rad"],
        "contact_agreement": all(comparison["contact"][s]["B_agreement"] >= comparison["contact"][s]["A_agreement"] - limits["realized_contact_agreement_drop_max_each"] for s in ("left", "right")),
        "stance_slip": all(comparison["contact"][s]["B_stance_slip_p95_mps"] <= comparison["contact"][s]["A_stance_slip_p95_mps"] + limits["realized_stance_slip_p95_increase_max_mps_each"] for s in ("left", "right")),
    }
    frozen_summary_keys = (
        "model", "stationary_model", "recovery_model", "template",
        "model_input_dim", "stationary_model_input_dim", "recovery_model_input_dim",
        "pd_profile", "pd_kp_multiplier", "pd_kd_multiplier", "default_pose_profile",
        "command_vx_mps", "policy_vx_floor_mps", "state_prediction_seconds", "state_qos_depth",
        "prepare_seconds", "stand_seconds", "move_seconds", "stop_seconds",
        "move_template_multiplier", "stationary_controller", "stop_controller",
        "stationary_warmup_seconds", "stationary_blend", "heading_gain",
        "heading_rate_limit_radps", "action_bias_mode", "action_bias",
        "action_bias_ramp_seconds", "ankle_roll_common_bias", "recovery_enter_m",
        "recovery_exit_m", "recovery_slew_rate_per_s", "waist_tilt_action_multiplier",
    )
    frozen_config_comparison = {
        key: {"A": a_summary.get(key), "B": b_summary.get(key), "exact": a_summary.get(key) == b_summary.get(key)}
        for key in frozen_summary_keys
    }
    static = {
        "adapter_sha256": sha256(ADAPTER),
        "adapter_hash_exact": sha256(ADAPTER) == prereg["static_gates_before_physics"]["phase34_adapter_sha_exact"],
        "upper_artifact_hash_exact": sha256(UPPER) == prereg["static_gates_before_physics"]["upper_artifact_sha_exact"],
        "upper_only_ast": static_upper_only_contract(ADAPTER.read_text()),
        "frozen_summary_config": frozen_config_comparison,
        "frozen_summary_config_exact": all(row["exact"] for row in frozen_config_comparison.values()),
    }
    promoted = all(gates.values()) and static["adapter_hash_exact"] and static["upper_artifact_hash_exact"] and static["upper_only_ast"]["pass"] and static["frozen_summary_config_exact"]
    report = {
        "schema_version": "x2_hybrid_upper_closed_phase50_v1",
        "hypothesis": prereg["hypothesis"],
        "execution": {
            "A_reused_not_rerun": True, "new_B_episodes": 1, "B_retries": 0,
            "prephysics_invalid_starts": 1,
            "prephysics_invalid_start_physics_episodes": 0,
            "prereg_amendment": str(PREREG_AMENDMENT),
            "prereg_amendment_sha256": sha256(PREREG_AMENDMENT),
            "A_rollout": str(a_rollout_path), "A_rollout_sha256": sha256(a_rollout_path),
            "B_rollout": str(args.b_rollout), "B_rollout_sha256": sha256(args.b_rollout),
            "B_mmap_audit": str(args.b_audit), "B_mmap_audit_sha256": sha256(args.b_audit),
            "prereg": str(PREREG), "prereg_sha256": sha256(PREREG),
        },
        "static": static,
        "comparison": comparison,
        "gates": gates,
        "truth_boundary": prereg["truth_boundary"],
        "decision": {
            "status": "PROMOTABLE_ONE_EPISODE_HYBRID_EXISTENCE" if promoted else "NOT_PROMOTABLE_UNDER_PREREGISTERED_GATE",
            "conclusion": (
                "单次closed官方物理中，上肢意图跟踪与全部冻结后端退化门同时通过；这里只证明hybrid composition存在性，不证明跨动作/跨扰动鲁棒。"
                if promoted else
                "唯一B episode未同时满足上肢改善与冻结后端的全部预注册门；不重试、不改腿腰补偿，不把局部改善表述为WBT晋级。"
            ),
            "next_step": (
                "先由主线裁决是否值得做固定多动作复核；本阶段不训练。" if promoted
                else "停止该固定upper14 composition候选；如需继续，应提出新的单变量假设，而不是重跑或放宽门。"
            ),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
