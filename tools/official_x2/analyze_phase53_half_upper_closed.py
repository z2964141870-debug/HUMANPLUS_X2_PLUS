#!/usr/bin/env python3
"""Apply the preregistered Phase53 gates to immutable A/B1 and sole Bhalf."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "reports/official_x2/phase53_half_upper_closed_prereg.json"
A_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase34_full_stage250_trace/phase34_stage250_closed_full_once_d230.json")
B1_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase50_hybrid_upper_closed/phase50_stage250_hybrid_upper_B_once_d229.json")
BH_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase53_half_upper_closed/phase53_stage250_half_upper_once_d227.json")
A_AUDIT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase34_full_stage250_trace/phase34_full_offline_audit.json")
B1_AUDIT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase50_hybrid_upper_closed/phase50_B_full_offline_audit.json")
BH_AUDIT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase53_half_upper_closed/phase53_Bhalf_full_offline_audit.json")
OUTPUT_JSON = REPO / "reports/retarget/x2_half_upper_closed_phase53.json"
OUTPUT_MD = REPO / "reports/retarget/x2_half_upper_closed_phase53.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def move(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in data["trace"] if row["stage"] == "move"]
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 move rows, got {len(rows)}")
    return rows


def contact(audit: dict[str, Any], side: str) -> dict[str, float]:
    row = audit["analysis"]["contact"][side]
    return {
        "agreement": float(row["generator_agreement_move_50hz"]),
        "slip_p95_mps": float(row["realized_stance_contact_point_slip_mps"]["p95"]),
        "clearance_p95_m": float(row["realized_swing_sole_clearance_m"]["p95"]),
    }


def render(r: dict[str, Any]) -> str:
    g, c = r["gates"], r["comparison"]
    return f"""# WBT Phase53：half-amplitude / half-slew closed 诊断

## 假设

若 Phase50 的问题主要是上肢目标幅值/速度超过冻结 Stage250 后端的吸扰范围，那么把 upper scale、excursion、slew 同时减半，应在保留实质上肢跟踪的同时显著回收横漂和航向偏差。

## 干预 / 对照

- A：immutable BASE Phase34。
- B1：immutable Phase50，scale=0.25、±0.12rad、0.20rad/s。
- Bhalf：唯一新 episode；scale=0.125、±0.06rad、0.10rad/s。其他 hash/PD/command/timing 全冻结。
- 新 physics episode=`1`，重试=`0`；未训练、未扫其他幅值。

## 结果

| 指标 | A/counterfactual A | B1 | Bhalf | Phase53门 | 通过 |
|---|---:|---:|---:|---:|:---:|
| full gate | True | {c['B1_full_gate']} | {c['Bhalf_full_gate']} | True | {g['functional']} |
| half-target upper RMSE | {c['A_counterfactual_half_upper_rmse_rad']:.5f} | - | {c['Bhalf_upper_rmse_rad']:.5f} | 改善≥10% | {g['upper_rmse_improvement']} |
| half-target upper p95 | {c['A_counterfactual_half_upper_p95_rad']:.5f} | - | {c['Bhalf_upper_p95_rad']:.5f} | 改善≥10% | {g['upper_p95_improvement']} |
| signed lateral (m) | {c['A_lateral_m']:.5f} | {c['B1_lateral_m']:.5f} | {c['Bhalf_lateral_m']:.5f} | excess回收≥25% | {g['lateral_recovery']} |
| signed yaw progress (rad) | {c['A_yaw_rad']:.5f} | {c['B1_yaw_rad']:.5f} | {c['Bhalf_yaw_rad']:.5f} | excess回收≥25% | {g['yaw_recovery']} |
| stop drift (m) | {c['A_stop_drift_m']:.5f} | {c['B1_stop_drift_m']:.5f} | {c['Bhalf_stop_drift_m']:.5f} | ≤B1+0.02 | {g['stop_drift']} |

- upper RMSE/p95实际改善：{c['upper_rmse_improvement_fraction']*100:.2f}% / {c['upper_p95_improvement_fraction']*100:.2f}% 。
- lateral/yaw excess回收：{c['lateral_recovery_fraction']*100:.2f}% / {c['yaw_recovery_fraction']*100:.2f}% 。
- Bhalf left/right slip p95：{c['contact']['Bhalf']['left']['slip_p95_mps']:.3f}/{c['contact']['Bhalf']['right']['slip_p95_mps']:.3f}m/s；contact保护门=`{g['contact_agreement']}`，slip保护门=`{g['stance_slip']}`。
- Bhalf 1kHz telemetry alignment strict max gate：`{r['evidence_boundary']['Bhalf_alignment_pass_1e8']}`；因此contact只作诊断，不改变上述telemetry核心否证。

## 结论

**{r['decision']['status']}**

{r['decision']['conclusion']}

## 下一步

{r['decision']['next_step']}

接触/滑移为 official MuJoCo 模型真值，不是实机 GRF/COP。
"""


def main() -> None:
    prereg = load(PREREG); threshold = prereg["success_gates"]
    data = {"A": load(A_ROLLOUT), "B1": load(B1_ROLLOUT), "Bhalf": load(BH_ROLLOUT)}
    summaries = {key: value["summary"] for key, value in data.items()}
    audits = {"A": load(A_AUDIT), "B1": load(B1_AUDIT), "Bhalf": load(BH_AUDIT)}
    a_move, bh_move = move(data["A"]), move(data["Bhalf"])
    half_target = np.asarray([row["upper_target_rad"] for row in bh_move], dtype=np.float64)
    a_actual = np.asarray([row["upper_actual_rad"] for row in a_move], dtype=np.float64)
    bh_actual = np.asarray([row["upper_actual_rad"] for row in bh_move], dtype=np.float64)
    a_error, bh_error = a_actual - half_target, bh_actual - half_target
    a_rmse, bh_rmse = float(np.sqrt(np.mean(a_error ** 2))), float(np.sqrt(np.mean(bh_error ** 2)))
    a_p95, bh_p95 = float(np.quantile(np.abs(a_error), .95)), float(np.quantile(np.abs(bh_error), .95))
    rmse_improvement, p95_improvement = 1.0 - bh_rmse / a_rmse, 1.0 - bh_p95 / a_p95
    a_lat, b1_lat, bh_lat = (float(summaries[x]["move_lateral_displacement_m"]) for x in ("A", "B1", "Bhalf"))
    a_yaw, b1_yaw, bh_yaw = (float(summaries[x]["move_yaw_progress_rad"]) for x in ("A", "B1", "Bhalf"))
    lateral_recovery = 1.0 - abs(bh_lat - a_lat) / max(abs(b1_lat - a_lat), 1e-12)
    yaw_recovery = 1.0 - abs(bh_yaw - a_yaw) / max(abs(b1_yaw - a_yaw), 1e-12)
    contacts = {arm: {side: contact(audits[arm], side) for side in ("left", "right")} for arm in audits}

    frozen_keys = (
        "model", "stationary_model", "recovery_model", "template", "pd_profile", "pd_kp_multiplier",
        "pd_kd_multiplier", "default_pose_profile", "command_vx_mps", "state_prediction_seconds",
        "state_qos_depth", "prepare_seconds", "stand_seconds", "move_seconds", "stop_seconds",
        "stop_controller", "heading_gain", "heading_rate_limit_radps", "action_bias_mode", "action_bias",
        "ankle_roll_common_bias", "recovery_enter_m", "recovery_exit_m", "recovery_slew_rate_per_s",
        "upper_time_scale", "upper_motion_source", "upper_stop_mode",
    )
    frozen = {key: {arm: summaries[arm].get(key) for arm in summaries} for key in frozen_keys}
    # A has no upper source and an irrelevant default upper_time_scale.  Exact
    # upper source/timing is required only between B1 and Bhalf; all backend
    # fields are required across A/B1/Bhalf.
    upper_only_keys = {"upper_time_scale", "upper_motion_source", "upper_stop_mode"}
    frozen_exact = all(
        (row["B1"] == row["Bhalf"] if key in upper_only_keys else row["A"] == row["B1"] == row["Bhalf"])
        for key, row in frozen.items()
    )
    comparison = {
        "B1_full_gate": bool(summaries["B1"]["full_gate_pass"]), "Bhalf_full_gate": bool(summaries["Bhalf"]["full_gate_pass"]),
        "Bhalf_target_excursion_rad": float(summaries["Bhalf"]["upper_target_excursion_abs_max_rad"]),
        "Bhalf_target_speed_radps": float(summaries["Bhalf"]["upper_target_speed_abs_max_radps"]),
        "A_counterfactual_half_upper_rmse_rad": a_rmse, "Bhalf_upper_rmse_rad": bh_rmse,
        "A_counterfactual_half_upper_p95_rad": a_p95, "Bhalf_upper_p95_rad": bh_p95,
        "upper_rmse_improvement_fraction": rmse_improvement, "upper_p95_improvement_fraction": p95_improvement,
        "A_lateral_m": a_lat, "B1_lateral_m": b1_lat, "Bhalf_lateral_m": bh_lat,
        "lateral_recovery_fraction": lateral_recovery,
        "A_yaw_rad": a_yaw, "B1_yaw_rad": b1_yaw, "Bhalf_yaw_rad": bh_yaw,
        "yaw_recovery_fraction": yaw_recovery,
        "A_stop_drift_m": float(summaries["A"]["stop_root_xy_drift_m"]),
        "B1_stop_drift_m": float(summaries["B1"]["stop_root_xy_drift_m"]),
        "Bhalf_stop_drift_m": float(summaries["Bhalf"]["stop_root_xy_drift_m"]),
        "contact": contacts,
    }
    gates = {
        "functional": all(bool(summaries["Bhalf"][key]) for key in ("startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")),
        "nonzero_bounded_target": threshold["nonzero_target_excursion_min_rad"] <= comparison["Bhalf_target_excursion_rad"] <= threshold["target_excursion_max_rad"] and comparison["Bhalf_target_speed_radps"] <= threshold["target_speed_max_radps"],
        "upper_rmse_improvement": rmse_improvement >= threshold["counterfactual_upper_rmse_improvement_vs_A_min_fraction"],
        "upper_p95_improvement": p95_improvement >= threshold["counterfactual_upper_p95_improvement_vs_A_min_fraction"],
        "lateral_recovery": lateral_recovery >= threshold["lateral_excess_over_A_recovery_vs_B1_min_fraction"],
        "yaw_recovery": yaw_recovery >= threshold["yaw_excess_over_A_recovery_vs_B1_min_fraction"],
        "stop_drift": comparison["Bhalf_stop_drift_m"] <= comparison["B1_stop_drift_m"] + threshold["stop_drift_increase_vs_B1_max_m"],
        "contact_agreement": all(contacts["Bhalf"][side]["agreement"] >= contacts["B1"][side]["agreement"] - threshold["contact_agreement_drop_vs_B1_max_each"] for side in ("left", "right")),
        "stance_slip": all(contacts["Bhalf"][side]["slip_p95_mps"] <= contacts["B1"][side]["slip_p95_mps"] + threshold["stance_slip_p95_increase_vs_B1_max_mps_each"] for side in ("left", "right")),
        "frozen_contract": frozen_exact,
    }
    success = all(gates.values())
    report = {
        "schema_version": "x2_half_upper_closed_phase53_v1", "scope": prereg["scope"],
        "execution": {"new_physics_episodes": 1, "retries": 0, "A_reused": True, "B1_reused": True},
        "provenance": {"prereg_sha256": sha256(PREREG), "A_sha256": sha256(A_ROLLOUT), "B1_sha256": sha256(B1_ROLLOUT), "Bhalf_sha256": sha256(BH_ROLLOUT), "Bhalf_audit_sha256": sha256(BH_AUDIT)},
        "static": {"frozen_contract_exact_except_preregistered_half_scale_excursion_slew": frozen_exact, "frozen_fields": frozen, "only_changed": prereg["only_changed"]},
        "comparison": comparison, "gates": gates,
        "evidence_boundary": {"Bhalf_alignment_pass_1e8": bool(audits["Bhalf"]["analysis"]["telemetry_alignment"]["pass_1e8"]), "contact_is_model_truth_not_hardware": True},
        "decision": {
            "status": "RETAIN_UPPER_AMPLITUDE_ROUTE" if success else "STOP_UPPER_AMPLITUDE_ROUTE",
            "conclusion": (
                "half幅值仍有实质上肢改善，并显著回收横漂/航向且保护功能/contact/slip；只保留为下一阶段候选，不自动训练。" if success else
                f"half幅值保持全功能门并改善部分contact/slip，但upper RMSE/p95只改善{rmse_improvement*100:.2f}%/{p95_improvement*100:.2f}%，横漂/yaw只回收{lateral_recovery*100:.2f}%/{yaw_recovery*100:.2f}%，均未达到预注册10%/25%门；幅值路线停止，不扫0.25/0.75。"
            ),
            "next_step": "停止upper幅值扫描；后续若继续WBT，应转向具有独立重复基线的闭环扰动建模或允许下层受限协调，而不是继续缩放上肢。" if not success else "先回报主线，不自动扩展。",
        },
        "truth_boundary": prereg["truth_boundary"],
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
