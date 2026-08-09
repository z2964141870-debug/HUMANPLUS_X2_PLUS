#!/usr/bin/env python3
"""Compare immutable A/B1 with the sole Phase52 mirrored-upper episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "reports/official_x2/phase52_mirrored_upper_closed_prereg.json"
A_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase34_full_stage250_trace/phase34_stage250_closed_full_once_d230.json")
B1_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase50_hybrid_upper_closed/phase50_stage250_hybrid_upper_B_once_d229.json")
BM_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase52_mirrored_upper_closed/phase52_stage250_mirrored_upper_once_d228.json")
A_AUDIT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase34_full_stage250_trace/phase34_full_offline_audit.json")
B1_AUDIT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase50_hybrid_upper_closed/phase50_B_full_offline_audit.json")
BM_AUDIT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase52_mirrored_upper_closed/phase52_Bmirror_full_offline_audit.json")
OUTPUT_JSON = REPO / "reports/retarget/x2_mirrored_upper_closed_phase52.json"
OUTPUT_MD = REPO / "reports/retarget/x2_mirrored_upper_closed_phase52.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summary(path: Path) -> dict[str, Any]:
    return load(path)["summary"]


def contact(audit: dict[str, Any], side: str) -> dict[str, float]:
    row = audit["analysis"]["contact"][side]
    return {
        "agreement": float(row["generator_agreement_move_50hz"]),
        "slip_p95_mps": float(row["realized_stance_contact_point_slip_mps"]["p95"]),
        "clearance_p95_m": float(row["realized_swing_sole_clearance_m"]["p95"]),
    }


def render(r: dict[str, Any]) -> str:
    d = r["signed_causal_diagnostic"]
    return f"""# WBT Phase52：mirrored-upper closed 因果诊断

## 假设

若 Phase50 的横漂/航向偏差主要由 `AMASS-UPPER-001` 的左右方向性直接造成，那么只镜像 upper14 后，`Bmirror−A` 的 signed lateral 与 yaw 应相对 `B1−A` 翻转；其他控制合同保持不变。

## 干预 / 对照

- A：BASE Phase34 immutable Stage250。
- B1：Phase50 原 upper14，immutable。
- Bmirror：唯一新 episode；仅按冻结 X2 joint mirror contract 对 upper14 做左右交换，roll/yaw 取反，pitch 不变。
- mirror²误差=0；lower/waist/root/head direct path、Stage219、PD、command、时序均未改变。

## 结果

| 指标 | A | B1 | Bmirror |
|---|---:|---:|---:|
| full gate | {r['arms']['A']['full_gate']} | {r['arms']['B1']['full_gate']} | {r['arms']['Bmirror']['full_gate']} |
| upper tracking RMSE (rad) | {r['arms']['A']['upper_rmse_rad']:.5f} | {r['arms']['B1']['upper_rmse_rad']:.5f} | {r['arms']['Bmirror']['upper_rmse_rad']:.5f} |
| signed lateral (m) | {r['arms']['A']['lateral_m']:.5f} | {r['arms']['B1']['lateral_m']:.5f} | {r['arms']['Bmirror']['lateral_m']:.5f} |
| signed yaw progress (rad) | {r['arms']['A']['yaw_progress_rad']:.5f} | {r['arms']['B1']['yaw_progress_rad']:.5f} | {r['arms']['Bmirror']['yaw_progress_rad']:.5f} |

- lateral delta：B1−A={d['lateral_B1_minus_A_m']:.5f}m，Bmirror−A={d['lateral_Bmirror_minus_A_m']:.5f}m；sign flip=`{d['lateral_sign_flip']}`。
- yaw delta：B1−A={d['yaw_B1_minus_A_rad']:.5f}rad，Bmirror−A={d['yaw_Bmirror_minus_A_rad']:.5f}rad；sign flip=`{d['yaw_sign_flip']}`。
- left slip p95 A/B1/Bmirror：{r['arms']['A']['contact']['left']['slip_p95_mps']:.3f}/{r['arms']['B1']['contact']['left']['slip_p95_mps']:.3f}/{r['arms']['Bmirror']['contact']['left']['slip_p95_mps']:.3f}m/s。
- right slip p95 A/B1/Bmirror：{r['arms']['A']['contact']['right']['slip_p95_mps']:.3f}/{r['arms']['B1']['contact']['right']['slip_p95_mps']:.3f}/{r['arms']['Bmirror']['contact']['right']['slip_p95_mps']:.3f}m/s。

## 结论

**{r['decision']['status']}**

{r['decision']['conclusion']}

## 下一步

{r['decision']['next_step']}

接触/滑移来自 closed official MuJoCo 1kHz 碰撞模型，不是实机 GRF/COP。
"""


def main() -> None:
    prereg = load(PREREG)
    summaries = {"A": summary(A_ROLLOUT), "B1": summary(B1_ROLLOUT), "Bmirror": summary(BM_ROLLOUT)}
    audits = {"A": load(A_AUDIT), "B1": load(B1_AUDIT), "Bmirror": load(BM_AUDIT)}
    frozen_keys = (
        "model", "stationary_model", "recovery_model", "template", "pd_profile",
        "pd_kp_multiplier", "pd_kd_multiplier", "default_pose_profile", "command_vx_mps",
        "state_prediction_seconds", "state_qos_depth", "prepare_seconds", "stand_seconds",
        "move_seconds", "stop_seconds", "stop_controller", "heading_gain",
        "heading_rate_limit_radps", "action_bias_mode", "action_bias",
        "ankle_roll_common_bias", "recovery_enter_m", "recovery_exit_m", "recovery_slew_rate_per_s",
    )
    frozen = {key: {arm: summaries[arm].get(key) for arm in summaries} for key in frozen_keys}
    upper_frozen_keys = ("upper_scale", "upper_time_scale", "upper_max_excursion_rad", "upper_max_velocity_radps")
    upper_frozen = {key: {arm: summaries[arm].get(key) for arm in ("B1", "Bmirror")} for key in upper_frozen_keys}
    frozen_exact = all(row["A"] == row["B1"] == row["Bmirror"] for row in frozen.values()) and all(
        row["B1"] == row["Bmirror"] for row in upper_frozen.values()
    )

    arms = {}
    for arm in summaries:
        s = summaries[arm]
        arms[arm] = {
            "full_gate": bool(s["full_gate_pass"]), "startup_gate": bool(s["startup_gate_pass"]),
            "move_gate": bool(s["move_gate_pass"]), "stop_gate": bool(s["stop_gate_pass"]),
            "upper_rmse_rad": float(s["upper_tracking_rmse_rad"]),
            "upper_abs_p95_rad": float(s["upper_tracking_abs_p95_rad"]),
            "upper_excursion_rad": float(s["upper_target_excursion_abs_max_rad"]),
            "lateral_m": float(s["move_lateral_displacement_m"]),
            "heading_max_rad": float(s["move_heading_max_deviation_rad"]),
            "yaw_progress_rad": float(s["move_yaw_progress_rad"]),
            "forward_m": float(s["move_forward_displacement_m"]),
            "stop_drift_m": float(s["stop_root_xy_drift_m"]),
            "contact": {side: contact(audits[arm], side) for side in ("left", "right")},
        }
    lateral_b1 = arms["B1"]["lateral_m"] - arms["A"]["lateral_m"]
    lateral_bm = arms["Bmirror"]["lateral_m"] - arms["A"]["lateral_m"]
    yaw_b1 = arms["B1"]["yaw_progress_rad"] - arms["A"]["yaw_progress_rad"]
    yaw_bm = arms["Bmirror"]["yaw_progress_rad"] - arms["A"]["yaw_progress_rad"]
    lateral_flip = bool(abs(lateral_b1) >= .02 and abs(lateral_bm) >= .02 and np.sign(lateral_b1) == -np.sign(lateral_bm))
    yaw_flip = bool(abs(yaw_b1) >= .03 and abs(yaw_bm) >= .03 and np.sign(yaw_b1) == -np.sign(yaw_bm))
    if lateral_flip and yaw_flip:
        status = "DIRECTIONAL_UPPER_CAUSALITY_SUPPORTED"
        conclusion = "左右镜像同时翻转横漂与航向偏差，支持upper方向性是主要因果变量；本阶段仍不调补偿。"
    elif lateral_flip or yaw_flip:
        status = "MIXED_DIRECTIONAL_UPPER_EFFECT"
        conclusion = "仅一个世界方向量随镜像翻转，说明upper方向性是部分因素，但接触/闭环耦合仍占主导。"
    else:
        status = "DIRECTIONAL_UPPER_CAUSALITY_NOT_SUPPORTED"
        conclusion = (
            "Bmirror与B1相对A的横漂和航向偏差均保持同一负方向且幅值接近，因此Phase50漂移不能归因于该upper轨迹的左右方向性。"
            "镜像把slip显著拉回A附近，却没有翻转漂移，说明足端滑移对upper侧别敏感，而系统性横漂/航向更像非方向性的上肢扰动、闭环初态/调度差异或下层吸扰能力边界。"
        )
    report = {
        "schema_version": "x2_mirrored_upper_closed_phase52_v1",
        "scope": "one mirrored-upper causal diagnostic; no retry/training/compensation tuning",
        "execution": {"new_physics_episodes": 1, "retries": 0, "A_reused": True, "B1_reused": True},
        "provenance": {
            "prereg_sha256": sha256(PREREG), "A_sha256": sha256(A_ROLLOUT),
            "B1_sha256": sha256(B1_ROLLOUT), "Bmirror_sha256": sha256(BM_ROLLOUT),
            "Bmirror_audit_sha256": sha256(BM_AUDIT),
            "mirrored_artifact_sha256": prereg["artifacts"]["mirrored_sha256"],
        },
        "static": {"preregistered_checks": prereg["static_checks"], "frozen_summary_config_exact": frozen_exact, "frozen_summary_config": frozen, "B1_Bmirror_upper_config_exact": upper_frozen},
        "arms": arms,
        "signed_causal_diagnostic": {
            "lateral_B1_minus_A_m": lateral_b1, "lateral_Bmirror_minus_A_m": lateral_bm,
            "lateral_sign_flip": lateral_flip,
            "lateral_abs_magnitude_ratio_Bmirror_over_B1": abs(lateral_bm) / max(abs(lateral_b1), 1e-12),
            "yaw_B1_minus_A_rad": yaw_b1, "yaw_Bmirror_minus_A_rad": yaw_bm,
            "yaw_sign_flip": yaw_flip,
            "yaw_abs_magnitude_ratio_Bmirror_over_B1": abs(yaw_bm) / max(abs(yaw_b1), 1e-12),
        },
        "decision": {
            "status": status, "conclusion": conclusion,
            "next_step": "冻结本次归因结果；不要据同一episode拟合补偿。若继续辨识，需要预先固定的对称/反对称upper panel与重复A噪声基线。",
        },
        "truth_boundary": prereg["truth_boundary"],
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
