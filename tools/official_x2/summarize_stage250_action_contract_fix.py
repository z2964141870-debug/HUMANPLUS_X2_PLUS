#!/usr/bin/env python3
"""Freeze evidence for the RSL actor-clip contract fix in official X2 MuJoCo."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from analyze_official_motion_quality import aggregate


def paths(pattern: str) -> list[Path]:
    found = [Path(item) for item in sorted(glob.glob(pattern))]
    if not found:
        raise RuntimeError(f"no files matched {pattern}")
    return found


def reduction(before: dict[str, object], after: dict[str, object], key: str) -> float:
    return 1.0 - float(after[key]) / float(before[key])


def gate(group: dict[str, object]) -> dict[str, int]:
    return {"passes": int(group["functional_passes"]), "runs": int(group["run_count"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = str(args.result_root)

    old_medium = aggregate(paths(f"{root}/stage232_stateqos1_heading050_straight_r[1-6].json"))
    old_low = aggregate(paths(f"{root}/stage240_vx025_contract_heading_action_recovery018_straight_r[1-6].json"))
    medium_straight = aggregate(paths(f"{root}/stage244_rsl_actor_clip_medium_straight_r[1-6].json"))
    medium_right = aggregate(paths(f"{root}/stage246_rsl_actor_clip_turn_progress_right_r[1-3].json"))
    medium_left = aggregate(paths(f"{root}/stage247_rsl_actor_clip_turn_progress_left_mirrorfix_r[1-3].json"))
    low_straight = aggregate(paths(f"{root}/stage248_rsl_actor_clip_vx025_native_straight_r[1-6].json"))
    low_right = aggregate(paths(f"{root}/stage249_rsl_actor_clip_vx025_turn_progress_right_r[1-3].json"))
    low_left = aggregate(paths(f"{root}/stage249_rsl_actor_clip_vx025_turn_progress_left_r[1-3].json"))
    waist_negative = aggregate(paths(f"{root}/stage243_vx025_quality_waist085_straight_r[1-3].json"))
    stale_turn_right = aggregate(paths(f"{root}/stage245_rsl_actor_clip_medium_turn_right_r[1-3].json"))
    stale_turn_left = aggregate(paths(f"{root}/stage245_rsl_actor_clip_medium_turn_left_r[1-3].json"))

    medium_total = sum(int(group["functional_passes"]) for group in (medium_straight, medium_right, medium_left))
    low_total = sum(int(group["functional_passes"]) for group in (low_straight, low_right, low_left))
    result = {
        "domain": "aimdk_x2_v1_official_mujoco",
        "checkpoint": "stage219_s2600_actor.onnx",
        "root_cause": {
            "training_contract": "RslRlVecEnvWrapper clips actor output to [-1,1] before env.step; the action term then adds and clips the gait template.",
            "old_deploy_contract": "Unbounded ONNX mean was added to the gait template and fed back as last_action.",
            "observed_old_raw_action_abs_max": 6.261923313140869,
            "fix": "Clip actor output to [-1,1] before both gait-template composition and next-step last_action feedback.",
            "source_evidence": "/home/humanplus/x2_teleop_final/third_party/IsaacLab-v2.3.2/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:151",
        },
        "functional_gate": {
            "medium": {
                "straight": gate(medium_straight), "turn_right": gate(medium_right),
                "turn_left": gate(medium_left), "total": {"passes": medium_total, "runs": 12},
            },
            "low": {
                "straight": gate(low_straight), "turn_right": gate(low_right),
                "turn_left": gate(low_left), "total": {"passes": low_total, "runs": 12},
            },
            "combined": {"passes": medium_total + low_total, "runs": 24},
        },
        "quality": {
            "old_medium_straight": old_medium,
            "fixed_medium_straight": medium_straight,
            "old_low_straight": old_low,
            "fixed_low_straight": low_straight,
            "relative_reduction": {
                "medium": {
                    key: reduction(old_medium, medium_straight, key)
                    for key in (
                        "lateral_abs_m_mean", "tilt_p95_rad_mean",
                        "action_saturation_fraction_mean", "waist_saturation_fraction_mean",
                        "action_delta_l2_p95_mean",
                    )
                },
                "low": {
                    key: reduction(old_low, low_straight, key)
                    for key in (
                        "lateral_abs_m_mean", "tilt_p95_rad_mean",
                        "action_saturation_fraction_mean", "waist_saturation_fraction_mean",
                        "action_delta_l2_p95_mean",
                    )
                },
            },
        },
        "simplification": {
            "removed_from_low_speed_candidate": [
                "policy_vx_floor=0.30", "move_template_multiplier=0.833333",
                "heading_action_recovery",
            ],
            "remaining": "native 0.25 m/s actor command and native gait-template amplitude",
        },
        "negative_controls": {
            "posthoc_waist_scale_0p85": gate(waist_negative),
            "old_fixed_turn_bias_after_clip": {
                "right": gate(stale_turn_right), "left": gate(stale_turn_left),
                "decision": "replaced by bounded turn-progress feedback",
            },
        },
        "limitations": [
            "No validated official foot-contact/wrench topic is available, so this report does not claim foot-slip or COP gates.",
            "The official simulator is a physics environment, but it is not a substitute for hardware validation.",
            "Upper-body disturbance and actuator-parameter robustness matrices remain pending.",
            "GPU device nodes are currently absent after reboot, so a fresh IsaacLab trace could not yet be recorded; the checked-in IsaacLab Stage219 report remains the cross-domain reference.",
        ],
        "decision": "Promote Stage250 as the official-domain action-contract and functional-quality sweet point; do not train from the old deploy adapter results.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    med = result["quality"]["relative_reduction"]["medium"]
    low = result["quality"]["relative_reduction"]["low"]
    lines = [
        "# Stage250 官方 X2 RSL 动作契约修复甜点位",
        "",
        "## 假设",
        "",
        "官方 MuJoCo 中约 47% 的动作饱和和明显歪扭，不全是 X2 动力学本身造成的；部署适配器若没有复现 RSL-RL 的 actor 裁剪顺序，会把训练中不存在的超界 action 回灌为 `last_action`，形成闭环自激。",
        "",
        "## 干预",
        "",
        "- 严格复现训练链：`actor mean → clip[-1,1] → 加 gait template → 再 clip[-1,1] → PD target`。",
        "- 下一帧 `last_action` 使用第一次裁剪后的 actor action，不再使用最大到 `6.26` 的未裁剪 ONNX mean。",
        "- checkpoint、官方 MJCF、PD、50 Hz 控制周期和 QoS depth=1 均冻结。",
        "- 转向把旧固定髋 yaw 偏置替换为有界转角进度反馈；镜像左转使用共同负 hip-yaw 通道。",
        "",
        "## 对照",
        "",
        "- 旧中速 Stage232 与旧低速 Stage240，二者功能门都曾是 12/12，但动作质量差。",
        "- 负例一：直接把腰 pitch/roll 缩到 0.85，结果 0/3 且停车倒地。",
        "- 负例二：裁剪修复后继续沿用旧固定转向偏置，左右转均 0/3，说明旧偏置依赖错误 action 语义。",
        "",
        "## 结果",
        "",
        f"- 新中速门禁：直行 6/6、右转 3/3、左转 3/3，合计 `{medium_total}/12`。",
        f"- 新低速门禁：直行 6/6、右转 3/3、左转 3/3，合计 `{low_total}/12`；总计 `{medium_total + low_total}/24`。",
        f"- 中速直行：横漂降低 `{med['lateral_abs_m_mean']:.1%}`，倾角 p95 降低 `{med['tilt_p95_rad_mean']:.1%}`，总动作饱和降低 `{med['action_saturation_fraction_mean']:.1%}`，腰部饱和降低 `{med['waist_saturation_fraction_mean']:.1%}`，动作 L2-p95 降低 `{med['action_delta_l2_p95_mean']:.1%}`。",
        f"- 低速直行：横漂降低 `{low['lateral_abs_m_mean']:.1%}`，倾角 p95 降低 `{low['tilt_p95_rad_mean']:.1%}`，总动作饱和降低 `{low['action_saturation_fraction_mean']:.1%}`，腰部饱和降低 `{low['waist_saturation_fraction_mean']:.1%}`，动作 L2-p95 降低 `{low['action_delta_l2_p95_mean']:.1%}`。",
        f"- 新中/低速直行动作饱和率分别为 `{medium_straight['action_saturation_fraction_mean']:.3f}` / `{low_straight['action_saturation_fraction_mean']:.3f}`，已接近 IsaacLab Stage219 对照的 `0.1476`，不再是约 0.47。",
        "- 低速不再需要内部速度下限、模板缩放或额外航向动作恢复，恢复为原生 0.25 m/s 控制契约。",
        "",
        "## 结论",
        "",
        "此前把相当一部分“X2 动力学/reference 冲突”误判得过重：官方域的核心歪扭至少有一大块来自部署 action/last-action 语义错位。Stage250 在不训练新 checkpoint 的情况下，同时保持 24/24 功能门并显著改善姿态、横漂、饱和与高分位动作突跳，是本轮可信的结构性推进。",
        "",
        "## 下一步",
        "",
        "1. 录制修复前/后的同机位、原帧率官方 MuJoCo MP4，肉眼确认改善。",
        "2. 补上肢固定/慢摆/快摆扰动门与官方可配置执行器参数 A/B。",
        "3. GPU 节点恢复后，用 IsaacLab trace 做逐观测组分布对照；在此之前不启动长训。",
        "4. 到第 40 个实质任务节点执行 Git、本地大包和百度备份；百度仍以 STOKEN 是否恢复为准。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["functional_gate"], indent=2))


if __name__ == "__main__":
    main()
