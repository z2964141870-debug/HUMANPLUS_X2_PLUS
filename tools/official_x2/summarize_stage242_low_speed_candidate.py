#!/usr/bin/env python3
"""Freeze the official-X2 low-speed control-contract evidence."""

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


def pass_count(pattern: str) -> dict[str, int]:
    runs = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in paths(pattern)]
    return {
        "passes": sum(bool(run.get("full_gate_pass")) for run in runs),
        "runs": len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = str(args.result_root)

    medium_straight = aggregate(paths(f"{root}/stage232_stateqos1_heading050_straight_r[1-6].json"))
    medium_right = pass_count(f"{root}/stage231_stateqos1_gate_turn_right_r[123].json")
    medium_left = pass_count(f"{root}/stage231_stateqos1_gate_turn_left_r[123].json")
    low_straight = aggregate(paths(f"{root}/stage240_vx025_contract_heading_action_recovery018_straight_r[1-6].json"))
    low_right = aggregate(paths(f"{root}/stage242_vx025_contract_turn_right_bias055_r[123].json"))
    low_left = aggregate(paths(f"{root}/stage241_vx025_contract_turn_left_r[123].json"))

    recovery_steps = [
        int(json.loads(path.read_text(encoding="utf-8"))["summary"]["heading_action_recovery_steps"])
        for path in paths(f"{root}/stage240_vx025_contract_heading_action_recovery018_straight_r[1-6].json")
    ]
    funnel = {
        "vx_0p20_native": pass_count(f"{root}/stage234_stateqos1_heading050_vx020_straight_r[123].json"),
        "vx_0p20_scaled_template": pass_count(f"{root}/stage235_stateqos1_heading050_vx020_tmpl067_straight_r[123].json"),
        "vx_0p25_native": pass_count(f"{root}/stage236_stateqos1_heading050_vx025_straight_r[123].json"),
        "vx_0p25_contract_without_active_recovery": pass_count(
            f"{root}/stage238_stateqos1_vx025_policy030_tmpl083_straight_r[1-6].json"
        ),
        "vx_0p25_contract_active_recovery": {
            "passes": low_straight["functional_passes"],
            "runs": low_straight["run_count"],
            "recovery_steps_per_run": recovery_steps,
        },
    }
    low_total_passes = (
        int(low_straight["functional_passes"])
        + int(low_right["functional_passes"])
        + int(low_left["functional_passes"])
    )
    low_total_runs = (
        int(low_straight["run_count"])
        + int(low_right["run_count"])
        + int(low_left["run_count"])
    )
    medium_total_passes = int(medium_straight["functional_passes"]) + medium_right["passes"] + medium_left["passes"]
    medium_total_runs = int(medium_straight["run_count"]) + medium_right["runs"] + medium_left["runs"]

    result = {
        "domain": "aimdk_x2_v1_official_mujoco",
        "checkpoint": "stage219_s2600_actor.onnx",
        "external_low_speed_command_mps": 0.25,
        "control_contract": {
            "policy_vx_floor_mps": 0.30,
            "move_template_multiplier": 0.833333,
            "state_qos_depth": 1,
            "straight_heading_command_gain": 0.5,
            "straight_heading_action_recovery": {
                "enter_rad": 0.18,
                "exit_rad": 0.08,
                "gain": 0.75,
                "limit": 0.20,
            },
            "right_turn": {"fixed_wz_radps": 0.15, "hip_yaw_common_bias": 0.55},
            "left_turn": {"fixed_wz_radps": -0.09, "hip_yaw_common_bias": -0.50, "mirrored": True},
        },
        "selection_funnel": funnel,
        "functional_gate": {
            "low_speed": {
                "straight": {"passes": low_straight["functional_passes"], "runs": low_straight["run_count"]},
                "turn_right": {"passes": low_right["functional_passes"], "runs": low_right["run_count"]},
                "turn_left": {"passes": low_left["functional_passes"], "runs": low_left["run_count"]},
                "total": {"passes": low_total_passes, "runs": low_total_runs},
            },
            "medium_speed_frozen_baseline": {
                "straight": {"passes": medium_straight["functional_passes"], "runs": medium_straight["run_count"]},
                "turn_right": medium_right,
                "turn_left": medium_left,
                "total": {"passes": medium_total_passes, "runs": medium_total_runs},
            },
            "combined": {
                "passes": low_total_passes + medium_total_passes,
                "runs": low_total_runs + medium_total_runs,
            },
        },
        "straight_quality": {"medium": medium_straight, "low": low_straight},
        "limitations": [
            "Functional gates include startup, skill semantics, survival and stop; they do not prove visual motion quality.",
            "Official simulator exposes no validated foot wrench/contact topic, so foot slip and COP are not claimed.",
            "The low-speed contract maps an external 0.25 m/s request to the policy's trained 0.30 m/s floor; it is not a new checkpoint.",
            "Action saturation, waist saturation and high-percentile action jumps remain too large for a final deployment claim.",
        ],
        "decision": "Freeze Stage242 as the low-speed functional candidate; begin a separate quality intervention without weakening the 24/24 composite gate.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage242 官方 X2 低速控制契约甜点位",
        "",
        "## 假设",
        "",
        "Stage219 的训练速度范围是 `0.25–0.60 m/s`，但官方域在范围下沿存在相位和航向裕量不足。将外部 `0.25 m/s` 意图映射到 actor 熟悉的 `0.30 m/s`，同时按比例缩小步态模板，并只在直行偏航超过阈值时施加有界髋 yaw 修正，可能恢复低速闭环而不改 checkpoint。",
        "",
        "## 干预",
        "",
        "- 外部命令保持 `0.25 m/s`；actor 内部速度下限为 `0.30 m/s`。",
        "- 步态模板乘 `0.833333`，避免把内部速度下限直接变成更大步幅。",
        "- 直行航向动作恢复：进入/退出阈值 `0.18/0.08 rad`，增益 `0.75`，限幅 `0.20`。",
        "- 右转髋 yaw 偏置由 `0.50` 单变量调整为 `0.55`；左转保持原镜像契约。",
        "- checkpoint、PD、状态 QoS、起步和停车时序不变。",
        "",
        "## 对照",
        "",
        f"- `0.20 m/s` 原生：`{funnel['vx_0p20_native']['passes']}/{funnel['vx_0p20_native']['runs']}`；缩模板：`{funnel['vx_0p20_scaled_template']['passes']}/{funnel['vx_0p20_scaled_template']['runs']}`。这是训练范围外命令，未继续包装成可用能力。",
        f"- `0.25 m/s` 原生：`{funnel['vx_0p25_native']['passes']}/{funnel['vx_0p25_native']['runs']}`。",
        f"- 仅速度/模板契约、无有效航向动作介入：`{funnel['vx_0p25_contract_without_active_recovery']['passes']}/{funnel['vx_0p25_contract_without_active_recovery']['runs']}`。",
        "- 中速冻结基线：Stage232 直行 6 次，Stage231 左右转各 3 次。",
        "",
        "## 结果",
        "",
        f"- 低速直行 `{low_straight['functional_passes']}/{low_straight['run_count']}`；恢复器每次实际触发 `{min(recovery_steps)}–{max(recovery_steps)}` 个控制步。",
        f"- 低速右转 `{low_right['functional_passes']}/{low_right['run_count']}`，低速左转 `{low_left['functional_passes']}/{low_left['run_count']}`；低速合计 `{low_total_passes}/{low_total_runs}`。",
        f"- 中速冻结门禁仍为 `{medium_total_passes}/{medium_total_runs}`；中低速合计 `{low_total_passes + medium_total_passes}/{low_total_runs + medium_total_runs}`。",
        f"- 低速直行横漂均值 `{low_straight['lateral_abs_m_mean']:.3f} m`，倾角 p95 均值 `{low_straight['tilt_p95_rad_mean']:.3f} rad`，总/腰部动作饱和率 `{low_straight['action_saturation_fraction_mean']:.3f}/{low_straight['waist_saturation_fraction_mean']:.3f}`，停车稳定时间均值 `{low_straight['stop_settle_s_mean']:.2f} s`。",
        "",
        "## 结论",
        "",
        "低速起步—直行—左右转—停车闭环已经在官方 AimDK MuJoCo 中形成可复现功能甜点位，且不是靠放松门槛得到的；但这是控制契约修复，不是新 checkpoint，也没有解决肉眼可见的歪扭、腰部高饱和和高分位动作突跳。",
        "",
        "## 下一步",
        "",
        "1. 冻结这套 `24/24` 中低速功能矩阵，后续质量干预不得破坏它。",
        "2. 只针对腰部饱和、身体倾斜和动作 p95 突跳做结构化 A/B，不使用全局滤波。",
        "3. 质量取得净改善后，再录制不插帧、不截断失败的官方 MuJoCo 报告视频。",
        "4. 官方域没有已验证的足底力/接触话题，因此暂不声称 foot-slip、COP 已通过。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["functional_gate"], indent=2))


if __name__ == "__main__":
    main()
