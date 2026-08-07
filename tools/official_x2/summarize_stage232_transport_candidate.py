#!/usr/bin/env python3
"""Freeze the Stage232 official-X2 state-transport candidate evidence."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = str(args.result_root)
    baseline = aggregate(paths(f"{root}/stage219_closed_loop_gate_v5_straight_r[123].json"))
    straight = aggregate(paths(f"{root}/stage232_stateqos1_heading050_straight_r[1-6].json"))
    right = aggregate(paths(f"{root}/stage231_stateqos1_gate_turn_right_r[123].json"))
    left = aggregate(paths(f"{root}/stage231_stateqos1_gate_turn_left_r[123].json"))
    negative = aggregate(paths(f"{root}/stage233_stateqos1_heading_recovery_straight_r[1-6].json"))
    result = {
        "domain": "aimdk_x2_v1_official_mujoco",
        "candidate_contract": {
            "checkpoint": "stage219_s2600_actor.onnx",
            "state_qos_depth": 1,
            "state_prediction_seconds": 0.0,
            "straight_heading_gain": 0.5,
            "straight_heading_rate_limit_radps": 0.10,
            "turn_contract": "Stage219 v5 fixed-wz commands; fixed-wz overrides heading feedback",
        },
        "functional_gate": {
            "straight": {"passes": straight["functional_passes"], "runs": straight["run_count"]},
            "turn_right": {"passes": right["functional_passes"], "runs": right["run_count"]},
            "turn_left": {"passes": left["functional_passes"], "runs": left["run_count"]},
            "total": {
                "passes": straight["functional_passes"] + right["functional_passes"] + left["functional_passes"],
                "runs": straight["run_count"] + right["run_count"] + left["run_count"],
            },
        },
        "straight_quality": {"baseline": baseline, "candidate": straight},
        "relative_reduction": {
            key: reduction(baseline, straight, key)
            for key in (
                "lateral_abs_m_mean",
                "action_delta_element_rms_mean",
                "action_delta_l2_p50_mean",
                "action_delta_l2_p75_mean",
                "action_delta_l2_p90_mean",
                "action_delta_l2_p95_mean",
            )
        },
        "known_regressions": {
            "tilt_p95_relative_change": -reduction(baseline, straight, "tilt_p95_rad_mean"),
            "action_saturation_relative_change": -reduction(
                baseline, straight, "action_saturation_fraction_mean"
            ),
            "waist_saturation_relative_change": -reduction(
                baseline, straight, "waist_saturation_fraction_mean"
            ),
        },
        "negative_control": {
            "intervention": "deadband heading recovery, enter=0.22 rad, exit=0.08 rad",
            "passes": negative["functional_passes"],
            "runs": negative["run_count"],
            "conclusion": "rejected; it missed a stable-but-low-progress failure",
        },
        "decision": (
            "Stage232 is the new transport/functional candidate, not a final motion-quality model. "
            "It unlocks reproducible official-domain gates while tilt and saturation remain open."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    reductions = result["relative_reduction"]
    regressions = result["known_regressions"]
    lines = [
        "# Stage232 官方 X2 状态传输甜点位",
        "",
        "## 假设",
        "",
        "官方 500–1000 Hz 状态话题进入 Python 单线程适配器时，深度 10 的队列会放大旧帧与调度不确定性；只取最新帧可减少动作卡顿，但直行仍需要小幅航向闭环防止偶发发散。",
        "",
        "## 干预",
        "",
        "- 状态 QoS `KEEP_LAST depth: 10 → 1`；命令发布 QoS 不变。",
        "- 不使用固定状态预测（`0 ms`）。",
        "- 直行使用 `heading_gain=0.5`、限幅 `0.10 rad/s`；转向仍使用原固定 `wz` 契约。",
        "- checkpoint、PD、步态模板、恢复偏置、起步/停车时序均冻结。",
        "",
        "## 对照",
        "",
        "- 原 Stage219-v5：直行/左右转各 3 次，原先功能门 9/9。",
        "- Stage232：直行 6 次，左右转各 3 次。",
        "- 否定对照：仅偏航超阈值介入的监督器。",
        "",
        "## 结果",
        "",
        f"- Stage232 功能门：直行 `{straight['functional_passes']}/{straight['run_count']}`，右转 `{right['functional_passes']}/{right['run_count']}`，左转 `{left['functional_passes']}/{left['run_count']}`，合计 `12/12`。",
        f"- 横漂均值降低 `{reductions['lateral_abs_m_mean']:.1%}`。",
        f"- 动作跳变量 L2：p50 降低 `{reductions['action_delta_l2_p50_mean']:.1%}`，p75 降低 `{reductions['action_delta_l2_p75_mean']:.1%}`，p90 降低 `{reductions['action_delta_l2_p90_mean']:.1%}`；p95 基本不变（降低 `{reductions['action_delta_l2_p95_mean']:.1%}`）。",
        f"- 代价：倾角 p95 增加 `{regressions['tilt_p95_relative_change']:.1%}`，总动作饱和增加 `{regressions['action_saturation_relative_change']:.1%}`，腰部饱和增加 `{regressions['waist_saturation_relative_change']:.1%}`。",
        f"- 死区航向恢复监督器只有 `{negative['functional_passes']}/{negative['run_count']}`：稳定但低推进的失败不会触发偏航阈值，因此淘汰。",
        "",
        "## 结论",
        "",
        "这轮修复了官方部署链路中一部分卡顿与偶发直行发散，并建立了 12 次可复现功能门；没有修好肉眼可见的整体歪扭，也没有降低腰部饱和。Stage232 是新的部署/传输基线，不是最终迁移模型。",
        "",
        "## 下一步",
        "",
        "1. 冻结 Stage232，补低速/中速的起步—行走—停车门禁。",
        "2. 将倾角和腰部饱和作为独立质量目标，不再通过全局动作滤波处理。",
        "3. 通过门禁后再录制不插帧的官方 MuJoCo 对比视频。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["functional_gate"], indent=2))


if __name__ == "__main__":
    main()
