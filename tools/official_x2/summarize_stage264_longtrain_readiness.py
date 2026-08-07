#!/usr/bin/env python3
"""Build the evidence-backed long-training readiness decision after Task50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=Path("reports/official_x2"))
    args = parser.parse_args()

    stage250 = read(args.report_root / "stage250_rsl_action_contract_fix_20260808.json")
    stage251 = read(args.report_root / "stage251_upper_straight_panel.json")
    stage252 = read(args.report_root / "stage252_upper_turn_panel.json")
    stage253 = read(args.report_root / "stage253_left_yaw_gain_panel.json")
    stage257 = read(args.report_root / "stage257_pd_upper_straight_panel.json")
    stage263 = read(args.report_root / "stage263_pd_upper_straight_panel.json")

    nominal_gate_passes = int(stage250["functional_gate"]["combined"]["passes"])
    nominal_gate_runs = int(stage250["functional_gate"]["combined"]["runs"])
    upper_straight_passes = sum(
        group["full_gate_passes"] for group in stage251["summary"].values()
    )
    upper_straight_runs = sum(group["runs"] for group in stage251["summary"].values())
    stage253_passes = sum(group["full_gate_passes"] for group in stage253["summary"].values())
    stage253_runs = sum(group["runs"] for group in stage253["summary"].values())
    pd_upper_passes = sum(group["full_gate_passes"] for group in stage257["summary"].values())
    pd_upper_runs = sum(group["runs"] for group in stage257["summary"].values())
    stage263_passes = sum(group["full_gate_passes"] for group in stage263["summary"].values())
    stage263_runs = sum(group["runs"] for group in stage263["summary"].values())

    result = {
        "stage": "stage264_longtrain_readiness_task50",
        "date": "2026-08-08",
        "official_domain": "AimDK X2 v1.0 MuJoCo",
        "checkpoint": "stage219_s2600_actor.onnx (frozen during all tests)",
        "readiness": {
            "official_action_contract_fixed": True,
            "nominal_backend_ready": nominal_gate_passes == nominal_gate_runs == 24,
            "nominal_upper_disturbance_ready": (
                upper_straight_passes == upper_straight_runs
                and stage253_passes == stage253_runs
            ),
            "compound_actuator_upper_ready": pd_upper_passes == pd_upper_runs,
            "long_train_unlocked": False,
        },
        "evidence": {
            "nominal_start_walk_turn_stop": f"{nominal_gate_passes}/{nominal_gate_runs}",
            "fixed_slow_fast_upper_straight": f"{upper_straight_passes}/{upper_straight_runs}",
            "left_turn_gain_2p5_fixed_slow_fast": f"{stage253_passes}/{stage253_runs}",
            "pd_0p9_1p0_1p2_x_fixed_fast": f"{pd_upper_passes}/{pd_upper_runs}",
            "aggressive_supervisor_pd_x_upper": f"{stage263_passes}/{stage263_runs}",
            "stage252_initial_turn_panel_pass": bool(stage252["panel_pass"]),
        },
        "failure_boundary": {
            "soft0p9_fixed": stage257["summary"]["soft0p9_fixed"],
            "soft0p9_fast": stage257["summary"]["soft0p9_fast"],
            "stiff1p2_fast": stage257["summary"]["stiff1p2_fast"],
            "aggressive_supervisor": stage263["summary"],
            "interpretation": (
                "PD perturbation or fast upper motion is individually manageable, but their "
                "composition is not reproducibly stable. The aggressive supervisor preserves "
                "the same aggregate 14/18 score while moving failures to other cells, so "
                "parameter-only tuning does not produce a universal 3/3 solution."
            ),
        },
        "next_training_contract": {
            "train_domains": ["PD scale 0.9-1.2", "fixed/slow/fast bounded upper motion"],
            "curriculum": [
                "nominal fixed upper",
                "nominal randomized upper",
                "PD randomized fixed upper",
                "joint PD and upper randomization",
            ],
            "frozen_contracts": [
                "RSL actor output clip and last-action feedback semantics",
                "31DOF joint order and official PD target interface",
                "14DOF upper relative target bounds",
                "existing start/walk/turn/stop thresholds",
            ],
            "unlock_gate": (
                "Every PD x upper cell must pass 3/3 straight, followed by left/right turn and "
                "8 s stop validation; no threshold relaxation and no safety fallback counted as tracking success."
            ),
        },
        "constraints": {
            "gpu_training": "pending because NVIDIA device nodes are unavailable after reboot",
            "explicit_delay_noise_in_official_mujoco": "not yet evaluated in this Task41-50 block",
            "real_robot": "not used",
        },
    }

    json_path = args.report_root / "stage264_longtrain_readiness_task50.json"
    md_path = args.report_root / "stage264_longtrain_readiness_task50.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage264 / Task50：X2 官方 MuJoCo 长训解锁裁决",
        "",
        "## 结论",
        "",
        "**标称后端已可用，但复合鲁棒性尚未达标，因此不能解锁高算力长训。**",
        "",
        "这不是回到原点：官方动作接口语义已经修正，起步/行走/左右转/停车和标称上肢扰动都已形成可复现门禁；现在剩下的是一个边界明确的联合域训练问题。",
        "",
        "## 已通过",
        "",
        f"- 标称起步、直行、左右转、停车：{nominal_gate_passes}/{nominal_gate_runs}。",
        f"- 固定/慢摆/快摆上肢直行：{upper_straight_passes}/{upper_straight_runs}。",
        f"- 修复后的左转（反馈增益 2.5）×固定/慢摆/快摆：{stage253_passes}/{stage253_runs}。",
        "- 上肢目标来自已有 X2 重定向动作；幅值≤0.12 rad，慢/快速度≤0.20/0.40 rad/s。",
        "",
        "## 未通过",
        "",
        f"- PD 0.9/1.0/1.2 × 固定/快摆臂：{pd_upper_passes}/{pd_upper_runs}。",
        "- PD=0.9、固定手臂可 3/3；PD=0.9、快摆臂为 0/3。",
        "- PD=1.2、快摆臂为 2/3。",
        f"- 更激进的横向 supervisor 完整矩阵仍为 {stage263_passes}/{stage263_runs}；它只改变失败落在哪些组合，没有提高总通过数。",
        "",
        "失败以航向越界和随后停车失稳为主，不是不会向前走。安全回退可以减少摔倒，但不能被计作完整上肢跟踪成功。",
        "",
        "## 下一训练任务",
        "",
        "1. 保持已修正的 RSL action clip/last-action 契约不变。",
        "2. 使用 PD 0.9–1.2 与固定/慢/快上肢扰动做课程式联合随机化。",
        "3. 先分别学会执行器扰动和上肢扰动，再进入联合域，避免一开始同时灌入两种难度。",
        "4. 解锁条件仍是每格 3/3；通过直行后再扩展左右转和 8 秒停车，不放宽门槛。",
        "",
        "## 当前限制",
        "",
        "本机重启后 NVIDIA 设备节点缺失，当前只能运行官方 MuJoCo CPU 门禁，不能启动新的 IsaacLab PPO 长训；官方域中的显式 delay/noise 也仍待补测。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
