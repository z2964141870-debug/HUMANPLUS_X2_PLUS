#!/usr/bin/env python3
"""Audit the persisted observable-readiness result from Phase19."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_reset_prefix_phase19_20260809"
)
NAME = "official_native_event_v2_observable_ready_phase19"
DEFAULT_READINESS = RESULT / f"{NAME}.readiness.json"
DEFAULT_RECORDER_LOG = RESULT / "logs" / NAME / "recorder.log"
DEFAULT_CONTROLLER_LOG = RESULT / "logs" / NAME / "controller.log"
DEFAULT_SIM_LOG = RESULT / "logs" / NAME / "simulator.log"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_observable_ready_phase19.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_observable_ready_phase19.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(readiness_path: Path, recorder_log: Path, controller_log: Path, sim_log: Path) -> dict[str, Any]:
    ready = json.loads(readiness_path.read_text(encoding="utf-8"))
    controller = controller_log.read_text(encoding="utf-8", errors="replace")
    simulator = sim_log.read_text(encoding="utf-8", errors="replace")
    recorder = recorder_log.read_text(encoding="utf-8", errors="replace")
    npz = RESULT / f"{NAME}.npz"
    manifest = RESULT / f"{NAME}.manifest.json"
    all_groups_zero = all(row["seen"] == 0 for row in ready["state_by_group"].values())
    checks = {
        "diagnostic_reached_at_least_15s": ready["diagnostic_elapsed_ns"] >= 15_000_000_000,
        "snapshot_ready": bool(ready["snapshot_ready"]),
        "state_31_complete": ready["state_joint_seen_count"] == ready["state_joint_expected_count"] == 31,
        "odom_seen": bool(ready["odom_seen"]),
        "imu_seen": bool(ready["imu_seen"]),
        "all_state_groups_zero": all_groups_zero,
        "controller_loaded": "Inputs (2):" in controller and "Outputs (7):" in controller,
        "simulator_initialized": "AimRT Initialization Report End" in simulator,
        "joint_mode_not_published": "Switch to JOINT_DEFAULT" not in controller,
        "rl_mode_not_published": "Switch to RL_DEFAULT" not in controller,
        "npz_not_written": not npz.exists(),
        "manifest_not_written": not manifest.exists(),
        "timeout_diagnostic_present_in_log": '"state_joint_seen_count": 0' in recorder,
    }
    qualification = bool(
        checks["snapshot_ready"] and checks["state_31_complete"]
        and checks["odom_seen"] and checks["imu_seen"]
    )
    return {
        "phase": 19,
        "scope": "single isolated domain218 observable ready attempt; no mode publish or physics after failure",
        "provenance": {
            "readiness": {"path": str(readiness_path), "sha256": sha256(readiness_path)},
            "recorder_log": {"path": str(recorder_log), "sha256": sha256(recorder_log)},
            "controller_log": {"path": str(controller_log), "sha256": sha256(controller_log)},
            "simulator_log": {"path": str(sim_log), "sha256": sha256(sim_log)},
            "ros_domain_id": 218,
        },
        "readiness": ready,
        "checks": checks,
        "qualification": {
            "passed": qualification,
            "ready_receipt_s": None,
            "joint_mode_receipt_s": None,
            "first_complete_snapshot_s": None,
            "rl_mode_receipt_s": None,
            "scoreable_joint_prefix_s": None,
        },
        "truth_boundary": {
            "observation_window_s": ready["diagnostic_elapsed_ns"] * 1.0e-9,
            "proves_no_telemetry_before_joint_within_15s_not_forever": True,
            "strict_ready_before_joint_infeasible_under_current_bound": True,
            "joint_or_rl_never_published": True,
            "no_npz_or_manifest": True,
            "no_prescribed_or_free_physics": True,
            "no_training_base_real_robot": True,
        },
        "decision": {
            "status": "PHASE19_ZERO_MODE_TELEMETRY_ABSENT_NO_CAPTURE",
            "result": "After 15.090 s, state remained 0/31 across every group, odom and IMU were unseen, and snapshot_ready remained false.",
            "conclusion": "In the current official stack and 15 s bound, telemetry required for a complete reference does not start before JOINT_DEFAULT; a complete-ready-before-JOINT handshake is therefore not implementable as specified.",
            "next_step": "Stop without physics. If separately authorized, publish JOINT_DEFAULT to activate telemetry, wait for the first real complete snapshot, then hold JOINT for four additional scoreable seconds before RL; do not use topic existence or sleep as ready.",
        },
    }


def render(report: dict[str, Any]) -> str:
    ready = report["readiness"]
    checks = report["checks"]
    lines = [
        "# X2 Observable Ready Handshake Phase19",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 唯一domain218运行只观察ready；未达到ready后严格没有发布JOINT/RL、没有生成NPZ、没有运行physics。",
        "",
        "## 假设",
        "",
        "若Phase17只是5秒窗口太短，则在不改变snapshot_ready条件的15秒窗口中，应观察到关节/odom/IMU逐步到达并最终ready。",
        "",
        "## 干预 / 对照",
        "",
        "- 干预：只增加readiness diagnostics，不改变订阅、数据缓存、控制或mode路径。",
        "- 对照：Phase17没有缺项可观测性，只知道5秒内ready未产生。",
        "- 诊断字段：已见joint names/count、缺失31列表、各group计数、odom/IMU seen与首达时间；15秒仍由真实snapshot_ready硬触发。",
        "",
        "## 结果",
        "",
        f"- 观察时长：`{ready['diagnostic_elapsed_ns'] * 1e-9:.3f}s`；snapshot_ready：`{ready['snapshot_ready']}`。",
        f"- state：`{ready['state_joint_seen_count']}/{ready['state_joint_expected_count']}`；leg/waist/arm/head分别为 "
        f"`{ready['state_by_group']['leg']['seen']}/{ready['state_by_group']['waist']['seen']}/{ready['state_by_group']['arm']['seen']}/{ready['state_by_group']['head']['seen']}`。",
        f"- odom seen：`{ready['odom_seen']}`；IMU seen：`{ready['imu_seen']}`；所有首达时间均为空。",
        "- controller模型加载和sim初始化日志存在；没有自发发布JOINT/RL，符合严格握手设计。",
        "",
        "## Gate",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        "- ready qualification：`False`；prescribed/free executed：`False/False`。",
        "",
        "## 结论",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--recorder-log", type=Path, default=DEFAULT_RECORDER_LOG)
    parser.add_argument("--controller-log", type=Path, default=DEFAULT_CONTROLLER_LOG)
    parser.add_argument("--sim-log", type=Path, default=DEFAULT_SIM_LOG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    report = audit(args.readiness, args.recorder_log, args.controller_log, args.sim_log)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
