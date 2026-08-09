#!/usr/bin/env python3
"""Phase18 read-only diagnosis of the aborted Phase17 readiness handshake."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_reset_prefix_phase17_20260809"
)
LOGS = RESULT / "logs/official_native_event_v2_reset_ready_phase17"
DEFAULT_RECORDER_LOG = LOGS / "recorder.log"
DEFAULT_CONTROLLER_LOG = LOGS / "controller.log"
DEFAULT_SIM_LOG = LOGS / "simulator.log"
DEFAULT_INNER = REPO / "tools/official_x2/run_official_native_reset_prefix_capture_v2_inner.sh"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_ready_diagnosis_phase18.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_ready_diagnosis_phase18.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(recorder_log: Path, controller_log: Path, sim_log: Path, inner: Path) -> dict[str, Any]:
    recorder = recorder_log.read_text(encoding="utf-8", errors="replace")
    controller = controller_log.read_text(encoding="utf-8", errors="replace")
    simulator = sim_log.read_text(encoding="utf-8", errors="replace")
    launcher = inner.read_text(encoding="utf-8")
    wait_match = re.search(
        r"for _ in \$\(seq 1 (\d+)\); do test -s \"\$READY\".*?sleep ([0-9.]+); done",
        launcher,
    )
    variable_wait = re.search(r'READY_WAIT_STEPS="\$\{READY_WAIT_STEPS:-([0-9]+)\}"', launcher)
    variable_loop = re.search(
        r'for _ in \$\(seq 1 "\$READY_WAIT_STEPS"\);.*?sleep ([0-9.]+); done',
        launcher,
    )
    iterations = (
        int(wait_match.group(1)) if wait_match else
        int(variable_wait.group(1)) if variable_wait else None
    )
    interval = (
        float(wait_match.group(2)) if wait_match else
        float(variable_loop.group(1)) if variable_loop else None
    )
    wait_budget = iterations * interval if iterations is not None and interval is not None else None
    partial_artifact = RESULT / "official_native_event_v2_reset_ready_phase17.npz"
    missing_diagnostic_tokens = any(
        token in recorder for token in (
            "missing_joint", "missing joint", "odom_missing", "imu_missing",
            "state_group_missing", "snapshot_ready=false",
        )
    )
    controller_started = "Inputs (2):" in controller and "Outputs (7):" in controller
    sim_initialized = "AimRT Initialization Report End" in simulator and "Start]Configurator manager start completed" in simulator
    mode_was_not_published = "Switch to JOINT_DEFAULT" not in controller and "Switch to RL_DEFAULT" not in controller
    recorder_cleanup_error = "the given context is not valid" in recorder
    spontaneous_failure_evidence = any(
        token in (controller + simulator).lower()
        for token in ("segmentation fault", "core dumped", "terminate called", "fatal error")
    )
    specific_missing_component_identified = bool(missing_diagnostic_tokens)
    wait_only_explanation_proven = bool(
        specific_missing_component_identified
        and not spontaneous_failure_evidence
        and partial_artifact.exists()
    )
    capture_allowed = wait_only_explanation_proven
    return {
        "phase": 18,
        "scope": "read-only Phase17 log audit before any domain217 capture",
        "provenance": {
            "recorder_log": {"path": str(recorder_log), "sha256": sha256(recorder_log), "bytes": recorder_log.stat().st_size},
            "controller_log": {"path": str(controller_log), "sha256": sha256(controller_log), "bytes": controller_log.stat().st_size},
            "simulator_log": {"path": str(sim_log), "sha256": sha256(sim_log), "bytes": sim_log.stat().st_size},
            "launcher": {"path": str(inner), "sha256": sha256(inner)},
        },
        "questions": {
            "a_specific_snapshot_ready_missing_component": {
                "answer": None,
                "evidence": "recorder emitted no per-joint/group/odom/IMU readiness diagnostic and no partial NPZ was written",
            },
            "b_only_more_than_5s_needed": {
                "answer": None,
                "evidence": "Phase16 first complete snapshot at about 5.195s is confounded by JOINT already being active; Phase17 has no partial state timeline",
            },
            "c_process_health": {
                "controller_loaded": controller_started,
                "simulator_initialized": sim_initialized,
                "spontaneous_fatal_evidence": spontaneous_failure_evidence,
                "recorder_terminal_error_is_cleanup_context_invalidation": recorder_cleanup_error,
                "mode_was_not_published": mode_was_not_published,
                "answer": "controller/simulator startup looks healthy; recorder traceback is consistent with external cleanup, but this does not prove state completeness would eventually occur",
            },
        },
        "launcher_wait": {
            "iterations": iterations,
            "poll_interval_s": interval,
            "nominal_budget_s": wait_budget,
            "strict_snapshot_ready_condition_preserved": "node.buffer.snapshot_ready()" in (REPO / "tools/official_x2/official_x2_rollout_recorder_v2.py").read_text(encoding="utf-8"),
        },
        "evidence_gate": {
            "specific_missing_component_identified": specific_missing_component_identified,
            "partial_state_timeline_available": partial_artifact.exists(),
            "wait_only_explanation_proven": wait_only_explanation_proven,
            "domain217_capture_allowed": capture_allowed,
            "passed": capture_allowed,
        },
        "truth_boundary": {
            "no_domain217_capture": not capture_allowed,
            "no_physics_replay": True,
            "zero_mode_complete_state_reachability_unknown": True,
            "five_second_timeout_suspected_but_not_proven": True,
            "no_training_base_real_robot": True,
        },
        "decision": {
            "status": "PHASE18_WAIT_ONLY_CAUSE_UNPROVEN_NO_CAPTURE",
            "result": "The logs show healthy controller/simulator startup and cleanup-induced recorder termination, but contain neither the missing readiness component nor a partial state timeline.",
            "conclusion": "Evidence is insufficient to classify Phase17 as a pure wait-window failure; zero/default-mode state completeness remains unknown.",
            "next_step": "Do not run domain217. A future authorized diagnostic must add read-only per-component readiness reporting or partial-state counters before another capture; never replace snapshot_ready with sleep.",
        },
    }


def render(report: dict[str, Any]) -> str:
    q = report["questions"]
    gate = report["evidence_gate"]
    health = q["c_process_health"]
    lines = [
        "# X2 Ready Failure Diagnosis Phase18",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 只读审计 Phase17 recorder/controller/simulator 日志；未改等待窗、未运行 domain217 capture、未运行physics。",
        "",
        "## 假设",
        "",
        "只有日志能证明 Phase17 只是完整state到达晚于5秒，而不是zero/default mode永远缺某组状态时，才允许把ready等待扩到15秒并重采。",
        "",
        "## 三项审计",
        "",
        "### (a) snapshot_ready 缺什么",
        "",
        f"- 裁决：`未知`。{q['a_specific_snapshot_ready_missing_component']['evidence']}。",
        "",
        "### (b) 是否只差等待超过5秒",
        "",
        f"- 裁决：`未知`。{q['b_only_more_than_5s_needed']['evidence']}。",
        "- 5秒窗口确实可疑，但“可疑”不满足预注册的纯等待证据门。",
        "",
        "### (c) 进程健康",
        "",
        f"- controller加载：`{health['controller_loaded']}`；sim初始化：`{health['simulator_initialized']}`；自发fatal证据：`{health['spontaneous_fatal_evidence']}`。",
        f"- recorder末尾RCLError符合cleanup后context失效：`{health['recorder_terminal_error_is_cleanup_context_invalidation']}`；JOINT/RL均未发布：`{health['mode_was_not_published']}`。",
        "- 因而进程启动基本健康，但不能由此推断等待更久一定产生完整state。",
        "",
        "## 证据门",
        "",
    ]
    for name, value in gate.items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
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
    parser.add_argument("--recorder-log", type=Path, default=DEFAULT_RECORDER_LOG)
    parser.add_argument("--controller-log", type=Path, default=DEFAULT_CONTROLLER_LOG)
    parser.add_argument("--sim-log", type=Path, default=DEFAULT_SIM_LOG)
    parser.add_argument("--inner", type=Path, default=DEFAULT_INNER)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    report = audit(args.recorder_log, args.controller_log, args.sim_log, args.inner)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
