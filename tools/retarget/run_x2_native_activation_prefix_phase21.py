#!/usr/bin/env python3
"""Phase21 qualified activation-prefix replay using the frozen Phase15 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
for value in (REPO / "tools", REPO / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import retarget.run_x2_forefoot_official_physics_screen as physics
import retarget.run_x2_native_event_replay_phase15 as phase15
import retarget.run_x2_native_gold_trackability_phase12 as phase12


RESULT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_activation_prefix_phase21_20260809"
)
NAME = "official_native_event_v2_activation_prefix_phase21"
DEFAULT_SOURCE = RESULT / f"{NAME}.npz"
DEFAULT_MANIFEST = RESULT / f"{NAME}.manifest.json"
DEFAULT_CONTROLLER = RESULT / "logs" / NAME / "controller.log"
DEFAULT_PHASE15 = REPO / "reports/retarget/x2_native_event_replay_phase15.json"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_activation_prefix_phase21.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_activation_prefix_phase21.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualification(source: Path, manifest_path: Path, controller_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    controller = controller_path.read_text(encoding="utf-8", errors="replace")
    with np.load(source, allow_pickle=False) as archive:
        times = archive["time_s"].astype(np.float64)
        modes = archive["mode_event_inferred_mode"].astype(str).tolist()
        mode_times = archive["mode_event_elapsed_ns"].astype(np.float64) * 1.0e-9
        valid = archive["command_valid"].astype(bool)
        subscription_elapsed = float(
            (archive["subscription_ready_monotonic_ns"] - archive["capture_start_monotonic_ns"]) * 1.0e-9
        )
    joint_time, rl_time = map(float, mode_times) if modes == ["JOINT_DEFAULT", "RL_DEFAULT"] else (float("nan"), float("nan"))
    start_index = int(np.searchsorted(times, joint_time, side="left"))
    first_complete = float(times[start_index]) if start_index < len(times) else None
    prefix = None if first_complete is None else rl_time - first_complete
    rl_coverage = float(times[-1] - rl_time)
    checks = {
        "npz_hash_matches_manifest": sha256(source) == manifest["npz"]["sha256"],
        "domain220": manifest["ros_domain_id"] == 220,
        "mode_sequence_exact": modes == ["JOINT_DEFAULT", "RL_DEFAULT"],
        "subscription_before_joint": subscription_elapsed < joint_time,
        "controller_ack_joint_then_rl": (
            "Switch to JOINT_DEFAULT" in controller and "Switch to RL_DEFAULT" in controller
            and controller.index("Switch to JOINT_DEFAULT") < controller.index("Switch to RL_DEFAULT")
        ),
        "first_complete_snapshot_after_joint": first_complete is not None and first_complete >= joint_time,
        "first_complete_command_context_31": first_complete is not None and int(valid[start_index].sum()) == 31,
        "scoreable_joint_prefix_at_least_4s": prefix is not None and prefix >= 4.0,
        "rl_segment_at_least_20s": rl_coverage >= 20.0,
    }
    return {
        "subscription_ready_s": subscription_elapsed,
        "joint_mode_s": joint_time,
        "first_complete_snapshot_s": first_complete,
        "first_complete_snapshot_index": start_index,
        "four_second_prefix_achieved_s": None if first_complete is None else first_complete + 4.0,
        "rl_mode_s": rl_time,
        "scoreable_joint_prefix_s": prefix,
        "rl_segment_s": rl_coverage,
        "checks": checks,
        "passed": all(checks.values()),
        "failed": [name for name, value in checks.items() if not value],
    }


def render(report: dict[str, Any]) -> str:
    q = report["capture_qualification"]
    replay = report.get("replay", {})
    prescribed = replay.get(phase12.MODE_PRESCRIBED)
    free = replay.get(phase12.MODE_FREE)
    lines = [
        "# X2 Activation Prefix Replay Phase21",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 唯一变量是激活/初始化顺序；模型、PD、reference定义、Phase15 gate均未改变。",
        "- reference始终为recorded actual q/root/model-contact；command event只作control input。",
        "",
        "## 假设",
        "",
        "Phase15 prescribed误差若主要来自动态中段冷启动，则JOINT激活telemetry后，从首个完整snapshot重放至少4秒稳定前缀，再进入RL段，应通过相同trackability门。",
        "",
        "## Capture资格",
        "",
        f"- subscription-ready `{q['subscription_ready_s']:.3f}s` → JOINT ack/receipt `{q['joint_mode_s']:.3f}s` → first complete `{q['first_complete_snapshot_s']:.3f}s`。",
        f"- complete+4s 达成 `{q['four_second_prefix_achieved_s']:.3f}s`；RL ack/receipt `{q['rl_mode_s']:.3f}s`；可评分prefix `{q['scoreable_joint_prefix_s']:.3f}s`；RL覆盖 `{q['rl_segment_s']:.3f}s`。",
        f"- qualification：`{q['passed']}`，失败项 `{q['failed']}`。",
        "",
        "## Replay结果",
        "",
    ]
    if prescribed is None:
        lines.append("- capture资格失败，未运行physics。")
    else:
        lines += [
            "| mode | survival | q RMSE(all31/active29/head2) | body rel-pos p95 | body ori p95 | contact | SS(ref/real) | slip p95 | torque sat |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [prescribed] + ([free] if free is not None else []):
            lines.append(
                f"| {row['mode']} | {row['simulated_duration_s']:.3f}/{row['reference_duration_s']:.3f}s | "
                f"{row['q_tracking']['rmse_rad']:.4f}/{row['q_tracking']['active29_rmse_rad']:.4f}/{row['q_tracking']['head2_rmse_rad']:.4f} | "
                f"{row['body_tracking']['root_relative_position_p95_max_m'][0]:.4f}m | {row['body_tracking']['orientation_p95_max_rad'][0]:.4f} | "
                f"{row['contact']['agreement']['mean']:.3f} | {row['contact']['reference_phase']['single_support_ratio']:.3f}/{row['contact']['realized_phase']['single_support_ratio']:.3f} | "
                f"{row['slip_p95_max_mps'][0]:.3f}m/s | {row['torque']['saturation_fraction']:.4f} |"
            )
        lines += [
            "",
            f"- prescribed gate：`{report['gate']['prescribed']['pass']}`，失败项 `{report['gate']['prescribed']['failed']}`。",
            f"- free-root executed：`{free is not None}`" + ("。" if free is not None else "；prescribed失败即停。"),
        ]
    lines += [
        "",
        "## 边界",
        "",
        "- JOINT阶段真实active31 context被保留；RL阶段空head callback不清除最后source head command，未伪造29/31。",
        "- subscriber receipt不是publisher timestamp；solver/contact warmstart仍不可得；source稳定不冒充replay稳定。",
        "- contact为模型/官方MuJoCo碰撞，不是实机GRF/COP/wrench。",
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--phase15", type=Path, default=DEFAULT_PHASE15)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    qualify = qualification(args.source, args.manifest, args.controller)
    replay: dict[str, Any] = {}
    prescribed_gate = free_gate = None
    source_metrics = None
    initialization = None
    if qualify["passed"]:
        reference, events = phase15.build_reference(
            args.source, start_snapshot_index=qualify["first_complete_snapshot_index"]
        )
        model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
        source_metrics = phase12.source_kinematics(reference, model)
        prescribed = phase15.simulate_event_replay(
            reference, events, phase12.MODE_PRESCRIBED,
            expected_active_joint_count=31, allow_head_active=True,
        )
        replay[phase12.MODE_PRESCRIBED] = prescribed
        prescribed_gate = phase12.prescribed_gate(prescribed, source_metrics)
        initialization = prescribed["initialization"]
        if prescribed_gate["pass"]:
            free = phase15.simulate_event_replay(
                reference, events, phase12.MODE_FREE,
                expected_active_joint_count=31, allow_head_active=True,
            )
            replay[phase12.MODE_FREE] = free
            free_gate = phase12.free_gate(free)

    if not qualify["passed"]:
        status = "PHASE21_CAPTURE_QUALIFICATION_REJECTED_NO_PHYSICS"
        result = "activation-prefix capture未通过资格门，未运行physics。"
        conclusion = "本次不能检验reset-compatible prefix。"
        next_step = "停止，不重采、不调参。"
    elif prescribed_gate is not None and not prescribed_gate["pass"]:
        source_failed = any(
            name in prescribed_gate["failed"]
            for name in ("source_joint_step", "source_joint_velocity", "target_limits_bounded")
        )
        if source_failed:
            status = "PHASE21_SOURCE_PREFIX_REJECTED_PRESCRIBED_FAILED"
            result = "时间资格通过，但captured actual reference自身违反冻结的连续性/速度/关节限位门，且prescribed失败；未运行free-root。"
            conclusion = "额外JOINT前缀不是合格的稳定reference，因此本次不能裁决动态中段冷启动假设。"
            next_step = "停止；先解释JOINT hold期间source失稳，不调replay增益/时移、不训练。"
        else:
            status = "PHASE21_ACTIVATION_PREFIX_PRESCRIBED_REJECTED"
            result = "合格reset-compatible prefix仍未通过同一prescribed trackability门；按规则未运行free-root。"
            conclusion = "动态中段冷启动不是剩余body/slip误差的充分解释；初始化修复有效性必须按具体指标判断。"
            next_step = "停止，不扫时移/增益/滤波、不训练。"
    elif free_gate is not None and free_gate["pass"]:
        status = "PHASE21_ACTIVATION_PREFIX_PRESCRIBED_AND_FREE_PASSED"
        result = "激活prefix同时通过prescribed与一次free-root门。"
        conclusion = "该source control replay可复现；仍不证明Any2Any。"
        next_step = "冻结sanity证据，本阶段不训练。"
    else:
        status = "PHASE21_ACTIVATION_PREFIX_PRESCRIBED_PASSED_FREE_REJECTED"
        result = "prescribed通过但一次free-root失败。"
        conclusion = "控制trackability成立，裸replay平衡不成立；不否定Any2Any。"
        next_step = "停止，不调参。"

    historical = json.loads(args.phase15.read_text(encoding="utf-8"))
    report = {
        "schema_version": "x2_native_activation_prefix_phase21_v1",
        "provenance": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "controller": {"path": str(args.controller), "sha256": sha256(args.controller)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": sha256(physics.DEFAULT_CONTROL)},
        },
        "hypothesis": "A qualified reset-compatible JOINT prefix should improve Phase15 if dynamic mid-clip cold start was the dominant error.",
        "intervention": "Subscription-ready then JOINT activation; start at first complete snapshot, preserve >=4s JOINT prefix, then recorded RL events.",
        "control": "Phase15 dynamic mid-segment start under the same prescribed gate; independent capture means historical, not frame-matched, comparison.",
        "pre_registered_gates": phase12.GATES,
        "capture_qualification": qualify,
        "source_reference": source_metrics,
        "initialization": initialization,
        "historical_phase15": {
            "status": historical["decision"]["status"],
            "prescribed": historical["replay"][phase12.MODE_PRESCRIBED],
            "not_frame_matched": True,
        },
        "replay": replay,
        "gate": {"prescribed": prescribed_gate, "free": free_gate},
        "truth_boundary": {
            "actual_state_is_reference_command_is_control": True,
            "active31_context_is_source_recorded": True,
            "subscriber_receipt_not_publisher_time": True,
            "contact_not_hardware_force_truth": True,
            "source_stability_not_replay_stability": True,
            "free_failure_does_not_reject_any2any": True,
            "no_training_base_real_robot": True,
        },
        "decision": {"status": status, "result": result, "conclusion": conclusion, "next_step": next_step},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
