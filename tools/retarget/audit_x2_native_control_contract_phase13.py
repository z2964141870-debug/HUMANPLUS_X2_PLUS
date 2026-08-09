#!/usr/bin/env python3
"""Audit whether Phase10 preserves an exact replayable source-control contract.

This phase is deliberately audit-first.  A 50 Hz snapshot of the latest ROS
command is useful response evidence, but it is not automatically equivalent to
the original 500 Hz command event stream.  Physics is prohibited when the
event timing/provenance contract is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy"
)
DEFAULT_SOURCE = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_strict_20260807/official_native_dance_teacher_50hz_60s.npz"
)
DEFAULT_MANIFEST = DEFAULT_SOURCE.with_suffix(".manifest.json")
DEFAULT_RECORDER = REPO / "tools/official_x2/official_x2_rollout_recorder.py"
DEFAULT_CONTROLLER = (
    OFFICIAL_ROOT / "x2_rl_deploy_controller/src/motion_control_node.cc"
)
DEFAULT_HEADER = (
    OFFICIAL_ROOT / "aimdk_msgs/interface/robot/common/msg/MessageHeader.msg"
)
DEFAULT_COMMAND_ARRAY = (
    OFFICIAL_ROOT / "aimdk_msgs/interface/robot/hal/msg/JointCommandArray.msg"
)
DEFAULT_ONNX = (
    OFFICIAL_ROOT / "x2_rl_deploy_controller/config/rl_model/kuailechongbai.onnx"
)
DEFAULT_CONTROL = OFFICIAL_ROOT / "x2_rl_deploy_controller/config/motion_control.yaml"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_control_contract_phase13.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_control_contract_phase13.md"


COMMAND_FIELDS = (
    "command_q_rad",
    "command_dq_radps",
    "command_effort",
    "command_kp",
    "command_kd",
    "command_valid",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_metrics(archive: Any, samples: int, joints: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in COMMAND_FIELDS:
        present = field in archive.files
        shape = list(archive[field].shape) if present else None
        expected = [samples, joints]
        finite = bool(np.all(np.isfinite(archive[field]))) if present and archive[field].dtype != bool else present
        result[field] = {
            "present": present,
            "shape": shape,
            "expected_shape": expected,
            "shape_exact": shape == expected,
            "finite": bool(finite),
        }
    return result


def render(report: dict[str, Any]) -> str:
    audit = report["audit"]
    checks = report["contract_gate"]["checks"]
    lines = [
        "# X2 Native Source Control Contract Phase13",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 只审计 Phase10 源 NPZ 与采集/官方发布代码；因控制事件时序合同不完整，未执行 prescribed/free 物理 A/B。",
        "- command 仍只作控制输入候选，评分 reference 始终应是 recorded actual q/root/contact；本阶段没有把 command 冒充 reference。",
        "",
        "## 假设",
        "",
        "若源 NPZ 保存了 source policy 真正发布的逐控制事件 command、Kp/Kd、关节顺序、原始消息时间戳/sequence 和可验证的模型配置身份，才允许把 Phase12 的 synthetic PD 换成 recorded control contract 做单变量 A/B。",
        "",
        "## 干预 / 对照",
        "",
        "- 对照：Phase10 源 NPZ/manifest 与原始采集器实现。",
        "- 干预：无；只读审计 shapes、finite、ordering、sample timing、ROS topic provenance、publisher/recorder语义。",
        "- 停止条件：缺少原始 command event timestamp/sequence、组间原子同步或 source policy/config identity 任一项，即不进入物理。",
        "",
        "## 已保存内容",
        "",
        f"- `{audit['samples']}` 帧 × `{audit['joint_count']}` 关节；snapshot time dt mean/std/min/max = "
        f"`{audit['snapshot_time']['dt_mean_s']:.6f}/{audit['snapshot_time']['dt_std_s']:.6f}/"
        f"{audit['snapshot_time']['dt_min_s']:.6f}/{audit['snapshot_time']['dt_max_s']:.6f}s`。",
        f"- command valid fraction `{audit['command_values']['valid_fraction']:.3f}`；Kp unique `{audit['command_values']['kp_unique']}`；Kd unique `{audit['command_values']['kd_unique']}`。",
        f"- archive joint order exact recorder contract：`{checks['archive_joint_order_exact_recorder']}`；command q/dq/effort/Kp/Kd/valid 形状与有限性：`{checks['all_command_arrays_complete']}`。",
        f"- official topic provenance：`{checks['values_are_sampled_from_official_command_topics']}`；官方 C++ 会把 position/stiffness/damping 写入这些 topic：`{checks['official_publisher_emits_control_fields']}`。",
        "",
        "## 缺失的可重放合同",
        "",
        "- 官方 publisher 500Hz；采集器只用独立 50Hz timer 读取四个 topic 的 latest-value 字典，因此 NPZ 是异步快照，不是完整 500Hz command event stream。",
        "- `JointCommandArray` 定义含 stamp/sequence，但官方 `createCommand()` 没有填 header；采集器也未保存 header、callback receipt time 或每组 sequence。",
        "- arm/leg/waist/head 四组独立 callback 更新同一字典，NPZ 未保存组级到达时刻，无法证明同一行31关节来自同一 publish cycle。",
        "- manifest 有人类可读 source 标签，但没有逐帧 control mode、ONNX hash、control YAML hash、process/launch identity；不能仅凭标签完整证明每一帧来自指定 policy/config。",
        "",
        "## Gate",
        "",
    ]
    for name, value in checks.items():
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recorder", type=Path, default=DEFAULT_RECORDER)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    archive = np.load(args.source, allow_pickle=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    recorder_text = args.recorder.read_text(encoding="utf-8")
    controller_text = args.controller.read_text(encoding="utf-8")
    header_text = DEFAULT_HEADER.read_text(encoding="utf-8")
    command_array_text = DEFAULT_COMMAND_ARRAY.read_text(encoding="utf-8")
    names = [str(value) for value in archive["joint_names"].tolist()]
    samples, joint_count = archive["command_q_rad"].shape
    fields = field_metrics(archive, samples, joint_count)
    times = np.asarray(archive["time_s"], dtype=np.float64)
    dt = np.diff(times)
    valid = np.asarray(archive["command_valid"], dtype=bool)
    kp = np.asarray(archive["command_kp"], dtype=np.float64)
    kd = np.asarray(archive["command_kd"], dtype=np.float64)

    recorder_joint_literal = tuple(names) == tuple(
        # Importing the recorder requires ROS; parse its explicit JOINTS tuple
        # by validating all names and their textual order instead.
        sorted(names, key=lambda value: recorder_text.index(f'"{value}"'))
    )
    source_hash_ok = manifest["sha256"] == sha256(args.source)
    topic_literals = all(
        f'/aima/hal/joint/{{area}}/command' in recorder_text
        for _ in [0]
    ) and 'for area in ("leg", "waist", "arm", "head")' in recorder_text
    publisher_fields = all(
        token in controller_text
        for token in (
            "ros_command.position  = command.position",
            "ros_command.damping   = command.damping",
            "ros_command.stiffness = command.stiffness",
        )
    )
    header_defined = all(token in header_text for token in ("Time stamp", "uint32 sequence"))
    command_has_header = "MessageHeader header" in command_array_text
    create_start = controller_text.index("MotionControlNode::createCommand")
    create_body = controller_text[create_start:create_start + 1600]
    publisher_populates_header = "msg.header" in create_body or "header." in create_body
    recorder_captures_header = any(
        token in recorder_text for token in ("msg.header", "header.stamp", "header.sequence")
    )
    recorder_captures_callback_time = "command_receive" in recorder_text or "command_timestamp" in recorder_text
    latest_cache_semantics = "self.command[joint.name]" in recorder_text and "self.create_timer(0.02, self.sample)" in recorder_text
    official_500hz = "kControlPeriod        = std::chrono::milliseconds(2)" in controller_text
    group_atomic_metadata = any(
        key in archive.files for key in ("command_group_time_s", "command_group_sequence", "command_header_stamp")
    )
    identity_fields = set(("policy_sha256", "control_config_sha256", "control_mode", "command_event_time_s"))
    identity_complete = identity_fields.issubset(archive.files) or identity_fields.issubset(manifest)

    checks = {
        "source_manifest_hash_matches": source_hash_ok,
        "all_command_arrays_complete": all(
            value["present"] and value["shape_exact"] and value["finite"]
            for value in fields.values()
        ),
        "all_joint_commands_valid": bool(np.all(valid)),
        "archive_joint_order_exact_recorder": recorder_joint_literal and len(names) == 31 and len(set(names)) == 31,
        "snapshot_timestamps_monotonic": bool(np.all(dt > 0.0)),
        "values_are_sampled_from_official_command_topics": topic_literals,
        "official_publisher_emits_control_fields": publisher_fields,
        "original_command_event_timestamp_preserved": recorder_captures_header or recorder_captures_callback_time,
        "original_command_sequence_preserved": recorder_captures_header,
        "four_group_atomic_cycle_preserved": group_atomic_metadata,
        "source_policy_and_config_identity_hashes_preserved": identity_complete,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    complete = all(checks.values())
    audit = {
        "samples": int(samples),
        "joint_count": int(joint_count),
        "joint_order": names,
        "fields": fields,
        "snapshot_time": {
            "semantics": "recorder 50Hz sample timer, not original command publication time",
            "dt_mean_s": float(np.mean(dt)),
            "dt_std_s": float(np.std(dt)),
            "dt_min_s": float(np.min(dt)),
            "dt_max_s": float(np.max(dt)),
        },
        "command_values": {
            "valid_fraction": float(np.mean(valid)),
            "kp_unique": np.unique(kp).tolist(),
            "kd_unique": np.unique(kd).tolist(),
            "kp_change_count": int(np.count_nonzero(np.diff(kp, axis=0))),
            "kd_change_count": int(np.count_nonzero(np.diff(kd, axis=0))),
            "command_dq_all_zero": bool(np.all(archive["command_dq_radps"] == 0.0)),
            "command_effort_all_zero": bool(np.all(archive["command_effort"] == 0.0)),
        },
        "event_contract": {
            "official_publisher_hz": 500 if official_500hz else None,
            "recorder_snapshot_hz": 50,
            "latest_value_cache": latest_cache_semantics,
            "message_header_defines_stamp_and_sequence": header_defined and command_has_header,
            "official_createCommand_populates_header": publisher_populates_header,
            "recorder_saves_header_or_callback_receipt_time": recorder_captures_header or recorder_captures_callback_time,
            "four_group_arrival_metadata_saved": group_atomic_metadata,
            "policy_config_identity_saved": identity_complete,
        },
        "source_policy_evidence": {
            "manifest_label": str(archive["source"].item()),
            "direct_subscription_to_official_output_topics": topic_literals,
            "controller_maps_policy_action_to_position_and_config_kp_kd": all(
                token in controller_text
                for token in (
                    "target_pos = action_vec_(i) * cfg_->rl_config.action_scale + default_pos",
                    "cmd.stiffness = kp",
                    "cmd.damping   = kd",
                )
            ),
            "strength": "strong value provenance, insufficient exact per-event timing/config identity",
        },
    }
    report = {
        "schema_version": "x2_native_control_contract_phase13_v1",
        "provenance": {
            "source_npz": {"path": str(args.source), "sha256": sha256(args.source)},
            "source_manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "recorder": {"path": str(args.recorder), "sha256": sha256(args.recorder)},
            "official_controller": {"path": str(args.controller), "sha256": sha256(args.controller)},
            "official_message_header": {"path": str(DEFAULT_HEADER), "sha256": sha256(DEFAULT_HEADER)},
            "official_onnx_current_file": {"path": str(DEFAULT_ONNX), "sha256": sha256(DEFAULT_ONNX)},
            "official_control_current_file": {"path": str(DEFAULT_CONTROL), "sha256": sha256(DEFAULT_CONTROL)},
        },
        "truth_boundary": {
            "command_is_control_input_candidate_not_reference": True,
            "source_trace_is_official_simulation_not_hardware": True,
            "captured_values_are_ros_command_topic_snapshots": True,
            "snapshots_are_not_exact_original_500hz_event_stream": True,
            "physics_prescribed_free_ab_executed": False,
            "training_ppo_lora_gold_source_base_real_robot_git_baidu": False,
        },
        "hypothesis": "Phase10 command/Kp/Kd may form an exact source actuator-control replay contract.",
        "intervention": "Read-only schema, timestamp, source-publisher and recorder-semantics audit before any physics.",
        "control": "Phase10 NPZ/manifest and immutable official/recorder source code.",
        "audit": audit,
        "contract_gate": {
            "required_all": True,
            "checks": checks,
            "pass": complete,
            "failed": [key for key, value in checks.items() if not value],
        },
        "decision": {
            "status": "PHASE13_CONTROL_CONTRACT_COMPLETE" if complete else "PHASE13_CONTROL_CONTRACT_INCOMPLETE_STOP",
            "result": (
                "源 control contract 完整，可进入一次 prescribed/free 单变量 A/B。"
                if complete else
                "q/dq/effort/Kp/Kd 值与关节顺序完整且来自官方 command topics，但缺少原始500Hz事件时间戳/sequence、四组原子同步和policy/config身份hash；按规则未运行物理。"
            ),
            "conclusion": (
                "可将 recorded control contract 视为source dynamic teacher控制输入。"
                if complete else
                "现有NPZ适合command-to-state响应证据，不能被提升为可精确复现source闭环的控制事件流；这不否定Gold、Any2Any或官方policy。"
            ),
            "next_step": (
                "复用Phase12进行唯一一次control A/B。" if complete else
                "若未来可重录，保存四个topic每条消息的callback monotonic time、header/sequence、group、mode、ONNX/config hash；在此之前不做Phase13 physics。"
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
