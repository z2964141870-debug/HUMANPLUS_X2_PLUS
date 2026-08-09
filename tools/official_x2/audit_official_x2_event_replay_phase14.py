#!/usr/bin/env python3
"""Audit the Phase14 event-level official X2 capture without replaying physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_event_v2_20260809"
)
DEFAULT_SOURCE = RESULT_ROOT / "official_native_event_v2_20s.npz"
DEFAULT_MANIFEST = RESULT_ROOT / "official_native_event_v2_20s.manifest.json"
DEFAULT_CONTROLLER_LOG = RESULT_ROOT / "logs/official_native_event_v2_20s/controller.log"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_event_replay_phase14.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_event_replay_phase14.md"
GROUPS = ("leg", "waist", "arm", "head")
EXPECTED_CHANGED = {"leg": 12, "waist": 3, "arm": 14, "head": 0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {name: float("nan") for name in ("min", "p01", "p50", "p95", "p99", "max")}
    q = np.quantile(values * scale, [0.0, 0.01, 0.5, 0.95, 0.99, 1.0])
    return {name: float(value) for name, value in zip(("min", "p01", "p50", "p95", "p99", "max"), q)}


def dominant(counter: Counter[tuple[str, ...]], limit: int = 8) -> list[dict[str, Any]]:
    return [
        {"order": list(order), "count": int(count)}
        for order, count in counter.most_common(limit)
    ]


def audit_capture(source: Path, manifest_path: Path, controller_log: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    groups = arrays["command_event_group"].astype(str)
    receipt = arrays["command_event_receipt_monotonic_ns"].astype(np.int64)
    elapsed = arrays["command_event_elapsed_ns"].astype(np.int64)
    global_index = arrays["command_event_global_index"].astype(np.int64)
    local_index = arrays["command_event_group_index"].astype(np.int64)
    changed = arrays["command_event_changed_joint_mask"].astype(bool)
    valid = arrays["command_event_valid"].astype(bool)
    event_count = len(groups)
    event_duration_s = float((receipt[-1] - receipt[0]) * 1.0e-9)

    group_audit: dict[str, Any] = {}
    for group in GROUPS:
        mask = groups == group
        group_receipt = receipt[mask]
        group_local = local_index[mask]
        changed_count = changed[mask].sum(axis=1)
        group_audit[group] = {
            "events": int(mask.sum()),
            "rate_hz": float((mask.sum() - 1) / ((group_receipt[-1] - group_receipt[0]) * 1.0e-9)),
            "interval_ms": quantiles(np.diff(group_receipt), 1.0e-6),
            "local_index_contiguous": bool(np.array_equal(group_local, np.arange(mask.sum()))),
            "changed_joint_counts": {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(changed_count, return_counts=True))
            },
            "expected_changed_joint_count": EXPECTED_CHANGED[group],
            "changed_contract_exact": bool(np.all(changed_count == EXPECTED_CHANGED[group])),
        }

    adjacent = Counter(zip(groups[:-1].tolist(), groups[1:].tolist()))
    nonoverlap = Counter(tuple(groups[i:i + 4].tolist()) for i in range(0, event_count - 3, 4))
    block_spans = receipt[3::4] - receipt[: len(receipt[3::4]) * 4:4]

    snapshot_receipt = arrays["snapshot_receipt_monotonic_ns"].astype(np.int64)
    snapshot_group_receipt = arrays["snapshot_state_group_receipt_monotonic_ns"].astype(np.int64)
    odom_receipt = arrays["snapshot_odom_receipt_monotonic_ns"].astype(np.int64)
    imu_receipt = arrays["snapshot_imu_receipt_monotonic_ns"].astype(np.int64)
    snapshot_dt = np.diff(snapshot_receipt) * 1.0e-9

    callback_age = {}
    for index, group in enumerate(arrays["snapshot_state_group_order"].astype(str).tolist()):
        callback_age[group] = quantiles(snapshot_receipt - snapshot_group_receipt[:, index], 1.0e-6)

    event_valid_count = valid.sum(axis=1)
    snapshot_valid_count = arrays["command_valid"].astype(bool).sum(axis=1)
    root = arrays["root_pos_w_m"].astype(np.float64)
    quat = arrays["root_quat_xyzw"].astype(np.float64)
    quat_norm = np.linalg.norm(quat, axis=1)
    # xyzw tilt magnitude, invariant to yaw.
    tilt = 2.0 * np.arctan2(np.linalg.norm(quat[:, :2], axis=1), np.maximum(1.0e-12, np.sqrt(quat[:, 2] ** 2 + quat[:, 3] ** 2)))
    controller_text = controller_log.read_text(encoding="utf-8", errors="replace")

    artifact_host = {
        "official_onnx": Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/rl_model/kuailechongbai.onnx"),
        "official_control_config": Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml"),
        "official_scene": Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"),
        "recorder": REPO / "tools/official_x2/official_x2_rollout_recorder_v2.py",
    }
    artifact_checks = {
        name: {
            "host_path": str(path),
            "exists": path.is_file(),
            "capture_manifest_sha256": manifest["artifacts"][name]["sha256"],
            "current_host_sha256": sha256(path) if path.is_file() else None,
            "current_bytes_match_capture_manifest": bool(
                path.is_file()
                and sha256(path) == manifest["artifacts"][name]["sha256"]
            ),
        }
        for name, path in artifact_host.items()
    }
    runtime_artifacts = ("official_onnx", "official_control_config", "official_scene")
    runtime_artifacts_recoverable = all(
        artifact_checks[name]["current_bytes_match_capture_manifest"]
        for name in runtime_artifacts
    )
    recorder_matches = artifact_checks["recorder"][
        "current_bytes_match_capture_manifest"
    ]

    header_available = arrays["command_event_header_available"].astype(bool)
    header_populated = arrays["command_event_header_populated"].astype(bool)
    header_all_zero = bool(
        np.all(arrays["command_event_header_stamp_sec"] == 0)
        and np.all(arrays["command_event_header_stamp_nanosec"] == 0)
        and np.all(arrays["command_event_header_sequence"] == 0)
    )
    expected_keys = {
        "joint_names", "time_s", "joint_q_rad", "joint_dq_radps", "joint_effort",
        "command_q_rad", "command_dq_radps", "command_effort", "command_kp", "command_kd",
        "command_valid", "root_pos_w_m", "root_quat_xyzw", "root_lin_vel_w_mps",
        "root_ang_vel", "torso_imu_quat_xyzw", "torso_imu_ang_vel", "torso_imu_lin_acc",
        "command_event_global_index", "command_event_group", "command_event_group_index",
        "command_event_receipt_monotonic_ns", "command_event_elapsed_ns",
        "command_event_header_available", "command_event_header_populated",
        "command_event_header_stamp_sec", "command_event_header_stamp_nanosec",
        "command_event_header_sequence", "command_event_changed_joint_mask",
        "command_event_valid", "command_event_q_rad", "command_event_dq_radps",
        "command_event_effort", "command_event_kp", "command_event_kd",
    }

    checks = {
        "npz_hash_matches_manifest": sha256(source) == manifest["npz"]["sha256"],
        "capture_runtime_artifacts_recoverable_exactly": runtime_artifacts_recoverable,
        "current_recorder_source_matches_capture_manifest": recorder_matches,
        "schema_keys_complete": expected_keys.issubset(arrays),
        "manifest_counts_match": manifest["snapshots"] == len(snapshot_receipt) and manifest["command_events"] == event_count,
        "global_event_index_exact": bool(np.array_equal(global_index, np.arange(event_count))),
        "global_receipt_monotonic": bool(np.all(np.diff(receipt) >= 0)),
        "elapsed_matches_receipt_origin": bool(np.array_equal(np.diff(elapsed), np.diff(receipt))),
        "all_group_local_indices_contiguous": all(row["local_index_contiguous"] for row in group_audit.values()),
        "all_group_rates_approximately_500hz": all(490.0 <= row["rate_hz"] <= 510.0 for row in group_audit.values()),
        "changed_joint_contract_exact": all(row["changed_contract_exact"] for row in group_audit.values()),
        "snapshot_rate_approximately_50hz": 0.019 <= float(np.median(snapshot_dt)) <= 0.021,
        "snapshot_receipt_monotonic": bool(np.all(snapshot_dt > 0.0)),
        "legacy_arrays_finite": all(np.all(np.isfinite(arrays[name])) for name in (
            "joint_q_rad", "joint_dq_radps", "joint_effort", "root_pos_w_m", "root_quat_xyzw"
        )),
        "root_quaternion_normalized": bool(np.max(np.abs(quat_norm - 1.0)) < 1.0e-4),
        "controller_log_proves_joint_then_rl_mode": (
            "Switch to JOINT_DEFAULT" in controller_text and "Switch to RL_DEFAULT" in controller_text
            and controller_text.index("Switch to JOINT_DEFAULT") < controller_text.index("Switch to RL_DEFAULT")
        ),
        "header_schema_present": bool(np.all(header_available)),
        "publisher_header_populated": bool(np.all(header_populated)),
        "effective_wbt29_after_warmup": bool(np.all(event_valid_count[2:] == 29) and np.all(snapshot_valid_count == 29)),
        "active_31_joint_control_available": bool(np.all(event_valid_count == 31)),
    }
    required = [
        name for name in checks
        if name not in {
            "publisher_header_populated",
            "active_31_joint_control_available",
            "current_recorder_source_matches_capture_manifest",
        }
    ]
    passed = all(checks[name] for name in required)

    return {
        "phase": 14,
        "scope": "offline audit of one isolated official AimDK v1.0 20s capture; no replay/training/real robot",
        "inputs": {
            "source": str(source), "manifest": str(manifest_path), "controller_log": str(controller_log),
            "source_sha256": sha256(source), "manifest_ros_domain_id": manifest["ros_domain_id"],
            "control_mode": manifest["control_mode"], "launch_identity": manifest["launch_identity"],
        },
        "identity": {"artifacts": artifact_checks},
        "recorder_provenance_boundary": {
            "capture_manifest_recorder_sha256": manifest["artifacts"]["recorder"]["sha256"],
            "current_recorder_sha256": artifact_checks["recorder"]["current_host_sha256"],
            "current_source_matches_capture": recorder_matches,
            "capture_time_recorder_source_archived_separately": False,
            "exact_capture_recorder_source_reproducible": recorder_matches,
            "meaning": (
                "The NPZ/manifest and capture-time recorder digest remain immutable, but the "
                "capture-time recorder source was not archived as a separate artifact. Later "
                "readiness instrumentation changed the current source, so exact source-level "
                "recorder reproduction is unavailable; this is not capture-data corruption."
            ),
        },
        "downstream_dependency_boundary": {
            "phase15_requires_current_recorder_source_hash": False,
            "phase15_requires_npz_hash_bound_to_capture_manifest": True,
            "meaning": "Phase15 consumes the immutable event arrays and verifies the NPZ digest from this manifest; its result does not rely on the current recorder source matching the historical digest.",
        },
        "snapshots": {
            "count": len(snapshot_receipt),
            "duration_s": float((snapshot_receipt[-1] - snapshot_receipt[0]) * 1.0e-9),
            "rate_hz": float((len(snapshot_receipt) - 1) / ((snapshot_receipt[-1] - snapshot_receipt[0]) * 1.0e-9)),
            "interval_s": {**quantiles(snapshot_dt), "mean": float(snapshot_dt.mean()), "std": float(snapshot_dt.std())},
            "state_callback_age_ms": callback_age,
            "odom_callback_age_ms": quantiles(snapshot_receipt - odom_receipt, 1.0e-6),
            "imu_callback_age_ms": quantiles(snapshot_receipt - imu_receipt, 1.0e-6),
            "command_valid_joint_counts": {str(int(v)): int(c) for v, c in zip(*np.unique(snapshot_valid_count, return_counts=True))},
        },
        "events": {
            "count": event_count, "duration_s": event_duration_s,
            "aggregate_rate_hz": float((event_count - 1) / event_duration_s),
            "by_group": group_audit,
            "first_40_groups": groups[:40].tolist(),
            "dominant_adjacent_transitions": dominant(adjacent),
            "dominant_nonoverlap_four_event_orders": dominant(nonoverlap),
            "four_event_receipt_span_ms": quantiles(block_spans, 1.0e-6),
            "event_valid_joint_counts": {str(int(v)): int(c) for v, c in zip(*np.unique(event_valid_count, return_counts=True))},
        },
        "header_boundary": {
            "schema_header_available_ratio": float(header_available.mean()),
            "header_populated_ratio": float(header_populated.mean()),
            "all_stamp_and_sequence_fields_zero": header_all_zero,
            "authoritative_time": "subscriber callback receipt time.monotonic_ns",
            "not_available": "publisher/source timestamp and publisher sequence",
        },
        "representation_boundary": {
            "stored_width": 31,
            "effective_valid_after_two_warmup_events": 29,
            "head_events": int(np.count_nonzero(groups == "head")),
            "head_changed_joint_count": 0,
            "meaning": "full31 latest-map plus validity mask; official RL publisher emitted empty head arrays, so this is exact for active WBT29, not active31",
        },
        "source_trace_observation_not_replay": {
            "root_z_min_m": float(root[:, 2].min()),
            "root_z_median_m": float(np.median(root[:, 2])),
            "root_z_max_m": float(root[:, 2].max()),
            "root_z_final_m": float(root[-1, 2]),
            "tilt_rad": quantiles(tilt),
            "warning": "source rollout stability is provenance evidence only; no replay stability was tested",
        },
        "gate": {"checks": checks, "required_checks": required, "passed": passed},
        "decision": {
            "status": "PHASE14_WBT29_EVENT_CAPTURE_PASSED_WITH_RECORDER_SOURCE_DRIFT_BOUNDARY" if passed else "PHASE14_EVENT_CAPTURE_AUDIT_FAILED",
            "result": (
                "The immutable capture data passes event/schema/timing integrity and its ONNX/config/scene bytes remain recoverable. The current recorder source has drifted from the capture-time digest and the historical source was not archived separately."
                if passed else "At least one required event-capture integrity check failed."
            ),
            "conclusion": "Phase13 event timing/provenance gap is closed for active WBT29 subscriber-receipt replay, but not for publisher-time exactness or active head control.",
            "next_step": "Stop at Phase14. Do not run Phase15 until explicitly authorized; any later replay must preserve receipt order and validity masks and must not claim source-trace stability as replay stability.",
        },
    }


def render(report: dict[str, Any]) -> str:
    snapshots = report["snapshots"]
    events = report["events"]
    headers = report["header_boundary"]
    checks = report["gate"]["checks"]
    lines = [
        "# Official X2 Event-Replay Recorder Phase14",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 范围：一次隔离的 AimDK v1.0 官方 MuJoCo + 官方 ONNX 20s 采集的离线审计；未训练、未做物理 replay、未上真机。",
        "- 边界：这是官方仿真 truth，不是真机 GRF/COP；source rollout 自身稳定不能冒充 replay 稳定。",
        "",
        "## 假设",
        "",
        "若逐 callback receipt 事件、各组局部索引、full31 latest-map/valid mask、50Hz 状态快照及模型/配置/scene/recorder 身份都可验证，则 Phase13 的 active-WBT29 事件时序缺口可关闭。",
        "",
        "## 干预 / 对照",
        "",
        "- 对照：Phase13 的 50Hz latest-value 快照，缺少完整 500Hz command event stream。",
        "- 干预：独立 v2 recorder；逐 leg/waist/arm/head callback 记录 monotonic receipt、global/group index、header字段、changed mask、full31 latest-map；另存50Hz state/odom/IMU callback receipt。",
        "- 本阶段只审计采集完整性；没有拿 command 当 reference，也没有做 replay 性能结论。",
        "",
        "## 结果",
        "",
        f"- snapshot：`{snapshots['count']}` 帧，`{snapshots['duration_s']:.3f}s`，`{snapshots['rate_hz']:.3f}Hz`；dt median/p95/max = "
        f"`{snapshots['interval_s']['p50']:.6f}/{snapshots['interval_s']['p95']:.6f}/{snapshots['interval_s']['max']:.6f}s`。",
        f"- command events：`{events['count']}`，aggregate `{events['aggregate_rate_hz']:.2f}Hz`（四组合计）。",
    ]
    for group in GROUPS:
        row = events["by_group"][group]
        lines.append(
            f"- `{group}`：`{row['events']}` events，`{row['rate_hz']:.3f}Hz`，interval p50/p95/p99/max = "
            f"`{row['interval_ms']['p50']:.3f}/{row['interval_ms']['p95']:.3f}/{row['interval_ms']['p99']:.3f}/{row['interval_ms']['max']:.3f}ms`，"
            f"changed joints `{row['changed_joint_counts']}`。"
        )
    lines += [
        "",
        "## Group ordering 边界",
        "",
        "- global index 与 receipt-monotonic 顺序完整，per-group index 均从0连续；后续 replay 应以保存的 global receipt order 为准。",
        f"- 开头40条呈 `leg→waist→arm→head` 周期，但线程调度会改变相邻到达顺序；最常见 transitions：`{events['dominant_adjacent_transitions'][:5]}`。",
        "- 因此四组不是可证明的 publisher-atomic cycle，不能把固定 leg/waist/arm/head 顺序硬编码为真值。",
        "",
        "## Header 与 31/29DOF 边界",
        "",
        f"- message schema 中 header 对象存在率 `{headers['schema_header_available_ratio']:.3f}`，但 populated率 `{headers['header_populated_ratio']:.3f}`；stamp/sequence 全零：`{headers['all_stamp_and_sequence_fields_zero']}`。",
        "- 唯一权威时间是 subscriber callback 的 `time.monotonic_ns()`；无法恢复 publisher/source timestamp 或 publisher sequence。",
        f"- 数组宽度为31，但事件 warmup 后与全部snapshot都只有29个valid关节；`head` 有 `{report['representation_boundary']['head_events']}` 条空callback，changed joints恒为0。",
        "- 所以该资产可描述 official WBT29 active-control contract；不能声称获得了31个主动关节或头部控制轨迹。",
        "",
        "## 身份与 Gate",
        "",
        f"- ROS domain `{report['inputs']['manifest_ros_domain_id']}`，control mode `{report['inputs']['control_mode']}`；NPZ、ONNX、control YAML、scene 的捕获证据可验证。",
        f"- recorder capture SHA `{report['recorder_provenance_boundary']['capture_manifest_recorder_sha256']}`；current SHA `{report['recorder_provenance_boundary']['current_recorder_sha256']}`；匹配：`{report['recorder_provenance_boundary']['current_source_matches_capture']}`。",
        "- 捕获时 recorder 源码未另存快照；因此当前漂移不等于 capture 数据损坏，但不能声称 recorder 源码可精确重现。",
        "- Phase15 只依赖 manifest 绑定的 NPZ 事件数组，不依赖 current recorder SHA；其既有裁决不受这次源码漂移影响。",
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
    parser.add_argument("--controller-log", type=Path, default=DEFAULT_CONTROLLER_LOG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    report = audit_capture(args.source, args.manifest, args.controller_log)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
