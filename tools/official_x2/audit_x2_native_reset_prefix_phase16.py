#!/usr/bin/env python3
"""Audit the single Phase16 reset-prefix capture before any physics replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RESULT = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_reset_prefix_v2_20260809"
)
DEFAULT_SOURCE = RESULT / "official_native_event_v2_reset_prefix_24s.npz"
DEFAULT_MANIFEST = DEFAULT_SOURCE.with_suffix(".manifest.json")
DEFAULT_CONTROLLER_LOG = RESULT / "logs/official_native_event_v2_reset_prefix_24s/controller.log"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_reset_prefix_phase16.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_reset_prefix_phase16.md"
REQUIRED_FULL_PREFIX_S = 4.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(source: Path, manifest_path: Path, controller_log: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    modes = arrays["mode_event_inferred_mode"].astype(str).tolist()
    mode_time = arrays["mode_event_elapsed_ns"].astype(np.float64) * 1.0e-9
    if modes != ["JOINT_DEFAULT", "RL_DEFAULT"]:
        joint_time = rl_time = float("nan")
    else:
        joint_time, rl_time = map(float, mode_time)
    snapshot_time = arrays["time_s"].astype(np.float64)
    first_after_joint_index = int(np.searchsorted(snapshot_time, joint_time, side="left"))
    first_full_time = (
        float(snapshot_time[first_after_joint_index])
        if first_after_joint_index < len(snapshot_time) else None
    )
    full_prefix = None if first_full_time is None else max(0.0, rl_time - first_full_time)
    raw_mode_prefix = float(rl_time - joint_time)
    rl_recorded = float(snapshot_time[-1] - rl_time)
    controller = controller_log.read_text(encoding="utf-8", errors="replace")
    event_time = arrays["command_event_elapsed_ns"].astype(np.int64)
    checks = {
        "npz_hash_matches_manifest": sha256(source) == manifest["npz"]["sha256"],
        "isolated_domain_215": manifest["ros_domain_id"] == 215,
        "mode_event_sequence_exact": modes == ["JOINT_DEFAULT", "RL_DEFAULT"],
        "mode_event_receipt_monotonic": bool(np.all(np.diff(mode_time) > 0.0)),
        "mode_event_buttons_exact": arrays["mode_event_buttons"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 1]],
        "command_global_index_exact": bool(np.array_equal(arrays["command_event_global_index"], np.arange(len(event_time)))),
        "command_receipt_monotonic": bool(np.all(np.diff(event_time) >= 0)),
        "snapshot_receipt_monotonic": bool(np.all(np.diff(snapshot_time) > 0.0)),
        "controller_acknowledged_sequence": (
            "Switch to JOINT_DEFAULT" in controller and "Switch to RL_DEFAULT" in controller
            and controller.index("Switch to JOINT_DEFAULT") < controller.index("Switch to RL_DEFAULT")
        ),
        "raw_mode_interval_at_least_4s": raw_mode_prefix >= REQUIRED_FULL_PREFIX_S,
        "first_complete_snapshot_after_joint_exists": first_full_time is not None,
        "complete_reference_prefix_at_least_4s": full_prefix is not None and full_prefix >= REQUIRED_FULL_PREFIX_S,
        "rl_segment_at_least_20s": rl_recorded >= 20.0,
    }
    contract_pass = all(checks.values())
    return {
        "phase": 16,
        "scope": "single domain215 capture contract audit; physics replay prohibited when prefix contract fails",
        "provenance": {
            "source": {"path": str(source), "sha256": sha256(source), "bytes": source.stat().st_size},
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "controller_log": {"path": str(controller_log), "sha256": sha256(controller_log)},
            "ros_domain_id": manifest["ros_domain_id"],
            "launch_identity": manifest["launch_identity"],
            "artifacts": manifest["artifacts"],
        },
        "capture": {
            "snapshots": int(len(snapshot_time)),
            "command_events": int(len(event_time)),
            "command_events_by_group": manifest["command_events_by_group"],
            "mode_events": manifest["control_mode_sequence"],
            "mode_header_populated": arrays["mode_event_header_populated"].astype(bool).tolist(),
            "first_command_event_s": float(event_time[0] * 1.0e-9),
            "last_command_event_s": float(event_time[-1] * 1.0e-9),
            "first_complete_snapshot_s": float(snapshot_time[0]),
            "last_complete_snapshot_s": float(snapshot_time[-1]),
            "joint_mode_event_s": joint_time,
            "rl_mode_event_s": rl_time,
            "raw_joint_to_rl_interval_s": raw_mode_prefix,
            "first_complete_snapshot_after_joint_s": first_full_time,
            "complete_reference_prefix_before_rl_s": full_prefix,
            "recorded_complete_snapshot_rl_segment_s": rl_recorded,
            "first_snapshot_command_valid_joints": int(arrays["command_valid"][0].sum()),
        },
        "pre_registered_contract": {
            "required_complete_actual_state_prefix_s": REQUIRED_FULL_PREFIX_S,
            "checks": checks,
            "passed": contract_pass,
            "failed": [name for name, value in checks.items() if not value],
        },
        "truth_boundary": {
            "mode_time_is_subscriber_receipt_not_controller_internal_transition": True,
            "mode_headers_unpopulated": bool(not np.any(arrays["mode_event_header_populated"])),
            "reference_requires_complete_actual_q_root_contact": True,
            "command_only_prefix_cannot_replace_missing_actual_reference": True,
            "source_is_official_simulation_not_real_hardware": True,
            "physics_replay_executed": False,
            "free_root_executed": False,
            "training_base_real_robot": False,
        },
        "decision": {
            "status": "PHASE16_RESET_PREFIX_CONTRACT_PASSED" if contract_pass else "PHASE16_RESET_PREFIX_CONTRACT_REJECTED_NO_PHYSICS",
            "result": (
                "The complete recorded-state prefix satisfies the frozen four-second contract."
                if contract_pass else
                f"JOINT→RL mode receipts span {raw_mode_prefix:.3f}s, but the first complete 31DOF snapshot appears at {first_full_time:.3f}s, leaving only {full_prefix:.3f}s of scoreable prefix."
            ),
            "conclusion": (
                "The reset-compatible prefix is eligible for the one-shot Phase15 replay."
                if contract_pass else
                "Mode-event capture succeeded, but this capture cannot test the reset-prefix hypothesis; missing complete actual-state reference is a capture-contract failure, not a dynamics or Any2Any result."
            ),
            "next_step": (
                "Run prescribed replay, then free only conditionally."
                if contract_pass else
                "Stop Phase16 without physics. A future separately authorized capture must wait until complete snapshots are available and then hold JOINT_DEFAULT for four additional seconds before RL_DEFAULT."
            ),
        },
    }


def render(report: dict[str, Any]) -> str:
    c = report["capture"]
    gate = report["pre_registered_contract"]
    lines = [
        "# X2 Reset-Compatible Prefix Phase16",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- v2 mode-event 扩展本身成功；但预注册的完整4秒 actual-state prefix 不成立，因此未运行 prescribed/free physics。",
        "- 本结果只否定这一次capture对Phase16假设的资格，不否定官方policy、X2动力学、Any2Any或reset-prefix思路。",
        "",
        "## 假设",
        "",
        "Phase15 的动态中段冷启动若是主要误差源，则从 JOINT_DEFAULT 后首个完整 actual q/dq/root/command context 开始，重放至少4秒稳定前缀，再进入RL段，应改善trackability。",
        "",
        "## 干预 / 对照",
        "",
        "- 唯一计划变量：Phase15动态中段冷启动 → reset-compatible JOINT_DEFAULT稳定前缀。",
        "- recorder先ready；domain215隔离；只采一次；mode-event保存callback monotonic、buttons、header和推断mode。",
        "- 合同停止门：从JOINT后首个完整snapshot到RL切换必须≥4.0s；否则不允许physics。",
        "",
        "## Capture结果",
        "",
        f"- `{c['snapshots']}` complete snapshots，`{c['command_events']}` command events；mode sequence为 `JOINT_DEFAULT → RL_DEFAULT`。",
        f"- JOINT receipt `{c['joint_mode_event_s']:.3f}s`；RL receipt `{c['rl_mode_event_s']:.3f}s`；表面间隔 `{c['raw_joint_to_rl_interval_s']:.3f}s`。",
        f"- 但首个完整31DOF snapshot直到 `{c['first_complete_snapshot_after_joint_s']:.3f}s` 才出现；到RL仅余 `{c['complete_reference_prefix_before_rl_s']:.3f}s`，低于4秒。",
        f"- RL后完整snapshot覆盖 `{c['recorded_complete_snapshot_rl_segment_s']:.3f}s`；该部分充足，但不能补回缺失的前缀reference。",
        "- controller日志确认先JOINT再RL；Joy header仍未填充，因此mode时间是subscriber receipt，不是controller内部生效时间。",
        "",
        "## Gate",
        "",
    ]
    for name, value in gate["checks"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        f"- 失败项：`{gate['failed']}`。",
        "- prescribed executed：`False`；free-root executed：`False`。",
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
    report = audit(args.source, args.manifest, args.controller_log)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
