#!/usr/bin/env python3
"""Phase31: frozen Phase29+30 transfer to three fixed held-out motions.

No held-out metric is used to select an action, weight, iteration count, or
time factor.  Each Phase28 official-v1 source is passed through the unchanged
Phase29 eight-iteration full-trajectory repair and unchanged Phase30 1.46x
retiming, then audited by the same offline Bronze/Silver gates.  This performs
no physics integration, policy forward, PPO, or robot operation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mujoco


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29
import retarget.run_x2_wbt_time_dilation_phase30 as phase30


MOTION_IDS = ("AMASS-TURN-R-001", "PHUMA-RAISE-R-001", "AMASS-KICK-R-001")
OUTPUT_ROOT = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "held_out/phase31_frozen_repair_retime"
)
REPAIR_CACHE = OUTPUT_ROOT / "x2_phase31_heldout_frozen_repair.pkl"
RETIME_CACHE = OUTPUT_ROOT / "x2_phase31_heldout_frozen_repair_time1p46.pkl"
REPORT_JSON = REPO / "reports/retarget/x2_wbt_heldout_transfer_phase31.json"
REPORT_MD = REPO / "reports/retarget/x2_wbt_heldout_transfer_phase31.md"
PREFLIGHT_JSON = REPO / "reports/retarget/x2_wbt_heldout_transfer_phase31_preflight.json"


def frozen_contract() -> dict[str, Any]:
    return {
        "motions": list(MOTION_IDS),
        "selection": "fixed held-out IDs before metrics",
        "held_metric_used_for_tuning": False,
        "source_adapter": "reuse explicit per-entry Phase28 source adapter",
        "phase29": phase29.configuration_manifest(),
        "phase30_time_dilation": phase30.TIME_DILATION,
        "phase30_target_fps": phase30.TARGET_FPS,
        "per_clip_tuning": False,
        "fallback_motion_substitution": False,
        "physics_steps": 0,
        "ppo_updates": 0,
    }


def preflight(rows: dict[str, dict[str, Any]], cache: dict[str, Any]) -> dict[str, Any]:
    records = []
    for motion_id in MOTION_IDS:
        row, entry = rows.get(motion_id), cache.get(motion_id)
        checks = {
            "panel_row_present": row is not None,
            "phase28_official_entry_present": entry is not None,
            "held_out_split": row is not None and row["recommended_split"] == "held_out",
            "source_sha_exact": row is not None and entry is not None and entry.get("source_sha256") == row["source_sha256"],
            "phase28_source_adapter_explicit": entry is not None and bool(entry.get("phase28_source_adapter")),
            "official_joint_count_31": entry is not None and len(entry["joint_names_mujoco"]) == 31,
        }
        records.append({
            "id": motion_id, "checks": checks, "pass": bool(all(checks.values())),
            "dataset": row["dataset"] if row else None,
            "source_adapter": entry.get("phase28_source_adapter") if entry else None,
            "frames": len(entry["dof"]) if entry else None,
        })
    contract = frozen_contract()
    invariant_checks = {
        "phase29_iterations_8": contract["phase29"]["gn_iterations"] == 8,
        "phase29_weights_exact_current": contract["phase29"]["weights"] == phase29.WEIGHTS,
        "phase29_motion_independent_config": contract["phase29"]["per_clip_tuning"] is False,
        "phase30_factor_1p46": contract["phase30_time_dilation"] == 1.46,
        "all_three_fixed_held_ready": all(record["pass"] for record in records),
    }
    return {
        "schema_version": "x2_wbt_heldout_transfer_phase31_preflight_v1",
        "mode": "no_held_metrics_read_before_frozen_transfer",
        "frozen_contract": contract,
        "motions": records,
        "invariant_checks": invariant_checks,
        "pass": bool(all(invariant_checks.values())),
    }


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# X2 WBT Phase31：冻结 repair＋retime held-out transfer", "",
        "## 裁决", "",
        f"- held-out Silver：`{report['summary']['retime_silver_count']}/3`；独立数据门：`{report['decision']['independent_data_gate_pass']}`。",
        "- 三条动作在看指标前固定；没有替换动作、逐clip调参、空间重优化或time-factor扫描。",
        "- Phase29权重/8轮/表示与Phase30 `1.46×` 完全冻结。",
        "- 本阶段 physics/PPO/policy optimizer/真机均为 `0`；contact/FK不是实机GRF/COP/足底力。", "",
        "## 结果", "",
        "| motion | adapter | frames original→retime | original→repair→retime | qstep p95 | root acc | timing L/R | final rejects |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in report["motions"]:
        b = row["retime_audit"]["bronze"]["metrics"]
        s = row["retime_audit"]["silver"]["metrics"]
        lines.append(
            f"| `{row['id']}` | `{row['source_adapter']}` | {row['original_frames']}→{row['retime_resample']['new_frames']} | "
            f"{row['original_audit']['tier']}→{row['repair_audit']['tier']}→{row['retime_audit']['tier']} | "
            f"{b['joint_step_p95_rad']:.4f} | {s['root_horizontal_acceleration_p95_mps2']:.3f} | "
            f"{s['contact_timing_error_p95_s']['left']:.3f}/{s['contact_timing_error_p95_s']['right']:.3f} | "
            f"`{' ; '.join(row['retime_audit']['reject_reasons'])}` |"
        )
    lines += ["", "## 结论", "", report["decision"]["conclusion"], "", "## 下一步", "", report["decision"]["next_step"]]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repair-cache", type=Path, default=REPAIR_CACHE)
    parser.add_argument("--retime-cache", type=Path, default=RETIME_CACHE)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()

    panel = json.loads(phase28.PANEL.read_text())
    panel_rows = {row["id"]: row for row in panel["motions"]}
    cache = joblib.load(phase29.PHASE28_CACHE)
    pre = preflight(panel_rows, cache)
    PREFLIGHT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_JSON.write_text(json.dumps(phase28.json_safe(pre), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"[phase31] preflight pass={pre['pass']} held={sum(row['pass'] for row in pre['motions'])}/3", flush=True)
    if args.preflight_only or not pre["pass"]:
        return

    gates = json.loads(phase28.TIER_GATES.read_text())
    mirror = json.loads(phase28.MIRROR_CONTRACT.read_text())
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase31_offline")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)
    joint_axes = phase28.amass_adapter.parse_joint_axes(phase28.OFFICIAL_MJCF)
    repaired_cache, retimed_cache, records = {}, {}, []

    for index, motion_id in enumerate(MOTION_IDS, start=1):
        started = time.perf_counter()
        row, original = panel_rows[motion_id], cache[motion_id]
        print(f"[phase31] frozen repair {index}/3 {motion_id}", flush=True)
        original_audit = phase28.audit_tier(row, original, model, reset, gates, mirror)
        repaired, diagnostics = phase29.solve_motion(row, original, model, reset, gates, mirror, joint_axes)
        repaired["source_sha256"] = row["source_sha256"]
        repaired["recommended_split"] = row["recommended_split"]
        repair_audit = phase28.audit_tier(row, repaired, model, reset, gates, mirror)
        retimed, resample, frozen_contact = phase30.dilate_entry(repaired, row, model, joint_axes)
        retimed["source_sha256"] = row["source_sha256"]
        retimed["recommended_split"] = row["recommended_split"]
        retime_audit = phase30.audit_with_frozen_contact(row, retimed, model, reset, gates, mirror, frozen_contact)
        repaired_cache[motion_id], retimed_cache[motion_id] = repaired, retimed
        records.append({
            "id": motion_id, "dataset": row["dataset"], "split": row["recommended_split"],
            "source_path": row["source_path"], "source_sha256": row["source_sha256"],
            "source_adapter": original["phase28_source_adapter"], "original_frames": len(original["dof"]),
            "original_entry_sha256": phase28.array_hash(original), "repair_entry_sha256": phase28.array_hash(repaired), "retime_entry_sha256": phase28.array_hash(retimed),
            "original_audit": original_audit, "repair_audit": repair_audit, "retime_audit": retime_audit,
            "repair_diagnostics": diagnostics, "retime_resample": resample,
            "elapsed_s": float(time.perf_counter()-started),
        })
        print(f"[phase31] result {motion_id} {original_audit['tier']}->{repair_audit['tier']}->{retime_audit['tier']}", flush=True)

    args.repair_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(repaired_cache, args.repair_cache, compress=True)
    joblib.dump(retimed_cache, args.retime_cache, compress=True)
    silver = [row["id"] for row in records if row["retime_audit"]["tier"] == "Silver"]
    passed = bool(silver)
    conclusion = (
        f"冻结Phase29+30链在 {len(silver)}/3 条固定 held-out 上产生Silver，独立数据门通过；仍仅为静态Silver，不是Gold或物理稳定证明。"
        if passed else
        "冻结Phase29+30链在三条固定 held-out 上没有产生Silver；独立数据门仍锁定，不得以train Silver替代held-out证据。"
    )
    report = {
        "schema_version": "x2_wbt_heldout_transfer_phase31_v1",
        "mode": "offline_frozen_repair_retime_no_physics_no_PPO",
        "truth_boundary": {"contact_and_fk_are_model_estimates": True, "not_real_grf_cop_force": True, "physics_steps": 0, "ppo_updates": 0, "policy_optimizer_steps": 0, "real_robot": False, "silver_is_not_gold": True},
        "frozen_contract": frozen_contract(), "preflight": pre,
        "provenance": {
            "phase28_cache": {"path": str(phase29.PHASE28_CACHE), "sha256": phase28.sha256(phase29.PHASE28_CACHE)},
            "phase29_script": {"path": str(Path(phase29.__file__)), "sha256": phase28.sha256(Path(phase29.__file__))},
            "phase30_script": {"path": str(Path(phase30.__file__)), "sha256": phase28.sha256(Path(phase30.__file__))},
            "tier_gates": {"path": str(phase28.TIER_GATES), "sha256": phase28.sha256(phase28.TIER_GATES)},
            "repair_cache": {"path": str(args.repair_cache)}, "retime_cache": {"path": str(args.retime_cache)},
        },
        "motions": records,
        "summary": {"completed": len(records), "repair_silver_count": sum(row["repair_audit"]["tier"] == "Silver" for row in records), "retime_silver_count": len(silver), "retime_silver_ids": silver, "retime_bronze_count": sum(row["retime_audit"]["tier"] == "Bronze" for row in records), "retime_reject_count": sum(row["retime_audit"]["tier"] == "Reject" for row in records)},
        "decision": {"independent_data_gate_pass": passed, "minimum_one_heldout_silver": passed, "static_silver_only_not_gold": True, "conclusion": conclusion, "next_step": "Stop for review; do not automatically run physics or PPO." if passed else "Keep faithful Any2Any/PPO locked; do not substitute easier held-out motions."},
    }
    report["provenance"]["repair_cache"]["sha256"] = phase28.sha256(args.repair_cache)
    report["provenance"]["retime_cache"]["sha256"] = phase28.sha256(args.retime_cache)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_md(report, args.report_md)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
