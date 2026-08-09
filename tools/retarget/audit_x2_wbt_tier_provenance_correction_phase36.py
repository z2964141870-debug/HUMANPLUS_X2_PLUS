#!/usr/bin/env python3
"""Phase36 provenance correction for Phase28/29/30 Silver contact auditing.

No motion is regenerated and no optimizer or MuJoCo integration is run.  The
existing caches are re-audited with static official-model FK.  Human/source
height contact remains an *intent* label, while Silver contact is the official
x2.xml 12-active-spheres-per-foot collision (signed surface distance <= 0).
Historical reports are read-only evidence and are never overwritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.run_x2_wbt_full_trajectory_repair_phase29 as phase29
import retarget.run_x2_wbt_time_dilation_phase30 as phase30


OUTPUT_JSON = REPO / "reports/retarget/x2_wbt_tier_provenance_correction_phase36.json"
OUTPUT_MD = REPO / "reports/retarget/x2_wbt_tier_provenance_correction_phase36.md"


def historic_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in report["motions"]}


def compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    metrics = audit["silver"]["metrics"]
    return {
        "tier": audit["tier"],
        "bronze_pass": audit["bronze"]["pass"],
        "silver_pass": audit["silver"]["pass"],
        "silver_checks": audit["silver"]["checks"],
        "reject_reasons": audit["reject_reasons"],
        "contact": {
            key: metrics[key] for key in (
                "source_intent_contact_ratio",
                "official_geometry_contact_ratio",
                "intent_official_frame_agreement",
                "legacy_bronze_tolerance_contact_ratio_not_silver",
                "official_signed_distance_threshold_m",
                "legacy_bronze_tolerance_threshold_m",
                "collision_signed_distance_exact",
                "target_contact_transitions",
                "complete_ds_ss_ds_cycles",
                "contact_timing_error_p95_s",
                "unintended_flight_fraction",
                "contact_provenance",
            ) if key in metrics
        },
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "completed": len(rows),
        "Silver": sum(row["corrected"]["tier"] == "Silver" for row in rows),
        "Bronze": sum(row["corrected"]["tier"] == "Bronze" for row in rows),
        "Reject": sum(row["corrected"]["tier"] == "Reject" for row in rows),
        "tier_changes": [
            {"id": row["id"], "historical": row["historical_tier"], "corrected": row["corrected"]["tier"]}
            for row in rows if row["historical_tier"] != row["corrected"]["tier"]
        ],
    }


def render(report: dict[str, Any]) -> str:
    lunge = next(row for row in report["phase30_corrected"] if row["id"] == "PHUMA-LUNGE-R-001")
    contact = lunge["corrected"]["contact"]
    lines = [
        "# X2 WBT Phase36：Silver contact provenance更正", "",
        "## 裁决", "",
        "- **Phase30 `PHUMA-LUNGE-R-001` 的历史Silver无效，现更正为Bronze；train GMR Silver回到0，WBT optimizer/PPO资格撤销。**",
        "- 历史Phase28/29/30报告和轨迹未覆写；本报告是追加式correction addendum。",
        "- 0个physics integration step、0 optimizer；只对原cache做official x2.xml静态FK/collision复审。", "",
        "## 根因", "",
        f"- Phase7 Bronze ground tolerance：`signed distance <= reset_clearance + radius = {report['contract_correction']['historical_tolerance_threshold_m']:.5f}m`。它用于容忍ground误差，不等于碰撞。",
        "- Phase28/30 tier auditor却把该Bronze容差复用成Silver contact，因此把距地面0–10.05mm的悬空sole误标为contact。",
        "- 更正后的Silver official contact：12 active sole spheres/foot与floor实际collision；与signed surface distance `<=0`逐帧完全一致。",
        "- source foot-height标签只保留为intent，不能覆盖official geometry contact；两者是否一致仍沿用原transition与0.10s timing门，不新增/放松阈值。", "",
        "## PHUMA-LUNGE-R-001", "",
        f"- historical→corrected tier：`{lunge['historical_tier']} → {lunge['corrected']['tier']}`。",
        f"- source intent contact L/R：`{contact['source_intent_contact_ratio']}`。",
        f"- historical tolerance-contact L/R：`{contact['legacy_bronze_tolerance_contact_ratio_not_silver']}`；因此旧报告得到flight=0。",
        f"- official collision-contact L/R：`{contact['official_geometry_contact_ratio']}`；corrected flight={contact['unintended_flight_fraction']:.4f}、transitions={contact['target_contact_transitions']}、DS→SS→DS={contact['complete_ds_ss_ds_cycles']}。",
        f"- intent↔official frame agreement：`{contact['intent_official_frame_agreement']}`；timing={contact['contact_timing_error_p95_s']}。", "",
        "## 重算漏斗", "",
        f"- Phase28固定24：`{report['phase28_summary']}`。",
        f"- Phase29固定3：`{report['phase29_summary']}`。",
        f"- Phase30固定3：`{report['phase30_summary']}`。", "",
        "| Phase30 motion | historical | corrected | corrected flight | official L/R contact | failed |",
        "|---|---|---|---:|---|---|",
    ]
    for row in report["phase30_corrected"]:
        metrics = row["corrected"]["contact"]
        failed = [name for name, value in row["corrected"]["silver_checks"].items() if not value]
        lines.append(
            f"| `{row['id']}` | {row['historical_tier']} | {row['corrected']['tier']} | "
            f"{metrics['unintended_flight_fraction']:.3f} | `{metrics['official_geometry_contact_ratio']}` | `{' ; '.join(failed)}` |"
        )
    lines += ["", "## 结论 / 下一步", "", report["decision"]["conclusion"], "", report["decision"]["next_step"], ""]
    return "\n".join(lines)


def main() -> None:
    panel = json.loads(phase28.PANEL.read_text())
    panel_rows = {row["id"]: row for row in panel["motions"]}
    gates = json.loads(phase28.TIER_GATES.read_text())
    mirror = json.loads(phase28.MIRROR_CONTRACT.read_text())
    historic28_report = json.loads(phase28.REPORT_JSON.read_text())
    historic29_report = json.loads(phase29.REPORT_JSON.read_text())
    historic30_report = json.loads(phase30.REPORT_JSON.read_text())
    historic28 = historic_rows(historic28_report)
    historic29 = historic_rows(historic29_report)
    historic30 = historic_rows(historic30_report)
    cache28 = joblib.load(phase29.PHASE28_CACHE)
    cache29 = joblib.load(phase29.OUTPUT_CACHE)
    cache30 = joblib.load(phase30.OUTPUT_CACHE)

    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_phase36_static_fk_helpers")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = phase7.official_reset_geometry(model, physics)

    corrected28 = []
    for motion_id, entry in cache28.items():
        audit = phase28.audit_tier(panel_rows[motion_id], entry, model, reset, gates, mirror)
        corrected28.append({
            "id": motion_id,
            "split": panel_rows[motion_id]["recommended_split"],
            "entry_sha256": phase28.array_hash(entry),
            "historical_tier": historic28[motion_id]["tier_audit"]["tier"],
            "corrected": compact_audit(audit),
        })
    print(f"[phase36] Phase28 static re-audit {len(corrected28)}/24", flush=True)

    corrected29 = []
    for motion_id, entry in cache29.items():
        audit = phase28.audit_tier(panel_rows[motion_id], entry, model, reset, gates, mirror)
        corrected29.append({
            "id": motion_id,
            "split": panel_rows[motion_id]["recommended_split"],
            "entry_sha256": phase28.array_hash(entry),
            "historical_tier": historic29[motion_id]["repair_audit"]["tier"],
            "corrected": compact_audit(audit),
        })
    print(f"[phase36] Phase29 static re-audit {len(corrected29)}/3", flush=True)

    corrected30 = []
    for motion_id, candidate in cache30.items():
        row = panel_rows[motion_id]
        source_contract = phase29.source_contract(row, cache29[motion_id], model)
        source_contact = {
            side: phase30.nearest_phase_resample(source_contract["contact"][side], len(candidate["dof"]))
            for side in phase29.SIDES
        }
        audit = phase30.audit_with_frozen_contact(row, candidate, model, reset, gates, mirror, source_contact)
        compact = compact_audit(audit)
        # audit_with_frozen_contact intentionally replaces source intent after
        # phase28.audit_tier; retain the corrected metrics it added.
        compact["contact"].update({
            key: audit["silver"]["metrics"][key] for key in (
                "source_intent_contact_ratio", "official_geometry_contact_ratio",
                "intent_official_frame_agreement", "official_signed_distance_threshold_m",
                "collision_signed_distance_exact",
            )
        })
        corrected30.append({
            "id": motion_id,
            "split": row["recommended_split"],
            "entry_sha256": phase28.array_hash(candidate),
            "historical_tier": historic30[motion_id]["phase30_audit"]["tier"],
            "corrected": compact,
        })
    print(f"[phase36] Phase30 static re-audit {len(corrected30)}/3", flush=True)

    summary28, summary29, summary30 = summary(corrected28), summary(corrected29), summary(corrected30)
    lunge = next(row for row in corrected30 if row["id"] == "PHUMA-LUNGE-R-001")
    revoked = lunge["historical_tier"] == "Silver" and lunge["corrected"]["tier"] != "Silver"
    report = {
        "schema_version": "x2_wbt_tier_provenance_correction_phase36_v1",
        "mode": "append_only_static_FK_collision_reaudit_no_mj_step_no_optimizer",
        "provenance": {
            "historical_phase28": {"path": str(phase28.REPORT_JSON), "sha256": phase28.sha256(phase28.REPORT_JSON)},
            "historical_phase29": {"path": str(phase29.REPORT_JSON), "sha256": phase28.sha256(phase29.REPORT_JSON)},
            "historical_phase30": {"path": str(phase30.REPORT_JSON), "sha256": phase28.sha256(phase30.REPORT_JSON)},
            "cache28": {"path": str(phase29.PHASE28_CACHE), "sha256": phase28.sha256(phase29.PHASE28_CACHE)},
            "cache29": {"path": str(phase29.OUTPUT_CACHE), "sha256": phase28.sha256(phase29.OUTPUT_CACHE)},
            "cache30": {"path": str(phase30.OUTPUT_CACHE), "sha256": phase28.sha256(phase30.OUTPUT_CACHE)},
            "tier_gates_unchanged": {"path": str(phase28.TIER_GATES), "sha256": phase28.sha256(phase28.TIER_GATES)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase28.sha256(physics.DEFAULT_SCENE)},
        },
        "truth_boundary": {
            "physics_integration_steps": 0, "optimizer_training_ppo": False,
            "trajectory_root_threshold_split_source_sha_changed": False,
            "static_mj_forward_fk_collision_only": True,
            "contact_is_model_estimate_not_hardware_grf_cop_wrench": True,
            "historical_reports_overwritten": False,
        },
        "contract_correction": {
            "bug": "Phase7 Bronze reset-clearance + one-sphere-radius tolerance was reused as Silver contact",
            "historical_tolerance_threshold_m": float(reset["reset_clearance_m"] + reset["sole_sphere_radius_m"]),
            "correct_official_collision_signed_distance_threshold_m": 0.0,
            "active_sole_spheres_per_foot": 12,
            "intent_definition_unchanged": phase28.OFFLINE_METRIC_CONTRACT["source_contact_intent"],
            "intent_geometry_consistency_gate": "existing intended-foot transition >=2 and event timing <=0.10s; no new numeric threshold",
        },
        "phase28_summary": summary28,
        "phase29_summary": summary29,
        "phase30_summary": summary30,
        "phase28_corrected": corrected28,
        "phase29_corrected": corrected29,
        "phase30_corrected": corrected30,
        "decision": {
            "phase30_lunge_silver_revoked": revoked,
            "minimum_train_gmr_silver": False,
            "faithful_any2any_optimizer_or_ppo_qualified": False,
            "conclusion": "Phase30唯一train Silver来自contact审计provenance错误；官方geometry复审后降级，历史Silver与其后续optimizer资格正式撤销。该更正不否定1.46 retime对连续性/PD trackability的改善。",
            "next_step": "保持optimizer/PPO锁定；回到contact-feasible reference生成，不做root-z repair或阈值扫描。",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
