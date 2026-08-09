#!/usr/bin/env python3
"""Phase45 Bronze-only exact-S7 static and zero-update gate.

This entrypoint deliberately has no optimizer or Isaac environment.  It
replaces the native-Gold *training* batch used by Phase26 with an explicitly
kinematic-Bronze GMR batch, while retaining native Gold solely as a held-out
pipeline/source-ability regression batch.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TOOLS = REPO / "tools"
SONIC = Path("/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox")
for path in (SRC, TOOLS, SONIC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot  # noqa: E402
from retarget.probe_x2_faithful_wbt29_gold_phase23 import native_config  # noqa: E402
from retarget.run_x2_faithful_zero_update_phase26 import (  # noqa: E402
    build_contract,
    create_models,
    default_source29,
    evaluate_batch,
    fixed_batch,
    load_split,
    tensor_mapping_hash,
)
from x2_kinematic_bronze_phase45 import (  # noqa: E402
    sha256,
    validate_bronze_train_artifact,
)
from x2_physics_provenance_guard import (  # noqa: E402
    assert_physics_contract,
    load_runtime_snapshot,
)
from retarget.audit_x2_faithful_entrypoint_phase24 import _load_source_checkpoint  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/x2_faithful_bronze_exact_s7_phase45.yaml"
DEFAULT_JSON = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45.json"
DEFAULT_MD = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen(config: Mapping[str, Any]) -> dict[str, bool]:
    return {
        name: Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
        for name, row in config["frozen_inputs"].items()
    }


def audit_tier_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    phase28 = _load_json(Path(config["frozen_inputs"]["phase28_tier_report"]["path"]))
    phase36 = _load_json(Path(config["frozen_inputs"]["phase36_tier_correction"]["path"]))
    phase28_by_id = {row["id"]: row for row in phase28["motions"]}
    phase30_by_id = {row["id"]: row for row in phase36["phase30_corrected"]}

    expected = config["train_bronze"]["sampler_keys"]
    evidence: dict[str, Any] = {}
    for key in expected:
        source = config["train_bronze"]["source_tier_evidence"][key]
        if source == "phase28_tier_report":
            row = phase28_by_id[key]
            tier = row["tier_audit"]["tier"]
            bronze_pass = row["tier_audit"]["bronze"]["pass"]
            evidence[key] = {
                "source": source,
                "tier": tier,
                "bronze_pass": bronze_pass,
                "silver_pass": row["tier_audit"]["silver"]["pass"],
                "recommended_split": row["recommended_split"],
                "entry_sha256": row["official_entry_sha256"],
            }
        elif source == "phase36_tier_correction":
            row = phase30_by_id[key]
            corrected = row["corrected"]
            tier = corrected["tier"]
            bronze_pass = corrected["bronze_pass"]
            evidence[key] = {
                "source": source,
                "historical_tier": row["historical_tier"],
                "tier": tier,
                "bronze_pass": bronze_pass,
                "silver_pass": corrected["silver_pass"],
                "split": row["split"],
                "entry_sha256": row["entry_sha256"],
                "historical_silver_revoked": phase36["decision"][
                    "phase30_lunge_silver_revoked"
                ],
            }
        else:
            raise ValueError(f"unknown tier evidence source for {key}: {source}")
        if tier != "Bronze" or bronze_pass is not True:
            raise ValueError(f"{key} is not frozen kinematic Bronze")
    return {
        "selected_keys_exact": list(evidence) == expected,
        "all_selected_are_bronze": all(row["tier"] == "Bronze" for row in evidence.values()),
        "none_selected_are_silver": all(row["silver_pass"] is False for row in evidence.values()),
        "evidence": evidence,
        "phase36_silver_revocation_preserved": phase36["decision"][
            "phase30_lunge_silver_revoked"
        ] is True,
    }


def load_bronze(config: Mapping[str, Any]) -> tuple[MotionLibRobot, dict[str, Any], dict[str, Any]]:
    row = config["train_bronze"]
    path = Path(row["path"])
    hook = validate_bronze_train_artifact(path, row["sha256"], row["sampler_keys"])
    payload = joblib.load(path)
    motion = MotionLibRobot(native_config(path), len(payload), "cpu")
    motion.load_motions_for_evaluation()
    if list(motion.curr_motion_keys) != row["sampler_keys"]:
        raise ValueError("live MotionLib sampler keys differ from Bronze contract")
    hook["motionlib_sampler_keys_exact"] = True
    hook["motionlib_internal_fps"] = float(motion._motion_fps[0])
    return motion, payload, hook


def render(report: Mapping[str, Any]) -> str:
    d = report["decision"]
    return "\n".join(
        [
            "# X2 Bronze-only Faithful exact-S7 — Phase45",
            "",
            f"- 裁决：**{d['status']}**。",
            "- 方法学更正：Any2Any的target PPO可使用目标本体GMR运动学reference；动态Silver不是论文先验硬门。",
            "- 真实性边界：3条训练reference只标为`kinematic Bronze`；绝不称Silver、动力学真值、实机GRF/COP或free-root稳定证据。",
            "",
            "## 假设",
            "",
            "Phase23–26的WBT29、B1–B5和exact-S7若能无改写接收Bronze sampler，则B=0应逐元素复现source，且Gold held-out不得进入optimizer。",
            "",
            "## 干预",
            "",
            "- train：AMASS stand、AMASS upper、PHUMA lunge三条Bronze；固定zero train batch使用唯一动态意图lunge。",
            "- held-out：Phase10 native Gold只作pipeline/原能力回归，不能冒充cross-embodiment held-out。",
            "- 只做2个CPU forward；Isaac环境、physics step、optimizer、PPO均为0。",
            "",
            "## 对照与结果",
            "",
            f"- Bronze provenance：`{report['bronze_provenance']}`",
            f"- B1–B5/hash guards：`{report['guards']}`",
            f"- lunge train B=0：`{report['fixed_batches']['train_bronze']['metrics']}`",
            f"- Gold held B=0：`{report['fixed_batches']['held_gold']['metrics']}`",
            f"- zero checks：`{report['zero_update_checks']}`",
            "",
            "## 结论",
            "",
            f"- 结果：{d['result']}",
            f"- 结论：{d['conclusion']}",
            f"- 下一步：{d['next_step']}",
            "",
            "## 1→5 update预注册边界",
            "",
            "- 1-update是20个optimizer minibatch的数值sanity，不是性能证明；本阶段没有执行。",
            "- 仅loss有限不能解锁5-update；必须lunge即时诊断无NaN、termination不恶化且tracking方向正确，同时原能力回归过门。",
            "- held Gold永不进optimizer，且只代表pipeline/原能力；当前没有独立cross-embodiment held-out性能证据。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("immutable") is not True:
        raise ValueError("Phase45 config must be immutable")

    frozen = verify_frozen(config)
    if not all(frozen.values()):
        raise RuntimeError(f"Phase45 frozen input drift: {[k for k,v in frozen.items() if not v]}")
    phase24 = yaml.safe_load(
        Path(config["frozen_inputs"]["phase24_config"]["path"]).read_text(encoding="utf-8")
    )
    phase24_report = _load_json(Path(config["frozen_inputs"]["phase24_report"]["path"]))
    phase23_report = _load_json(Path(config["frozen_inputs"]["phase23_report"]["path"]))
    phase25_report = _load_json(Path(config["frozen_inputs"]["phase25_report"]["path"]))
    phase26_report = _load_json(Path(config["frozen_inputs"]["phase26_report"]["path"]))
    tier = audit_tier_provenance(config)
    bronze_motion, bronze_payload, bronze_hook = load_bronze(config)

    held_motion, held_payload, held_hook = load_split(phase24, "held_out")
    held_key = config["fixed_zero_batches"]["held_key"]
    if held_key not in held_payload or held_key not in held_hook["sampler_keys"]:
        raise ValueError("held Gold fixed key missing")

    phase25_guard = assert_physics_contract(
        Path(config["frozen_inputs"]["phase25_manifest"]["path"]),
        load_runtime_snapshot(Path(config["frozen_inputs"]["phase25_runtime"]["path"])),
    )
    guards = {
        "all_frozen_hashes_exact": all(frozen.values()),
        "phase25_declared_train_domain_ready": phase25_report["decision"]["status"]
        == "B5_READY_AS_DECLARED_TRAIN_DOMAIN",
        "phase25_files_exact": all(phase25_guard["file_checks"].values()),
        "phase25_runtime_exact": phase25_guard["runtime_check"]["exact"],
        "phase26_B1_B5_zero_historical_pass": all(
            phase26_report["zero_update_checks"].values()
        ),
        "phase24_exact_s7_static_pass": phase24_report["lora_manifests"][
            "figure7_exact_s7"
        ]["pass"],
    }

    contract = build_contract(phase23_report)
    default29 = default_source29(contract, yaml.safe_load(
        Path(config["frozen_inputs"]["phase26_config"]["path"]).read_text(encoding="utf-8")
    )["x2_default_joint_positions"])
    source14 = phase24["reward_semantics"]["source14_body_names"]
    checkpoint = _load_source_checkpoint(Path(phase24["source"]["checkpoint"]))
    dense_before = tensor_mapping_hash(
        checkpoint["policy_state_dict"], checkpoint["value_state_dict"]
    )
    manifest = phase24["lora_manifests"][config["selected_lora_manifest"]]
    models = create_models(checkpoint, manifest)

    h = config["fixed_zero_batches"]["history_frames"]
    f = config["fixed_zero_batches"]["future_frames"]
    train_key = config["fixed_zero_batches"]["train_key"]
    train_batch = fixed_batch(
        bronze_motion, train_key, contract, default29, source14, h, f
    )
    held_batch = fixed_batch(
        held_motion, held_key, contract, default29, source14, h, f
    )
    train_metrics = evaluate_batch(train_batch, models, checkpoint, contract)
    held_metrics = evaluate_batch(held_batch, models, checkpoint, contract)
    dense_after = tensor_mapping_hash(
        checkpoint["policy_state_dict"], checkpoint["value_state_dict"]
    )
    expected_trainable = phase24_report["lora_manifests"][
        config["selected_lora_manifest"]
    ]["trainable_names"]
    tolerance = float(config["zero_update"]["tolerance"])
    both = (train_metrics, held_metrics)
    zero_checks = {
        "guards_pass": all(guards.values()),
        "bronze_provenance_pass": tier["selected_keys_exact"]
        and tier["all_selected_are_bronze"]
        and tier["none_selected_are_silver"]
        and tier["phase36_silver_revocation_preserved"],
        "bronze_sampler_keys_exact": bronze_hook["sampler_keys"]
        == config["train_bronze"]["sampler_keys"],
        "dynamic_train_key_is_lunge": train_key
        == config["train_bronze"]["dynamic_diagnostic_key"],
        "held_gold_optimizer_ineligible": held_hook["optimizer_eligible"] is False,
        "held_gold_absent_from_train_sampler": not bool(
            set(held_hook["sampler_keys"]) & set(bronze_hook["sampler_keys"])
        ),
        "fixed_forward_batches_exactly_two": config["zero_update"][
            "fixed_forward_batches"
        ] == 2,
        "zero_B_token_exact": all(x["reference_token_max_abs"] <= tolerance for x in both),
        "zero_B_target_action_exact": all(x["action_target_max_abs"] <= tolerance for x in both),
        "zero_B_value_exact": all(x["value_max_abs"] <= tolerance for x in both),
        "head_nominal_exact": all(x["head_nominal_exact"] for x in both),
        "std_source_exact": all(x["std_source_exact"] for x in both),
        "all_finite": all(x["input_finite"] and x["all_outputs_finite"] for x in both),
        "dense_checkpoint_hash_unchanged": dense_before == dense_after,
        "exact_s7_trainable_names": models["trainable_names"] == expected_trainable,
        "no_optimizer_or_physics": all(
            config["zero_update"][name] == 0
            for name in ("environment_instances", "physics_steps", "optimizer_instances", "optimizer_minibatch_steps")
        ),
        "one_update_not_executed": config["one_update_preregistered"][
            "authorized_in_phase45"
        ] is False,
        "five_update_locked": config["five_update_preregistered"][
            "authorized_in_phase45"
        ] is False,
    }
    passed = all(zero_checks.values())
    one = config["one_update_preregistered"]
    one_contract = {
        "static_contract_closed": passed,
        "workload_exact": one["transitions"]
        == one["envs"] * one["rollout_steps_per_env"]
        and one["optimizer_minibatch_steps"]
        == one["ppo_epochs"] * one["mini_batches_per_epoch"],
        "optimizer_train_keys": list(bronze_hook["sampler_keys"]),
        "optimizer_held_keys": one["held_optimizer_keys_exact"],
        "dynamic_train_diagnostic": copy.deepcopy(one["lunge_immediate_diagnostic"]),
        "held_gold_role": copy.deepcopy(config["held_out_gold"]),
        "five_update_unlock": copy.deepcopy(config["five_update_preregistered"]),
        "live_optimizer_executed": False,
    }
    report = {
        "schema_version": "x2_faithful_bronze_exact_s7_phase45_v1",
        "provenance": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "source_checkpoint": {
                "path": phase24["source"]["checkpoint"],
                "sha256": phase24["source"]["checkpoint_sha256"],
            },
            "dense_hash_before": dense_before,
            "dense_hash_after": dense_after,
        },
        "truth_boundary": {
            "train_reference_tier": "kinematic_bronze",
            "train_reference_is_silver": False,
            "train_reference_is_dynamics_truth": False,
            "model_contact_is_hardware_grf_cop_wrench": False,
            "held_gold_role": "pipeline/source-ability regression, not cross-embodiment held-out",
            "isaac_environment_instances": 0,
            "physics_steps": 0,
            "optimizer_minibatch_steps": 0,
        },
        "bronze_provenance": tier,
        "bronze_hook": bronze_hook,
        "held_gold_hook": held_hook,
        "guards": guards,
        "fixed_batches": {
            "train_bronze": {
                "key": train_key,
                "input_source_frames": len(bronze_payload[train_key]["dof"]),
                "metrics": train_metrics,
            },
            "held_gold": {
                "key": held_key,
                "source_frame_range": held_payload[held_key]["source_frame_range"],
                "metrics": held_metrics,
            },
        },
        "exact_s7": {
            "manifest": config["selected_lora_manifest"],
            "trainable_names_actual": models["trainable_names"],
            "trainable_names_expected": expected_trainable,
            "input_masks": models["input_masks"],
        },
        "zero_update_checks": zero_checks,
        "one_update_preregistered": one_contract,
        "decision": {
            "status": (
                "BRONZE_EXACT_S7_ZERO_PASSED_ONE_UPDATE_PREREGISTERED_NOT_RUN"
                if passed
                else "BRONZE_EXACT_S7_ZERO_REJECTED"
            ),
            "result": (
                "Three explicitly Bronze GMR references are isolated to the train sampler; the dynamic lunge and native-Gold held batch both reproduce source token/action/value at exact-S7 B=0."
                if passed
                else "At least one Bronze provenance, B1-B5 guard, split, or B=0 check failed."
            ),
            "conclusion": (
                "The corrected Bronze-only numerical path is ready for a separately executed one-update sanity. This is not PPO performance evidence and does not unlock five updates."
                if passed
                else "Optimizer remains blocked before one update."
            ),
            "next_step": (
                "Run exactly one preregistered 20-minibatch update only after review; evaluate lunge direction/termination and Gold original-ability regression. Five-update remains locked unless every additional gate passes."
                if passed
                else "Fix only the failed static/zero contract and rerun; do not start optimizer."
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
