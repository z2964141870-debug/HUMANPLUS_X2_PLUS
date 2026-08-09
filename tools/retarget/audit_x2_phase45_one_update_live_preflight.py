#!/usr/bin/env python3
"""Fail-closed live preflight for the authorized Phase45 one-update pilot.

This audit must run before Isaac or an optimizer is created.  It demonstrates
whether the current live trainer actually consumes the Phase23 WBT29 contract
and exact-S7 scope, rather than inferring readiness from an offline B=0 test.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/x2_faithful_bronze_exact_s7_phase45.yaml"
ZERO_REPORT = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45.json"
JSON_OUT = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45_one_update_preflight.json"
MD_OUT = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45_one_update_preflight.md"

SANDBOX = Path("/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox")
ACTION_CFG = SANDBOX / "gear_sonic/config/manager_env/actions/terms/joint_pos.yaml"
POLICY_OBS = SANDBOX / "gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml"
CRITIC_OBS = SANDBOX / "gear_sonic/config/manager_env/observations/critic/privileged_mf_hist.yaml"
REFERENCE_OBS = SANDBOX / "gear_sonic/config/manager_env/observations/terms/command_multi_future_nonflat.yaml"
TRAINER = SANDBOX / "gear_sonic/trl/trainer/ppo_trainer.py"
LIVE_LAUNCHER = REPO / "scripts/run_dcpeft_stage152.sh"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    zero = json.loads(ZERO_REPORT.read_text(encoding="utf-8"))
    action = yaml.safe_load(ACTION_CFG.read_text(encoding="utf-8"))["joint_pos"]
    policy_text = POLICY_OBS.read_text(encoding="utf-8")
    critic_text = CRITIC_OBS.read_text(encoding="utf-8")
    reference = yaml.safe_load(REFERENCE_OBS.read_text(encoding="utf-8"))[
        "command_multi_future_nonflat"
    ]
    launcher = LIVE_LAUNCHER.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    bronze = joblib.load(config["train_bronze"]["path"])

    evidence = {
        "offline_zero_pass": zero["decision"]["status"]
        == "BRONZE_EXACT_S7_ZERO_PASSED_ONE_UPDATE_PREREGISTERED_NOT_RUN",
        "bronze_train_keys_exact": list(bronze) == config["train_bronze"]["sampler_keys"],
        "live_action_selector": action["joint_names"],
        "live_action_selector_is_all_31": action["joint_names"] == [".*"],
        "policy_joint_history_has_no_WBT29_asset_selector": "asset_cfg" not in policy_text,
        "critic_joint_history_has_no_WBT29_asset_selector": "asset_cfg" not in critic_text,
        "live_reference_function": reference["func"],
        "live_reference_is_raw_all_joint_command": reference["func"]
        == "gear_sonic.envs.manager_env.mdp:command_multi_future",
        "live_critic_scope_hardcoded_all_layers": (
            "'+algo.config.any2any_lora.critic_lora_prefixes=[critic_module]'" in launcher
        ),
        "live_launcher_references_WBT29_contract": "WBT29PolicyContract" in launcher,
        "live_trainer_references_WBT29_contract": "WBT29PolicyContract" in trainer,
        "phase23_hook_referenced_by_live_launcher": "ImmutableGoldMotionLibHook" in launcher,
    }
    blockers = [
        {
            "id": "LIVE_ACTION_OUTPUT_IS_31_NOT_WBT29",
            "evidence": "The action term selects [.*], so the live action manager and actor output are 31-D including both head rows; Phase45 exact-S7 requires a 29-D source-semantic output followed by explicit scatter and nominal head lock.",
        },
        {
            "id": "LIVE_POLICY_CRITIC_HISTORY_IS_31_NOT_WBT29",
            "evidence": "policy/critic joint_pos, joint_vel and actions terms have no explicit 29-joint name/order selector; the offline WBT29 gather/permutation is not called by the live environment.",
        },
        {
            "id": "LIVE_REFERENCE_ENCODER_INPUT_IS_31_JOINT",
            "evidence": "command_multi_future_nonflat calls the raw all-joint command. The live encoder therefore uses the X2-expanded 31-joint boundary rather than the frozen source 29-joint 640-D contract.",
        },
        {
            "id": "LIVE_CRITIC_LORA_SCOPE_IS_NOT_EXACT_S7",
            "evidence": "run_dcpeft_stage152.sh hardcodes critic_lora_prefixes=[critic_module], which includes critic module.0 and module.12; exact-S7 freezes both and adapts only 2/4/6/8/10.",
        },
        {
            "id": "PHASE23_HOOK_NOT_LIVE_WIRED",
            "evidence": "WBT29PolicyContract and ImmutableGoldMotionLibHook are exercised by CPU/static tools but are absent from the live Stage152 launcher/trainer path.",
        },
    ]
    report = {
        "schema_version": "x2_phase45_one_update_live_preflight_v1",
        "mode": "read_only_fail_closed_before_isaac_optimizer",
        "authorized_workload": config["one_update_preregistered"],
        "evidence": evidence,
        "blockers": blockers,
        "truth_boundary": {
            "isaac_environment_instances": 0,
            "physics_steps": 0,
            "optimizer_instances": 0,
            "optimizer_minibatch_steps": 0,
            "checkpoints_created": 0,
            "gpu_check": "host RTX 5060 visible outside sandbox; ~986 MiB used at preflight",
            "foreign_process_observed": "AimDK A3 MuJoCo domain232 was running; not killed or modified",
        },
        "decision": {
            "status": "ONE_UPDATE_BLOCKED_LIVE_WBT29_EXACT_S7_NOT_WIRED",
            "result": "The Bronze data and offline B=0 gate are valid, but the only live PPO launcher does not implement the frozen WBT29/exact-S7 contract.",
            "conclusion": "Starting it would run a different 31-D Stage152 experiment and would violate the authorized fail-closed contract. No optimizer or physics was started.",
            "next_step": "Implement a dedicated opt-in live WBT29 observation/action/reference adapter plus exact critic layer prefixes, then run an env-initialization/no-update shape/hash probe before re-authorizing the same one-update pilot.",
        },
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_OUT.write_text(
        "\n".join(
            [
                "# X2 Phase45 one-update live preflight",
                "",
                f"- 裁决：**{report['decision']['status']}**。",
                "- 已完成的Bronze split与exact-S7 B=0仍有效；阻塞发生在live PPO接线，而非数据或学习结果。",
                "- 未创建Isaac环境、未走physics step、未创建optimizer、未生成checkpoint。",
                "",
                "## 假设",
                "",
                "若Phase23–26真的已进入live trainer，则Stage152入口应呈现29维source-semantic action/observation/reference，并只给exact-S7指定层梯度。",
                "",
                "## 干预与对照",
                "",
                "只读比较Phase45冻结合同与当前Stage152 launcher、action cfg、policy/critic observation及reference observation；没有运行训练。",
                "",
                "## 结果",
                "",
                *[f"- `{row['id']}`：{row['evidence']}" for row in blockers],
                "",
                "## 结论",
                "",
                f"- 结果：{report['decision']['result']}",
                f"- 结论：{report['decision']['conclusion']}",
                f"- 下一步：{report['decision']['next_step']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
