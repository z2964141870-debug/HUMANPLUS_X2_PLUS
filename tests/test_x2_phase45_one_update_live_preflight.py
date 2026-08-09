from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45_one_update_preflight.json"
TOOL = REPO / "tools/retarget/audit_x2_phase45_one_update_live_preflight.py"


def test_preflight_blocks_before_physics_optimizer():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"]["status"] == (
        "ONE_UPDATE_BLOCKED_LIVE_WBT29_EXACT_S7_NOT_WIRED"
    )
    truth = report["truth_boundary"]
    assert truth["isaac_environment_instances"] == 0
    assert truth["physics_steps"] == 0
    assert truth["optimizer_instances"] == 0
    assert truth["optimizer_minibatch_steps"] == 0
    assert truth["checkpoints_created"] == 0


def test_all_five_live_contract_blockers_are_explicit():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert {row["id"] for row in report["blockers"]} == {
        "LIVE_ACTION_OUTPUT_IS_31_NOT_WBT29",
        "LIVE_POLICY_CRITIC_HISTORY_IS_31_NOT_WBT29",
        "LIVE_REFERENCE_ENCODER_INPUT_IS_31_JOINT",
        "LIVE_CRITIC_LORA_SCOPE_IS_NOT_EXACT_S7",
        "PHASE23_HOOK_NOT_LIVE_WIRED",
    }
    evidence = report["evidence"]
    assert evidence["offline_zero_pass"] is True
    assert evidence["bronze_train_keys_exact"] is True
    assert evidence["live_action_selector_is_all_31"] is True
    assert evidence["live_critic_scope_hardcoded_all_layers"] is True
    assert evidence["live_launcher_references_WBT29_contract"] is False
    assert evidence["live_trainer_references_WBT29_contract"] is False


def test_audit_tool_itself_has_no_training_path():
    source = TOOL.read_text(encoding="utf-8")
    assert "import isaac" not in source
    assert "torch.optim" not in source
    assert "env.step(" not in source
    assert "optimizer.step(" not in source
