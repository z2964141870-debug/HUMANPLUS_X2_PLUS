from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "reports/retarget/x2_upper_robust_lower_one_update_phase56_prereg.json"
RESULT = REPO / "reports/retarget/x2_upper_robust_lower_one_update_phase56.json"


def test_phase56_prereg_is_one_update_and_domain_balanced():
    cfg = json.loads(PREREG.read_text())
    assert cfg["status"] == "PREREGISTERED_BEFORE_OPTIMIZER"
    assert cfg["train"]["transitions"] == 1536
    assert cfg["train"]["optimizer_steps"] == 20
    assert cfg["train"]["domain_balance"] == {
        "none_ideal": 24,
        "none_response": 8,
        "bounded_ideal": 24,
        "bounded_response": 8,
    }
    assert "never auto-start" in cfg["stop"]


def test_phase56_launcher_has_only_source_and_final_weights():
    launcher = (REPO / "scripts/run_x2_upper_robust_one_update_phase56.sh").read_text()
    runner = (REPO / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    assert "source_stage219.pt" in launcher
    assert "final_one_update.pt" in launcher
    assert "model_0.pt" not in launcher
    assert "model_1.pt" not in launcher
    assert 'for _ in range(24)' in runner
    assert 'step_counter["count"] != 20' in runner
    assert "CWI_UPPER_SPLIT_MODE=interleaved" in launcher


def test_phase56_result_is_fail_closed():
    result = json.loads(RESULT.read_text())
    assert result["decision"] in {
        "PASS_ONE_UPDATE_TREND_GATE_STOP",
        "FAIL_ONE_UPDATE_TREND_GATE_STOP",
    }
    assert result["training"]["optimizer_steps"] == 20
    assert result["training"]["checkpoint_count"] == 2
    assert result["training"]["std_hash_before"] == result["training"]["std_hash_after"]
    assert "never auto-unlocks five updates" in result["boundary"]
