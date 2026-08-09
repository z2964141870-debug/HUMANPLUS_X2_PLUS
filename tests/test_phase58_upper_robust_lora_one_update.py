from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "reports/retarget/x2_upper_robust_lower_lora_one_update_phase58_prereg.json"
RESULT = REPO / "reports/retarget/x2_upper_robust_lower_lora_one_update_phase58.json"


def test_phase58_prereg_freezes_single_lora_update():
    cfg = json.loads(PREREG.read_text())
    assert cfg["learning_rate"] == 5.0e-5
    assert cfg["learning_rate_schedule"] == "fixed"
    assert cfg["rank"] == cfg["alpha"] == 4
    assert cfg["transitions"] == 1536
    assert cfg["optimizer_steps"] == 20
    assert "do not sweep" in cfg["stop"]


def test_phase58_result_is_lora_only_and_fail_closed():
    result = json.loads(RESULT.read_text())
    assert result["decision"] in {
        "PASS_LORA_ONE_UPDATE_TREND_GATE_STOP",
        "FAIL_LORA_ONE_UPDATE_TREND_GATE_STOP",
    }
    train = result["training"]
    assert train["learning_rate"] == 5.0e-5
    assert train["learning_rate_schedule"] == "fixed"
    assert train["optimizer_steps"] == 20
    assert train["checkpoint_count"] == 2
    assert train["dense_hash_before"] == train["dense_hash_after"]
    assert train["std_hash_before"] == train["std_hash_after"]
    assert all("lora_" in name for name in train["trainable_names"])
