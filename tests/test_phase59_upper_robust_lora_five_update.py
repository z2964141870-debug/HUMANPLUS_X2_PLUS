from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "reports/retarget/x2_upper_robust_lower_lora_five_update_phase59_prereg.json"
RESULT = REPO / "reports/retarget/x2_upper_robust_lower_lora_five_update_phase59.json"


def test_phase59_prereg_is_fixed_and_early_stop():
    cfg = json.loads(PREREG.read_text())
    assert cfg["maximum_updates"] == 5
    assert cfg["per_update"]["learning_rate"] == 5.0e-5
    assert cfg["per_update"]["rank"] == cfg["per_update"]["alpha"] == 4
    assert cfg["per_update"]["transitions"] == 1536
    assert "Hard gate failure stops immediately" in cfg["stop"]


def test_phase59_result_keeps_only_evidence_weights():
    result = json.loads(RESULT.read_text())
    assert 1 <= result["updates_attempted"] <= 5
    assert result["retained_checkpoint_count"] in (2, 3)
    assert result["weights"]["source"]
    assert result["weights"]["best"]
    artifact_dir = REPO / "artifacts/retarget/x2_upper_robust_lower_lora_phase59"
    assert len(list(artifact_dir.glob("*.pt"))) == result["retained_checkpoint_count"]
    assert "no automatic longer training" in result["boundary"]
