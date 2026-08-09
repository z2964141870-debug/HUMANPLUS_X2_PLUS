from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/retarget/x2_faithful_any2any_readiness_phase22.json"
TOOL = ROOT / "tools/retarget/audit_x2_faithful_any2any_readiness_phase22.py"


def test_phase22_report_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED_BEFORE_ZERO_UPDATE"
    assert report["not_rejected_by_phase21"] is True
    assert all(report["checks"].values())
    assert report["x2_alignment"]["offline_ready"] is True
    assert report["x2_alignment"]["runtime_ready"] is False
    assert len(report["x2_alignment"]["target_wbt29_order"]) == 29
    assert len(report["x2_alignment"]["head_locked2"]) == 2
    assert report["data_isolation"]["checks"]["embargo_frames"] == 200
    assert report["data_isolation"]["checks"]["embargo_seconds"] == 4.0
    assert len(report["ready"]) == 5
    assert len(report["blockers"]) == 6


def test_phase22_gates_are_fail_closed_and_small() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    zero = report["zero_update_gate"]
    one = report["one_update_gate"]
    assert zero["exact_compute"]["env_steps"] == 0
    assert zero["exact_compute"]["optimizer_steps"] == 0
    assert "blocks the optimizer" in zero["stop"]
    assert one["exact_compute"] == {
        "envs": 64,
        "rollout_steps_per_env": 24,
        "transitions": 1536,
        "ppo_epochs": 5,
        "mini_batches_per_epoch": 4,
        "optimizer_minibatch_steps": 20,
    }
    assert "zero-update" in one["precondition"]
    assert "not evidence" in one["interpretation"]


def test_phase22_tool_is_read_only_with_respect_to_training() -> None:
    source = TOOL.read_text(encoding="utf-8")
    forbidden = ("mj_step(", ".backward(", "optimizer.step(", "torch.save(")
    assert all(token not in source for token in forbidden)
    assert "joblib.load" in source
    assert "source_frame_range" in source
