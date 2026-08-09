from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/x2_faithful_any2any_phase24.yaml"
ENTRY = REPO / "tools/retarget/audit_x2_faithful_entrypoint_phase24.py"


def test_faithful_config_locks_source_ppo_std_and_b5_gate():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["immutable"] is True
    assert config["ppo"]["num_learning_epochs"] == 5
    assert config["ppo"]["num_mini_batches"] == 4
    assert config["source"]["std"] == {
        "source_key": "std",
        "load": True,
        "trainable": False,
    }
    assert len(config["reward_semantics"]["source3_body_names"]) == 3
    assert len(config["reward_semantics"]["source14_body_names"]) == 14
    assert config["physics_gate"]["live_zero_update_allowed"] is False
    assert config["physics_gate"]["fail_closed_exit_code"] == 42


def test_phase24_report_has_independent_exact_manifests():
    report = json.loads(
        (REPO / "reports/retarget/x2_faithful_any2any_phase24.json").read_text()
    )
    assert report["decision"]["status"] == "B3_B4_IMPLEMENTATION_READY_B5_FAIL_CLOSED"
    assert all(report["static_checks"].values())
    primary = report["lora_manifests"]["sonic_decoder_critic_primary"]
    s7 = report["lora_manifests"]["figure7_exact_s7"]
    assert primary["critic_layers"] == [f"critic_module.module.{i}" for i in (0, 2, 4, 6, 8, 10, 12)]
    assert s7["critic_layers"] == [f"critic_module.module.{i}" for i in (2, 4, 6, 8, 10)]
    assert s7["input_masks"]["module.0"]["selected_columns"] == [64, 994]
    assert all(item["pass"] for item in (primary, s7))
    assert not any(name.endswith("std") for name in primary["trainable_names"])


def test_static_entrypoints_are_split_locked_and_live_zero_fails_closed():
    env = {"PYTHONPATH": str(REPO / "src")}
    train = subprocess.run(
        [sys.executable, str(ENTRY), "--config", str(CONFIG), "--mode", "train-static"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    held = subprocess.run(
        [sys.executable, str(ENTRY), "--config", str(CONFIG), "--mode", "eval-static"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    train_row = json.loads(train.stdout.strip())
    held_row = json.loads(held.stdout.strip())
    assert train_row["split"] == "train" and train_row["optimizer_eligible"] is True
    assert held_row["split"] == "held_out" and held_row["optimizer_eligible"] is False
    refused = subprocess.run(
        [sys.executable, str(ENTRY), "--config", str(CONFIG), "--mode", "live-zero"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert refused.returncode == 42
    assert json.loads(refused.stdout.strip())["status"] == "REFUSED_B5_NOT_FROZEN"
