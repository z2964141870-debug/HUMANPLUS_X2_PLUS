from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/x2_faithful_bronze_exact_s7_phase45.yaml"
REPORT = REPO / "reports/retarget/x2_faithful_bronze_exact_s7_phase45.json"
RUNNER = REPO / "tools/retarget/run_x2_faithful_bronze_zero_phase45.py"
LAUNCHER = REPO / "scripts/run_x2_faithful_bronze_zero_phase45.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase45_config_is_bronze_only_and_fail_closed():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["immutable"] is True
    assert config["methodology"]["training_tier"] == "kinematic_bronze"
    assert "silver_contact" in config["methodology"]["forbidden_claims"]
    assert config["one_update_preregistered"]["authorized_in_phase45"] is False
    assert config["five_update_preregistered"]["authorized_in_phase45"] is False
    assert config["five_update_preregistered"]["note"] == (
        "finite_loss_alone_never_unlocks"
    )
    assert config["held_out_gold"]["optimizer_eligible"] is False
    assert config["held_out_gold"]["cross_embodiment_performance_claim_allowed"] is False
    for row in config["frozen_inputs"].values():
        path = Path(row["path"])
        assert path.is_file()
        assert sha256(path) == row["sha256"]
    assert sha256(Path(config["train_bronze"]["path"])) == config["train_bronze"][
        "sha256"
    ]

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert sha256(CONFIG) in launcher
    refused = subprocess.run(
        ["bash", str(LAUNCHER), "one-update"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 45


def test_derived_bronze_sampler_is_exact_and_truthfully_labeled():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload = joblib.load(config["train_bronze"]["path"])
    assert list(payload) == config["train_bronze"]["sampler_keys"]
    assert list(payload) == [
        "AMASS-STAND-001",
        "AMASS-UPPER-001",
        "PHUMA-LUNGE-R-001",
    ]
    for entry in payload.values():
        assert entry["split"] == "train"
        assert entry["phase45_tier"] == "kinematic_bronze"
        boundary = entry["phase45_truth_boundary"]
        assert "not Silver" in boundary
        assert "not Silver" in boundary and "hardware GRF/COP/wrench" in boundary


def test_phase45_zero_report_passes_without_optimizer_or_physics():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"]["status"] == (
        "BRONZE_EXACT_S7_ZERO_PASSED_ONE_UPDATE_PREREGISTERED_NOT_RUN"
    )
    assert all(report["guards"].values())
    assert all(report["zero_update_checks"].values())
    assert report["truth_boundary"]["train_reference_tier"] == "kinematic_bronze"
    assert report["truth_boundary"]["train_reference_is_silver"] is False
    assert report["truth_boundary"]["train_reference_is_dynamics_truth"] is False
    assert report["truth_boundary"]["physics_steps"] == 0
    assert report["truth_boundary"]["optimizer_minibatch_steps"] == 0
    assert report["bronze_provenance"]["all_selected_are_bronze"] is True
    assert report["bronze_provenance"]["none_selected_are_silver"] is True


def test_b0_batches_and_split_contract_are_exact():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["fixed_batches"]["train_bronze"]["key"] == "PHUMA-LUNGE-R-001"
    assert report["fixed_batches"]["held_gold"]["key"] == (
        "official_native_dance_held_out_000"
    )
    for label in ("train_bronze", "held_gold"):
        metrics = report["fixed_batches"][label]["metrics"]
        assert metrics["reference_token_max_abs"] == 0.0
        assert metrics["action_target_max_abs"] == 0.0
        assert metrics["value_max_abs"] == 0.0
        assert metrics["head_nominal_exact"] is True
        assert metrics["all_outputs_finite"] is True
    one = report["one_update_preregistered"]
    assert one["optimizer_train_keys"] == [
        "AMASS-STAND-001",
        "AMASS-UPPER-001",
        "PHUMA-LUNGE-R-001",
    ]
    assert one["optimizer_held_keys"] == []
    assert one["live_optimizer_executed"] is False
    assert one["five_update_unlock"]["locked_until_one_update_passes"] is True


def test_runner_contains_no_optimizer_or_isaac_step_path():
    source = RUNNER.read_text(encoding="utf-8")
    assert "import isaac" not in source
    assert "torch.optim" not in source
    assert "env.step(" not in source
    assert "optimizer.step(" not in source
