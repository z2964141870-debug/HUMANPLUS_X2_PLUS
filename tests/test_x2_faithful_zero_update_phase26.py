from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/x2_faithful_zero_update_phase26.yaml"
REPORT = REPO / "reports/retarget/x2_faithful_zero_update_phase26.json"
TOOL = REPO / "tools/retarget/run_x2_faithful_zero_update_phase26.py"
LAUNCHER = REPO / "scripts/run_x2_faithful_zero_update_phase26.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase26_config_and_launcher_are_fail_closed():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["immutable"] is True
    contract = config["zero_update_contract"]
    assert contract["environment_instances"] == 0
    assert contract["physics_steps"] == 0
    assert contract["optimizer_instances"] == 0
    assert contract["optimizer_steps"] == 0
    assert contract["fixed_forward_batches"] == 2
    assert contract["one_update_authorized"] is False
    for row in config["frozen_inputs"].values():
        path = Path(row["path"])
        assert path.is_file()
        assert _sha256(path) == row["sha256"]

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert _sha256(CONFIG) in launcher
    assert "run_x2_faithful_zero_update_phase26.py" in launcher
    assert "Phase26 accepts no positional arguments or live-training mode" in launcher
    refused = subprocess.run(
        ["bash", str(LAUNCHER), "one-update"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 44


def test_phase26_report_passes_exact_preregistered_zero_gate():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"]["status"] == (
        "FAITHFUL_ZERO_UPDATE_PASSED_ONE_UPDATE_STILL_LOCKED"
    )
    assert all(report["guards"].values())
    assert all(report["zero_update_checks"].values())
    assert report["truth_boundary"] == {
        "cpu_motionlib_kinematic_loader_fk_instances": 2,
        "isaac_environment_instances": 0,
        "physics_steps": 0,
        "optimizer_instances_and_steps": [0, 0],
        "callback_instances": 0,
        "fixed_forward_batches": 2,
        "framework_initialization": "CPU torch modules + CPU MotionLib kinematic loader only; no SimulationApp or manager environment",
        "official_mujoco_is_held_out_sim_to_sim_not_training_truth": True,
        "model_contact_is_not_hardware_grf_cop_wrench": True,
    }
    assert report["provenance"]["dense_checkpoint_tensor_hash_before"] == report[
        "provenance"
    ]["dense_checkpoint_tensor_hash_after"]


def test_two_batches_are_split_locked_and_base_equals_zero_lora():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert set(report["fixed_batches"]) == {"train", "held_out"}
    assert report["fixed_batches"]["train"]["key"] == (
        "official_native_dance_train_000"
    )
    assert report["fixed_batches"]["held_out"]["key"] == (
        "official_native_dance_held_out_000"
    )
    for split in ("train", "held_out"):
        metrics = report["fixed_batches"][split]["metrics"]
        assert metrics["reference_token_max_abs"] == 0.0
        assert metrics["action_source_max_abs"] == 0.0
        assert metrics["action_target_max_abs"] == 0.0
        assert metrics["value_max_abs"] == 0.0
        assert metrics["head_nominal_exact"] is True
        assert metrics["all_outputs_finite"] is True
    assert report["split_isolation"]["optimizer_path_keys"] == []
    assert report["split_isolation"]["callback_path_keys"] == []
    assert report["split_isolation"]["four_second_embargo_exact"] is True
    assert report["split_hooks"]["held_out"]["optimizer_eligible"] is False


def test_exact_s7_scope_and_no_optimizer_or_physics_code_path():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["lora"]["selected"] == "figure7_exact_s7"
    assert report["lora"]["trainable_names_actual"] == report["lora"][
        "trainable_names_expected"
    ]
    assert len(report["lora"]["trainable_names_actual"]) == 24
    assert report["lora"]["input_masks"]["module.0"] == {
        "selected_columns": [64, 994],
        "selected_count": 930,
        "input_width": 994,
    }
    source = TOOL.read_text(encoding="utf-8")
    assert "import isaac" not in source
    assert "torch.optim" not in source
    assert "env.step(" not in source
    assert "optimizer.step(" not in source
