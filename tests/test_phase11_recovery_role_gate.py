import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools/official_x2/run_phase11_recovery_role_paired_gate.sh"
ADAPTER = REPO_ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py"
HANDOFF = REPO_ROOT / "tools/official_x2/skill_handoff_contract.py"
REPORT = REPO_ROOT / "reports/baseline/x2_recovery_phase11_role_semantics.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase11_gate_freezes_repaired_role_contract() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    historical = report["frozen_sha256"]
    drift = report["historical_source_reproducibility"]

    assert f'ADAPTER_SHA="{historical["adapter"]}"' in runner
    assert f'HANDOFF_SHA="{historical["handoff_contract"]}"' in runner
    assert _sha256(ADAPTER) == drift["current_adapter_sha256"]
    assert _sha256(HANDOFF) == drift["current_handoff_sha256"]
    assert drift["current_adapter_matches_historical"] is False
    assert drift["current_handoff_matches_historical"] is False
    assert _sha256(ADAPTER) != historical["adapter"]
    assert _sha256(HANDOFF) != historical["handoff_contract"]
    assert "check_sha \"$ADAPTER_SHA\"" in runner
    assert "check_sha \"$HANDOFF_SHA\"" in runner
    assert "REPEATS=5" in runner
    assert 'STOP_CONTROLLER=curriculum_then_policy' in runner
    assert 'STOP_TRANSITION_SECONDS=2.0' in runner
    assert 'STOP_INTENT_DECELERATE_SECONDS=2.0' in runner
    assert 'PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2' in runner


def test_phase11_gate_changes_only_recovery_role_model() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert "labels=(source_recovery candidate_recovery)" in runner
    assert "recovery_models=(" in runner
    assert runner.count("stand_backend_scratch_i150_actor.onnx") >= 3
    assert runner.count("phase9_stateful_recovery_f005_u5_actor.onnx") >= 2
    assert "STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx" in runner
    assert 'RECOVERY_MODEL_PATH="/models/$recovery"' in runner
    assert 'expected_counts = {"main": 360, "stationary": 100, "recovery": 300}' in runner
    assert '"single_variable": "recovery_model source -> phase9 f005"' in runner
    assert '"updates25_unlocked": False' in runner
    assert "train_x2" not in runner
