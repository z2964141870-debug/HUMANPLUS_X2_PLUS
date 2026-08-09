import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from official_x2.physical_handoff_support_gate import (
    build_contract,
    load_contract,
    physical_gate_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "manifests/x2_phase24_physical_handoff_support_gate.json"


def test_frozen_contract_loads_and_exact_reference_is_supported() -> None:
    file_hash = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    payload = load_contract(CONTRACT, file_hash)
    assert payload["reference_count"] == 432
    angular = payload["groups"]["base_ang_vel"]["reference"][0]
    gravity = payload["groups"]["projected_gravity"]["reference"][0]
    decision = physical_gate_decision(angular, gravity, payload)
    assert decision["base_ang_vel"]["in_support"]
    assert decision["projected_gravity"]["in_support"]
    assert decision["allow_handoff"]


def test_both_groups_are_required_independently() -> None:
    payload = load_contract(CONTRACT)
    angular = np.asarray(payload["groups"]["base_ang_vel"]["reference"][0])
    gravity = np.asarray(payload["groups"]["projected_gravity"]["reference"][0])
    far_angular = angular + 100.0 * np.asarray(payload["groups"]["base_ang_vel"]["scale"])
    far_gravity = gravity + 100.0 * np.asarray(payload["groups"]["projected_gravity"]["scale"])
    a = physical_gate_decision(far_angular, gravity, payload)
    g = physical_gate_decision(angular, far_gravity, payload)
    assert not a["base_ang_vel"]["in_support"] and a["projected_gravity"]["in_support"]
    assert not g["projected_gravity"]["in_support"] and g["base_ang_vel"]["in_support"]
    assert not a["allow_handoff"] and not g["allow_handoff"]


def test_thresholds_are_frozen_leave_one_out_not_zero() -> None:
    rng = np.random.default_rng(24)
    payload = build_contract(
        rng.normal(size=(40, 3)), rng.normal(size=(40, 3)), source_manifest_sha256="b" * 64
    )
    for group in payload["groups"].values():
        assert group["threshold_value"] > 0.0
        assert group["threshold_method"].endswith("p95")


def test_tamper_fails_closed(tmp_path: Path) -> None:
    copy = tmp_path / "physical.json"
    copy.write_bytes(CONTRACT.read_bytes())
    payload = json.loads(copy.read_text())
    payload["groups"]["projected_gravity"]["threshold_value"] += 1.0
    copy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_contract(copy)


def test_adapter_physical_gate_is_default_off_and_only_strengthens_phase23() -> None:
    source = (ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py").read_text()
    start = source.index('elif self.args.stop_controller == "curriculum_then_policy":')
    end = source.index('elif self.args.stop_controller == "ramp_policy":', start)
    branch = source[start:end]
    assert source.count("physical_gate_decision(") == 1
    assert "self.curriculum_handoff_physical_gate_contract is not None" in branch
    assert 'gate["allow_handoff"] = bool(' in branch
    assert 'gate["allow_handoff"]\n                                and physical_gate["allow_handoff"]' in branch
    parser = source[source.index("def parse_args()") : source.index("def main()")]
    assert "physical handoff gate can only strengthen the Phase23 handoff gate" in parser
    assert 'parser.error("physical handoff gate requires --state-prediction-seconds 0")' in parser


def test_runner_plumbing_is_conditional() -> None:
    inner = (ROOT / "tools/official_x2/run_official_gate_inner.sh").read_text()
    outer = (ROOT / "tools/official_x2/run_official_gate_case.sh").read_text()
    assert 'CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT="${CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT:-}"' in inner
    assert '[[ -n "$CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT" ]]' in inner
    assert "--curriculum-recovery-physical-gate-contract-sha256" in inner
    assert "CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT_SHA256" in outer


def test_phase24_runner_reuses_controls_and_runs_only_five_candidates() -> None:
    runner = (ROOT / "tools/official_x2/run_phase24_physical_handoff_gate.sh").read_text()
    assert "REPEATS=5" in runner
    assert "phase23_fixed_handoff" in runner and "phase23_gated_handoff" in runner
    assert "missing Phase23 control" in runner
    assert "phase24_physical_gated_handoff" in runner
    assert "CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT=" in runner
    assert "phase21_outcome_aware_f005_u5_actor.onnx" in runner
    assert "train_x2" not in runner
