import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from official_x2.handoff_support_gate import (
    build_contract,
    handoff_gate_decision,
    load_contract,
    previous_action_distance,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "manifests/x2_phase23_previous_action_support_gate.json"


def test_frozen_contract_loads_with_file_hash_and_reference_is_supported() -> None:
    expected = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    payload = load_contract(CONTRACT, expected)
    reference = np.asarray(payload["reference_previous_action"], dtype=np.float64)
    assert reference.shape == (432, 15)
    assert previous_action_distance(reference[0], payload) == pytest.approx(0.0)
    decision = handoff_gate_decision(reference[0], [0.0, 0.0, 1.0, 1.0], payload)
    assert decision["previous_action_in_support"]
    assert decision["generator_double_support"]
    assert decision["allow_handoff"]


def test_gate_requires_both_support_domain_and_generator_double_support() -> None:
    payload = load_contract(CONTRACT)
    reference = np.asarray(payload["reference_previous_action"], dtype=np.float64)
    assert not handoff_gate_decision(reference[0], [0.0, 0.0, 1.0, 0.0], payload)[
        "allow_handoff"
    ]
    far = reference[0] + 100.0 * np.asarray(payload["scale"])
    decision = handoff_gate_decision(far, [0.0, 0.0, 1.0, 1.0], payload)
    assert not decision["previous_action_in_support"]
    assert not decision["allow_handoff"]


def test_contract_threshold_is_leave_one_out_not_self_distance() -> None:
    rng = np.random.default_rng(7)
    payload = build_contract(rng.normal(size=(32, 15)), source_manifest_sha256="a" * 64)
    assert payload["threshold_value"] > 0.0
    assert payload["reference_count"] == 32


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    copy = tmp_path / "gate.json"
    copy.write_bytes(CONTRACT.read_bytes())
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_contract(copy, "0" * 64)
    payload = json.loads(copy.read_text())
    payload["threshold_value"] += 1.0
    copy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_contract(copy)


def test_adapter_gate_is_default_off_and_isolated_to_curriculum_branch() -> None:
    source = (ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py").read_text()
    parser = source[source.index("def parse_args()") : source.index("def main()")]
    assert '"--curriculum-recovery-handoff-gate-contract"' in parser
    assert "handoff gate requires --stop-controller curriculum_then_policy" in parser
    start = source.index('elif self.args.stop_controller == "curriculum_then_policy":')
    end = source.index('elif self.args.stop_controller == "ramp_policy":', start)
    branch = source[start:end]
    assert source.count("handoff_gate_decision(") == 1
    assert "self.curriculum_handoff_gate_contract is not None" in branch
    assert "gate_waiting = False" in branch
    assert "if not gate_waiting:" in branch
    # A wait must flow through the existing common publish/record path so the
    # 50 Hz control-step clock and stop horizon remain unchanged.
    assert "return" not in branch
    assert "self._publish(" not in branch
    assert "self._record(" not in branch


def test_default_off_path_keeps_historical_handoff_statement_order() -> None:
    source = (ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py").read_text()
    start = source.index('elif self.args.stop_controller == "curriculum_then_policy":')
    end = source.index('elif self.args.stop_controller == "ramp_policy":', start)
    branch = source[start:end]
    # When contract is None gate_waiting stays false and the existing sequence
    # remains copy-history -> latch -> recovery inference -> target blend.
    sequence = (
        "self.previous_actions[stop_slot] = self.previous_actions[",
        "self.stop_hold_latch_s = stop_elapsed",
        "targets, obs, action = self._policy_targets(",
        "targets = blend_curriculum_recovery_targets(",
    )
    positions = []
    cursor = 0
    for item in sequence:
        cursor = branch.index(item, cursor)
        positions.append(cursor)
    assert positions == sorted(positions)


def test_official_runner_plumbs_contract_only_when_nonempty() -> None:
    inner = (ROOT / "tools/official_x2/run_official_gate_inner.sh").read_text()
    outer = (ROOT / "tools/official_x2/run_official_gate_case.sh").read_text()
    assert 'CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT="${CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT:-}"' in inner
    assert '[[ -n "$CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT" ]]' in inner
    assert "--curriculum-recovery-handoff-gate-contract-sha256" in inner
    assert "CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT_SHA256" in outer


def test_phase23_runner_is_training_free_single_variable_five_by_five() -> None:
    runner = (ROOT / "tools/official_x2/run_phase23_delayed_handoff_ab.sh").read_text()
    assert "labels=(fixed gated)" in runner
    assert "REPEATS=5" in runner
    assert "phase21_outcome_aware_f005_u5_actor.onnx" in runner
    assert "CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS=0.5" in runner
    assert "STOP_TRANSITION_SECONDS=2.0" in runner
    assert "PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2" in runner
    assert "CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT=/repo/manifests/x2_phase23_previous_action_support_gate.json" in runner
    assert 'if [[ "$label" == "gated" ]]' in runner
    assert "train_x2" not in runner
