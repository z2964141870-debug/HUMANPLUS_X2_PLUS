from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_x2_privileged_posture_stop_teacher_v4.py"
SHELL = ROOT / "scripts/run_x2_privileged_posture_stop_teacher_v4.sh"
V3_RUNNER = ROOT / "scripts/run_x2_privileged_feedback_teacher_search_v3.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_is_preserved_and_superseded_not_rewritten() -> None:
    assert sha256(V3_RUNNER) == "bcbddf8a7584bb48842db4253bad2f50f563b45e8aa2bd4cbf1b09d5cd17004a"
    assert "privileged_feedback_teacher_search_v3" not in SHELL.read_text()


def test_runner_has_three_treatments_and_no_learning_path() -> None:
    source = RUNNER.read_text()
    ast.parse(source)
    for treatment in ("source_direct", "fixed_half_direct", "feedback_gated"):
        assert treatment in source
    for forbidden in ("torch.optim", ".backward(", ".learn(", "compute_returns("):
        assert forbidden not in source
    assert '"scalar_reward_used": False' in source
    assert '"critic_used": False' in source


def test_horizon_and_phase_contract_are_explicit() -> None:
    source = RUNNER.read_text()
    assert "CRUISE_START_S = 2.0" in source
    assert "DECEL_START_S = 6.2" in source
    assert "HOLD_START_S = 8.2" in source
    assert "default=820" in source
    shell = SHELL.read_text()
    assert "--steps 820" in shell
    assert "--num-envs 256" in shell
    assert "--timeout-seconds 1200" in shell


def test_shell_is_fail_closed_and_never_unlocks_training() -> None:
    source = SHELL.read_text()
    assert "set -euo pipefail" in source
    assert "sha256sum -c" in source
    assert ".technical_checks[]" in source
    assert ".evidence_boundary.training_unlocked==false" in source
    assert ".evidence_boundary.deployment_unlocked==false" in source
    assert "disk_used_delta_bytes<=536870912" in source
    assert "finalize_x2_privileged_posture_stop_teacher_v4.py" in source
