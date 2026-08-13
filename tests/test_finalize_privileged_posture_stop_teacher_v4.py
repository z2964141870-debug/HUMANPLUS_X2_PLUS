from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "tools/retarget/finalize_x2_privileged_posture_stop_teacher_v4.py"


def test_finalizer_recomputes_rows_gates_bootstrap_and_decision() -> None:
    source = FINALIZER.read_text()
    ast.parse(source)
    assert "summarize(rows)" in source
    assert "feasibility_gates" in source
    assert "paired_bootstrap" in source
    assert 'screen["decision"] != decision' in source
    assert "verify_sidecar(path)" in source
    assert "resource_checks" in source


def test_finalizer_never_unlocks_training_or_deployment() -> None:
    source = FINALIZER.read_text()
    assert '"training_unlocked": False' in source
    assert '"deployment_unlocked": False' in source
    assert '"whole_body_training_unlocked": False' in source
