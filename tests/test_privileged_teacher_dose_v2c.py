from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_x2_privileged_teacher_dose_v2c.py"
BASE = ROOT / "scripts/run_x2_privileged_teacher_reachability_v2_attempt0.py"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cyclic_assignment_exposes_every_env_to_every_scale_once() -> None:
    ids = np.arange(64)
    observed = {scale_id: [] for scale_id in range(8)}
    for pass_index in range(8):
        scale_ids = (ids + pass_index) % 8
        for scale_id in range(8):
            observed[scale_id].extend(ids[scale_ids == scale_id].tolist())
    assert all(sorted(env_ids) == list(range(64)) for env_ids in observed.values())


def test_runner_is_anchored_to_frozen_attempt0_and_has_no_policy_update() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    assert file_sha256(BASE) == "c221fdfb75de51dd99e1dd77af4da8fc485d9aa017a7a4ffea39d381f31b5d06"
    assert "scale_by_env * full_direction" in text
    assert '"candidate_zero_termination"' in text
    assert '"cyclic_assignment_complete"' in text
    assert ".backward(" not in text
    assert "torch.optim" not in text
    assert "torch.save" not in text


def test_mechanical_runner_expansion_compiles() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    replacement = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "replacement" for target in node.targets):
            continue
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)
        dedent_call = node.value.func.value
        assert isinstance(dedent_call, ast.Call)
        assert isinstance(dedent_call.args[0], ast.Constant)
        replacement = str(dedent_call.args[0].value).lstrip()
        break
    assert replacement is not None
    source = BASE.read_text(encoding="utf-8").replace(
        "    with torch.inference_mode():\n", "    with torch.no_grad():\n", 1
    )
    start = source.index("def run() -> dict:\n")
    end = source.index("\ndef fail(error: BaseException) -> None:\n", start)
    expanded = source[:start] + replacement + source[end:]
    compile(expanded, "<phase-v2c-expanded>", "exec")
    assert expanded.count("def run() -> dict:") == 1


def test_dose_grid_brackets_material_threshold_region() -> None:
    scales = np.asarray([0.0, 0.10, 0.125, 0.15, 0.20, 0.25, 0.50, 1.0])
    assert scales[0] == 0.0 and scales[-1] == 1.0
    assert np.all(np.diff(scales) > 0.0)
    assert {0.10, 0.125, 0.15, 0.20, 0.25}.issubset(set(scales.tolist()))
