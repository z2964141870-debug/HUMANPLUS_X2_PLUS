from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from cwi_x2.phase76_pairing_preflight import copy_pair_rows, pair_diagnostics, paired_lane_ids


ROOT = Path(__file__).resolve().parents[1]
EXEC_LINE = 'exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())'


def mechanically_expand_phase76_runner() -> str:
    wrapper_path = ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.py"
    wrapper = wrapper_path.read_text()
    assert wrapper.count(EXEC_LINE) == 1
    namespace: dict[str, object] = {"__file__": str(wrapper_path)}
    exec(compile(wrapper.replace(EXEC_LINE, "TRANSFORMED_WRAPPER = source"), str(wrapper_path), "exec"), namespace)
    transformed_wrapper = namespace["TRANSFORMED_WRAPPER"]
    assert isinstance(transformed_wrapper, str)
    assert transformed_wrapper.count(EXEC_LINE) == 1
    second: dict[str, object] = {"__file__": str(wrapper_path)}
    exec(
        compile(transformed_wrapper.replace(EXEC_LINE, "FINAL_RUNNER = source"), str(wrapper_path), "exec"),
        second,
    )
    final_runner = second["FINAL_RUNNER"]
    assert isinstance(final_runner, str)
    return final_runner


def test_phase75_frozen_anchor_is_exact() -> None:
    frozen = ROOT / "scripts/run_x2_phase75_pairing_preflight.py"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        "a08cc5d79b87beec71394c71b2c83800fc1e84fded0d7672123dbf27dc24c107"
    )


def test_phase76_transform_only_changes_lifecycle_after_frozen_clone_contract() -> None:
    text = (ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.py").read_text()
    assert "EXPECTED_PHASE75_SHA256" in text
    assert "Phase76 underlying ledger guard anchor changed" in text
    assert "run_with_gpu_deadline_ledger_phase76.py" in text
    assert "os._exit(0)" in text
    assert "os.fsync" in text
    assert "if 'simulation_app.close()' in source" in text
    assert "phase77_shadow_preregistration_unlocked" not in text  # produced only after runtime transform


def test_mechanically_expanded_runner_has_only_new_success_lifecycle() -> None:
    expanded = mechanically_expand_phase76_runner()
    assert "simulation_app.close()" not in expanded
    assert "os._exit(0)" in expanded
    assert "os.fsync" in expanded
    assert "run_with_gpu_deadline_ledger_phase76.py" in expanded
    assert '"supervisor_test_sha256": ROOT / "tests/test_phase76_deadline_supervisor.py"' in expanded
    assert '"phase77_shadow_preregistration_unlocked": False' in expanded
    assert '"phase77_scientific_preregistration_unlocked": False' in expanded
    assert '"phase77_launch_unlocked": False' in expanded
    assert "env.step(" not in expanded
    assert "reward_manager.compute" not in expanded
    compile(expanded, str(ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.py"), "exec")


def test_phase76_helpers_preserve_mixed_device_pairing_contract() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    donors, recipients = paired_lane_ids(0, 4)
    copy_pair_rows(value, donors, recipients)
    assert pair_diagnostics(value, donors, recipients)["passed"] is True


def test_phase76_remains_initial_only() -> None:
    frozen = (ROOT / "scripts/run_x2_phase74_pairing_preflight.py").read_text()
    assert "env.step(" not in frozen
    assert "reward_manager.compute" not in frozen
    assert "torch.optim" not in frozen
    assert ".backward(" not in frozen
