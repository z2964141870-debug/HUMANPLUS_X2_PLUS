from pathlib import Path

import pytest

from cwi_x2.locomotion_zero_confirmation import checkerboard_assignment
from tools.retarget.finalize_x2_locomotion_zero_hold_confirmation import (
    resource_checks,
    validate_rows,
)


def test_validate_rows_rejects_treatment_mismatch():
    assignments = checkerboard_assignment(256, 0)
    rows = [
        {"env_id": env_id, "segment": segment, "treatment": assignments[env_id]}
        for env_id in range(256)
        for segment in ("cruise", "decelerate", "hold")
    ]
    validate_rows(rows, assignments)
    rows[0]["treatment"] = "locomotion_zero" if rows[0]["treatment"] == "direct_mix" else "direct_mix"
    with pytest.raises(RuntimeError, match="assignment"):
        validate_rows(rows, assignments)


def test_resource_checks_fail_closed_on_signal(tmp_path: Path):
    screen = tmp_path / "screen.json"
    screen.write_text("{}", encoding="utf-8")
    resource = {
        "schema": "x2_gpu_deadline_ledger_phase76_v1",
        "label": "expected",
        "exit_code": 0,
        "raw_returncode": 0,
        "autonomous_exit": True,
        "timed_out": False,
        "term_sent": False,
        "kill_sent": False,
        "forced_cleanup": False,
        "elapsed_s": 100.0,
        "gpu": {"memory_used_peak_mib": 3000},
        "disk_after": {"free_bytes": 100_000},
    }
    prereg = {"resources": {
        "elapsed_max_s": 300, "gpu_peak_mib_max": 8192,
        "free_after_bytes_min": 10_000, "screen_bytes_max": 1000,
    }}
    assert all(resource_checks(resource, screen, prereg, "expected").values())
    resource["term_sent"] = True
    assert not resource_checks(resource, screen, prereg, "expected")["no_timeout_or_signal"]


def test_runner_success_path_bypasses_native_close():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_x2_locomotion_zero_hold_confirmation.py").read_text(
        encoding="utf-8"
    )
    assert "simulation_app.close" not in source
    assert "atomic_json(args.report, report)" in source
    assert "os._exit(0)" in source
    assert source.index("atomic_json(args.report, report)") < source.index("os._exit(0)")
