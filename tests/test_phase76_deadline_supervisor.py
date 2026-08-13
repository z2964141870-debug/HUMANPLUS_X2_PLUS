from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "tools/retarget/run_with_gpu_deadline_ledger_phase76.py"


def run_supervisor(tmp_path: Path, child_code: str, timeout: float = 1.0, grace: float = 0.2):
    resource = tmp_path / "resource.json"
    log = tmp_path / "child.log"
    command = [
        sys.executable,
        str(SUPERVISOR),
        "--label",
        "unit",
        "--resource-output",
        str(resource),
        "--log",
        str(log),
        "--disk-path",
        str(tmp_path),
        "--timeout-seconds",
        str(timeout),
        "--term-grace-seconds",
        str(grace),
        "--",
        sys.executable,
        "-c",
        child_code,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    payload = json.loads(resource.read_text())
    return completed, payload, log


def test_autonomous_direct_exit_is_the_only_success(tmp_path: Path) -> None:
    completed, payload, log = run_supervisor(tmp_path, "import os; print('ready', flush=True); os._exit(0)")
    assert completed.returncode == 0
    assert payload["raw_returncode"] == 0
    assert payload["autonomous_exit"] is True
    assert payload["timed_out"] is False
    assert payload["term_sent"] is False
    assert payload["kill_sent"] is False
    assert payload["forced_cleanup"] is False
    assert payload["exit_code"] == 0
    assert log.read_text().strip() == "ready"
    assert resource_sidecar_ok(tmp_path / "resource.json")
    assert resource_sidecar_ok(log)


def test_timeout_remains_failure_when_child_handles_term_as_success(tmp_path: Path) -> None:
    code = (
        "import signal,time,sys; "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
        "print('waiting', flush=True); time.sleep(10)"
    )
    completed, payload, _ = run_supervisor(tmp_path, code, timeout=0.25, grace=0.5)
    assert completed.returncode != 0
    assert payload["timed_out"] is True
    assert payload["term_sent"] is True
    assert payload["raw_returncode"] == 0
    assert payload["autonomous_exit"] is False
    assert payload["exit_code"] != 0


def test_ignored_term_escalates_to_kill_and_fails(tmp_path: Path) -> None:
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)"
    completed, payload, _ = run_supervisor(tmp_path, code, timeout=0.25, grace=0.15)
    assert completed.returncode != 0
    assert payload["timed_out"] is True
    assert payload["term_sent"] is True
    assert payload["kill_sent"] is True
    assert payload["raw_returncode"] < 0
    assert payload["exit_code"] != 0


def resource_sidecar_ok(path: Path) -> bool:
    import hashlib

    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.read_text() == f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
