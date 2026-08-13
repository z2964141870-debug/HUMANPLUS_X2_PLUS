from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_phase73_shell_is_fresh_and_fail_closed() -> None:
    text = (ROOT / "scripts/run_x2_phase73_antithetic.sh").read_text()
    for token in (
        "730041",
        "730045",
        "phase73_seed",
        "cumulative_disk_delta",
        "seen_initial_hashes",
        "bundle_bytes",
        "gpu_peak",
        "free_after",
    ):
        assert token in text
    assert "phase72_seed" not in text


def test_phase73_wrappers_pin_phase72_sources() -> None:
    expected = {
        "scripts/run_x2_phase73_antithetic.py": "444482b29d05303b6b957b0f043d588c7f67b5bedc231912e33f7e47cccfb335",
        "tools/retarget/validate_x2_phase73_pair.py": "f744bbb03f1dd1ce7de4ad79c2b17aa52b76cf49225455ad219f941aadd587bf",
        "tools/retarget/finalize_x2_phase73_antithetic.py": "0e7e3cbf4c292f53984d54607d382e1c4326d9da56d6ecce6af3de61f8c20197",
    }
    for relative, digest in expected.items():
        assert digest in (ROOT / relative).read_text()


def test_gpu_ledger_propagates_deliberate_child_failure(tmp_path: Path) -> None:
    resource = tmp_path / "resource.json"
    log = tmp_path / "child.log"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/retarget/run_with_gpu_ledger.py"),
            "--label",
            "phase73_deliberate_failure_contract",
            "--resource-output",
            str(resource),
            "--log",
            str(log),
            "--disk-path",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "raise RuntimeError('deliberate Phase73 exit propagation probe')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(resource.read_text())["exit_code"] == 1
    assert "deliberate Phase73 exit propagation probe" in log.read_text()


def test_repaired_close_cannot_mask_failure_from_ledger(tmp_path: Path) -> None:
    resource = tmp_path / "resource_close_mask.json"
    log = tmp_path / "close_mask.log"
    sentinel = tmp_path / "unreachable.txt"
    harness = tmp_path / "repaired_exit_harness.py"
    harness.write_text(
        """import os
import sys
import traceback
from pathlib import Path

sentinel = Path(sys.argv[1])

class DummyApp:
    def close(self):
        raise SystemExit(0)

simulation_app = DummyApp()

def run():
    raise RuntimeError('deliberate Phase73 repaired-close probe')

try:
    run()
except BaseException:
    traceback.print_exc()
    try:
        simulation_app.close()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
else:
    simulation_app.close()

sentinel.write_text('unreachable')
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/retarget/run_with_gpu_ledger.py"),
            "--label",
            "phase73_close_mask_failure_contract",
            "--resource-output",
            str(resource),
            "--log",
            str(log),
            "--disk-path",
            str(tmp_path),
            "--",
            sys.executable,
            str(harness),
            str(sentinel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(resource.read_text())["exit_code"] == 1
    assert "deliberate Phase73 repaired-close probe" in log.read_text()
    assert not sentinel.exists()
