#!/usr/bin/env python3
"""Offline regression checks for the preloaded fixed-support launcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = DEPLOY_ROOT / "run_x2_suspended_sonic.sh"
WRAPPER = DEPLOY_ROOT / "run_x2_supported_preloaded_fixed100.sh"
HANDOFF = DEPLOY_ROOT / "deploy_x2.sh"
TOKEN = "X2_SUSPENDED_SONIC_START"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assignment(script: str, name: str) -> str:
    match = re.search(rf'^{re.escape(name)}="([^"]+)"$', script, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing shell assignment: {name}")
    return match.group(1)


def touch(path: Path, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if executable:
        path.chmod(0o755)


def capture_runner_args(mode: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="x2_fixed100_profile.") as tmp:
        root = Path(tmp)
        project = root / "project"
        deploy = project / "GR00T_audit" / "gear_sonic_deploy"
        deploy.mkdir(parents=True)
        runner_copy = deploy / RUNNER.name
        shutil.copy2(RUNNER, runner_copy)
        # macOS ships Bash 3.2, where nounset treats an explicitly empty array
        # expansion as unbound. The robot uses Bash 5; disabling nounset only
        # in this disposable fixture lets us inspect the generated argv.
        runner_text = runner_copy.read_text(encoding="utf-8")
        runner_copy.write_text(
            runner_text.replace("set -euo pipefail", "set -eo pipefail", 1),
            encoding="utf-8",
        )
        runner_copy.chmod(0o755)

        stub = deploy / "deploy_x2.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' __X2_ARGS_BEGIN__\n"
            "printf '%s\\n' \"$@\"\n",
            encoding="ascii",
        )
        stub.chmod(0o755)

        runtime = deploy / "runtime_suspended"
        touch(runtime / "ws" / "install" / "setup.bash")
        touch(runtime / "venv" / "bin" / "python3", executable=True)
        touch(runtime / "onnxruntime" / "lib" / "libonnxruntime.so")
        candidate_ws = runtime / "ws_ground_debug_20260831"
        touch(candidate_ws / "install" / "setup.bash")
        touch(project / "models" / "x2_sonic_frozen_g1core_lora_v2.onnx")

        fake_home = root / "home"
        (fake_home / "aimdk_ws_0_8_18" / "install" / "aimdk_msgs" /
         "local" / "lib" / "python3.10" / "dist-packages").mkdir(
             parents=True)

        env = os.environ.copy()
        env.update(
            HOME=str(fake_home),
            X2_ONBOT_WS=str(candidate_ws),
            X2_SONIC_WRITER_HZ="250",
        )
        for name in ("X2_ADOPT_PAUSED_MC", "X2_ADOPT_DEBUG_PORT",
                     "X2_OBS_DUMP_PATH"):
            env.pop(name, None)

        completed = subprocess.run(
            [str(deploy / RUNNER.name), TOKEN, mode],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"runner fixture failed for {mode}: rc={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        lines = completed.stdout.splitlines()
        marker = lines.index("__X2_ARGS_BEGIN__")
        return lines[marker + 1 :]


def option_value(args: list[str], option: str) -> str:
    index = args.index(option)
    return args[index + 1]


def normalized_profile(args: list[str]) -> list[str]:
    result = list(args)
    index = result.index("--supported-policy-seconds")
    result[index + 1] = "<duration>"
    for option in (
        "--model",
        "--log-dir",
        "--onbot-prefix",
        "--onbot-ws",
        "--onbot-venv",
        "--onbot-onnxruntime",
        "--onbot-aimdk-prefix",
    ):
        index = result.index(option)
        result[index + 1] = f"<{option[2:]}>"
    return result


def test_wrapper_identity() -> None:
    script = WRAPPER.read_text(encoding="utf-8")
    assert assignment(script, "EXPECTED_POWERED_LAUNCHER_SHA256") == sha256(RUNNER)
    assert assignment(script, "EXPECTED_HANDOFF_LAUNCHER_SHA256") == sha256(HANDOFF)
    assert '"${1:-}" == "--verify-only"' in script
    assert "FIXED100_ARTIFACTS_VERIFIED: no process started" in script
    assert "unset X2_ADOPT_PAUSED_MC X2_ADOPT_DEBUG_PORT X2_OBS_DUMP_PATH" in script
    assert "--supported-neutral-damped-relative-100" in script


def test_fixed100_profile_matches_frozen_parent() -> None:
    fixed100 = capture_runner_args("--supported-neutral-damped-relative-100")
    frozen300 = capture_runner_args("--supported-neutral-damped-relative-300")

    assert option_value(fixed100, "--supported-policy-seconds") == "100.0"
    assert option_value(frozen300, "--supported-policy-seconds") == "300.0"
    assert normalized_profile(fixed100) == normalized_profile(frozen300)

    assert option_value(fixed100, "--writer-hz") == "250"
    assert option_value(fixed100, "--target-lpf-hz") == "8.0"
    assert option_value(fixed100, "--supported-policy-target-rate") == "0.12"
    assert "--supported-policy-entry-relative-to-load" in fixed100
    assert "--support-step-probe" not in fixed100
    assert "--vla" not in fixed100


if __name__ == "__main__":
    test_wrapper_identity()
    test_fixed100_profile_matches_frozen_parent()
    print("FIXED100_LAUNCHER_TEST_PASS")
