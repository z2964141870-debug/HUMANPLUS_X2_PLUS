#!/usr/bin/env python3
"""Execute the frozen Phase75 clone contract with fail-fast Phase76 success termination."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PHASE75 = ROOT / "scripts/run_x2_phase75_pairing_preflight.py"
EXPECTED_PHASE75_SHA256 = "a08cc5d79b87beec71394c71b2c83800fc1e84fded0d7672123dbf27dc24c107"

source = FROZEN_PHASE75.read_text(encoding="utf-8")
if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_PHASE75_SHA256:
    raise RuntimeError("frozen Phase75 runner drifted before Phase76 lifecycle transform")

source = source.replace("Phase75", "Phase76").replace("phase75", "phase76")
for permission_suffix in (
    "shadow_preregistration_unlocked",
    "scientific_preregistration_unlocked",
    "launch_unlocked",
):
    current = f'"phase76_{permission_suffix}"'
    if source.count(current) != 2:
        raise RuntimeError(f"Phase76 permission transform changed: {permission_suffix}")
    source = source.replace(current, f'"phase75_{permission_suffix}"', 1)
    source = source.replace(current, f'"phase77_{permission_suffix}"', 1)

if source.count('ROOT / "tools/retarget/run_with_gpu_ledger.py"') != 0:
    # The wrapper source does not contain the underlying Phase74 guard. The loaded source does.
    raise RuntimeError("unexpected direct ledger path in Phase76 wrapper")

loop_anchor = 'for old, new in replacements.items():\n'
if source.count(loop_anchor) != 1:
    raise RuntimeError("Phase76 wrapper transform loop changed")

injection_anchor = 'if "value[donors]" in source or "value[recipients]" in source or "value[:, recipients]" in source:\n'
if source.count(injection_anchor) != 1:
    raise RuntimeError("Phase76 wrapper post-transform anchor changed")

success_old = "else:\n    simulation_app.close()"
success_new = (
    "else:\n"
    "    evidence_paths = (\n"
    "        args.commit, args.commit.with_suffix(args.commit.suffix + '.sha256'),\n"
    "        args.output, args.output.with_suffix(args.output.suffix + '.sha256'),\n"
    "    )\n"
    "    for evidence_path in evidence_paths:\n"
    "        with evidence_path.open('rb') as evidence_stream:\n"
    "            os.fsync(evidence_stream.fileno())\n"
    "    for evidence_dir in {path.parent for path in evidence_paths}:\n"
    "        directory_fd = os.open(evidence_dir, os.O_RDONLY)\n"
    "        try:\n"
    "            os.fsync(directory_fd)\n"
    "        finally:\n"
    "            os.close(directory_fd)\n"
    "    sys.stdout.flush()\n"
    "    sys.stderr.flush()\n"
    "    os._exit(0)"
)
injection = (
    "if source.count('ROOT / \\\"tools/retarget/run_with_gpu_ledger.py\\\"') != 1:\n"
    "    raise RuntimeError('Phase76 underlying ledger guard anchor changed')\n"
    "source = source.replace(\n"
    "    'ROOT / \\\"tools/retarget/run_with_gpu_ledger.py\\\"',\n"
    "    'ROOT / \\\"tools/retarget/run_with_gpu_deadline_ledger_phase76.py\\\"',\n"
    ")\n"
    "phase76_ledger_guard = '\\\"ledger_sha256\\\": ROOT / \\\"tools/retarget/run_with_gpu_deadline_ledger_phase76.py\\\",'\n"
    "if source.count(phase76_ledger_guard) != 1:\n"
    "    raise RuntimeError('Phase76 ledger code guard entry changed')\n"
    "source = source.replace(\n"
    "    phase76_ledger_guard,\n"
    "    phase76_ledger_guard + '\\n        \\\"supervisor_test_sha256\\\": ROOT / \\\"tests/test_phase76_deadline_supervisor.py\\\",',\n"
    ")\n"
    "phase76_code_paths = {\n"
    "    'ROOT / \\\"scripts/run_x2_phase76_pairing_preflight.sh\\\"': 'ROOT / \\\"scripts/run_x2_phase76_pairing_lifecycle_preflight.sh\\\"',\n"
    "    'ROOT / \\\"tests/test_phase76_pairing_preflight.py\\\"': 'ROOT / \\\"tests/test_phase76_pairing_lifecycle_preflight.py\\\"',\n"
    "    'ROOT / \\\"tools/retarget/finalize_x2_phase76_pairing_preflight.py\\\"': 'ROOT / \\\"tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py\\\"',\n"
    "    'ROOT / \\\"tests/test_phase76_pairing_finalizer.py\\\"': 'ROOT / \\\"tests/test_phase76_pairing_lifecycle_finalizer.py\\\"',\n"
    "}\n"
    "for phase76_old_path, phase76_new_path in phase76_code_paths.items():\n"
    "    if source.count(phase76_old_path) != 1:\n"
    "        raise RuntimeError(f'Phase76 code guard path anchor changed: {phase76_old_path}')\n"
    "    source = source.replace(phase76_old_path, phase76_new_path)\n"
    f"success_old = {success_old!r}\n"
    f"success_new = {success_new!r}\n"
    "if source.count(success_old) != 1:\n"
    "    raise RuntimeError('Phase76 success shutdown anchor changed')\n"
    "source = source.replace(success_old, success_new)\n"
    "if 'simulation_app.close()' in source:\n"
    "    raise RuntimeError('Phase76 transformed runner retains blocking success shutdown')\n\n"
)
source = source.replace(injection_anchor, injection + injection_anchor)

exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())
