#!/usr/bin/env python3
"""Finalize the frozen initial-pairing contract with Phase76 lifecycle gates."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PHASE75 = ROOT / "tools/retarget/finalize_x2_phase75_pairing_preflight.py"
EXPECTED_PHASE75_SHA256 = "ee946567b0fc50afc41fee18fc72f3159aff8701d3083416b60bd41411308e5b"

source = FROZEN_PHASE75.read_text(encoding="utf-8")
if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_PHASE75_SHA256:
    raise RuntimeError("frozen Phase75 finalizer drifted before Phase76 lifecycle transform")
source = source.replace("Phase75", "Phase76").replace("phase75", "phase76")
for permission_suffix in (
    "shadow_preregistration_unlocked",
    "scientific_preregistration_unlocked",
    "launch_unlocked",
):
    current = f'"phase76_{permission_suffix}"'
    if source.count(current) != 2:
        raise RuntimeError(f"Phase76 finalizer permission transform changed: {permission_suffix}")
    source = source.replace(current, f'"phase75_{permission_suffix}"', 1)
    source = source.replace(current, f'"phase77_{permission_suffix}"', 1)

exec_anchor = 'exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())'
if source.count(exec_anchor) != 1:
    raise RuntimeError("Phase76 finalizer execution anchor changed")
injection = '''lifecycle_anchor = 'int(row.get("exit_code", -1)) == 0\\n'
if source.count(lifecycle_anchor) != 1:
    raise RuntimeError("Phase76 resource lifecycle anchor changed")
source = source.replace(
    lifecycle_anchor,
    lifecycle_anchor
    + '        and int(row.get("raw_returncode", -1)) == 0\\n'
    + '        and row.get("autonomous_exit") is True\\n'
    + '        and row.get("timed_out") is False\\n'
    + '        and row.get("term_sent") is False\\n'
    + '        and row.get("kill_sent") is False\\n'
    + '        and row.get("forced_cleanup") is False\\n',
)
'''
source = source.replace(exec_anchor, injection + exec_anchor)
exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())
