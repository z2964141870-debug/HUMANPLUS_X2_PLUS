#!/usr/bin/env python3
"""Run the frozen Phase74 finalizer under the new Phase75 technical namespace."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "tools/retarget/finalize_x2_phase74_pairing_preflight.py"
EXPECTED_SHA256 = "48ea1b3fc8de1500c13f371b110a901c162d457227d43e3421f75d3927b16955"

source = FROZEN.read_text(encoding="utf-8")
if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("frozen Phase74 finalizer drifted before Phase75 transform")
source = source.replace("Phase74", "Phase75").replace("phase74", "phase75")
source = source.replace("phase75_shadow_preregistration_unlocked", "phase76_shadow_preregistration_unlocked")
source = source.replace("phase75_scientific_preregistration_unlocked", "phase76_scientific_preregistration_unlocked")
source = source.replace("phase75_launch_unlocked", "phase76_launch_unlocked")
exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())

