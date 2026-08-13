#!/usr/bin/env python3
"""Execute the frozen Phase72 finalizer with Phase73 namespaces."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/retarget/finalize_x2_phase72_antithetic.py"
SOURCE_SHA256 = "0e7e3cbf4c292f53984d54607d382e1c4326d9da56d6ecce6af3de61f8c20197"
raw = SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("Phase73 frozen Phase72 finalizer hash changed")
text = raw.decode().replace("phase72", "phase73").replace("Phase72", "Phase73")
exec(compile(text, str(Path(__file__).resolve()), "exec"), globals())
