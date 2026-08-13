#!/usr/bin/env python3
"""Execute the frozen Phase72 pair validator with Phase73 namespaces."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/retarget/validate_x2_phase72_pair.py"
SOURCE_SHA256 = "f744bbb03f1dd1ce7de4ad79c2b17aa52b76cf49225455ad219f941aadd587bf"
raw = SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("Phase73 frozen Phase72 pair validator hash changed")
text = raw.decode().replace("phase72", "phase73").replace("Phase72", "Phase73")
exec(compile(text, str(Path(__file__).resolve()), "exec"), globals())
