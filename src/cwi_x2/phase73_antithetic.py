"""Phase73 uses the frozen Phase72 pure estimand helpers unchanged."""

import hashlib
from pathlib import Path


_SOURCE = Path(__file__).with_name("phase72_antithetic.py")
_SOURCE_SHA256 = "9201a323e593215394879fe61c3d066020dbbe4a4574d33c250db26906310cb2"
if hashlib.sha256(_SOURCE.read_bytes()).hexdigest() != _SOURCE_SHA256:
    raise RuntimeError("Phase73 frozen Phase72 estimand helper changed")

from cwi_x2.phase72_antithetic import *  # noqa: E402,F401,F403
