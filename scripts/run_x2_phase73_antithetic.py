#!/usr/bin/env python3
"""Execute the audited Phase72 runner as a fresh Phase73 serialization repair."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_x2_phase72_antithetic.py"
SOURCE_SHA256 = "444482b29d05303b6b957b0f043d588c7f67b5bedc231912e33f7e47cccfb335"
OLD_FINITE = '"finite": all(torch.isfinite(value).all() for value in tensors.values()) and torch.isfinite(encoded).all(),'
NEW_FINITE = '"finite": bool(all(bool(torch.isfinite(value).all()) for value in tensors.values()) and bool(torch.isfinite(encoded).all())), '
OLD_EXIT = '''failure = None
try:
    run()
except Exception as exc:
    failure = exc
    traceback.print_exc()
finally:
    simulation_app.close()
if failure is not None:
    raise failure
'''
NEW_EXIT = '''try:
    run()
except BaseException:
    traceback.print_exc()
    try:
        simulation_app.close()
    finally:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
else:
    simulation_app.close()
'''


def repaired_source() -> str:
    """Return the only authorized Phase73 runner source transformation."""

    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Phase73 frozen Phase72 source runner hash changed")
    text = raw.decode()
    if text.count(OLD_FINITE) != 1 or text.count(OLD_EXIT) != 1:
        raise RuntimeError("Phase73 serialization/exit repair anchor changed")
    text = text.replace(OLD_FINITE, NEW_FINITE)
    text = text.replace(OLD_EXIT, NEW_EXIT)
    text = text.replace("phase72", "phase73").replace("Phase72", "Phase73")
    return text


exec(compile(repaired_source(), str(Path(__file__).resolve()), "exec"), globals())
