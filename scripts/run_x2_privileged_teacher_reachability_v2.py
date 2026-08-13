#!/usr/bin/env python3
"""Repair-only wrapper for the frozen privileged-teacher v2 attempt0 runner."""

from __future__ import annotations

import hashlib
from pathlib import Path


BASE = Path(__file__).with_name("run_x2_privileged_teacher_reachability_v2_attempt0.py")
EXPECTED_BASE_SHA256 = "c221fdfb75de51dd99e1dd77af4da8fc485d9aa017a7a4ffea39d381f31b5d06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(BASE) != EXPECTED_BASE_SHA256:
    raise RuntimeError("frozen attempt0 runner drifted")

source = BASE.read_text(encoding="utf-8")
needle = "    with torch.inference_mode():\n"
replacement = "    with torch.no_grad():\n"
if source.count(needle) != 1:
    raise RuntimeError("repair anchor count changed")
transformed = source.replace(needle, replacement, 1)
if transformed.count("with torch.inference_mode():") != 0:
    raise RuntimeError("unrepaired inference-mode environment loop remains")

namespace = {
    "__file__": str(Path(__file__).resolve()),
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(transformed, str(BASE), "exec"), namespace, namespace)
