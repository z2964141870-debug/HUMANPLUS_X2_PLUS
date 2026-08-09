#!/usr/bin/env python3
"""Prelaunch fail-closed guard for the frozen Phase25 physics domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from x2_physics_provenance_guard import (
    assert_physics_contract,
    load_runtime_snapshot,
    sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    actual = sha256(args.manifest)
    if actual != args.expected_manifest_sha256:
        print(json.dumps({"status": "B5_MANIFEST_HASH_REFUSED", "actual": actual}))
        return 43
    try:
        result = assert_physics_contract(args.manifest, load_runtime_snapshot(args.runtime))
    except (ValueError, RuntimeError) as error:
        print(json.dumps({"status": "B5_RUNTIME_REFUSED", "error": str(error)}))
        return 43
    print(
        json.dumps(
            {
                "status": "B5_PRELAUNCH_GUARD_PASSED",
                "manifest_sha256": actual,
                "file_count": len(result["file_checks"]),
                "runtime_exact": result["runtime_check"]["exact"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
