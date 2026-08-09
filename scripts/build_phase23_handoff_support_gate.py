#!/usr/bin/env python3
"""Build the immutable Phase23 previous-action support contract."""

import argparse
import json
from pathlib import Path

import numpy as np

from official_x2.handoff_support_gate import build_contract, file_sha256
from official_x2.outcome_aware_state_role_v2 import load_manifest, resolve_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/x2_phase19_outcome_aware_state_role.json"))
    parser.add_argument("--output", type=Path, default=Path("manifests/x2_phase23_previous_action_support_gate.json"))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    indices = [int(row["manifest_row_index"]) for row in manifest["rows"] if row["eligible"]]
    rows = resolve_rows(manifest, indices)
    previous = np.asarray([row["actual_previous_action_input"] for row in rows], dtype=np.float64)
    contract = build_contract(previous, source_manifest_sha256=file_sha256(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "file_sha256": file_sha256(args.output), "content_sha256": contract["content_sha256"], "threshold": contract["threshold_value"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
