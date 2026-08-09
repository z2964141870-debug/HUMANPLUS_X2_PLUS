#!/usr/bin/env python3
"""Build immutable source-reference-only BASE Phase20 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_x2.outcome_aware_state_role_v2 import build_manifest, write_manifest
from official_x2.recovery_suffix_aggregation import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("manifests/x2_phase19_outcome_aware_state_role.json"),
    )
    parser.add_argument(
        "--urdf", type=Path,
        default=Path("/home/humanplus/x2_teleop_final/assets/agibot_x2/x2_ultra_sonic_sole12.urdf"),
    )
    args = parser.parse_args()
    root = Path(
        "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
        "official_native_strict_20260807"
    )
    sidecars = [
        root / f"phase19_stage326_stop_event_stateful_v2_r{index}_sidecar_v2.json"
        for index in (2, 3, 4, 5)
    ]
    manifest = build_manifest(sidecars, args.urdf)
    write_manifest(args.output, manifest)
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": sha256_file(args.output),
        "content_sha256": manifest["content_sha256"],
        "counts": manifest["counts"],
        "training_unlocked": False,
    }, indent=2))


if __name__ == "__main__":
    main()
