#!/usr/bin/env python3
"""Build the immutable source-reference-only BASE Phase18 curriculum asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_x2.recovery_suffix_aggregation import sha256_file
from official_x2.role_aware_recovery_curriculum import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    official = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
    result_root = official / "results/official_native_strict_20260807"
    parser.add_argument(
        "--output", type=Path,
        default=Path("manifests/x2_phase17_role_aware_reset_curriculum.json"),
    )
    parser.add_argument(
        "--urdf", type=Path,
        default=Path("/home/humanplus/x2_teleop_final/assets/agibot_x2/x2_ultra_sonic_sole12.urdf"),
    )
    args = parser.parse_args()
    sidecars = [
        result_root / f"phase17_stage326_stop_event_stateful_r{index}_sidecar.json"
        for index in range(1, 5)
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
