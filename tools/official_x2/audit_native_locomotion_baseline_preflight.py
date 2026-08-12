#!/usr/bin/env python3
"""Audit immutable inputs required to freeze and train the X2 locomotion baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "source_checkpoint": "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb",
    "actor_onnx": "b95bad3680658c7c25be50f236f070c80b7ff7ba8992355cec2ddfb1ee53c0f9",
    "gait_template": "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d",
    "official_bundle": "5bbcf724d54fb28f153db0d272f9acb7906bb1d2cac7dd7ccdc699a5c7eeab35",
    "x2_xml": "3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3",
    "scene_xml": "7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63",
    "stand_onnx": "edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    home = args.home.expanduser().resolve()
    repo = home / "projects/ZHY/CWI_CrossEmbodiment_Sim"
    official = home / "projects/ZHY/x2_official_rl_deploy_v1"
    model_info = official / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info"
    paths = {
        "source_checkpoint": home / "x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt",
        "actor_onnx": official / "models/stage219_s2600_actor.onnx",
        "gait_template": home / "x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz",
        "official_bundle": home / "x2_migration_20260812/downloads/official/aimdk-x2-v1.0.0-official.zip",
        "x2_xml": model_info / "x2.xml",
        "scene_xml": model_info / "scene.xml",
        "stand_onnx": official / "models/stand_backend_scratch_i150_actor.onnx",
    }
    assets = {}
    for name, path in paths.items():
        regular_file = path.is_file()
        actual = sha256(path) if regular_file else None
        assets[name] = {
            "path": str(path),
            "regular_file": regular_file,
            "expected_sha256": EXPECTED[name],
            "actual_sha256": actual,
            "sha256_matches": actual == EXPECTED[name],
            "bytes": path.stat().st_size if regular_file else None,
        }
    deploy_ready = all(assets[name]["sha256_matches"] for name in (
        "actor_onnx", "gait_template", "official_bundle", "x2_xml", "scene_xml", "stand_onnx"
    ))
    training_ready = deploy_ready and assets["source_checkpoint"]["sha256_matches"]
    report = {
        "stage": "x2_native_locomotion_baseline_preflight_20260812",
        "semantics": {
            "stage264": "Task50 readiness decision using frozen Stage219-s2600 weights and Stage250 deployment contract; not a separate checkpoint",
            "deploy_ready": "all exact official closed-loop inputs are present",
            "training_ready": "deploy_ready plus exact source PPO checkpoint is present",
        },
        "assets": assets,
        "decision": {
            "deploy_ready": deploy_ready,
            "training_ready": training_ready,
            "missing_or_mismatched": [name for name, row in assets.items() if not row["sha256_matches"]],
            "allow_gate_replay": deploy_ready,
            "allow_continued_training": training_ready,
        },
        "disk_policy": {
            "do_not_download_amass_before_task4": True,
            "smoke_before_long_train": True,
            "large_artifacts_outside_git": True,
        },
    }
    output = args.output or repo / "reports/official_x2/x2_native_locomotion_baseline_preflight_20260812.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if training_ready else 2)


if __name__ == "__main__":
    main()
