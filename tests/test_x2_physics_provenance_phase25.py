from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs/x2_faithful_physics_phase25.yaml"
RUNTIME = REPO / "configs/x2_faithful_physics_phase25_runtime.json"
REPORT = REPO / "reports/retarget/x2_physics_provenance_phase25.json"
GUARD = REPO / "tools/retarget/guard_x2_physics_phase25.py"
LAUNCHER = REPO / "scripts/run_x2_faithful_any2any_phase25.sh"

sys.path.insert(0, str(REPO / "src"))
from x2_physics_provenance_guard import (  # noqa: E402
    assert_physics_contract,
    check_runtime_snapshot,
    load_manifest,
    load_runtime_snapshot,
    sha256,
)


def _matrix(report: dict) -> dict[str, dict]:
    return {row["item"]: row for row in report["comparison_matrix"]}


def test_declared_train_domain_and_held_out_boundary_are_explicit():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"]["status"] == "B5_READY_AS_DECLARED_TRAIN_DOMAIN"
    assert report["truth_boundary"] == {
        "isaaclab_is_training_domain": True,
        "official_mujoco_is_held_out_sim_to_sim": True,
        "domains_numerically_equivalent": False,
        "physics_step_training_zero_update": False,
    }

    matrix = _matrix(report)
    for item in (
        "pelvis mass",
        "full active collision set",
        "default pose/root height",
        "application PD",
        "joint armature",
        "action scale",
        "physics dt",
    ):
        assert matrix[item]["status"] == "different"
    for item in (
        "velocity limits",
        "cross-engine solver",
        "raw URDF official generation lineage",
    ):
        assert matrix[item]["status"] == "unknown"


def test_collision_and_static_contract_evidence_are_not_silently_equated():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    collision = report["collision"]
    assert collision["sole_spheres_exact"] is True
    assert collision["sole_spheres_per_foot"] == {"left": 12, "right": 12}
    assert collision["isaac_source_active_count"] == 52
    assert collision["official_active_count"] == 49
    assert set(collision["body_count_differences"]) == {
        "head_yaw_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    }
    assert collision["body_type_differences"]["head_pitch_link"] == {
        "isaac_source": [7],
        "official": [5],
    }

    assert report["joint_limits"]["range_abs_max_rad"] == pytest.approx(0.0)
    assert report["joint_limits"]["effort_abs_max_nm"] == pytest.approx(0.0)
    assert report["fk"]["neutral_position_max_m"] < 1e-6
    assert report["fk"]["neutral_orientation_max_rad"] < 1e-6


def test_hash_runtime_guard_accepts_only_the_declared_snapshot():
    manifest = load_manifest(MANIFEST)
    runtime = load_runtime_snapshot(RUNTIME)
    positive = assert_physics_contract(MANIFEST, runtime)
    assert all(positive["file_checks"].values())
    assert positive["runtime_check"]["exact"] is True

    changed = dict(runtime)
    changed["sim_dt"] = 0.01
    mismatch = check_runtime_snapshot(manifest, changed)
    assert mismatch["exact"] is False
    assert mismatch["different"]["sim_dt"] == {
        "expected": 0.005,
        "actual": 0.01,
    }
    with pytest.raises(RuntimeError, match="runtime contract differs"):
        assert_physics_contract(MANIFEST, changed)


def test_guard_cli_and_launcher_are_fail_closed_before_live_zero():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["immutable"] is True
    assert manifest["guard"] == {
        "fail_closed": True,
        "require_all_file_hashes": True,
        "require_runtime_snapshot_exact": True,
        "allow_environment_overrides": False,
        "b5_ready_as_declared_train_domain": True,
        "live_zero_update_authorized": False,
    }

    expected = sha256(MANIFEST)
    passed = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--manifest",
            str(MANIFEST),
            "--runtime",
            str(RUNTIME),
            "--expected-manifest-sha256",
            expected,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(passed.stdout)["status"] == "B5_PRELAUNCH_GUARD_PASSED"

    refused = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--manifest",
            str(MANIFEST),
            "--runtime",
            str(RUNTIME),
            "--expected-manifest-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 43
    assert json.loads(refused.stdout)["status"] == "B5_MANIFEST_HASH_REFUSED"

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert expected in launcher
    assert "guard_x2_physics_phase25.py" in launcher
    assert "run_x2_faithful_any2any_phase24.sh" in launcher
    assert "live-zero pending explicit post-Phase25 review" in launcher
