from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from official_x2.role_aware_recovery_curriculum import (
    eligibility_reasons,
    fixed_batch_indices,
    outcome_role,
    reset_from_role_aware_suffix,
    load_manifest,
)


def _row():
    return {
        "stop_elapsed_s": 1.0,
        "physical_state": {
            "root_position_m": [0.0, 0.0, 0.62],
            "root_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_names": ["joint"],
            "joint_position_rad": [0.0],
            "joint_velocity_radps": [0.0],
        },
    }


def test_outcome_roles_are_explicit_and_failure_is_not_success():
    assert outcome_role({"full_gate_pass": True}) == "success"
    assert outcome_role({"full_gate_pass": False, "survived_stop_height_gate": True}) == "critical"
    assert outcome_role({"full_gate_pass": False, "survived_stop_height_gate": False}) == "height_fail"


def test_eligibility_rejects_post_collapse_and_limit_violations():
    limits = {"joint": {"lower": -1.0, "upper": 1.0, "velocity": 2.0}}
    assert eligibility_reasons(_row(), limits) == []
    collapsed = _row()
    collapsed["physical_state"]["root_position_m"][2] = 0.2
    collapsed["physical_state"]["joint_position_rad"][0] = 2.0
    reasons = eligibility_reasons(collapsed, limits)
    assert "root_z_below_safe_reset_floor" in reasons
    assert "joint_position_outside_urdf_limit" in reasons


def test_fraction_zero_does_not_open_manifest_or_consume_rng():
    env = SimpleNamespace(device=torch.device("cpu"))
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    reset_from_role_aware_suffix(
        env, torch.arange(4), manifest_path="/definitely/missing.json",
        reset_fraction=0.0, outcome_role_name="success",
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert env._x2_role_reset_last["strict_no_op"] is True
    assert len(env._x2_role_reset_last["selected_env_ids"]) == 0


def test_fixed_batch_indices_are_role_and_eligibility_filtered():
    manifest = {"rows": [
        {"manifest_row_index": 0, "outcome_role": "success", "eligible": False},
        {"manifest_row_index": 1, "outcome_role": "success", "eligible": True},
        {"manifest_row_index": 2, "outcome_role": "critical", "eligible": True},
    ]}
    assert fixed_batch_indices(manifest, "success", 3) == [1, 1, 1]


def test_frozen_manifest_contains_references_not_copied_training_targets():
    path = Path("manifests/x2_phase17_role_aware_reset_curriculum.json")
    manifest = load_manifest(
        path,
        "8904ebd256ee49279ac8011db04dc0044b2ad75ea86d85f53d6dce174cc2c7a7",
    )
    assert manifest["action_semantics"]["imitation_target"] is False
    assert set(manifest["counts"]) == {"success", "critical", "height_fail"}
    forbidden = {"physical_state", "observation_93d", "actual_issued_action"}
    assert all(not (forbidden & set(row)) for row in manifest["rows"])
