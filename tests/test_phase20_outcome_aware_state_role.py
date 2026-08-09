import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from official_x2.outcome_aware_state_role_v2 import (
    fixed_batch_indices,
    load_manifest,
    reset_from_outcome_aware_v2,
    validate_manifest,
)


def test_fraction_zero_is_strict_io_and_rng_noop():
    env = SimpleNamespace(device=torch.device("cpu"))
    torch.manual_seed(2020)
    before = torch.random.get_rng_state().clone()
    reset_from_outcome_aware_v2(
        env, torch.arange(16), manifest_path="/missing/phase20.json",
        reset_fraction=0.0, state_role_name="success_safe",
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert env._x2_phase20_reset_last["strict_no_op"] is True
    assert len(env._x2_phase20_reset_last["selected_env_ids"]) == 0


def test_frozen_manifest_has_two_nonempty_state_roles_without_fabricated_episode():
    path = Path("manifests/x2_phase19_outcome_aware_state_role.json")
    manifest = load_manifest(path)
    assert set(manifest["counts"]) == {"success_safe", "critical_from_failure"}
    assert all(manifest["counts"][role]["eligible"] > 0 for role in manifest["counts"])
    assert manifest["role_semantics"]["critical_episode_fabricated"] is False
    assert manifest["action_semantics"]["imitation_target"] is False
    assert manifest["action_semantics"]["failure_action_is_expert_label"] is False
    assert all("physical_state" not in row for row in manifest["rows"])


def test_critical_state_role_is_safe_failure_prefix_with_bounded_future_collapse():
    manifest = load_manifest("manifests/x2_phase19_outcome_aware_state_role.json")
    critical = [
        row for row in manifest["rows"]
        if row["state_role"] == "critical_from_failure" and row["eligible"]
    ]
    assert critical
    assert all(row["episode_outcome"] == "height_failure" for row in critical)
    assert all(row["root_z_m"] >= 0.55 and row["root_tilt_rad"] <= 0.30 for row in critical)
    assert all(0.5 - 1e-9 <= row["time_to_height_collapse_s"] <= 1.5 + 1e-9 for row in critical)


def test_manifest_hash_and_source_hashes_fail_closed():
    manifest = load_manifest("manifests/x2_phase19_outcome_aware_state_role.json")
    broken = copy.deepcopy(manifest)
    broken["counts"]["success_safe"]["eligible"] += 1
    try:
        validate_manifest(broken, validate_sources=False)
    except ValueError as exc:
        assert "content hash" in str(exc)
    else:
        raise AssertionError("tampered manifest was accepted")


def test_fixed_batches_are_role_filtered():
    manifest = load_manifest("manifests/x2_phase19_outcome_aware_state_role.json")
    for role in ("success_safe", "critical_from_failure"):
        indices = fixed_batch_indices(manifest, role, 16)
        assert len(indices) == 16
        assert all(manifest["rows"][index]["state_role"] == role for index in indices)
        assert all(manifest["rows"][index]["eligible"] for index in indices)
