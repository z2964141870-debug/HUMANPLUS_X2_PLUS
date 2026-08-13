import json
from pathlib import Path

import torch

from cwi_x2.phase73_antithetic import generate_epsilon, tensor_hash


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "444482b29d05303b6b957b0f043d588c7f67b5bedc231912e33f7e47cccfb335"
OLD_FINITE = '"finite": all(torch.isfinite(value).all() for value in tensors.values()) and torch.isfinite(encoded).all(),'
NEW_FINITE = '"finite": bool(all(bool(torch.isfinite(value).all()) for value in tensors.values()) and bool(torch.isfinite(encoded).all())), '


def test_phase73_schedule_is_fresh_and_repeatable() -> None:
    phase73 = generate_epsilon(731041)
    replay = generate_epsilon(731041)
    phase72 = generate_epsilon(721041)
    assert torch.equal(phase73, replay)
    assert not torch.equal(phase73, phase72)
    assert tensor_hash(phase73) == tensor_hash(replay)


def test_phase73_runner_repair_is_one_scalar_conversion() -> None:
    wrapper = (ROOT / "scripts/run_x2_phase73_antithetic.py").read_text()
    assert SOURCE_SHA in wrapper
    assert "text.count(OLD_FINITE) != 1" in wrapper
    assert "text.replace(OLD_FINITE, NEW_FINITE)" in wrapper
    assert "text.replace(OLD_EXIT, NEW_EXIT)" in wrapper
    assert "os._exit(1)" in wrapper
    assert '.replace("phase72", "phase73")' in wrapper


def test_complete_dummy_screen_json_roundtrip_has_no_tensor() -> None:
    finite = bool(
        all(bool(torch.isfinite(value).all()) for value in (torch.ones(2), torch.zeros(2)))
        and bool(torch.isfinite(torch.ones(2)).all())
    )
    report = {
        "schema": "x2_phase73_antithetic_screen_v1",
        "decision": "VALID_PENDING_PAIR",
        "seed_index": 0,
        "env_seed": 730041,
        "latent_seed": 731041,
        "sign": "plus",
        "init_mode": "write",
        "initial_commit": "/tmp/seed0_initial_commit.json",
        "initial_commit_sha256": "a" * 64,
        "initial_combined_sha256": "b" * 64,
        "bundle": "/tmp/seed0_plus_rollout.pt",
        "bundle_sha256": "c" * 64,
        "bundle_bytes": 24_000_000,
        "epsilon_tensor_sha256": "d" * 64,
        "requested_max_abs_rad": 0.0027,
        "effective_requested_abs_ratio": 1.0,
        "reward_closure_max_abs": 0.0,
        "phase_counts": {str(index): 6400 for index in range(4)},
        "technical_checks": {
            "pair_initial_state_exact": True,
            "nondegenerate_reset_perturbation": True,
            "zero_residual_mean": True,
            "epsilon_replay_exact": True,
            "finite": finite,
            "all_25600_active": True,
            "no_done_or_timeout": True,
            "all_phase_samples_classified": True,
            "every_anchor_env_covers_four_phases": True,
            "reward_closure": True,
            "non_knee_exact_zero": True,
            "physical_bound": True,
            "effective_exact": True,
            "source_state_unchanged": True,
            "residual_state_unchanged": True,
            "source_and_residual_grad_none": True,
            "optimizer_steps_zero": True,
            "optimizer_state_entries_zero": True,
            "backward_calls_zero": True,
            "checkpoint_count_zero": True,
        },
        "physics_steps": 25600,
        "optimizer_steps": 0,
        "optimizer_state_entries": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "optimizer_unlocked": False,
        "long_training_unlocked": False,
        "deployment_unlocked": False,
        "task2_complete": False,
    }
    restored = json.loads(json.dumps(report))
    assert restored == report
    assert set(restored) == {
        "schema", "decision", "seed_index", "env_seed", "latent_seed", "sign", "init_mode",
        "initial_commit", "initial_commit_sha256", "initial_combined_sha256", "bundle",
        "bundle_sha256", "bundle_bytes", "epsilon_tensor_sha256", "requested_max_abs_rad",
        "effective_requested_abs_ratio", "reward_closure_max_abs", "phase_counts",
        "technical_checks", "physics_steps", "optimizer_steps", "optimizer_state_entries",
        "backward_calls", "checkpoint_count", "optimizer_unlocked", "long_training_unlocked",
        "deployment_unlocked", "task2_complete",
    }
    assert len(restored["technical_checks"]) == 20
    assert type(restored["technical_checks"]["finite"]) is bool

    def assert_native(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_native(child)
        elif isinstance(value, list):
            for child in value:
                assert_native(child)

    assert_native(restored)


def test_phase73_repaired_source_compiles_without_execution() -> None:
    source = (ROOT / "scripts/run_x2_phase72_antithetic.py").read_text()
    assert source.count(OLD_FINITE) == 1
    repaired = source.replace(OLD_FINITE, NEW_FINITE)
    old_exit = '''failure = None
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
    new_exit = '''try:
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
    assert source.count(old_exit) == 1
    repaired = repaired.replace(old_exit, new_exit)
    repaired = repaired.replace("phase72", "phase73").replace("Phase72", "Phase73")
    compile(repaired, "phase73_repaired_runner", "exec")
