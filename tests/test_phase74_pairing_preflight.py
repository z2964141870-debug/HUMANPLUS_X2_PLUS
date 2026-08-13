import torch
from pathlib import Path

from cwi_x2.phase74_pairing_preflight import copy_pair_rows, pair_diagnostics, paired_lane_ids


ROOT = Path(__file__).resolve().parents[1]


def test_alternating_pair_donors() -> None:
    donors0, recipients0 = paired_lane_ids(0, 4)
    donors1, recipients1 = paired_lane_ids(1, 4)
    assert donors0.tolist() == [0, 3, 4, 7]
    assert recipients0.tolist() == [1, 2, 5, 6]
    assert torch.equal(donors0, recipients1)
    assert torch.equal(recipients0, donors1)


def test_copy_and_exact_diagnostics() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    donors, recipients = paired_lane_ids(0, 4)
    copy_pair_rows(value, donors, recipients)
    result = pair_diagnostics(value, donors, recipients)
    assert result["passed"] is True
    assert result["pair_pass_count"] == 4
    assert result["max_abs"] == 0.0


def test_tolerance_is_explicit() -> None:
    value = torch.zeros(4, 1)
    donors, recipients = paired_lane_ids(0, 2)
    value[recipients] = 5.0e-7
    assert pair_diagnostics(value, donors, recipients, atol=1.0e-6)["passed"] is True
    assert pair_diagnostics(value, donors, recipients, atol=1.0e-7)["passed"] is False


def test_matching_contact_sentinels_are_not_false_mismatches() -> None:
    value = torch.tensor([[float("nan")], [float("nan")], [float("inf")], [float("inf")]])
    donors, recipients = paired_lane_ids(0, 2)
    assert pair_diagnostics(value, donors, recipients)["passed"] is True
    value[recipients[1]] = float("-inf")
    assert pair_diagnostics(value, donors, recipients)["passed"] is False


def test_runner_is_initial_only_and_failure_exits_before_native_close() -> None:
    text = (ROOT / "scripts/run_x2_phase74_pairing_preflight.py").read_text()
    assert "residual_env.step" not in text
    assert "env.step(" not in text
    assert "reward_manager.compute" not in text
    assert "torch.optim" not in text
    assert ".backward(" not in text
    assert "env.observation_manager.compute(update_history=False)" in text
    assert "get_root_transforms()" in text
    assert "get_root_velocities()" in text
    assert "get_dof_positions()" in text
    assert "get_dof_velocities()" in text
    assert '"x2_gait_action_module_sha256"' in text
    assert '"upper_hook_sha256"' in text
    assert "unknown_mutable_tensor_fields_empty" in text
    assert 'finite_required = not name.endswith(".data.contact_pos_w")' in text
    assert "for sensor_name, sensor in sorted(env.scene.sensors.items())" in text
    assert 'add("robot.data._previous_joint_vel"' in text
    assert "ROBOT_DATA_TIMESTAMPED_BUFFERS" in text
    assert "invalidate_articulation_lazy_caches(env)" in text
    assert 'for name in ("_scale", "_offset", "_clip")' in text
    assert "sys.excepthook = phase74_excepthook" in text
    failure_tail = text.split("except BaseException as exc:", 1)[1]
    assert "atomic_json(args.failure, failure)" in failure_tail
    assert "os._exit(1)" in failure_tail
    assert failure_tail.index("os._exit(1)") < failure_tail.index("else:\n    simulation_app.close()")


def test_runner_does_not_close_environment_before_failure_evidence() -> None:
    text = (ROOT / "scripts/run_x2_phase74_pairing_preflight.py").read_text()
    run_body = text.split("def run() -> None:", 1)[1].split("\n\ntry:\n    run()", 1)[0]
    assert "wrapped.close()" not in run_body
    assert "env.close()" not in run_body
