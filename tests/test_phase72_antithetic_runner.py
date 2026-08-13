from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase72_runner_is_independent_and_zero_optimizer() -> None:
    text = (ROOT / "scripts/run_x2_phase72_antithetic.py").read_text()
    assert "run_x2_upper_robust_one_update_phase56" not in text
    assert "torch.inference_mode()" in text
    assert "optimizer_steps_zero" in text
    assert "backward_calls_zero" in text
    assert "checkpoint_count_zero" in text
    assert "torch.optim" not in text
    assert ".backward(" not in text
    assert "autograd" not in text


def test_phase72_reset_is_non_degenerate_and_pair_guarded() -> None:
    text = (ROOT / "scripts/run_x2_phase72_antithetic.py").read_text()
    for token in (
        "root_pose_roll_rad",
        "root_pose_pitch_rad",
        "root_linear_x_mps",
        "root_linear_y_mps",
        "root_angular_roll_radps",
        "root_angular_pitch_radps",
        "root_angular_yaw_radps",
        "pair initial tensors are not bitwise identical",
        "combined_sha256",
    ):
        assert token in text


def test_phase72_uses_forced_frozen_schedule_not_normal_sample() -> None:
    text = (ROOT / "scripts/run_x2_phase72_antithetic.py").read_text()
    assert "sign_value * sigma * innovation" in text
    assert ".sample()" not in text
    assert "epsilon_replay_exact" in text
    assert "zero_residual_mean" in text


def test_phase72_shell_enforces_every_launch_resource_gate() -> None:
    text = (ROOT / "scripts/run_x2_phase72_antithetic.sh").read_text()
    for token in (
        "gpu_peak",
        "bundle_bytes",
        "free_after",
        "cumulative_disk_delta",
        "536870912",
        "4294967296",
        "seen_initial_hashes",
    ):
        assert token in text
