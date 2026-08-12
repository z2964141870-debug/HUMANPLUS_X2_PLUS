from pathlib import Path

import torch

from src.x2_privileged_generator_live import (
    finalize_native_generator_reset,
    reset_from_native_generator_seed,
)


class _FakeAsset:
    def __init__(self, names):
        self.joint_names = names
        self.device = torch.device("cpu")
        n = len(names)
        self.data = type("Data", (), {
            "default_joint_pos": torch.zeros(2, n),
            "joint_pos_limits": torch.tensor([[[-10.0, 10.0]]] * n * 2).reshape(2, n, 2),
            "soft_joint_pos_limits": torch.tensor([[[-10.0, 10.0]]] * n * 2).reshape(2, n, 2),
            "soft_joint_vel_limits": torch.full((2, n), 100.0),
            "joint_vel_limits": torch.full((2, n), 100.0),
        })()
        self.writes = {}

    def write_root_pose_to_sim(self, value, env_ids): self.writes["root_pose"] = value
    def write_root_velocity_to_sim(self, value, env_ids): self.writes["root_velocity"] = value
    def write_joint_state_to_sim(self, q, dq, env_ids): self.writes.update(joint_pos=q, joint_vel=dq)


def _fake_env(seed):
    import numpy as np
    with np.load(seed, allow_pickle=False) as payload:
        names = list(payload["joint_names"].tolist())
    asset = _FakeAsset(names)
    class Scene(dict):
        pass
    scene = Scene(robot=asset)
    scene.env_origins = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    env = type("Env", (), {"device": torch.device("cpu"), "scene": scene})()
    return env, asset


def test_native_reset_event_writes_exact_state_and_no_contacts():
    root = Path(__file__).resolve().parents[1]
    seed = root / "research/dynamic_retargeting_20260811/phase36_native_generator_seed.npz"
    env, asset = _fake_env(seed)
    reset_from_native_generator_seed(
        env, torch.tensor([0, 1]), seed_path=str(seed),
        expected_sha256="4bd8bc428cc47fea3c5a783af9277babb45cc5cfdbdf2ce451628e81be5e7ad2",
        fixed_frame_indices=[0, 199],
        sampling_mode="evenly_spaced", asset_name="robot",
    )
    assert set(asset.writes) == {"root_pose", "root_velocity", "joint_pos", "joint_vel"}
    assert env._x2_native_generator_reset_last["contact_labels_written"] is False
    assert env._x2_native_generator_reset_last["hard_position_projection_applied"] is False
    assert torch.equal(env._x2_native_generator_reset_last["frame_indices"], torch.tensor([0, 199]))
    assert torch.equal(
        env._x2_native_generator_reset_last["selected_env_ids"], torch.tensor([0, 1])
    )
    assert env._x2_native_generator_reset_last["seed_sha256"] == (
        "4bd8bc428cc47fea3c5a783af9277babb45cc5cfdbdf2ce451628e81be5e7ad2"
    )
    assert env._x2_native_generator_reset_last["sampling_mode"] == "evenly_spaced"
    assert env._x2_native_generator_reset_last["asset_name"] == "robot"
    assert set(("root_pose", "root_velocity", "joint_pos", "joint_vel")) <= set(
        env._x2_native_generator_reset_last
    )
    asset.writes.clear()
    result = finalize_native_generator_reset(env)
    assert result == {"selected_env_count": 2, "no_op": False}
    assert set(asset.writes) == {"root_pose", "root_velocity", "joint_pos", "joint_vel"}
    assert env._x2_native_generator_reset_last["finalizer_applied"] is True


def test_fraction_zero_returns_before_opening_seed():
    env = type("Env", (), {"device": torch.device("cpu")})()
    reset_from_native_generator_seed(
        env, torch.tensor([0]), seed_path="/does/not/exist",
        expected_sha256="bad", reset_fraction=0.0,
    )
    assert env._x2_native_generator_reset_last["strict_no_op"] is True


def test_stage152_default_and_opt_in_targets_are_explicit():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_dcpeft_stage152.sh").read_text()
    launcher = (root / "scripts/run_phase40_privileged_generator_live_zero.sh").read_text()
    assert "FAITHFUL_WBT29_ENV_TARGET" in text
    assert "x2_faithful_live_actions_phase46.FaithfulWBT29TrackingEnvCfg" in text
    assert "x2_privileged_generator_live_env.PrivilegedGeneratorTrackingEnvCfg" in launcher
    assert "train_agent_trl_privileged_generator.py" in launcher
    assert "ITERS=0" in launcher and "SAVE_FREQUENCY=-1" in launcher
    assert "PHASE40_OUTPUT" in launcher
    assert 'decision.get("result") != "PASS_LIVE_ZERO_ONLY"' in launcher
    assert 'runtime.get("optimizer_steps") != 0' in launcher


def test_privileged_entrypoint_imports_upstream_from_stage152_cwd():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts/train_agent_trl_privileged_generator.py").read_text()
    assert "_sandbox_cwd = os.getcwd()" in wrapper
    assert "sys.path.insert(0, _sandbox_cwd)" in wrapper
    assert wrapper.index("sys.path.insert(0, _sandbox_cwd)") < wrapper.index(
        "import train_agent_trl as upstream"
    )
    assert 'os.environ.setdefault("HYDRA_MAIN_MODULE", "__main__")' in wrapper
    assert "sys.path.append(_sandbox_cwd)" not in wrapper
    assert 'sys.modules["isaaclab.envs"]' in wrapper
