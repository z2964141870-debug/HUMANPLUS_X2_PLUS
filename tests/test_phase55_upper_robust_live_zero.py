from __future__ import annotations

import json
from pathlib import Path

from cwi_x2.upper_motion_contract import UPPER_JOINT_NAMES, load_upper_motion

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_upper_robust_lower_live_zero_phase55.json"


def test_phase50_npz_adapter_is_exact_and_finite():
    clip = load_upper_motion(REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz")
    assert clip.joint_names == UPPER_JOINT_NAMES
    assert clip.q_rad.shape == (200, 14)
    assert clip.fps == 50.0


def test_phase55_launcher_is_zero_only_and_standard_93d():
    runner = (REPO / "scripts/run_x2_upper_robust_live_zero_phase55.py").read_text()
    launcher = (REPO / "scripts/run_x2_upper_robust_live_zero_phase55.sh").read_text()
    assert "FutureIntentActorCritic" not in runner
    assert "ActorCritic(" in runner
    assert ".step(" not in runner
    assert '"optimizer_steps": 0' in runner
    assert "CWI_UPPER_ZERO_FRACTION=0.50" in launcher
    assert "CWI_UPPER_DETERMINISTIC_SPLIT=1" in launcher
    assert "CWI_UPPER_SPLIT_MODE=interleaved" in launcher
    hook = (REPO / "hooks/sitecustomize.py").read_text()
    assert "_cwi_upper_deterministic_split" in hook
    assert "x2_hybrid_phase44_upper_motion.npz" in launcher


def test_live_report_is_fail_closed_and_exact():
    result = json.loads(REPORT.read_text())
    assert result["decision"] == "PASS_LIVE_ZERO_UPDATE_ONLY"
    assert result["runtime"]["policy_obs"] == [64, 93]
    assert result["runtime"]["critic_obs"] == [64, 93]
    assert result["runtime"]["policy_action_dim"] == 15
    assert result["runtime"]["sim_joint_count"] == 31
    assert result["runtime"]["fixed_upper_envs"] == 32
    assert result["runtime"]["active_upper_envs"] == 32
    assert result["runtime"]["domain_upper_counts"] == {
        "none_ideal": 24, "none_response": 8,
        "bounded_ideal": 24, "bounded_response": 8,
    }
    assert result["runtime"]["upper_target_in_action_residual"] is False
    assert result["parameters"]["frozen_names"] == ["std"]
    assert result["parameters"]["hash_before"] == result["parameters"]["hash_after"]
    assert result["optimizer_constructed"] is False
    assert result["optimizer_steps"] == 0
    assert result["environment_control_steps"] == 0
    assert result["checkpoint_created"] is False
    assert all(row["action_max_abs"] == row["value_max_abs"] == 0.0 for row in result["fixed_forward_batches"])
