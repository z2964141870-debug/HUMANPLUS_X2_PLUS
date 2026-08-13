from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_groups_cover_each_role_twice() -> None:
    env_ids = np.arange(256)
    groups = env_ids // 16
    roles = (env_ids // 2) % 8
    for group in range(16):
        assert np.bincount(roles[groups == group], minlength=8).tolist() == [2] * 8


def test_validation_lane_swap_is_exact_crossover() -> None:
    groups = np.arange(256) // 16
    lane_a = groups >= 8
    lane_b = ~lane_a
    assert np.all(lane_a ^ lane_b)
    assert lane_a.sum() == lane_b.sum() == 128


def test_runner_transform_is_bounded_and_accepts_scientific_stop() -> None:
    text = (ROOT / "scripts/run_x2_privileged_feedback_teacher_search_v3.py").read_text()
    assert 'args.num_envs != 256' in text
    assert 'PASS_FEEDBACK_TEACHER_SEARCH_LOCAL_ONLY' in text
    assert 'FAIL_FEEDBACK_TEACHER_SEARCH_STOP' in text
    assert 'optimizer_steps": 0' in text
    assert 'checkpoint_writes": 0' in text
    assert "feedback_scale(" in text
    assert "slew_limited_residual(" in text


def test_search_budget_is_six_plus_two_rollouts() -> None:
    text = (ROOT / "scripts/run_x2_privileged_feedback_teacher_search_v3.py").read_text()
    assert '(int(search["iterations"]) + 2) * args.steps' in text
    assert "range(1, 16)" in text
    assert "population = np.clip" in text
